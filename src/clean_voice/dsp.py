"""The clean-up chain, at 48 kHz mono (DESIGN Features, J10):

    rumble filter -> Denoise (RNNoise + strength mix) -> Voice style (5-band EQ) -> Fine-tune filters
        -> compressor -> Voice volume -> limiter

Denoise, Voice style and Fine-tune are the guest's Step 2 switches; only what they switch changes the sound, and the
fixed stages stay as they are. Fine-tune runs the guest's list of OBS filters (``obs_filters``), top first.

There is no noise gate: RNNoise alone quiets the pauses, as in OBS's usual voice chain (noise suppression, then
compressor and limiter). A gate cut words on a quiet mic in a noisy room (user decision, 2026-11-11).

The guest's settings (``clean_voice.settings``) arrive through a ``SettingsCell`` read once per block, and every change
is ramped, so it is click-free. Every stage keeps its state between calls and accepts any block length; a chain made
with ``block=480`` (the worker's) takes whole 480-sample frames only, which saves the denoiser's 10 ms queue. Per-sample work is vectorized numpy/scipy (Numba is ruled out: no macOS
x86_64 wheels); the compressor's envelope is a plain loop over the block, sample by sample, as OBS runs it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import signal

from clean_voice import obs_filters
from clean_voice.filters import sosfilt
from clean_voice.fine_tune import FilterSpec
from clean_voice.ring import Ring
from clean_voice.rnnoise import FRAME_SIZE, RNNoise
from clean_voice.settings import Band, Settings, SettingsCell, bands_for
from clean_voice.vst import KIND as VST_KIND
from clean_voice.vst import VstPlugin, VstRegistry

SAMPLE_RATE = 48_000
CONTROL_STEP = 48  # 1 ms: the EQ glide's step

Audio = npt.NDArray[np.float32]


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


# ------------------------------------------------------------------ tuned defaults

RUMBLE_HZ = 80.0
RUMBLE_ORDER = 4

COMP_THRESHOLD_DB = -24.0
COMP_RATIO = 3.0
COMP_ATTACK_MS = 5.0
COMP_RELEASE_MS = 120.0
COMP_MAKEUP_DB = 6.0

LIMIT_CEILING_DB = -1.0  # DESIGN: the output never exceeds -1 dBFS
LIMIT_RELEASE_DB_PER_S = 80.0

RAMP_SAMPLES = 480  # 10 ms: every gain change (volume, strength, bypass, echo, Fine-tune) is a linear ramp this long
EQ_GLIDE_S = 0.020  # an EQ band moves to its new gain, frequency and Q over 20 ms
VOLUME_RANGE_DB = 12.0  # Voice volume 0..100 is -12..+12 dB


class RumbleFilter:
    """80 Hz high-pass (4th-order Butterworth) that removes desk thumps, traffic and mic-stand rumble."""

    def __init__(self) -> None:
        sos = signal.butter(RUMBLE_ORDER, RUMBLE_HZ, btype="highpass", fs=SAMPLE_RATE, output="sos")
        self._sos: npt.NDArray[np.float64] = np.asarray(sos, dtype=np.float64)
        self._zi = np.zeros((self._sos.shape[0], 2))

    def process(self, x: Audio) -> Audio:
        y, self._zi = signal.sosfilt(self._sos, x, zi=self._zi)
        return y.astype(np.float32)


class NoiseSuppressor:
    """RNNoise on 480-sample frames, buffering the block size in and out.

    Follows OBS plugins/obs-filters/noise-suppress-filter.c: input is queued until a full 10 ms segment exists (:196-206,
    :441-442), each segment is denoised, and the output is handed on as soon as it exists (:451-452). With ``block`` a
    multiple of 480 (the worker's blocks) every call completes its frames, so nothing more is queued. For any other
    block size (``block=None``) the output queue starts with one frame of zeros, a fixed 10 ms that OBS does not need
    (it returns nothing until a packet is whole), so every call can return a full block.
    """

    LATENCY = FRAME_SIZE  # the queue's latency for any block size; 0 for whole frames

    def __init__(self, block: int | None = None) -> None:
        self._rnnoise = RNNoise()
        self._pending = np.zeros(FRAME_SIZE, np.float32)
        self._pending_len = 0
        self._frame_out = np.zeros(FRAME_SIZE, np.float32)
        self._out = Ring(1 << 16)
        self.whole_frames = block is not None and block % FRAME_SIZE == 0
        self.latency = 0 if self.whole_frames else self.LATENCY
        self._out.write(np.zeros(self.latency, np.float32))  # the fixed latency, so output never runs short
        self.wet: Audio = np.zeros(0, np.float32)

    def process(self, x: Audio) -> Audio:
        pos = 0
        while pos < len(x):
            take = min(FRAME_SIZE - self._pending_len, len(x) - pos)
            self._pending[self._pending_len : self._pending_len + take] = x[pos : pos + take]
            self._pending_len += take
            pos += take
            if self._pending_len == FRAME_SIZE:
                self._rnnoise.process(self._pending, self._frame_out)  # its voice probability is not used
                self._out.write(self._frame_out)
                self._pending_len = 0
        y = np.empty(len(x), np.float32)
        if self._out.read_into(y) < len(x):
            raise ValueError(f"a {len(x)}-sample block: this chain was made for whole 480-sample frames")
        self.wet = y  # the fully denoised block (test S1 lines it up with the dry path)
        return y


def strength_mix(strength: float) -> float:
    """How much of the un-denoised signal stays in: d = 10^((-6 - 54 s) / 20), the original's strength formula
    (simple-clean-mic dsp.rs:229, where it set the gate's floor) moved to the RNNoise mix, so strength 0 keeps the dry
    signal at -6 dB and 70 at -43.8 dB. Strength 100 is fully wet, RNNoise alone, as OBS runs it: OBS has no RNNoise
    level (its suppress level is Speex-only, noise-suppress-filter.c:271, :473-477)."""
    if strength >= 100.0:
        return 0.0
    return db_to_lin(-6.0 - 54.0 * strength / 100.0)


class Compressor(obs_filters.Compressor):
    """The chain's fixed downward compressor with makeup gain: OBS's Compressor filter (``obs_filters.Compressor``,
    plugins/obs-filters/compressor-filter.c, sample by sample) at the J06 Step 6 values: -24 dB, 3:1, 5/120 ms and
    +6 dB of output gain, the makeup. Voice volume is a separate, ramped stage after."""

    def __init__(self) -> None:
        super().__init__(
            {
                "threshold": COMP_THRESHOLD_DB,
                "ratio": COMP_RATIO,
                "attack_time": COMP_ATTACK_MS,
                "release_time": COMP_RELEASE_MS,
                "output_gain": COMP_MAKEUP_DB,
            }
        )


class Limiter:
    """Peak limiter whose output never exceeds -1 dBFS.

    Follows OBS plugins/obs-filters/limiter-filter.c: slope 1, so the gain is threshold - envelope in dB, reduction only
    (:100, :153-166), with an instant attack, as OBS's: its ATK_TIME 0.001 is taken as milliseconds (:39, :88) and
    divided by 1000 again (:95), so the attack is 1 us. One change: the release (OBS: a one-pole, 60 ms by default, :96,
    :185) is linear in dB (LIMIT_RELEASE_DB_PER_S), which lets the whole block be solved at once:
    gain_db[n] = min(need_db[n], gain_db[n-1] + release).
    """

    def __init__(self) -> None:
        self._ceiling = db_to_lin(LIMIT_CEILING_DB)
        self._release = LIMIT_RELEASE_DB_PER_S / SAMPLE_RATE  # dB per sample
        self._gain_db = 0.0
        self.last_gain_db: npt.NDArray[np.float64] = np.zeros(0)  # the last block's gain, for the level tests

    def process(self, x: Audio) -> Audio:
        n = len(x)
        if n == 0:
            return x.copy()
        peak = np.maximum(np.abs(x).astype(np.float64), 1e-12)
        need_db = np.minimum(0.0, 20.0 * np.log10(self._ceiling / peak))
        k = np.arange(n, dtype=np.float64)
        # min over j <= n of (need_db[j] + release * (n - j)), also bounded by the carried gain.
        held = np.minimum.accumulate(need_db - self._release * k) + self._release * k
        carried = self._gain_db + self._release * (k + 1.0)
        gain_db = np.minimum(np.minimum(held, carried), 0.0)
        self._gain_db = float(gain_db[-1])
        self.last_gain_db = gain_db
        y = (x * np.power(10.0, gain_db / 20.0)).astype(np.float32)
        # Belt and braces against float rounding at the ceiling.
        return np.clip(y, -self._ceiling, self._ceiling)


class Ramp:
    """A gain that moves linearly to each new target over RAMP_SAMPLES (10 ms), whatever the block size: the way
    every user-facing gain change is made click-free (Voice volume, the strength mix, the bypass, echo on/off)."""

    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.target = self.value
        self._step = 0.0

    def set(self, target: float) -> None:
        target = float(target)
        if target != self.target:
            self.target = target
            self._step = (target - self.value) / RAMP_SAMPLES

    @property
    def moving(self) -> bool:
        return self.value != self.target

    def gains(self, n: int) -> npt.NDArray[np.float64] | float:
        """The gain for each of the next ``n`` samples (a plain float when it is not moving)."""
        if not self.moving:
            return self.value
        g = self.value + self._step * np.arange(1, n + 1, dtype=np.float64)
        g = np.minimum(g, self.target) if self._step > 0 else np.maximum(g, self.target)
        self.value = float(g[-1])
        return g


class Delay:
    """A fixed delay of ``n`` samples for any block size."""

    def __init__(self, n: int) -> None:
        self._buf = np.zeros(n, np.float32)

    def process(self, x: Audio) -> Audio:
        joined = np.concatenate((self._buf, x))
        self._buf = joined[len(x) :]
        return joined[: len(x)]


def _eq_sos(freq: npt.NDArray[np.float64], gain: npt.NDArray[np.float64], q: npt.NDArray[np.float64]) -> Any:
    """Second-order sections for the five bands: RBJ low shelf (band 0, slope 1), peaking (1-3) and high shelf (4),
    the same formulas as the original (simple-clean-mic src-tauri/src/dsp.rs:39-92), for all bands at once."""
    w = 2.0 * np.pi * freq / SAMPLE_RATE
    s, c = np.sin(w), np.cos(w)
    a = 10.0 ** (gain / 40.0)
    sos = np.empty((len(freq), 6))
    # Peaking.
    alpha = s / (2.0 * q)
    a0 = 1.0 + alpha / a
    sos[:, 0] = (1.0 + alpha * a) / a0
    sos[:, 1] = -2.0 * c / a0
    sos[:, 2] = (1.0 - alpha * a) / a0
    sos[:, 3] = 1.0
    sos[:, 4] = -2.0 * c / a0
    sos[:, 5] = (1.0 - alpha / a) / a0
    # Shelves: sign -1 for the low shelf, +1 for the high shelf (main.js:237-246 writes both with one sign).
    for i, sign in ((0, -1.0), (len(freq) - 1, 1.0)):
        ai, ci = a[i], c[i]
        sa = 2.0 * np.sqrt(ai) * (s[i] / 2.0 * np.sqrt(2.0))
        d0 = (ai + 1) - sign * (ai - 1) * ci + sa
        sos[i] = (
            ai * ((ai + 1) + sign * (ai - 1) * ci + sa) / d0,
            -sign * 2 * ai * ((ai - 1) + sign * (ai + 1) * ci) / d0,
            ai * ((ai + 1) + sign * (ai - 1) * ci - sa) / d0,
            1.0,
            sign * 2 * ((ai - 1) - sign * (ai + 1) * ci) / d0,
            ((ai + 1) - sign * (ai - 1) * ci - sa) / d0,
        )
    return sos


def _band_params(bands: tuple[Band, ...]) -> npt.NDArray[np.float64]:
    """(gain dB, log frequency, log Q) per band: the space the glide moves in."""
    return np.array([[b.gain, math.log(b.freq), math.log(b.q)] for b in bands])


class Equalizer:
    """The 5-band voice EQ (DESIGN Features; the original's bands and formulas, dsp.rs:106-133).

    The original swaps coefficients in one step and keeps the filter memory (dsp.rs:70-76), which can pop on a big
    jump. Here each band glides in (gain dB, log frequency, log Q) to its target over EQ_GLIDE_S: while it moves, the
    block is filtered in 1 ms sub-blocks on a fixed sample grid, with coefficients recomputed for each and the filter
    state carried over, so the output does not depend on the block size. When nothing moves it is one ``sosfilt``.
    """

    def __init__(self, bands: tuple[Band, ...] | None = None) -> None:
        params = _band_params(bands if bands is not None else Settings().bands)
        self._now = params
        self._from = params
        self._to = params
        self._elapsed = 0
        self._glide = int(EQ_GLIDE_S * SAMPLE_RATE)
        self._sos = self._coefs(params)
        self._zi = np.zeros((len(params), 2))
        self._pos = 0  # samples processed so far: the sub-block grid is fixed to it
        self._sub: tuple[int, int, Any, Any, Any, Any] | None = None  # the glide sub-block in progress

    @staticmethod
    def _coefs(p: npt.NDArray[np.float64]) -> Any:
        return _eq_sos(np.exp(p[:, 1]), p[:, 0], np.exp(p[:, 2]))

    @property
    def moving(self) -> bool:
        return self._elapsed < self._glide

    def set_bands(self, bands: tuple[Band, ...]) -> None:
        target = _band_params(bands)
        if np.array_equal(target, self._to):
            return
        self._from, self._to, self._elapsed = self._now, target, 0

    def _next_sub(self) -> None:
        """Start the next sub-block of the glide: run from the current coefficients towards the ones at its end."""
        start = self._pos
        end = (start // CONTROL_STEP + 1) * CONTROL_STEP
        self._elapsed = min(self._glide, self._elapsed + (end - start))
        t = 0.5 - 0.5 * math.cos(math.pi * self._elapsed / self._glide)  # no corners, so no sidebands
        self._now = self._from + (self._to - self._from) * t
        self._sub = (start, end, self._sos, self._coefs(self._now), self._zi, self._zi.copy())

    def process(self, x: Audio) -> Audio:
        n = len(x)
        if not self.moving and self._sub is None:
            y, self._zi = sosfilt(self._sos, x, self._zi)
            self._pos += n
            return y.astype(np.float32)
        out = np.empty(n, np.float32)
        i = 0
        while i < n:
            if self._sub is None:
                if not self.moving:
                    y, self._zi = sosfilt(self._sos, x[i:], self._zi)
                    out[i:] = y
                    self._pos += n - i
                    break
                self._next_sub()
            assert self._sub is not None
            start, end, sos_a, sos_b, zi_a, zi_b = self._sub
            take = min(n - i, end - self._pos)
            chunk = x[i : i + take]
            # Both coefficient sets from the same state, crossfaded across the sub-block: the gain moves every
            # sample instead of in 1 ms stairs, whose steps put images of a tone at multiples of 1 kHz (test E3).
            ya, zi_a = sosfilt(sos_a, chunk, zi_a)  # filters.sosfilt: the 1 ms steps made the wrapper the cost
            yb, zi_b = sosfilt(sos_b, chunk, zi_b)
            w = (np.arange(self._pos - start + 1, self._pos - start + take + 1)) / (end - start)
            out[i : i + take] = ya + (yb - ya) * w
            self._pos += take
            i += take
            if self._pos == end:
                self._sos, self._zi, self._sub = sos_b, zi_b, None
            else:
                self._sub = (start, end, sos_a, sos_b, zi_a, zi_b)
        return out


FLAT_BANDS = bands_for("flat")  # Voice style off: the style's EQ glides to flat (style_bands(preset, 0))


Runnable = obs_filters.Filter | VstPlugin  # a built-in OBS filter, or a loaded VST 2 plug-in


class _Slot:
    """One filter of the running list: its spec, its instance (None: nothing to run: a VST filter with no plug-in
    loaded, which passes audio through as OBS's does, VSTPlugin.cpp:230-233) and its eye's 10 ms crossfade (1: the
    filter's output, 0: its input)."""

    __slots__ = ("eye", "filter", "spec")

    def __init__(self, spec: FilterSpec, filt: Runnable | None) -> None:
        self.spec = spec
        self.filter = filt
        self.eye = Ramp(1.0 if spec.enabled else 0.0)


class FineTune:
    """Fine-tune: the guest's list of OBS filters, top first, with click-free changes (J10 plan §3).

    The order follows OBS: a new filter goes to index 0 of a source's array and the array runs from its end
    (libobs/obs-source.c filter_async_audio :3908-3925), so the list's top runs first; a disabled filter is skipped
    (:3914). Clean Voice's own rule that every change is click-free adds the 10 ms ramps OBS does not have:

    - ``fine_tune`` off: the section crossfades from its output to its input over 10 ms; once fully off, no filter
      in it runs.
    - An eye: that filter's output crossfades with its input; once fully off, the filter is skipped.
    - A setting: the instance's ``update`` between blocks, OBS's semantics (the Noise Gate resets its state).
    - Add, remove or reorder: the section ramps to its input, swaps to the new list at a block boundary, then ramps
      back. An instance whose uid and kind stay in the list is kept, state and all; a new built-in one is made there
      and then. No instance runs twice and none is copied.

    ``applied``: the ``Settings`` whose list is fully in place. It still names a removed filter until the swap.

    VST plug-ins come from the engine's ``vst.VstRegistry`` (``vst``), loaded there by the UI thread; without one (the
    harness, the drift simulation) a VST filter passes audio through. ``held`` is the handshake that lets the UI
    unload a removed plug-in safely: every VST uid this Fine-tune may still call, a uid added before it is taken from
    the registry and dropped once its slot is gone. ``close`` (the engine's, when its streams close) lets go of all.
    """

    def __init__(self, settings: Settings, block: int | None = None, vst: VstRegistry | None = None) -> None:
        self.block = block
        self.instances: dict[int, Runnable] = {}
        self.held: frozenset[int] = frozenset()
        self._closed = False
        self._slots: list[_Slot] = []
        self._want = settings
        self._vst = vst
        if vst is not None:
            vst.attach(self)  # before any plug-in is taken
        self._swap()  # fully in place from the start: nothing to fade
        self._section = Ramp(1.0 if settings.fine_tune else 0.0)  # 1: the filters' output, 0: the input

    def _instance(self, spec: FilterSpec, old: dict[int, _Slot]) -> Runnable | None:
        slot = old.get(spec.uid)
        if slot is not None and slot.filter is not None and slot.spec.kind == spec.kind:
            if slot.spec.settings != spec.settings:
                slot.filter.update(spec.settings)
            return slot.filter
        if spec.kind == VST_KIND:
            return self._vst.get(spec.uid) if self._vst is not None else None
        return obs_filters.make(spec.kind, spec.settings, self.block)

    def _swap(self) -> None:
        s = self._want
        self.held = self.held | {f.uid for f in s.filters if f.kind == VST_KIND}  # held before it is taken
        old = {slot.spec.uid: slot for slot in self._slots}
        self._slots = [_Slot(spec, self._instance(spec, old)) for spec in s.filters]
        self.instances = {slot.spec.uid: slot.filter for slot in self._slots if slot.filter is not None}
        self.held = frozenset(uid for uid, f in self.instances.items() if isinstance(f, VstPlugin))
        self._pending = False
        self.applied = s

    def close(self) -> None:
        """On the thread that runs ``process``, after its last block: run nothing more and hold no plug-in."""
        self._closed = True
        self._slots = []
        self.instances = {}
        self.held = frozenset()

    def retarget(self, s: Settings) -> None:
        """Called by the chain between blocks when the settings changed."""
        self._want = s
        same = [(f.uid, f.kind) for f in s.filters] == [(x.spec.uid, x.spec.kind) for x in self._slots]
        if same and not self._pending:
            for slot, spec in zip(self._slots, s.filters, strict=True):
                if slot.filter is not None and spec.settings != slot.spec.settings:
                    slot.filter.update(spec.settings)
                slot.eye.set(1.0 if spec.enabled else 0.0)
                slot.spec = spec
            self.applied = s
            self._section.set(1.0 if s.fine_tune else 0.0)
        else:
            self._pending = True
            self._section.set(0.0)

    def process(self, x: Audio) -> Audio:
        if self._closed:
            return x
        section = self._section
        if self._pending and not section.moving and section.value == 0.0:
            self._swap()  # at a block boundary, with the section fully at its input
            section.set(1.0 if self._want.fine_tune else 0.0)
        if not section.moving and section.value == 0.0:
            return x
        y = x
        for slot in self._slots:
            eye = slot.eye
            if slot.filter is None or (not eye.moving and eye.value == 0.0):
                continue
            out = slot.filter.process(y)
            g = eye.gains(len(y))
            y = out if isinstance(g, float) and g == 1.0 else (y + g * (out - y)).astype(np.float32)
        g = section.gains(len(x))
        if isinstance(g, float) and g == 1.0:
            return y
        return (x + g * (y - x)).astype(np.float32)


def loudness_db(loudness: float) -> float:
    """Voice volume 0..100 as a gain: 50 is 0 dB, the range is +-12 dB (the default 50 is main.js:200)."""
    return VOLUME_RANGE_DB * (loudness - 50.0) / 50.0


class VoiceVolume:
    """The Voice volume bar: a gain after the compressor, ramped over 10 ms on every change."""

    def __init__(self, loudness: float = 50.0) -> None:
        self._ramp = Ramp(db_to_lin(loudness_db(loudness)))

    def set_loudness(self, loudness: float) -> None:
        self._ramp.set(db_to_lin(loudness_db(loudness)))

    def process(self, x: Audio) -> Audio:
        return (x * self._ramp.gains(len(x))).astype(np.float32)


class Chain:
    """The whole clean-up chain. ``process`` returns a block the same length as its input.

    rumble -> Denoise -> Voice style EQ -> Fine-tune filters -> compressor -> Voice volume -> limiter
    (DESIGN Features, J10). The settings come from a ``SettingsCell``, read once per block.

    - Denoise off: the strength mix's ramp goes to 1.0, fully dry (y = wet + d (dry - wet)). RNNoise keeps running
      and the dry path stays delayed by its latency, so switching is a 10 ms ramp with no jump in timing.
    - Voice style off: the style EQ glides to flat over its 20 ms glide.
    - Fine-tune off: ``FineTune`` crossfades to its input over 10 ms and then runs nothing.

    With ``cleaning`` off, the raw mic, delayed by the chain's latency so the crossfade lines up, goes to the output
    instead, still through the limiter, as the original's Off did (main.js:622, audio.rs:278-287). The chain keeps
    running underneath, so switching back is seamless. Fine-tune's own latency (Noise Suppression, a VST) is not
    compensated, as OBS does not (noise-suppress-filter.c:467 only shifts timestamps).
    """

    STAGES = ("rumble", "denoise", "eq", "fine_tune", "compressor", "volume", "limiter")  # in order, for the bench
    # RNNoise 0.2's own 960 samples (measured by cross-correlation, test S1: the plan expected 480 inside RNNoise, but the
    # wet output lags its input by 960 more than our queue), plus the queue's 480 when the block size is not whole frames.
    RNNOISE_LATENCY = 2 * FRAME_SIZE
    LATENCY = NoiseSuppressor.LATENCY + RNNOISE_LATENCY  # for any block size (``block=None``): 1440 samples, 30 ms

    def __init__(
        self, cell: SettingsCell | None = None, block: int | None = None, vst: VstRegistry | None = None
    ) -> None:
        """``block``: the block size every ``process`` call will use. A multiple of 480 (the worker's 480) saves the
        10 ms output queue (``latency`` 960 instead of ``LATENCY``); None allows any block size. ``vst``: the engine's
        loaded VST plug-ins (none offline)."""
        self.cell = cell if cell is not None else SettingsCell()
        s = self.cell.value
        self.applied: Settings = s  # the settings the stages were last retargeted to
        self.rumble = RumbleFilter()
        self.denoise = NoiseSuppressor(block)
        self.latency = self.denoise.latency + self.RNNOISE_LATENCY
        self._dry = Delay(self.latency)  # the rumble-filtered input, lined up with RNNoise's output
        self._mix = Ramp(self._mix_target(s))
        self.dry: Audio = np.zeros(0, np.float32)  # the last block's dry path (test S1)
        self.eq = Equalizer(self._bands(s))
        self.fine_tune = FineTune(s, block, vst)
        self.compressor = Compressor()
        self.volume = VoiceVolume(s.loudness)
        self.limiter = Limiter()
        self._raw = Delay(self.latency)
        self._bypass = Ramp(0.0 if s.cleaning else 1.0)  # 0: the clean voice, 1: the raw mic

    @staticmethod
    def _mix_target(s: Settings) -> float:
        return strength_mix(s.strength) if s.denoise else 1.0  # Denoise off: all dry

    @staticmethod
    def _bands(s: Settings) -> tuple[Band, ...]:
        return s.bands if s.style else FLAT_BANDS

    def _retarget(self, s: Settings) -> None:
        self.applied = s
        self._mix.set(self._mix_target(s))
        self.eq.set_bands(self._bands(s))
        self.fine_tune.retarget(s)
        self.volume.set_loudness(s.loudness)
        self._bypass.set(0.0 if s.cleaning else 1.0)

    def process(self, x: Audio, raw: Audio | None = None) -> Audio:
        """``raw``: what Off sends instead, when it is not ``x`` (the mic before echo removal)."""
        s = self.cell.value  # read once per block: a block never sees half a change
        if s is not self.applied:
            self._retarget(s)
        rumbled = self.rumble.process(x)
        self.dry = self._dry.process(rumbled)
        wet = self.denoise.process(rumbled)
        # The strength mix: wet + d * (dry - wet), d ramped on a change.
        y = (wet + self._mix.gains(len(x)) * (self.dry - wet)).astype(np.float32)
        y = self.eq.process(y)
        y = self.fine_tune.process(y)
        y = self.compressor.process(y)
        y = self.volume.process(y)
        late_raw = self._raw.process(x if raw is None else raw)
        b = self._bypass.gains(len(x))
        if not isinstance(b, float) or b > 0.0:
            y = (y * (1.0 - b) + late_raw * b).astype(np.float32)
        return self.limiter.process(y)


class PassThrough:
    """A chain that changes nothing, for measuring the audio path on its own."""

    def process(self, x: Audio, raw: Audio | None = None) -> Audio:
        return x.astype(np.float32, copy=True)


def eq_response_db(bands: tuple[Band, ...], freqs: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The EQ's magnitude response in dB at ``freqs`` (Hz): the Fine-tune curve. It is computed from the same
    second-order sections ``Equalizer`` runs (``_eq_sos``), so the drawn curve is the filter, not a second copy of the
    formulas (the original drew its own, main.js:231-258)."""
    p = _band_params(bands)
    sos = _eq_sos(np.exp(p[:, 1]), p[:, 0], np.exp(p[:, 2]))
    f = np.asarray(freqs, dtype=np.float64)
    _, h = signal.sosfreqz(sos, worN=f, fs=SAMPLE_RATE)
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
