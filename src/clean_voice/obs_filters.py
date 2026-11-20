"""OBS Studio's audio filters as Clean Voice DSP, mono at 48 kHz (J10 plan §4).

Each class follows one file under ``reference/obs-studio/plugins/obs-filters/``: its ``update`` (coefficients from the
settings), its processing and its state, written here in our own Python from the algorithm (OBS is GPL, Clean Voice is
MIT; no OBS code is copied). The settings are OBS's own keys, filled in and clamped by ``fine_tune.resolve`` from the
one ``fine_tune.KINDS`` table.

Every filter has ``__init__(settings)``, ``update(settings)`` and ``process(block) -> block``, takes any block length,
keeps its state between calls, and allocates its scratch buffers once (grown only if a longer block arrives, as OBS's
``resize_env_buffer`` does). ``process`` returns a new float32 block and never changes its input.

Mono: OBS runs each channel (or takes the maximum over channels for a shared detector); with one channel both are
the mono sample. OBS computes in float32; these classes compute in float64 and return float32.

The recursions OBS runs per sample (envelope followers, gain ballistics, the gate's state) are plain loops over the
block, as ``dsp.Compressor`` already runs its follower (Numba is ruled out: no macOS x86_64 wheels). Everything that
is not recursive is vectorized, and the 3-Band EQ's one-pole cascades run as second-order sections.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from clean_voice import fine_tune, nvafx
from clean_voice.filters import sosfilt
from clean_voice.ring import Ring

SAMPLE_RATE = 48_000
SCRATCH = 480  # the worker's block: the scratch buffers' first size (OBS: DEFAULT_AUDIO_BUF_MS, 10 ms)

Audio = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


def db_to_mul(db: float) -> float:
    """libobs/media-io/audio-math.h db_to_mul :36-39 (0 for a non-finite dB)."""
    return 10.0 ** (db / 20.0) if math.isfinite(db) else 0.0


def gain_coefficient(time_s: float) -> float:
    """A one-pole follower's coefficient exp(-1 / (rate * time)) (compressor-filter.c gain_coefficient :133-136)."""
    return math.exp(-1.0 / (SAMPLE_RATE * time_s))


def ms_coefficient(ms: float) -> float:
    """``gain_coefficient`` for a time in ms, computed as J06's fixed compressor always did (rate x ms x 0.001), so
    that stage stays bit for bit."""
    return math.exp(-1.0 / (SAMPLE_RATE * ms * 0.001))


class Filter:
    """One OBS audio filter. ``settings``: the full, valid settings it runs (``fine_tune.resolve``)."""

    KIND: ClassVar[str]

    def __init__(self, settings: Mapping[str, object] | None = None) -> None:
        self.settings: dict[str, fine_tune.Value] = {}
        self._scratch: dict[str, F64] = {}
        self.update(settings or {})

    def update(self, settings: Mapping[str, object]) -> None:
        self.settings = fine_tune.resolve(self.KIND, settings)

    def _buf(self, name: str, n: int) -> F64:
        """A preallocated float64 scratch buffer of at least ``n`` samples (its first ``n``)."""
        buf = self._scratch.get(name)
        if buf is None or len(buf) < n:
            buf = np.zeros(max(n, SCRATCH))
            self._scratch[name] = buf
        return buf[:n]

    def process(self, x: Audio) -> Audio:
        raise NotImplementedError

    def _f(self, key: str) -> float:
        return float(self.settings[key])


class Gain(Filter):
    """Gain: multiply by db_to_mul(db).

    Follows plugins/obs-filters/gain-filter.c: gain_update :34-40, gain_filter_audio :50-66, gain_defaults :68-71,
    gain_properties :73-82 (-30..30 dB).
    """

    KIND = "gain_filter"

    def update(self, settings: Mapping[str, object]) -> None:
        super().update(settings)
        self.multiple = db_to_mul(self._f("db"))

    def process(self, x: Audio) -> Audio:
        return (x * self.multiple).astype(np.float32)


class InvertPolarity(Filter):
    """Invert Polarity: every sample times -1.

    Follows plugins/obs-filters/invert-audio-polarity.c: invert_polarity_filter_audio :20-38. No properties.
    """

    KIND = "invert_polarity_filter"

    def process(self, x: Audio) -> Audio:
        return np.negative(x, dtype=np.float32)


class ThreeBandEq(Filter):
    """3-Band Equalizer: low, mid and high bands split at 800 Hz and 5 kHz.

    Follows plugins/obs-filters/eq-filter.c: LOW_FREQ/HIGH_FREQ :8-9, eq_update :44-50 (only the gains change; the
    filter state carries on), eq_create :77-89 (lf = 2 sin(pi f / fs)), EQ_EPSILON :97 and eq_process :99-127.
    Each crossover is four cascaded one-pole low-passes y += c (u - y), the first with the 1/4294967295 offset that
    keeps denormals away. Then h = x[n-3] - hf4, m = x[n-3] - (h + l), and the output is l Gl + m Gm + h Gh.

    Vectorized: a one-pole y[n] = (1 - c) y[n-1] + c u[n] + eps is c / (1 - (1 - c) z^-1) on u + eps / c, so each
    cascade of four is two second-order sections run by ``filters.sosfilt``, with its state carried between blocks.
    The 3-sample delay is a carried buffer. test_obs_filters checks it sample for sample against a per-sample oracle.
    """

    KIND = "basic_eq_filter"
    LOW_FREQ = 800.0
    HIGH_FREQ = 5000.0
    EPSILON = 1.0 / 4294967295.0

    def __init__(self, settings: Mapping[str, object] | None = None) -> None:
        self._lf = 2.0 * math.sin(math.pi * self.LOW_FREQ / SAMPLE_RATE)
        self._hf = 2.0 * math.sin(math.pi * self.HIGH_FREQ / SAMPLE_RATE)
        self._low_sos = self._cascade(self._lf)
        self._high_sos = self._cascade(self._hf)
        self._low_zi = np.zeros((2, 2))
        self._high_zi = np.zeros((2, 2))
        self._delay = np.zeros(3)  # x[n-3], x[n-2], x[n-1]
        super().__init__(settings)

    @staticmethod
    def _cascade(c: float) -> F64:
        """Four one-poles c / (1 - (1 - c) z^-1) as two second-order sections."""
        p = 1.0 - c
        section = [c * c, 0.0, 0.0, 1.0, -2.0 * p, p * p]
        return np.array([section, section])

    def update(self, settings: Mapping[str, object]) -> None:
        super().update(settings)
        self.low_gain = db_to_mul(self._f("low"))
        self.mid_gain = db_to_mul(self._f("mid"))
        self.high_gain = db_to_mul(self._f("high"))

    def process(self, x: Audio) -> Audio:
        n = len(x)
        if n == 0:
            return np.zeros(0, np.float32)
        u = self._buf("u", n)
        u[:] = x
        low, self._low_zi = sosfilt(self._low_sos, u + self.EPSILON / self._lf, self._low_zi)
        high_lp, self._high_zi = sosfilt(self._high_sos, u + self.EPSILON / self._hf, self._high_zi)
        late = self._buf("late", n + 3)
        late[:3] = self._delay
        late[3:] = u
        self._delay[:] = late[n:]
        x3 = late[:n]
        h = x3 - high_lp
        m = x3 - (h + low)
        return (low * self.low_gain + m * self.mid_gain + h * self.high_gain).astype(np.float32)


class NoiseGate(Filter):
    """Noise Gate: opens above the open threshold, closes once the decaying level falls under the close threshold,
    with linear attack and release and a hold before the release.

    Follows plugins/obs-filters/noise-gate-filter.c: noise_gate_update :67-100 (thresholds as multipliers, attack and
    release rates 1 / (time * fs), decay (open - close) / (fs / 75), and the state reset on every update, :96-99),
    noise_gate_filter_audio :110-154 (the per-sample state machine), noise_gate_defaults :156-163. A 0 ms attack or
    release is an infinite rate, as OBS's float division gives: the gain jumps.
    """

    KIND = "noise_gate_filter"

    def update(self, settings: Mapping[str, object]) -> None:
        super().update(settings)
        self.open_threshold = db_to_mul(self._f("open_threshold"))
        self.close_threshold = db_to_mul(self._f("close_threshold"))
        attack_s = self._f("attack_time") / 1000.0
        release_s = self._f("release_time") / 1000.0
        self.attack_rate = 1.0 / (attack_s * SAMPLE_RATE) if attack_s > 0 else math.inf
        self.release_rate = 1.0 / (release_s * SAMPLE_RATE) if release_s > 0 else math.inf
        self.decay_rate = (self.open_threshold - self.close_threshold) / (SAMPLE_RATE / 75.0)
        self.hold_time = self._f("hold_time") / 1000.0
        self.is_open = False
        self.attenuation = 0.0
        self.level = 0.0
        self.held_time = 0.0

    def process(self, x: Audio) -> Audio:
        n = len(x)
        gains = self._buf("gain", n)
        open_t, close_t = self.open_threshold, self.close_threshold
        attack, release, decay = self.attack_rate, self.release_rate, self.decay_rate
        hold, step = self.hold_time, 1.0 / SAMPLE_RATE
        is_open, att, level, held = self.is_open, self.attenuation, self.level, self.held_time
        for i, cur in enumerate(np.abs(x).astype(np.float64).tolist()):
            if cur > open_t and not is_open:
                is_open = True
            if level < close_t and is_open:
                held = 0.0
                is_open = False
            level = max(level, cur) - decay
            if is_open:
                att = min(1.0, att + attack)
            else:
                held += step
                if held > hold:
                    att = max(0.0, att - release)
            gains[i] = att
        self.is_open, self.attenuation, self.level, self.held_time = is_open, att, level, held
        return (x * gains).astype(np.float32)


class _Follower(Filter):
    """The attack/release follower of |x| shared by the Compressor and the Limiter (compressor-filter.c
    analyze_envelope :294-321, limiter-filter.c analyze_envelope :124-151), and their gain: slope x (threshold -
    envelope dB), reduction only, times the output gain (compressor-filter.c process_compression :356-369,
    limiter-filter.c :153-166). The envelope carries between blocks; ``update`` does not reset it."""

    attack: float
    release: float
    threshold: float
    slope: float
    output_gain: float

    def __init__(self, settings: Mapping[str, object] | None = None) -> None:
        self._envelope = 0.0
        super().__init__(settings)

    def process(self, x: Audio) -> Audio:
        n = len(x)
        if n == 0:
            return x.copy()
        env_buf = self._buf("env", n)
        env, attack, release = self._envelope, self.attack, self.release
        for i, v in enumerate(np.abs(x).astype(np.float64).tolist()):
            env = v + (attack if env < v else release) * (env - v)
            env_buf[i] = env
        self._envelope = env
        # mul_to_db's -inf for a zero envelope (audio-math.h :31-34) only ever meant "no reduction": the floor gives
        # the same gain without a log of zero.
        env_db = 20.0 * np.log10(np.maximum(env_buf, 1e-30))
        reduction_db = np.minimum(0.0, self.slope * (self.threshold - env_db))
        return (x * (np.power(10.0, reduction_db / 20.0) * self.output_gain)).astype(np.float32)


class Compressor(_Follower):
    """Compressor: a downward compressor with an output gain.

    Follows plugins/obs-filters/compressor-filter.c: compressor_update :181-199 (coefficients exp(-1 / (fs t)) with
    the attack and release in whole ms, slope 1 - 1/ratio, output gain db_to_mul), analyze_envelope :294-321,
    process_compression :356-369, compressor_defaults :438-446. The Sidechain/Ducking Source is left out (J10 plan,
    question 7). ``dsp.Compressor``, the chain's fixed stage, is this class at the J06 values.
    """

    KIND = "compressor_filter"

    def update(self, settings: Mapping[str, object]) -> None:
        super().update(settings)
        self.attack = ms_coefficient(self._f("attack_time"))
        self.release = ms_coefficient(self._f("release_time"))
        self.threshold = self._f("threshold")
        self.slope = 1.0 - 1.0 / self._f("ratio")
        self.output_gain = db_to_mul(self._f("output_gain"))


class Limiter(_Follower):
    """Limiter: OBS's, exactly. A separate class from ``dsp.Limiter`` (the chain's fixed -1 dBFS stage, whose linear-dB
    release is J06's own).

    Follows plugins/obs-filters/limiter-filter.c: ATK_TIME :39 is 0.001, used as milliseconds and divided by 1000 again
    in limiter_update :88, :95, so the attack is 1 us; the release is a one-pole of release_time ms (:90, :96); slope 1
    and no output gain (:91, :97, :100); analyze_envelope :124-151; process_compression :153-166;
    limiter_defaults :182-186.
    """

    KIND = "limiter_filter"
    ATTACK_MS = 0.001

    def update(self, settings: Mapping[str, object]) -> None:
        super().update(settings)
        self.attack = gain_coefficient(self.ATTACK_MS / 1000.0)
        self.release = gain_coefficient(self._f("release_time") / 1000.0)
        self.threshold = self._f("threshold")
        self.slope = 1.0
        self.output_gain = 1.0


class _Expander(Filter):
    """The Expander and the Upward Compressor: one file, one detector, two gain laws.

    Follows plugins/obs-filters/expander-filter.c: expander_update :179-231, analyze_envelope :275-320 (a 10 ms RMS
    one-pole, rmscoef = 2^(-100 / fs), or the peak |x|), process_sample :322-374 and process_expansion :377-407 (the
    gain, then dB-domain attack/release against the previous sample's gain, then db_to_mul and the output gain).
    The RMS one-pole is vectorized (``scipy.signal.lfilter`` with its state carried); the ballistics are a loop.
    """

    UPWARD: ClassVar[bool] = False

    def __init__(self, settings: Mapping[str, object] | None = None) -> None:
        self._runave = 0.0  # cd->runave: the detector's state
        self._gain_db = 0.0  # cd->gain_db_buf: the last sample's gain
        self._rmscoef = 2.0 ** (-100.0 / SAMPLE_RATE)
        super().__init__(settings)

    def update(self, settings: Mapping[str, object]) -> None:
        super().update(settings)
        s = self.settings
        self.attack = gain_coefficient(self._f("attack_time") / 1000.0)
        self.release = gain_coefficient(self._f("release_time") / 1000.0)
        self.threshold = self._f("threshold")
        self.slope = 1.0 - self._f("ratio")
        self.output_gain = db_to_mul(self._f("output_gain"))
        self.knee = self._f("knee_width") if self.UPWARD else 0.0
        self.rms = self.UPWARD or s.get("detector") == "RMS"

    def _envelope_db(self, x: Audio) -> F64:
        n = len(x)
        sq = self._buf("sq", n)
        np.square(x, out=sq, dtype=np.float64)
        if self.rms:
            from scipy.signal import lfilter

            c = self._rmscoef
            runave, _ = lfilter([1.0 - c], [1.0, -c], sq, zi=[c * self._runave])
            self._runave = float(runave[-1])
            env = np.sqrt(np.maximum(runave, 0.0))
        else:
            self._runave = float(sq[-1])  # OBS keeps x^2 as the detector state in peak mode too (:310, :315)
            env = np.abs(x).astype(np.float64)
        with np.errstate(divide="ignore"):
            return 20.0 * np.log10(env)  # mul_to_db: -inf for a zero envelope (audio-math.h :31-34)

    def _target_db(self, env_db: F64) -> F64:
        raise NotImplementedError

    def process(self, x: Audio) -> Audio:
        n = len(x)
        if n == 0:
            return x.copy()
        target = self._target_db(self._envelope_db(x)).tolist()
        gains = self._buf("gain", n)
        a, r = self.attack, self.release
        ia, ir = 1.0 - a, 1.0 - r
        prev = self._gain_db
        upward = self.UPWARD
        for i, g in enumerate(target):
            if upward and prev < 0.0:
                prev = 0.0
            prev = a * prev + ia * g if g > prev else r * prev + ir * g
            gains[i] = prev
        self._gain_db = prev
        if not upward:
            np.minimum(gains, 0.0, out=gains)
        return (x * (np.power(10.0, gains / 20.0) * self.output_gain)).astype(np.float32)


class Expander(_Expander):
    """Expander (and its Gate preset): turns the signal down below the threshold, by (1 - ratio) x the difference,
    never more than 60 dB.

    Follows plugins/obs-filters/expander-filter.c: expander_defaults :153-166, expander_update :179-231 (switching
    the preset clears the settings to that preset's defaults, :182-196; a filter created on the gate preset keeps its
    values, compressor_expander_create :242-245), process_sample's expander branches :352-353, :367-368 (the gain is
    fmaxf(slope x diff, -60): a NaN from a 1:1 ratio on silence also gives -60, as fmaxf does), expander_properties
    :432-468.
    """

    KIND = "expander_filter"

    def __init__(self, settings: Mapping[str, object] | None = None) -> None:
        self.is_gate = (settings or {}).get("presets") == "gate"
        super().__init__(settings)

    def update(self, settings: Mapping[str, object]) -> None:
        preset = settings.get("presets")
        if preset == "expander" and self.is_gate:
            settings, self.is_gate = {"presets": "expander"}, False
        elif preset == "gate" and not self.is_gate:
            settings, self.is_gate = {"presets": "gate"}, True
        super().update(settings)

    def _target_db(self, env_db: F64) -> F64:
        diff = self.threshold - env_db
        with np.errstate(invalid="ignore"):
            g = np.fmax(self.slope * diff, -60.0)
        return np.where(diff > 0.0, g, 0.0)


class UpwardCompressor(_Expander):
    """Upward Compressor: turns the signal up below the threshold, by (1 - ratio) x the difference, with a soft knee.

    Follows plugins/obs-filters/expander-filter.c: upward_compressor_defaults :168-177, expander_update :179-231,
    process_sample's is_upwcomp branches :333-334 (at or below (threshold - 60) / 2 the difference becomes env + 60,
    floored at 0), :340-350 (no gain above the knee, slope x diff below it, slope (diff + knee/2)^2 / (2 knee) in it;
    the previous gain floored at 0) and :369-370 (db_to_mul of the gain, no ceiling). The detector is RMS (its
    default, with no list in expander_properties :458-466).
    """

    KIND = "upward_compressor_filter"
    UPWARD = True

    def _target_db(self, env_db: F64) -> F64:
        thr, knee, slope = self.threshold, self.knee, self.slope
        diff = thr - env_db
        low = env_db <= (thr - 60.0) / 2.0
        diff = np.where(low, np.where(env_db + 60.0 > 0.0, env_db + 60.0, 0.0), diff)
        gain = np.zeros_like(diff)
        below = thr - knee / 2.0 >= env_db
        gain = np.where(below, slope * diff, gain)
        if knee > 0.0:
            inside = (env_db > thr - knee / 2.0) & (thr + knee / 2.0 > env_db)
            gain = np.where(inside, slope * (diff + knee / 2.0) ** 2 / (2.0 * knee), gain)
        return gain


class NoiseSuppression(Filter):
    """Noise Suppression with RNNoise, fully wet (OBS has no RNNoise level).

    Follows plugins/obs-filters/noise-suppress-filter.c: noise_suppress_update :190-255 (10 ms segments),
    noise_suppress_filter_audio :396-469 (queue, process each whole segment, hand on what is ready),
    noise_suppress_defaults_v2 :493-501 (RNNoise), noise_suppress_properties :503-526 (no properties without Speex).
    It is ``dsp.NoiseSuppressor``, which already follows that file. ``block``: the chain's block size (a multiple of
    480 adds no queue latency; None takes any block size with 10 ms more).
    """

    KIND = "noise_suppress_filter"

    def __init__(self, settings: Mapping[str, object] | None = None, block: int | None = 480) -> None:
        from clean_voice.dsp import NoiseSuppressor  # dsp builds its fixed compressor on this module

        self._rnnoise = NoiseSuppressor(block)
        super().__init__(settings)

    @property
    def latency(self) -> int:
        return self._rnnoise.latency

    def process(self, x: Audio) -> Audio:
        return self._rnnoise.process(x)


class Denoise(Filter):
    """Clean Voice's own Denoise filter (DESIGN J13; OBS has none): the chain's Denoise stage, RNNoise and the
    Strength mix, ``dsp.Chain``'s own parts: ``NoiseSuppressor``, ``strength_mix``, the 10 ms ``Ramp`` on a change and
    the dry path delayed by RNNoise's latency. Fully wet at Strength 100. ``block`` is as for ``NoiseSuppression``."""

    KIND = fine_tune.DENOISE_KIND

    def __init__(self, settings: Mapping[str, object] | None = None, block: int | None = 480) -> None:
        from clean_voice import dsp  # dsp builds its fixed compressor on this module

        self._rnnoise = dsp.NoiseSuppressor(block)
        self.latency = self._rnnoise.latency + dsp.Chain.RNNOISE_LATENCY  # the wet output's delay, dry matches it
        self._dry = dsp.Delay(self.latency)
        self._mix = dsp.Ramp(dsp.strength_mix(float(fine_tune.resolve(self.KIND, settings)["strength"])))  # type: ignore[arg-type]
        super().__init__(settings)

    def update(self, settings: Mapping[str, object]) -> None:
        from clean_voice import dsp

        super().update(settings)
        self._mix.set(dsp.strength_mix(self._f("strength")))  # a 10 ms ramp, as the chain's

    def process(self, x: Audio) -> Audio:
        dry = self._dry.process(x)
        wet = self._rnnoise.process(x)
        return (wet + self._mix.gains(len(x)) * (dry - wet)).astype(np.float32)


class VoiceStyle(Filter):
    """Clean Voice's own Voice Style filter (DESIGN J13; OBS has none): the chain's style EQ, ``settings.style_bands``
    into ``dsp.Equalizer``, whose 20 ms glide makes a change of style or Strength click-free."""

    KIND = fine_tune.STYLE_KIND

    def __init__(self, settings: Mapping[str, object] | None = None) -> None:
        from clean_voice import dsp

        self._eq = dsp.Equalizer(self._bands(fine_tune.resolve(self.KIND, settings)))
        super().__init__(settings)

    @staticmethod
    def _bands(settings: Mapping[str, Any]) -> Any:
        from clean_voice.settings import style_bands

        return style_bands(str(settings["style"]), float(settings["strength"]))

    def update(self, settings: Mapping[str, object]) -> None:
        super().update(settings)
        self._eq.set_bands(self._bands(self.settings))

    def process(self, x: Audio) -> Audio:
        return self._eq.process(x)


class NvidiaAudioEffects(Filter):
    """NVIDIA Audio Effects: noise removal, room echo removal, or both, on NVIDIA's runtime (J12), mono at 48 kHz.

    Follows plugins/nv-filters/nvidia-audiofx-filter.c: the settings are method, intensity and Voice Optimized
    (nvidia_audio_update :444-527, nvidia_audio_defaults :807-813); the audio is cut into 10 ms segments of 480 samples
    that wait in a queue until whole (BUFFER_SIZE_MSEC :41, nvidia_audio_filter_audio :739-805, process :704-717); each
    goes through ``nvafx.Effect`` (process_fx :625-702; Clean Voice's chain is already 48 kHz mono, so OBS's
    resamplers and its mono-to-stereo copy are not needed). The effect loads on a thread (``nvafx.Effect``): until it
    is ready the audio passes through unchanged, as in OBS.

    Without the runtime (a saved filter on a computer that has none, or macOS) there is no effect: the audio passes
    through, as a filter whose module is missing does in OBS (libobs/obs-source.c filter_async_audio :3917 runs only
    a filter with ``context.data``). ``block``: as for ``NoiseSuppression``: whole 480-sample blocks add no queue
    latency, any other size adds 10 ms.
    """

    KIND = fine_tune.NVIDIA_KIND

    def __init__(self, settings: Mapping[str, object] | None = None, block: int | None = 480) -> None:
        self._effect: nvafx.Effect | None = None
        self._checked = False  # the runtime check has answered, and an effect was made if it found the runtime
        self.whole_frames = block is not None and block % nvafx.FRAME_SIZE == 0
        self.latency = 0  # 10 ms for any other block size, once an effect exists (the queue, below)
        self._pending = np.zeros(nvafx.FRAME_SIZE, np.float32)
        self._pending_len = 0
        self._frame = np.zeros(nvafx.FRAME_SIZE, np.float32)
        self._out = Ring(1 << 16)
        super().__init__(settings)
        self._effect_now()

    def _effect_now(self) -> nvafx.Effect | None:
        """The effect, made as soon as the runtime check (started at launch, off every audio thread) has found the
        runtime: this never waits for it, so until then the audio passes through, as with no runtime at all."""
        if self._effect is None and not self._checked:
            checked, rt = nvafx.peek()
            if checked:
                self._checked = True
                if rt is not None and rt.api is not None:
                    self._effect = nvafx.Effect(rt, *self._params())
                    if not self.whole_frames:
                        self.latency = nvafx.FRAME_SIZE
                        self._out.write(np.zeros(self.latency, np.float32))  # fixed, so every call returns a block
        return self._effect

    def _params(self) -> tuple[str, float, bool]:
        return str(self.settings["method"]), self._f("intensity"), self.settings["vad"] is True

    def update(self, settings: Mapping[str, object]) -> None:
        super().update(settings)
        if self._effect is not None:
            self._effect.set(*self._params())

    @property
    def failed(self) -> bool:
        """The effect could not load or run (OBS switches the filter off then, :333-345); the audio passes through."""
        return self._effect is not None and self._effect.failed

    def close(self) -> None:
        if self._effect is not None:
            self._effect.close()
            self._effect = None
        self._checked = True

    def process(self, x: Audio) -> Audio:
        effect = self._effect_now()
        if effect is None:
            return x.astype(np.float32, copy=True)
        size = len(self._pending)
        pos = 0
        while pos < len(x):
            take = min(size - self._pending_len, len(x) - pos)
            self._pending[self._pending_len : self._pending_len + take] = x[pos : pos + take]
            self._pending_len += take
            pos += take
            if self._pending_len == size:
                self._frame[:] = self._pending
                effect.run(self._frame)  # untouched while the effect loads or has failed
                self._out.write(self._frame)
                self._pending_len = 0
        y = np.empty(len(x), np.float32)
        if self._out.read_into(y) < len(x):
            raise ValueError(f"a {len(x)}-sample block: this chain was made for whole 480-sample frames")
        return y


CLASSES: dict[str, type[Filter]] = {
    c.KIND: c
    for c in (
        Denoise,
        VoiceStyle,
        NvidiaAudioEffects,
        ThreeBandEq,
        Compressor,
        Expander,
        Gain,
        InvertPolarity,
        Limiter,
        NoiseGate,
        NoiseSuppression,
        UpwardCompressor,
    )
}


def make(kind: str, settings: Mapping[str, object] | None = None, block: int | None = 480) -> Filter:
    """A built-in filter of ``kind`` (any ``fine_tune.KINDS`` id but ``vst_filter``). ``block``: the chain's block
    size, which only Noise Suppression's queue needs."""
    if kind == NoiseSuppression.KIND:
        return NoiseSuppression(settings, block=block)
    if kind == Denoise.KIND:
        return Denoise(settings, block=block)
    if kind == NvidiaAudioEffects.KIND:
        return NvidiaAudioEffects(settings, block=block)
    cls: Any = CLASSES[kind]
    return cls(settings)
