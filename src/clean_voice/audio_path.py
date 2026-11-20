r"""The audio path between the device callbacks, with no devices or threads in it.

    mic callback --copy--> input ring --worker: chain, WAVs, drift resampler--> output ring --copy--> cable callback

The mic and VB-CABLE run on different clocks. Instead of dropping or starving like the original app
(docs/simple-clean-mic-audio-report.md, cause 1), the worker resamples the clean voice to the cable's clock, within
+-0.2 %. Each device callback copies its audio and stamps one (time, frames so far) entry, as OBS stamps every capture
packet (plugins/win-wasapi/win-wasapi.cpp:1062-1070). A ``ClockEstimator`` per stream turns the stamps into the
device's true rate. OBS never estimates a device's rate (its mix is clocked by the system); this is a needed deviation
for the cable's own clock, which borrows OBS's timestamp thresholds (70 ms jitter, 2 s jump; libobs/obs-source.c:1493,
:1513-1524, :1640-1643).
The ratio of the two rates is the feed-forward correction; a small trim on the ring fill holds the cushion. Late audio
adds buffering instead of silence (libobs/obs-audio.c:361-422, add_audio_buffering). An underrun fades out and back in
and is counted; nothing is ever skipped. OBS's output is clocked by the system; Clean Voice plays into another
device's clock, so it also needs the adaptive resampling (as JACK's alsa_out and PipeWire do).
"""

from __future__ import annotations

import enum
import logging
import math
import time as _time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import samplerate

from clean_voice.dsp import Ramp
from clean_voice.ring import Ring
from clean_voice.wav import Recorder

SAMPLE_RATE = 48_000  # the chain's rate
BLOCK = 480  # the worker's processing block: 10 ms at 48 kHz
# The start cushion: max(this, the largest cable callback + fade + 10 ms). DESIGN records about 60 ms, the
# orchestrator's J05 decision (review r2, F1). The bursty mic's rate is trusted only after about 100 s (CLOCK_WINDOW_S),
# so until then the fill trim alone holds the cushion, and with the cable 100 ppm or more faster than the mic a 40 ms
# start drained to an underrun within a minute on the real callback pattern. Late audio still adds buffering on top.
TARGET_FILL_S = 0.060
MAX_CORRECTION = 0.002  # +-0.2 %
FADE_S = 0.003
INPUT_RING_S = 2.0
OUTPUT_RING_S = 1.0
MAX_CALLBACK_FRAMES = 16_384
STAMPS = 4096  # timestamp entries per stream: 40 s of 10 ms callbacks, far more than the worker ever lags

# Timestamp handling (O3). OBS smooths audio timestamps that are within TS_SMOOTHING_THRESHOLD of where it expects
# them (libobs/obs-source.c:1493, :1640-1643) and resets its timing on a jump over MAX_TS_VAR
# (libobs/obs-internal.h:1070, obs-source.c:1513-1524, handle_ts_jump).
TS_SMOOTHING_S = 0.070
MAX_TS_JUMP_S = 2.0
CLOCK_WINDOW_S = 120.0  # a bursty mic needs about this long for a sure fit (J05 review r1, F4)
CLOCK_BIN_S = 0.5
CLOCK_READY_S = 10.0  # at least this much history before a rate is trusted
# ...and its fit must be this sure: the standard error of the slope, in ppm. On the real bursty mic a fit taken at
# 10 s regardless was off by up to 1433 ppm (J05 review r1, F4); behind this gate, replaying every recorded real
# trace from its stream start, no admitted rate was more than 29 ppm off.
CLOCK_CONFIDENT_PPM = 10.0
OUTLIER_LOG_S = 10.0  # at most one "off its clock" line per stream this often

# The trim on the fill: 10 ms of error -> 50 ppm, at most +-500 ppm, on the fill averaged over 1 s.
TRIM_PER_S = 50e-6 / 0.010
MAX_TRIM = 500e-6
FILL_AVERAGE_S = 1.0

# Late audio adds latency at once, as OBS's add_audio_buffering does (libobs/obs-audio.c:361-390): when a source's
# data starts after the time the mix needs (ts->start > min_ts, :377), the lateness is rounded up to whole ticks
# (:378-379) and added to the buffering in one go (:381), capped at the maximum (:383-387, "Max audio buffering
# reached!"). Here the cable's callbacks measure the lateness: the least the ring held beyond what a callback needed
# (its spare). Every block the worker looks at it; under SAFETY_S, before the ring runs dry, the shortfall is rounded
# up to 10 ms ticks and added at once: the block fades out, that much silence follows, the next block fades in, and
# the target rises by the same amount, so the trim keeps it. An underrun adds its measured deficit the same way (the
# underrun itself already played the silence). OBS's cap (libobs/obs.c:1619-1625) is far larger; 150 ms is the
# user's choice for lip sync. The cushion is never lowered while the devices stay open.
PRIME_SLACK_S = 0.001  # playback (re)starts once the ring holds the target less this (see Sink._prime_slack)
SAFETY_S = 0.010
BUFFERING_STEP_S = 0.010  # the tick lateness is rounded up to (OBS: AUDIO_OUTPUT_FRAMES)
MAX_TARGET_S = 0.150

Audio = npt.NDArray[np.float32]

log = logging.getLogger(__name__)


class Processor(Protocol):
    def process(self, x: Audio, raw: Audio | None = None) -> Audio: ...


@dataclass
class Counters:
    underruns: int = 0  # the cable callback found the output ring empty (faded, counted)
    input_overflows: int = 0  # the input ring was full: mic samples lost (the worker fell behind)
    output_overflows: int = 0  # the output ring was full: processed samples lost
    input_device_errors: int = 0  # PortAudio reported an input over/underflow
    output_device_errors: int = 0  # PortAudio reported an output under/overflow
    monitor_underruns: int = 0  # Hear myself / the mic test: the speaker sink ran dry (faded, counted)
    monitor_overflows: int = 0  # ...or its ring was full
    monitor_device_errors: int = 0  # ...or PortAudio reported an output under/overflow on the headphones or speakers
    reference_gaps: int = 0  # echo removal: the loopback ran short, zeros were fed for the shortfall
    reference_overflows: int = 0  # ...or the loopback ring was full: speaker audio lost

    def __add__(self, other: Counters) -> Counters:
        """Field by field, so a new counter is never left out of the totals."""
        return Counters(**{k: getattr(self, k) + getattr(other, k) for k in vars(self)})


class MicTestState(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"  # copying what others hear into a 5 s buffer
    PLAYING = "playing"  # playing it back on the speakers


MIC_TEST_S = 5.0  # like Google Meet's "Check your audio": a few seconds of what others hear
MIC_TEST_TAIL_S = 0.100  # silence after the playback, before the speaker sink is let to run dry


class MicTest:
    """Test your mic: records exactly what goes to the cable, then plays it back once. Worker thread only."""

    def __init__(self) -> None:
        self.buffer = np.zeros(int(MIC_TEST_S * SAMPLE_RATE), np.float32)  # preallocated
        self.state = MicTestState.IDLE
        self.recorded = 0
        self.played = 0
        self._tail = 0
        self._silence = np.zeros(BLOCK, np.float32)

    def start(self) -> None:
        self.state, self.recorded, self.played, self._tail = MicTestState.RECORDING, 0, 0, 0

    def cancel(self) -> None:
        self.state = MicTestState.IDLE

    @property
    def seconds(self) -> float:
        n = self.recorded if self.state is MicTestState.RECORDING else self.played
        return n / SAMPLE_RATE

    def record(self, clean: Audio) -> None:
        n = min(len(clean), len(self.buffer) - self.recorded)
        self.buffer[self.recorded : self.recorded + n] = clean[:n]
        self.recorded += n
        if self.recorded >= len(self.buffer):
            self.state = MicTestState.PLAYING

    def next_block(self) -> Audio:
        """The next 10 ms of the playback, then MIC_TEST_TAIL_S of silence; then the test is over (IDLE)."""
        if self.played < self.recorded:
            block = self.buffer[self.played : self.played + BLOCK]
            self.played += len(block)
            return block
        self._tail += BLOCK
        if self._tail >= MIC_TEST_TAIL_S * SAMPLE_RATE:
            self.state = MicTestState.IDLE
        return self._silence


def callback_time(time: Any) -> float:
    """When the callback was due: PortAudio's ``currentTime``, taken before the callback waited for the GIL.

    On WASAPI it is on ``time.perf_counter``'s clock (measured in J05 Step 2a); the buffer ADC time is not usable
    there (it ran up to 30 ms either side of the call). Without stream time info, the time at entry.
    """
    t = time.currentTime if time is not None else 0.0
    return t if t > 0.0 else _time.perf_counter()


class Stamps:
    """One (time, frames so far, fill) entry per device callback, in preallocated arrays.

    The callback writes an entry and then bumps ``count``; the worker reads entries up to ``count``.
    """

    def __init__(self) -> None:
        self.time = np.zeros(STAMPS)
        self.frames = np.zeros(STAMPS)
        self.fill = np.zeros(STAMPS)
        self.count = 0  # entries ever written; only the callback changes it
        self.read = 0  # entries the worker has taken
        self.total_frames = 0

    def put(self, t: float, frames: int, fill: float = 0.0) -> None:
        i = self.count % STAMPS
        self.time[i] = t
        self.frames[i] = self.total_frames
        self.fill[i] = fill
        self.total_frames += frames
        self.count += 1


class ClockEstimator:
    """A device's true sample rate, from the (time, frames so far) stamps of its callbacks.

    OBS has no rate estimator: it smooths each timestamp to where it expects it and re-places audio that drifts too
    far (libobs/obs-source.c:1613-1683). This is a needed deviation for the cable's own clock, and it borrows OBS's
    thresholds: a stamp within 70 ms of the device's fitted clock is normal jitter (TS_SMOOTHING_THRESHOLD, :1493); one
    70 ms to 2 s off is logged and left out; a jump over 2 s (MAX_TS_VAR, a stall or a resume) restarts the estimate.

    Device callbacks come late, never early, so the earliest stamp of each half second lies closest to the device's
    clock. The rate is the least-squares slope through those earliest stamps over the last 120 s (CLOCK_WINDOW_S).
    A fit is used once at least 10 s are in and the slope's standard error is under CLOCK_CONFIDENT_PPM; ``rate``
    stays the nominal rate until then, and afterwards keeps the last confident value whenever a fit is less sure.
    """

    def __init__(self, nominal_rate: float, name: str) -> None:
        self.nominal = float(nominal_rate)
        self.name = name
        self.outliers = 0
        self.resets = 0
        self._bins: deque[tuple[float, float, float]] = deque(maxlen=int(CLOCK_WINDOW_S / CLOCK_BIN_S))
        self._last_log = -math.inf
        self._reset()

    def _reset(self) -> None:
        self._bins.clear()
        self._bin_t0: float | None = None  # when the open bin started
        self._best: tuple[float, float, float] | None = None  # (time, frames, offset) earliest in the open bin
        self._first_t = 0.0
        self._line: tuple[float, float, float] | None = None  # (frames, offset, slope) of the fitted clock
        self.ready = False
        self.rate = self.nominal
        self.uncertainty_ppm = math.inf  # the latest fit's standard error

    def add(self, t: float, frames: float) -> None:
        offset = t - frames / self.nominal  # seconds ahead of an ideal clock at the nominal rate
        if self._bin_t0 is not None:
            off = offset - self._predict(frames)
            if abs(off) > MAX_TS_JUMP_S:
                log.info("%s: timing jumped by %.0f ms, restarting the rate estimate", self.name, off * 1000)
                self.resets += 1
                self._reset()
            elif abs(off) > TS_SMOOTHING_S:
                self.outliers += 1
                if t - self._last_log >= OUTLIER_LOG_S:
                    self._last_log = t
                    log.info(
                        "%s: a packet came %.0f ms off its clock; left out of the rate estimate (%d so far)",
                        self.name,
                        off * 1000,
                        self.outliers,
                    )
                return
        if self._bin_t0 is None:
            self._bin_t0 = self._first_t = t
            self._best = (t, frames, offset)
            return
        if t - self._bin_t0 >= CLOCK_BIN_S:
            assert self._best is not None
            self._bins.append(self._best)
            self._bin_t0 = t
            self._best = (t, frames, offset)
            self._fit(t)
        elif self._best is None or offset < self._best[2]:
            self._best = (t, frames, offset)

    def _predict(self, frames: float) -> float:
        if self._line is not None:
            f0, y0, slope = self._line
            return y0 + slope * (frames - f0)
        lowest = min((b[2] for b in self._bins), default=math.inf)
        return min(lowest, self._best[2]) if self._best is not None else lowest

    def _fit(self, now: float) -> None:
        bins = self._bins
        if len(bins) < 3:
            return
        f0 = bins[0][1]
        x = np.fromiter((b[1] - f0 for b in bins), np.float64, len(bins))
        y = np.fromiter((b[2] for b in bins), np.float64, len(bins))
        xm, ym = float(x.mean()), float(y.mean())
        xc = x - xm
        sxx = float(np.dot(xc, xc))
        if sxx <= 0.0:
            return
        slope = float(np.dot(xc, y - ym)) / sxx  # seconds per frame, beyond 1 / nominal
        resid = y - ym - slope * xc
        stderr = math.sqrt(float(np.dot(resid, resid)) / (len(bins) - 2) / sxx)
        self._line = (xm + f0, ym, slope)
        self.uncertainty_ppm = stderr * self.nominal * 1e6
        if now - self._first_t >= CLOCK_READY_S and self.uncertainty_ppm < CLOCK_CONFIDENT_PPM:
            self.ready = True
            self.rate = 1.0 / (1.0 / self.nominal + slope)


class DriftController:
    """The resampling correction: feed-forward from the two clocks, plus a trim that holds the cushion."""

    def __init__(self, out_rate: int, trim_per_s: float = TRIM_PER_S, max_trim: float = MAX_TRIM) -> None:
        self.out_rate = out_rate
        self.trim_per_s = trim_per_s
        self.max_trim = max_trim
        self.target = TARGET_FILL_S * out_rate
        self.feed_forward = 0.0
        self.trim = 0.0
        self.correction = 0.0
        self.min_correction = 0.0
        self.max_correction = 0.0

    def raise_target(self, frames: float) -> None:
        """The cushion was raised to ``frames`` (the ring was given the difference at once): hold it there."""
        self.target = frames

    def update(self, fill_average: float | None, feed_forward: float, running: bool) -> float:
        """Returns the correction factor (ratio = nominal * (1 + correction))."""
        self.feed_forward = feed_forward
        trim = 0.0
        if running and fill_average is not None:
            error_s = (self.target - fill_average) / self.out_rate
            trim = min(self.max_trim, max(-self.max_trim, self.trim_per_s * error_s))
        self.trim = trim
        self.correction = min(MAX_CORRECTION, max(-MAX_CORRECTION, feed_forward + trim))
        self._record()
        return self.correction

    def _record(self) -> None:
        self.min_correction = min(self.min_correction, self.correction)
        self.max_correction = max(self.max_correction, self.correction)


class Sink:
    """One output device fed by the worker. ``callback`` only copies from the ring and stamps its time."""

    def __init__(
        self,
        out_rate: int,
        out_channels: int,
        counters: Counters | None = None,
        name: str = "cable",
        source: ClockEstimator | None = None,
    ) -> None:
        self.out_rate = out_rate
        self.out_channels = out_channels
        self.name = name
        self.counters = counters if counters is not None else Counters()
        self.ring = Ring(int(OUTPUT_RING_S * out_rate), 1)
        self.controller = DriftController(out_rate)
        self.clock = ClockEstimator(out_rate, name)
        self.source = source  # the clock of the audio arriving, for the feed-forward
        self.stamps = Stamps()
        self._resampler = samplerate.Resampler("sinc_fastest", channels=1)
        self._nominal_ratio = out_rate / SAMPLE_RATE
        self._blocks = 0
        self._safety = SAFETY_S * out_rate
        # The resampler holds back a few frames of its first block (20 at 48 kHz), so a ring that has been handed
        # exactly the target holds slightly less; waiting for those frames meant waiting for the next block.
        self._prime_slack = int(PRIME_SLACK_S * out_rate)
        self._step = BUFFERING_STEP_S * out_rate
        self._max_target = MAX_TARGET_S * out_rate
        self._max_warned = False
        self._underruns_seen = self.counters.underruns  # obs-audio.c:361-390: only lateness now adds buffering
        self._fill_sum = 0.0
        self._fill_n = 0
        self._fill_t0: float | None = None
        self.fill_average: float | None = None  # the fill before each read, averaged over the last full second
        self.buffering_changes = 0
        self._fade48 = np.linspace(1.0, 0.0, max(1, int(FADE_S * SAMPLE_RATE)), endpoint=False, dtype=np.float32)
        self._fade_in_block = False  # the next pushed block starts with a fade-in (after inserted silence)
        # Callback-side state, all preallocated.
        self._scratch = np.zeros(MAX_CALLBACK_FRAMES, np.float32)
        fade = max(1, int(FADE_S * out_rate))
        self._fade_in = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        self._fade_out = self._fade_in[::-1].copy()
        self.prime = int(self.controller.target)  # playback (re)starts once the ring holds this much
        self.primed = False
        self.ever_primed = False
        self._fade_in_next = True
        # The least the ring held beyond what a callback needed, since the worker last looked.
        self.low_water: float = math.inf
        self.callbacks = 0
        self.frames_played = 0
        # Set by the worker when the sink is meant to run dry (the mic test's playback has ended): the empty ring then
        # fades out as usual but is not an underrun, and adds no buffering.
        self.finishing = False

    # ------------------------------------------------------------------ worker

    def push(self, clean: Audio) -> None:
        """Resample one 48 kHz block to the device clock and queue it; add buffering first if the audio came late."""
        self._read_stamps()
        insert = self._lateness()
        if self._fade_in_block:
            clean = clean.copy()
            n = min(len(self._fade48), len(clean))
            clean[:n] *= self._fade48[::-1][:n]
            self._fade_in_block = False
        if insert:
            clean = clean.copy()
            n = min(len(self._fade48), len(clean))
            clean[len(clean) - n :] *= self._fade48[len(self._fade48) - n :]
        correction = self.controller.update(self.fill_average, self._feed_forward(), self.primed)
        ratio = self._nominal_ratio * (1.0 + correction)
        self._write(self._resampler.process(clean, ratio))
        if insert:
            # The added buffering, at once (obs-audio.c:381): silence through the same resampler, so nothing reorders.
            self._write(self._resampler.process(np.zeros(insert, np.float32), ratio))
            self._fade_in_block = True
        self._blocks += 1

    def _write(self, out: Any) -> None:
        out = np.asarray(out, np.float32)
        if self.ring.write(out) < len(out):
            self.counters.output_overflows += 1

    def _lateness(self) -> int:
        """How much buffering to add now, in 48 kHz samples (0: none). An underrun already played its silence, so it
        only raises the target; lateness before the ring runs dry is added to the ring too."""
        low, self.low_water = self.low_water, math.inf
        if self.counters.underruns != self._underruns_seen:
            self._underruns_seen = self.counters.underruns
            self._add_buffering(self._ticks(low), "an underrun: ", "")
            return 0
        if low >= self._safety or not self.primed:
            return 0
        added = self._add_buffering(
            self._ticks(low), "", f" ({low * 1000 / self.out_rate:.1f} ms to spare at the worst callback)"
        )
        return round(added * SAMPLE_RATE / self.out_rate)

    def _ticks(self, low: float) -> float:
        """The shortfall below the safety margin, rounded up to whole 10 ms ticks (obs-audio.c:378-379), in frames."""
        short = self._safety - (low if math.isfinite(low) else 0.0)
        return max(1, math.ceil(short / self._step - 1e-9)) * self._step

    def _read_stamps(self) -> None:
        st = self.stamps
        end = st.count
        for i in range(max(st.read, end - STAMPS), end):
            j = i % STAMPS
            t = float(st.time[j])
            self.clock.add(t, float(st.frames[j]))
            fill = float(st.fill[j])
            if fill < 0.0:  # not playing: the ring is filling up, not a cushion
                continue
            if self._fill_t0 is None:
                self._fill_t0 = t
            elif t - self._fill_t0 >= FILL_AVERAGE_S and self._fill_n:
                self.fill_average = self._fill_sum / self._fill_n
                self._fill_sum, self._fill_n, self._fill_t0 = 0.0, 0, t
            self._fill_sum += fill
            self._fill_n += 1
        st.read = end

    def _feed_forward(self) -> float:
        source = self.source
        if source is None or not source.ready or not self.clock.ready:
            return 0.0
        return (self.clock.rate / self.out_rate) / (source.rate / source.nominal) - 1.0

    def _add_buffering(self, add: float, reason: str, detail: str) -> float:
        """Raise the target by ``add`` frames, capped (obs-audio.c:381-387); returns what was added."""
        target = self.controller.target
        if target >= self._max_target:
            return 0.0
        new = min(self._max_target, target + add)
        self.controller.raise_target(new)
        self.prime = int(new)
        self.buffering_changes += 1
        log.info(
            "%s: %sadding %d ms of audio buffering, total now %d ms%s",
            self.name,
            reason,
            round((new - target) * 1000 / self.out_rate),
            round(new * 1000 / self.out_rate),
            detail,
        )
        if new >= self._max_target and not self._max_warned:
            self._max_warned = True
            log.warning("%s: maximum audio buffering reached (%d ms)", self.name, round(new * 1000 / self.out_rate))
        return new - target

    @property
    def target_ms(self) -> float:
        return self.controller.target * 1000.0 / self.out_rate

    # ------------------------------------------------------------------ callback (copy only)

    def callback(self, outdata: Audio, frames: int, time: Any, status: Any) -> None:
        self.stamps.put(callback_time(time), frames, float(self.ring.available()) if self.primed else -1.0)
        if status:
            self.counters.output_device_errors += 1
        self.callbacks += 1
        cushion = min(frames, MAX_CALLBACK_FRAMES) + len(self._fade_out) + int(self._safety)
        if cushion > self.prime:
            # The device asks for more per block than the cushion holds: add buffering rather than starve (OBS adds
            # latency for a late source the same way, libobs/obs-audio.c:361-422).
            self.prime = cushion
            self.controller.target = float(cushion)
        done = 0
        while done < frames:  # only loops for blocks larger than the scratch buffer
            n = min(frames - done, MAX_CALLBACK_FRAMES)
            self._fill_output(outdata[done : done + n], n)
            done += n

    def _fill_output(self, outdata: Audio, n: int) -> None:
        if not self.primed:
            if self.ring.available() < self.prime - self._prime_slack:
                outdata.fill(0.0)
                return
            self.primed = True
            self.ever_primed = True
            self._fade_in_next = True
        buf = self._scratch[:n]
        fade_len = len(self._fade_out)
        available = self.ring.available()
        spare = available - n - fade_len  # below 0 is an underrun
        self.low_water = min(self.low_water, spare)
        if spare < 0:
            # Underrun: a fade's worth is always kept back, so there is audio left to fade out even when the ring
            # runs dry on a block boundary. Play it all, fade it, then wait for the cushion to rebuild.
            got = self.ring.read_into(buf[: min(n, available)])
            if not self.finishing:
                self.counters.underruns += 1
            fade = min(fade_len, got)
            if fade:
                buf[got - fade : got] *= self._fade_out[fade_len - fade :]
            buf[got:] = 0.0
            self.primed = False
        else:
            got = self.ring.read_into(buf)
        if self.primed and self._fade_in_next:
            fade = min(len(self._fade_in), n)
            buf[:fade] *= self._fade_in[:fade]
            self._fade_in_next = False
        self.frames_played += got
        outdata[:] = buf[:, None]


def _scaled(block: Audio, gain: npt.NDArray[np.float64] | float) -> Audio:
    if isinstance(gain, float) and gain == 1.0:
        return block
    return (block * gain).astype(np.float32)


class AudioPath:
    """Everything between the callbacks. ``input_callback`` and the sinks' callbacks only copy and stamp."""

    def __init__(
        self,
        in_rate: int,
        in_channels: int,
        out_rate: int,
        out_channels: int,
        chain: Processor,
        recorder: Recorder | None = None,
    ) -> None:
        self.in_rate = in_rate
        self.in_channels = in_channels
        self.out_rate = out_rate
        self.out_channels = out_channels
        self.chain = chain
        self.recorder = recorder
        self.counters = Counters()
        self.in_ring = Ring(int(INPUT_RING_S * in_rate), in_channels)
        self.mic_stamps = Stamps()
        self.mic_clock = ClockEstimator(in_rate, "mic")
        self.cable = Sink(out_rate, out_channels, self.counters, source=self.mic_clock)
        self.cable_live = True  # False while the cable's stream is closed for a retry: the mic and backup go on
        # Hear myself and the mic test: a second sink on the speakers, with its own ring, clock and drift correction
        # (OBS feeds monitoring from the audio thread, resampled to the device: libobs/audio-monitoring/win32/
        # wasapi-output.c:204-221, :279-340). OBS has no drift correction there: a faster OBS fills the 1 s device buffer
        # until the monitor is rebuilt (:325-327, :347-348), a slower one underruns. Its late-audio cutting (:52-128)
        # runs only for sources with video (:316-322), so for a mic it skips nothing either; this sink never does.
        self.monitor_counters = Counters()  # kept across Hear myself on/off, for the totals
        self.monitor: Sink | None = None
        self.monitor_live = False  # Hear myself on: the clean voice goes to the speakers
        self.mic_test = MicTest()
        # Hear myself volume: a gain on Hear myself only, 10 ms ramp. The mic test plays at unity (final-handoff r1 m1).
        self.monitor_volume = 1.0  # set by the worker; read once per block
        self._monitor_gain = Ramp(1.0)
        self._silence = np.zeros(BLOCK, np.float32)
        # Remove speaker echo (J06 Step 8): the mic is held back so the speakers' sound reaches AEC3 before its echo.
        self.echo: Any = None  # clean_voice.echo.EchoStage while echo removal runs (or ramps out)
        self.reference: Any = None  # clean_voice.echo.Reference while the loopback is open
        self.render: Any = None  # clean_voice.echo.RenderMarker while the silent render is open
        self.mic_first_times: deque[float] = deque(maxlen=4096)  # when each block's first mic frame arrived
        self._mic_arrivals: deque[tuple[float, float]] = deque(maxlen=STAMPS)  # (frames before, time) per callback

        # Worker-side buffers.
        self._in_block = round(BLOCK * in_rate / SAMPLE_RATE)
        self._device_frames = np.zeros((self._in_block, in_channels), np.float32)
        self._in_resampler = None if in_rate == SAMPLE_RATE else samplerate.Resampler("sinc_medium", channels=1)
        self._stage = Ring(SAMPLE_RATE, 1)  # 48 kHz mono, waiting to make a full block
        self._block = np.zeros(BLOCK, np.float32)
        self.output_peak = 0.0  # peak of the clean voice since the UI last took it
        self.input_peak = 0.0
        self.blocks_processed = 0

    # ------------------------------------------------------------------ the cable sink, by its old names

    @property
    def out_ring(self) -> Ring:
        return self.cable.ring

    @property
    def controller(self) -> DriftController:
        return self.cable.controller

    @property
    def primed(self) -> bool:
        return self.cable.primed

    @property
    def frames_played(self) -> int:
        return self.cable.frames_played

    @property
    def output_callbacks(self) -> int:
        return self.cable.callbacks

    # ------------------------------------------------------------------ callbacks (copy only)

    def input_callback(self, indata: Audio, frames: int, time: Any, status: Any) -> None:
        self.mic_stamps.put(callback_time(time), frames)
        if status:
            self.counters.input_device_errors += 1
        if self.in_ring.write(indata) < frames:
            self.counters.input_overflows += 1

    def output_callback(self, outdata: Audio, frames: int, time: Any, status: Any) -> None:
        self.cable.callback(outdata, frames, time, status)

    def monitor_callback(self, outdata: Audio, frames: int, time: Any, status: Any) -> None:
        sink = self.monitor
        if sink is None:
            outdata.fill(0.0)
            return
        sink.callback(outdata, frames, time, status)

    def loopback_callback(self, indata: Audio, frames: int, time: Any, status: Any) -> None:
        """The speakers' loopback: copy into the reference ring and stamp, nothing else (A5)."""
        ref = self.reference
        if ref is not None:
            ref.callback(indata, frames, time, status)

    def render_callback(self, outdata: Audio, frames: int, time: Any, status: Any) -> None:
        """The silent render on the speakers (W1): the inaudible timing marker, copied from a table, and a stamp."""
        render = self.render
        if render is None:
            outdata.fill(0.0)
            return
        render.callback(outdata, frames, time, status)

    # ------------------------------------------------------------------ Hear myself, mic test, echo (worker thread)

    def add_monitor(self, out_rate: int, out_channels: int) -> Sink:
        self.monitor = Sink(out_rate, out_channels, self.monitor_counters, name="monitor", source=self.mic_clock)
        return self.monitor

    def remove_monitor(self) -> None:
        self.monitor = None

    def detach_cable(self) -> None:
        """The cable's stream is closed (it stalled) while the mic goes on: stop feeding it, keep everything else."""
        self.cable_live = False

    def attach_cable(self, out_rate: int, out_channels: int) -> Sink:
        """A reopened cable stream: a fresh sink at its rate (call before the stream starts, from the worker). Each
        device recovers on its own, as each OBS source does (plugins/win-wasapi/win-wasapi.cpp:959-993)."""
        self.out_rate, self.out_channels = out_rate, out_channels
        self.cable = Sink(out_rate, out_channels, self.counters, source=self.mic_clock)
        self.cable_live = True
        return self.cable

    def enable_echo(self, ref_rate: int, ref_channels: int) -> None:
        """Start echo removal with the loopback at ``ref_rate`` x ``ref_channels`` (call from the worker thread:
        AEC3 instances are not thread-safe). The loopback stream may start delivering right after."""
        from clean_voice import echo

        self.reference = echo.Reference(ref_rate, ref_channels, self.mic_clock, self.counters, self.mic_stamps)
        self.render = echo.RenderMarker(ref_rate)
        if self.echo is None:
            self.echo = echo.EchoStage()
        self.echo.start(self.reference, self.render, self.mic_first_times)

    def set_echo_paused(self, paused: bool) -> None:
        """While Hear myself is on, the speakers carry the guest's own voice: crossfade to the uncancelled mic, but
        keep the reference flowing so resuming is seamless (audio.rs:429-431)."""
        if self.echo is not None:
            self.echo.set_apply(not paused)

    def disable_echo(self) -> None:
        """Stop echo removal (the loopback and render streams are already closed): crossfade out, then ramp the mic
        delay back to 0 (nothing skipped)."""
        if self.echo is not None:
            self.echo.stop()
        self.reference = None
        self.render = None

    def status_counters(self) -> Counters:
        """This path's counters, the speaker sink's included, by field."""
        mon = self.monitor_counters
        return self.counters + Counters(
            monitor_underruns=mon.underruns,
            monitor_overflows=mon.output_overflows,
            monitor_device_errors=mon.output_device_errors,
        )

    # ------------------------------------------------------------------ worker

    def process(self) -> int:
        """Run every complete block waiting in the input ring; return how many 10 ms blocks ran."""
        self._read_mic_stamps()
        ran = 0
        while self.in_ring.available() >= self._in_block:
            if ran:
                _time.sleep(0)  # a backlog: let a waiting device callback have the GIL between blocks
            self.in_ring.read_into(self._device_frames)
            frames = self._device_frames
            mono = frames[:, 0] if self.in_channels == 1 else frames.mean(axis=1, dtype=np.float32)
            if self._in_resampler is not None:
                mono = self._in_resampler.process(mono, SAMPLE_RATE / self.in_rate).astype(np.float32)
            self._stage.write(mono)
            while self._stage.available() >= BLOCK:
                self._stage.read_into(self._block)
                self._run_block(self._block)
                ran += 1
        return ran

    def _read_mic_stamps(self) -> None:
        st = self.mic_stamps
        end = st.count
        for i in range(max(st.read, end - STAMPS), end):
            j = i % STAMPS
            self.mic_clock.add(float(st.time[j]), float(st.frames[j]))
            self._mic_arrivals.append((float(st.frames[j]), float(st.time[j])))
        st.read = end

    def _block_arrival(self) -> float:
        """When the mic callback that delivered this block's first frame ran (for the echo timing)."""
        first = self.blocks_processed * BLOCK * self.in_rate / SAMPLE_RATE
        arrivals = self._mic_arrivals
        while len(arrivals) > 1 and arrivals[1][0] <= first:
            arrivals.popleft()
        return arrivals[0][1] if arrivals else 0.0

    def _run_block(self, raw: Audio) -> None:
        stage = self.echo
        if stage is not None:
            self.mic_first_times.append(self._block_arrival())
            delayed = stage.delay(raw)
            x = stage.cancel(delayed)
            clean = self.chain.process(x, delayed)  # Off sends the mic before echo removal, as the original did
            if stage.finished:
                self.echo = None
        else:
            clean = self.chain.process(raw)
        self.input_peak = max(self.input_peak, float(np.max(np.abs(raw))))
        self.output_peak = max(self.output_peak, float(np.max(np.abs(clean))))
        if self.recorder is not None:
            self.recorder.write(clean, raw)  # the raw backup is the mic itself, never delayed or cancelled
        if (
            self.cable_live and self.cable.callbacks > 0
        ):  # wasapi-output.c:279-345: nothing queues before the device runs
            self.cable.push(clean)
        self._feed_monitor(clean)
        self.blocks_processed += 1

    def _feed_monitor(self, clean: Audio) -> None:
        test = self.mic_test
        if test.state is MicTestState.RECORDING:
            test.record(clean)  # exactly what went to the cable
        # The Hear myself volume moves every block, heard or not, so the speakers start at the level that is set.
        self._monitor_gain.set(self.monitor_volume)
        gain = self._monitor_gain.gains(BLOCK)
        sink = self.monitor
        if sink is None or sink.callbacks == 0:
            # Not playing yet: the stream is still starting (a Bluetooth headset can take half a second). What would
            # queue up now would only add its length to the delay, so nothing is queued until the first callback.
            return
        if test.state is MicTestState.PLAYING:
            sink.finishing = False
            sink.push(test.next_block())  # at full level: at Hear myself 0 % the test would be silent
            if test.state is MicTestState.IDLE and not self.monitor_live:
                sink.finishing = True  # the playback is over: the ring may now run dry
        elif self.monitor_live:
            sink.finishing = False
            sink.push(_scaled(clean, gain))  # a copy: the cable's block stays as it is
        elif test.state is MicTestState.RECORDING:
            sink.finishing = False
            sink.push(self._silence)  # keep the speakers primed for the playback
        else:
            sink.finishing = True

    def take_peaks(self) -> tuple[float, float]:
        """(input peak, clean output peak) since the last call, for the level meters."""
        peaks = (self.input_peak, self.output_peak)
        self.input_peak = 0.0
        self.output_peak = 0.0
        return peaks
