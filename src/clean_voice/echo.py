r"""Remove speaker echo: WebRTC AEC3 (pywebrtc-audio) fed with a continuous, timestamp-aligned loopback of the speakers.

    speakers' loopback --copy--> reference ring --worker: resample to the mic's clock--> 480 samples per mic block
    silent render on the same speakers: keeps the loopback delivering, and carries an inaudible timing marker
    mic block --MicDelay--> AEC3(mic, reference) --crossfade--> the clean-up chain

This fixes the original app's echo canceller (docs/simple-clean-mic-audio-report.md, cause 3, section 4 P0.3):
- the reference never breaks: during silence a continuously running render on the same speakers keeps the loopback
  delivering. That render is Clean Voice's own, needed by AEC3: OBS's "silent loopback fix" only fills one render
  buffer with zeros and never starts the client (plugins/win-wasapi/win-wasapi.cpp:751-797, ClearBuffer, called at
  :883-885). A silent stretch is real zeros, never a gap (OBS keeps silent packets on the timeline, :1044-1054). A real shortfall is filled with zeros and
  counted, and no reference audio is ever skipped, except once after a timing jump over 2 s (logged and counted);
- the reference is resampled to the mic's clock by a J05 drift controller: feed-forward from the two streams' clock
  estimators (timestamps, as OBS judges timing, libobs/obs-source.c:1613-1683) plus a trim that holds the ring's
  lead, so the lag AEC3 sees stays constant;
- the mic is held back so the reference reaches AEC3 before its echo (J06 Step 3 measured the loopback 19-35 ms behind
  the render on real speakers, 119 ms on VB-CABLE). The silent render writes a -94 dBFS pseudo-random marker; the
  worker finds it in the reference and measures the reference's age against the mic's arrival times; the mic delay is
  that age plus a margin and one AEC3 frame, capped at MAX_DELAY_S, and moves at DELAY_SLEW so nothing is ever skipped
  or repeated;
- AEC3 runs on the worker thread only; the callbacks only copy and stamp.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

import numpy as np
import numpy.typing as npt
import samplerate

from clean_voice.audio_path import (
    BLOCK,
    FILL_AVERAGE_S,
    SAMPLE_RATE,
    STAMPS,
    ClockEstimator,
    Counters,
    DriftController,
    Stamps,
    callback_time,
)
from clean_voice.dsp import Ramp
from clean_voice.ring import Ring

log = logging.getLogger(__name__)

Audio = npt.NDArray[np.float32]

REF_RING_S = 4.0
# The reference ring's lead: this plus the longest wait between loopback callbacks (or the largest callback), as the
# plan's "20 ms plus the largest callback" but in time, because a loopback can hold audio back like a bursty mic does
# (J06 Step 9: the measured mic pattern as a loopback had 19 gaps in 5 min with the frame-count rule).
REF_LEAD_S = 0.020
REF_MAX_LEAD_S = 0.150
# The least fill held each second: the block being read (10 ms) and a 20 ms margin. With a 10 ms margin the measured
# mic pattern ran short once in 5 min: a second's minimum can miss a rarer, deeper dip (J06 Step 10).
REF_MIN_S = 0.030
# The reference's trim: 10 ms of error -> 100 ppm, at most 500 ppm (twice the cable's gain; resampling the reference is
# inaudible). Measured in the drifting room (A3): at 500 ppm per 10 ms the least-fill loop swung the age AEC3 sees by
# 4.7 ms over a minute; at 200, 2.0 ms; at 100, 0.9 ms, with no gaps on the measured mic pattern either.
REF_TRIM_PER_S = 100e-6 / 0.010
REF_MAX_TRIM = 500e-6
# The reference starts once the loopback has been watched this long, so its waits are known: at 0.5 s the measured mic
# pattern replayed as a loopback ran short once at 1.5 s; at 1.0 and 2.0 s never in 5 min (J06 Step 9).
REF_OBSERVE_S = 1.0
REF_RAISE_S = 0.010  # a gap adds this much lead
REF_GAP_IGNORE_S = 0.5  # a wait longer than this is a stall (a gap), not jitter to plan the lead for
# The stream's first callbacks come while the device starts (and the silent render with it): their waits say nothing
# about how it runs (J06 Step 10: a 140 ms start-up wait on Bluetooth headphones set a 150 ms lead; running, 13 ms).
REF_SKIP_FIRST = 25
REF_CHUNK = 240  # loopback frames resampled at a time
REALIGN_MIN_S = 0.100  # after a timing jump, re-align only when the ring holds this much more than its lead
MARKER_LEVEL = 10.0 ** (-94.0 / 20.0)  # inaudible: far below any DAC's noise, but exact in the float mix
MARKER_PERIOD = 1 << 16
MARKER_SEED = 20261111
AGE_WINDOW = 4800  # 0.1 s of reference per measurement (the clocks drift under a sample in that time)
AGE_EVERY_BLOCKS = 25  # a measurement every 0.25 s
AGE_OFFSET_S = 0.5  # the loopback and render streams start within this of each other: the first search range
AGE_TRACK_S = 0.02  # once found, the offset is tracked within this
AGE_MIN_Z = 9.0  # the marker's correlation peak must stand this many standard deviations out
AGE_KEEP = 5  # the age used is the median of the last few measurements
AGE_WINDOWS = 40  # measurements combined: the last 10 s
AGE_FIT_S = 5.0  # the clock lines are fitted through this much of the callbacks
AGE_FIT_BLOCKS = 500
DEFAULT_AGE_S = 0.060  # until the marker is found: the largest real-speaker age expected (35 ms loopback + lead)
MARGIN_S = 0.010
AEC_FRAME_S = 0.010
MAX_DELAY_S = 0.150  # J05's buffering cap (lip sync)
DELAY_SLEW = 0.005  # the mic delay changes by at most 0.5 % of real time: a 9-cent pitch shift while it moves
DELAY_RETARGET_S = 0.003  # smaller changes of the measured age leave the delay where it is
MAX_DELAY_SAMPLES = int(MAX_DELAY_S * SAMPLE_RATE)


def marker_table() -> npt.NDArray[np.float32]:
    """The timing marker: a fixed pseudo-random +-MARKER_LEVEL sequence, twice over so any slice is contiguous."""
    rng = np.random.default_rng(MARKER_SEED)
    one = (np.where(rng.random(MARKER_PERIOD) < 0.5, -1.0, 1.0) * MARKER_LEVEL).astype(np.float32)
    return np.concatenate((one, one))


_MARKER = marker_table()


class RenderMarker:
    """The silent render's callback side: copy the marker from the table and stamp the callback (copy only)."""

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.stamps = Stamps()
        self.pos = 0

    def set_rate(self, rate: int) -> None:
        self.rate = rate

    def callback(self, outdata: Audio, frames: int, time: Any, status: Any) -> None:
        self.stamps.put(callback_time(time), frames)
        done = 0
        while done < frames:
            n = min(frames - done, MARKER_PERIOD)
            i = self.pos % MARKER_PERIOD
            outdata[done : done + n] = _MARKER[i : i + n, None]
            self.pos += n
            done += n


class Reference:
    """The speakers' loopback, continuous and on the mic's clock: exactly 480 samples per mic block."""

    def __init__(
        self,
        ref_rate: int,
        ref_channels: int,
        mic_clock: ClockEstimator,
        counters: Counters,
        mic_stamps: Stamps | None = None,
    ) -> None:
        self.rate = ref_rate
        self.channels = ref_channels
        self.counters = counters
        self.ring = Ring(int(REF_RING_S * ref_rate), ref_channels)
        self.stamps = Stamps()
        self.clock = ClockEstimator(ref_rate, "loopback")
        self.mic_clock = mic_clock
        self.mic_stamps = mic_stamps  # the mic paces the reads: its waits size the start
        # The trim holds the ring's *least* fill each second at REF_MIN_S (one block and a margin), not its average: a
        # bursty mic takes several blocks at once after holding back, so the fill swings well below its average
        # (J06 Step 10: the average rule ran short once 2 s in on the real Focusrite mic).
        self.controller = DriftController(ref_rate, REF_TRIM_PER_S, REF_MAX_TRIM)
        self._min = math.inf
        self._min_n = 0
        self.fill_min: float | None = None
        self._longest = 0.0  # frames: the longest wait between callbacks, or the largest callback
        self._raised = 0.0  # frames added after gaps
        self._first_t: float | None = None
        self._last_t: float | None = None
        self._started = False
        self._callbacks = 0
        self.controller.target = REF_MIN_S * ref_rate
        self._resampler = samplerate.Resampler("sinc_fastest", channels=1)
        self._fifo = Ring(SAMPLE_RATE, 1)
        self._chunk = np.zeros((REF_CHUNK, ref_channels), np.float32)
        self.output = np.zeros(BLOCK, np.float32)
        self.primed = False
        self.realigns = 0
        self.frames_used = 0  # loopback frames taken from the ring (dropped ones included)
        self.frames_dropped = 0
        self._resets_seen = 0
        self._fill_sum = 0.0
        self._fill_n = 0
        self.fill_average: float | None = None
        self.observer: Any = None  # tests: called with every output block
        self.meter: AgeMeter | None = None

    @property
    def block_start(self) -> float:
        """The loopback frame the last output block's first sample came from."""
        pending = (self._fifo.available() + BLOCK) * self.rate / SAMPLE_RATE
        return self.frames_used - pending

    # ------------------------------------------------------------------ callback (copy only)

    def callback(self, indata: Audio, frames: int, time: Any, status: Any) -> None:
        self.stamps.put(callback_time(time), frames)
        if self.ring.write(indata) < frames:
            self.counters.reference_overflows += 1

    # ------------------------------------------------------------------ worker

    def _read_stamps(self) -> None:
        st = self.stamps
        end = st.count
        for i in range(max(st.read, end - STAMPS), end):
            j = i % STAMPS
            t = float(st.time[j])
            self.clock.add(t, float(st.frames[j]))
            self._callbacks += 1
            if self._first_t is None:
                self._first_t = t
            elif self._last_t is not None and t - self._last_t < REF_GAP_IGNORE_S and self._callbacks > REF_SKIP_FIRST:
                # The longest wait between loopback callbacks: the audio that arrives late by that much must already
                # be in the ring when it is due. A wait over REF_GAP_IGNORE_S is a stall, handled as a gap.
                self._longest = max(self._longest, (t - self._last_t) * self.rate)
            self._last_t = t
            k = (i + 1) % STAMPS
            if i + 1 < end:
                self._longest = max(self._longest, float(st.frames[k] - st.frames[j]))
        st.read = end
        if end:
            self._longest = max(self._longest, float(st.total_frames - st.frames[(end - 1) % STAMPS]))
        self.controller.target = REF_MIN_S * self.rate + self._raised

    def _mic_longest(self) -> float:
        """The mic's longest wait between callbacks lately, in loopback frames."""
        st = self.mic_stamps
        if st is None or st.count < 3:
            return 0.0
        n = min(st.count, STAMPS, 300)
        idx = np.arange(st.count - n, st.count) % STAMPS
        waits = np.diff(st.time[idx])
        waits = waits[waits < REF_GAP_IGNORE_S]
        return float(waits.max() * self.rate) if len(waits) else 0.0

    @property
    def start_lead(self) -> float:
        """The fill to start from: REF_LEAD_S, the loopback's and the mic's longest waits, and any raise."""
        lead = REF_LEAD_S * self.rate + self._longest + self._mic_longest() + self._raised
        return min(lead, REF_MAX_LEAD_S * self.rate)

    def _feed_forward(self) -> float:
        if not (self.clock.ready and self.mic_clock.ready):
            return 0.0
        return (self.mic_clock.rate / self.mic_clock.nominal) / (self.clock.rate / self.clock.nominal) - 1.0

    def _realign(self) -> None:
        """A timing jump over 2 s (the loopback's clock estimate restarted, as OBS resets on one, obs-source.c:
        1513-1524): drop what is beyond the lead, once. The only time reference audio is dropped."""
        excess = self.ring.available() - int(self.start_lead)
        if excess <= REALIGN_MIN_S * self.rate:
            return  # the lead is where it should be: nothing to re-align
        scratch = np.zeros((min(excess, self.ring.capacity), self.channels), np.float32)
        self.ring.read_into(scratch)
        self.frames_used += excess
        self.frames_dropped += excess
        self.realigns += 1
        log.info(
            "loopback: timing jumped; re-aligned the echo reference once (%d ms dropped)", excess * 1000 // self.rate
        )

    def read_block(self) -> Audio:
        self._read_stamps()
        if self.clock.resets != self._resets_seen:
            self._resets_seen = self.clock.resets
            self._realign()
        available = self.ring.available()
        watched = self._last_t is not None and self._first_t is not None and self._last_t - self._first_t
        start = self.start_lead
        if not self.primed and available >= start and (watched or 0.0) >= REF_OBSERVE_S:
            self.primed = True
            if not self._started:
                # The very first start: what came in while the loopback was being watched was never handed to anyone;
                # start from the lead, as a sink starts from its cushion. (After a gap nothing is ever dropped.)
                self._started = True
                excess = available - int(start)
                if excess > 0:
                    self.ring.read_into(np.zeros((excess, self.channels), np.float32))
                    self.frames_used += excess
                    available -= excess
        if self.primed:
            self._fill_sum += available
            self._min = min(self._min, available)
            self._fill_n += 1
            if self._fill_n * BLOCK >= FILL_AVERAGE_S * SAMPLE_RATE:
                self.fill_average = self._fill_sum / self._fill_n
                self.fill_min = self._min
                self._fill_sum, self._fill_n, self._min = 0.0, 0, math.inf
            correction = self.controller.update(self.fill_min, self._feed_forward(), True)
            ratio = SAMPLE_RATE / self.rate * (1.0 + correction)
            while self._fifo.available() < BLOCK:
                n = min(REF_CHUNK, self.ring.available())
                if n == 0:
                    break
                chunk = self._chunk[:n]
                self.ring.read_into(chunk)
                self.frames_used += n
                mono = chunk[:, 0] if self.channels == 1 else chunk.mean(axis=1, dtype=np.float32)
                if self.meter is not None:
                    self.meter.feed_loopback(mono, self.frames_used - n)
                self._fifo.write(self._resampler.process(mono, ratio).astype(np.float32))
        got = self._fifo.read_into(self.output)
        if got < BLOCK:
            self.output[got:] = 0.0
            if self.primed:
                # Ran short: zeros for the shortfall, counted, and wait for a 10 ms longer lead; nothing is skipped
                # (late audio adds buffering, as for the cable: libobs/obs-audio.c:361-422).
                self.counters.reference_gaps += 1
                self.primed = False
                self._raised += REF_RAISE_S * self.rate
                log.info("loopback: the echo reference ran short; its lead grows by %d ms", REF_RAISE_S * 1000)
        if self.observer is not None:
            self.observer(self.output)
        return self.output


def _line(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Least-squares y = a + b * x: callback times jitter (and bursty devices bunch them), a line through seconds of
    them does not."""
    xm, ym = float(x.mean()), float(y.mean())
    xc = x - xm
    sxx = float(np.dot(xc, xc))
    b = float(np.dot(xc, y - ym)) / sxx if sxx > 0 else 0.0
    return ym - b * xm, b


class AgeMeter:
    """How old the reference is when AEC3 gets it: the time a mic block arrived minus the time the speaker sound paired
    with it was rendered.

    The render and the loopback run on the speakers' one clock, so loopback frame n holds render frame n - offset, with
    one offset per session. The offset is found by correlating the raw loopback frames the reference consumes (no
    resampler in the way) with the marker the render wrote; the correlations of the last AGE_WINDOWS 0.1 s windows are
    combined, each weighted by its own noise (inverse variance), so windows where the far end talks add almost nothing
    and the marker is found 20-30 dB under other sound. Each block's age then follows from the stream positions: the
    mic's smoothed arrival line at the block, minus the render line's time of the loopback frames the block used.
    """

    def __init__(self, reference: Reference, render: RenderMarker, mic_times: deque[float]) -> None:
        self.reference = reference
        self.render = render
        self.mic_times = mic_times
        self._raw = np.zeros(AGE_WINDOW)  # the last consumed loopback frames, mono
        self._raw_end = 0  # loopback index one past self._raw's last frame
        self._raw_len = 0
        self._arrivals: deque[tuple[float, float]] = deque(maxlen=AGE_FIT_BLOCKS)  # (block number, arrival time)
        self._count = 0
        self._num: deque[npt.NDArray[np.float64]] = deque(maxlen=AGE_WINDOWS)  # corr / var per window
        self._inv: deque[float] = deque(maxlen=AGE_WINDOWS)
        self._search: tuple[int, int] = (-int(AGE_OFFSET_S * reference.rate), int(AGE_OFFSET_S * reference.rate))
        self.offset: int | None = None  # loopback index - render index
        self.ages: deque[float] = deque(maxlen=AGE_KEEP)
        self.measurements = 0
        self.last_z = 0.0

    @property
    def age_s(self) -> float | None:
        return float(np.median(self.ages)) if self.ages else None

    def feed_loopback(self, mono: Audio, start: int) -> None:
        """The reference consumed loopback frames [start, start + len(mono)) (worker thread)."""
        if start != self._raw_end:
            self._raw_len = 0  # frames were dropped (a re-align): start a fresh window
        n = len(mono)
        if n >= AGE_WINDOW:
            self._raw[:] = mono[-AGE_WINDOW:]
        else:
            self._raw[:-n] = self._raw[n:]
            self._raw[-n:] = mono
        self._raw_end = start + n
        self._raw_len = min(AGE_WINDOW, self._raw_len + n)

    def feed(self, block: Audio, loopback_start: float) -> None:
        """One reference block went to AEC3; its first frame came from loopback frame ``loopback_start`` (the
        first frames on both sides: ``mic_times`` holds when each mic block's first frame arrived)."""
        if self.mic_times:
            self._arrivals.append((float(self._count), self.mic_times[-1]))
        self._count += 1
        if self._count % AGE_EVERY_BLOCKS == 0:
            self._find_offset()
            if self.offset is not None:
                self._age(loopback_start)

    def _render_line(self) -> tuple[float, float] | None:
        """time = a + b * frames, through the render callbacks of the last AGE_FIT_S."""
        st = self.render.stamps
        n = min(st.count, STAMPS, int(AGE_FIT_S * 100))
        if n < 20:
            return None
        idx = np.arange(st.count - n, st.count) % STAMPS
        return _line(st.frames[idx].astype(np.float64), st.time[idx].astype(np.float64))

    def _find_offset(self) -> None:
        if self._raw_len < AGE_WINDOW:
            return
        lo, hi = self._search
        n0 = self._raw_end - AGE_WINDOW
        first = n0 - hi  # the earliest render index searched
        if first < 0:
            return
        span = hi - lo + AGE_WINDOW
        seg = _MARKER[(first + np.arange(span)) % MARKER_PERIOD].astype(np.float64)
        size = 1 << math.ceil(math.log2(span + AGE_WINDOW))
        corr = np.fft.irfft(np.fft.rfft(seg, size) * np.conj(np.fft.rfft(self._raw, size)), size)[: hi - lo + 1]
        corr = corr[::-1]  # index k: offset lo + k
        var = float(np.var(corr))
        if var <= 0.0:
            return
        self._num.append(corr / var)
        self._inv.append(1.0 / var)
        combined = np.sum(self._num, axis=0)
        combined = (combined - combined.mean()) / (combined.std() or 1.0)
        k = int(np.argmax(combined))
        self.last_z = float(combined[k])
        if self.last_z >= AGE_MIN_Z:
            found = lo + k
            if self.offset is None:
                # Found: from now on search only around it (cheaper), keeping the combined windows aligned.
                self._narrow(found)
            self.offset = found

    def _narrow(self, offset: int) -> None:
        lo, hi = self._search
        half = int(AGE_TRACK_S * self.reference.rate)
        new = (max(lo, offset - half), min(hi, offset + half))
        a, b = new[0] - lo, new[1] - lo + 1
        self._num = deque((c[a:b] for c in self._num), maxlen=AGE_WINDOWS)
        self._search = new

    def _age(self, loopback_start: float) -> None:
        line = self._render_line()
        if line is None or len(self._arrivals) < 20 or self.offset is None:
            return
        blocks = np.array([a[0] for a in self._arrivals])
        times = np.array([a[1] for a in self._arrivals])
        a, b = _line(blocks, times)
        arrived = a + b * (self._count - 1)  # this block, on the mic's smoothed arrival line
        rendered = line[0] + line[1] * (loopback_start - self.offset)
        age = arrived - rendered
        if -AGE_OFFSET_S < age < 2 * AGE_OFFSET_S:
            self.ages.append(age)
            self.measurements += 1


class MicDelay:
    """A variable delay on the mic that moves at most DELAY_SLEW samples per sample (a fractional read with linear
    interpolation while it moves, an exact copy while it stands): nothing is skipped or repeated, and no click."""

    def __init__(self) -> None:
        self._hist = np.zeros(MAX_DELAY_SAMPLES + 4, np.float32)
        self.current = 0.0  # samples
        self.target = 0.0

    @property
    def ms(self) -> float:
        return self.current * 1000.0 / SAMPLE_RATE

    def set_target_s(self, seconds: float) -> None:
        self.target = float(min(MAX_DELAY_SAMPLES, max(0.0, round(seconds * SAMPLE_RATE))))

    def process(self, x: Audio) -> Audio:
        n = len(x)
        buf = np.concatenate((self._hist, x))
        self._hist = buf[-len(self._hist) :]
        if self.current == self.target:
            d = int(self.current)
            end = len(buf) - d
            return buf[end - n : end].copy()
        k = np.arange(1, n + 1, dtype=np.float64)
        move = np.clip(self.target - self.current, -DELAY_SLEW * k, DELAY_SLEW * k)
        delays = self.current + move
        self.current = float(delays[-1])
        if abs(self.current - self.target) < 1e-9:
            self.current = self.target
        pos = (len(buf) - n + k - 1) - delays
        i = np.floor(pos).astype(np.intp)
        frac = (pos - i).astype(np.float32)
        nxt = np.minimum(i + 1, len(buf) - 1)
        return (buf[i] * (1.0 - frac) + buf[nxt] * frac).astype(np.float32)


class EchoCanceller:
    """WebRTC AEC3 through pywebrtc-audio (Apache-2.0; WebRTC BSD-3), one 10 ms block at a time. Create and use it on
    one thread only: instances are not thread-safe."""

    def __init__(self, stream_delay_ms: int = round((MARGIN_S + AEC_FRAME_S) * 1000)) -> None:
        from pywebrtc_audio import EchoCanceller as Aec3

        # The delay hint: with the mic held back by the reference's age plus the margin and one frame, the echo
        # trails its reference by about that margin.
        self._aec = Aec3(sample_rate=SAMPLE_RATE, num_channels=1, stream_delay_ms=stream_delay_ms)

    def process(self, near: Audio, far: Audio) -> Audio:
        if len(near) % BLOCK:
            raise ValueError("echo removal runs on whole 10 ms blocks")
        return self._aec.process(np.ascontiguousarray(near, np.float32), np.ascontiguousarray(far, np.float32))


class EchoStage:
    """The mic delay, AEC3 and the on/off crossfade, as the worker runs them for each block."""

    def __init__(self) -> None:
        self.mic_delay = MicDelay()
        self._aec: EchoCanceller | None = None
        self._apply = Ramp(0.0)
        self.reference: Reference | None = None
        self.meter: AgeMeter | None = None
        self._stopping = False
        self.finished = False
        self._zeros = np.zeros(BLOCK, np.float32)
        self._capped_warned = False

    @property
    def delay_ms(self) -> float:
        return self.mic_delay.ms

    def start(self, reference: Reference, render: RenderMarker, mic_times: deque[float]) -> None:
        self.reference = reference
        self.meter = AgeMeter(reference, render, mic_times)
        reference.meter = self.meter
        if self._aec is None:
            self._aec = EchoCanceller()
        self._stopping = False
        self.finished = False
        self._apply.set(1.0)
        age = self.meter.age_s
        self._retarget(DEFAULT_AGE_S if age is None else age)

    def _retarget(self, age: float) -> None:
        target = age + MARGIN_S + AEC_FRAME_S
        if abs(target * SAMPLE_RATE - self.mic_delay.target) >= DELAY_RETARGET_S * SAMPLE_RATE:
            self.mic_delay.set_target_s(target)
            if target > MAX_DELAY_S and not self._capped_warned:
                self._capped_warned = True
                log.warning(
                    "echo removal: the speakers' sound arrives %.0f ms late; the mic delay is capped at %.0f ms",
                    age * 1000,
                    MAX_DELAY_S * 1000,
                )

    def set_apply(self, on: bool) -> None:
        if not self._stopping:
            self._apply.set(1.0 if on else 0.0)

    def stop(self) -> None:
        """Switch echo removal off. The reference (and AEC3) stay until the crossfade back to the uncancelled mic has
        finished: the ring still holds its 30 ms lead after the loopback stream closes. Dropping it at once brought
        the echo back in one sample (J06 review r1, F1)."""
        self._stopping = True
        self._apply.set(0.0)
        self.meter = None
        self.mic_delay.set_target_s(0.0)

    def delay(self, raw: Audio) -> Audio:
        return self.mic_delay.process(raw)

    def cancel(self, near: Audio) -> Audio:
        ref = self.reference
        far = ref.read_block() if ref is not None else self._zeros
        if self.meter is not None and ref is not None:
            before = self.meter.measurements
            self.meter.feed(far, ref.block_start)
            if self.meter.measurements != before and self.meter.age_s is not None:
                self._retarget(self.meter.age_s)
        out = self._aec.process(near, far) if self._aec is not None and ref is not None else near
        w = self._apply.gains(len(near))
        y = near if isinstance(w, float) and w == 0.0 else (near + w * (out - near)).astype(np.float32)
        if self._stopping and not self._apply.moving:
            self.reference = None  # crossfaded out: the reference has done its job
            self._aec = None
            if self.mic_delay.current == 0.0:
                self.finished = True
        return y
