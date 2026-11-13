"""Offline simulation of the audio path with a mic and a cable on different clocks.

Drives the real ``AudioPath`` callbacks and worker step on a virtual timeline: the mic delivers blocks at
48 kHz x (1 + mic_ppm), the cable asks for blocks at 48 kHz x (1 + cable_ppm), callbacks arrive with jitter and the
worker wakes late now and then, like a busy desktop.

``bursty=True`` models what real Windows devices did on the development PC (measured with bare sounddevice
streams): a Focusrite "Video Call" input that holds back 20-30 ms of audio now and then and then delivers it
back to back (residual -970..+650 frames around its average rate), and VB-CABLE pulling up to 13 ms late and then
catching up (-600..+200 frames).
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from clean_voice import fine_tune
from clean_voice.audio_path import SAMPLE_RATE, AudioPath, Processor
from clean_voice.dsp import Chain, PassThrough
from clean_voice.settings import Settings, SettingsCell

FOUR_FILTERS = ("noise_gate_filter", "basic_eq_filter", "compressor_filter", "gain_filter")


def four_filter_chain() -> Chain:
    """The J10 done-when's test chain: Denoise and Voice style off, Fine-tune on with OBS's Noise Gate, 3-Band
    Equalizer, Compressor and Gain at OBS's defaults, top first."""
    filters: list[fine_tune.FilterSpec] = []
    for kind in FOUR_FILTERS:
        filters.append(fine_tune.new_filter(kind, filters))
    return Chain(SettingsCell(Settings(denoise=False, style=False, fine_tune=True, filters=tuple(filters))))


TONE_HZ = 440.0
TONE_AMP = 0.5


@dataclass
class SimResult:
    seconds: float
    mic_ppm: float
    cable_ppm: float
    underruns: int
    input_overflows: int
    output_overflows: int
    discontinuities: int
    frames_in: int
    frames_played: int
    min_correction: float
    max_correction: float
    final_fill_ms: float
    min_fill_ms: float
    max_fill_ms: float
    target_ms: float = 0.0
    buffering_changes: int = 0
    frames_lost: int = 0  # mic audio the host buffer overflowed (bursty devices with a small buffer)
    # The least the ring held beyond a cable callback's need: after the first 10 s (simulate), from the first prime
    # (replay).
    min_spare_ms: float = 0.0
    fills_ms: list[float] = field(default_factory=list)
    corrections_ppm: list[float] = field(default_factory=list)  # the correction once a second
    underrun_times: list[float] = field(default_factory=list)  # (simulate only)
    # Hear myself (J06): the speaker sink on its own clock.
    monitor_underruns: int = 0
    monitor_overflows: int = 0
    monitor_discontinuities: int = 0
    monitor_frames_played: int = 0
    monitor_correction_ppm: float = 0.0  # its correction at the end
    monitor_feed_forward_ppm: float = 0.0
    # Remove speaker echo (J06): the loopback reference.
    reference_gaps: int = 0
    reference_overflows: int = 0
    reference_discontinuities: int = 0


class SimTime:
    """Stands in for PortAudio's callback time info: the virtual time the callback ran."""

    def __init__(self) -> None:
        self.currentTime = 0.0
        self.inputBufferAdcTime = 0.0
        self.outputBufferDacTime = 0.0


# Measured burst shapes (see the module docstring): probability per callback and the late range in seconds.
MIC_BURST = (0.12, 0.008, 0.025)
CABLE_BURST = (0.10, 0.004, 0.013)


def simulate(
    seconds: float,
    mic_ppm: float,
    cable_ppm: float,
    chain: Processor | None = None,
    mic_block: int = 480,
    cable_block: int = 441,
    mic_channels: int = 2,
    callback_jitter_s: float = 0.002,
    worker_period_s: float = 0.003,
    hiccup_every_s: float = 2.0,
    hiccup_s: float = 0.015,
    check_tone: bool = True,
    seed: int = 1,
    bursty: bool = False,
    mic_host_buffer_s: float | None = None,
    monitor_ppm: float | None = None,
    monitor_block: int = 441,
    loopback_ppm: float | None = None,
) -> SimResult:
    """``monitor_ppm``: Hear myself on, a speaker sink pulling ``monitor_block`` frames on its own clock.
    ``loopback_ppm``: echo removal on, a bursty loopback stream on its own clock feeding the reference (a tone too)."""
    rng = np.random.default_rng(seed)

    def late(burst: tuple[float, float, float]) -> float:
        base = rng.uniform(0.0, callback_jitter_s)
        if bursty and rng.random() < burst[0]:
            base += rng.uniform(burst[1], burst[2])
        return base

    path = AudioPath(SAMPLE_RATE, mic_channels, SAMPLE_RATE, 2, chain or PassThrough())
    mic_rate = SAMPLE_RATE * (1.0 + mic_ppm * 1e-6)
    cable_rate = SAMPLE_RATE * (1.0 + cable_ppm * 1e-6)
    mic_period = mic_block / mic_rate
    cable_period = cable_block / cable_rate
    phase_step = 2.0 * math.pi * TONE_HZ / SAMPLE_RATE  # the tone is in the mic's own sample clock

    events: list[tuple[float, int, int]] = []  # (time, kind, sequence)
    MIC, CABLE, WORKER, MONITOR, LOOPBACK = 0, 1, 2, 3, 4
    heapq.heappush(events, (mic_period, MIC, 1))
    heapq.heappush(events, (cable_period, CABLE, 1))
    heapq.heappush(events, (worker_period_s, WORKER, 0))
    monitor = None
    monitor_check = Continuity(TONE_AMP * phase_step * 1.25)
    monitor_out = np.zeros((monitor_block, 2), np.float32)
    monitor_clock = SimTime()
    monitor_period = 0.0
    if monitor_ppm is not None:
        monitor = path.add_monitor(SAMPLE_RATE, 2)
        path.monitor_live = True
        monitor_period = monitor_block / (SAMPLE_RATE * (1.0 + monitor_ppm * 1e-6))
        heapq.heappush(events, (monitor_period, MONITOR, 1))
    loop = None
    if loopback_ppm is not None:
        loop = LoopbackFeed(path, loopback_ppm, rng, bursty)
        if check_tone:
            path.set_echo_paused(True)  # AEC3 runs but is crossfaded out: the cable carries the (delayed) tone
        heapq.heappush(events, (loop.period, LOOPBACK, 1))

    outbuf = np.zeros((cable_block, 2), np.float32)
    mic_clock, cable_clock = SimTime(), SimTime()
    mic_index = 0
    last_sample: float | None = None
    max_step = TONE_AMP * phase_step * 1.25  # the steepest step a clean 440 Hz tone can take, plus margin
    discontinuities = 0
    fills: list[float] = []
    corrections: list[float] = []
    underrun_times: list[float] = []
    next_fill_log = 1.0
    next_hiccup = hiccup_every_s
    primed_once = False
    frames_lost = 0
    min_spare = 1 << 40
    # The host buffer holds the requested latency plus the period being delivered.
    host_capacity = int(mic_host_buffer_s * mic_rate) + mic_block if mic_host_buffer_s else 1 << 40

    while events:
        t, kind, seq = heapq.heappop(events)
        if t > seconds:
            break
        if kind == MIC:
            # The device has captured everything up to now; what it held back beyond the host buffer is lost, as a
            # WASAPI capture buffer overflows (silently, on the measured device).
            pending = int(t * mic_rate) - mic_index
            if pending > host_capacity:
                frames_lost += pending - host_capacity
                mic_index += pending - host_capacity  # the lost stretch of the tone never arrives
                pending = host_capacity
            mic_clock.currentTime = t
            for _ in range(pending // mic_block):
                n = np.arange(mic_index, mic_index + mic_block)
                tone = (TONE_AMP * np.sin(phase_step * n)).astype(np.float32)
                path.input_callback(np.repeat(tone[:, None], mic_channels, axis=1), mic_block, mic_clock, None)
                mic_index += mic_block
            due = (seq + 1) * mic_period + late(MIC_BURST)
            heapq.heappush(events, (max(due, t), MIC, seq + 1))
        elif kind == CABLE:
            underruns_before = path.counters.underruns
            if path.primed and t > 10.0:
                min_spare = min(min_spare, path.out_ring.available() - cable_block)
            cable_clock.currentTime = t
            path.output_callback(outbuf, cable_block, cable_clock, None)
            if path.counters.underruns != underruns_before:
                underrun_times.append(t)
            if path.primed:
                primed_once = True
            mono = outbuf[:, 0]
            if check_tone and primed_once and path.counters.underruns == underruns_before:
                if last_sample is not None:
                    steps = np.abs(np.diff(np.concatenate(([last_sample], mono))))
                else:
                    steps = np.abs(np.diff(mono))
                discontinuities += int(np.count_nonzero(steps > max_step))
            last_sample = float(mono[-1]) if primed_once else None
            due = (seq + 1) * cable_period + late(CABLE_BURST)
            heapq.heappush(events, (max(due, t), CABLE, seq + 1))
        elif kind == MONITOR and monitor is not None:
            before = monitor.counters.underruns
            monitor_clock.currentTime = t
            monitor.callback(monitor_out, monitor_block, monitor_clock, None)
            if check_tone:
                monitor_check.feed(monitor_out[:, 0], monitor.primed, monitor.counters.underruns != before)
            due = (seq + 1) * monitor_period + late(CABLE_BURST)
            heapq.heappush(events, (max(due, t), MONITOR, seq + 1))
        elif kind == LOOPBACK and loop is not None:
            loop.deliver(t)
            heapq.heappush(events, (max(loop.next_due(seq + 1), t), LOOPBACK, seq + 1))
        else:
            path.process()
            wake = t + worker_period_s
            if t >= next_hiccup:  # the worker is held up, e.g. by the UI or the disk
                wake += hiccup_s * rng.uniform(0.5, 1.0)
                next_hiccup += hiccup_every_s
            heapq.heappush(events, (wake, WORKER, 0))
        if t >= next_fill_log:
            fills.append(path.out_ring.available() / SAMPLE_RATE * 1000.0)
            corrections.append(path.controller.correction * 1e6)
            next_fill_log += 1.0

    result = _result(
        path,
        seconds,
        mic_ppm,
        cable_ppm,
        discontinuities,
        mic_index - frames_lost,
        frames_lost,
        min_spare,
        fills,
        corrections,
    )
    result.underrun_times = underrun_times
    _extras(result, path, monitor_check, loop)
    return result


class Continuity:
    """Counts jumps in a clean tone, block by block, from the first primed block on (underrun blocks excepted)."""

    def __init__(self, max_step: float) -> None:
        self.max_step = max_step
        self.last: float | None = None
        self.started = False
        self.discontinuities = 0

    def feed(self, mono: npt.NDArray[np.float32], primed: bool, underrun: bool) -> None:
        self.started = self.started or primed
        if self.started and not underrun:
            prev = mono[:1] if self.last is None else np.array([self.last], np.float32)
            steps = np.abs(np.diff(np.concatenate((prev, mono))))
            self.discontinuities += int(np.count_nonzero(steps > self.max_step))
        self.last = float(mono[-1]) if self.started else None


class LoopbackFeed:
    """The speakers' loopback: a 330 Hz tone on its own clock, delivered in bursty 10 ms callbacks like the measured
    WASAPI loopback (J06 Step 3: median 10 ms, max 24 ms), into ``path.loopback_callback``; the reference it produces
    for the AEC is checked for continuity at every worker block."""

    HZ = 330.0

    def __init__(self, path: AudioPath, ppm: float, rng: np.random.Generator, bursty: bool, block: int = 480) -> None:
        self.path = path
        self.rate = SAMPLE_RATE * (1.0 + ppm * 1e-6)
        self.block = block
        self.period = block / self.rate
        self.rng = rng
        self.bursty = bursty
        self.index = 0
        self.clock = SimTime()
        self.step = 2.0 * math.pi * self.HZ / SAMPLE_RATE  # in the loopback's own sample clock
        self.check = Continuity(TONE_AMP * self.step * 1.3)
        self._seen_start = False
        path.enable_echo(SAMPLE_RATE, 2)
        assert path.reference is not None
        path.reference.observer = self._observe

    def _observe(self, block: npt.NDArray[np.float32]) -> None:
        ref = self.path.reference
        if ref is None or not ref._started:
            return
        if not self._seen_start:
            # The reference's first block steps from silence into the tone (as the cable's first prime does): the
            # check starts after it.
            self._seen_start = True
            self.check.last = float(block[-1])
            self.check.started = True
            return
        self.check.feed(block, True, False)

    def next_due(self, seq: int) -> float:
        late = self.rng.uniform(0.0, 0.002)
        if self.bursty and self.rng.random() < 0.05:
            late += self.rng.uniform(0.004, 0.014)
        return seq * self.period + late

    def deliver(self, t: float) -> None:
        pending = int(t * self.rate) - self.index
        for _ in range(max(0, pending) // self.block):
            self.deliver_frames(t, self.block)

    def deliver_frames(self, t: float, frames: int) -> None:
        self.clock.currentTime = t
        n = np.arange(self.index, self.index + frames)
        tone = (TONE_AMP * np.sin(self.step * n)).astype(np.float32)
        self.path.loopback_callback(np.repeat(tone[:, None], 2, axis=1), frames, self.clock, None)
        self.index += frames


def _extras(result: SimResult, path: AudioPath, monitor_check: Continuity, loop: LoopbackFeed | None) -> None:
    counters = path.status_counters()
    result.monitor_underruns = counters.monitor_underruns
    result.monitor_overflows = counters.monitor_overflows
    result.monitor_discontinuities = monitor_check.discontinuities
    if path.monitor is not None:
        result.monitor_frames_played = path.monitor.frames_played
        result.monitor_correction_ppm = path.monitor.controller.correction * 1e6
        result.monitor_feed_forward_ppm = path.monitor.controller.feed_forward * 1e6
    result.reference_gaps = counters.reference_gaps
    result.reference_overflows = counters.reference_overflows
    if loop is not None:
        result.reference_discontinuities = loop.check.discontinuities


FIXTURE = Path(__file__).resolve().parent / "data" / "real_pattern.npz"


@dataclass
class Pattern:
    """A real callback pattern (tools/audio_soak.py --export-fixture): times in seconds from the window's start."""

    seconds: float
    mic_time: npt.NDArray[np.float64]
    mic_frames: npt.NDArray[np.int32]
    cable_time: npt.NDArray[np.float64]
    cable_frames: npt.NDArray[np.int32]
    worker_start: npt.NDArray[np.float64]
    worker_end: npt.NDArray[np.float64]
    chain_cost: npt.NDArray[np.float64]
    source_underruns: int


def load_pattern(path: Path = FIXTURE) -> Pattern:
    with np.load(path) as d:
        return Pattern(
            seconds=float(d["seconds"]),
            mic_time=d["mic_time"],
            mic_frames=d["mic_frames"],
            cable_time=d["cable_time"],
            cable_frames=d["cable_frames"],
            worker_start=d["worker_start"],
            worker_end=d["worker_end"],
            chain_cost=d["chain_cost"],
            source_underruns=int(d["source_underruns"]),
        )


def _looped(
    times: npt.NDArray[np.float64], values: npt.NDArray[Any], period: float, ppm: float, kind: int
) -> Iterator[tuple[float, int, int]]:
    """The pattern repeated every ``period`` seconds, on a clock ``ppm`` fast."""
    scale = 1.0 / (1.0 + ppm * 1e-6)
    pairs = list(zip(times.tolist(), values.tolist(), strict=True))
    loop = 0
    while True:
        base = loop * period
        for t, v in pairs:
            yield ((base + t) * scale, kind, int(v))
        loop += 1


def loop_bias_ppm(p: Pattern) -> float:
    """How much faster the looped mic runs than the looped cable, as recorded: a window cuts a mic burst at its edge,
    so one loop holds a callback more of mic frames than of cable frames (480 of 2 880 000: -167 ppm)."""
    return (float(p.mic_frames.sum()) / float(p.cable_frames.sum()) - 1.0) * 1e6


def replay_events(p: Pattern, mic_ppm: float = 0.0, cable_ppm: float = 0.0) -> Iterator[tuple[float, int, int]]:
    """The looped pattern's mic, cable and worker events in time order, with the clocks exactly ``mic_ppm`` and
    ``cable_ppm`` off: the mic is slowed by the loop's own bias first, so at 0/0 both deliver the same frames per
    loop (J05 review r2, F2: every replay used to run 167 ppm off what it named)."""
    mic_clock_ppm = ((1.0 + mic_ppm * 1e-6) / (1.0 + loop_bias_ppm(p) * 1e-6) - 1.0) * 1e6
    MIC, CABLE, WORKER = 0, 1, 2
    return heapq.merge(
        _looped(p.mic_time, p.mic_frames, p.seconds, mic_clock_ppm, MIC),
        _looped(p.cable_time, p.cable_frames, p.seconds, cable_ppm, CABLE),
        _looped(p.worker_end, np.zeros(len(p.worker_end), np.int32), p.seconds, 0.0, WORKER),
    )


def replay(
    fixture: Pattern | Path = FIXTURE,
    seconds: float = 300.0,
    mic_ppm: float = 0.0,
    cable_ppm: float = 0.0,
    chain: Processor | None = None,
    mic_channels: int = 2,
    monitor: bool = False,
    echo: bool = False,
    monitor_ppm: float = 0.0,
    loopback_ppm: float = 0.0,
) -> SimResult:
    """Drive the real ``AudioPath`` at the recorded callback and worker times.

    Mic callbacks deliver their recorded frame counts at their recorded times, the cable asks for its recorded frame
    counts at its times, and each worker step runs at the end of its recorded span: what it produced becomes visible
    only once it has held the clock for as long as it took on the real machine. The pattern loops for runs longer
    than the fixture; ``mic_ppm`` and ``cable_ppm`` speed up those two clocks on top of their measured rates. The tone
    check (discontinuities) runs only with the pass-through chain.

    ``monitor``: Hear myself on; the speaker sink pulls the recorded cable pattern shifted by half its length.
    ``echo``: echo removal on; the loopback delivers a tone in the recorded mic pattern shifted the same way.
    """
    p = fixture if isinstance(fixture, Pattern) else load_pattern(fixture)
    check_tone = chain is None
    path = AudioPath(SAMPLE_RATE, mic_channels, SAMPLE_RATE, 2, chain or PassThrough())
    MIC, CABLE, MONITOR, LOOPBACK = 0, 1, 3, 4  # replay_events gives 0-2; 2 is a worker step
    streams = [replay_events(p, mic_ppm, cable_ppm)]
    half = p.seconds / 2
    sink = None
    monitor_check = Continuity(TONE_AMP * 2.0 * math.pi * TONE_HZ / SAMPLE_RATE * 1.25)
    if monitor:
        sink = path.add_monitor(SAMPLE_RATE, 2)
        path.monitor_live = True
        t_mon, f_mon = _shifted(p.cable_time, p.cable_frames, p.seconds, half)
        streams.append(_looped(t_mon, f_mon, p.seconds, monitor_ppm, MONITOR))
    loop = None
    if echo:
        loop = LoopbackFeed(path, 0.0, np.random.default_rng(0), False)
        if check_tone:
            path.set_echo_paused(True)  # AEC3 runs but is crossfaded out: the cable carries the (delayed) tone
        t_lb, f_lb = _shifted(p.mic_time, p.mic_frames, p.seconds, half)
        lb_ppm = ((1.0 + loopback_ppm * 1e-6) / (1.0 + loop_bias_ppm(p) * 1e-6) - 1.0) * 1e6
        streams.append(_looped(t_lb, f_lb, p.seconds, lb_ppm, LOOPBACK))
    events = heapq.merge(*streams) if len(streams) > 1 else streams[0]
    mon_out = np.zeros((int(p.cable_frames.max()), 2), np.float32)
    mon_clock = SimTime()
    phase_step = 2.0 * math.pi * TONE_HZ / SAMPLE_RATE
    max_step = TONE_AMP * phase_step * 1.25
    outbuf = np.zeros((int(p.cable_frames.max()), 2), np.float32)
    mic_clock, cable_clock = SimTime(), SimTime()
    corrections: list[float] = []
    mic_index = 0
    last_sample: float | None = None
    primed_once = False
    discontinuities = 0
    fills: list[float] = []
    next_fill_log = 1.0
    min_spare = 1 << 40
    for t, kind, frames in events:
        if t > seconds:
            break
        if kind == MIC:
            n = np.arange(mic_index, mic_index + frames)
            tone = (TONE_AMP * np.sin(phase_step * n)).astype(np.float32)
            mic_clock.currentTime = t
            path.input_callback(np.repeat(tone[:, None], mic_channels, axis=1), frames, mic_clock, None)
            mic_index += frames
        elif kind == CABLE:
            underruns_before = path.counters.underruns
            if path.primed:  # from the very first prime: a start-up cushion that is too small shows here
                min_spare = min(min_spare, path.out_ring.available() - frames)
            out = outbuf[:frames]
            cable_clock.currentTime = t
            path.output_callback(out, frames, cable_clock, None)
            primed_once = primed_once or path.primed
            mono = out[:, 0]
            if check_tone and primed_once and path.counters.underruns == underruns_before:
                prev = mono[:1] if last_sample is None else np.array([last_sample], np.float32)
                steps = np.abs(np.diff(np.concatenate((prev, mono))))
                discontinuities += int(np.count_nonzero(steps > max_step))
            last_sample = float(mono[-1]) if primed_once else None
        elif kind == MONITOR and sink is not None:
            before = sink.counters.underruns
            out = mon_out[:frames]
            mon_clock.currentTime = t
            sink.callback(out, frames, mon_clock, None)
            if check_tone:
                monitor_check.feed(out[:, 0], sink.primed, sink.counters.underruns != before)
        elif kind == LOOPBACK and loop is not None:
            loop.deliver_frames(t, frames)
        else:
            path.process()
        if t >= next_fill_log:
            fills.append(path.out_ring.available() / SAMPLE_RATE * 1000.0)
            corrections.append(path.controller.correction * 1e6)
            next_fill_log += 1.0
    result = _result(path, seconds, mic_ppm, cable_ppm, discontinuities, mic_index, 0, min_spare, fills, corrections)
    _extras(result, path, monitor_check, loop)
    return result


def _shifted(
    times: npt.NDArray[np.float64], values: npt.NDArray[Any], period: float, shift: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any]]:
    """A pattern started ``shift`` seconds later in its loop, so a second stream is not in step with the first."""
    t = (times + shift) % period
    order = np.argsort(t, kind="stable")
    return t[order], values[order]


def _result(
    path: AudioPath,
    seconds: float,
    mic_ppm: float,
    cable_ppm: float,
    discontinuities: int,
    frames_in: int,
    frames_lost: int,
    min_spare: float,
    fills: list[float],
    corrections: list[float],
) -> SimResult:
    settled = fills[10:] or fills
    return SimResult(
        seconds=seconds,
        mic_ppm=mic_ppm,
        cable_ppm=cable_ppm,
        underruns=path.counters.underruns,
        input_overflows=path.counters.input_overflows,
        output_overflows=path.counters.output_overflows,
        discontinuities=discontinuities,
        frames_in=frames_in,
        frames_lost=frames_lost,
        min_spare_ms=min_spare / SAMPLE_RATE * 1000.0,
        frames_played=path.frames_played,
        min_correction=path.controller.min_correction,
        max_correction=path.controller.max_correction,
        final_fill_ms=fills[-1] if fills else 0.0,
        min_fill_ms=min(settled) if settled else 0.0,
        max_fill_ms=max(settled) if settled else 0.0,
        target_ms=path.cable.target_ms,
        buffering_changes=path.cable.buffering_changes,
        fills_ms=fills,
        corrections_ppm=corrections,
    )
