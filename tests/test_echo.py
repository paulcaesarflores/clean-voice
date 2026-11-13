"""Remove speaker echo (J06 Step 8): AEC3, the continuous reference, the mic delay, the engine's echo states."""

from __future__ import annotations

import heapq
import logging
import sys
import time

import numpy as np
import pytest
from drift_sim import SimTime, simulate
from speech import RATE, active_rms_db, speech, word_spans
from test_engine import CABLE, HEADPHONES, MIC, SPEAKERS, FakeBackend, wait_for

import clean_voice.engine as engine_module
from clean_voice.audio_path import BLOCK, ClockEstimator, Counters
from clean_voice.dsp import PassThrough
from clean_voice.engine import ECHO_MAC, ECHO_NO_SPEAKERS, ECHO_PAUSED, EchoState, State

try:
    import pywebrtc_audio  # noqa: F401

    HAVE_AEC = True
except ImportError:
    HAVE_AEC = False
    if sys.platform == "win32":
        raise  # echo removal is pinned for Windows: a missing package there is a failure, not a skip

needs_aec = pytest.mark.skipif(not HAVE_AEC, reason="echo removal is Windows only in v1 (no AEC package here)")
pytestmark = needs_aec


def db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-30))


def voice(level: float = -20.0) -> np.ndarray:
    x = speech().astype(np.float32)
    return x * np.float32(10 ** ((level - active_rms_db(x, word_spans(x))) / 20))


def run_aec(near: np.ndarray, far: np.ndarray) -> np.ndarray:
    from clean_voice.echo import EchoCanceller

    aec = EchoCanceller()
    return np.concatenate([aec.process(near[i : i + BLOCK], far[i : i + BLOCK]) for i in range(0, len(near), BLOCK)])


# ------------------------------------------------------------------ A1, A2: the package (J06 Step 3)


def test_echo_is_removed() -> None:
    # A1: a port of simple-clean-mic audio.rs:551-589 (40 ms speaker -> mic path at half volume), speech as the far end.
    far = voice()
    near = np.zeros_like(far)
    near[48 * 40 :] = 0.5 * far[: -48 * 40]
    out = run_aec(near, far)
    after = slice(5 * RATE, None)
    assert db(out[after]) - db(near[after]) <= -15.0


def test_no_ducking_without_an_echo_path() -> None:
    # A2, as restated by the orchestrator from the J06 Step 3 measurements: unrelated far-end speech, the near end
    # talking, no echo path (headphones). AEC3's fixed capture high-pass costs <= 0.5 dB; beyond it the near-end loss
    # is <= 6 dB in the first 10 s (convergence) and <= 2 dB in every 10 s window after.
    x1 = voice()
    spans1 = word_spans(x1)
    x = np.tile(x1, 3)
    spans = [(s + k * len(x1), e + k * len(x1)) for k in range(3) for s, e in spans1]
    fixed_out = run_aec(x, np.zeros_like(x))
    fixed = active_rms_db(x, spans) - active_rms_db(fixed_out, spans)
    assert fixed <= 0.5
    for shift in (10, 25):
        out = run_aec(x, np.roll(x, shift * RATE))
        for w0 in range(0, len(x) - 10 * RATE + 1, 10 * RATE):
            ws = [(s, e) for s, e in spans if w0 <= s and e <= w0 + 10 * RATE]
            ducking = active_rms_db(fixed_out, ws) - active_rms_db(out, ws)
            assert ducking <= (6.0 if w0 == 0 else 2.0), f"shift {shift} s, window {w0 // RATE} s: {ducking:.2f} dB"


# ------------------------------------------------------------------ the room: render, loopback, echo, mic


class RecordingChain(PassThrough):
    """Keeps what reached the chain: the echo stage's output and the delayed mic it worked on (or, with
    ``energies_only``, just each block's energy of both, for hour-long runs)."""

    def __init__(self, energies_only: bool = False) -> None:
        self.energies_only = energies_only
        self.out: list[np.ndarray] = []
        self.near: list[np.ndarray] = []
        self.e_out: list[float] = []
        self.e_near: list[float] = []
        self.off_block: int | None = None

    def process(self, x, raw=None):
        near = x if raw is None else raw
        if self.energies_only:
            self.e_out.append(float(np.dot(x.astype(np.float64), x)))
            self.e_near.append(float(np.dot(near.astype(np.float64), near)))
        else:
            self.out.append(x.copy())
            self.near.append(near.copy())
        return x.copy()

    def signals(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(self.near), np.concatenate(self.out)


def room(
    loopback_lag_s: float,
    echo_s: float,
    seconds: float = 60.0,
    loopback_ppm: float = 0.0,
    no_delay: bool = False,
    energies_only: bool = False,
    on_second=None,
    far: np.ndarray | None = None,
    off_at_s: float | None = None,
):
    """A simulated speaker -> mic room around the real AudioPath (pass-through chain).

    The far end (speech) is rendered on the speakers, mixed with what the silent render writes (the marker). The mic
    hears the speakers ``echo_s`` after the render callback wrote it, at half level; the loopback delivers the mix
    ``loopback_lag_s`` after the render. The speakers' clock runs ``loopback_ppm`` off the mic's. Returns the path
    and the chain (``near``/``out`` blocks, or per-block energies with ``energies_only``). ``on_second`` is called once a
    simulated second with the measured reference age, when known. ``far``: what the far end plays (default: the speech
    fixture). ``off_at_s``: echo removal is switched off then, as the engine does when the guest turns it off or the
    loopback is lost (the loopback stops delivering); ``chain.off_block`` records the block.
    """
    from clean_voice.audio_path import AudioPath

    chain = RecordingChain(energies_only)
    path = AudioPath(RATE, 1, RATE, 2, chain)
    path.enable_echo(RATE, 1)
    assert path.render is not None and path.echo is not None
    if no_delay:
        path.echo.mic_delay.set_target_s = lambda s: None  # type: ignore[method-assign]
        path.echo.mic_delay.target = path.echo.mic_delay.current = 0.0
    far = voice() if far is None else far
    spk_rate = RATE * (1 + loopback_ppm * 1e-6)
    ring = 4 * RATE  # the speaker mix, kept for the last 4 s
    mix = np.zeros(ring, np.float32)
    rendered = 0
    looped = 0
    mic_index = 0
    MIC, RENDER, LOOP, WORK, CABLE_OUT = 0, 1, 2, 3, 4
    # The first render at 1 us, not 0: a stamp time of 0 means "no time info" to callback_time().
    events = [(0.01, MIC, 1), (1e-6, RENDER, 0), (loopback_lag_s, LOOP, 0), (0.003, WORK, 0), (0.01, CABLE_OUT, 1)]
    heapq.heapify(events)
    t_mic, t_spk, t_loop = SimTime(), SimTime(), SimTime()
    out = np.zeros((480, 2), np.float32)
    render_block = np.zeros((480, 1), np.float32)
    rng = np.random.default_rng(3)
    next_second = 1.0
    while events:
        t, kind, seq = heapq.heappop(events)
        if t > seconds:
            break
        if off_at_s is not None and chain.off_block is None and t >= off_at_s:
            path.disable_echo()
            chain.off_block = len(chain.out)
        if on_second is not None and t >= next_second:
            next_second += 1.0
            stage = path.echo
            age = stage.meter.age_s if stage is not None and stage.meter is not None else None
            if age is not None:
                on_second(age)
        if kind == RENDER:
            t_spk.currentTime = t
            path.render_callback(render_block, 480, t_spk, None)
            k = rendered % ring
            src = (rendered + np.arange(480)) % len(far)
            mix[k : k + 480] = render_block[:, 0] + far[src]
            rendered += 480
            heapq.heappush(events, ((seq + 1) * 480 / spk_rate + 1e-6, RENDER, seq + 1))
        elif kind == LOOP:
            # Everything rendered loopback_lag_s ago, in 480-frame callbacks.
            t_loop.currentTime = t
            due = int((t - loopback_lag_s) * spk_rate)
            while looped + 480 <= min(due, rendered) and chain.off_block is None:  # off: the loopback is closed
                k = looped % ring
                path.loopback_callback(mix[k : k + 480, None], 480, t_loop, None)
                looped += 480
            heapq.heappush(events, (t + 0.01 + rng.uniform(0, 0.002), LOOP, seq + 1))
        elif kind == MIC:
            # Mic block seq-1 captured samples ending at t; the echo of render time tau arrives at tau + echo_s.
            c = np.arange(mic_index, mic_index + 480) / RATE
            r = (c - echo_s) * spk_rate
            r0 = np.floor(r).astype(np.int64)
            frac = (r - r0).astype(np.float32)
            v = mix[r0 % ring] * (1 - frac) + mix[(r0 + 1) % ring] * frac
            block = np.where(r >= 0, 0.5 * v, 0.0).astype(np.float32)
            t_mic.currentTime = t
            path.input_callback(block[:, None], 480, t_mic, None)
            mic_index += 480
            heapq.heappush(events, ((seq + 1) * 0.01, MIC, seq + 1))
        elif kind == CABLE_OUT:
            path.output_callback(out, 480, None, None)
            heapq.heappush(events, ((seq + 1) * 0.01 + 0.005, CABLE_OUT, seq + 1))
        else:
            path.process()
            heapq.heappush(events, (t + 0.003, WORK, 0))
    return path, chain


def removal_db(near: np.ndarray, out: np.ndarray, from_s: float) -> float:
    s = int(from_s * RATE)
    return db(near[s:]) - db(out[s:])


@pytest.mark.parametrize("echo_ms", [20, 40])
@pytest.mark.parametrize("lag_ms", [19, 35, 119])
def test_the_reference_reaches_aec3_before_its_echo(lag_ms: int, echo_ms: int) -> None:
    # The orchestrator's alignment test (J06 Step 3 decision 2): the loopback as late as measured on this PC's
    # headphones (19 ms), speakers (35 ms) and VB-CABLE (119 ms), the echo 20-40 ms after the render. The mic delay,
    # measured from the render marker, must make AEC3 remove >= 15 dB once settled.
    path, chain = room(lag_ms / 1000, echo_ms / 1000)
    near, out = chain.signals()
    stage = path.echo
    assert stage is not None and stage.meter is not None
    age = stage.meter.age_s
    print(
        f"\nloopback {lag_ms} ms, echo {echo_ms} ms: reference age {age * 1000 if age else float('nan'):.1f} ms, "
        f"mic delay {stage.delay_ms:.1f} ms, removal {removal_db(near, out, 45.0):.1f} dB"
    )
    assert age is not None and stage.meter.measurements > 10
    assert removal_db(near, out, 45.0) >= 15.0
    assert path.counters.reference_gaps == 0


def test_without_the_mic_delay_the_reference_arrives_after_its_echo() -> None:
    # The same room with the mic delay held at 0: the reference reaches AEC3 after its echo and little is removed. This
    # is what the test above guards (it goes red the same way if the delay stops working).
    _, chain = room(0.119, 0.020, seconds=45.0, no_delay=True)
    near, out = chain.signals()
    assert removal_db(near, out, 30.0) < 6.0


@pytest.mark.slow
def test_echo_stays_removed_for_an_hour_under_drift() -> None:
    # A8: one hour, the speakers' clock (render and loopback) 100 ppm fast against the mic, the real canceller.
    path, chain = room(0.035, 0.030, seconds=3600.0, loopback_ppm=100, energies_only=True)
    e_near, e_out = np.array(chain.e_near), np.array(chain.e_out)
    per_min = 6000  # blocks
    worst = min(
        10 * np.log10(e_near[m : m + per_min].sum() / max(e_out[m : m + per_min].sum(), 1e-30))
        for m in range(per_min, len(e_near) - per_min + 1, per_min)
    )
    stage = path.echo
    age = stage.meter.age_s if stage is not None and stage.meter is not None else None
    print(
        f"\n1 h echo removal, speakers +100 ppm: worst minute after the first {worst:.1f} dB; reference gaps "
        f"{path.counters.reference_gaps}, overflows {path.counters.reference_overflows}; age "
        f"{age * 1000 if age else float('nan'):.1f} ms, mic delay {stage.delay_ms if stage else 0:.1f} ms"
    )
    assert worst >= 15.0
    assert path.counters.reference_gaps == 0 and path.counters.reference_overflows == 0


# ------------------------------------------------------------------ A3-A5: the reference


def test_the_reference_is_continuous_under_drift_and_bursts() -> None:
    # A3: 10 simulated minutes, the loopback +150 ppm and bursty, a tone through it.
    r = simulate(600.0, 0, 0, loopback_ppm=150, bursty=True)
    print(
        f"\nreference: gaps={r.reference_gaps} overflows={r.reference_overflows} "
        f"discontinuities={r.reference_discontinuities}"
    )
    assert r.underruns == 0 and r.discontinuities == 0
    assert r.reference_gaps == 0 and r.reference_overflows == 0 and r.reference_discontinuities == 0


def test_the_reference_lead_holds_under_drift() -> None:
    # A3 (lag): with the speakers' clock 150 ppm fast, the trim holds the ring's least fill each second at its target
    # (so the lag AEC3 sees stays put: the measured age moves under 2 ms after 30 s); and with an echo path the removal
    # holds in every 10 s window after the first 45 s.
    ages: list[float] = []
    path, chain = room(0.035, 0.030, seconds=90.0, loopback_ppm=150, on_second=ages.append)
    near, out = chain.signals()
    ref = path.reference
    assert ref is not None and ref.fill_min is not None
    assert abs(ref.fill_min - ref.controller.target) / RATE <= 0.005  # a second's minimum wobbles a few ms
    settled = ages[30:]
    assert len(settled) > 50 and max(settled) - min(settled) <= 0.002, (min(settled), max(settled))
    for w0 in range(45, 81, 10):
        assert removal_db(near[: (w0 + 10) * RATE], out[: (w0 + 10) * RATE], w0) >= 15.0, f"window {w0} s"


def deliver(ref, t: float, frames: int, start: int) -> int:
    clock = SimTime()
    clock.currentTime = t + 1e-6  # never 0: a stamp time of 0 means "no time info" to callback_time()
    block = (np.arange(start, start + frames, dtype=np.float32) % 1000 / 1000.0)[:, None]
    ref.callback(block, frames, clock, None)
    return start + frames


def test_a_stalled_loopback_is_filled_and_counted_never_skipped(caplog: pytest.LogCaptureFixture) -> None:
    # A4
    from clean_voice.echo import Reference

    counters = Counters()
    ref = Reference(RATE, 1, ClockEstimator(RATE, "mic"), counters)
    sent = 0
    t = 0.0
    zeros_seen = 0
    for k in range(3000):  # 30 s: a 300 ms stop at 10 s
        t = k * 0.01
        if not (10.0 <= t < 10.3):
            sent = deliver(ref, t, 480, sent)
        out = ref.read_block()
        if 10.0 <= t < 10.6:
            zeros_seen += int(np.count_nonzero(out == 0.0))
    assert counters.reference_gaps >= 1 and zeros_seen > 0
    assert ref.frames_dropped == 0 and ref.frames_used + ref.ring.available() == sent  # nothing skipped
    # A 3 s stop, then the device hands over its backlog at once: re-aligned once, logged and counted.
    caplog.set_level(logging.INFO)
    t += 3.0
    for _ in range(300):
        sent = deliver(ref, t, 480, sent)
    for k in range(200):
        deliver(ref, t + 0.01 * (k + 1), 480, sent)
        sent += 480
        ref.read_block()
    assert ref.realigns == 1 and ref.frames_dropped > 0
    assert "re-aligned the echo reference" in caplog.text


def test_aec_and_the_reference_only_run_in_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    # A5 (as R7): the loopback callback only copies and stamps.
    from clean_voice.audio_path import AudioPath

    path = AudioPath(RATE, 1, RATE, 2, PassThrough())
    path.enable_echo(RATE, 2)
    assert path.echo is not None and path.reference is not None
    inside = [False]

    def guarded(real):
        def call(*args, **kwargs):
            assert not inside[0], "AEC or resampling inside the loopback callback"
            return real(*args, **kwargs)

        return call

    class GuardedResampler:
        def __init__(self, real) -> None:
            self.process = guarded(real.process)

    aec = path.echo._aec
    assert aec is not None
    monkeypatch.setattr(aec, "process", guarded(aec.process))
    monkeypatch.setattr(path.reference, "_resampler", GuardedResampler(path.reference._resampler))
    clock = SimTime()
    block = np.full((480, 2), 0.1, np.float32)
    mic = np.full((480, 1), 0.1, np.float32)
    for k in range(1, 201):
        clock.currentTime = k * 0.01
        inside[0] = True
        try:
            path.loopback_callback(block, 480, clock, None)
        finally:
            inside[0] = False
        path.input_callback(mic, 480, clock, None)
        path.process()
    assert path.reference.stamps.count == 200 and path.blocks_processed == 200


# ------------------------------------------------------------------ A6: the engine's echo states


def test_echo_states_and_devices(make_engine, monkeypatch: pytest.MonkeyPatch) -> None:
    # A6
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    assert engine.status().echo is EchoState.OFF and backend.streams_of("loopback") == []
    main = list(backend.opened)
    engine.set_echo(True)
    wait_for(lambda: engine.status().echo is EchoState.ON)
    loops, renders = backend.streams_of("loopback", SPEAKERS), backend.streams_of("output", SPEAKERS)
    assert len(loops) == 1 and len(renders) == 1  # the loopback and the silent render, on the same speakers
    wait_for(lambda: engine.status().echo_delay_ms > 0)
    # Hear myself on: paused, with the reason; the reference keeps flowing.
    engine.set_monitor(True)
    wait_for(lambda: engine.status().echo is EchoState.PAUSED)
    assert engine.status().echo_reason == ECHO_PAUSED and not loops[0].closed
    engine.set_monitor(False)
    wait_for(lambda: engine.status().echo is EchoState.ON and engine.status().echo_reason is None)
    # A lost loopback: unavailable, retried every RETRY_S (0.2 s here), the mic and cable never touched.
    backend.loopback_stalled = True
    wait_for(lambda: engine.status().echo is EchoState.UNAVAILABLE)
    assert engine.status().echo_reason == ECHO_NO_SPEAKERS and loops[0].closed
    opens = len(backend.streams_of("loopback"))
    time.sleep(1.6)  # each try: the watchdog (0.3 s here) finds it stalled, then the retry (0.2 s)
    assert len(backend.streams_of("loopback")) - opens >= 3
    assert backend.opened == main and not any(s.closed for s in main) and engine.status().state is State.ON
    backend.loopback_stalled = False
    wait_for(lambda: engine.status().echo is EchoState.ON)
    # Off: both close.
    engine.set_echo(False)
    wait_for(lambda: engine.status().echo is EchoState.OFF)
    assert all(s.closed for s in backend.streams_of("loopback")) and all(
        s.closed for s in backend.streams_of("output", SPEAKERS)
    )
    # Echo on survives a mic reconnect.
    engine.set_echo(True)
    wait_for(lambda: engine.status().echo is EchoState.ON)
    backend.present.remove(MIC)
    wait_for(lambda: engine.status().state is State.MIC_LOST)
    backend.present.append(MIC)
    wait_for(lambda: engine.status().state is State.ON and len(backend.opened) == 2)
    wait_for(lambda: sum(not s.closed for s in backend.streams_of("loopback")) == 1)  # reopened on the new path
    wait_for(lambda: engine.status().echo is EchoState.ON)


class CachedDefaultBackend(FakeBackend):
    """A fake whose scanned default output, like PortAudio's, is the one seen at the last rescan, while
    ``default_speakers`` asks the system now (the real backend asks PyAudioWPatch's own PortAudio copy)."""

    def __init__(self, present: list) -> None:
        super().__init__(present)
        self.system_default = HEADPHONES  # what Windows says now
        self._cached_default = self.system_default

    def rescan(self) -> None:
        super().rescan()
        self._cached_default = self.system_default

    def default_output(self):
        return self._cached_default if self._cached_default in self._listed else None

    def default_speakers(self):
        return self.system_default if self.system_default in self.present else None


def test_echo_follows_the_new_default_speakers(tmp_path) -> None:
    # OBS audit I-4b: a default-device source restarts on the new default at once (win-wasapi.cpp:1237-1265), and every
    # (re)connect looks the device up afresh (:607-636). Echo removal listens to a Bluetooth speaker; its battery dies
    # and Windows makes the laptop speakers the default: the far end now plays into the mic. It used to retry the
    # vanished speaker forever ("can't reach your speakers") until the guest turned Clean Voice off and on.
    backend = CachedDefaultBackend([MIC, CABLE, SPEAKERS, HEADPHONES])
    engine = engine_module.Engine(tmp_path / "Recordings", backend, retry_s=0.2, watchdog_s=0.3)
    try:
        engine.start(MIC.name)
        wait_for(lambda: engine.status().state is State.ON)
        engine.set_echo(True)
        wait_for(lambda: engine.status().echo is EchoState.ON)
        assert backend.streams_of("loopback", HEADPHONES)
        main = list(backend.opened)
        backend.present.remove(HEADPHONES)  # the battery dies
        backend.system_default = SPEAKERS  # Windows switches the default to the laptop speakers
        wait_for(
            lambda: bool(backend.streams_of("loopback", SPEAKERS)) and engine.status().echo is EchoState.ON, timeout=5
        )
        assert backend.streams_of("output", SPEAKERS)  # the silent render moved with it
        assert backend.opened == main and engine.status().state is State.ON  # the mic and cable were never touched
    finally:
        engine.close()


def test_echo_says_starting_until_its_streams_are_open(make_engine) -> None:
    # J06 review r1, minor: asked for while the audio is off (or its streams are still opening), Status.echo said OFF.
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    engine.set_echo(True)
    wait_for(lambda: engine.status().echo is EchoState.STARTING)
    engine.start(MIC.name)
    wait_for(lambda: engine.status().echo is EchoState.ON)
    assert len(backend.streams_of("loopback")) == 1


def test_echo_is_unavailable_without_speakers(make_engine, monkeypatch: pytest.MonkeyPatch) -> None:
    # A6: no speakers, the speakers are the cable, or not Windows: unavailable, with the plain reason.
    for default in (None, CABLE):
        engine, backend = make_engine([MIC, CABLE, SPEAKERS])
        backend.default_out = default
        engine.start(MIC.name)
        wait_for(lambda engine=engine: engine.status().state is State.ON)
        engine.set_echo(True)
        wait_for(lambda engine=engine: engine.status().echo is EchoState.UNAVAILABLE)
        assert engine.status().echo_reason == ECHO_NO_SPEAKERS and backend.streams_of("loopback") == []
        engine.close()
    monkeypatch.setattr(engine_module, "echo_platform_reason", lambda: ECHO_MAC)
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    engine.set_echo(True)
    wait_for(lambda: engine.status().echo is EchoState.UNAVAILABLE)
    assert engine.status().echo_reason == ECHO_MAC and backend.streams_of("loopback") == []


# ------------------------------------------------------------------ A7: click-free


def test_echo_on_and_off_is_click_free() -> None:
    # A7: on at 5 s, off at 25 s, on speech; the mic delay moves in and out at 0.5 % speed and AEC3 crossfades.
    from clean_voice.audio_path import AudioPath

    chain = RecordingChain()
    path = AudioPath(RATE, 1, RATE, 2, chain)
    x = np.tile(voice(), 2)
    zeros = np.zeros((480, 2), np.float32)
    out = np.zeros((480, 2), np.float32)
    for k in range(len(x) // 480):
        if k == 500:
            path.enable_echo(RATE, 2)
        if k == 2500:
            path.disable_echo()
        if path.reference is not None:
            path.loopback_callback(zeros, 480, None, None)
        path.input_callback(x[k * 480 : (k + 1) * 480, None], 480, None, None)
        path.process()
        path.output_callback(out, 480, None, None)
    y = np.concatenate(chain.out).astype(np.float64)
    steps = np.abs(np.diff(y))
    steady = max(steps[: 4 * RATE].max(), steps[-5 * RATE :].max())
    for at in (5, 25):
        assert steps[at * RATE - 480 : (at + 20) * RATE].max() <= 1.2 * steady, f"a click near {at} s"
    assert path.echo is None  # ramped all the way out, then removed


def test_echo_off_while_removing_an_echo_crossfades() -> None:
    # A7 with a real echo (J06 review r1, F1): turning echo removal off (the guest, or the loopback watchdog) used to
    # drop the reference at once, so the echo came back in one sample. The stage keeps its reference until the 10 ms
    # crossfade is done. The far end is a 317 Hz tone (not locked to the 480-sample blocks), so a hard switch shows as
    # energy above 3 kHz.
    from test_dsp import energy_above_db, max_step, tone

    path, chain = room(0.035, 0.030, seconds=41.0, far=tone(317, 8.0, 0.1), off_at_s=40.0404)
    near, y = chain.signals()
    assert chain.off_block is not None
    s = chain.off_block * 480
    before = slice(s - 2 * RATE, s - 480)
    assert db(near[before]) - db(y[before]) >= 15.0  # the echo really was being removed
    first = slice(s, s + 480)  # the 10 ms the crossfade takes
    assert float(np.max(np.abs(y[first] - near[first]))) > 1e-3, "a hard switch: the raw mic from the first sample"
    steady = max_step(y[s + 4800 : s + 24000])  # the returned echo's own steps
    assert max_step(y[s - 480 : s + 960]) <= 1.2 * steady
    assert energy_above_db(y, s) <= -45.0  # a hard switch measured -22.8 dB (review r1); the crossfade -54.9 dB
    assert path.echo is None or path.echo.reference is None  # and it lets go of the reference once crossfaded


# ------------------------------------------------------------------ what the screen shows (J06 review r2, M1 and M2)


def test_echo_stops_saying_on_once_the_audio_stops(make_engine) -> None:
    # M1: after Stop, or a lost mic, nothing listens to the speakers, so the echo state must not say ON.
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    engine.set_echo(True)
    engine.start(MIC.name)
    wait_for(lambda: engine.status().echo is EchoState.ON)
    engine.stop()
    wait_for(lambda: engine.status().state is State.OFF)
    assert engine.status().echo is EchoState.STARTING
    engine.start(MIC.name)
    wait_for(lambda: engine.status().echo is EchoState.ON)
    backend.present.remove(MIC)
    wait_for(lambda: engine.status().state is State.MIC_LOST)
    assert engine.status().echo is EchoState.STARTING
    backend.present.append(MIC)
    wait_for(lambda: engine.status().state is State.ON and engine.status().echo is EchoState.ON)


def _echo_states_for(engine, seconds: float) -> set[tuple[EchoState, str | None]]:
    seen = set()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = engine.status()
        seen.add((status.echo, status.echo_reason))
        time.sleep(0.005)
    return seen


def test_echo_holds_unavailable_while_a_stalled_loopback_is_retried(make_engine) -> None:
    # M2: the state used to cycle UNAVAILABLE, STARTING, ON on every retry, so the reason text blinked.
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    engine.set_echo(True)
    wait_for(lambda: engine.status().echo is EchoState.ON)
    backend.loopback_stalled = True
    wait_for(lambda: engine.status().echo is EchoState.UNAVAILABLE)
    opens = len(backend.streams_of("loopback"))
    assert _echo_states_for(engine, 1.6) == {(EchoState.UNAVAILABLE, ECHO_NO_SPEAKERS)}
    assert len(backend.streams_of("loopback")) - opens >= 3  # it kept retrying meanwhile
    backend.loopback_stalled = False
    wait_for(lambda: engine.status().echo is EchoState.ON and engine.status().echo_reason is None)


def test_echo_holds_unavailable_while_the_speakers_will_not_open(make_engine) -> None:
    # M2: speakers that refuse to open cycled STARTING and UNAVAILABLE.
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    backend.supports_loopback = False
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    engine.set_echo(True)
    wait_for(lambda: engine.status().echo is EchoState.UNAVAILABLE)
    assert _echo_states_for(engine, 1.2) == {(EchoState.UNAVAILABLE, ECHO_NO_SPEAKERS)}
    backend.supports_loopback = True
    wait_for(lambda: engine.status().echo is EchoState.ON)
