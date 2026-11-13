"""Hear myself (the speaker sink) and Test your mic (J06 Step 7): H1-H5, T1-T5."""

from __future__ import annotations

import time

import numpy as np
import pytest
from drift_sim import SimTime, simulate
from test_engine import CABLE, CABLE_REC, HEADPHONES, MIC, SPEAKERS, VOICEMEETER, wait_for

import clean_voice.engine as engine_module
from clean_voice.audio_path import MIC_TEST_S, SAMPLE_RATE, AudioPath, MicTestState
from clean_voice.devices import Device, monitor_device
from clean_voice.dsp import PassThrough
from clean_voice.engine import State


def test_hear_myself_has_its_own_drift_correction() -> None:
    # H1: a third clock, 150 ppm slow, on top of mic +100 and cable -100.
    r = simulate(120.0, 100, -100, monitor_ppm=-150)
    print(
        f"\nmonitor: underruns={r.monitor_underruns} overflows={r.monitor_overflows} "
        f"discontinuities={r.monitor_discontinuities} correction={r.monitor_correction_ppm:+.0f} ppm "
        f"(feed-forward {r.monitor_feed_forward_ppm:+.0f})"
    )
    assert r.underruns == 0 and r.discontinuities == 0
    assert r.monitor_underruns == 0 and r.monitor_overflows == 0 and r.monitor_discontinuities == 0
    gap = ((1 - 150e-6) / (1 + 100e-6) - 1) * 1e6  # what the monitor needs relative to the mic: -250 ppm
    assert abs(r.monitor_feed_forward_ppm - gap) < 60.0
    assert r.monitor_frames_played > 100 * SAMPLE_RATE


def test_monitor_device_rule() -> None:
    # H2 (simple-clean-mic ui/main.js:452-458): the default output unless it is the cable or virtual, else the first
    # output that is neither, else none.
    outs = [CABLE, VOICEMEETER, SPEAKERS, HEADPHONES]
    assert monitor_device(outs, HEADPHONES, CABLE) == HEADPHONES
    assert monitor_device(outs, CABLE, CABLE) == SPEAKERS
    assert monitor_device(outs, VOICEMEETER, CABLE) == SPEAKERS
    assert monitor_device(outs, None, CABLE) == SPEAKERS
    assert monitor_device([CABLE, VOICEMEETER, Device(9, "Virtual Speakers", 2, 48_000.0)], CABLE, CABLE) is None


def test_hear_myself_is_off_at_start_and_never_reopens_the_main_streams(make_engine) -> None:
    # H3 (report cause 5: restarting the main streams caused dropouts)
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    assert not engine.status().monitor
    opened = list(backend.opened)
    engine.set_monitor(True)
    wait_for(lambda: engine.status().monitor and len(backend.streams_of("output", SPEAKERS)) == 1)
    wait_for(lambda: backend.streams_of("output", SPEAKERS)[0].calls > 20)
    assert engine.status().monitor_device == SPEAKERS.name
    engine.set_monitor(False)
    wait_for(lambda: not engine.status().monitor and backend.streams_of("output", SPEAKERS)[0].closed)
    assert backend.opened == opened and not any(s.closed for s in opened)
    assert engine.status().state is State.ON


def test_a_lost_monitor_device_turns_hear_myself_off(make_engine) -> None:
    # H4: the headphones are unplugged; the voice to Meet carries on untouched.
    engine, backend = make_engine([MIC, CABLE, HEADPHONES])
    backend.default_out = HEADPHONES
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    engine.set_monitor(True)
    wait_for(lambda: engine.status().monitor)
    wait_for(lambda: engine.status().recording_seconds > 1.0)
    before = engine.status().counters
    opened = list(backend.opened)
    backend.present.remove(HEADPHONES)
    t0 = time.monotonic()
    wait_for(lambda: not engine.status().monitor, timeout=2.0)
    assert time.monotonic() - t0 < 2.0
    status = engine.status()
    assert status.monitor_error and "Hear myself" in status.monitor_error
    assert status.state is State.ON and backend.opened == opened
    after = status.counters
    assert (after.underruns, after.input_overflows, after.output_overflows) == (
        before.underruns,
        before.input_overflows,
        before.output_overflows,
    )


def guard(real, inside: list[bool]):
    def call(*args, **kwargs):
        assert not inside[0], "DSP inside a device callback"
        return real(*args, **kwargs)

    return call


def test_the_monitor_callback_only_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    # H5 (as R7): no chain and no resampler inside the speaker sink's callback.
    path = AudioPath(SAMPLE_RATE, 2, SAMPLE_RATE, 2, PassThrough())
    sink = path.add_monitor(SAMPLE_RATE, 2)
    path.monitor_live = True
    inside = [False]

    class Guarded:
        def __init__(self, real) -> None:
            self.process = guard(real.process, inside)

    monkeypatch.setattr(path.chain, "process", guard(path.chain.process, inside))
    monkeypatch.setattr(sink, "_resampler", Guarded(sink._resampler))
    block = np.full((480, 2), 0.25, np.float32)
    out = np.zeros((441, 2), np.float32)
    clock = SimTime()
    for k in range(1, 101):
        clock.currentTime = k * 0.01
        path.input_callback(block, 480, clock, None)
        path.process()
        inside[0] = True
        try:
            path.monitor_callback(out, 441, clock, None)
        finally:
            inside[0] = False
    assert sink.primed and sink.stamps.count == 100


def test_speaker_device_errors_reach_the_totals() -> None:
    # J06 review r1, minor: a glitching headset was invisible in the totals (status_counters left it out).
    path = AudioPath(SAMPLE_RATE, 2, SAMPLE_RATE, 2, PassThrough())
    path.add_monitor(SAMPLE_RATE, 2)
    out = np.zeros((480, 2), np.float32)
    for _ in range(3):
        path.monitor_callback(out, 480, None, True)  # PortAudio reports an output underflow
    counters = path.status_counters()
    assert counters.monitor_device_errors == 3 and counters.output_device_errors == 0


def test_nothing_queues_before_the_speakers_start() -> None:
    # J06 Step 10: a Bluetooth headset took ~0.5 s to start; the worker queued into its sink meanwhile, and Hear
    # myself then ran ~480 ms late for good. Nothing is queued until the stream's first callback.
    path = AudioPath(SAMPLE_RATE, 2, SAMPLE_RATE, 2, PassThrough())
    sink = path.add_monitor(SAMPLE_RATE, 2)
    path.monitor_live = True
    block = np.full((480, 2), 0.25, np.float32)
    for _ in range(50):  # 0.5 s with the stream not started
        path.input_callback(block, 480, None, None)
        path.process()
    assert sink.ring.available() == 0
    out = np.zeros((480, 2), np.float32)
    for _ in range(50):
        path.input_callback(block, 480, None, None)
        path.process()
        path.monitor_callback(out, 480, None, None)
    assert sink.ring.available() < 0.1 * SAMPLE_RATE  # about the cushion, not half a second


def test_slow_speaker_streams_never_hold_up_the_voice(make_engine) -> None:
    # J06 Step 10: opening the loopback, the silent render and the monitor on the worker thread starved the cable
    # (an underrun when they came on). A device that takes 0.4 s to open must not stop the worker.
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    real_output, real_loopback = backend.open_output, backend.open_loopback

    def slow(real):
        def call(*args, **kwargs):
            time.sleep(0.4)
            return real(*args, **kwargs)

        return call

    backend.open_output = slow(real_output)  # type: ignore[method-assign]
    backend.open_loopback = slow(real_loopback)  # type: ignore[method-assign]
    path = engine._path
    assert path is not None
    engine.set_monitor(True)
    engine.set_echo(True)
    t0 = time.monotonic()
    blocks = path.blocks_processed
    longest = 0.0
    last = (blocks, t0)
    while time.monotonic() - t0 < 2.0:
        n = path.blocks_processed
        now = time.monotonic()
        if n != last[0]:
            last = (n, now)
        longest = max(longest, now - last[1])
        time.sleep(0.005)
    assert longest < 0.15, f"the worker stood still for {longest * 1000:.0f} ms"
    # Echo removal opens the loopback on Windows only; on a Mac it is unavailable and never opens one (DESIGN).
    loopbacks = 1 if engine_module.echo_platform_reason() is None else 0
    wait_for(lambda: len(backend.streams_of("loopback")) == loopbacks and engine.status().monitor)


# ------------------------------------------------------------------ Test your mic


def drive(path: AudioPath, blocks: int, value_of=lambda k: 0.1) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run the path block by block; return what went to the cable sink and to the speaker sink (48 kHz blocks)."""
    to_cable: list[np.ndarray] = []
    to_monitor: list[np.ndarray] = []
    real_cable = path.cable.push

    def cable_push(clean):
        to_cable.append(clean.copy())
        real_cable(clean)

    path.cable.push = cable_push  # type: ignore[method-assign]
    out = np.zeros((480, 2), np.float32)
    for k in range(blocks):
        m = path.monitor
        if m is not None and not hasattr(m, "_spied"):
            real = m.push

            def mon_push(clean, _real=real):
                to_monitor.append(clean.copy())
                _real(clean)

            m.push = mon_push  # type: ignore[method-assign]
            m._spied = True  # type: ignore[attr-defined]
        rng = np.random.default_rng(k)
        path.input_callback((rng.standard_normal((480, 2)) * value_of(k)).astype(np.float32), 480, None, None)
        path.process()
        path.output_callback(out, 480, None, None)
        if path.monitor is not None:
            path.monitor_callback(out, 480, None, None)
    path.cable.push = real_cable  # type: ignore[method-assign]
    return to_cable, to_monitor


def test_mic_test_records_what_others_hear_and_plays_it_back() -> None:
    # T1
    path = AudioPath(SAMPLE_RATE, 2, SAMPLE_RATE, 2, PassThrough())
    path.add_monitor(SAMPLE_RATE, 2)
    states = [path.mic_test.state]
    path.mic_test.start()
    states.append(path.mic_test.state)
    start = path.blocks_processed
    to_cable, to_monitor = drive(path, 2, lambda k: 0.1)
    n = int(MIC_TEST_S * SAMPLE_RATE / 480)
    for _ in range(2 * n + 40):
        c, m = drive(path, 1, lambda k: 0.1)
        to_cable += c
        to_monitor += m
        if path.mic_test.state is not states[-1]:
            states.append(path.mic_test.state)
    assert states[:3] == [MicTestState.IDLE, MicTestState.RECORDING, MicTestState.PLAYING]
    recorded = np.concatenate(to_cable[:n])
    assert len(recorded) == pytest.approx(MIC_TEST_S * SAMPLE_RATE, abs=480)
    played = np.concatenate(to_monitor)
    k = int(np.flatnonzero(np.any(np.abs(played.reshape(-1, 480)) > 0, axis=1))[0]) * 480
    assert np.array_equal(played[k : k + len(recorded)], recorded)  # the same samples, in order
    assert path.blocks_processed > start


def test_mic_test_ends_idle_with_silence_then_finishing() -> None:
    # T1 (end) and T5: after the playback, 100 ms of silence, then the empty ring is not an underrun.
    path = AudioPath(SAMPLE_RATE, 2, SAMPLE_RATE, 2, PassThrough())
    sink = path.add_monitor(SAMPLE_RATE, 2)
    path.mic_test.start()
    n = int(MIC_TEST_S * SAMPLE_RATE / 480)
    drive(path, 2 * n + 10)
    assert path.mic_test.state is MicTestState.IDLE
    assert sink.finishing
    drive(path, 50)  # the stream would be closed by now; the empty ring fades out uncounted
    assert path.status_counters().monitor_underruns == 0


def test_cancel_returns_to_idle(make_engine) -> None:
    # T2
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    engine.start_mic_test()
    wait_for(lambda: engine.status().mic_test is MicTestState.RECORDING)
    wait_for(lambda: len(backend.streams_of("output", SPEAKERS)) == 1)
    engine.cancel_mic_test()
    wait_for(lambda: engine.status().mic_test is MicTestState.IDLE)
    wait_for(lambda: backend.streams_of("output", SPEAKERS)[0].closed)
    # ...and while playing.
    path = engine._path
    assert path is not None
    engine.start_mic_test()
    wait_for(lambda: engine.status().mic_test is MicTestState.RECORDING)
    path.mic_test.recorded = path.mic_test.buffer.shape[0]  # skip ahead to the playback
    wait_for(lambda: engine.status().mic_test is MicTestState.PLAYING)
    engine.cancel_mic_test()
    wait_for(lambda: engine.status().mic_test is MicTestState.IDLE)
    wait_for(lambda: all(s.closed for s in backend.streams_of("output", SPEAKERS)))
    assert engine.status().state is State.ON


def test_hear_myself_resumes_after_a_mic_test(make_engine) -> None:
    # T3: live voice, the test playback, live voice again, on one stream.
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    engine.set_monitor(True)
    wait_for(lambda: engine.status().monitor)
    path = engine._path
    assert path is not None
    engine.start_mic_test()
    wait_for(lambda: engine.status().mic_test is MicTestState.RECORDING)
    path.mic_test.recorded = path.mic_test.buffer.shape[0] - 4800
    wait_for(lambda: engine.status().mic_test is MicTestState.PLAYING)
    # The playback moves one block per fake mic block: 5.1 s of them, whose 10 ms sleeps ran ~4x long on the macOS
    # runner (CI run 43368578097), so the wait allows for a slow machine.
    wait_for(lambda: engine.status().mic_test is MicTestState.IDLE, timeout=60)
    time.sleep(0.3)
    streams = backend.streams_of("output", SPEAKERS)
    assert len(streams) == 1 and not streams[0].closed and engine.status().monitor
    assert path.monitor is not None and not path.monitor.finishing
    assert path.status_counters().monitor_underruns == 0


def test_speakers_lost_during_a_mic_test_blame_only_the_test(make_engine) -> None:
    # J06 review r1, minor: with only the mic test on the speakers, losing them said "Hear myself turned off".
    engine, backend = make_engine([MIC, CABLE, SPEAKERS])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    engine.start_mic_test()
    wait_for(lambda: engine.status().mic_test is MicTestState.RECORDING)
    wait_for(lambda: len(backend.streams_of("output", SPEAKERS)) == 1 and backend.streams_of("output")[0].calls > 5)
    backend.present.remove(SPEAKERS)
    wait_for(lambda: engine.status().mic_test_error is not None)
    status = engine.status()
    assert status.mic_test is MicTestState.IDLE and status.monitor_error is None and not status.monitor
    assert status.state is State.ON


def test_no_speakers_means_no_mic_test(make_engine) -> None:
    # T4
    engine, backend = make_engine([MIC, CABLE, CABLE_REC])
    backend.default_out = CABLE
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    engine.start_mic_test()
    wait_for(lambda: engine.status().mic_test_error is not None)
    status = engine.status()
    assert status.mic_test is MicTestState.IDLE and "speakers" in (status.mic_test_error or "")
    assert backend.streams_of("output") == []
    path = engine._path
    assert path is not None and path.mic_test.recorded == 0


# ------------------------------------------------------------------ Hear myself volume


def live_path(volume: float | None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Hear myself on for 60 blocks; ``volume`` (a gain) is set after the first 30 when not None."""
    path = AudioPath(SAMPLE_RATE, 2, SAMPLE_RATE, 2, PassThrough())
    path.add_monitor(SAMPLE_RATE, 2)
    path.monitor_live = True
    to_cable, to_monitor = drive(path, 30)
    if volume is not None:
        path.monitor_volume = volume
    c, m = drive(path, 30)
    return to_cable + c, to_monitor + m


def test_hear_myself_volume_moves_the_monitor_and_never_the_cable() -> None:
    cable_ref, monitor_ref = live_path(None)
    cable, monitor = live_path(0.25)
    assert all(np.array_equal(a, b) for a, b in zip(cable, cable_ref, strict=True))  # the cable is bit-identical
    assert len(monitor) == len(monitor_ref)
    before, after = len(monitor) - 30 - 1, len(monitor) - 1
    assert np.array_equal(monitor[before], monitor_ref[before])  # untouched until the change
    assert np.allclose(monitor[after], 0.25 * monitor_ref[after], atol=1e-7)  # then a quarter of the level
    # The change is a 10 ms linear ramp (dsp.Ramp), not a step: the first changed block glides from 1 to 0.25.
    first = len(monitor) - 30
    gain = monitor[first] / np.where(monitor_ref[first] == 0, 1, monitor_ref[first])
    assert gain[0] > 0.99 and gain[-1] == pytest.approx(0.25, abs=1e-5)
    assert np.all(np.diff(gain[np.abs(monitor_ref[first]) > 1e-3]) <= 1e-5)


@pytest.mark.parametrize("volume", [0.0, 0.25])
def test_the_mic_test_plays_at_full_level_whatever_the_hear_myself_volume(volume: float) -> None:
    # final-handoff r1 m1: at Hear myself 0 % the test played silence, and the mic looked broken. The test is about
    # hearing yourself at all, so its playback ignores the Hear myself volume.
    path = AudioPath(SAMPLE_RATE, 2, SAMPLE_RATE, 2, PassThrough())
    path.add_monitor(SAMPLE_RATE, 2)
    path.monitor_volume = volume
    path.mic_test.start()
    n = int(MIC_TEST_S * SAMPLE_RATE / 480)
    to_cable, to_monitor = drive(path, 2 * n + 10)
    recorded = np.concatenate(to_cable[:n])
    played = np.concatenate(to_monitor)
    nonzero = np.flatnonzero(np.any(np.abs(played.reshape(-1, 480)) > 0, axis=1))
    assert len(nonzero), "the playback was silent"
    k = int(nonzero[0]) * 480
    assert np.array_equal(played[k : k + len(recorded)], recorded)  # unity: the same samples


def test_the_engine_hands_the_hear_myself_volume_to_the_path(make_engine) -> None:
    engine, _ = make_engine([MIC, CABLE, SPEAKERS])
    # An audio taper, gain = (percent/100)^2 (coordinator's decision): 50 % is about -12 dB, roughly half as loud.
    engine.set_monitor_volume(50.0)  # before start: kept for the path that opens later
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    path = engine._path
    assert path is not None
    wait_for(lambda: path.monitor_volume == pytest.approx(0.25))
    engine.set_monitor_volume(0.0)
    wait_for(lambda: path.monitor_volume == 0.0)  # silent
    engine.set_monitor_volume(100.0)
    wait_for(lambda: path.monitor_volume == pytest.approx(1.0))  # unity
