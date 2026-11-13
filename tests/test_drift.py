"""The mic and VB-CABLE on clocks +-100 ppm apart: zero underruns, zero dropped chunks.

The normal suite simulates minutes of audio; ``-m slow`` runs the full hour (WORKPLAN J05/J06 done-when).
"""

import logging
from itertools import pairwise

import numpy as np
import pytest
from drift_sim import (
    FIXTURE,
    SimResult,
    SimTime,
    four_filter_chain,
    load_pattern,
    loop_bias_ppm,
    replay,
    replay_events,
    simulate,
)

from clean_voice.audio_path import (
    FADE_S,
    MAX_CORRECTION,
    PRIME_SLACK_S,
    SAMPLE_RATE,
    TARGET_FILL_S,
    AudioPath,
    Sink,
)

START_MS = TARGET_FILL_S * 1000  # the start cushion (60 ms)
START_FRAMES = int(TARGET_FILL_S * SAMPLE_RATE)
from clean_voice.dsp import Chain, PassThrough


def assert_clean_run(r: SimResult, max_fill_ms: float = 70.0) -> None:
    assert r.underruns == 0
    assert r.input_overflows == 0
    assert r.output_overflows == 0
    assert r.discontinuities == 0  # a dropped or skipped chunk would break the tone
    assert -MAX_CORRECTION <= r.min_correction <= r.max_correction <= MAX_CORRECTION
    # The cushion stays near its target: never close to empty, never growing without bound.
    assert 20.0 < r.min_fill_ms and r.max_fill_ms < max_fill_ms
    # Every mic sample reaches the cable: counted in the cable's clock, only the cushion, the block in the worker
    # and the resampler's delay are still in flight.
    in_flight = r.frames_in * (1 + r.cable_ppm * 1e-6) / (1 + r.mic_ppm * 1e-6) - r.frames_played
    assert 0 <= in_flight < (max_fill_ms / 1000 + 0.03) * SAMPLE_RATE


@pytest.mark.parametrize(("mic_ppm", "cable_ppm"), [(100, -100), (-100, 100), (100, 0), (0, -100)])
def test_minutes_of_drift_have_no_underruns_or_drops(mic_ppm: int, cable_ppm: int) -> None:
    r = simulate(240.0, mic_ppm, cable_ppm)
    assert_clean_run(r)
    assert r.buffering_changes == 0  # the correction holds the cushion: no late audio, so no added buffering


def test_a_cable_with_large_blocks_gets_a_bigger_cushion() -> None:
    # Some drivers ask for 100 ms at a time, more than the start cushion: add buffering instead of starving.
    r = simulate(60.0, 100, -100, cable_block=4800)
    assert r.underruns == 0
    assert r.discontinuities == 0


def test_full_chain_minute_of_drift_has_no_underruns() -> None:
    r = simulate(60.0, 100, -100, chain=Chain(), check_tone=False)
    assert_clean_run(r)


def test_four_filter_chain_minute_of_drift_has_no_underruns() -> None:
    # J10: the done-when's Fine-tune chain (gate, EQ, compressor, gain) in the real audio path.
    chain = four_filter_chain()
    assert [f.kind for f in chain.cell.value.filters] == [
        "noise_gate_filter",
        "basic_eq_filter",
        "compressor_filter",
        "gain_filter",
    ]
    r = simulate(60.0, 100, -100, chain=chain, check_tone=False)
    assert_clean_run(r)


def test_fixed_ratio_would_have_failed() -> None:
    # The same clocks with the correction disabled (the original app's fixed ratio) drift out of the cushion. The
    # +-0 clamp applies to the whole correction, feed-forward included, so neither part can act. Since late audio adds
    # its lateness at once (engine-obs r1, B1), the drain shows as buffering raises (each a faded 10 ms gap) before it
    # shows as underruns; with the correction the same clocks need none (test_minutes_of_drift_have_no_underruns_...).
    from clean_voice import audio_path

    saved = audio_path.MAX_CORRECTION
    audio_path.MAX_CORRECTION = 0.0
    try:
        r = simulate(240.0, -100, 100)
    finally:
        audio_path.MAX_CORRECTION = saved
    assert r.underruns + r.buffering_changes >= 3, (r.underruns, r.buffering_changes)


@pytest.mark.slow
@pytest.mark.parametrize(("mic_ppm", "cable_ppm"), [(100, -100), (-100, 100)])
def test_one_hour_of_drift_pass_through(mic_ppm: int, cable_ppm: int) -> None:
    r = simulate(3600.0, mic_ppm, cable_ppm)
    print(
        f"\n1 h mic {mic_ppm:+} ppm / cable {cable_ppm:+} ppm (pass-through): underruns={r.underruns} "
        f"input_overflows={r.input_overflows} output_overflows={r.output_overflows} "
        f"discontinuities={r.discontinuities} "
        f"correction={r.min_correction * 1e6:+.0f}..{r.max_correction * 1e6:+.0f} ppm "
        f"fill={r.min_fill_ms:.1f}..{r.max_fill_ms:.1f} ms frames_in={r.frames_in} frames_played={r.frames_played}"
    )
    assert_clean_run(r)


@pytest.mark.slow
def test_one_hour_of_drift_four_filter_chain() -> None:
    # T17 (J10 done-when 4): the hour with Fine-tune's test chain, gate, EQ, compressor and gain, at OBS's defaults.
    r = simulate(3600.0, 100, -100, chain=four_filter_chain(), check_tone=False)
    print(
        f"\n1 h mic +100 ppm / cable -100 ppm (four-filter chain): underruns={r.underruns} "
        f"input_overflows={r.input_overflows} output_overflows={r.output_overflows} "
        f"correction={r.min_correction * 1e6:+.0f}..{r.max_correction * 1e6:+.0f} ppm "
        f"fill={r.min_fill_ms:.1f}..{r.max_fill_ms:.1f} ms frames_in={r.frames_in} frames_played={r.frames_played}"
    )
    assert_clean_run(r)


@pytest.mark.slow
def test_one_hour_of_drift_full_chain() -> None:
    r = simulate(3600.0, 100, -100, chain=Chain(), check_tone=False)
    print(
        f"\n1 h mic +100 ppm / cable -100 ppm (full chain): underruns={r.underruns} "
        f"input_overflows={r.input_overflows} output_overflows={r.output_overflows} "
        f"correction={r.min_correction * 1e6:+.0f}..{r.max_correction * 1e6:+.0f} ppm "
        f"fill={r.min_fill_ms:.1f}..{r.max_fill_ms:.1f} ms"
    )
    assert_clean_run(r)


# ------------------------------------------------------------------ J06: everything on (D1, D2)


def assert_side_streams_clean(r: SimResult, monitor: bool = True, echo: bool = True) -> None:
    if monitor:
        assert r.monitor_underruns == 0 and r.monitor_overflows == 0 and r.monitor_discontinuities == 0
        assert r.monitor_frames_played > 0
    if echo:
        assert r.reference_gaps == 0 and r.reference_overflows == 0 and r.reference_discontinuities == 0


def print_all(label: str, r: SimResult) -> None:
    print(
        f"\n{label}: underruns={r.underruns} input_overflows={r.input_overflows} output_overflows={r.output_overflows}"
        f" discontinuities={r.discontinuities} correction={r.min_correction * 1e6:+.0f}..{r.max_correction * 1e6:+.0f}"
        f" ppm fill={r.min_fill_ms:.1f}..{r.max_fill_ms:.1f} ms | monitor underruns={r.monitor_underruns}"
        f" overflows={r.monitor_overflows} discontinuities={r.monitor_discontinuities}"
        f" correction={r.monitor_correction_ppm:+.0f} ppm | reference gaps={r.reference_gaps}"
        f" overflows={r.reference_overflows} discontinuities={r.reference_discontinuities}"
    )


def test_minutes_of_drift_with_everything_on() -> None:
    # D1: the full chain, echo removal on (loopback +50 ppm) and Hear myself on (monitor -150 ppm), mic +100 / cable
    # -100; the monitor and the reference carry tones, checked for continuity.
    r = simulate(120.0, 100, -100, chain=Chain(), check_tone=False, monitor_ppm=-150, loopback_ppm=50)
    print_all("2 min everything on (full chain)", r)
    assert_clean_run(r)
    assert_side_streams_clean(r, monitor=False)  # the monitor carries the chain's output, not a tone here
    assert r.monitor_underruns == 0 and r.monitor_overflows == 0
    p = simulate(120.0, 100, -100, monitor_ppm=-150, loopback_ppm=50)  # pass-through: the tone checks everywhere
    print_all("2 min everything on (pass-through)", p)
    assert_clean_run(p)
    assert_side_streams_clean(p)


def test_real_pattern_with_hear_myself_and_echo_on() -> None:
    # D1: the measured callback pattern for 5 min, the monitor and the loopback on shifted copies of it.
    r = replay(load_pattern(), 300.0, monitor=True, echo=True)
    print_all("real pattern 300 s, monitor + echo", r)
    assert r.underruns == 0 and r.input_overflows == 0 and r.output_overflows == 0 and r.discontinuities == 0
    assert_side_streams_clean(r)


@pytest.mark.slow
def test_one_hour_full_chain_echo_on() -> None:
    # D2
    r = simulate(3600.0, 100, -100, chain=Chain(), check_tone=False, loopback_ppm=50)
    print_all("1 h full chain, echo on (mic +100, cable -100, loopback +50)", r)
    assert_clean_run(r)
    assert_side_streams_clean(r, monitor=False)


@pytest.mark.slow
def test_one_hour_full_chain_hear_myself_on() -> None:
    # D2
    r = simulate(3600.0, 100, -100, chain=Chain(), check_tone=False, monitor_ppm=-100)
    print_all("1 h full chain, Hear myself on (mic +100, cable -100, monitor -100)", r)
    assert_clean_run(r)
    assert r.monitor_underruns == 0 and r.monitor_overflows == 0 and r.monitor_frames_played > 0


@pytest.mark.slow
def test_one_hour_real_pattern_everything_on() -> None:
    # D2
    r = replay(load_pattern(), 3600.0, monitor=True, echo=True)
    print_all("real pattern 1 h, monitor + echo", r)
    assert r.underruns == 0 and r.input_overflows == 0 and r.output_overflows == 0 and r.discontinuities == 0
    assert_side_streams_clean(r)


# ------------------------------------------------------------------ the output side on its own


def make_path() -> AudioPath:
    return AudioPath(SAMPLE_RATE, 1, SAMPLE_RATE, 2, PassThrough())


def feed(path: AudioPath, seconds: float, value: float = 0.5) -> None:
    block = np.full((480, 1), value, np.float32)
    for _ in range(int(seconds * 100)):
        path.input_callback(block, 480, None, None)
        path.process()


def test_output_waits_for_the_cushion_then_fades_in() -> None:
    path = make_path()
    out = np.ones((480, 2), np.float32)
    path.output_callback(out, 480, None, None)
    assert np.all(out == 0.0) and not path.primed
    feed(path, TARGET_FILL_S + 0.01)
    path.output_callback(out, 480, None, None)
    assert path.primed
    assert out[0, 0] == 0.0 and out[-1, 0] == pytest.approx(0.5, abs=1e-3)  # a ramp, not a step
    assert np.all(np.diff(out[:144, 0]) >= -1e-6)
    assert np.array_equal(out[:, 0], out[:, 1])  # mono to both cable channels


def test_underrun_on_a_block_boundary_still_fades() -> None:
    # The ring runs dry exactly at the end of a callback block: the fade must still come before the silence.
    path = make_path()
    path.out_ring.write(np.full(START_FRAMES, 0.5, np.float32))  # exactly the start cushion, then nothing
    out = np.zeros((480, 2), np.float32)
    played = []
    for _ in range(START_FRAMES // 480 + 2):
        path.output_callback(out, 480, None, None)
        played.append(out[:, 0].copy())
    audio = np.concatenate(played)
    last = np.nonzero(audio)[0][-1]
    assert audio[last] < 0.05
    assert path.counters.underruns == 1
    assert path.frames_played == START_FRAMES


def test_underrun_fades_out_counts_and_never_skips() -> None:
    path = make_path()
    feed(path, TARGET_FILL_S + 0.01)
    out = np.zeros((480, 2), np.float32)
    played = []
    for _ in range(int(TARGET_FILL_S * 100) + 3):  # drain past the end of the audio
        path.output_callback(out, 480, None, None)
        played.append(out[:, 0].copy())
    assert path.counters.underruns == 1
    audio = np.concatenate(played)
    nonzero = np.nonzero(audio)[0]
    last = nonzero[-1]
    # The last samples fade down to silence instead of stopping dead.
    assert audio[last] < 0.05 and np.max(audio[last - 400 : last - 200]) == pytest.approx(0.5, abs=1e-3)
    # Every sample that went in came out (after the resampler's own small delay): nothing was skipped.
    assert path.frames_played == path.out_ring.total_written


# ------------------------------------------------------------------ the real device pattern (J05 Step 2b)


def print_run(label: str, r: SimResult) -> None:
    print(
        f"\n{label}: underruns={r.underruns} input_overflows={r.input_overflows} "
        f"output_overflows={r.output_overflows} discontinuities={r.discontinuities} "
        f"correction={r.min_correction * 1e6:+.0f}..{r.max_correction * 1e6:+.0f} ppm "
        f"fill={r.min_fill_ms:.1f}..{r.max_fill_ms:.1f} ms target={r.target_ms:.0f} ms "
        f"buffering_changes={r.buffering_changes} min_spare={r.min_spare_ms:.1f} ms"
    )


def test_real_pattern_fixture_holds_only_times_and_frame_counts() -> None:
    with np.load(FIXTURE) as d:
        assert set(d.files) == {
            "seconds",
            "in_rate",
            "out_rate",
            "mic_time",
            "mic_frames",
            "cable_time",
            "cable_frames",
            "worker_start",
            "worker_end",
            "chain_cost",
            "source_underruns",
        }
        assert all(d[k].dtype.kind in "fiu" for k in d.files)  # numbers only: no names, no audio
    assert FIXTURE.stat().st_size < 1_000_000


HEADSET = FIXTURE.parent / "headset_pattern.npz"


class _Stamp:
    currentTime = 0.0


def replay_headset(key: str) -> AudioPath:
    """Replay one real 16 kHz Bluetooth headset soak (engine-obs r1, B1) through the real AudioPath in its real event
    order: the mic's 160-frame callbacks, the cable's, and a worker step at each recorded worker step."""
    with np.load(HEADSET) as d:
        mic, cable, work, (in_rate, out_rate) = d[f"{key}_mic"], d[f"{key}_cable"], d[f"{key}_work"], d[f"{key}_rates"]
    path = AudioPath(int(in_rate), 1, int(out_rate), 2, PassThrough())
    events = sorted([(r[0], 0, i) for i, r in enumerate(mic)] + [(r[0], 1, i) for i, r in enumerate(cable)])
    events = sorted(events + [(t, 2, 0) for t in work])
    block = np.zeros((int(mic[:, 2].max()), 1), np.float32)
    out = np.zeros((int(cable[:, 2].max()), 2), np.float32)
    phase = 0
    t_mic, t_cable = _Stamp(), _Stamp()
    for _, kind, i in events:
        if kind == 0:
            n = int(mic[i, 2])
            block[:n, 0] = 0.1 * np.sin(2 * np.pi * 440 * (phase + np.arange(n)) / in_rate)
            phase += n
            t_mic.currentTime = float(mic[i, 1])
            path.input_callback(block[:n], n, t_mic, None)
        elif kind == 1:
            n = int(cable[i, 2])
            t_cable.currentTime = float(cable[i, 1])
            path.output_callback(out[:n], n, t_cable, None)
        else:
            path.process()
    return path


@pytest.mark.parametrize("key", ["a", "b"])
def test_a_headset_that_loses_audio_never_runs_the_cable_dry(key: str) -> None:
    # engine-obs r1, B1: the 16 kHz Bluetooth hands-free mic loses 2-15 ms of audio at a time upstream (28 ms in its
    # first 35 s), and its clock is trusted only after about 150 s. One 10 ms raise per underrun, built up at 100 ppm,
    # let the cable run dry: trace "a" (the reviewer's real run) underran at 39.7 s, "b" (the builder's) at 113.2 s.
    # Late audio now adds its lateness at once, before the ring runs dry (libobs/obs-audio.c:361-390).
    path = replay_headset(key)
    c = path.counters
    print(
        f"\nheadset {key}: underruns {c.underruns}, buffering changes {path.cable.buffering_changes}, "
        f"target {path.cable.target_ms:.0f} ms"
    )
    assert c.underruns == 0 and c.output_overflows == 0
    assert path.cable.target_ms <= 100.0  # it took a few ticks, far from the 150 ms cap


def test_headset_fixture_holds_only_times_and_frame_counts() -> None:
    with np.load(HEADSET) as d:
        assert {k.split("_", 1)[1] for k in d.files} == {"mic", "cable", "work", "rates"}
        assert all(d[k].dtype.kind in "fiu" for k in d.files)  # numbers only: no names, no audio
    assert HEADSET.stat().st_size < 1_000_000


@pytest.mark.parametrize(("mic_ppm", "cable_ppm"), [(0, 0), (100, -100), (-100, 100)])
def test_the_replay_runs_the_clock_gap_it_names(mic_ppm: int, cable_ppm: int) -> None:
    # F2 (J05 review r2): one 60 s loop of the fixture holds 480 more mic frames than cable frames, so every replay
    # ran 167 ppm off what it named, and R2's "cable 200 ppm faster" case drained at only +33 ppm.
    p = load_pattern()
    frames = {0: 0, 1: 0}
    for t, kind, n in replay_events(p, mic_ppm, cable_ppm):
        if t > 3600.0:
            break
        if kind in frames:
            frames[kind] += n
    gap = (frames[1] / frames[0] - 1.0) * 1e6
    named = ((1 + cable_ppm * 1e-6) / (1 + mic_ppm * 1e-6) - 1.0) * 1e6
    print(f"\nreplay mic {mic_ppm:+} / cable {cable_ppm:+} ppm: cable/mic gap {gap:+.1f} ppm (named {named:+.1f})")
    assert abs(loop_bias_ppm(p)) > 100.0  # the fixture's own loop really is off: the correction is needed
    assert abs(gap - named) < 5.0  # one callback at a loop edge is 2.8 ppm of an hour


@pytest.mark.parametrize("gap_ppm", [100, 200, -100, -200])
def test_real_pattern_with_the_clocks_apart_has_no_underruns(gap_ppm: int) -> None:
    # F1 (J05 review r2): the bursty mic's rate is trusted only after about 100 s, so until then only the fill trim
    # holds the cushion. With the cable 100 ppm or more faster than the mic, a 40 ms start drained to an underrun at
    # 46-57 s. A 60 ms start (TARGET_FILL_S) has the room. gap_ppm: how much faster the cable asks than the mic gives.
    r = replay(load_pattern(), 300.0, mic_ppm=-gap_ppm / 2, cable_ppm=gap_ppm / 2)
    print_run(f"real pattern 300 s, cable {gap_ppm:+} ppm against the mic", r)
    assert r.underruns == 0
    assert r.input_overflows == 0 and r.output_overflows == 0
    assert r.discontinuities == 0


@pytest.mark.parametrize("chain", [None, "full"], ids=["pass-through", "full-chain"])
def test_real_device_pattern_has_no_underruns(chain: str | None) -> None:
    # R1: the callback pattern measured on the development PC (Focusrite mic, VB-CABLE, WASAPI), replayed for 5 min.
    # The measured run had 0 underruns on the Step 1 code too (the reviewer's 8 came from the mic host buffer, see
    # devices.INPUT_LATENCY_S), so this pins the pattern; it is not a red-before test (accepted deviation from plan
    # Step 2b). It replays the pattern with its loop bias cancelled, so the clocks run exactly 0 ppm apart (review r2,
    # F2). The least spare is what makes it bite: the replay holds 22.3 ms beyond a callback's frames on the 60 ms
    # start; a 40 ms start drops that to 7.4 ms, a 20 ms one to 3.8 ms. At 0 ppm it cannot see a missing low-water
    # raise or feed-forward (22.3 and 22.8 ms): R5, the catch-up and large-block tests, and R3, the drift minutes and
    # the clocks-apart replays catch those.
    r = replay(load_pattern(), 300.0, chain=Chain() if chain else None)
    print_run(f"real pattern 300 s ({chain or 'pass-through'})", r)
    assert r.min_spare_ms >= 15.0
    assert r.underruns == 0
    assert r.input_overflows == 0
    assert r.output_overflows == 0
    assert r.discontinuities == 0


@pytest.mark.slow
@pytest.mark.parametrize(("mic_ppm", "cable_ppm"), [(100, -100), (-100, 100)])
def test_real_pattern_one_hour(mic_ppm: int, cable_ppm: int) -> None:
    # R2: the same pattern for an hour, with the clocks another +-100 ppm apart on top of their measured rates.
    r = replay(load_pattern(), 3600.0, mic_ppm, cable_ppm)
    print_run(f"real pattern 1 h mic {mic_ppm:+} ppm / cable {cable_ppm:+} ppm", r)
    # The loop bias is cancelled (review r2, F2), so the cable really asks 200 ppm more, or less, than the mic gives.
    # Cable fast, the cushion drains until the mic's rate is trusted (about 100 s): one raise to 70 ms; a 40 ms start
    # underran here. The real mic's 30-40 ms bursts make the fill swing about 35 ms above the target, so the fill bound
    # is the target plus 50 ms rather than the even simulations' 70 ms (M5, J05 review r1).
    assert r.target_ms <= 70.0
    assert_clean_run(r, max_fill_ms=r.target_ms + 50.0)


# ------------------------------------------------------------------ the controller (J05 Step 2c)


def test_correction_settles_on_the_clock_offset() -> None:
    # R3: bursty devices with clocks 200 ppm apart. The correction must settle on that offset, not swing.
    r = simulate(240.0, 100, -100, bursty=True)
    offset = ((1 - 100e-6) / (1 + 100e-6) - 1) * 1e6  # -200 ppm: the cable is slower, so the worker makes fewer frames
    window = np.array(r.corrections_ppm[60:])
    print(f"\ncorrection after 60 s: {window.min():+.0f}..{window.max():+.0f} ppm, mean {window.mean():+.0f}")
    # A buffering raise is added at once (engine-obs r1, B1), so there is no catch-up stretch to leave out.
    assert len(window) >= 30
    assert window.max() - window.min() < 100.0
    assert abs(window.mean() - offset) < 50.0
    # The 60 ms start covers this model's worst start too (a 27 ms mic burst on a 15 ms worker hiccup), which took a
    # 40 ms start to one underrun in its first 10 s (J05 review r2, F1).
    assert r.underruns == 0, r.underrun_times


def drive_sink(
    sink: Sink,
    seconds: float,
    late_at: tuple[float, ...] = (),
    late_s: float = 0.025,
    cable_frames: int = 480,
) -> list[float]:
    """The worker pushes 480 samples every 10 ms and the cable takes ``cable_frames`` every 10 ms, 5 ms later.

    From each time in ``late_at`` the worker's blocks arrive ``late_s`` late, all at once. Returns the target in ms
    after every event.
    """
    events: list[tuple[float, int]] = []
    for k in range(1, int(seconds * 100)):
        t = k * 0.01
        push = next((at + late_s for at in late_at if at <= t < at + late_s), t)
        events += [(push, 0), (t + 0.005, 1)]
    events.sort()
    block = np.full(480, 0.5, np.float32)
    out = np.zeros((cable_frames, 2), np.float32)
    clock = SimTime()
    targets = []
    for t, kind in events:
        if kind == 0:
            sink.push(block)
        else:
            clock.currentTime = t
            sink.callback(out, cable_frames, clock, None)
        targets.append(sink.target_ms)
    return targets


def test_late_audio_adds_buffering_in_10_ms_steps(caplog: pytest.LogCaptureFixture) -> None:
    # R5 (libobs/obs-audio.c:361-422, add_audio_buffering): audio late by the cushion less 24 ms (36 ms on the 60 ms
    # start) leaves about 7 ms to spare at the last cable read it misses, under the 10 ms low-water mark: add 10 ms.
    caplog.set_level(logging.INFO)
    sink = Sink(SAMPLE_RATE, 2)
    targets = drive_sink(sink, 8.0, late_at=(3.0,), late_s=TARGET_FILL_S - 0.024)
    assert sink.counters.underruns == 0
    assert sink.target_ms == pytest.approx(START_MS + 10)
    assert f"cable: adding 10 ms of audio buffering, total now {START_MS + 10:.0f} ms" in caplog.text
    assert all(b >= a for a, b in pairwise(targets))  # never lowered


def test_buffering_is_capped_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    sink = Sink(SAMPLE_RATE, 2)
    # Each 200 ms gap is one underrun, one 10 ms step: nine take the 60 ms start to the 150 ms cap, the rest find it.
    targets = drive_sink(sink, 22.0, late_at=tuple(3.0 + 1.5 * k for k in range(12)), late_s=0.2)
    assert sink.counters.underruns == 12
    assert sink.target_ms == pytest.approx(150.0)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warnings] == ["cable: maximum audio buffering reached (150 ms)"]
    assert all(b >= a for a, b in pairwise(targets))


def test_an_underrun_raises_the_target_by_its_deficit_once(caplog: pytest.LogCaptureFixture) -> None:
    # The underrun's measured deficit below the safety margin, rounded up to 10 ms ticks (obs-audio.c:378-379), once:
    # the same underrun must not count again later (J05 review r1, F3: it used to add 20 ms more, 70 ms in all).
    caplog.set_level(logging.INFO)
    sink = Sink(SAMPLE_RATE, 2)
    targets = drive_sink(sink, 8.0, late_at=(5.3,), late_s=TARGET_FILL_S + 0.02)
    assert sink.counters.underruns == 1
    added = sink.target_ms - START_MS
    assert targets[0] == pytest.approx(START_MS) and added > 0
    assert added == pytest.approx(round(added / 10) * 10)  # whole ticks
    assert f"cable: an underrun: adding {added:.0f} ms of audio buffering" in caplog.text
    assert caplog.text.count("adding") == 1


def test_playback_starts_once_the_target_is_queued_even_if_the_mic_pauses() -> None:
    # M3 (J05 review r1): six 10 ms blocks are the 60 ms target, but the resampler holds back ~19 frames of the
    # first one. Playback waited for the next block; on the real mic that came 30 ms later in a burst, so it started
    # with 80 ms queued and the trim spent minutes bringing it down.
    sink = Sink(SAMPLE_RATE, 2)
    block = np.full(480, 0.5, np.float32)
    out = np.zeros((480, 2), np.float32)
    blocks = round(TARGET_FILL_S * 100)
    for _ in range(blocks):
        sink.push(block)
    queued = sink.ring.available()
    assert queued < blocks * 480  # the resampler's start-up delay
    sink.callback(out, 480, None, None)  # the mic pauses here: no fifth block
    assert sink.primed and np.any(out), f"waited with {queued} frames queued"


def test_a_raise_is_added_at_once_and_faded() -> None:
    # engine-obs r1, B1 (libobs/obs-audio.c:381): the buffering is added in one go, not built up by playing slower,
    # so the pitch never moves (J05 F5: a fast build swung a real run's correction to +2000 ppm). The block before
    # the added silence fades out and the one after fades in (DESIGN: never hard silence).
    sink = Sink(SAMPLE_RATE, 2)
    block = np.full(480, 0.5, np.float32)
    out = np.zeros((480, 2), np.float32)
    played: list[np.ndarray] = []
    corrections: list[float] = []
    for k in range(1, 3000):
        late = 2000 <= k < 2000 + round((TARGET_FILL_S - 0.024) * 100)  # 36 ms of the worker's blocks arrive late
        if not late:
            for _ in range(
                1 + (k == 2000 + round((TARGET_FILL_S - 0.024) * 100)) * round((TARGET_FILL_S - 0.024) * 100)
            ):
                sink.push(block)
                corrections.append(sink.controller.correction)
        sink.callback(out, 480, None, None)
        played.append(out[:, 0].copy())
    y = np.concatenate(played)
    assert sink.counters.underruns == 0 and sink.target_ms == pytest.approx(START_MS + 10)
    assert sink.buffering_changes == 1
    assert (max(corrections) - min(corrections)) * 1e6 < 1.0  # equal clocks: no correction, before or after
    steady = y[5 * SAMPLE_RATE : 15 * SAMPLE_RATE]
    assert np.all(steady == pytest.approx(0.5, abs=1e-3))
    gap = np.flatnonzero(y[16 * SAMPLE_RATE :] < 1e-4) + 16 * SAMPLE_RATE
    assert len(gap) >= 0.009 * SAMPLE_RATE  # about 10 ms of silence was added, once
    fade = int(FADE_S * SAMPLE_RATE)
    assert np.max(np.abs(np.diff(y[16 * SAMPLE_RATE :]))) <= 1.5 * 0.5 / fade  # faded both ways, no hard edge


@pytest.mark.parametrize(("cable_frames", "target_ms"), [(441, 60.0), (2400, 63.0)])
def test_playback_starts_at_the_target(cable_frames: int, target_ms: float) -> None:
    # R6: the first audio plays once the ring holds the target: 60 ms, or the largest callback + fade + 10 ms.
    sink = Sink(SAMPLE_RATE, 2)
    out = np.zeros((cable_frames, 2), np.float32)
    block = np.full(480, 0.5, np.float32)
    before: list[int] = []
    for _ in range(200):
        sink.push(block)
        before.append(sink.ring.available())
        sink.callback(out, cable_frames, None, None)
        if np.any(out):
            break
    target = target_ms * SAMPLE_RATE / 1000
    assert sink.target_ms == pytest.approx(target_ms, abs=0.1)
    start = target - PRIME_SLACK_S * SAMPLE_RATE  # less the resampler's start-up delay (M3)
    assert before[-1] >= start and before[-2] < start


def test_callbacks_only_copy_and_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    # R7: no chain, no resampler inside the device callbacks; each one writes one timestamp entry.
    path = AudioPath(SAMPLE_RATE, 2, SAMPLE_RATE, 2, PassThrough())
    inside = [False]

    def guard(real):
        def call(*args, **kwargs):
            assert not inside[0], "DSP inside a device callback"
            return real(*args, **kwargs)

        return call

    class GuardedResampler:  # samplerate's Resampler.process is read-only, so wrap the object
        def __init__(self, real) -> None:
            self.process = guard(real.process)

    monkeypatch.setattr(path.chain, "process", guard(path.chain.process))
    monkeypatch.setattr(path.cable, "_resampler", GuardedResampler(path.cable._resampler))

    def in_callback(real):
        def call(*args):
            inside[0] = True
            try:
                real(*args)
            finally:
                inside[0] = False

        return call

    input_callback, output_callback = in_callback(path.input_callback), in_callback(path.output_callback)
    block = np.full((480, 2), 0.25, np.float32)
    out = np.zeros((441, 2), np.float32)
    clock = SimTime()
    for k in range(1, 101):
        clock.currentTime = k * 0.01
        input_callback(block, 480, clock, None)
        path.process()
        output_callback(out, 441, clock, None)
    assert path.blocks_processed == 100 and path.primed
    assert path.mic_stamps.count == 100
    assert path.cable.stamps.count == 100
    assert path.mic_stamps.time[99] == pytest.approx(1.0)
    assert path.cable.stamps.frames[99] == 99 * 441  # frames read before that callback
