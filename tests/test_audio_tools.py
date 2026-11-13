"""The offline tools: the fault analyzer finds each signature (F1) and the harness runs the real chain (F2)."""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from speech import RATE, read_wav, speech, word_spans

from clean_voice.wav import WavWriter

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))


def write(path: Path, x: np.ndarray) -> Path:
    w = WavWriter(path, RATE)
    w.write(np.asarray(x, np.float32))
    w.close()
    return path


def check(path: Path, reference: Path | None = None) -> tuple[int, str]:
    cmd = [sys.executable, str(TOOLS / "audio_check.py"), str(path)]
    if reference is not None:
        cmd += ["--speech", str(reference)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    return r.returncode, r.stdout + r.stderr


def faults(path: Path, reference: Path | None = None) -> list:
    import audio_check

    return audio_check.check_file(path, reference)


def inside_word(t: float) -> bool:
    return any(s <= t * RATE < e for s, e in word_spans(speech()))


# ------------------------------------------------------------------ F1: the analyzer


@pytest.fixture(scope="module")
def clean_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("check")
    x = speech()
    t = np.arange(3 * RATE) / RATE
    tone = 0.25 * np.sin(2 * np.pi * 440 * t)
    ramp = np.linspace(0, 1, 480)
    tone[:480] *= ramp
    tone[-480:] *= ramp[::-1]
    return {"speech": write(d / "speech.wav", x), "tone": write(d / "tone.wav", tone)}


def test_clean_speech_and_tone_have_no_faults(clean_files: dict[str, Path]) -> None:
    for name, path in clean_files.items():
        code, out = check(path)
        assert code == 0, f"{name}: {out}"
        assert "0 faults" in out
    code, out = check(clean_files["speech"], clean_files["speech"])  # against itself: no gate dips either
    assert code == 0, out


def test_zero_run_is_found(tmp_path: Path) -> None:
    x = speech()
    at = 7.30
    assert inside_word(at)
    s = int(at * RATE)
    x[s : s + int(0.003 * RATE)] = 0.0
    found = faults(write(tmp_path / "zero.wav", x))
    assert [f.kind for f in found] == ["zero run"]
    assert found[0].time == pytest.approx(at, abs=0.001)
    code, out = check(tmp_path / "zero.wav")
    assert code == 1 and "zero run" in out and "underrun" in out


def test_discontinuity_is_found(tmp_path: Path) -> None:
    x = speech()
    at = 13.0
    assert inside_word(at)
    s = int(at * RATE)
    # Drop the 1-10 ms chunk that makes the largest jump: a skipped buffer.
    k = 48 + int(np.argmax(np.abs(x[s + 48 : s + 480] - x[s - 1])))
    y = np.concatenate((x[:s], x[s + k :]))
    found = faults(write(tmp_path / "jump.wav", y))
    assert [f.kind for f in found] == ["discontinuity"]
    assert found[0].time == pytest.approx(at, abs=0.001)


def test_gate_dip_is_found_against_the_clean_speech(tmp_path: Path) -> None:
    x = speech()
    ref = write(tmp_path / "ref.wav", x)
    at, length = 7.50, 0.200
    assert inside_word(at) and inside_word(at + length)
    s, e = int(at * RATE), int((at + length) * RATE)
    gain = np.ones(len(x))
    gain[s:e] = 10 ** (-15 / 20)
    ramp = int(0.002 * RATE)
    gain[s : s + ramp] = np.linspace(1, gain[s + ramp], ramp)
    gain[e - ramp : e] = np.linspace(gain[s + ramp], 1, ramp)
    dipped = write(tmp_path / "dip.wav", x * gain)
    found = faults(dipped, ref)
    assert [f.kind for f in found] == ["gate dip"]
    assert found[0].time == pytest.approx(at, abs=0.02)
    assert faults(dipped) == []  # the waveform alone cannot tell a dip from a stop consonant
    _, out = check(dipped)
    assert "gate dips not checked" in out


def test_long_gap_is_found(tmp_path: Path) -> None:
    x = speech()
    at = 17.0
    assert inside_word(at)
    s = int(at * RATE)
    y = np.concatenate((x[:s], np.zeros(int(0.150 * RATE)), x[s:]))
    found = faults(write(tmp_path / "gap.wav", y))
    assert [f.kind for f in found] == ["long gap"]
    assert found[0].time == pytest.approx(at, abs=0.001)


# ------------------------------------------------------------------ F2: the harness


def test_harness_refuses_a_file_at_another_rate(tmp_path: Path) -> None:
    # J06 review r1, minor: a 44.1 kHz file used to be processed as if it were 48 kHz, without a word.
    src = tmp_path / "in44.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44_100)
        w.writeframes(np.zeros(4410, "<i2").tobytes())
    r = subprocess.run(
        [sys.executable, str(TOOLS / "harness.py"), str(src), str(tmp_path / "out.wav")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert r.returncode != 0 and "44100 Hz" in r.stderr
    assert not (tmp_path / "out.wav").exists()


def _cable_of(clean: np.ndarray, lag: int, step: float) -> np.ndarray:
    """What CABLE Output records of ``clean``: ``lag`` samples later, truncated toward zero to the cable's step."""
    cable = np.concatenate((np.zeros(lag), clean))[: len(clean)]
    return np.trunc(cable / step) * step


def test_the_soak_explains_a_zero_run_only_when_the_aligned_clean_voice_is_too_small_for_the_cable(capsys) -> None:
    # engine-obs r1, m6: VB-CABLE truncates toward zero, so it shows exact zeros wherever the clean voice is under one
    # step (not half: the old rule assumed rounding), and the drift resampler's interpolation can take a peak a little
    # over one step below it: a run is quiet when the lined-up clean voice peaks at 2 steps or less (orchestrator
    # decision; measured zeros at 1.00-1.75 steps). The old rule also searched a 250 ms window without lining the cable
    # up with the backup, so a quiet stretch elsewhere in it explained a real dropout.
    sys.path.insert(0, str(TOOLS))
    import audio_soak

    step = 2.0**-18  # VB-CABLE's step (measured)
    rng = np.random.default_rng(4)
    n = 12 * RATE
    clean = rng.standard_normal(n) * 40 * step  # a quiet voice, about 40 steps
    lag = int(0.037 * RATE)  # the cushion and the resampler: 37 ms from the backup to the cable
    runs = {
        "dropout": (int(4.0 * RATE), int(0.004 * RATE)),  # loud there: zeros on the cable are a real gap
        "under one step": (int(7.0 * RATE), int(0.003 * RATE)),  # between half a step and one: truncated to 0
        "quiet elsewhere": (int(10.0 * RATE), int(0.004 * RATE)),  # loud there; quiet 150 ms earlier in the backup
        "under two steps": (int(5.5 * RATE), int(0.002 * RATE)),  # peaks at 1.8 steps: the resampler's zeros
        "three steps": (int(8.5 * RATE), int(0.002 * RATE)),  # peaks at 3 steps: zeros there are a fault
    }
    s, m = runs["under one step"]
    clean[s - lag : s - lag + m] = 0.7 * step * np.sign(rng.standard_normal(m))
    for name, peak in (("under two steps", 1.8), ("three steps", 3.0)):
        s, m = runs[name]
        clean[s - lag : s - lag + m] = peak * step * np.sign(rng.standard_normal(m))
    s, m = runs["quiet elsewhere"]
    clean[s - lag - int(0.15 * RATE) : s - lag - int(0.15 * RATE) + m] = 0.2 * step
    cable = _cable_of(clean, lag, step)
    s, m = runs["dropout"]
    cable[s : s + m] = 0.0
    for name in ("quiet elsewhere", "under two steps", "three steps"):
        s, m = runs[name]
        cable[s : s + m] = 0.0
    t0 = 100.0  # the backup's first block and the listener's first sample, on the same clock
    stamps = np.array([[t0 + i / RATE, i] for i in range(0, n, 480)], np.float64)
    unmatched = audio_soak._zero_runs(cable.astype(np.float32), stamps, np.zeros(0), t0 - 5.0, (clean, t0))
    out = capsys.readouterr().out
    lines = {
        name: next(line for line in out.splitlines() if f"t={s / RATE + 5.0:8.3f}" in line)
        for name, (s, _) in runs.items()
    }
    assert "NO counted event" in lines["dropout"], out
    assert "quiet" in lines["under one step"], out
    assert "NO counted event" in lines["quiet elsewhere"], out
    assert "quiet" in lines["under two steps"], out
    assert "NO counted event" in lines["three steps"], out
    assert unmatched == 3, out
    # A buffering raise adds its silence to the cable at once (engine-obs r1, B1): a gap right after one is its own.
    s, _ = runs["dropout"]
    unmatched = audio_soak._zero_runs(
        cable.astype(np.float32), stamps, np.zeros(0), t0 - 5.0, (clean, t0), np.array([t0 + s / RATE - 0.05])
    )
    out = capsys.readouterr().out
    assert "matches a buffering raise" in next(line for line in out.splitlines() if f"t={s / RATE + 5.0:8.3f}" in line)
    assert unmatched == 2, out


Z2_TRACES = ROOT / "scratch" / "review" / "fine-tune-updates-r1" / "soak"
Z2_GAPS = {  # trace -> (s after on, ms) of the one real VB-CABLE start-up gap in it (fine-tune-updates r1, Z2)
    "2026-11-11_22-56-45": (1.455, 10.2),
    "2026-11-11_23-12-01": (2.064, 11.1),
    "2026-11-11_23-14-18": (0.688, 11.1),
    "2026-11-11_23-14-53": (0.481, 10.0),
}


def _zero_run_lines(trace: Path, capsys) -> tuple[int, dict[str, float], list[str]]:
    sys.path.insert(0, str(TOOLS))
    import audio_soak

    d = np.load(trace)
    work, cable = d["work"], d["cable"]
    clean = (audio_soak._read_wav24(Path(str(d["clean_wav"]))), float(work[work[:, 2] > 0][0, 0]))
    und = np.diff(np.concatenate(([0.0], cable[:, 5])))
    info: dict[str, float] = {}
    raises = cable[1:, 0][np.diff(cable[:, 6]) > 0]
    unmatched = audio_soak._zero_runs(
        d["listen_audio"], d["listen_stamps"], cable[und > 0, 0], float(d["meta_t_on"]), clean, raises, cable, info
    )
    return unmatched, info, [line for line in capsys.readouterr().out.splitlines() if line.startswith("  t=")]


@pytest.mark.skipif(not Z2_TRACES.exists(), reason="the reviewer's recorded soak traces are local only (scratch/)")
@pytest.mark.parametrize(
    "trace", sorted(Z2_TRACES.glob("2026-*.npz")) if Z2_TRACES.exists() else [], ids=lambda p: p.stem
)
def test_the_soak_labels_vbcables_start_gap_in_the_recorded_traces(trace: Path, capsys) -> None:
    # fine-tune-updates r1, Z2: a 10-11 ms exact-zero run 0.5-2 s after start in 4 of 12 soaks, inside audio, nothing
    # counted. VB-CABLE (or the Windows engine) does it with no Clean Voice too (docs/evidence/Z2-vbcable-start-gap.md):
    # each is labelled, at most once per stream start; every other run in the 12 traces is explained.
    unmatched, info, lines = _zero_run_lines(trace, capsys)
    gaps = [line for line in lines if "VB-CABLE inserted gap" in line]
    assert unmatched == 0, lines
    if trace.stem in Z2_GAPS:
        at, ms = Z2_GAPS[trace.stem]
        assert len(gaps) == 1 and f"t={at:8.3f} s  {ms:.1f} ms" in gaps[0], lines
        assert info["vbcable_gaps"] == 1
    else:
        assert gaps == [] and info["vbcable_gaps"] == 0, lines


def _gap_case(
    gaps: list[tuple[float, int]],
    late_after: float | None = None,
    seconds: float = 8.0,
    holes: list[tuple[float, int]] = (),  # type: ignore[assignment]
    quiet: list[tuple[float, int]] = (),  # type: ignore[assignment]
):
    """A synthetic soak: a quiet voice on the cable 100 ms behind the backup, VB-CABLE gaps of (time, samples) inserted
    (the audio after each comes that much later), cable callbacks every 10 ms from 0.1 s before the listener's first
    sample. ``late_after``: from then on the cable callbacks run 20 ms late (render starvation: the hand-over count
    steps). ``holes``: zeros written over the audio, nothing after them delayed (a hole Clean Voice wrote itself).
    ``quiet``: pauses of (cable time, samples) where the clean voice is under one cable step (VB-CABLE truncates them
    to exact zeros)."""
    step = 2.0**-18
    rng = np.random.default_rng(9)
    n = int(seconds * RATE)
    clean = rng.standard_normal(n) * 40 * step
    lag = 4800
    for at, length in quiet:
        i = int(at * RATE) - lag
        clean[i : i + length] = 0.4 * step
    cable = np.concatenate((np.zeros(lag), clean))[:n]
    for at, length in sorted(gaps, reverse=True):
        i = int(at * RATE)
        cable = np.concatenate((cable[:i], np.zeros(length), cable[i:]))[:n]
    for at, length in holes:
        i = int(at * RATE)
        cable[i : i + length] = 0.0
    cable = np.trunc(cable / step) * step
    t0 = 100.0
    stamps = np.array([[t0 + i / RATE, i] for i in range(0, n, 480)], np.float64)
    times = t0 - 0.1 + np.arange(int(seconds * 100) + 10) * 0.01
    if late_after is not None:
        times[times >= t0 + late_after] += 0.020
    rows = np.zeros((len(times), 7))
    rows[:, 0], rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 6] = times, times, 480, 2880, 2880
    return cable.astype(np.float32), stamps, rows, (clean, t0), t0


@pytest.mark.parametrize(
    ("name", "gaps", "late_after", "labelled", "unmatched"),
    [
        ("one period, early", [(1.5, 528)], None, 1, 0),
        ("a second one", [(1.5, 528), (3.0, 528)], None, 1, 1),
        ("30 ms", [(1.5, 1440)], None, 0, 1),
        ("at 6 s", [(6.0, 528)], None, 0, 1),
        ("hand-over count drops", [(1.5, 528)], 1.5, 0, 1),
    ],
)
def test_the_soak_labels_only_vbcables_start_gap(name, gaps, late_after, labelled, unmatched, capsys) -> None:
    # The orchestrator's rule: within 5 s of the stream start, one period (9-13 ms), the hand-over count constant within
    # 3 ms, the audio after it later by its length within 2 samples where alignment exists; at most one per start.
    sys.path.insert(0, str(TOOLS))
    import audio_soak

    cable, stamps, rows, clean, t0 = _gap_case(gaps, late_after)
    info: dict[str, float] = {}
    got = audio_soak._zero_runs(cable, stamps, np.zeros(0), t0, clean, np.zeros(0), rows, info)
    out = capsys.readouterr().out
    assert info["vbcable_gaps"] == labelled, out
    assert got == unmatched, out


def test_a_hole_written_over_the_voice_is_not_a_vbcable_gap(capsys) -> None:
    # fine-tune-updates r2, T1: what tells VB-CABLE's insertion from a hole Clean Voice wrote itself is the audio after
    # it: an insertion delays it by the run's length, a hole written over the voice leaves it in place. A 10 ms hole
    # 1.5 s after the stream start (inside the 5 s window, one period long, the hand-over count unchanged) must fail.
    sys.path.insert(0, str(TOOLS))
    import audio_soak

    cable, stamps, rows, clean, t0 = _gap_case([], holes=[(1.5, 480)])
    info: dict[str, float] = {}
    got = audio_soak._zero_runs(cable, stamps, np.zeros(0), t0, clean, np.zeros(0), rows, info)
    out = capsys.readouterr().out
    assert got == 1 and info["vbcable_gaps"] == 0, out


def test_every_zero_run_is_judged_not_only_the_first_40(capsys) -> None:
    # fine-tune-updates r2, S1: the judge looked at the first 40 runs only, so a real hole after 40 quiet pauses (the
    # clean voice under one cable step, which VB-CABLE truncates to exact zeros) was neither printed nor counted.
    sys.path.insert(0, str(TOOLS))
    import audio_soak

    pauses = [(5.0 + 0.5 * k, 144) for k in range(45)]
    cable, stamps, rows, clean, t0 = _gap_case([], seconds=60.0, holes=[(40.0, 480)], quiet=pauses)
    info: dict[str, float] = {}
    got = audio_soak._zero_runs(cable, stamps, np.zeros(0), t0, clean, np.zeros(0), rows, info)
    out = capsys.readouterr().out
    assert got == 1, out  # the hole at 40 s, the 46th run
    assert "exact-zero runs of 1 ms or more: 46" in out


def test_the_soak_judges_only_what_came_after_the_cable_stream_started(capsys) -> None:
    # A reader opening CABLE Output can get VB-CABLE's stale buffer from whatever played last (a real 30 s soak opened
    # with 14 ms of an earlier -40 dBFS test tone, then silence, before Clean Voice's stream existed). That is not
    # Clean Voice's audio, so it must not make Clean Voice's own leading silence look like a hole.
    sys.path.insert(0, str(TOOLS))
    import audio_soak

    cable, stamps, rows, clean, t0 = _gap_case([], seconds=6.0)
    stale = np.zeros(int(1.2 * RATE), np.float32)
    stale[:672] = 0.01  # the stale tail
    stale[int(0.5 * RATE) : int(0.5 * RATE) + 200] = 0.01
    joined = np.concatenate((stale, cable))
    stamps = np.array([[t0 - 1.2 + i / RATE, i] for i in range(0, len(joined), 480)], np.float64)
    # The cable stream's first callback (rows) is at t0 - 0.1, after the stale audio.
    info: dict[str, float] = {}
    got = audio_soak._zero_runs(joined, stamps, np.zeros(0), t0, clean, np.zeros(0), rows, info)
    out = capsys.readouterr().out
    assert "not Clean Voice's" in out and got == 0, out


def test_the_drift_mean_is_information_when_the_mic_clock_was_never_trusted(capsys) -> None:
    # Orchestrator decision (fine-tune-updates r1): in a soak too short for the mic clock estimator (it stays at the
    # nominal rate until its fit is sure: 30 s soaks), the correction's mean cannot follow the rate gap yet, so that
    # check is information, with the reason printed. With the estimate trusted (every 300 s soak) it stays hard.
    sys.path.insert(0, str(TOOLS))
    import audio_soak

    run = {
        "underruns": 0,
        "input_overflows": 0,
        "output_overflows": 0,
        "corr_spread_ppm": 40.0,
        "corr_mean_ppm": -110.0,
        "gap_ppm": 15.0,
        "min_spare_ms": 20.0,
        "zero_runs_unmatched": 0,
    }
    assert audio_soak._verdict({**run, "mic_clock_trusted": 0.0}) == []
    out = capsys.readouterr().out
    assert "information only" in out and "never trusted" in out and "PASS" in out
    assert audio_soak._verdict({**run, "mic_clock_trusted": 1.0}) == [
        "correction mean from 10 s within 100 ppm of the rate gap"
    ]


def test_the_window_soak_runs_with_the_backup_on(monkeypatch) -> None:
    # engine-obs r2, n4: --with-ui saved prefs without the backup, which is off by default, so the window soak had no
    # clean voice to tell a quiet zero run from a fault (every run there was "NO counted event").
    import argparse

    sys.path.insert(0, str(TOOLS))
    import audio_soak

    from clean_voice import app, prefs
    from clean_voice.settings import Settings

    seen: dict[str, bool] = {}

    class Opened(Exception):
        pass

    def window(engine: object, path: Path) -> None:
        seen["recording"] = prefs.load(path).recording
        raise Opened

    monkeypatch.setattr(app, "MainWindow", window)
    engine = type("Engine", (), {"settings": lambda self: Settings()})()
    with pytest.raises(Opened):
        audio_soak._run_with_ui(engine, argparse.Namespace(mic=None), {})  # type: ignore[arg-type]
    assert seen == {"recording": True}


def test_the_soak_reports_the_spread_of_a_mic_losing_audio_as_information(capsys) -> None:
    # Orchestrator decision (engine-obs r1): when the mic is measured losing audio (its estimated rate more than 300 ppm
    # off nominal), the correction follows it, so its spread is information, not a failure, and the reason is printed.
    # Underruns, overflows and unexplained zero runs stay hard checks.
    sys.path.insert(0, str(TOOLS))
    import audio_soak

    clean_run = {
        "underruns": 0,
        "input_overflows": 0,
        "output_overflows": 0,
        "corr_spread_ppm": 623.0,
        "corr_mean_ppm": 209.0,
        "gap_ppm": 248.0,
        "min_spare_ms": 7.8,
        "zero_runs_unmatched": 0,
    }
    assert audio_soak._verdict({**clean_run, "mic_rate_off_ppm": -503.0}) == []
    out = capsys.readouterr().out
    assert "information only" in out and "-503 ppm" in out and "Step 2d verdict (judged from 10 s): PASS" in out
    steady_mic = audio_soak._verdict({**clean_run, "mic_rate_off_ppm": -40.0})  # a steady mic: the spread still fails
    assert steady_mic == ["correction spread from 10 s under 300 ppm"]
    losing_with_underrun = audio_soak._verdict({**clean_run, "mic_rate_off_ppm": -503.0, "underruns": 1})
    assert losing_with_underrun == ["underruns 0"]
    losing_with_zero_run = audio_soak._verdict({**clean_run, "mic_rate_off_ppm": -503.0, "zero_runs_unmatched": 1})
    assert losing_with_zero_run == ["every exact-zero run on CABLE Output explained"]


def test_harness_takes_a_wav_of_any_length(tmp_path: Path) -> None:
    # engine-obs r1, F3: the whole-frame chain (m-1) refused the file's last, partial block, so any recording whose
    # length is not a multiple of 480 samples (an Audacity export, a trimmed clip) crashed the tool.
    x = speech()[: 3 * RATE + 100]
    src = tmp_path / "in.wav"
    w = WavWriter(src, RATE)
    w.write(x.astype(np.float32))
    w.close()
    out = tmp_path / "out.wav"
    r = subprocess.run(
        [sys.executable, str(TOOLS / "harness.py"), str(src), str(out)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert r.returncode == 0, r.stderr[-800:]
    assert len(read_wav(out)) == len(x)


def test_harness_runs_the_real_chain(tmp_path: Path) -> None:
    from clean_voice import settings as st
    from clean_voice.dsp import Chain

    src = write(tmp_path / "in.wav", speech()[: 5 * RATE])
    out = tmp_path / "out.wav"
    r = subprocess.run(
        [sys.executable, str(TOOLS / "harness.py"), str(src), str(out), "--settings", "preset=flat", "--bench"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    x = read_wav(src).astype(np.float32)
    chain = Chain(st.SettingsCell(st.Settings(bands=st.bands_for("flat"))), block=480)
    expected = np.concatenate([chain.process(x[i : i + 480]) for i in range(0, len(x), 480)])
    got = read_wav(out)
    assert len(got) == len(expected)
    assert np.max(np.abs(got - expected)) <= 1.0 / (1 << 23)  # one 24-bit step
    rows = {line.split()[0] for line in r.stdout.splitlines() if line.strip()}
    for stage in ("rumble", "denoise", "eq", "compressor", "volume", "limiter", "total"):
        assert stage in rows, r.stdout
