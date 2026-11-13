"""The chain on real speech: licensed fixtures, every word coming through, noise between them, levels and clipping."""

from __future__ import annotations

import platform
import re
import sys
import wave
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from speech import FAN, HUM, NOISES, RATE, active_rms_db, mix, pauses, speech, word_drops_db, word_spans

from clean_voice.dsp import Chain
from clean_voice.settings import Settings, SettingsCell

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

DATA = Path(__file__).resolve().parent / "data"
README = DATA / "README.md"


# ------------------------------------------------------------------ F3: the fixtures


@pytest.mark.parametrize(("name", "min_seconds"), [("speech.wav", 30.0), ("noise.wav", 10.0)])
def test_fixture_format(name: str, min_seconds: float) -> None:
    path = DATA / name
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 48_000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() / 48_000 >= min_seconds
    assert path.stat().st_size < 3_000_000


@pytest.mark.parametrize("name", ["speech.wav", "noise.wav"])
def test_fixture_licence_is_recorded(name: str) -> None:
    text = README.read_text(encoding="utf-8")
    section = text.split(f"## {name}", 1)[1].split("\n## ", 1)[0]
    licence = re.search(r"^- Licence: (.+)$", section, re.MULTILINE)
    assert licence, f"no licence line for {name}"
    assert re.search(r"Public Domain|CC0|CC BY 4\.0", licence[1]), licence[1]
    assert "CC BY-SA" not in licence[1] and "NC" not in licence[1]
    assert re.search(r"^- Source: https://\S+$", section, re.MULTILINE), f"no source URL for {name}"
    assert re.search(r"^- Author: .+$", section, re.MULTILINE), f"no author for {name}"
    assert re.search(r"^- Conversion: .+$", section, re.MULTILINE), f"no conversion for {name}"


@dataclass
class Run:
    x: np.ndarray  # the input
    clean: np.ndarray  # the clean speech at the same level
    out: np.ndarray
    spans: list[tuple[int, int]]


@cache
def run_case(level: float, snr: float | None, strength: float, noise: str = HUM, seed: int = 1) -> Run:
    x, clean, spans = mix(level, snr, seed=seed, noise_name=noise)
    chain = Chain(SettingsCell(Settings(strength=strength)))
    out = np.concatenate([chain.process(x[i : i + 480]) for i in range(0, len(x), 480)])
    return Run(x, clean, out, spans)


LEVELS = [(-26.0, "normal"), (-38.0, "quiet")]
SNRS = [None, 20.0, 10.0]
STRENGTHS = [0.0, 70.0, 100.0]
CASES = [
    (lv, snr, s, n)
    for n in NOISES
    for lv, _ in LEVELS
    for snr in SNRS
    for s in STRENGTHS
    if not (snr is None and n != HUM)  # the clean cases have no noise: run them once
]


def case_id(case: tuple[float, float | None, float, str]) -> str:
    lv, snr, s, n = case
    where = "clean" if snr is None else f"{'hum' if n == HUM else 'fan'}{snr:.0f}"
    return f"{'quiet' if lv < -30 else 'normal'}-{where}-s{s:.0f}"


# No word may come out more than this far below the typical word (tests/speech.py word_drops_db). Measured without a
# gate: at most 2.2 dB on clean speech and at 10-20 dB SNR, 5.6 dB at 0-3 dB SNR (RNNoise thins the first word most).
# The removed floor-relative gate failed it at 0 dB SNR on the hum: the word at 11.54 s came out 7.6 dB (normal level)
# and 10.9 dB (quiet) down, because the gate never opened in it.
WORD_DROP_MAX_DB = 7.0


def assert_every_word_comes_through(r: Run) -> None:
    drops = word_drops_db(r.out, r.clean, r.spans, Chain.LATENCY)
    worst = int(np.argmin(drops))
    assert drops[worst] >= -WORD_DROP_MAX_DB, (
        f"the word at {r.spans[worst][0] / RATE:.2f} s came out {-drops[worst]:.1f} dB below the typical word"
    )


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_every_word_comes_through_on_real_speech(case: tuple[float, float | None, float, str]) -> None:
    # G1, restated for the chain without a gate (user decision, 2026-11-11): nothing turns a word down on its own.
    r = run_case(*case)
    assert_every_word_comes_through(r)


NOISY_SNRS = [3.0, 0.0]  # a poor mic in a noisy room: the voice barely above the noise
NOISY_CASES = [(-38.0, snr, s, n) for n in NOISES for snr in NOISY_SNRS for s in STRENGTHS]


@pytest.mark.parametrize("case", NOISY_CASES, ids=case_id)
def test_voice_survives_a_noisy_room(case: tuple[float, float | None, float, str]) -> None:
    # Bug report (real Realtek laptop mic, words a few dB above its own noise): the voice must never be wiped out. At
    # 0-3 dB SNR every word still comes through. Red with the removed gate (hum, 0 dB, strengths 70 and 100).
    r = run_case(*case)
    assert_every_word_comes_through(r)


# "Clearly reduced" (J06) in the pauses, where a guest hears the noise. Two figures, kept apart: (a) noise alone at
# strength 70 is turned down >= 10 dB (hum) and >= 12 dB (fan), test_dsp NOISE_ALONE_MIN_DB (S2); (b) here, the whole
# pauses of the 20 dB SNR mix at the defaults, pinned as a floor at what the chain reaches (J06 plan: "pinned at about
# 19 dB as a floor"). It measures 18.61 dB (hum) and 17.34 dB (fan) with the OBS-style compressor (OBS audit I-2); the
# floors sit 0.2 dB under, for other CPUs' float rounding: CI's Windows runner came up to 0.06 dB (hum) and 0.09 dB
# (fan) under this PC (run 43368578097 printed 18.6 / 17.3 against those pins), which leaves it >= 0.15 dB above these
# floors. They stay above strength 50's 17.92 / 16.75 dB, so a default that falls back to weaker cleaning fails here
# (S2's floors would pass it). 30 dB is out of reach: the speech recording alone leaves -61.8 dBFS in these pauses (its
# word tails and breaths).
PAUSE_FLOOR_DB = {HUM: 18.4, FAN: 17.1}


@pytest.mark.parametrize("noise", NOISES)
def test_background_noise_is_clearly_reduced_on_real_speech(noise: str) -> None:
    # N1 as restated by the orchestrator: (a) noise alone at strength 70 is turned down >= 10 dB (tonal hum) and >= 12 dB
    # (broadband fan) -- see test_dsp S2; (b) here: the pauses of the 20 dB SNR mix, defaults, pinned at the reachable
    # floor (PAUSE_FLOOR_DB), and audio_check finds 0 faults.
    r = run_case(-26.0, 20.0, 70.0, noise)
    lat = Chain.LATENCY
    quiet = pauses(r.spans, len(r.x))
    xin = np.concatenate([r.x[s:e] for s, e in quiet]).astype(np.float64)
    xout = np.concatenate([r.out[s + lat : e + lat] for s, e in quiet]).astype(np.float64)
    reduction = 10 * np.log10(np.mean(xin**2) / np.mean(xout**2))
    assert reduction >= PAUSE_FLOOR_DB[noise], f"{reduction:.1f} dB"
    import audio_check

    assert audio_check.analyze(r.out.astype(np.float64), RATE, r.clean.astype(np.float64)) == []


# At 10 dB SNR, RNNoise itself thins speech for 50-90 ms in one place per case (the same dips with and without the
# removed gate). audio_check reports these as "gate dip"; they are pinned here, from the strength they appear at:
# (noise, level) -> (from strength, time s, longest ms). The fan's 16.89 s dip at the normal level crosses the 10 dB
# line at strength 100 since the per-sample compressor (OBS audit I-2); at the quiet level it did from strength 70.
# At the normal level and strength 70 it sits on the line: 9.91 dB down on x86-64, where RNNoise 0.2 runs its network
# with the SSE kernels (src/vec_avx.h: activations quantised to unsigned 8 bits, _mm_rcp_ps), and just past 10 dB on
# Apple silicon, where it runs the NEON ones (src/vec_neon.h: signed 8 bits, the coarser vrecpeq_f32). Same -O2 build,
# no fast-math; the same dip at the same place and length (CI run 43368578097, macOS arm64), so arm64 pins it from 70.
ARM64 = platform.machine().lower() in ("arm64", "aarch64")
RNNOISE_DIPS = {
    (HUM, -26.0): (70.0, 1.12, 60),
    (HUM, -38.0): (70.0, 1.09, 90),
    (FAN, -38.0): (70.0, 16.89, 50),
    (FAN, -26.0): (70.0 if ARM64 else 100.0, 16.89, 50),
}


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_audio_check_finds_no_faults(case: tuple[float, float | None, float, str]) -> None:
    lv, snr, strength, noise = case
    r = run_case(lv, snr, strength, noise)
    import audio_check

    faults = audio_check.analyze(r.out.astype(np.float64), RATE, r.clean.astype(np.float64))
    pinned = RNNOISE_DIPS.get((noise, lv)) if snr == 10.0 else None
    if pinned is None or strength < pinned[0]:
        assert faults == [], [str(f) for f in faults]
        return
    _, at, longest_ms = pinned
    assert len(faults) == 1 and faults[0].kind == "gate dip", [str(f) for f in faults]
    assert faults[0].time == pytest.approx(at, abs=0.02)
    assert float(faults[0].detail.split()[0]) <= longest_ms


def test_voice_level_is_sensible() -> None:
    # N2: a typical USB mic (-26 dBFS active RMS) comes out near the original's -24 dBFS target (dsp.rs:234-235).
    r = run_case(-26.0, None, 70.0)
    lat = Chain.LATENCY
    out_db = active_rms_db(r.out, [(s + lat, e + lat) for s, e in r.spans])
    assert out_db == pytest.approx(-24.0, abs=3.0)
    chain = Chain()
    limited = 0
    for i in range(0, len(r.x), 480):
        chain.process(r.x[i : i + 480])
        limited += int(np.count_nonzero(chain.limiter.last_gain_db < -1.0))
    assert limited / len(r.x) < 0.02


def test_word_spans_of_the_speech_fixture_are_pinned() -> None:
    # The span rule (tests/speech.py) decides where G1 looks; a changed rule or fixture shows up here first.
    spans = word_spans(speech())
    assert len(spans) == 14
    assert len(pauses(spans, len(speech()))) == 7  # the lead-in and six pauses of 0.6 s or more
