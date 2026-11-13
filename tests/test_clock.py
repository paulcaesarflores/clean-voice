"""R4: the clock estimator follows OBS's timestamp handling (libobs/obs-source.c, source_output_audio_data).

Jitter within 70 ms of the fitted clock is smoothed over, a packet 70 ms to 2 s off is logged and left out, and a jump
over 2 s resets the timing.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from clean_voice.audio_path import ClockEstimator

RATE = 48_000
TRUE_RATE = RATE * (1 + 100e-6)
PACKET = 480


def feed(clock: ClockEstimator, seconds: float, seed: int = 0, start_frames: int = 0, t0: float = 100.0) -> int:
    """Packets of 480 frames at TRUE_RATE, each up to 30 ms late; returns the frame count reached."""
    rng = np.random.default_rng(seed)
    frames = start_frames
    for _ in range(int(seconds * TRUE_RATE / PACKET)):
        late = rng.uniform(0.0, 0.030) if rng.random() < 0.5 else 0.0
        clock.add(t0 + (frames - start_frames) / TRUE_RATE + late, frames)
        frames += PACKET
    return frames


def ppm(clock: ClockEstimator) -> float:
    return (clock.rate / TRUE_RATE - 1.0) * 1e6


@pytest.mark.parametrize("seed", range(5))
def test_rate_within_10_ppm_after_30_s_of_late_jitter(seed: int) -> None:
    clock = ClockEstimator(RATE, "mic")
    feed(clock, 30.0, seed)
    assert clock.ready
    assert abs(ppm(clock)) < 10.0


def test_not_ready_before_10_s() -> None:
    clock = ClockEstimator(RATE, "mic")
    feed(clock, 9.0)
    assert not clock.ready
    assert clock.rate == RATE  # nominal until ready
    feed(clock, 21.0, start_frames=int(9.0 * TRUE_RATE / PACKET) * PACKET, t0=109.0)
    assert clock.ready  # once its fit is sure (CLOCK_CONFIDENT_PPM), well before 30 s with this jitter


STARTUP = Path(__file__).resolve().parent / "data" / "real_startup.npz"


def startup_patterns() -> list[tuple[str, np.ndarray, np.ndarray]]:
    with np.load(STARTUP) as d:
        return [
            (f"p{i} {name}", d[f"p{i}_{name}_time"], d[f"p{i}_{name}_frames"].astype(np.float64))
            for i in range(int(d["patterns"]))
            for name in ("mic", "cable")
        ]


@pytest.mark.parametrize(
    ("label", "times", "frames"), startup_patterns(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_real_stream_starts_never_give_a_wrong_rate(label: str, times: np.ndarray, frames: np.ndarray) -> None:
    # F4 (J05 review r1): the stamps of real streams from their very first callback (Focusrite mic and VB-CABLE,
    # tools/audio_soak.py --export-startup). Taken at 10 s regardless, the old fit was off by up to 1433 ppm here,
    # and the feed-forward swung the correction to -1722 ppm. Every rate the estimator reports must be close.
    before = np.concatenate(([0.0], np.cumsum(frames)[:-1]))
    late = times > 5.0  # the truth: least squares over everything after the first 5 s
    tc = times[late] - times[late].mean()
    truth = float(np.dot(tc, before[late] - before[late].mean()) / np.dot(tc, tc))
    clock = ClockEstimator(RATE, label)
    worst, next_check = 0.0, 1.0
    for t, f in zip(times.tolist(), before.tolist(), strict=True):
        clock.add(100.0 + t, f)
        if t >= next_check:
            next_check += 1.0
            if clock.ready:
                worst = max(worst, abs(clock.rate / truth - 1.0) * 1e6)
    print(f"\n{label}: truth {truth:.2f} Hz, ready={clock.ready}, worst reported error {worst:.0f} ppm")
    assert worst < 50.0
    if times[-1] >= 140.0:
        # Sure by the end: the bursty mic needs its 120 s window (CLOCK_WINDOW_S), VB-CABLE about 20 s.
        assert clock.ready


def test_a_packet_200_ms_late_is_logged_and_left_out(caplog: pytest.LogCaptureFixture) -> None:
    clock = ClockEstimator(RATE, "mic")
    frames = feed(clock, 20.0)
    before = clock.rate
    caplog.set_level(logging.INFO)
    clock.add(100.0 + frames / TRUE_RATE + 0.200, frames)
    assert clock.outliers == 1
    assert clock.resets == 0
    assert "mic: a packet came 200 ms off its clock" in caplog.text
    assert clock.rate == before


def test_a_3_s_jump_resets_the_estimator(caplog: pytest.LogCaptureFixture) -> None:
    clock = ClockEstimator(RATE, "cable")
    frames = feed(clock, 20.0)
    assert clock.ready
    caplog.set_level(logging.INFO)
    clock.add(100.0 + frames / TRUE_RATE + 3.0, frames)
    assert clock.resets == 1
    assert not clock.ready
    assert "cable: timing jumped by 3000 ms, restarting the rate estimate" in caplog.text
    # It learns the clock again from the new timeline.
    feed(clock, 30.0, start_frames=frames, t0=100.0 + frames / TRUE_RATE + 3.0)
    assert clock.ready and abs(ppm(clock)) < 10.0
