"""The real speech and noise fixtures (tests/data/README.md): loading, mixing at a level and SNR, and word spans."""

from __future__ import annotations

import wave
from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

DATA = Path(__file__).resolve().parent / "data"
RATE = 48_000
FRAME = RATE // 100  # 10 ms
SPAN_BELOW_PEAK_DB = 40.0  # a frame is speech when its RMS is within this of the loudest frame
SPAN_MERGE_S = 0.150  # gaps shorter than this inside a span are merged (stop consonants)
SPAN_MIN_S = 0.080  # spans shorter than this are dropped
SEAM_S = 0.050  # the crossfade where the looped noise wraps around

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
Spans = list[tuple[int, int]]


def read_wav(path: Path) -> F64:
    """A 16-, 24- or 32-bit PCM WAV as float64 in -1..1, channels averaged."""
    with wave.open(str(path), "rb") as w:
        width, channels, n = w.getsampwidth(), w.getnchannels(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float64) / float(1 << 23)
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float64) / float(1 << 31)
    else:
        raise ValueError(f"{path}: {width * 8}-bit WAV not supported")
    return x.reshape(-1, channels).mean(axis=1) if channels > 1 else x


@cache
def _fixture(name: str) -> F64:
    return read_wav(DATA / name)


def speech() -> F64:
    return _fixture("speech.wav").copy()


HUM = "noise.wav"  # tonal: air-conditioner hum
FAN = "fan.wav"  # broadband: an electric fan
NOISES = (HUM, FAN)


def noise(name: str = HUM) -> F64:
    return _fixture(name).copy()


def frame_rms_db(x: npt.NDArray[np.floating], frame: int = FRAME) -> F64:
    n = len(x) // frame
    blocks = np.asarray(x[: n * frame], np.float64).reshape(n, frame)
    return 10.0 * np.log10(np.mean(blocks**2, axis=1) + 1e-20)


def word_spans(clean: npt.NDArray[np.floating]) -> Spans:
    """(start, end) sample indices of the words in a clean recording.

    A 10 ms frame is speech when its RMS is above the loudest frame's less 40 dB; gaps under 150 ms inside a span are
    merged (stop consonants) and spans under 80 ms are dropped.
    """
    db = frame_rms_db(clean)
    active = db > db.max() - SPAN_BELOW_PEAK_DB
    edges = np.flatnonzero(np.diff(np.concatenate(([0], active.astype(np.int8), [0]))))
    raw = list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))
    merged: list[list[int]] = []
    for s, e in raw:
        if merged and (s - merged[-1][1]) * FRAME < SPAN_MERGE_S * RATE:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s * FRAME, e * FRAME) for s, e in merged if (e - s) * FRAME >= SPAN_MIN_S * RATE]


def pauses(spans: Spans, length: int, min_s: float = 0.6) -> Spans:
    """The noise-only stretches: the lead-in, the gaps between spans and the tail, each at least ``min_s`` long."""
    bounds = [0, *[v for span in spans for v in span], length]
    gaps = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds), 2)]
    return [(s, e) for s, e in gaps if e - s >= min_s * RATE]


def active_rms_db(x: npt.NDArray[np.floating], spans: Spans) -> float:
    inside = np.concatenate([np.asarray(x[s:e], np.float64) for s, e in spans])
    return float(10.0 * np.log10(np.mean(inside**2) + 1e-20))


WORD_LOUD_DB = 20.0  # a word's level is taken over its frames within this of the word's own loudest frame


def word_drops_db(out: npt.NDArray[np.floating], clean: npt.NDArray[np.floating], spans: Spans, latency: int) -> F64:
    """How far each word comes out below the typical word, in dB (0 for the median word, negative when quieter).

    Per word: the output's level over the word's loud frames (``latency`` samples later) less the clean speech's level
    over the same frames. Taking each word against the median word removes the chain's overall gain (makeup,
    compression), so what is left is a word turned down on its own: a gate cutting it, or the denoiser removing it.
    """
    c = frame_rms_db(clean)
    o = frame_rms_db(np.asarray(out, np.float64)[latency:])
    n = min(len(c), len(o))
    levels = []
    for s, e in spans:
        f0, f1 = s // FRAME, min(e // FRAME, n)
        cw = c[f0:f1]
        loud = cw > cw.max() - WORD_LOUD_DB
        levels.append(10 * np.log10(np.mean(10 ** (o[f0:f1][loud] / 10)) / np.mean(10 ** (cw[loud] / 10))))
    d = np.array(levels)
    return d - np.median(d)


def looped_noise(length: int, seed: int, name: str = HUM) -> F64:
    """``length`` samples of a noise fixture from a seeded random start, looped with an equal-power crossfade."""
    src = noise(name)
    seam = int(SEAM_S * RATE)
    fade = np.sin(np.linspace(0.0, np.pi / 2, seam))
    loop = src[:-seam].copy()
    loop[:seam] = src[:seam] * fade + src[-seam:] * fade[::-1]  # the tail flows into the head
    start = int(np.random.default_rng(seed).integers(0, len(loop)))
    reps = (start + length) // len(loop) + 1
    return np.tile(loop, reps)[start : start + length]


def mix(
    level_db: float = -26.0, snr_db: float | None = None, seed: int = 0, noise_name: str = HUM
) -> tuple[F32, F32, Spans]:
    """The speech fixture at ``level_db`` active RMS, plus real noise ``snr_db`` below it (None: clean).

    Returns (mixed input, the clean speech at the same level, word spans). Everything is float32, as the engine sees it.
    """
    clean = speech()
    spans = word_spans(clean)
    clean *= 10.0 ** ((level_db - active_rms_db(clean, spans)) / 20.0)
    mixed = clean.copy()
    if snr_db is not None:
        n = looped_noise(len(clean), seed, noise_name)
        n_rms_db = 10.0 * np.log10(np.mean(n**2))
        mixed += n * 10.0 ** ((level_db - snr_db - n_rms_db) / 20.0)
    return mixed.astype(np.float32), clean.astype(np.float32), spans
