"""Convert a downloaded recording into a test fixture: 48 kHz, mono, 16-bit WAV (tests/data/README.md).

    python tools/convert_fixture.py <source> <dest.wav> --start 42.5 --seconds 30.5

Needs the ``tools`` dependency group (soundfile, to decode MP3), never the app. Channels are averaged, the stretch is
cut at the source rate, resampled exactly by 160/147 (44.1 -> 48 kHz) or by the rational ratio of the rates, and
rounded to 16 bits. Nothing else is done to the audio: no gain, no filtering, no fades.
"""

from __future__ import annotations

import argparse
import math
import sys
import wave
from pathlib import Path

import numpy as np
from scipy import signal

RATE = 48_000


def convert(source: Path, dest: Path, start: float, seconds: float) -> None:
    # The optional `tools` group, which CI does not install.
    import soundfile as sf  # pyright: ignore[reportMissingImports]

    info = sf.info(str(source))
    rate = int(info.samplerate)
    x, _ = sf.read(str(source), start=round(start * rate), frames=round(seconds * rate), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if rate != RATE:
        g = math.gcd(RATE, rate)
        x = signal.resample_poly(x, RATE // g, rate // g)
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    print(f"{dest}: {len(pcm) / RATE:.2f} s, {dest.stat().st_size} bytes, from {source.name} ({rate} Hz) at {start} s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="convert_fixture", description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the source")
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args(argv)
    convert(args.source, args.dest, args.start, args.seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
