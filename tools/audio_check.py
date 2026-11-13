"""Find dropouts and processing faults in a WAV: each fault leaves a signature in the waveform (audio-check skill).

    python tools/audio_check.py out.wav                       # zero runs, discontinuities, long gaps
    python tools/audio_check.py out.wav --speech clean.wav    # also gate dips, against the clean speech

Prints every fault with its time, type and likely cause, and exits 1 when there is any. The signatures and causes come
from docs/simple-clean-mic-audio-report.md:

- zero run: 2 ms or more of exact 0.0 inside active audio (an output underrun);
- discontinuity: one sample-to-sample jump far above the local level on both sides (a dropped chunk or a skip);
- gate dip: inside a word, the level falls 10 dB or more below the word's own level for 50-500 ms, then comes back
  (a word cut by the chain). Natural stop consonants look the same in the waveform alone, so this needs the clean
  speech (--speech) to compare against; without it the check is skipped and the summary says so;
- long gap: more than 100 ms of near-silence with a hard edge, a click, at one end (an engine or device restart).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from speech import FRAME, RATE, frame_rms_db, read_wav, word_spans

F64 = npt.NDArray[np.float64]

ZERO_RUN_S = 0.002
ACTIVE_DB = -60.0  # the audio around a zero run is this loud or more: it sits inside active audio
CONTEXT_S = 0.010
# A discontinuity jumps this many times the local sample-to-sample RMS, on both sides. At 10 the release burst of a
# plosive in the speech fixture (8.65 s: 11x) counted; a skipped chunk in voiced speech jumps 30x or more.
JUMP_RATIO = 20.0
JUMP_MIN = 0.01  # ...and by at least this much (-40 dBFS)
LOCAL_S = 0.005
QUIET_SIDE = 1e-4  # a side quieter than this is silence: an onset or a zero-run edge, not a discontinuity
DIP_DB = 10.0
DIP_MIN_S, DIP_MAX_S = 0.050, 0.500
DIP_EVAL_DB = 30.0  # gate dips are judged on frames within this of the clean speech's loudest frame
GAP_S = 0.100
GAP_LEVEL = 1e-4  # near-silence: every sample under -80 dBFS
CLICK_LEVEL = 0.01  # a hard edge: the audio within 0.5 ms of the gap reaches -40 dBFS at once
MAX_DELAY_S = 0.2  # the processed file may lag the clean speech by this much (the chain's latency)

CAUSES = {
    "zero run": "output underrun (buffer starvation)",
    "discontinuity": "a dropped chunk or a buffer skip",
    # The name is kept from J06, when a noise gate could do it. The chain has had no gate since 2026-11-11: the dips
    # left (10 dB SNR or less) are RNNoise thinning the word.
    "gate dip": "the denoiser thinned the word (or a gain stage cut it mid-word)",
    "long gap": "an engine or device restart",
}


@dataclass(frozen=True)
class Fault:
    time: float
    kind: str
    detail: str

    @property
    def cause(self) -> str:
        return CAUSES[self.kind]

    def __str__(self) -> str:
        return f"{self.time:9.3f} s  {self.kind:13}  {self.detail}; likely: {self.cause}"


def _runs(mask: npt.NDArray[np.bool_]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return edges[::2], edges[1::2]


def _rms_db(x: F64) -> float:
    return float(10 * np.log10(np.mean(x**2) + 1e-20)) if len(x) else -200.0


def long_gaps(x: F64, rate: int = RATE) -> list[tuple[int, int]]:
    starts, ends = _runs(np.abs(x) < GAP_LEVEL)
    edge = max(1, int(0.0005 * rate))
    found = []
    for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
        if e - s <= GAP_S * rate or s == 0 or e == len(x):
            continue
        before = np.max(np.abs(x[max(0, s - edge) : s]))
        after = np.max(np.abs(x[e : e + edge]))
        if max(before, after) >= CLICK_LEVEL:
            found.append((s, e))
    return found


def zero_runs(x: F64, rate: int = RATE, skip: list[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    nz = np.flatnonzero(x)
    if len(nz) == 0:
        return []
    starts, ends = _runs(x == 0.0)
    ctx = int(CONTEXT_S * rate)
    found = []
    for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
        if e - s < ZERO_RUN_S * rate or s <= nz[0] or e > nz[-1]:
            continue
        if any(a <= s and e <= b for a, b in skip or []):
            continue
        around = np.concatenate((x[max(0, s - ctx) : s], x[e : e + ctx]))
        if _rms_db(around) >= ACTIVE_DB:
            found.append((s, e))
    return found


def discontinuities(x: F64, rate: int = RATE) -> list[int]:
    dx = np.abs(np.diff(x))
    half = int(LOCAL_S * rate)
    c = np.concatenate(([0.0], np.cumsum(dx**2)))
    idx = np.arange(len(dx))
    # The sample-to-sample RMS just before and just after each step, two samples clear of it.
    lo0, lo1 = np.clip(idx - half, 0, len(dx)), np.clip(idx - 2, 0, len(dx))
    hi0, hi1 = np.clip(idx + 3, 0, len(dx)), np.clip(idx + 3 + half, 0, len(dx))
    left = np.sqrt((c[lo1] - c[lo0]) / np.maximum(lo1 - lo0, 1))
    right = np.sqrt((c[hi1] - c[hi0]) / np.maximum(hi1 - hi0, 1))
    hits = (dx > JUMP_MIN) & (dx > JUMP_RATIO * np.maximum(left, right)) & (np.minimum(left, right) > QUIET_SIDE)
    hits &= (lo1 - lo0 >= half - 2) & (hi1 - hi0 >= half)
    found: list[int] = []
    for i in np.flatnonzero(hits).tolist():
        if not found or i - found[-1] > CONTEXT_S * rate:
            found.append(i)
    return found


def _delay(out: F64, ref: F64, rate: int) -> int:
    n = min(len(out), len(ref))
    size = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(out[:n], size) * np.conj(np.fft.rfft(ref[:n], size)), size)
    return int(np.argmax(corr[: int(MAX_DELAY_S * rate)]))


def gate_dips(out: F64, ref: F64, rate: int = RATE) -> list[tuple[int, int]]:
    """Stretches inside the clean speech's words where the processed level fell DIP_DB below the word's own."""
    delay = _delay(out, ref, rate)
    aligned = out[delay:]
    n = min(len(aligned), len(ref)) // FRAME
    ref_db = frame_rms_db(ref[: n * FRAME])
    gain = frame_rms_db(aligned[: n * FRAME]) - ref_db
    judged = ref_db > ref_db.max() - DIP_EVAL_DB
    found = []
    for s, e in word_spans(ref[: n * FRAME]):
        fs, fe = s // FRAME, e // FRAME
        inside = judged[fs:fe]
        if inside.sum() < 3:
            continue
        # Only the judged frames count: one too quiet in the clean speech neither starts, ends nor breaks a dip.
        where = np.flatnonzero(inside)
        g = gain[fs:fe][where]
        low = g < float(np.median(g)) - DIP_DB
        for a, b in zip(*_runs(low), strict=True):
            if a == 0:
                continue  # the word starts low: an onset the denoiser eased in, not a fall inside speech
            if b == len(g):
                continue  # it never came back inside this word
            start, end = fs + int(where[a]), fs + int(where[b - 1]) + 1
            if DIP_MIN_S <= (end - start) * FRAME / rate <= DIP_MAX_S:
                found.append((start * FRAME + delay, end * FRAME + delay))
    return found


def analyze(x: F64, rate: int = RATE, reference: F64 | None = None) -> list[Fault]:
    faults: list[Fault] = []
    gaps = long_gaps(x, rate)
    for s, e in gaps:
        faults.append(Fault(s / rate, "long gap", f"{(e - s) * 1000 / rate:.0f} ms of near-silence with a hard edge"))
    zeros = zero_runs(x, rate, skip=gaps)
    for s, e in zeros:
        faults.append(Fault(s / rate, "zero run", f"{(e - s) * 1000 / rate:.1f} ms of exact zeros inside audio"))
    for i in discontinuities(x, rate):
        if any(a - 2 <= i <= b + 1 for a, b in gaps + zeros):
            continue  # the edge of a gap or zero run, already reported
        faults.append(Fault((i + 1) / rate, "discontinuity", f"a jump of {abs(x[i + 1] - x[i]):.3f}"))
    if reference is not None:
        for s, e in gate_dips(x, reference, rate):
            faults.append(
                Fault(s / rate, "gate dip", f"{(e - s) * 1000 / rate:.0f} ms at {DIP_DB:.0f} dB or more down")
            )
    return sorted(faults, key=lambda f: f.time)


def check_file(path: Path, speech_path: Path | None = None) -> list[Fault]:
    x = read_wav(path)
    ref = read_wav(speech_path) if speech_path is not None else None
    return analyze(x, RATE, ref)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audio_check", description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--speech", type=Path, help="the clean speech the file was made from, for the gate-dip check")
    args = parser.parse_args(argv)
    faults = check_file(args.wav, args.speech)
    for f in faults:
        print(f)
    checks = "zero runs, discontinuities, long gaps" + (", gate dips" if args.speech else "")
    skipped = "" if args.speech else " (gate dips not checked: give --speech)"
    print(f"{args.wav.name}: {len(faults)} fault{'s' if len(faults) != 1 else ''} [{checks}]{skipped}")
    return 1 if faults else 0


if __name__ == "__main__":
    sys.exit(main())
