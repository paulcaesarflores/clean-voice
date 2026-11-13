"""Run the real clean-up chain offline on a WAV, in the worker's 480-sample blocks (audio-check skill).

    python tools/harness.py in.wav out.wav
    python tools/harness.py in.wav out.wav --settings preset=radio loudness=80 strength=70
    python tools/harness.py in.wav out.wav --settings denoise=0 style=0 fine_tune=1 \\
        --filters noise_gate_filter,basic_eq_filter,compressor_filter:ratio=4,gain_filter:db=3
    python tools/harness.py in.wav out.wav --reference far.wav --echo     # echo removal first, far.wav as the speakers
    python tools/harness.py in.wav out.wav --bench                        # time per stage and in total
    python tools/harness.py in.wav out.wav --bench --moving-filters       # ...with everything retuned every block

``out.wav`` is 48 kHz 24-bit mono, like the backup recording. Settings keys: preset (natural, warm, crisp, radio, deep,
flat), strength and loudness (0-100), bass, mid and treble (0-100, 50 = no change), and the switches cleaning,
denoise, style and fine_tune (on/off, 1/0). ``--filters``: Fine-tune's list, top first, as OBS filter ids at OBS's
defaults, each optionally followed by ``:key=value`` overrides with OBS's keys (VST filters need the app). With --echo
the reference is fed to AEC3 sample for sample with the input (an offline far end, no timing).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from speech import RATE, read_wav

from clean_voice import fine_tune, obs_filters
from clean_voice import settings as st
from clean_voice.dsp import Chain
from clean_voice.echo import EchoCanceller
from clean_voice.fine_tune import FilterSpec
from clean_voice.wav import WavWriter

BLOCK = 480
F32 = npt.NDArray[np.float32]
SWITCHES = ("cleaning", "denoise", "style", "fine_tune")


def _on(value: str) -> bool:
    return value.strip().lower() in ("1", "on", "true", "yes")


def parse_filters(text: str) -> tuple[FilterSpec, ...]:
    """``kind[:key=value...],kind...`` as Fine-tune's list, top first, named by OBS's rule."""
    out: list[FilterSpec] = []
    for item in text.split(","):
        if not item.strip():
            continue
        kind, *pairs = (part.strip() for part in item.split(":"))
        if kind not in obs_filters.CLASSES:
            raise SystemExit(f"unknown filter {kind!r} (built-in OBS filters: {', '.join(obs_filters.CLASSES)})")
        keys = {p.key for p in fine_tune.KINDS[kind].props if p.is_setting}
        settings: dict[str, object] = {}
        for pair in pairs:
            key, _, value = pair.partition("=")
            if key not in keys:
                raise SystemExit(f"{kind} has no setting {key!r} (it has: {', '.join(sorted(keys)) or 'none'})")
            try:
                settings[key] = float(value)
            except ValueError:
                settings[key] = value  # a list's value, e.g. detector=peak
        out.append(fine_tune.new_filter(kind, out, settings=settings))
    return tuple(out)


def parse_settings(pairs: list[str], filters: str | None = None) -> st.Settings:
    """A ``Settings`` from key=value words and a ``--filters`` list."""
    s = st.Settings()
    for pair in pairs:
        key, _, value = pair.partition("=")
        key = key.strip().lower()
        if key == "preset":
            s = st.replace(s, bands=st.bands_for(value.strip().lower()))
        elif key in ("strength", "loudness"):
            s = st.replace(s, **{key: float(value)})
        elif key in st.TONE_BANDS:
            s = st.with_tone(s, key, float(value))
        elif key in SWITCHES:
            s = st.replace(s, **{key: _on(value)})
        else:
            raise SystemExit(f"unknown setting {key!r}")
    if filters:
        s = st.replace(s, filters=parse_filters(filters))
    return s


class _Timed:
    """Wraps a stage's ``process`` and keeps each call's duration."""

    def __init__(self, process: Any, times: list[float]) -> None:
        self._process = process
        self.times = times

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        y = self._process(*args, **kwargs)
        self.times.append(time.perf_counter() - t0)
        return y


def _retuned(base: FilterSpec, now: FilterSpec, k: int) -> FilterSpec:
    """``now`` with one of its numbers moved 5 % of its range off the ``base`` value, or back, by ``k``."""
    numeric = [p for p in fine_tune.KINDS[base.kind].props if p.widget in (fine_tune.SLIDER, fine_tune.INT)]
    if not numeric:
        return now
    p = numeric[(k // 2) % len(numeric)]
    v = float(base.settings[p.key])  # pyright: ignore[reportArgumentType]
    delta = 0.05 * (p.hi - p.lo)
    moved = v + delta if v + delta <= p.hi else v - delta
    return dataclasses.replace(now, settings={**now.settings, p.key: moved if k % 2 else v})


def moving_filters(s: st.Settings, base: tuple[FilterSpec, ...], k: int, presets: list[str]) -> st.Settings:
    """Block ``k`` of --moving-filters: a different style every block, one filter retuned and one eye toggled every
    block, and the list rotated by one every 10th block (each a section fade out and back in)."""
    s = st.replace(s, bands=st.bands_for(presets[k % len(presets)]))
    if not s.filters:
        return s
    order = list(s.filters)
    if k % 10 == 0:
        order = order[1:] + order[:1]
    by_uid = {f.uid: f for f in base}
    j = k % len(order)
    order[j] = _retuned(by_uid[order[j].uid], order[j], k // len(order))
    e = (k + 1) % len(order)
    order[e] = dataclasses.replace(order[e], enabled=not order[e].enabled)
    return st.replace(s, filters=tuple(order))


def run(
    x: F32,
    settings: st.Settings | None = None,
    far: F32 | None = None,
    bench: dict[str, list[float]] | None = None,
    block: int = BLOCK,
    moving: bool = False,
) -> F32:
    """The chain (and echo removal when ``far`` is given) over ``x``, block by block; returns the output.
    ``moving``: --moving-filters, the style EQ, the Fine-tune filters and their eyes changed every block and the list
    reordered every 10th block (their worst case)."""
    cell = st.SettingsCell(settings if settings is not None else st.Settings())
    chain = Chain(cell, block=block)  # the worker's whole 480-sample frames: no extra queue
    canceller = None
    if far is not None:
        canceller = EchoCanceller()
        far = np.concatenate((far, np.zeros(max(0, len(x) - len(far)), np.float32)))[: len(x)]
    total: list[float] = []
    aec_times: list[float] = []
    if bench is not None:
        bench.clear()
        if canceller is not None:
            bench["echo"] = aec_times
        for name in chain.STAGES:
            stage = getattr(chain, name)
            bench[name] = []
            stage.process = _Timed(stage.process, bench[name])
            if name == "fine_tune":
                for spec in cell.value.filters:  # each filter's own time, inside fine_tune's
                    inst = chain.fine_tune.instances.get(spec.uid)
                    if inst is not None:
                        bench[f" {spec.name}"] = []
                        setattr(inst, "process", _Timed(inst.process, bench[f" {spec.name}"]))  # noqa: B010
        bench["total"] = total
    presets = list(st.PRESETS)
    base = cell.value.filters
    # The worker only ever runs whole blocks: pad the file's last, partial one with silence, and trim it off again.
    length = len(x)
    pad = -length % block
    x = np.concatenate((x, np.zeros(pad, np.float32)))
    if far is not None:
        far = np.concatenate((far, np.zeros(pad, np.float32)))
    out = np.zeros(len(x), np.float32)
    for k, i in enumerate(range(0, len(x), block)):
        if moving:
            cell.value = moving_filters(cell.value, base, k, presets)
        b = x[i : i + block]
        t0 = time.perf_counter()
        if canceller is not None and far is not None and len(b) == block:
            ta = time.perf_counter()
            b = canceller.process(b, far[i : i + block])
            aec_times.append(time.perf_counter() - ta)
        out[i : i + len(b)] = chain.process(b)
        total.append(time.perf_counter() - t0)  # echo removal included
    return out[:length]


def print_bench(bench: dict[str, list[float]]) -> None:
    n = len(bench["total"])
    print(f"{'stage':20} {'mean ms':>8} {'p50 ms':>8} {'p99 ms':>8} {'max ms':>8}   (per 10 ms block, {n} blocks)")
    for name, times in bench.items():
        t = np.array(times) * 1000.0
        if len(t):
            print(f"{name:20} {t.mean():8.3f} {np.median(t):8.3f} {np.percentile(t, 99):8.3f} {t.max():8.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness", description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--settings", nargs="*", default=None, metavar="KEY=VALUE")
    parser.add_argument("--filters", metavar="KIND[:KEY=VALUE...],...", help="Fine-tune's filters, top first")
    parser.add_argument("--reference", type=Path, help="what the speakers played (the far end), for --echo")
    parser.add_argument("--echo", action="store_true", help="run echo removal first, with --reference")
    parser.add_argument("--bench", action="store_true", help="print the time per stage and in total")
    parser.add_argument(
        "--moving-filters",
        action="store_true",
        help="change the style, a filter's setting and an eye every block, and reorder every 10th (the worst case)",
    )
    args = parser.parse_args(argv)
    if args.echo and args.reference is None:
        parser.error("--echo needs --reference")
    for wav_path in (args.input, args.reference if args.echo else None):
        if wav_path is not None:
            with wave.open(str(wav_path), "rb") as w:
                if w.getframerate() != RATE:
                    parser.error(
                        f"{wav_path} is {w.getframerate()} Hz; the chain runs at {RATE} Hz (resample it first)"
                    )
    x = read_wav(args.input).astype(np.float32)
    far = read_wav(args.reference).astype(np.float32) if args.echo and args.reference else None
    settings = parse_settings(args.settings or [], args.filters) if args.settings or args.filters else None
    bench: dict[str, list[float]] | None = {} if args.bench else None
    y = run(x, settings, far, bench, moving=args.moving_filters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = WavWriter(args.output, RATE)
    writer.write(y)
    writer.close()
    print(f"{args.output}: {len(y) / RATE:.2f} s")
    if bench is not None:
        print_bench(bench)
    return 0


if __name__ == "__main__":
    sys.exit(main())
