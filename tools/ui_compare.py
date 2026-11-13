"""Pair the original's baseline screenshots with the latest ui-snapshot round, side by side with captions.

    .venv/Scripts/python tools/ui_compare.py [--baseline docs/ui/baseline] [--round docs/ui/rounds/<N>]
                                             [--out docs/ui/compare]

``--round`` defaults to the highest numbered folder in docs/ui/rounds/. Each pair in ``PAIRS`` whose two PNGs exist
becomes ``<out>/<baseline name>.png``; a pair with a missing side is listed and skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUTTER = 24  # px between the two screenshots
CAPTION = 36  # px band above them for the captions
BACKGROUND = "#ffffff"
INK = "#202124"

# Baseline screenshot (tools/ui_baseline.py) -> (Clean Voice screen, the round state that shows the same thing). Since
# J10 the original's Voice, Cleaning and Settings tabs are all in the one window (Steps 1-5 and the footer), and its
# Fine-tune tab is the Fine-tune filter window.
PAIRS = {
    "voice": ("One window", "running"),
    "cleaning": ("One window", "echo-on"),
    "fine-tune": ("Fine-tune filters", "filters-basic-eq"),
    "settings": ("One window", "voice-to-speakers"),
    "cable-missing": ("One window", "cable-missing"),
    "error": ("One window", "error"),
    "cleaning-off": ("One window", "power-off"),
    "mic-menu-open": ("One window", "mic-menu-open"),
}


def _app() -> object:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # The offscreen platform has no font list of its own: point it at the system fonts.
    for fonts in (Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts", Path("/System/Library/Fonts")):
        if fonts.is_dir():
            os.environ.setdefault("QT_QPA_FONTDIR", str(fonts))
            break
    from PySide6.QtGui import QGuiApplication

    return QGuiApplication.instance() or QGuiApplication([sys.argv[0]])


def side_by_side(left: Path, right: Path, left_caption: str, right_caption: str, out: Path) -> None:
    """Write ``left`` and ``right`` next to each other, each under its caption, to ``out``."""
    _app()
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter

    images = [QImage(str(left)), QImage(str(right))]
    for path, image in zip((left, right), images, strict=True):
        if image.isNull():
            raise ValueError(f"cannot read {path}")
    lw, rw = images[0].width(), images[1].width()
    height = CAPTION + max(i.height() for i in images)
    canvas = QImage(lw + GUTTER + rw, height, QImage.Format.Format_RGB32)
    canvas.fill(QColor(BACKGROUND))
    painter = QPainter(canvas)
    try:
        font = QFont()
        font.setFamilies(["Segoe UI", "Helvetica Neue", "Arial"])
        font.setPixelSize(16)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(INK))
        for x, width, image, caption in ((0, lw, images[0], left_caption), (lw + GUTTER, rw, images[1], right_caption)):
            painter.drawText(
                QRect(x + 4, 0, width - 4, CAPTION), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, caption
            )
            painter.drawImage(x, CAPTION, image)
    finally:
        painter.end()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not canvas.save(str(out)):
        raise OSError(f"could not write {out}")


def latest_round(rounds: Path) -> Path | None:
    numbered = [p for p in rounds.iterdir() if p.is_dir() and p.name.isdigit()] if rounds.is_dir() else []
    return max(numbered, key=lambda p: int(p.name), default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, default=ROOT / "docs" / "ui" / "baseline")
    parser.add_argument("--round", type=Path, help="round folder (default: the latest in docs/ui/rounds)")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "ui" / "compare")
    args = parser.parse_args(argv)

    round_dir = args.round or latest_round(ROOT / "docs" / "ui" / "rounds")
    if round_dir is None:
        print("no numbered round in docs/ui/rounds; pass --round", file=sys.stderr)
        return 1
    written = 0
    for base_name, (page, state) in PAIRS.items():
        left, right = args.baseline / f"{base_name}.png", round_dir / f"{state}.png"
        missing = [str(p) for p in (left, right) if not p.is_file()]
        if missing:
            print(f"skipped {base_name}: missing {', '.join(missing)}", file=sys.stderr)
            continue
        out = args.out / f"{base_name}.png"
        caption = f"Clean Voice, round {round_dir.name}: {page} ({state})"
        side_by_side(left, right, f"Original: {base_name}", caption, out)
        print(out)
        written += 1
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
