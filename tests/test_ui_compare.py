"""S3: tools/ui_compare.py puts the baseline and the round screenshot side by side, each under its caption."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtGui import QColor, QImage

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))


def solid(path: Path, width: int, height: int, colour: str) -> Path:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(colour))
    assert image.save(str(path))
    return path


def ink_in(image: QImage, x0: int, x1: int, y0: int, y1: int) -> int:
    """Pixels in the rectangle that differ from the white background (caption text)."""
    white = QColor("#ffffff").rgb()
    return sum(image.pixel(x, y) != white for x in range(x0, x1) for y in range(y0, y1))


def test_side_by_side_is_both_widths_plus_gutter_with_both_captions(tmp_path: Path) -> None:
    import ui_compare

    left = solid(tmp_path / "left.png", 200, 300, "#101010")
    right = solid(tmp_path / "right.png", 150, 340, "#2040a0")
    out = tmp_path / "out" / "pair.png"
    ui_compare.side_by_side(left, right, "Original: voice", "Clean Voice, round 1: running", out)

    image = QImage(str(out))
    assert image.width() == 200 + ui_compare.GUTTER + 150
    assert image.height() == ui_compare.CAPTION + 340
    # Each screenshot sits under the caption band, on its own side of the gutter.
    assert image.pixel(100, ui_compare.CAPTION + 150) == QColor("#101010").rgb()
    assert image.pixel(200 + ui_compare.GUTTER + 75, ui_compare.CAPTION + 150) == QColor("#2040a0").rgb()
    # Both captions are drawn: ink in the band above each screenshot.
    assert ink_in(image, 0, 200, 0, ui_compare.CAPTION) > 20
    assert ink_in(image, 200 + ui_compare.GUTTER, image.width(), 0, ui_compare.CAPTION) > 20


def test_cli_pairs_the_latest_round_and_skips_missing_states(tmp_path: Path) -> None:
    baseline, rounds, out = tmp_path / "baseline", tmp_path / "rounds", tmp_path / "compare"
    for d in (baseline, rounds / "1", rounds / "2"):
        d.mkdir(parents=True)
    solid(baseline / "voice.png", 52, 80, "#111111")
    solid(baseline / "fine-tune.png", 52, 80, "#111111")
    solid(rounds / "2" / "running.png", 52, 80, "#222222")
    result = subprocess.run(
        [sys.executable, str(TOOLS / "ui_compare.py"), "--baseline", str(baseline), "--round", str(rounds / "2")]
        + ["--out", str(out)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in out.iterdir()) == ["voice.png"]
    assert "skipped fine-tune" in result.stderr

    import ui_compare

    assert ui_compare.latest_round(rounds) == rounds / "2"
    # J10: the original's four tabs are all in the one window now, and its Fine-tune tab is the filter window.
    assert ui_compare.PAIRS["voice"] == ("One window", "running")
    assert ui_compare.PAIRS["fine-tune"] == ("Fine-tune filters", "filters-basic-eq")


def test_every_paired_state_is_rendered() -> None:
    import ui_compare

    from clean_voice.preview import DeviceNames, filter_samples, sample_states

    rendered = set(sample_states(DeviceNames())) | set(filter_samples())
    assert {state for _, state in ui_compare.PAIRS.values()} <= rendered
