"""J10 Step 11, the app icon from the mark (Tests T35-T37; DESIGN "App icon"): one source (mark.py), the files
Briefcase packs, crisp small sizes, and one QIcon for the window, the app and the tray."""

from __future__ import annotations

import importlib.util
import itertools
import tomllib
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from pytestqt.qtbot import QtBot

from clean_voice import mark

ROOT = Path(__file__).resolve().parents[1]
ICONS = ROOT / "icons"


def _tool():
    spec = importlib.util.spec_from_file_location("make_icons", ROOT / "tools" / "make_icons.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pixels(data: bytes) -> QImage:
    image = QImage.fromData(data)
    assert not image.isNull()
    return image.convertToFormat(QImage.Format.Format_ARGB32)


# The antialiased edges (the rounded corners, and the bars above 32 px) come out a little different from one Qt build
# and CPU to another: CI's Windows runner differed at 48 px and its Mac at 20 px (run 43368578097) from the files made
# on Windows. Only there may a pixel differ, by at most this much of 255 in each premultiplied channel: an edge moved by
# 0.2 px passes, 0.5 px fails, and so does any change of colour, bar or size (checked below by redrawing the mark so).
EDGE_TOLERANCE = 32


def _premultiplied(image: QImage) -> np.ndarray:
    """The image's pixels as (height, width, 4) premultiplied ints, alpha last: what the screen shows of each."""
    image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    bgra = rows[:, : image.width() * 4].reshape(image.height(), image.width(), 4).astype(int)
    return bgra[..., [2, 1, 0, 3]]  # ARGB32 words are B, G, R, A bytes on the little-endian machines we run on


def drawn_differently(fresh: QImage, committed: QImage) -> list[str]:
    """Where ``fresh`` is not the ``committed`` icon beyond antialiasing: a pixel that differs although neither it nor
    a neighbour is blended in ``committed`` (an edge's partly covered pixel), or that differs by more than
    EDGE_TOLERANCE."""
    if fresh.size() != committed.size():
        return [f"{fresh.size().toTuple()} instead of {committed.size().toTuple()}"]
    new, old = _premultiplied(fresh), _premultiplied(committed)
    # Unblended: transparent, and the committed icon's own two commonest opaque colours (the ground and the bars).
    opaque, counts = np.unique(old[old[..., 3] == 255], axis=0, return_counts=True)
    solid = [np.all(old == 0, axis=2)] + [np.all(old == opaque[i], axis=2) for i in np.argsort(counts)[-2:]]
    blended = np.pad(~np.logical_or.reduce(solid), 1)
    h, w = committed.height(), committed.width()
    edges = np.logical_or.reduce(
        [blended[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    )
    diff = np.abs(new - old).max(axis=2)
    wrong = (diff > 0) & (~edges | (diff > EDGE_TOLERANCE))
    return [f"({x}, {y}): {new[y, x].tolist()} instead of {old[y, x].tolist()}" for y, x in np.argwhere(wrong)]


def test_t35_make_icons_gives_the_committed_files(qapp, tmp_path: Path) -> None:
    tool = _tool()
    made_ico, made_icns = tool.make(tmp_path)
    committed_ico = tool.read_ico((ICONS / "clean-voice.ico").read_bytes())
    fresh_ico = tool.read_ico(made_ico.read_bytes())
    assert sorted(fresh_ico) == sorted(committed_ico)
    for size, data in fresh_ico.items():
        diff = drawn_differently(pixels(data), pixels(committed_ico[size]))
        assert not diff, f"{size} px: the .ico is not what mark.py draws: {diff[:8]}"
    committed_icns = tool.read_icns((ICONS / "clean-voice.icns").read_bytes())
    fresh_icns = tool.read_icns(made_icns.read_bytes())
    assert [k for k, _ in fresh_icns] == [k for k, _ in committed_icns]
    for (kind, data), (_, old) in zip(fresh_icns, committed_icns, strict=True):
        diff = drawn_differently(pixels(data), pixels(old))
        assert not diff, f"{kind}: the .icns is not what mark.py draws: {diff[:8]}"


def _drawn(size: int, radius_px: float = 0.0, green: str = mark.GREEN, shift: int = 0) -> QImage:
    """The mark as paint_mark draws it, its corners' radius ``radius_px`` larger, its bars ``green`` and ``shift`` px
    to the right."""
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(0)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(mark.GROUND))
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.2 + radius_px, size * 0.2 + radius_px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, size > mark.SNAP_AT)
    p.setBrush(QColor(green))
    for rect in mark.bar_rects(size):
        p.drawRect(rect.translated(shift, 0))
    p.end()
    return image


@pytest.mark.parametrize("size", [16, 20, 48, 64, 256])
def test_t35_the_comparison_passes_antialiasing_and_catches_a_changed_icon(qapp, size: int) -> None:
    tool = _tool()
    committed = _drawn(size)
    assert drawn_differently(pixels(tool.png(size)), committed) == []  # _drawn is the mark
    for wobble in (0.01, 0.05, -0.05, 0.2):  # edges a fraction of a pixel away: antialiasing
        assert drawn_differently(_drawn(size, radius_px=wobble), committed) == [], wobble
    for changed in (_drawn(size, radius_px=0.5), _drawn(size, green="#1E9A69"), _drawn(size, shift=1)):
        assert drawn_differently(changed, committed)
    assert drawn_differently(pixels(tool.png(size, two_bars=True)), committed)


def test_t36_the_files_hold_every_size(qapp) -> None:
    tool = _tool()
    ico = tool.read_ico((ICONS / "clean-voice.ico").read_bytes())
    assert sorted(ico) == [16, 20, 24, 32, 40, 48, 64, 256]
    for size, data in ico.items():
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and pixels(data).size().toTuple() == (size, size)
    icns = dict(tool.read_icns((ICONS / "clean-voice.icns").read_bytes()))
    expected = {
        "icp4": 16,
        "icp5": 32,
        "icp6": 64,
        "ic07": 128,
        "ic08": 256,
        "ic09": 512,
        "ic10": 1024,
        "ic11": 32,  # 16@2x
        "ic12": 64,  # 32@2x
        "ic13": 256,  # 128@2x
        "ic14": 512,  # 256@2x
    }
    assert set(icns) == set(expected)
    for kind, size in expected.items():
        assert pixels(icns[kind]).size().toTuple() == (size, size), kind


@pytest.mark.parametrize("size", [16, 20, 24, 32])
def test_t36_small_bars_sit_on_whole_pixels_with_no_partly_lit_column(qapp, size: int) -> None:
    rects = mark.bar_rects(size)
    for r in rects:
        assert all(float(v).is_integer() for v in (r.left(), r.right(), r.top(), r.bottom())), (size, r)
    gaps = {round(b.left() - a.right()) for a, b in itertools.pairwise(rects)}
    assert len(gaps) == 1 and min(gaps) >= 1  # equal gaps, never touching
    image = mark.pixmap(size).toImage().convertToFormat(QImage.Format.Format_ARGB32)
    green, ground = QColor(mark.GREEN), QColor(mark.GROUND)
    y = round(rects[1].bottom()) - 1  # a row through every bar
    row = [image.pixelColor(x, y) for x in range(round(rects[0].left()) - 1, round(rects[-1].right()) + 1)]
    assert all(c.rgb() in (green.rgb(), ground.rgb()) for c in row), f"{size} px: a partly lit column"
    lit = [x for x in range(size) if image.pixelColor(x, y).rgb() == green.rgb()]
    assert len(lit) == sum(round(r.width()) for r in rects)


def test_t36_one_qicon_for_the_window_the_app_and_the_tray(qtbot: QtBot, tmp_path: Path) -> None:
    from test_main_screen import MICS, ON, OUTPUTS, FakeLoginItem

    from clean_voice import tray
    from clean_voice.app import MainWindow
    from clean_voice.preview import PreviewEngine

    icon = mark.icon()
    assert icon is mark.icon()  # built once
    assert {s.width() for s in icon.availableSizes()} >= {16, 20, 24, 32, 40, 48, 64, 256}
    window = MainWindow(PreviewEngine(ON, MICS, tmp_path, OUTPUTS), tmp_path / "s.json", FakeLoginItem(), "win32")
    qtbot.addWidget(window)
    window.tray = tray.Tray(window)
    try:
        key = icon.cacheKey()
        assert window.windowIcon().cacheKey() == key
        assert window.tray.icon.icon().cacheKey() == key
    finally:
        window.tray.hide()
        window.tray = None
    from clean_voice import app as app_module

    source = Path(str(app_module.__file__)).read_text(encoding="utf-8")
    assert "app.setWindowIcon(mark.icon())" in source  # main() gives the application the same icon


def test_t37_briefcase_takes_the_icon_from_icons(qapp) -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["briefcase"]["app"]["clean-voice"]["icon"] == "icons/clean-voice"
    for suffix in (".ico", ".icns"):
        assert (ROOT / f"icons/clean-voice{suffix}").is_file()


def test_the_icon_sheet_shows_both_16_px_versions(qapp, tmp_path: Path) -> None:
    tool = _tool()
    tool.sheet(tmp_path / "icon-sheet.png")
    assert not QImage(str(tmp_path / "icon-sheet.png")).isNull()
    two = pixels(tool.png(16, two_bars=True))
    lit = {x for x in range(16) for y in range(16) if two.pixelColor(x, y).rgb() == QColor(mark.GREEN).rgb()}
    runs = sum(1 for x in sorted(lit) if x - 1 not in lit)
    assert runs == 2  # the two-bar fallback really has two bars
