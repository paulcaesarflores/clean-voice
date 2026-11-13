"""Clean Voice's mark, the one source of every place it shows: the window's footer, the title bar, the taskbar and
Dock, the tray, and the app icon files that tools/make_icons.py writes (J10 Step 11, DESIGN "App icon").

Three LED bars, lit, on a dark square: a level meter at speaking level. Drawn on a 24-unit grid. At 32 px and below
every bar edge is snapped to a whole pixel, so the small icons stay crisp; 16 px keeps all three bars (the icon sheet
shows the two-bar fallback beside it for comparison)."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap

GREEN = "#1E9A68"  # the light theme's signal green, in both themes, like any app icon
GROUND = "#15212B"
GRID = 24.0
# (left, height) of each bar on the grid; every bar is 3 units wide and stands on the line y = 19.
BARS = ((6.0, 7.0), (11.0, 12.0), (16.0, 9.0))
TWO_BARS = ((7.5, 8.0), (13.5, 12.0))  # the fallback for 16 px, shown in the icon sheet only
BAR_W = 3.0
BASE = 19.0
SNAP_AT = 32  # this size and below: whole-pixel bars
# Every size the icon files hold (Briefcase needs 16/32/48/64/256 in the .ico; the .icns members go to 1024).
ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256, 512, 1024)


def bar_rects(size: int, bars: tuple[tuple[float, float], ...] = BARS) -> list[QRectF]:
    """The bars at ``size`` px. At SNAP_AT and below, the bar width and the gap are each rounded to whole pixels and
    the group is centred where the grid puts it, so every bar edge is on a pixel and the gaps stay equal."""
    k = size / GRID
    if size > SNAP_AT:
        return [QRectF(x * k, (BASE - h) * k, BAR_W * k, h * k) for x, h in bars]
    width = max(1, round(BAR_W * k))
    gap = max(1, round((bars[1][0] - bars[0][0] - BAR_W) * k))
    group = len(bars) * width + (len(bars) - 1) * gap
    left = round((bars[0][0] + bars[-1][0] + BAR_W) / 2 * k - group / 2)
    bottom = round(BASE * k)
    rects = []
    for i, (_, h) in enumerate(bars):
        top = bottom - max(1, round(h * k))
        rects.append(QRectF(left + i * (width + gap), top, width, bottom - top))
    return rects


def paint_mark(p: QPainter, size: int, bars: tuple[tuple[float, float], ...] = BARS) -> None:
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(GROUND))
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.2, size * 0.2)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, size > SNAP_AT)  # snapped bars need no smoothing
    p.setBrush(QColor(GREEN))
    for rect in bar_rects(size, bars):
        p.drawRect(rect)


def pixmap(size: int, bars: tuple[tuple[float, float], ...] = BARS) -> QPixmap:
    image = QPixmap(size, size)
    image.fill(Qt.GlobalColor.transparent)
    p = QPainter(image)
    paint_mark(p, size, bars)
    p.end()
    return image


_icon: QIcon | None = None


def icon() -> QIcon:
    """One QIcon with the mark at every size, shared by the application (window, taskbar, Dock) and the tray."""
    global _icon
    if _icon is None:
        made = QIcon()
        for size in ICON_SIZES:
            made.addPixmap(pixmap(size))
        _icon = made
    return _icon
