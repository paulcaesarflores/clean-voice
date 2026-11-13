"""Barlow (SIL Open Font License 1.1, fonts/OFL.txt), bundled with Clean Voice (docs/ui/DIRECTION.md v2).

The files sit inside the package, so Briefcase ships them with the app. ``load()`` registers them with Qt once; the
system fonts stay behind them for Japanese and other scripts.
"""

from __future__ import annotations

from pathlib import Path

FOLDER = Path(__file__).resolve().parent / "fonts"
FILES = ("Barlow-Regular.ttf", "Barlow-Medium.ttf", "BarlowSemiCondensed-SemiBold.ttf")
BODY = "Barlow"
HEADLINE = "Barlow Semi Condensed"
# After Barlow: the platform UI fonts, then fonts with Japanese glyphs (Qt falls back per glyph through the list).
FALLBACK = ["Segoe UI", "SF Pro Text", ".AppleSystemUIFont", "Yu Gothic UI", "Meiryo UI", "Hiragino Sans", "Arial"]

_loaded: list[str] | None = None


def load() -> list[str]:
    """Register the bundled fonts with Qt (needs a QGuiApplication); returns the families found. Safe to call twice."""
    global _loaded
    if _loaded is not None:
        return _loaded
    from PySide6.QtGui import QFontDatabase

    families: list[str] = []
    for name in FILES:
        font_id = QFontDatabase.addApplicationFont(str(FOLDER / name))
        if font_id >= 0:
            families += [f for f in QFontDatabase.applicationFontFamilies(font_id) if f not in families]
    _loaded = families
    return families


def body() -> list[str]:
    return [BODY, *FALLBACK]


def headline() -> list[str]:
    return [HEADLINE, BODY, *FALLBACK]
