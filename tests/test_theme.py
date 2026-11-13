"""J10 Step 8: the one window's design tokens (docs/plans/J10-one-window.md §7 Tokens) and T25, their contrast."""

from __future__ import annotations

import dataclasses

import pytest

from clean_voice import theme
from clean_voice.theme import DARK, LIGHT, Palette


def luminance(color: str) -> float:
    """WCAG 2.x relative luminance of #RRGGBB."""

    def channel(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# §7 Tokens, exactly.
PLAN = {
    "light": {
        "surface": "#F7F9FA",
        "band": "#E8ECEF",
        "line": "#CDD5DC",
        "ink": "#15212B",
        "muted": "#56636F",
        "desk": "#2F56C8",
        "signal": "#1E9A68",
        "hot": "#D99A17",
        "clip": "#D2413A",
        "hot_text": "#8F6200",
    },
    "dark": {
        "surface": "#1A232B",
        "band": "#11181E",
        "line": "#2C3843",
        "ink": "#E4EBF0",
        "muted": "#97A4AF",
        "desk": "#88A6FF",
        "signal": "#3DCB8D",
        "hot": "#F0B43C",
        "clip": "#F06A63",
    },
}


@pytest.mark.parametrize("pal", [LIGHT, DARK], ids=["light", "dark"])
def test_the_tokens_are_the_plans(pal: Palette) -> None:
    for name, value in PLAN[pal.name].items():
        assert getattr(pal, name) == value, name


def test_the_old_names_and_the_band_colours_are_gone() -> None:
    fields = {f.name for f in dataclasses.fields(Palette)}
    assert not fields & {"wall", "panel", "rule", "bands"}


# Every text pair the window draws: (text token, background token, minimum ratio). Body text and hints need 4.5:1;
# the 26 px headline is large text, 3:1 (WCAG 1.4.3).
BODY = 4.5
LARGE = 3.0
PAIRS = [
    ("ink", "surface", BODY),  # body, step titles, controls
    ("ink", "band", BODY),  # the header's subline area, the "raw mic" headline
    ("muted", "surface", BODY),  # hints, the version
    ("muted", "band", BODY),  # the subline
    ("desk", "surface", BODY),  # Check for updates (a link)
    ("on_desk", "desk", BODY),  # the primary button's words
    ("ink", "hover", BODY),  # a hovered menu item
    ("signal", "band", LARGE),  # "Your call hears your clean voice"
    ("hot_text", "band", LARGE),  # "Your call can't hear you yet"
    ("hot_text", "surface", BODY),  # a warning line in a step (echo paused, a backup error)
]


@pytest.mark.parametrize("pal", [LIGHT, DARK], ids=["light", "dark"])
@pytest.mark.parametrize(("text", "ground", "minimum"), PAIRS)
def test_t25_every_text_pair_meets_wcag(pal: Palette, text: str, ground: str, minimum: float) -> None:
    ratio = contrast(getattr(pal, text), getattr(pal, ground))
    assert ratio >= minimum, f"{pal.name}: {text} on {ground} is {ratio:.2f}:1, needs {minimum}:1"


def test_t25_the_plain_hot_on_the_band_is_why_hot_text_exists() -> None:
    # §7 "Fix, light theme": hot on band is about 2.1:1, under the 3:1 a large headline needs.
    assert contrast(LIGHT.hot, LIGHT.band) < LARGE
    assert contrast(LIGHT.hot_text, LIGHT.band) >= LARGE


def test_current_and_use() -> None:
    before = theme.current()
    try:
        theme.use(DARK)
        assert theme.current() is DARK and theme.for_name("dark") is DARK and theme.for_name("light") is LIGHT
    finally:
        theme.use(before)


def test_palette_for_each_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    # DESIGN "Theme choice" (2026-11-13): Light and Dark are fixed; Automatic is the system's.
    assert theme.MODES == ("auto", "light", "dark")
    for dark in (False, True):
        monkeypatch.setattr(theme, "system_is_dark", lambda dark=dark: dark)
        assert theme.palette_for("light") is LIGHT and theme.palette_for("dark") is DARK
        assert theme.palette_for("auto") is (DARK if dark else LIGHT)


def test_request_scheme_asks_qt_for_each_choice(qtbot) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication

    hints = QGuiApplication.styleHints()
    calls: list[object] = []
    # The offscreen platform ignores the request, so record what Qt is asked (the native result is checked by hand).
    real_set, real_unset = hints.setColorScheme, hints.unsetColorScheme
    try:
        hints.setColorScheme = calls.append  # type: ignore[method-assign]
        hints.unsetColorScheme = lambda: calls.append("unset")  # type: ignore[method-assign]
        for mode in ("dark", "light", "auto"):
            theme.request_scheme(mode)
    finally:
        hints.setColorScheme, hints.unsetColorScheme = real_set, real_unset  # type: ignore[method-assign]
    assert calls == [Qt.ColorScheme.Dark, Qt.ColorScheme.Light, "unset"]
