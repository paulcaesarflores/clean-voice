"""Clean Voice's colours for the one window (docs/plans/J10-one-window.md §7 Tokens, evolved from "studio wall"): cool
neutrals, so the signal colours are the only saturated thing on screen. One set of tokens in a light and a dark
version; the guest chooses Light, Dark or Automatic (the system's setting, live), and every painted widget reads
``current()`` when it paints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    name: str
    surface: str  # the window body, the filter window, dropdowns
    band: str  # the header band
    line: str  # hairlines, slider tracks, the meter's unlit zones, the off power ring, outlines of off toggles
    ink: str  # text
    muted: str  # secondary text (at least 4.5:1 on surface and band)
    desk: str  # controls: the on-outline of Step 2's toggles, slider fill, focus ring, primary button
    signal: str  # voice reaching the call: the headline, the power ring, the meter's nominal zone
    hot: str  # the meter's warning zone; a problem's marker and power ring
    clip: str  # the meter's error zone; the recording dot
    hot_text: str  # a problem's words: the "can't hear you" headline (hot itself is too light for text on band)
    on_desk: str  # text on a desk or signal fill
    hover: str  # a hovered control

    @property
    def dark(self) -> bool:
        return self.name == "dark"


LIGHT = Palette(
    name="light",
    surface="#F7F9FA",
    band="#E8ECEF",
    line="#CDD5DC",
    ink="#15212B",
    muted="#56636F",
    desk="#2F56C8",
    signal="#1E9A68",
    hot="#D99A17",
    clip="#D2413A",
    hot_text="#8F6200",
    on_desk="#FFFFFF",
    hover="#E1E6EA",
)

DARK = Palette(
    name="dark",
    surface="#1A232B",
    band="#11181E",
    line="#2C3843",
    ink="#E4EBF0",
    muted="#97A4AF",
    desk="#88A6FF",
    signal="#3DCB8D",
    hot="#F0B43C",
    clip="#F06A63",
    hot_text="#F0B43C",  # hot itself reads on the dark band (test_theme.py T25)
    on_desk="#11181E",
    hover="#24303A",
)

_current = LIGHT


def current() -> Palette:
    return _current


def use(palette: Palette) -> None:
    global _current
    _current = palette


def for_name(name: str) -> Palette:
    return DARK if name == "dark" else LIGHT


def system_is_dark() -> bool:
    """Whether Windows or macOS is set to dark (Qt 6.5+ reads it from the platform). No answer means light. While
    ``request_scheme`` fixes Light or Dark, Qt answers with that choice instead."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication

    if QGuiApplication.instance() is None:
        return False
    return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark


MODES = ("auto", "light", "dark")  # the guest's choice in the footer (DESIGN "Theme choice", 2026-11-13)


def palette_for(mode: str) -> Palette:
    """The palette a choice shows now: Light or Dark, or for Automatic the system's."""
    if mode == "auto":
        return DARK if system_is_dark() else LIGHT
    return for_name(mode)


def request_scheme(mode: str) -> None:
    """Ask Qt for ``mode`` app-wide (Qt 6.8+), so what Clean Voice does not paint itself follows the choice too: the
    title bars of every window (the VST editor's frame included), the tray menu and native dialogs. "auto" gives the
    system's setting back, and Qt then reports the system's changes again."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication

    if QGuiApplication.instance() is None:
        return
    hints = QGuiApplication.styleHints()
    if mode == "auto":
        hints.unsetColorScheme()
    else:
        hints.setColorScheme(Qt.ColorScheme.Dark if mode == "dark" else Qt.ColorScheme.Light)


def rgba(color: str, alpha: float) -> str:
    """``color`` (#RRGGBB) at ``alpha`` (0..1), for a style sheet."""
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"
