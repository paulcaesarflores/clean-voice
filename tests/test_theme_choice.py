"""The window's own Light / Dark / Automatic choice (DESIGN "Theme choice", user, 2026-11-13): in the footer beside
Start with Windows; Automatic follows the system live, Light and Dark fix the theme; everything open follows at once;
saved in prefs."""

from __future__ import annotations

import collections
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import footer_fit
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QProxyStyle,
    QStyle,
    QWidget,
)
from pytestqt.qtbot import QtBot
from test_main_screen import DEVICES, MICS, ON, OUTPUTS, FakeLoginItem, make  # noqa: F401

from clean_voice import filter_window, fine_tune, prefs, theme
from clean_voice.app import MIN_SIZE, THEME_H, WINDOW_SIZE, CloseNote, MainWindow
from clean_voice.preview import PreviewEngine, sample_states
from clean_voice.settings import Settings


@pytest.fixture
def screen(request: pytest.FixtureRequest):
    """test_main_screen's window factory, under this file's own name."""
    return request.getfixturevalue("make")


@pytest.fixture
def system(monkeypatch: pytest.MonkeyPatch):
    """The computer's light/dark setting, simulated; and what the window asks Qt to use app-wide, recorded."""

    class System:
        def __init__(self) -> None:
            self.dark = False
            self.requested: list[str] = []

    state = System()
    monkeypatch.setattr(theme, "system_is_dark", lambda: state.dark)
    monkeypatch.setattr(theme, "request_scheme", state.requested.append)
    return state


def pick(window: MainWindow, mode: str) -> None:
    """Choose ``mode`` in the footer's dropdown, as a click on its item does."""
    combo = window.theme_combo
    index = combo.findData(mode)
    assert index >= 0, mode
    combo.setCurrentIndex(index)
    combo.activated.emit(index)


def has_colour(widget: QWidget, color: str, tolerance: int = 6) -> bool:
    image = widget.grab().toImage()
    want = QColor(color)
    for y in range(0, image.height(), 2):
        for x in range(0, image.width(), 2):
            c = image.pixelColor(x, y)
            if abs(c.red() - want.red()) + abs(c.green() - want.green()) + abs(c.blue() - want.blue()) < tolerance:
                return True
    return False


def background(widget: QWidget) -> QColor:
    """The colour most of ``widget`` shows: its background, whatever its icon or text are drawn in."""
    image = widget.grab().toImage()
    counts = collections.Counter(
        image.pixelColor(x, y).rgb() for y in range(0, image.height(), 2) for x in range(0, image.width(), 2)
    )
    return QColor.fromRgb(counts.most_common(1)[0][0])


def near(color: QColor, want: str, tolerance: int = 6) -> bool:
    w = QColor(want)
    return abs(color.red() - w.red()) + abs(color.green() - w.green()) + abs(color.blue() - w.blue()) < tolerance


# ------------------------------------------------------------------ the control


def test_the_footer_has_a_theme_dropdown_beside_start_with_windows(screen, system) -> None:
    window, _ = screen(ON)
    combo = window.theme_combo
    assert window.footer.isAncestorOf(combo) and window.footer.isAncestorOf(window.theme_label)
    assert [combo.itemText(i) for i in range(combo.count())] == ["Automatic", "Light", "Dark"]
    assert [combo.itemData(i) for i in range(combo.count())] == ["auto", "light", "dark"]
    assert combo.currentData() == "auto" and window.theme_mode == "auto"  # the default
    assert combo.accessibleName() == "Theme" and window.theme_label.text() == "Theme"
    assert window.theme_label.buddy() is combo
    assert combo.focusPolicy() == Qt.FocusPolicy.TabFocus  # the focus ring for the keyboard only, like the others
    chain = window.tab_chain()
    assert chain.index(combo) == chain.index(window.autostart_switch) + 1
    # Beside Start with Windows: on its row, just right of its switch.
    switch = window.autostart_switch
    assert (
        window.theme_label.mapTo(window, window.theme_label.rect().topLeft()).x()
        > switch.mapTo(window, switch.rect().topRight()).x()
    )


def test_the_dropdown_draws_a_focus_ring_for_the_keyboard(screen, system, qtbot: QtBot) -> None:
    window, _ = screen(ON)
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow, timeout=5000)
    combo = window.theme_combo
    plain = combo.grab().toImage()
    combo.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert combo.hasFocus() and combo.grab().toImage() != plain
    assert has_colour(combo, theme.current().desk)  # the ring is the desk colour, as on every other control


# ------------------------------------------------------------------ choosing


@pytest.mark.parametrize("dark_system", [False, True])
def test_choosing_light_or_dark_switches_the_palette_at_once(screen, system, dark_system: bool) -> None:
    system.dark = dark_system
    window, _ = screen(ON)
    for mode, pal in (("dark", theme.DARK), ("light", theme.LIGHT), ("dark", theme.DARK)):
        pick(window, mode)
        assert window.palette_now is pal and theme.current() is pal
        assert pal.surface in window.style_text()
        assert has_colour(window.footer, pal.surface)
        assert system.requested[-1] == mode  # the title bars and native menus are asked for the same


def test_automatic_follows_the_system_live(screen, system) -> None:
    window, _ = screen(ON)
    assert window.palette_now is theme.LIGHT
    system.dark = True
    window.follow_system_theme()  # what the system's colorSchemeChanged signal calls
    assert window.palette_now is theme.DARK
    system.dark = False
    window.follow_system_theme()
    assert window.palette_now is theme.LIGHT


def test_automatic_follows_the_systems_own_signal(screen, system) -> None:
    """Through the real path: Qt's colorSchemeChanged, as the system sends it, reaches the window (r1 m1)."""
    window, _ = screen(ON)
    hints = QGuiApplication.styleHints()
    system.dark = True
    hints.colorSchemeChanged.emit(Qt.ColorScheme.Dark)
    assert window.palette_now is theme.DARK and has_colour(window.footer, theme.DARK.surface)
    system.dark = False
    hints.colorSchemeChanged.emit(Qt.ColorScheme.Light)
    assert window.palette_now is theme.LIGHT


@pytest.mark.parametrize(("mode", "pal"), [("light", theme.LIGHT), ("dark", theme.DARK)])
def test_light_and_dark_ignore_the_system(screen, system, mode: str, pal: theme.Palette) -> None:
    window, _ = screen(ON)
    pick(window, mode)
    for dark in (True, False, True):
        system.dark = dark
        window.follow_system_theme()
        assert window.palette_now is pal


def test_back_to_automatic_takes_the_system_scheme_now(screen, system) -> None:
    window, _ = screen(ON)
    pick(window, "light")
    system.dark = True  # the system went dark while Light was fixed
    window.follow_system_theme()
    assert window.palette_now is theme.LIGHT
    pick(window, "auto")  # no system change needed: Automatic reads it at once
    assert window.palette_now is theme.DARK and system.requested[-1] == "auto"


# ------------------------------------------------------------------ saved


def test_the_choice_is_saved_after_the_usual_delay(screen, system, tmp_path: Path) -> None:
    window, _ = screen(ON)
    pick(window, "dark")
    assert window._save_timer.isActive()  # the existing debounce, not an immediate write
    window.save_prefs()
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert saved["theme"] == "dark"
    # The next launch starts dark although the system is light, and shows the choice.
    again = MainWindow(PreviewEngine(ON, MICS), tmp_path / "settings.json", FakeLoginItem(), "win32")
    try:
        assert again.theme_mode == "dark" and again.theme_combo.currentData() == "dark"
        assert again.palette_now is theme.DARK and system.requested[-1] == "dark"
    finally:
        again.close()


@pytest.mark.parametrize("content", [{"loudness": 61}, {"theme": "midnight"}, {"theme": 3}])
def test_an_old_or_bad_saved_theme_starts_automatic(qtbot: QtBot, tmp_path: Path, system, content: dict) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    system.dark = True
    window = MainWindow(PreviewEngine(ON, MICS), path, FakeLoginItem(), "win32")
    qtbot.addWidget(window)
    assert window.theme_mode == "auto" and window.theme_combo.currentData() == "auto"
    assert window.palette_now is theme.DARK  # follows the (dark) system


@pytest.mark.parametrize("name", ["light", "dark"])
def test_the_theme_name_argument_still_fixes_the_theme(qtbot: QtBot, tmp_path: Path, system, name: str) -> None:
    path = tmp_path / "settings.json"
    prefs.save(path, prefs.Prefs(theme="auto"))
    system.dark = name == "light"  # the system says the opposite
    window = MainWindow(PreviewEngine(ON, MICS), path, FakeLoginItem(), "win32", theme_name=name)
    qtbot.addWidget(window)
    window.follow_system_theme()
    assert window.palette_now is theme.for_name(name) and window.theme_combo.currentData() == name
    window.save_prefs()
    assert prefs.load(path).theme == "auto"  # a render's fixed theme is not written over the guest's choice


# ------------------------------------------------------------------ everything open follows


def test_the_filter_window_and_its_question_follow(screen, system) -> None:
    chain = (fine_tune.new_filter("gain_filter", ()),)
    window, _ = screen(ON, saved=prefs.Prefs(settings=Settings(fine_tune=True, filters=chain)))
    window.show_filters()
    fwindow = window.filter_window
    assert fwindow is not None and fwindow.palette_now is theme.LIGHT
    box = fwindow.confirm_box(filter_window.REMOVE_TITLE, filter_window.remove_text("Gain"))
    box.show()  # the remove question, open while the choice changes
    QApplication.processEvents()
    # Its background and its words, not every pixel: the platform's "?" icon has near-white pixels of its own (r1 m3).
    words = box.findChild(QLabel, "qt_msgbox_label")
    assert words is not None
    assert near(background(box), theme.LIGHT.surface)
    assert words.palette().color(words.foregroundRole()) == QColor(theme.LIGHT.ink)
    pick(window, "dark")
    QApplication.processEvents()
    assert fwindow.palette_now is theme.DARK and theme.DARK.surface in fwindow.styleSheet()
    assert has_colour(fwindow, theme.DARK.surface)
    assert near(background(box), theme.DARK.surface)
    assert words.palette().color(words.foregroundRole()) == QColor(theme.DARK.ink)
    box.close()
    fwindow.close()


def test_the_open_mic_menu_follows(screen, system) -> None:
    window, _ = screen(ON)
    window.open_mic_menu()
    assert has_colour(window.mic_menu, theme.LIGHT.surface)
    pick(window, "dark")
    QApplication.processEvents()
    assert window.mic_menu.isVisible()
    assert has_colour(window.mic_menu, theme.DARK.surface) and not has_colour(window.mic_menu, theme.LIGHT.surface)


def test_a_filter_window_opened_later_starts_in_the_chosen_theme(screen, system) -> None:
    window, _ = screen(ON)
    pick(window, "dark")
    window.show_filters()
    assert window.filter_window is not None and window.filter_window.palette_now is theme.DARK
    window.filter_window.close()


# ------------------------------------------------------------------ fit


def _sample(qtbot: QtBot, tmp_path: Path, size: tuple[int, int], platform: str) -> MainWindow:
    sample = sample_states(DEVICES, platform)["running"]
    prefs.save(tmp_path / "s.json", sample.saved())
    window = MainWindow(
        PreviewEngine(sample.status, MICS, tmp_path, OUTPUTS), tmp_path / "s.json", FakeLoginItem(), platform
    )
    qtbot.addWidget(window)
    window.resize(*size)
    window.show()
    window.tick()
    QApplication.processEvents()
    return window


@pytest.mark.parametrize("platform", ["win32", "darwin"])
@pytest.mark.parametrize("size", [MIN_SIZE, WINDOW_SIZE])
def test_the_footer_fits_with_the_theme_choice(qtbot: QtBot, tmp_path: Path, system, size, platform: str) -> None:
    """In this process: offscreen, at 100 % metrics, with the slack the real platform's scaling takes (F1)."""
    window = _sample(qtbot, tmp_path, size, platform)
    assert footer_fit.problems(window, opening=size == WINDOW_SIZE) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' own text at its display scales")
@pytest.mark.parametrize("scale", ["1", "1.25", "1.5", "1.75", "2"])
def test_the_footer_and_the_steps_fit_on_windows_at_every_display_scale(scale: str) -> None:
    """On the real Windows platform, where 125 % and 150 % text is wider and taller than offscreen's (r1 F1), and
    every line of text up to 1 px taller even at 100 %: the footer fits and the five steps need no scrolling."""
    env = {**os.environ, "QT_QPA_PLATFORM": "windows", "QT_ENABLE_HIGHDPI_SCALING": "0", "QT_SCALE_FACTOR": scale}
    result = subprocess.run(
        [sys.executable, str(Path(footer_fit.__file__))],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_two_note_lines_fit_the_switch_row_whatever_the_font() -> None:
    """Offscreen's stand-in for 125 % text: a 13 px font's natural lines are 16 px apart, too tall for two in the
    28 px row, so the note packs them into it; one line stays as tall as the font."""
    note = CloseNote("Closing keeps Clean Voice open in the tray.")
    font = QFont(note.font())
    font.setPixelSize(13)
    note.setFont(font)
    assert QFontMetricsF(font).lineSpacing() * 2 > THEME_H
    one = math.ceil(QFontMetricsF(font).horizontalAdvance(note.text())) + 2
    assert note.heightForWidth(one) >= QFontMetricsF(font).height() - 1
    assert note.heightForWidth(one * 2 // 3) <= THEME_H  # two lines
    note.resize(one * 2 // 3, note.heightForWidth(one * 2 // 3))
    assert footer_fit.inked_edges(note) == []  # and the last line's descenders are drawn whole (r2 F2)


# ------------------------------------------------------------------ title bars and the tray menu (r1 m2)


@pytest.mark.skipif(sys.platform != "win32", reason="DWM's title bars")
@pytest.mark.parametrize("style", [None, "windowsvista"])  # this Windows' own style, and Windows 10's default
def test_title_bars_and_the_tray_menu_follow_on_windows(style: str | None) -> None:
    env = {**os.environ, "QT_QPA_PLATFORM": "windows"}
    env.pop("QT_STYLE_OVERRIDE", None)
    if style is not None:
        env["QT_STYLE_OVERRIDE"] = style
    script = Path(footer_fit.__file__).with_name("theme_follow.py")
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("style", "kept"), [("windowsvista", False), ("windows11", True), ("fusion", True), ("macos", True)]
)
def test_only_a_style_that_keeps_a_light_palette_is_replaced(qapp: QApplication, style: str, kept: bool) -> None:
    """Windows 10's windowsvista keeps a light palette in Dark, and Qt then keeps the title bars and menus light: the
    app runs Fusion there. Every other style already follows the scheme and stays."""
    from PySide6.QtWidgets import QStyleFactory

    from clean_voice.app import follow_scheme_style

    styles = QStyleFactory.keys()  # Qt's list, not a dict's
    if style.lower() not in [k.lower() for k in styles]:
        pytest.skip(f"{style} is not on this platform")
    before = qapp.style().name()
    try:
        qapp.setStyle(style)
        follow_scheme_style(qapp)
        assert style_key(qapp.style()) == (style if kept else "fusion")
    finally:
        qapp.setStyle(before)


def style_key(style: QStyle) -> str:
    """The style's factory name, through the app's own proxy over Fusion."""
    return (style.baseStyle() if isinstance(style, QProxyStyle) else style).name().lower()


def test_fusion_asks_its_questions_as_windowsvista_did(qapp: QApplication, qtbot: QtBot) -> None:
    """Fusion on Windows 10 keeps the replaced style's question box: Yes and No underline their letters exactly when
    windowsvista would on this machine (while Alt is held, or always with Windows' "Underline access keys" on), and the
    "?" icon at the Windows size, not Fusion's 9 px taller box (theme-choice r2 m4, r3 m5)."""
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QStyleFactory

    from clean_voice.app import follow_scheme_style

    vista = QStyleFactory.create("windowsvista")
    if vista is None:
        pytest.skip("windowsvista is not on this platform")
    icon = vista.pixelMetric(QStyle.PixelMetric.PM_MessageBoxIconSize)

    def underlines() -> tuple[int, int]:
        """The style's answer for a question's Yes button, with Alt up and then with Alt held."""
        box = QMessageBox(QMessageBox.Icon.Question, "t", "q")
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        qtbot.addWidget(box)
        box.show()
        qtbot.waitExposed(box)
        yes = box.button(QMessageBox.StandardButton.Yes)
        hint = QStyle.StyleHint.SH_UnderlineShortcut
        up = qapp.style().styleHint(hint, None, yes)
        QTest.keyPress(box, Qt.Key.Key_Alt, Qt.KeyboardModifier.AltModifier)
        held = qapp.style().styleHint(hint, None, yes)
        QTest.keyRelease(box, Qt.Key.Key_Alt, Qt.KeyboardModifier.NoModifier)
        box.close()
        return up, held

    def gap_above_buttons() -> int:
        box = QMessageBox()
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.ensurePolished()
        buttons = box.findChild(QDialogButtonBox)
        assert buttons is not None
        gap = buttons.contentsMargins().top()
        box.deleteLater()
        return gap

    before = qapp.style().name()
    try:
        qapp.setStyle("windowsvista")
        vista_gap = gap_above_buttons()
        vista_underlines = underlines()
        follow_scheme_style(qapp)
        assert style_key(qapp.style()) == "fusion"
        assert underlines() == vista_underlines
        assert vista_underlines[1] == 1  # Alt shows them under both, whatever the Windows setting
        assert qapp.style().pixelMetric(QStyle.PixelMetric.PM_MessageBoxIconSize) == icon
        assert gap_above_buttons() == vista_gap > 0
    finally:
        qapp.setStyle(before)


def test_the_app_sets_up_its_style_at_launch(qapp: QApplication, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from clean_voice import app as app_module

    calls: list[object] = []
    monkeypatch.setattr(app_module, "follow_scheme_style", calls.append)
    assert app_module.main(["--quit-after-ms", "300"], engine=PreviewEngine(ON), prefs_path=tmp_path / "s.json") == 0
    assert calls == [qapp]
