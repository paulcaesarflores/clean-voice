"""J10 Step 9, the one window (docs/plans/J10-one-window.md §7, Tests T26-T30), and the behaviours of the deleted page
tests (test_presets_page.py, the page parts of test_screen_round4.py) that still exist."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QMargins, QRect, Qt
from PySide6.QtGui import QColor, QFontMetricsF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QStackedWidget, QWidget
from pytestqt.qtbot import QtBot
from test_main_screen import DEVICES, MICS, ON, OUTPUTS, FakeLoginItem, make, visible  # noqa: F401

from clean_voice import app as app_module
from clean_voice import fine_tune, prefs, theme, tray, widgets
from clean_voice.app import (
    HEAD_CLEAN,
    HEAD_NO_CALL,
    HEAD_NO_DEVICE,
    HEAD_NO_MIC,
    HEAD_RAW,
    HEAD_STARTING,
    MIN_SIZE,
    NO_CLEANUP,
    STEPS,
    WINDOW_SIZE,
    MainWindow,
)
from clean_voice.preview import PreviewEngine, sample_states
from clean_voice.settings import PRESETS, Settings, replace, style_bands
from clean_voice.widgets import keyboard_focus


@pytest.fixture
def screen(request: pytest.FixtureRequest):
    """test_main_screen's window factory, under this file's own name."""
    return request.getfixturevalue("make")


def top(window: MainWindow, widget: QWidget) -> int:
    return widget.mapTo(window, widget.rect().topLeft()).y()


def bottom(window: MainWindow, widget: QWidget) -> int:
    return widget.mapTo(window, widget.rect().bottomLeft()).y()


def in_view(window: MainWindow, widget: QWidget) -> bool:
    """Whether ``widget`` lies wholly inside the steps' visible area (no scrolling needed)."""
    viewport = window.body.viewport()
    rect = QRect(widget.mapTo(viewport, widget.rect().topLeft()), widget.size())
    return viewport.rect().contains(rect)


def grab_has(widget: QWidget, color: str, tolerance: int = 20) -> bool:
    image = widget.grab().toImage()
    want = QColor(color)
    for y in range(image.height()):
        for x in range(image.width()):
            c = image.pixelColor(x, y)
            if abs(c.red() - want.red()) + abs(c.green() - want.green()) + abs(c.blue() - want.blue()) < tolerance:
                return True
    return False


# ------------------------------------------------------------------ T26 one window


def test_t26_one_window_with_the_header_and_five_steps(screen) -> None:
    window, _ = screen(ON)
    assert not window.findChildren(QStackedWidget)
    for gone in ("pages", "tab_buttons", "show_tab", "current_tab", "pill", "banner", "quick_bar", "mic_combo"):
        assert not hasattr(window, gone), gone
    for gone in ("PAGES", "SIDEBAR_W", "NavButton", "StatusPill", "PAGE_ICONS"):
        assert not hasattr(app_module, gone), gone
    assert window.findChild(QWidget, "sidebar") is None
    titles = [step.title.text() for step in window.steps]
    assert (
        titles
        == list(STEPS)
        == [
            "Choose your microphone",
            "Clean your voice",
            "Adjust volume",
            "Send clean voice to",
            "Record backup",
        ]
    )
    assert [step.number for step in window.steps] == [1, 2, 3, 4, 5]
    tops = [top(window, step) for step in window.steps]
    assert tops == sorted(tops)
    assert window.header.isAncestorOf(window.power_button) and window.header.isAncestorOf(window.headline)
    assert not window.body.isAncestorOf(window.header)  # the header never scrolls
    assert WINDOW_SIZE == (760, 684) and MIN_SIZE == (680, 520)
    assert (window.minimumWidth(), window.minimumHeight()) == MIN_SIZE


def test_t26_the_window_opens_at_760_by_684(qtbot: QtBot, tmp_path: Path) -> None:
    window = MainWindow(PreviewEngine(ON, MICS, tmp_path, OUTPUTS), tmp_path / "s.json", FakeLoginItem(), "win32")
    qtbot.addWidget(window)
    assert (window.width(), window.height()) == WINDOW_SIZE


WIN11_FRAME = QMargins(0, 30, 0, 0)  # what Qt measures on Windows 11 (the title bar; the borders are invisible)
WIN10_FRAME = QMargins(8, 31, 8, 8)


@pytest.mark.parametrize(
    ("available", "frame", "want"),
    [
        # 1366 x 768 at 100 %, a 48 px taskbar at the bottom: 684 + 30 fits in 720.
        (QRect(0, 0, 1366, 720), WIN11_FRAME, QRect(303, 33, 760, 684)),
        # The same laptop with a thicker frame: 720 - 39 leaves 681.
        (QRect(0, 0, 1366, 720), WIN10_FRAME, QRect(303, 31, 760, 681)),
        # 1920 x 1080 at 150 %: 1280 x 720 logical, and its taskbar (48 px) leaves 672.
        (QRect(0, 0, 1280, 672), WIN11_FRAME, QRect(260, 30, 760, 642)),
        # A large screen, its taskbar on the left: the full size, centred in what is left.
        (QRect(62, 0, 2498, 1440), WIN11_FRAME, QRect(931, 393, 760, 684)),
        # A second screen to the left of the first, the Mac's menu bar on top.
        (QRect(-1440, 25, 1440, 875), QMargins(0, 28, 0, 0), QRect(-1100, 134, 760, 684)),
        # Smaller than the minimum: the minimum, with the title bar kept on the screen.
        (QRect(0, 0, 640, 480), WIN11_FRAME, QRect(0, 30, 680, 520)),
    ],
    ids=["1366x768", "1366x768-win10-frame", "1920x1080-at-150", "large", "mac-second-screen", "tiny"],
)
def test_m4_the_opening_size_fits_the_screen_with_its_frame(available: QRect, frame: QMargins, want: QRect) -> None:
    got = app_module.opening_geometry(available, frame)
    assert got == want
    outer = got.marginsAdded(frame)
    if available.height() >= MIN_SIZE[1] + frame.top() + frame.bottom():
        assert available.contains(outer)  # the footer is never under the taskbar


class FakeScreen:
    def __init__(self, available: QRect) -> None:
        self.available = available

    def availableGeometry(self) -> QRect:  # QScreen's name
        return self.available


@pytest.mark.parametrize("available", [QRect(0, 0, 1280, 672), QRect(0, 0, 1366, 720), QRect(0, 0, 2560, 1400)])
def test_m4_the_window_shows_fitted_to_its_screen_and_the_steps_scroll(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, available: QRect
) -> None:
    window = MainWindow(PreviewEngine(ON, MICS, tmp_path, OUTPUTS), tmp_path / "s.json", FakeLoginItem(), "win32")
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "screen", lambda: FakeScreen(available))
    app_module.show_window(window, minimized=False)
    QApplication.processEvents()
    frame = window.frameGeometry()
    assert available.contains(frame)
    want = min(WINDOW_SIZE[1], available.height() - (frame.height() - window.height()))
    assert (window.width(), window.height()) == (WINDOW_SIZE[0], want)
    assert abs(frame.center().x() - available.center().x()) <= 1
    assert abs(frame.center().y() - available.center().y()) <= 1
    assert visible(window, window.footer) and bottom(window, window.footer) < window.height()  # pinned
    if want < WINDOW_SIZE[1]:
        bar = window.body.verticalScrollBar()
        assert bar.maximum() > 0
        bar.setValue(bar.maximum())
        QApplication.processEvents()
        assert in_view(window, window.steps[4])
    # Once only: showing again from the tray keeps where the guest put it.
    window.hide()
    window.move(window.x() + 40, window.y() + 20)
    moved = window.pos()
    window.show_from_tray()
    assert window.pos() == moved


def test_m4_a_minimized_launch_is_fitted_when_the_tray_first_shows_it(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # J10 r2 m2: a --minimized login launch never goes through show_window's fit; show_from_tray is its only one.
    available = QRect(0, 0, 1280, 672)
    window = MainWindow(PreviewEngine(ON, MICS, tmp_path, OUTPUTS), tmp_path / "s.json", FakeLoginItem(), "win32")
    qtbot.addWidget(window)
    monkeypatch.setattr(window, "screen", lambda: FakeScreen(available))
    assert not window.isVisible()  # never shown, as after a --minimized launch
    window.show_from_tray()
    QApplication.processEvents()
    frame = window.frameGeometry()
    assert available.contains(frame)
    assert window.height() == available.height() - (frame.height() - window.height()) < WINDOW_SIZE[1]
    assert abs(frame.center().x() - available.center().x()) <= 1
    assert abs(frame.center().y() - available.center().y()) <= 1


def test_the_steps_fit_at_the_opening_size_and_scroll_at_the_minimum(screen) -> None:
    # Open question 9 and round 13: every step fully visible at the opening size; below it the steps scroll and the
    # header stays.
    window, _ = screen(ON)
    window.resize(*WINDOW_SIZE)
    QApplication.processEvents()
    for step in window.steps:
        assert in_view(window, step), step.title.text()
    window.resize(*MIN_SIZE)
    QApplication.processEvents()
    bar = window.body.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(bar.maximum())
    QApplication.processEvents()
    assert in_view(window, window.steps[4]) and top(window, window.header) == 0
    assert visible(window, window.power_button) and visible(window, window.updates_view.check_button)


# ------------------------------------------------------------------ T27 every state

# state: (headline, its tone, the power ring, the step that holds the notice or None)
EXPECTED = {
    "running": (HEAD_CLEAN, "signal", "signal", None),
    "power-off": (HEAD_RAW, "ink", "line", None),
    "starting": (HEAD_STARTING, "ink", "line", None),
    "cable-missing": (HEAD_NO_CALL, "hot", "hot", 4),
    "cable-lost": (HEAD_NO_CALL, "hot", "hot", 4),
    "voice-to-speakers": (HEAD_NO_CALL, "hot", "hot", 4),
    "no-mic": (HEAD_NO_MIC, "hot", "hot", 1),
    "mic-unplugged": (HEAD_NO_MIC, "hot", "hot", 1),
    "error": (HEAD_NO_DEVICE, "hot", "hot", 1),
}
CLEAN = (HEAD_CLEAN, "signal", "signal", None)


@pytest.mark.parametrize("state", list(sample_states(DEVICES)))
def test_t27_every_state_says_what_the_call_hears_and_where_to_look(qtbot: QtBot, tmp_path: Path, state: str) -> None:
    sample = sample_states(DEVICES)[state]
    prefs.save(tmp_path / "s.json", sample.saved())
    window = MainWindow(sample.engine(tmp_path), tmp_path / "s.json", FakeLoginItem(), "win32")
    qtbot.addWidget(window)
    window.show()
    window.tick()
    headline, tone, power, step = EXPECTED.get(state, CLEAN)
    assert window.headline.text() == headline and window.headline.property("tone") == tone
    assert window.power_button.tone == power and window.header.lit == (power == "signal")
    mic_notice = visible(window, window.mic_title)
    call_notice = visible(window, window.cable_title)
    assert (mic_notice, call_notice) == (step == 1, step == 4)
    if step == 1:
        assert window.steps[0].isAncestorOf(window.mic_title) and window.steps[0].isAncestorOf(window.mic_body)
    if step == 4:
        assert window.steps[3].isAncestorOf(window.cable_title) and not visible(window, window.use_hint)
    if step is not None:
        assert window.subline.text() == f"Step {step} shows what to do."
    if state == "voice-to-speakers":
        assert DEVICES.cable is not None
        assert "Choose" in window.cable_body.text() and DEVICES.cable in window.cable_body.text()
        assert window.focusWidget() is window.output_combo  # no Settings page: the dropdown that fixes it is focused
    if state == "backup-error":
        assert visible(window, window.rec_detail) and window.steps[4].isAncestorOf(window.rec_detail)
    if state == "hear-myself-error":
        assert visible(window, window.dock_note) and window.steps[0].isAncestorOf(window.dock_note)
    if state in ("echo-paused", "echo-unavailable"):
        assert window.echo_desc.property("warn")
    window.close()


def test_t27_a_problem_headline_is_drawn_in_hot_text() -> None:
    for pal in (theme.LIGHT, theme.DARK):
        sheet = app_module.style_sheet(pal)
        assert f'QLabel[role="headline"][tone="hot"] {{ color: {pal.hot_text}; }}' in sheet
        assert f'QLabel[role="headline"][tone="signal"] {{ color: {pal.signal}; }}' in sheet


def test_t27_the_subline_names_only_what_is_on(screen) -> None:
    chain = (fine_tune.new_filter("gain_filter", ()),)
    cases = [
        (Settings(), "natural", "Denoise 70, Natural style"),
        (Settings(strength=55.0, bands=style_bands("warm", 0.5)), "warm", "Denoise 55, Warm style"),
        (Settings(denoise=False), "natural", "Natural style"),
        (Settings(style=False, fine_tune=True, filters=chain), "natural", "Denoise 70, 1 Fine-tune filter"),
        (Settings(fine_tune=True), "natural", "Denoise 70, Natural style, Fine-tune with no filters yet"),
        (Settings(denoise=False, style=False), "natural", NO_CLEANUP),
    ]
    for settings, preset, text in cases:
        window, _ = screen(ON, saved=prefs.Prefs(preset=preset, settings=settings))
        window.tick()
        assert window.subline.text() == text
        window.close()


# ------------------------------------------------------------------ T28 the steps, wired


def test_t28_the_power_button_toggles_cleaning_and_the_tray_follows(screen) -> None:
    window, engine = screen(ON)
    window.tray = tray.Tray(window)
    try:
        window.power_button.click()
        assert engine.sent[-1].cleaning is False and not window.tray.clean_action.isChecked()
        window.tray.clean_action.trigger()
        assert engine.sent[-1].cleaning is True and window.power_button.isChecked()
    finally:
        window.tray.hide()
        window.tray = None


@pytest.mark.parametrize(("name", "switch"), [("denoise", "denoise"), ("style", "style"), ("fine_tune", "fine_tune")])
def test_t28_each_step_2_toggle_sends_its_switch_and_shows_its_outline(screen, name: str, switch: str) -> None:
    window, engine = screen(ON)
    toggle = getattr(window, f"{name}_toggle")
    assert window.steps[1].isAncestorOf(toggle)
    before = toggle.isChecked()
    assert before == getattr(Settings(), switch)
    toggle.click()
    assert getattr(engine.sent[-1], switch) is (not before) and toggle.isChecked() is (not before)
    on = toggle.isChecked()
    assert grab_has(toggle, theme.current().desk) is on  # the 2 px desk outline only when on
    assert grab_has(toggle, theme.current().signal) is on  # and its LED lit
    toggle.click()
    assert getattr(engine.sent[-1], switch) is before


def test_t28_a_rows_controls_are_greyed_but_work_while_its_switch_or_the_power_is_off(screen) -> None:
    window, engine = screen(ON)
    assert not window.strength_bar.dimmed and not window.style_strength_bar.dimmed
    window.denoise_toggle.click()
    assert window.strength_bar.dimmed and window.strength_bar.isEnabled()
    window.strength_bar.setFocus()
    QTest.keyClick(window.strength_bar, Qt.Key.Key_Right)
    assert engine.sent[-1].strength == 71 and engine.sent[-1].denoise is False  # preset while off
    window.power_button.click()
    assert window.style_strength_bar.dimmed and window.style_combo.property("dim") is True
    assert window.style_combo.isEnabled()


def test_t28_strength_scales_the_style_and_is_saved(screen, tmp_path: Path) -> None:
    # Moved from test_presets_page.py: 0.5 is the style, 0 bypasses its EQ, 1 doubles it; a new style keeps it.
    window, engine = screen(ON)
    window.style_combo.activated.emit(list(PRESETS).index("radio"))
    assert engine.sent[-1].bands == style_bands("radio", 0.5)
    assert window.style_strength_bar.value_text() == "0.50"
    window.style_strength_bar.set_value(0)
    window.style_strength_bar.valueChanged.emit(0)
    assert all(b.gain == 0.0 for b in engine.sent[-1].bands)
    window.style_strength_bar.set_value(100)
    window.style_strength_bar.valueChanged.emit(100)
    assert engine.sent[-1].bands == style_bands("radio", 1.0) and window.style_strength_bar.value_text() == "1.00"
    window.style_combo.activated.emit(list(PRESETS).index("warm"))
    assert engine.sent[-1].bands == style_bands("warm", 1.0)
    window.save_prefs()
    saved = prefs.load(tmp_path / "settings.json")
    assert saved.style_strength == 1.0 and saved.preset == "warm"


def test_t28_test_your_mic_and_hear_myself_sit_in_step_1_in_view(screen) -> None:
    # Open question 1: both on one row under the mic dropdown, visible without scrolling at the opening size.
    window, _ = screen(ON)
    window.resize(*WINDOW_SIZE)
    QApplication.processEvents()
    step1 = window.steps[0]
    for widget in (window.input_button, window.test_button, window.monitor_button, window.monitor_volume):
        assert step1.isAncestorOf(widget) and in_view(window, widget)
    assert top(window, window.test_button) == top(window, window.monitor_button)  # one row
    assert top(window, window.test_button) > bottom(window, window.input_button)  # under the dropdown


def test_t28_hear_myself_volume_is_saved_and_reaches_the_engine(screen, tmp_path: Path) -> None:
    # Moved from test_presets_page.py (now one slider, in Step 1).
    window, engine = screen(ON, saved=prefs.Prefs(monitor_volume=30.0))
    assert ("monitor_volume", "30.0") in engine.commands  # sent at launch, with the other saved choices
    window.monitor_volume.set_value(100)
    window.monitor_volume.valueChanged.emit(100)
    assert engine.commands[-1] == ("monitor_volume", "100.0") and engine.monitor_volume == 100.0
    window.save_prefs()
    assert prefs.load(tmp_path / "settings.json").monitor_volume == 100.0


def test_t28_the_output_dropdown_is_in_step_4_and_start_with_windows_in_the_footer(screen) -> None:
    login = FakeLoginItem()
    login.enabled = True  # already on in the system: the switch reads it
    window, _ = screen(ON, login=login)
    assert window.steps[3].isAncestorOf(window.output_combo)
    assert window.footer.isAncestorOf(window.autostart_switch) and window.autostart_switch.isChecked()
    assert window.footer.isAncestorOf(window.updates_view.check_button)
    assert window.version_label.text().startswith("v")


def test_t28_show_plugin_and_the_filter_count(screen) -> None:
    chain = tuple(fine_tune.new_filter(k, ()) for k in ("gain_filter",))
    window, _ = screen(ON, saved=prefs.Prefs(settings=Settings(fine_tune=True, filters=chain)))
    assert window.show_plugin_button.text() == "Show plugin" and window.filters_label.text() == "1 filter"
    window.show_plugin_button.click()  # the filter window comes with Step 10: the stub changes nothing
    assert window.prefs.settings.filters == chain
    window._send(replace(window.prefs.settings, filters=()))
    assert window.filters_label.text() == "No filters yet"


def test_the_mic_dropdown_shows_long_names_on_two_lines(screen) -> None:
    # Moved from test_screen_round4.py.
    import dataclasses

    long_name = "Video Call L + R (Focusrite USB Audio)"
    window, _ = screen(dataclasses.replace(ON, mic=long_name), mics=[long_name, *MICS])
    window.resize(*MIN_SIZE)
    window.tick()
    QApplication.processEvents()
    first, second, cut = window.input_button.lines()
    assert (first + " " + second).strip() == long_name and not cut
    assert window.input_button.toolTip() == long_name


def test_the_backup_is_off_by_default(qtbot: QtBot, tmp_path: Path) -> None:
    # Moved from test_presets_page.py.
    engine = PreviewEngine(ON, MICS, tmp_path, OUTPUTS)
    window = MainWindow(engine, tmp_path / "new.json", FakeLoginItem(), "win32")  # a first launch: no saved file
    qtbot.addWidget(window)
    window.tick()
    assert ("record", "off") in engine.commands and not window.rec_switch.isChecked()
    assert window.rec_label.text() == "Backup off"
    window.close()


# ------------------------------------------------------------------ T29 deleted for real

OLD_FILE = {  # T11's file: a J07/J09 settings.json with the 7-band EQ and the sidebar page
    "preset": "warm",
    "style_strength": 0.8,
    "user_eq": [{"kind": "bell", "freq": 1000.0, "gain": -4.5, "q": 1.0, "on": True, "slope": 12}] * 7,
    "loudness": 62,
    "strength": 35,
    "cleaning": False,
    "echo": True,
    "recording": True,
    "mic": MICS[1],
    "output": OUTPUTS[0],
    "tab": "tune",
    "monitor_volume": 40,
    "bass": 9,
    "mid": -9,
    "treble": 9,
}


def test_t29_the_old_modules_and_widgets_are_gone() -> None:
    with pytest.raises(ImportError):
        importlib.import_module("clean_voice.user_eq")
    old = (
        "UserEqCanvas",
        "BandStrip",
        "BandColumn",
        "BandToggle",
        "TypeButton",
        "TypeMenu",
        "_TypeItem",
        "Knob",
        "DragSpin",
        "paint_type_icon",
        "Chip",
        "PowerSwitch",
        "SignalArrow",
    )
    for name in old:
        assert not hasattr(widgets, name) and not hasattr(app_module, name), name
    assert not hasattr(prefs, "TABS") and not hasattr(Settings(), "user_eq")


def test_t29_the_spectrum_module_is_gone() -> None:
    with pytest.raises(ImportError):
        importlib.import_module("clean_voice.spectrum")


def test_t29_the_live_app_starts_with_an_old_settings_file(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(OLD_FILE), encoding="utf-8")
    engine = PreviewEngine(ON, MICS, tmp_path, OUTPUTS)
    window = MainWindow(engine, path, FakeLoginItem(), "win32")
    qtbot.addWidget(window)
    window.show()
    window.tick()
    sent = engine.sent[0]
    assert sent.bands == style_bands("warm", 0.8) and (sent.loudness, sent.strength) == (62.0, 35.0)
    assert sent.cleaning is False and sent.fine_tune is False and sent.filters == ()
    assert window.style_combo.currentText() == "Warm" and window.style_strength_bar.value() == 80
    assert window.call_volume.value() == 62 and window.strength_bar.value() == 35
    assert not window.power_button.isChecked() and window.echo_switch.isChecked() and window.rec_switch.isChecked()
    assert ("start", MICS[1]) in engine.commands and ("monitor_volume", "40.0") in engine.commands
    window.strength_bar.setFocus()
    QTest.keyClick(window.strength_bar, Qt.Key.Key_Right)  # a change, so the file is written again
    window.close()  # saves: the old keys are not written back
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["strength"] == 36.0
    assert not {"user_eq", "tab", "bass", "mid", "treble"} & set(saved)


def test_t29_the_worker_writes_no_spectrum() -> None:
    # The live spectrum had no view left: the worker no longer copies each block anywhere for it.
    from clean_voice.audio_path import BLOCK, AudioPath
    from clean_voice.dsp import Chain

    path = AudioPath(48_000, 1, 48_000, 2, Chain())
    assert not hasattr(path, "spectrum")  # no tap to write to
    path.input_callback(np.zeros((BLOCK * 3, 1), np.float32), BLOCK * 3, None, None)
    assert path.process() == 3


# ------------------------------------------------------------------ T30 keyboard


def _activate(window: MainWindow, qtbot: QtBot) -> None:
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow, timeout=5000)


def _tab_through(window: MainWindow, start: QWidget, count: int) -> list[QWidget]:
    start.setFocus(Qt.FocusReason.TabFocusReason)
    seen = [start]
    for _ in range(count):
        focused = window.focusWidget()
        assert focused is not None
        QTest.keyClick(focused, Qt.Key.Key_Tab)
        now = window.focusWidget()
        assert now is not None
        seen.append(now)
    return seen


def test_t30_tab_runs_header_then_steps_1_to_5_then_the_footer(screen, qtbot: QtBot) -> None:
    window, _ = screen(ON)
    _activate(window, qtbot)
    chain = [w for w in window.tab_chain() if w.isVisibleTo(window) and w.isEnabled()]
    seen = _tab_through(window, window.power_button, len(chain) - 1)
    assert seen == chain
    owners = [next((i for i, s in enumerate(window.steps) if s.isAncestorOf(w)), None) for w in seen]
    steps = [o for o in owners if o is not None]
    assert steps == sorted(steps) and steps[0] == 0 and steps[-1] == 4
    assert seen[0] is window.power_button and window.footer.isAncestorOf(seen[-1])


def test_t30_the_focus_ring_shows_for_the_keyboard_only(screen, qtbot: QtBot) -> None:
    window, _ = screen(ON)
    _activate(window, qtbot)
    toggle = window.denoise_toggle
    qtbot.mouseClick(toggle, Qt.MouseButton.LeftButton)
    assert toggle.hasFocus() and not keyboard_focus(toggle)
    # The style-sheet buttons and dropdowns take the focus only from the keyboard, so a click draws no ring.
    for widget in (window.show_plugin_button, window.style_combo, window.open_folder_button):
        assert widget.focusPolicy() == Qt.FocusPolicy.TabFocus
    _tab_through(window, window.power_button, 1)  # a real Tab press
    picker = window.input_button
    assert picker.hasFocus() and keyboard_focus(picker)
    ringed = picker.grab().toImage()
    QTest.keyClick(picker, Qt.Key.Key_Tab)
    following = window.focusWidget()
    assert following is not None and following is not picker and keyboard_focus(following)
    assert picker.grab().toImage() != ringed  # the ring was drawn for the keyboard, and goes with the focus
    qtbot.mouseClick(window.style_toggle, Qt.MouseButton.LeftButton)
    assert window.style_toggle.hasFocus() and not keyboard_focus(window.style_toggle)  # a click: no ring again


@pytest.mark.parametrize("key", [Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter])
def test_t30_space_and_enter_work_the_power_button_and_the_toggles(screen, key: Qt.Key) -> None:
    window, engine = screen(ON)
    for button, switch in (
        (window.power_button, "cleaning"),
        (window.denoise_toggle, "denoise"),
        (window.style_toggle, "style"),
        (window.fine_tune_toggle, "fine_tune"),
    ):
        button.setFocus(Qt.FocusReason.TabFocusReason)
        before = getattr(window.prefs.settings, switch)
        QTest.keyClick(button, key)
        assert getattr(engine.sent[-1], switch) is (not before), (button.text(), key)


# ------------------------------------------------------------------ round 13's critique (docs/ui/rounds/13)


def test_r13_the_echo_description_fits_one_line_at_the_opening_size(screen) -> None:
    window, _ = screen(ON)
    window.resize(*WINDOW_SIZE)
    QApplication.processEvents()
    desc = window.echo_desc
    assert desc.fontMetrics().horizontalAdvance(desc.text()) <= desc.width()


def test_r13_hear_myself_volume_has_a_visible_label(screen) -> None:
    window, _ = screen(ON)
    label = window.monitor_volume_label
    assert label.text() == "Volume" and visible(window, label) and window.steps[0].isAncestorOf(label)
    assert top(window, label) <= top(window, window.monitor_volume) + window.monitor_volume.height()


def _sample_window(
    qtbot: QtBot, tmp_path: Path, state: str, size: tuple[int, int] = WINDOW_SIZE, platform: str = "win32"
) -> MainWindow:
    sample = sample_states(DEVICES, platform)[state]
    prefs.save(tmp_path / "s.json", sample.saved())
    window = MainWindow(sample.engine(tmp_path), tmp_path / "s.json", FakeLoginItem(), platform)
    qtbot.addWidget(window)
    window.resize(*size)
    window.show()
    window.tick()
    QApplication.processEvents()
    return window


# States that add a problem notice: the steps scroll to it by design (rounds 13 and 15).
NOTICE_STATES = {
    "cable-missing",
    "cable-lost",
    "voice-to-speakers",
    "no-mic",
    "mic-unplugged",
    "error",
    "backup-error",
    "hear-myself-error",
}
NORMAL_STATES = [s for s in sample_states(DEVICES) if s not in NOTICE_STATES and s != "minimum-size"]


@pytest.mark.parametrize("platform", ["win32", "darwin"])
@pytest.mark.parametrize("state", NORMAL_STATES)
def test_r13_all_five_steps_fit_at_the_opening_size(qtbot: QtBot, tmp_path: Path, state: str, platform: str) -> None:
    # The user's mockup (docs/ui/mockups/J10/3-mockup-one-window.png): every step in view, no scrolling, at the opening
    # size, which stays inside a 1366 x 768 laptop's usable height (768 - 48 px taskbar - about 32 px title bar).
    # J10 r2 F1/m1: on both platforms and in every state without a notice (a wrapped echo note broke it on the Mac).
    assert WINDOW_SIZE[1] <= 688
    window = _sample_window(qtbot, tmp_path, state, platform=platform)
    assert (window.width(), window.height()) == WINDOW_SIZE
    assert window.body.verticalScrollBar().maximum() == 0  # nothing to scroll
    for step in window.steps:
        assert in_view(window, step.title), step.title.text()
        controls = [w for w in step.findChildren(QWidget) if w.isVisibleTo(window) and w.width() and w.height()]
        assert controls, step.title.text()
        for widget in controls:
            assert in_view(window, widget), (step.title.text(), type(widget).__name__, widget.accessibleName())
    footer_top = top(window, window.footer)
    assert bottom(window, window.steps[-1]) < footer_top and footer_top + window.footer.height() == window.height()


def _x_and_width(window: MainWindow, widget: QWidget) -> tuple[int, int]:
    return widget.mapTo(window, widget.rect().topLeft()).x(), widget.width()


@pytest.mark.parametrize("size", [WINDOW_SIZE, MIN_SIZE, (1000, 700)])
def test_r13_the_mic_and_call_meters_share_width_and_scale(qtbot: QtBot, tmp_path: Path, size) -> None:
    window = _sample_window(qtbot, tmp_path, "running", size)
    assert _x_and_width(window, window.in_meter) == _x_and_width(window, window.out_meter)
    assert window.in_meter.tick_labels() == window.out_meter.tick_labels()  # -60 ... 0 at the same x


@pytest.mark.parametrize("size", [WINDOW_SIZE, MIN_SIZE, (1000, 700)])
def test_r13_the_two_strength_sliders_line_up(qtbot: QtBot, tmp_path: Path, size) -> None:
    window = _sample_window(qtbot, tmp_path, "running", size)
    assert _x_and_width(window, window.strength_bar) == _x_and_width(window, window.style_strength_bar)
    assert window.strength_bar.track_rect() == window.style_strength_bar.track_rect()


@pytest.mark.parametrize("state", ["cable-missing", "cable-lost"])
def test_r13_a_step_4_problem_scrolls_step_4_into_view(qtbot: QtBot, tmp_path: Path, state: str) -> None:
    sample = sample_states(DEVICES)[state]
    prefs.save(tmp_path / "s.json", sample.saved())
    window = MainWindow(sample.engine(tmp_path), tmp_path / "s.json", FakeLoginItem(), "win32")
    qtbot.addWidget(window)
    window.show()
    window.tick()
    QApplication.processEvents()
    assert in_view(window, window.cable_title) and in_view(window, window.cable_body)


# ------------------------------------------------------------------ round 15's critique (docs/ui/rounds/15)


def test_r15_a_backup_error_scrolls_step_5_into_view(qtbot: QtBot, tmp_path: Path) -> None:
    # The folder error sits under Step 5, below the fold whenever a note above it takes room: it must be seen.
    window = _sample_window(qtbot, tmp_path, "backup-error")
    assert visible(window, window.rec_detail) and in_view(window, window.rec_detail)


def test_r15_a_backup_error_that_comes_later_scrolls_into_view_too(qtbot: QtBot, tmp_path: Path) -> None:
    import dataclasses

    window = _sample_window(qtbot, tmp_path, "running", MIN_SIZE)
    assert not in_view(window, window.steps[4])
    status = dataclasses.replace(sample_states(DEVICES)["backup-error"].status)
    window.engine.set_status(status)  # type: ignore[attr-defined]
    window.tick()
    qtbot.waitUntil(lambda: in_view(window, window.rec_detail), timeout=2000)


def test_r15_the_mic_name_breaks_before_its_brackets() -> None:
    picker = app_module.DevicePicker()
    picker.setFixedWidth(widgets.StepToggle.W)
    picker.setText("Video Call L + R (Focusrite USB Audio)")
    assert picker.lines() == ("Video Call L + R", "(Focusrite USB Audio)", False)
    picker.setText("Microphone Array on the front of this laptop")  # no brackets: fill the first line
    first, second, cut = picker.lines()
    assert second and not cut and not first.endswith(" ")
    metrics = QFontMetricsF(widgets.ui_font(picker, 13))
    assert metrics.horizontalAdvance(f"{first} {second.split(' ')[0]}") > picker._text_rect().width()
    picker.setText("Mic (A very long interface name that cannot fit on one line at all)")  # bracket too long
    first, second, cut = picker.lines()
    assert first.startswith("Mic (") and cut


@pytest.mark.parametrize("name", ["test_button", "monitor_button"])
def test_r15_the_focus_ring_shows_on_a_filled_button(screen, qtbot: QtBot, name: str) -> None:
    window, _ = screen(ON)
    _activate(window, qtbot)
    button = getattr(window, name)
    if name == "monitor_button":
        button.setChecked(True)  # Hear myself on: filled desk
    window.power_button.setFocus(Qt.FocusReason.TabFocusReason)
    for _ in range(len(window.tab_chain())):  # real Tab presses, as a keyboard guest reaches it
        if window.focusWidget() is button:
            break
        focused = window.focusWidget()
        assert focused is not None
        QTest.keyClick(focused, Qt.Key.Key_Tab)
    assert keyboard_focus(button)
    image = button.grab().toImage()
    edge = image.pixelColor(1, image.height() // 2)
    desk, ink = QColor(theme.current().desk), QColor(theme.current().ink)

    def distance(a: QColor, b: QColor) -> int:
        return abs(a.red() - b.red()) + abs(a.green() - b.green()) + abs(a.blue() - b.blue())

    assert distance(edge, ink) < distance(edge, desk)  # the ring stands out from the fill


def test_r15_a_render_state_shows_the_keyboard_focus() -> None:
    sample = sample_states(DEVICES)["keyboard-focus"]
    assert sample.focus == "test_button"


# ------------------------------------------------------------------ round 16's critique (docs/ui/rounds/16)


def test_r16_the_no_mic_notice_says_what_to_do_without_repeating_the_dropdown(qtbot: QtBot, tmp_path: Path) -> None:
    window = _sample_window(qtbot, tmp_path, "no-mic")
    assert window.input_button.text() == app_module.NO_MIC_TEXT
    assert window.mic_title.text() == "Plug in a microphone" != window.input_button.text()


@pytest.mark.parametrize(
    ("state", "muted"), [("running", False), ("no-mic", True), ("mic-unplugged", True), ("error", True)]
)
def test_r16_the_mic_meter_greys_out_while_there_is_no_mic_to_show(
    qtbot: QtBot, tmp_path: Path, state: str, muted: bool
) -> None:
    window = _sample_window(qtbot, tmp_path, state)
    assert window.in_meter.muted is muted


@pytest.mark.parametrize("state", ["cable-missing", "cable-lost", "voice-to-speakers", "backup-error"])
def test_r16_a_revealed_notice_leaves_whole_steps_at_the_top(qtbot: QtBot, tmp_path: Path, state: str) -> None:
    window = _sample_window(qtbot, tmp_path, state)
    qtbot.wait(50)  # the reveal runs once the window is shown
    value = window.body.verticalScrollBar().value()
    tops = [step.y() for step in window.steps]
    for step in window.steps:  # and each row of controls in it
        items = [step.body.itemAt(i) for i in range(step.body.count())]
        tops += [step.y() + item.geometry().top() for item in items if item is not None and not item.isEmpty()]
    # The view starts at the top or just above a step or a row, never through one.
    assert value == 0 or any(t - 8 <= value <= t for t in tops), (value, sorted(tops))
    notice = window.rec_detail if state == "backup-error" else window.cable_body
    assert in_view(window, notice)
