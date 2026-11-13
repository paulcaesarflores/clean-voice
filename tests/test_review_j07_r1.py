"""J07 review round 1 (docs/reviews/J07-main-screen-r1.md): each finding as a permanent test, made from the reviewer's
probes in scratch/review/J07-r1/. Each one failed before its fix."""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel
from pytestqt.qtbot import QtBot
from test_engine import CABLE, MIC, FakeBackend
from test_main_screen import CABLE_NAME, MICS, ON, OUTPUTS, FakeLoginItem, make  # noqa: F401

from clean_voice import prefs, tray
from clean_voice.app import HEAD_NO_CALL, HEAD_NO_MIC, HIDDEN_TICK_MS, TICK_MS, MainWindow, show_window
from clean_voice.engine import Engine, State, Status
from clean_voice.preview import PreviewEngine

ROOT = Path(__file__).resolve().parents[1]
SPEAKERS = OUTPUTS[1]


@pytest.fixture
def screen(request: pytest.FixtureRequest):
    """test_main_screen's window factory, under this file's own name."""
    return request.getfixturevalue("make")


# ------------------------------------------------------------------ MAJOR


def test_f1_a_minimized_start_runs_the_slow_tick(qtbot: QtBot, tmp_path: Path) -> None:
    # The plan: the tick slows to 500 ms while the window is hidden. A --minimized start (what the login item runs)
    # is never shown, so it was never hidden either, and ticked every 33 ms all day.
    window = MainWindow(PreviewEngine(ON, MICS, tmp_path, OUTPUTS), tmp_path / "s.json", login_item=FakeLoginItem())
    qtbot.addWidget(window)
    window.tray = tray.Tray(window)
    try:
        show_window(window, minimized=True)  # as main() does for --minimized with a tray
        assert not window.isVisible() and window._timer.interval() == HIDDEN_TICK_MS
        window.show_from_tray()
        assert window._timer.interval() == TICK_MS
        window.hide()  # closed to the tray
        assert window._timer.interval() == HIDDEN_TICK_MS
    finally:
        window.tray.hide()


def test_f2_sending_the_voice_to_the_speakers_says_what_to_do(screen) -> None:
    # J10: there is no Settings page any more; Step 4 says it and its dropdown, the fix, has the focus.
    window, _ = screen(dataclasses.replace(ON, output=SPEAKERS))
    window.tick()
    assert window.headline.text() == HEAD_NO_CALL
    step4 = [w.text() for w in window.steps[3].findChildren(QLabel) if w.isVisibleTo(window)]
    told = [t for t in step4 if "Choose" in t and CABLE_NAME in t]
    assert told, step4
    assert not window.use_hint.isVisibleTo(window)  # not the usual "Pick CABLE Output" advice: that isn't the problem
    assert window.focusWidget() is window.output_combo


def test_f3_a_change_is_saved_shortly_after_without_closing(screen, qtbot: QtBot, tmp_path: Path) -> None:
    # "Written 500 ms after the last change": a crash later keeps the change.
    window, _ = screen(ON)
    window.strength_bar.setFocus()
    QTest.keyClick(window.strength_bar, Qt.Key.Key_Right)
    qtbot.waitUntil(lambda: prefs.load(tmp_path / "settings.json").settings.strength == 71, timeout=2000)
    assert window.isVisible()


def test_f4_the_mac_microphone_prompt_names_any_call_app() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    text = config["tool"]["briefcase"]["app"]["clean-voice"]["permission"]["microphone"]
    assert "Zoom" in text and "Meet" in text, text


# ------------------------------------------------------------------ MINOR


def test_m1_escape_closes_the_menu_and_the_picker_with_it(screen) -> None:
    window, _ = screen(ON)
    window.open_mic_menu()
    current = next(i for i in window.mic_menu.items if i.isChecked())  # the item the menu gives the focus to
    QTest.keyClick(current, Qt.Key.Key_Escape)  # an item ignores Escape, so it reaches the menu, as when focused
    assert not window.mic_menu.isVisible() and not window.input_button.expanded


def test_m2_a_device_that_will_not_open_is_not_blamed_on_the_microphone(qtbot: QtBot, tmp_path: Path) -> None:
    # The real engine on the fake sound system: the cable's stream is the first one backend.open opens.
    backend = FakeBackend([MIC, CABLE])
    backend.fail_open = "Error opening OutputStream: Device unavailable [PaErrorCode -9985]"
    engine = Engine(tmp_path / "Recordings", backend, retry_s=0.2, watchdog_s=0.3)
    window = MainWindow(engine, tmp_path / "s.json", login_item=FakeLoginItem(), platform="win32")
    qtbot.addWidget(window)
    window.show()
    try:
        qtbot.waitUntil(lambda: engine.status().state is State.ERROR, timeout=10_000)
        window.tick()
        assert window.headline.text() != HEAD_NO_MIC
        shown = f"{window.headline.text()} {window.mic_title.text()} {window.mic_body.text()}"
        assert "OutputStream" not in shown and "PaErrorCode" not in shown  # no raw sound-system text on screen
        assert "OutputStream" in window.mic_body.toolTip()  # the details stay one hover away
    finally:
        window.close()
        engine.close()


def test_m4_the_backup_dot_is_steady(screen) -> None:
    # DIRECTION.md v2, Motion: nothing animates on its own except the meters.
    window, _ = screen(ON)
    window.tick()
    first = window.rec_dot.grab().toImage()
    for _ in range(10):
        window.tick()
    assert window.rec_dot.grab().toImage() == first


def test_m6_the_tray_does_not_say_recording_while_the_backup_is_paused(screen) -> None:
    window, _ = screen(dataclasses.replace(ON, state=State.MIC_LOST, level=0.0, input_level=0.0))
    window.tray = tray.Tray(window)
    try:
        window.tick()
        assert window.rec_label.text() == "Backup paused"
        assert "recording" not in window.tray.tooltip(), window.tray.tooltip()
    finally:
        window.tray.hide()
        window.tray = None


def test_m7_reopening_the_app_on_a_mac_shows_the_hidden_window(qtbot: QtBot, tmp_path: Path) -> None:
    # macOS starts no second process when the guest opens Clean Voice again from Finder or the Dock: it only
    # activates the running app, so the window must come back on that activation.
    window = MainWindow(PreviewEngine(ON, MICS, tmp_path, OUTPUTS), tmp_path / "s.json", FakeLoginItem(), "darwin")
    qtbot.addWidget(window)
    window.tray = tray.Tray(window)
    try:
        show_window(window, minimized=True)
        assert not window.isVisible()
        app = QApplication.instance()
        assert app is not None
        QApplication.sendEvent(app, QEvent(QEvent.Type.ApplicationActivate))
        assert window.isVisible()
    finally:
        window.tray.hide()


def test_m8_send_to_shows_nothing_chosen_while_there_is_no_cable(screen) -> None:
    speakers_first = [OUTPUTS[1], OUTPUTS[2]]  # no cable installed, nothing saved
    window, _ = screen(Status(state=State.NO_CABLE, mic=MICS[0]), outputs=speakers_first)
    window.tick()
    assert window.output_combo.currentText() not in speakers_first
    assert window.output_combo.placeholderText() == "VB-CABLE (not installed)"


# ------------------------------------------------------------------ J07 review round 2 (J07-main-screen-r2.md)


def test_n2_the_samples_show_the_voice_on_the_speakers_and_a_lost_cable() -> None:
    # The renders, the cut-off sweep and the clone check iterate sample_states, so these states must be in it.
    from test_main_screen import DEVICES

    from clean_voice.app import describe
    from clean_voice.preview import sample_states

    samples = sample_states(DEVICES)
    speakers = describe(samples["voice-to-speakers"].status)
    assert samples["voice-to-speakers"].status.output == OUTPUTS[1]
    assert speakers.notice is not None and speakers.notice.kind == "output"
    assert samples["cable-lost"].status.state is State.CABLE_LOST


REALTEK = "Microphone (Realtek(R) Audio)"


def test_n3_a_common_mic_name_fits_on_two_lines_at_the_minimum_size(qtbot: QtBot, tmp_path: Path) -> None:
    # In every state at the window's minimum, the picker shows a usual Windows name in full on its two lines.
    from test_main_screen import DEVICES

    from clean_voice.app import MIN_SIZE
    from clean_voice.preview import sample_states

    cut = []
    for state, sample in sample_states(DEVICES).items():
        if not sample.status.mic or not sample.mics:
            continue
        status = dataclasses.replace(sample.status, mic=REALTEK)
        path = tmp_path / f"{state}.json"
        prefs.save(path, sample.saved())
        engine = PreviewEngine(status, [REALTEK, *MICS], tmp_path, list(sample.outputs))
        window = MainWindow(engine, path, FakeLoginItem(), "win32")
        qtbot.addWidget(window)
        window.resize(*MIN_SIZE)
        window.show()
        window.tick()
        QApplication.processEvents()
        if window.input_button.lines()[2]:
            cut.append(state)
        window.close()
    assert not cut, cut


def test_n4_the_call_meter_is_never_green_while_the_call_cannot_hear(screen) -> None:
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QWidget

    from clean_voice import theme

    signal = QColor(theme.current().signal)

    def greens(widget: QWidget) -> int:
        image = widget.grab().toImage()
        count = 0
        for x in range(0, image.width(), 2):
            for y in range(image.height()):
                c = image.pixelColor(x, y)
                count += (
                    abs(c.red() - signal.red()) + abs(c.green() - signal.green()) + abs(c.blue() - signal.blue()) < 24
                )
        return count

    window, engine = screen(dataclasses.replace(ON, output=SPEAKERS))
    for _ in range(3):
        window.tick()
    assert not window.header.lit
    assert window.out_meter.fraction > 0  # the voice is there (on the speakers) ...
    assert greens(window.out_meter) == 0  # ... but nothing says the call hears it
    engine.set_status(ON)  # back to the cable: green again
    for _ in range(3):
        window.tick()
    assert greens(window.out_meter) > 0
