"""The window's custom widgets behave like the original's controls (J07 W1-W3; J10's PowerButton and StepToggle) and
the style EQ's curve is the filter that runs (W4). The meter is in test_meters.py."""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QLineEdit, QWidget
from pytestqt.qtbot import QtBot
from scipy import signal

from clean_voice import theme
from clean_voice.dsp import SAMPLE_RATE, Equalizer, eq_response_db
from clean_voice.settings import Band, bands_for
from clean_voice.widgets import BarSlider, IconButton, PowerButton, StepToggle, ToggleSwitch
from clean_voice.widgets import TestButton as MicTestButton  # not a pytest class

# tests/conftest.py sets QT_QPA_PLATFORM=offscreen.


def show(qtbot: QtBot, widget: QWidget, width: int, height: int) -> None:
    qtbot.addWidget(widget)
    widget.resize(width, height)
    widget.show()
    qtbot.waitExposed(widget)


def mouse(
    widget: QWidget, kind: QEvent.Type, pos: QPointF, buttons: Qt.MouseButton = Qt.MouseButton.LeftButton
) -> None:
    """Send one mouse event straight to the widget (QTest's moves carry no button state)."""
    button = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton
    held = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else buttons
    event = QMouseEvent(kind, pos, widget.mapToGlobal(pos), button, held, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, event)


# ------------------------------------------------------------------ W1: BarSlider


def make_bar(qtbot: QtBot, value: int = 50, reset: int = 50) -> tuple[BarSlider, list[int]]:
    bar = BarSlider("Voice volume", value=value, reset=reset)
    show(qtbot, bar, 400, 48)
    got: list[int] = []
    bar.valueChanged.connect(got.append)
    return bar, got


def at(bar: BarSlider, fraction: float) -> QPoint:
    track = bar.track_rect()
    return QPoint(round(track.left() + fraction * track.width()), round(track.center().y()))


def test_w1_a_click_anywhere_sets_the_value(qtbot: QtBot) -> None:
    bar, got = make_bar(qtbot)
    qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=at(bar, 0.25))
    assert bar.value() == 25
    assert got == [25]
    qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=at(bar, 0.9))
    assert bar.value() == 90
    assert got == [25, 90]


def test_w1_a_click_beyond_the_ends_clamps(qtbot: QtBot) -> None:
    bar, got = make_bar(qtbot)
    track = bar.track_rect()
    qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=QPoint(round(track.right()) + 2, 24))  # past the track end
    assert bar.value() == 100
    qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=QPoint(0, 24))
    assert bar.value() == 0
    assert got == [100, 0]


def test_w1_drag(qtbot: QtBot) -> None:
    bar, got = make_bar(qtbot)
    mouse(bar, QEvent.Type.MouseButtonPress, QPointF(at(bar, 0.1)))
    mouse(bar, QEvent.Type.MouseMove, QPointF(at(bar, 0.4)))
    mouse(bar, QEvent.Type.MouseMove, QPointF(at(bar, 0.6)))
    mouse(bar, QEvent.Type.MouseButtonRelease, QPointF(at(bar, 0.6)))
    assert got == [10, 40, 60]
    mouse(bar, QEvent.Type.MouseMove, QPointF(at(bar, 0.2)), Qt.MouseButton.NoButton)  # released: no more drag
    assert bar.value() == 60


def test_w1_keys(qtbot: QtBot) -> None:
    bar, got = make_bar(qtbot)
    qtbot.keyClick(bar, Qt.Key.Key_Right)
    qtbot.keyClick(bar, Qt.Key.Key_Up)
    assert bar.value() == 52
    qtbot.keyClick(bar, Qt.Key.Key_Left)
    qtbot.keyClick(bar, Qt.Key.Key_Down)
    assert bar.value() == 50
    qtbot.keyClick(bar, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert bar.value() == 60
    qtbot.keyClick(bar, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    qtbot.keyClick(bar, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
    assert bar.value() == 40
    qtbot.keyClick(bar, Qt.Key.Key_Home)
    assert bar.value() == 0
    qtbot.keyClick(bar, Qt.Key.Key_End)
    assert bar.value() == 100
    assert got == [51, 52, 51, 50, 60, 50, 40, 0, 100]


def test_w1_double_click_resets(qtbot: QtBot) -> None:
    bar, got = make_bar(qtbot, value=80, reset=70)
    qtbot.mouseDClick(bar, Qt.MouseButton.LeftButton, pos=at(bar, 0.25))
    assert bar.value() == 70
    assert got[-1] == 70


def test_w1_no_signal_when_the_value_is_unchanged(qtbot: QtBot) -> None:
    bar, got = make_bar(qtbot)
    qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=at(bar, 0.5))  # already 50
    qtbot.keyClick(bar, Qt.Key.Key_End)
    qtbot.keyClick(bar, Qt.Key.Key_End)
    qtbot.keyClick(bar, Qt.Key.Key_Right)  # at 100 already
    qtbot.keyClick(bar, Qt.Key.Key_Home)
    qtbot.keyClick(bar, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)  # at 0 already
    assert got == [100, 0]
    bar.set_value(30)  # the screen moving the bar (the EQ changed the tone) is not the guest's change
    assert bar.value() == 30
    assert got == [100, 0]
    qtbot.mouseDClick(bar, Qt.MouseButton.LeftButton, pos=at(bar, 0.5))  # press lands on 50, the reset value
    assert got == [100, 0, 50]


def test_w1_value_box(qtbot: QtBot) -> None:
    bar = BarSlider("Strength", value=50, reset=50)
    show(qtbot, bar, 400, 48)
    assert bar.value_text() == "50"
    bar.set_value(71.6)
    assert bar.value() == 72
    assert bar.value_text() == "72"
    assert bar.track_rect().right() < bar.width() - 50  # the value box sits on the right, outside the track
    bar.grab()  # paints without error


def test_w1_the_label_row_does_not_look_clickable(qtbot: QtBot) -> None:
    """home-gain-labels r2 m1: the hand and the grown handle only where a press acts (below the label row); a drag
    from the track keeps both over the label row. The handle is judged by its pixels."""
    page = QWidget()
    bar = BarSlider("Voice volume", value=50, reset=50, parent=page)
    bar.setGeometry(0, 0, 400, bar.height())
    show(qtbot, page, 400, 200)
    hx = round(bar.track_rect().center().x())  # the handle's x at 50, so a press there keeps the value
    label, track = QPoint(hx, 10), QPoint(hx, BarSlider.LABEL_H + 12)
    qtbot.mouseMove(page, QPoint(200, 150))  # the pointer on the page, off the slider
    assert not bar.underMouse()
    idle = bar.grab().toImage()
    hand, arrow = Qt.CursorShape.PointingHandCursor, Qt.CursorShape.ArrowCursor

    qtbot.mouseMove(bar, label)
    assert bar.cursor().shape() == arrow
    assert bar.grab().toImage() == idle  # the normal handle

    qtbot.mouseMove(bar, track)
    assert bar.cursor().shape() == hand
    grown = bar.grab().toImage()
    assert grown != idle

    qtbot.mouseMove(bar, label)
    assert bar.cursor().shape() == arrow
    assert bar.grab().toImage() == idle

    # A drag that starts on the track keeps its cursor and grown handle over the label row, until the release.
    qtbot.mousePress(bar, Qt.MouseButton.LeftButton, pos=track)
    qtbot.mouseMove(bar, label)
    assert bar.dragging
    assert bar.cursor().shape() == hand
    assert bar.grab().toImage() == grown
    qtbot.mouseRelease(bar, Qt.MouseButton.LeftButton, pos=label)
    assert bar.cursor().shape() == arrow
    assert bar.grab().toImage() == idle


def test_w1_compact_and_disabled_sliders_keep_the_hand(qtbot: QtBot) -> None:
    compact = BarSlider("Strength", compact=True)
    show(qtbot, compact, 300, compact.height())
    assert compact.cursor().shape() == Qt.CursorShape.PointingHandCursor
    qtbot.mouseMove(compact, QPoint(150, 1))
    assert compact.handle_grown
    bar = BarSlider("Mic level")
    show(qtbot, bar, 400, bar.height())
    bar.setEnabled(False)
    qtbot.mouseMove(bar, QPoint(200, 10))
    assert bar.cursor().shape() == Qt.CursorShape.PointingHandCursor
    qtbot.mouseMove(bar, QPoint(200, BarSlider.LABEL_H + 12))
    assert not bar.handle_grown


def test_w1_dimmed_is_greyed_but_still_works(qtbot: QtBot) -> None:
    # Step 2's rows while their switch is off: drawn like a disabled slider, and still set by the guest.
    bar, got = make_bar(qtbot)
    lit = bar.grab().toImage()
    bar.set_dimmed(True)
    assert bar.isEnabled() and bar.grab().toImage() != lit
    qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=at(bar, 0.25))
    assert got == [25]


def test_w1_the_value_box_width_is_the_callers(qtbot: QtBot) -> None:
    bar = BarSlider("Voice volume", value_width=70, formatter=lambda v: "-12.0 dB")
    show(qtbot, bar, 300, bar.height())
    assert bar.VALUE_W == 70 and not bar.cut_off()
    narrow = BarSlider("Voice volume", compact=True, value_width=20, formatter=lambda v: "-12.0 dB")
    show(qtbot, narrow, 300, narrow.height())
    assert narrow.cut_off()  # a readout wider than its box counts as cut off


def value_click(bar: BarSlider) -> QPoint:
    return QPoint(bar.width() - bar.VALUE_W // 2, bar.LABEL_H + bar.TRACK_H // 2)


def editor_of(bar: BarSlider) -> QLineEdit:
    field = bar.findChild(QLineEdit)
    assert field is not None
    return field


def test_typed_value_click_opens_a_field_with_the_value_and_enter_applies(qtbot: QtBot) -> None:
    bar, got = make_bar(qtbot, value=50)
    qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=value_click(bar))
    assert bar.value() == 50 and bar.editing()  # the click on the value is not a drag to 100
    field = editor_of(bar)
    assert field.text() == "50"
    field.setText("73")
    qtbot.keyClick(field, Qt.Key.Key_Return)
    assert (bar.value(), got, bar.editing()) == (73, [73], False)


def test_typed_value_clamps_and_leaving_the_field_applies(qtbot: QtBot) -> None:
    bar, got = make_bar(qtbot, value=50)
    bar.activateWindow()
    qtbot.waitUntil(bar.isActiveWindow)
    bar.begin_edit()
    editor_of(bar).setText("250")
    bar.setFocus()  # leaving the field
    qtbot.waitUntil(lambda: not bar.editing())
    assert (bar.value(), got) == (100, [100])


def test_typed_value_escape_and_bad_text_leave_it_unchanged(qtbot: QtBot) -> None:
    bar, got = make_bar(qtbot, value=50)
    bar.begin_edit()
    editor_of(bar).setText("12")
    qtbot.keyClick(editor_of(bar), Qt.Key.Key_Escape)
    assert (bar.value(), got, bar.editing()) == (50, [], False)
    bar.begin_edit()
    editor_of(bar).setText("loud")
    qtbot.keyClick(editor_of(bar), Qt.Key.Key_Return)
    assert (bar.value(), got, bar.editing()) == (50, [], False)


def test_typed_value_keys_open_it_and_a_disabled_slider_does_not(qtbot: QtBot) -> None:
    bar, _ = make_bar(qtbot)
    bar.setFocus()
    qtbot.keyClick(bar, Qt.Key.Key_F2)
    assert bar.editing()
    qtbot.keyClick(editor_of(bar), Qt.Key.Key_Escape)
    qtbot.keyClick(bar, Qt.Key.Key_Return)
    assert bar.editing()
    qtbot.keyClick(editor_of(bar), Qt.Key.Key_Escape)
    bar.setEnabled(False)
    qtbot.mouseClick(bar, Qt.MouseButton.LeftButton, pos=value_click(bar))
    assert not bar.editing()


def test_typed_value_keeps_double_click_reset_on_the_track(qtbot: QtBot) -> None:
    bar, _ = make_bar(qtbot, value=20, reset=50)
    qtbot.mouseDClick(bar, Qt.MouseButton.LeftButton, pos=at(bar, 0.5))
    assert bar.value() == 50 and not bar.editing()


# ------------------------------------------------------------------ W3: PowerButton, StepToggle, ToggleSwitch


@pytest.mark.parametrize("kind", [PowerButton, ToggleSwitch, lambda: StepToggle("Denoise")])
def test_w3_toggles_on_click_space_and_enter(qtbot: QtBot, kind) -> None:
    button = kind()
    qtbot.addWidget(button)
    button.show()
    qtbot.waitExposed(button)
    toggled: list[bool] = []
    button.toggled.connect(toggled.append)
    assert button.isCheckable()
    start = button.isChecked()
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert button.isChecked() is (not start)
    qtbot.keyClick(button, Qt.Key.Key_Space)
    assert button.isChecked() is start
    qtbot.keyClick(button, Qt.Key.Key_Return)
    assert button.isChecked() is (not start)
    assert toggled == [not start, start, not start]
    button.grab()


def test_w3_the_power_ring_takes_its_tone_over_150_ms(qtbot: QtBot) -> None:
    button = PowerButton()
    qtbot.addWidget(button)
    button.show()
    qtbot.waitExposed(button)
    pal = theme.current()
    assert button.tone == "line" and button.ring_color().name().upper() == pal.line.upper()
    button.set_tone("signal")
    assert button.tone == "signal" and button.ring_color().name().upper() != pal.signal.upper()  # it fades in...
    qtbot.wait(PowerButton.FADE_MS + 100)
    assert button.ring_color().name().upper() == pal.signal.upper()  # ...and lands on signal
    hidden = PowerButton()
    hidden.set_tone("hot")  # not shown: no fade, the ring is hot at once
    assert hidden.ring_color().name().upper() == pal.hot.upper()
    qtbot.addWidget(hidden)


def test_w3_a_step_toggle_shows_its_state_by_outline_and_led(qtbot: QtBot) -> None:
    toggle = StepToggle("Voice style")
    qtbot.addWidget(toggle)
    toggle.resize(toggle.sizeHint())
    off = toggle.grab().toImage()
    toggle.setChecked(True)
    on = toggle.grab().toImage()
    pal = theme.current()
    led = (StepToggle.PAD + 18, toggle.height() // 2)  # the LED's centre
    assert on.pixelColor(*led).name().upper() == pal.signal.upper()
    assert off.pixelColor(*led).name().upper() == pal.line.upper()
    edge = (toggle.width() // 2, StepToggle.PAD)  # the outline's top edge
    assert on.pixelColor(*edge).name().upper() == pal.desk.upper()
    assert not toggle.cut_off()


def test_other_widgets_paint(qtbot: QtBot) -> None:
    round_button = IconButton("headphones", "Hear myself", text="Hear myself")
    test = MicTestButton()
    for w in (round_button, test):
        qtbot.addWidget(w)
        w.grab()
    assert round_button.toolTip() == "Hear myself"
    test.set_phase("Speak now  4 s", True, 0.3)
    assert test.running and test.progress == 0.3


# ------------------------------------------------------------------ W4: the EQ curve


FREQS = np.geomspace(20.0, 20_000.0, 300)


def test_w4_flat_bands_are_0_db() -> None:
    assert np.allclose(eq_response_db(bands_for("flat"), FREQS), 0.0, atol=0.01)


def test_w4_a_6_db_bell_peaks_at_its_frequency() -> None:
    bands = list(bands_for("flat"))
    bands[2] = Band(1000.0, 6.0, 1.0)
    db = eq_response_db(tuple(bands), np.array([1000.0, 20.0, 20_000.0]))
    assert abs(db[0] - 6.0) <= 0.1
    assert abs(db[1]) < 0.1 and abs(db[2]) < 0.1


def test_w4_natural_is_the_filter_that_runs() -> None:
    eq = Equalizer(bands_for("natural"))
    _, h = signal.sosfreqz(eq._sos, worN=FREQS, fs=SAMPLE_RATE)
    expected = 20.0 * np.log10(np.abs(h))
    assert np.allclose(eq_response_db(bands_for("natural"), FREQS), expected, atol=1e-9)
