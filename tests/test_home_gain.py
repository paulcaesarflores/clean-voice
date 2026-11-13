"""The two gain sliders (DESIGN "Home gain sliders", 2026-11-12; J10 moved them): the mic's own OS level in Step 1,
beside the mic's meter, and Voice volume in Step 3, beside the call's meter (round 13: both rows alike, so the two
meters share their width and scale)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from pytestqt.qtbot import QtBot
from test_main_screen import DEVICES, MICS, ON, OUTPUTS, FakeLoginItem

from clean_voice import prefs
from clean_voice.app import CALL_VOLUME_TIP, MIC_LEVEL_FIXED, MIC_LEVEL_TIP, MainWindow
from clean_voice.mic_level import RETRY_S, SystemMicLevel
from clean_voice.preview import PreviewEngine, sample_states
from clean_voice.settings import Settings


class FakeMicLevel:
    """Each mic's OS level (None: it has none), and every level the window set."""

    def __init__(self, levels: dict[str, float | None]) -> None:
        self.levels = dict(levels)
        self.sets: list[tuple[str, float]] = []
        self.fail = False
        self.closed = False

    def level(self, mic: str) -> float | None:
        if self.fail:
            raise OSError("the sound system failed")
        return self.levels.get(mic)

    def set_level(self, mic: str, level: float) -> bool:
        self.sets.append((mic, level))
        if self.levels.get(mic) is None:
            return False
        self.levels[mic] = level
        return True

    def level_db(self, mic: str) -> float | None:
        return None  # no dB: the slider shows its percent (the dB readout is in test_meters.py)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def home(qtbot: QtBot, tmp_path: Path):
    def make(
        levels: dict[str, float | None] | None = None, saved: prefs.Prefs | None = None
    ) -> tuple[MainWindow, PreviewEngine, FakeMicLevel]:
        path = tmp_path / "settings.json"
        prefs.save(path, saved or prefs.Prefs())
        engine = PreviewEngine(ON, MICS, tmp_path / "Recordings", OUTPUTS)
        fake = FakeMicLevel({MICS[0]: 0.6, MICS[1]: 0.25} if levels is None else levels)
        window = MainWindow(engine, path, login_item=FakeLoginItem(), platform="win32", mic_levels=fake)
        qtbot.addWidget(window)
        window.show()
        qtbot.waitExposed(window)
        return window, engine, fake

    return make


def _beside(window: MainWindow, meter, slider) -> bool:
    """``slider`` to the right of ``meter``, on the same rows."""
    right = meter.mapTo(window.root, meter.rect().topRight()).x()
    left = slider.mapTo(window.root, slider.rect().topLeft())
    meter_top = meter.mapTo(window.root, meter.rect().topLeft()).y()
    return left.x() > right and left.y() <= meter_top <= left.y() + slider.height()


# ------------------------------------------------------------------ the visible labels (user decision, 2026-11-12)


@pytest.mark.parametrize("levels", [None, {}], ids=["adjustable", "greyed"])
def test_both_sliders_show_their_label_and_a_screen_reader_hears_it(home, levels) -> None:
    """Each slider shows its name above its track, the same name a screen reader hears; greyed, the name stays."""
    window, _, _ = home(levels)
    for slider, name in ((window.mic_gain, "Mic level"), (window.call_volume, "Voice volume")):
        assert slider.LABEL_H > 0, f"{name}: no visible label row"
        assert slider.label == name and slider.accessibleName() == name
        assert not slider.cut_off(), f"{name}: the label does not fit"
    assert _beside(window, window.in_meter, window.mic_gain)
    assert window.mic_gain.isEnabled() is (levels is None)
    assert window.mic_gain.toolTip() == (MIC_LEVEL_TIP if levels is None else MIC_LEVEL_FIXED)
    assert window.call_volume.toolTip() == CALL_VOLUME_TIP


# ------------------------------------------------------------------ under Your mic: the mic's own OS level


def test_the_mic_slider_sits_beside_the_mic_meter_and_reads_the_level(home) -> None:
    window, _, _ = home()
    slider = window.mic_gain
    assert window.steps[0].isAncestorOf(slider) and _beside(window, window.in_meter, slider)
    assert slider.isEnabled() and slider.value() == 60 and slider.value_text() == "60 %"
    assert slider.toolTip() == MIC_LEVEL_TIP


def test_moving_the_mic_slider_sets_the_selected_mics_level(home, qtbot: QtBot) -> None:
    window, engine, fake = home()
    before = len(engine.sent)
    window.mic_gain.setFocus()
    qtbot.keyClick(window.mic_gain, Qt.Key.Key_Right)
    assert fake.sets == [(MICS[0], pytest.approx(0.61))]
    assert len(engine.sent) == before  # Voice volume untouched: no engine update at all
    window.tick()
    assert window.mic_gain.value() == 61


def _press_at(qtbot: QtBot, slider, fraction: float) -> QPoint:
    pos = QPoint(int(slider.track_rect().left() + slider.track_rect().width() * fraction), slider.height() // 2)
    qtbot.mousePress(slider, Qt.MouseButton.LeftButton, pos=pos)
    return pos


def test_the_tick_does_not_move_the_handle_while_the_guest_drags(home, qtbot: QtBot) -> None:
    """The tick reads the OS level every 33 ms; while the guest holds the handle it must not write it back."""
    window, _, fake = home()
    slider = window.mic_gain
    pos = _press_at(qtbot, slider, 0.3)
    pressed = slider.value()  # about 30: the press lands on a whole pixel of the track
    assert slider.dragging and abs(pressed - 30) <= 2
    fake.levels[MICS[0]] = 0.9  # the OS reports something else meanwhile (a stepped driver, a late apply)
    window.tick()
    assert slider.value() == pressed, "the tick moved the handle under the guest's finger"
    qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=pos)
    window.tick()
    assert slider.value() == 90  # released: the slider follows the OS again


def test_unplugging_mid_drag_does_not_leave_the_slider_dragging(home, qtbot: QtBot) -> None:
    """A greyed slider never gets the release, so greying it out ends the drag; after a replug it follows the OS."""
    window, _, fake = home()
    slider = window.mic_gain
    pos = _press_at(qtbot, slider, 0.3)
    fake.levels[MICS[0]] = None  # unplugged mid-drag
    window.tick()
    assert not slider.isEnabled() and not slider.dragging
    qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=pos)
    fake.levels[MICS[0]] = 0.7  # plugged back in, at 70 in the system settings
    window.tick()
    assert slider.isEnabled() and not slider.dragging and slider.value() == 70


def test_a_mic_with_no_level_greys_the_slider_out(home, qtbot: QtBot) -> None:
    window, _, fake = home({MICS[0]: None})
    slider = window.mic_gain
    assert not slider.isEnabled()
    assert slider.toolTip() == MIC_LEVEL_FIXED and slider.value_text() == "–"
    assert slider.accessibleDescription() == "–"
    qtbot.mouseClick(slider, Qt.MouseButton.LeftButton, pos=QPoint(slider.width() - 60, slider.height() // 2))
    assert fake.sets == []


def test_a_greyed_slider_keeps_nothing_of_the_previous_mic(home) -> None:
    """Switching to a mic with no level: the handle goes to the start and the screen reader hears "–", as shown."""
    window, _, _ = home({MICS[0]: 0.6, MICS[2]: None})
    slider = window.mic_gain
    assert slider.accessibleDescription() == "60 %"
    window.choose_mic(MICS[2])
    window.tick()
    assert not slider.isEnabled() and slider.value() == 0
    assert slider.value_text() == "–" and slider.accessibleDescription() == "–"
    window.choose_mic(MICS[0])
    window.tick()
    assert slider.value() == 60 and slider.accessibleDescription() == "60 %"


def test_a_failing_read_greys_the_slider_out_without_crashing(home) -> None:
    window, _, fake = home()
    assert window.mic_gain.isEnabled()
    fake.fail = True
    window.tick()
    assert not window.mic_gain.isEnabled() and window.mic_gain.toolTip() == MIC_LEVEL_FIXED


def test_the_mic_slider_follows_a_change_in_the_system_settings(home) -> None:
    window, _, fake = home()
    fake.levels[MICS[0]] = 0.33
    window.tick()
    assert window.mic_gain.value() == 33


def test_the_mic_slider_follows_the_selected_mic(home, qtbot: QtBot) -> None:
    window, _, fake = home({MICS[0]: 0.6, MICS[1]: 0.25, MICS[2]: None})
    window.choose_mic(MICS[1])
    window.tick()
    assert window.mic_gain.isEnabled() and window.mic_gain.value() == 25
    window.mic_gain.setFocus()
    qtbot.keyClick(window.mic_gain, Qt.Key.Key_Left)
    assert fake.sets == [(MICS[1], pytest.approx(0.24))]
    window.choose_mic(MICS[2])
    window.tick()
    assert not window.mic_gain.isEnabled()


def test_the_mic_level_is_not_saved_and_the_control_is_closed_on_quit(home, tmp_path: Path, qtbot: QtBot) -> None:
    window, _, fake = home()
    window.mic_gain.setFocus()
    qtbot.keyClick(window.mic_gain, Qt.Key.Key_Right)
    window.save_prefs()
    assert prefs.load(tmp_path / "settings.json") == prefs.Prefs()  # nothing new to save: the OS keeps the level
    window.close()
    assert fake.closed


def _label_row(slider, fraction: float) -> QPoint:
    """A point on the label row above the track, at the track's x for ``fraction``."""
    track = slider.track_rect()
    return QPoint(int(track.left() + track.width() * fraction), slider.LABEL_H // 2)


def test_a_press_on_the_mic_level_label_leaves_the_os_level_alone(home, qtbot: QtBot) -> None:
    """home-gain-labels L1: a click on the words "Mic level" must not set (or mute) the OS mic level."""
    window, _, fake = home()
    slider = window.mic_gain
    for fraction in (0.0, 0.2, 0.5):
        qtbot.mousePress(slider, Qt.MouseButton.LeftButton, pos=_label_row(slider, fraction))
        assert not slider.dragging
        qtbot.mouseMove(slider, pos=_label_row(slider, 0.9))
        qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=_label_row(slider, 0.9))
        qtbot.mouseDClick(slider, Qt.MouseButton.LeftButton, pos=_label_row(slider, fraction))
    assert fake.sets == [] and slider.value() == 60


def test_a_press_on_the_voice_volume_label_sets_nothing(home, qtbot: QtBot) -> None:
    """The fix is in BarSlider: a labelled slider ignores a press on its label; a drag that starts on the track still
    follows the pointer over the label row."""
    window, engine, _ = home(saved=prefs.Prefs(settings=Settings(loudness=62.0)))
    bar = window.call_volume
    before = len(engine.sent)
    qtbot.mousePress(bar, Qt.MouseButton.LeftButton, pos=_label_row(bar, 0.1))
    assert not bar.dragging
    qtbot.mouseRelease(bar, Qt.MouseButton.LeftButton, pos=_label_row(bar, 0.1))
    qtbot.mouseDClick(bar, Qt.MouseButton.LeftButton, pos=_label_row(bar, 0.1))  # no reset to 50 either
    assert len(engine.sent) == before and bar.value() == 62
    pos = _press_at(qtbot, bar, 0.3)
    assert bar.dragging
    qtbot.mouseMove(bar, pos=_label_row(bar, 0.8))
    qtbot.mouseRelease(bar, Qt.MouseButton.LeftButton, pos=pos)
    assert abs(bar.value() - 80) <= 2 and engine.sent[-1].loudness == bar.value()


def test_double_click_does_not_jump_the_mic_level(home, qtbot: QtBot) -> None:
    window, _, fake = home()
    qtbot.mouseDClick(window.mic_gain, Qt.MouseButton.LeftButton, pos=QPoint(10, window.mic_gain.height() // 2))
    assert all(level < 0.1 for _, level in fake.sets[1:])  # the press sets where it lands, the double-click nothing


# ------------------------------------------------------------------ under Your call: Voice volume


def test_the_call_slider_sits_beside_the_call_meter_and_shows_voice_volume(home) -> None:
    window, _, _ = home(saved=prefs.Prefs(settings=Settings(loudness=62.0)))
    assert window.steps[2].isAncestorOf(window.call_volume) and window.steps[2].isAncestorOf(window.out_meter)
    assert _beside(window, window.out_meter, window.call_volume)
    assert window.call_volume.value() == 62


@pytest.mark.parametrize(("state", "enabled", "text"), [("running", True, "-4.0 dB"), ("mic-unplugged", False, "–")])
def test_the_preview_mic_level_matches_the_app(
    qtbot: QtBot, tmp_path: Path, state: str, enabled: bool, text: str
) -> None:
    """The renders the user approves show what the app shows: an unplugged mic has no level, so it is greyed."""
    sample = sample_states(DEVICES)[state]
    path = tmp_path / "settings.json"
    prefs.save(path, sample.saved())
    window = MainWindow(
        sample.engine(), path, login_item=FakeLoginItem(), platform="win32", mic_levels=sample.mic_levels()
    )
    qtbot.addWidget(window)
    window.tick()
    assert window.mic_gain.isEnabled() is enabled
    assert window.mic_gain.value_text() == text


def test_the_call_slider_moves_voice_volume_once(home, qtbot: QtBot) -> None:
    window, engine, _ = home()
    before = len(engine.sent)
    window.call_volume.setFocus()
    qtbot.keyClick(window.call_volume, Qt.Key.Key_Right)
    assert len(engine.sent) == before + 1 and engine.sent[-1].loudness == 51
    assert window.prefs.settings.loudness == 51


# ------------------------------------------------------------------ the control behind the slider


class FakeEndpoint:
    def __init__(self, value: float | None) -> None:
        self.value = value
        self.closed = False

    def get(self) -> float | None:
        return self.value

    def set(self, level: float) -> bool:
        self.value = level
        return True

    def get_db(self) -> float | None:
        return None

    def close(self) -> None:
        self.closed = True


def test_the_system_control_keeps_one_endpoint_open_per_mic() -> None:
    opened: list[str] = []
    endpoints: dict[str, FakeEndpoint] = {}

    def opener(name: str) -> FakeEndpoint:
        opened.append(name)
        endpoints[name] = FakeEndpoint(0.5)
        return endpoints[name]

    levels = SystemMicLevel(opener)
    assert [levels.level("a") for _ in range(3)] == [0.5, 0.5, 0.5]
    assert opened == ["a"]
    assert levels.set_level("a", 1.7) and endpoints["a"].value == 1.0  # clamped
    levels.level("b")
    assert opened == ["a", "b"] and endpoints["a"].closed
    levels.close()
    assert endpoints["b"].closed


def test_the_system_control_asks_again_later_for_a_mic_with_no_level() -> None:
    now = [0.0]
    calls: list[str] = []
    endpoint = FakeEndpoint(0.4)

    def opener(name: str) -> FakeEndpoint | None:
        calls.append(name)
        return endpoint if len(calls) > 1 else None

    levels = SystemMicLevel(opener, clock=lambda: now[0])
    assert levels.level("a") is None and levels.level("a") is None
    assert calls == ["a"]  # not every tick
    now[0] = RETRY_S
    assert levels.level("a") == 0.4
    endpoint.value = None  # unplugged
    assert levels.level("a") is None and endpoint.closed


def test_the_system_control_survives_an_opener_that_raises() -> None:
    def opener(name: str) -> FakeEndpoint:
        raise OSError("no sound system")

    levels = SystemMicLevel(opener)
    assert levels.level("a") is None and not levels.set_level("a", 0.5)


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows endpoint volume")
def test_the_real_default_mic_level_reads_without_changing_it() -> None:
    """The default WASAPI input, found by the name PortAudio lists it under: its level reads, and is left as it is."""
    sd = pytest.importorskip("sounddevice")
    from clean_voice.endpoints import open_capture_volume

    wasapi = next((api for api in sd.query_hostapis() if api["name"] == "Windows WASAPI"), None)
    if wasapi is None or wasapi["default_input_device"] < 0:
        pytest.skip("no default capture device")
    name = sd.query_devices(wasapi["default_input_device"])["name"]
    volume = open_capture_volume(name)
    assert volume is not None, f"no endpoint volume for {name!a}"
    try:
        level = volume.get()
        assert level is not None and 0.0 <= level <= 1.0
    finally:
        volume.close()
    assert volume.get() is None  # closed
