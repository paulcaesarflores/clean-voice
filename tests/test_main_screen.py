"""The main screen (J07 U1-U18), now the one window (J10 Step 9): every state, every control, wired to the engine and
the saved choices. The window's own layout checks (T26-T30) are in test_one_window.py."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from PySide6.QtCore import QPointF, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFontMetrics
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel
from pytestqt.qtbot import QtBot
from test_engine import CABLE, MIC, wait_for

from clean_voice import prefs
from clean_voice.app import (
    ECHO_TEXT,
    HEAD_CLEAN,
    HEAD_NO_CALL,
    HEAD_NO_DEVICE,
    HEAD_NO_MIC,
    HEAD_RAW,
    MainWindow,
    describe,
    format_duration,
)
from clean_voice.devices import CABLE_DOWNLOAD_URL
from clean_voice.engine import ECHO_MAC, ECHO_NO_SPEAKERS, ECHO_PAUSED, EchoState, MicTestState, State, Status
from clean_voice.preview import DeviceNames, PreviewEngine, sample_states
from clean_voice.settings import PRESETS, Settings, bands_for

# tests/conftest.py sets QT_QPA_PLATFORM=offscreen. The device names are this file's own, never the machine's.
JAPANESE = "マイク (USB オーディオ デバイス)"
MICS = ["Microphone (Test USB Mic)", "Microphone Array (Test Laptop)", JAPANESE]
CABLE_NAME = "CABLE Input (VB-Audio Virtual Cable)"
OUTPUTS = [CABLE_NAME, "Speakers (Test Speakers)", "Headphones (Test Headset)"]
DEVICES = DeviceNames(tuple(MICS), tuple(OUTPUTS), CABLE_NAME, MICS[0], OUTPUTS[2])
ON = Status(
    state=State.ON,
    mic=MICS[0],
    cable=CABLE_NAME,
    output=CABLE_NAME,
    level=0.42,
    input_level=0.3,
    recording=True,
    recording_seconds=754.0,
)


class FakeLoginItem:
    def __init__(self, works: bool = True) -> None:
        self.works = works
        self.enabled = False
        self.commands: list[list[str]] = []

    def is_enabled(self) -> bool:
        return self.enabled

    def enable(self, command: list[str]) -> None:
        self.commands.append(command)
        self.enabled = self.works

    def disable(self) -> None:
        self.enabled = False


@pytest.fixture
def make(qtbot: QtBot, tmp_path: Path):
    """A window on a PreviewEngine and a prefs file in tmp_path. Unless a test saves its own choices, the backup is
    switched on, as most states here show a recording (the default is off since 2026-11-11, tested on its own in
    test_one_window.py)."""

    def make(
        status: Status = ON,
        mics: list[str] | None = None,
        outputs: list[str] | None = None,
        saved: prefs.Prefs | None = None,
        platform: str = "win32",
        login: FakeLoginItem | None = None,
        engine: PreviewEngine | None = None,
    ) -> tuple[MainWindow, PreviewEngine]:
        path = tmp_path / "settings.json"
        prefs.save(path, saved if saved is not None else prefs.Prefs(recording=True))
        engine = engine or PreviewEngine(
            status, MICS if mics is None else mics, tmp_path / "Recordings", OUTPUTS if outputs is None else outputs
        )
        window = MainWindow(engine, path, login_item=login or FakeLoginItem(), platform=platform)
        qtbot.addWidget(window)
        window.show()
        qtbot.waitExposed(window)
        return window, engine

    return make


def visible(window: MainWindow, widget) -> bool:
    return widget.isVisibleTo(window)


def settle(window: MainWindow, ticks: int = 5) -> None:
    for _ in range(ticks):
        window.tick()


# ------------------------------------------------------------------ the states


def test_running(make) -> None:
    # U1
    window, _ = make(ON)
    settle(window)
    assert window.headline.text() == HEAD_CLEAN and window.power_button.tone == "signal" and window.header.lit
    assert window.in_meter.fraction > 0 and window.out_meter.fraction > 0
    assert window.power_button.isChecked()
    assert window.rec_dot.recording and visible(window, window.rec_dot)
    assert window.rec_label.text() == "Backup recording" and window.rec_time.text() == "12:34"


def test_the_two_meters_show_the_mic_and_the_clean_voice_apart(make) -> None:
    # U1: Step 1's meter is the raw input peak, Step 3's the clean one.
    window, _ = make(dataclasses.replace(ON, input_level=0.9, level=0.01))
    settle(window)
    assert window.in_meter.fraction > window.out_meter.fraction + 0.3


def test_stopped_says_starting(make) -> None:
    # U2: before the streams open the engine is OFF; the screen makes no "On" claim.
    window, _ = make(Status(state=State.OFF, cable=CABLE_NAME))
    settle(window)
    assert window.headline.text() == "Starting…" and window.power_button.tone == "line" and not window.header.lit
    assert window.in_meter.fraction == 0 and window.out_meter.fraction == 0
    assert not window.rec_dot.recording


@pytest.mark.parametrize(("platform", "name"), [("win32", "CABLE Output"), ("darwin", "VB-Cable")])
def test_cable_missing(make, monkeypatch: pytest.MonkeyPatch, platform: str, name: str) -> None:
    # U3: Step 4 becomes the install steps, with the VB-Audio credit and the device to pick per OS.
    window, _ = make(Status(state=State.NO_CABLE, mic=MICS[0]), platform=platform)
    assert window.headline.text() == HEAD_NO_CALL and window.power_button.tone == "hot"
    assert visible(window, window.cable_title) and window.steps[3].isAncestorOf(window.cable_title)
    assert window.cable_title.text() == "Install VB-CABLE" and not visible(window, window.use_hint)
    body = window.cable_body.text()
    assert "1." in body and "2." in body and "3." in body and name in body
    assert "VB-Audio" in window.cable_credit.text() and "donations" in window.cable_credit.text()
    opened: list[QUrl] = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    window.download_button.click()
    assert [u.toString() for u in opened] == [CABLE_DOWNLOAD_URL]


def test_cable_missing_shows_when_cleaning_is_off(make) -> None:
    # U3
    window, _ = make(Status(state=State.NO_CABLE), saved=prefs.Prefs(settings=Settings(cleaning=False)))
    assert visible(window, window.cable_title) and window.cable_title.text() == "Install VB-CABLE"


def test_mic_unplugged_and_back_with_the_real_engine(qtbot: QtBot, tmp_path: Path, make_engine) -> None:
    # U4: the real Engine on the fake sound system; the window starts it, as at launch.
    engine, backend = make_engine([MIC, CABLE])
    prefs.save(tmp_path / "settings.json", prefs.Prefs(recording=True))  # the backup (off by default) on
    window = MainWindow(engine, tmp_path / "settings.json", login_item=FakeLoginItem(), platform="win32")
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda: window.headline.text() == HEAD_CLEAN, timeout=10_000)
    files = engine.status().recording_files
    assert files
    backend.present.remove(MIC)
    qtbot.waitUntil(lambda: window.headline.text() == HEAD_NO_MIC, timeout=10_000)
    assert visible(window, window.mic_title) and window.mic_title.text() == "Your microphone was unplugged"
    backend.present.append(MIC)
    qtbot.waitUntil(lambda: window.headline.text() == HEAD_CLEAN, timeout=10_000)
    assert not visible(window, window.mic_title)
    assert engine.status().recording_files == files
    window.close()
    engine.stop()
    wait_for(lambda: engine.status().state is State.OFF)


def test_no_mic_and_error(make) -> None:
    # U5
    window, _ = make(Status(state=State.NO_MIC, cable=CABLE_NAME, output=CABLE_NAME), mics=[])
    assert window.mic_title.text() == "Plug in a microphone" and window.headline.text() == HEAD_NO_MIC
    assert window.input_button.text() == "No microphone found"
    window.open_mic_menu()
    assert [i.text() for i in window.mic_menu.items] == ["No microphone found"]
    assert not any(i.isEnabled() for i in window.mic_menu.items)
    window.close_mic_menu()
    window.apply(Status(state=State.ERROR, cable=CABLE_NAME, error="The microphone is in use by another app"))
    assert visible(window, window.mic_title)
    # J07 r1 m2: the engine does not say which device failed, so the words name neither; its own text is the tooltip.
    assert "Close it" in window.mic_body.text() and "microphone or VB-CABLE" in window.mic_body.text()
    assert "The microphone is in use by another app" in window.mic_body.toolTip()
    assert window.headline.text() == HEAD_NO_DEVICE and window.power_button.tone == "hot"


def test_the_power_button_switches_cleaning_and_it_is_saved(make, tmp_path: Path) -> None:
    # U6: off is a bypass (main.js:622), the engine keeps running.
    window, engine = make(ON)
    window.power_button.click()
    assert engine.sent[-1].cleaning is False
    assert ("stop", None) not in engine.commands
    window.tick()
    assert window.headline.text() == HEAD_RAW and not window.power_button.isChecked()
    assert window.power_button.tone == "line" and not window.header.lit
    window.save_prefs()
    assert prefs.load(tmp_path / "settings.json").settings.cleaning is False


def test_backup_switch_labels_and_open_folder(make, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # U7
    window, engine = make(ON)
    window.tick()
    assert window.rec_switch.isChecked() and window.rec_label.text() == "Backup recording"
    window.rec_switch.click()
    assert engine.commands[-1] == ("record", "off")
    window.tick()
    assert not window.rec_switch.isChecked() and window.rec_label.text() == "Backup off"
    assert not visible(window, window.rec_dot)
    window.save_prefs()
    assert prefs.load(tmp_path / "settings.json").recording is False
    opened: list[QUrl] = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    window.open_folder_button.click()
    assert (tmp_path / "Recordings").is_dir()
    assert Path(opened[0].toLocalFile()) == tmp_path / "Recordings"


def test_a_backup_error_names_the_folder(make) -> None:
    # U7
    message = "Can't save backups to C:\\Users\\guest\\Music\\Clean Voice. Your voice still reaches the call."
    window, _ = make(dataclasses.replace(ON, recording=False, recording_error=message))
    assert window.rec_label.text() == "Backup not saved"
    assert window.rec_detail.text() == message and "Music\\Clean Voice" in message
    assert window.rec_detail.property("warn") and window.steps[4].isAncestorOf(window.rec_detail)
    assert window.headline.text() == HEAD_CLEAN


def test_mic_test(make) -> None:
    # U8
    window, engine = make(ON)
    assert window.test_button.text() == "Test your mic" and window.test_button.isEnabled()
    window.test_button.click()
    assert engine.commands[-1] == ("mic_test", "start")
    engine.set_status(dataclasses.replace(ON, mic_test=MicTestState.RECORDING, mic_test_seconds=1.6))
    window.tick()
    assert window.test_button.text() == "Speak now  4 s" and window.test_button.running
    assert abs(window.test_button.progress - 1.6 / 5) < 1e-9
    engine.set_status(dataclasses.replace(ON, mic_test=MicTestState.PLAYING, mic_test_seconds=1.0))
    window.tick()
    assert window.test_button.text() == "Playing back" and abs(window.test_button.progress - 0.2) < 1e-9
    window.test_button.click()
    assert engine.commands[-1] == ("mic_test", "cancel")
    engine.set_status(dataclasses.replace(ON, mic_test_error="Clean Voice can't find speakers or headphones."))
    window.tick()
    assert window.test_label.text() == "Clean Voice can't find speakers or headphones."
    assert visible(window, window.test_label) and window.test_button.text() == "Test your mic"


def test_echo(make, tmp_path: Path) -> None:
    # U9: off by default (DESIGN wins over the original's on); the live reason in the warning colour.
    window, engine = make(ON)
    assert not window.echo_switch.isChecked() and ("echo", "off") in engine.commands
    assert window.steps[1].isAncestorOf(window.echo_switch)  # Step 2, under Denoise
    window.echo_switch.click()
    assert engine.commands[-1] == ("echo", "on")
    window.tick()
    assert window.echo_desc.text() == ECHO_TEXT and not window.echo_desc.property("warn")
    for state, reason in ((EchoState.PAUSED, ECHO_PAUSED), (EchoState.UNAVAILABLE, ECHO_NO_SPEAKERS)):
        engine.set_status(dataclasses.replace(ON, echo=state, echo_reason=reason))
        window.tick()
        assert window.echo_desc.text() == reason and window.echo_desc.property("warn")
    window.save_prefs()
    assert prefs.load(tmp_path / "settings.json").echo is True


def test_echo_on_a_mac(make) -> None:
    # U9
    window, _ = make(ON, platform="darwin")
    assert not window.echo_switch.isEnabled() and window.echo_desc.text() == ECHO_MAC


def test_hear_myself(make) -> None:
    # U10
    window, engine = make(ON)
    assert window.monitor_button.toolTip() == "Hear myself"
    window.monitor_button.click()
    assert engine.commands[-1] == ("monitor", "on")
    window.tick()
    assert window.monitor_button.isChecked() and window.monitor_button.toolTip() == "Stop hearing myself"
    window._monitor_pending_until = 0.0  # the engine has answered
    engine.set_status(dataclasses.replace(ON, monitor=False, monitor_error="Your headphones or speakers stopped."))
    window.tick()
    assert not window.monitor_button.isChecked()
    assert visible(window, window.dock_note) and window.dock_note.text() == "Your headphones or speakers stopped."


def test_voice_styles(make) -> None:
    # U11: six styles in Step 2's dropdown, each sent at the Strength set (0.5: the style as designed).
    window, engine = make(ON)
    combo = window.style_combo
    assert [combo.itemText(i) for i in range(combo.count())] == ["Natural", "Warm", "Crisp", "Radio", "Deep", "Flat"]
    assert combo.currentText() == "Natural"
    combo.activated.emit(list(PRESETS).index("warm"))
    assert engine.sent[-1].bands == bands_for("warm") and window.prefs.preset == "warm"
    combo.activated.emit(list(PRESETS).index("deep"))
    assert engine.sent[-1].bands == bands_for("deep") and combo.currentText() == "Deep"


def test_voice_volume(make) -> None:
    # U12: Voice volume stays (Step 3); Bass, Mid and Treble are gone.
    window, engine = make(ON)
    window.call_volume.setFocus()
    QTest.keyClick(window.call_volume, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert engine.sent[-1].loudness == 60
    assert not hasattr(window, "tone_bars")


def test_noise_removal(make) -> None:
    # U13: Denoise's Strength.
    window, engine = make(ON)
    assert window.strength_bar.value() == 70
    window.strength_bar.setFocus()
    QTest.keyClick(window.strength_bar, Qt.Key.Key_Right)
    assert engine.sent[-1].strength == 71
    track = window.strength_bar.track_rect()
    QTest.mouseDClick(window.strength_bar, Qt.MouseButton.LeftButton, pos=track.center().toPoint())
    assert window.strength_bar.value() == 70 and engine.sent[-1].strength == 70


def test_pickers_list_exactly_the_engines_devices(make) -> None:
    # U14: the mic menu under Step 1's dropdown, the output dropdown in Step 4.
    window, engine = make(ON)
    assert [window.output_combo.itemText(i) for i in range(window.output_combo.count())] == OUTPUTS
    assert window.output_combo.currentText() == CABLE_NAME and window.input_button.text() == MICS[0]
    window.open_mic_menu()
    assert window.mic_menu.isVisible() and window.mic_menu.names() == MICS
    assert [i.isChecked() for i in window.mic_menu.items] == [True, False, False]
    window.mic_menu.items[1].click()
    assert engine.commands[-1] == ("select", MICS[1]) and not window.mic_menu.isVisible()
    window.output_combo.activated.emit(2)
    assert engine.commands[-1] == ("select_output", OUTPUTS[2])
    window.tick()
    assert window.headline.text() == HEAD_NO_CALL  # the voice no longer goes to the call


def test_a_japanese_device_name_shows_verbatim(make) -> None:
    # U15
    window, _ = make(ON, outputs=[*OUTPUTS, JAPANESE])
    index = window.output_combo.findText(JAPANESE)
    assert index >= 0 and window.output_combo.itemText(index) == JAPANESE
    window.open_mic_menu()
    item = next(i for i in window.mic_menu.items if i.text() == JAPANESE)
    kana = [ch for ch in JAPANESE if ord(ch) > 0x3000]
    for font in (window.output_combo.font(), item.font(), window.input_button.font()):
        metrics = QFontMetrics(font)
        assert all(metrics.inFontUcs4(ord(ch)) for ch in kana), font.families()


@pytest.mark.parametrize(("platform", "name"), [("win32", "CABLE Output"), ("darwin", "VB-Cable")])
def test_wording_names_any_call_app(qtbot: QtBot, tmp_path: Path, platform: str, name: str) -> None:
    # U16: no text names Meet alone; the use hint names the cable's recording side.
    for state, sample in sample_states(DEVICES).items():
        path = tmp_path / f"{state}.json"
        prefs.save(path, sample.saved())
        window = MainWindow(sample.engine(tmp_path), path, login_item=FakeLoginItem(), platform=platform)
        qtbot.addWidget(window)
        window.show()
        window.tick()
        texts = [w.text() for w in window.findChildren(QLabel) if w.isVisibleTo(window)]
        for text in texts:
            if "Meet" in text:
                assert "Zoom" in text, (state, text)
        window.close()
    window = MainWindow(PreviewEngine(ON, MICS, tmp_path, OUTPUTS), tmp_path / "x.json", FakeLoginItem(), platform)
    qtbot.addWidget(window)
    hint = window.use_hint.text()
    assert name in hint and "Zoom" in hint and "noise cancellation" in hint


def test_saved_choices_come_back(make, tmp_path: Path) -> None:
    # U17: a second window on the same file shows the saved style and strength, sends them before start, and starts
    # with the saved mic.
    window, _ = make(ON)
    window.style_combo.activated.emit(list(PRESETS).index("warm"))
    window.strength_bar.setFocus()
    QTest.keyClick(window.strength_bar, Qt.Key.Key_Right)
    window.choose_mic(MICS[1])
    window.close()  # saves
    engine = PreviewEngine(ON, MICS, tmp_path / "Recordings", OUTPUTS)
    second = MainWindow(engine, tmp_path / "settings.json", login_item=FakeLoginItem(), platform="win32")
    try:
        assert second.style_combo.currentText() == "Warm" and second.strength_bar.value() == 71
        start = engine.commands.index(("start", MICS[1]))
        assert engine.commands.index(("settings", None)) < start
        assert engine.commands.index(("echo", "off")) < start and engine.commands.index(("record", "on")) < start
        assert engine.sent[0].strength == 71 and engine.sent[0].bands == bands_for("warm")
    finally:
        second.close()


@pytest.mark.parametrize(("platform", "label"), [("win32", "Start with Windows"), ("darwin", "Start at login")])
def test_start_at_login_switch(make, platform: str, label: str) -> None:
    # U18: the switch shows what the login item really says after each change (main.js:629-640).
    login = FakeLoginItem()
    window, _ = make(ON, platform=platform, login=login)
    assert window.autostart_label.text() == label and not window.autostart_switch.isChecked()
    assert window.footer.isAncestorOf(window.autostart_switch)
    window.autostart_switch.click()
    assert login.enabled and window.autostart_switch.isChecked()
    assert login.commands and login.commands[-1][-1] == "--minimized"
    window.autostart_switch.click()
    assert not login.enabled and not window.autostart_switch.isChecked()
    login.works = False  # the system refused: the switch reads the real state back
    window.autostart_switch.click()
    assert not window.autostart_switch.isChecked()


# ------------------------------------------------------------------ plain words


def test_every_state_has_plain_words() -> None:
    for sample in sample_states(DEVICES).values():
        view = describe(sample.status)
        assert view.headline
        words = [view.headline]
        text = " ".join([*words, view.notice.title, view.notice.body] if view.notice else words).lower()
        for jargon in ("dbfs", "underrun", "sample rate", "portaudio", "wasapi", "buffer"):
            assert jargon not in text


def test_format_duration() -> None:
    assert format_duration(5) == "00:05"
    assert format_duration(754) == "12:34"
    assert format_duration(3723) == "1:02:03"


def test_the_mic_menu_closes_on_a_click_elsewhere(make) -> None:
    # main.js:579-581
    window, _ = make(ON)
    window.open_mic_menu()
    assert window.mic_menu.isVisible() and window.input_button.expanded
    QTest.mouseClick(window.headline, Qt.MouseButton.LeftButton, pos=QPointF(5, 5).toPoint())
    assert not window.mic_menu.isVisible() and not window.input_button.expanded


def test_short_mic_names() -> None:
    from clean_voice.app import short_mic_name

    assert short_mic_name("Video Call L + R (Focusrite USB Audio)") == "Video Call L + R"
    assert short_mic_name("Microphone (A4tech FHD 1080P PC Camera)") == "A4tech FHD 1080P PC Camera"
    assert short_mic_name("マイク (Realtek(R) Audio)") == "Realtek(R) Audio"
    assert short_mic_name("Headset (ATH-M20xBT )") == "ATH-M20xBT"
    assert short_mic_name("Blue Yeti") == "Blue Yeti"


def test_the_window_follows_the_system_light_or_dark_setting(make, monkeypatch: pytest.MonkeyPatch) -> None:
    from clean_voice import theme
    from clean_voice.app import MainWindow

    monkeypatch.setattr(theme, "system_is_dark", lambda: True)
    window, _ = make(ON)
    assert window.palette_now is theme.DARK and theme.DARK.surface in window.style_text()
    monkeypatch.setattr(theme, "system_is_dark", lambda: False)
    window.follow_system_theme()  # what the system's colorSchemeChanged signal calls
    assert window.palette_now is theme.LIGHT and theme.LIGHT.surface in window.style_text()
    assert theme.DARK.surface not in window.style_text()
    for name in ("light", "dark"):  # both themes paint
        window.set_theme(theme.for_name(name))
        assert not window.grab().isNull()
    # A window given a theme (the renders) keeps it whatever the system says.
    fixed = MainWindow(PreviewEngine(ON, MICS), window.prefs_path, FakeLoginItem(), "win32", theme_name="dark")
    fixed.follow_system_theme()
    assert fixed.palette_now is theme.DARK
    fixed.close()


# The original's palette (style.css), and the v1 redesign's (round 3), which v2 replaced.
ORIGINAL_COLOURS = ["#0e1011", "#17191b", "#202326", "#08090a", "#22e4e4", "#03292b", "#7ff5f5", "34, 228, 228"]
V1_COLOURS = ["#f6f4ef", "#e3ded4", "#4b44c9", "#8f89f2", "#211f29", "#17161c"]


def test_no_colour_of_the_original_or_of_v1_survives() -> None:
    from clean_voice import app, mark, theme, widgets

    for pal in (theme.LIGHT, theme.DARK):
        sheet = app.style_sheet(pal).lower()
        assert not [c for c in ORIGINAL_COLOURS + V1_COLOURS if c in sheet], pal.name
    for module in (widgets, theme, app, mark):
        source = Path(str(module.__file__)).read_text(encoding="utf-8").lower()
        assert not [c for c in ORIGINAL_COLOURS + V1_COLOURS if c in source], module.__name__
    assert (theme.LIGHT.band, theme.DARK.band) == ("#E8ECEF", "#11181E")  # the header band, "studio wall"'s wall
