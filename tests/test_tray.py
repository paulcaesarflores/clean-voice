"""The tray, close to tray, --minimized and the single instance (J07 T1-T4)."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot
from test_main_screen import MICS, ON, OUTPUTS, FakeLoginItem

from clean_voice import prefs, tray
from clean_voice.app import SHOWN_MESSAGE, MainWindow, main, show_window
from clean_voice.preview import PreviewEngine


def window_with_tray(qtbot: QtBot, tmp_path: Path) -> tuple[MainWindow, PreviewEngine, tray.Tray]:
    engine = PreviewEngine(ON, MICS, tmp_path / "Recordings", OUTPUTS)
    prefs.save(tmp_path / "settings.json", prefs.Prefs(recording=True))  # the backup (off by default) on
    window = MainWindow(engine, tmp_path / "settings.json", login_item=FakeLoginItem(), platform="win32")
    qtbot.addWidget(window)
    window.tray = tray.Tray(window)
    window.show()
    qtbot.waitExposed(window)
    return window, engine, window.tray


def test_tray_menu_mirrors_the_power_button_and_closing_hides(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # T1 (main.rs:160-193, 230-236)
    window, engine, icon = window_with_tray(qtbot, tmp_path)
    assert icon.actions() == ["Open Clean Voice", "Clean my voice", "", "Quit"]
    assert icon.clean_action.isCheckable() and icon.clean_action.isChecked()
    icon.clean_action.trigger()  # unchecks it
    assert engine.sent[-1].cleaning is False and not window.power_button.isChecked()
    window.power_button.click()
    assert icon.clean_action.isChecked() and engine.sent[-1].cleaning is True
    window.tick()
    assert icon.tooltip() == "Clean Voice: on, recording a backup"
    # Closing the window hides it; the voice keeps going.
    window.close()
    assert not window.isVisible() and ("close", None) not in engine.commands
    icon.open_action.trigger()
    assert window.isVisible()
    quits: list[bool] = []
    monkeypatch.setattr(QApplication, "quit", staticmethod(lambda: quits.append(True)))
    icon.quit_action.trigger()
    assert ("close", None) in engine.commands and quits and not window.isVisible()


def test_minimized_starts_hidden_only_with_a_tray(qtbot: QtBot, tmp_path: Path) -> None:
    # T2 (main.rs:206-209): without a tray there would be no way back to the window.
    engine = PreviewEngine(ON, MICS, tmp_path, OUTPUTS)
    window = MainWindow(engine, tmp_path / "settings.json", login_item=FakeLoginItem())
    qtbot.addWidget(window)
    show_window(window, minimized=True)
    assert window.isVisible()
    window.hide()
    window.tray = tray.Tray(window)
    show_window(window, minimized=True)
    assert not window.isVisible()
    show_window(window, minimized=False)
    assert window.isVisible()


SECOND_LAUNCH = """
import sys
from PySide6.QtCore import QCoreApplication
from clean_voice import prefs, tray
app = QCoreApplication([])
print(tray.notify_existing(sys.argv[1]))
"""


def test_a_second_launch_asks_the_first_to_show(qtbot: QtBot) -> None:
    # T3 (main.rs:198): the second launch is another process, as it is for real.
    name = f"clean-voice-test-{uuid.uuid4().hex}"
    assert not tray.notify_existing(name)  # nobody listening: this launch carries on
    server = tray.InstanceServer(name)
    try:
        assert server.listening()
        with qtbot.waitSignal(server.show_requested, timeout=10_000):
            second = subprocess.Popen(
                [sys.executable, "-c", SECOND_LAUNCH, name],
                stdout=subprocess.PIPE,
                text=True,
                env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
            )
        out, _ = second.communicate(timeout=30)
        assert out.strip() == "True"
    finally:
        server.close()


LEFT_BEHIND = """
import shiboken6
from PySide6.QtWidgets import QWidget

left = []

def test_one(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    window.show()
    left.append(window)

def test_two():
    assert not shiboken6.isValid(left[0]), "the last test's window still exists"
"""


def test_each_tests_windows_are_gone_before_the_next(tmp_path: Path) -> None:
    # Undeleted windows from earlier tests once made the second-launch test above spend its first seconds destroying
    # them, so the second launch gave up and its "show" was lost (tests/conftest.py, pytest_runtest_teardown).
    conftest = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")
    (tmp_path / "conftest.py").write_text(conftest, encoding="utf-8")
    (tmp_path / "test_left_behind.py").write_text(LEFT_BEHIND, encoding="utf-8")
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "2 passed" in run.stdout


def test_launch_checks_never_use_the_single_instance(
    qtbot: QtBot, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # T4: a --quit-after-ms run neither asks a running copy to show nor creates the server, so a launch check never
    # depends on (or disturbs) a copy that is already running.
    # A name of its own: on Windows a second server can listen on the real name beside the user's running Clean Voice.
    name = f"clean-voice-test-{uuid.uuid4().hex}"
    monkeypatch.setattr(tray, "server_name", lambda: name)
    server = tray.InstanceServer(tray.server_name())
    shown: list[bool] = []
    server.show_requested.connect(lambda: shown.append(True))
    try:
        code = main(["--quit-after-ms", "300"], engine=PreviewEngine(ON, MICS), prefs_path=tmp_path / "s.json")
        qtbot.wait(100)
    finally:
        server.close()
    assert code == 0
    assert SHOWN_MESSAGE in capsys.readouterr().out
    assert not shown
