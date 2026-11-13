"""Start at login: the command it registers, the Windows Run value and the macOS LaunchAgent (J07 A1-A3)."""

import plistlib
import sys
from pathlib import Path

import pytest

from clean_voice.autostart import LABEL, MacLoginItem, WindowsLoginItem, launch_command


@pytest.mark.parametrize(
    "app",
    [
        r"C:\Program Files\Clean Voice\Clean Voice.exe",
        "/Applications/Clean Voice.app/Contents/MacOS/Clean Voice",
        r"C:\Users\guest\AppData\Local\Programs\Clean Voice\Clean Voice.exe",
    ],
)
def test_a1_an_app_runs_itself_minimized(app: str) -> None:
    assert launch_command(app) == [app, "--minimized"]


@pytest.mark.parametrize(
    "python",
    [
        r"C:\Users\guest\clean-voice\.venv\Scripts\python.exe",
        r"C:\Python312\pythonw.exe",
        r"C:\Python312\PYTHON.EXE",
        "/usr/local/bin/python3",
        "/Users/guest/clean-voice/.venv/bin/python3.12",
    ],
)
def test_a1_a_python_interpreter_runs_the_module(python: str) -> None:
    assert launch_command(python) == [python, "-m", "clean_voice", "--minimized"]


def test_a1_a_name_that_only_starts_like_python_is_an_app() -> None:
    assert launch_command(r"C:\Tools\pythonista-helper.exe") == [r"C:\Tools\pythonista-helper.exe", "--minimized"]


TEST_KEY = r"Software\Clean Voice tests\Run"


@pytest.fixture
def scratch_run_key():
    if sys.platform != "win32":
        pytest.skip("the Run key is Windows only")
    import winreg

    yield TEST_KEY
    for key in (TEST_KEY, r"Software\Clean Voice tests"):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)
        except FileNotFoundError:
            pass


def test_a2_windows_enable_disable_round_trip(scratch_run_key: str) -> None:
    assert sys.platform == "win32"
    import winreg

    item = WindowsLoginItem(key=scratch_run_key)
    assert not item.is_enabled()
    command = [r"C:\Program Files\Clean Voice\Clean Voice.exe", "--minimized"]
    item.enable(command)
    assert item.is_enabled()
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, scratch_run_key) as key:
        value, kind = winreg.QueryValueEx(key, "Clean Voice")
    assert kind == winreg.REG_SZ
    assert value == '"C:\\Program Files\\Clean Voice\\Clean Voice.exe" --minimized'  # the path quoted: it has spaces
    item.enable(command)  # enabling twice is harmless
    assert item.is_enabled()
    item.disable()
    assert not item.is_enabled()
    item.disable()  # so is disabling twice
    assert not item.is_enabled()


def test_a2_the_real_run_key_is_the_default() -> None:
    item = WindowsLoginItem()
    assert item.key == r"Software\Microsoft\Windows\CurrentVersion\Run"
    assert item.name == "Clean Voice"


def test_a3_mac_writes_a_launch_agent(tmp_path: Path) -> None:
    path = tmp_path / "Library" / "LaunchAgents" / f"{LABEL}.plist"  # the folder does not exist yet
    item = MacLoginItem(path)
    assert not item.is_enabled()
    command = ["/Applications/Clean Voice.app/Contents/MacOS/Clean Voice", "--minimized"]
    item.enable(command)
    assert item.is_enabled()
    with path.open("rb") as f:
        plist = plistlib.load(f)
    assert plist == {"Label": "io.github.paulcaesarflores.cleanvoice", "ProgramArguments": command, "RunAtLoad": True}
    item.disable()
    assert not item.is_enabled()
    assert not path.exists()
    item.disable()  # disabling twice is harmless
    assert not item.is_enabled()


def test_a3_the_real_launch_agent_is_the_default() -> None:
    assert (
        MacLoginItem().path == Path.home() / "Library" / "LaunchAgents" / "io.github.paulcaesarflores.cleanvoice.plist"
    )
