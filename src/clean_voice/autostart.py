"""Start at login ("Start with Windows" in the original, src-tauri/src/main.rs:126-141 and its LaunchAgent on macOS).

Windows: a value "Clean Voice" under HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run. macOS: a LaunchAgent
plist with RunAtLoad. Either runs the app with ``--minimized``, so it starts quietly in the tray. Off by default (J07
plan, open question 1, answer b): the guest opens Clean Voice for the call.
"""

from __future__ import annotations

import plistlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Protocol

APP_NAME = "Clean Voice"
LABEL = "io.github.paulcaesarflores.cleanvoice"  # the Briefcase bundle id (pyproject.toml)
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
MINIMIZED = "--minimized"
_PYTHON = re.compile(r"pythonw?(\d+(\.\d+)*)?", re.IGNORECASE)


def launch_command(app_path: str) -> list[str]:
    """What the login item runs. ``app_path`` is the running process image: the Briefcase stub in a built app, the
    Python interpreter under ``briefcase dev`` or ``python -m clean_voice``, which then needs the module."""
    name = re.split(r"[\\/]", app_path)[-1]
    stem = name[:-4] if name.lower().endswith(".exe") else name
    if _PYTHON.fullmatch(stem):
        return [app_path, "-m", "clean_voice", MINIMIZED]
    return [app_path, MINIMIZED]


def current_command() -> list[str]:
    """The launch command for this process (QCoreApplication.applicationFilePath, else sys.executable)."""
    from PySide6.QtCore import QCoreApplication

    path = QCoreApplication.applicationFilePath() if QCoreApplication.instance() is not None else ""
    return launch_command(path or sys.executable)


class LoginItem(Protocol):
    def is_enabled(self) -> bool: ...
    def enable(self, command: list[str]) -> None: ...
    def disable(self) -> None: ...


class WindowsLoginItem:
    """A Run value under HKEY_CURRENT_USER; ``key`` is replaceable so tests use a scratch key."""

    def __init__(self, key: str = RUN_KEY, name: str = APP_NAME) -> None:
        self.key = key
        self.name = name

    def is_enabled(self) -> bool:
        assert sys.platform == "win32"
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.key) as key:
                winreg.QueryValueEx(key, self.name)
        except FileNotFoundError:
            return False
        return True

    def enable(self, command: list[str]) -> None:
        assert sys.platform == "win32"
        import winreg

        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, self.key, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, self.name, 0, winreg.REG_SZ, subprocess.list2cmdline(command))

    def disable(self) -> None:
        assert sys.platform == "win32"
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self.key, 0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, self.name)
        except FileNotFoundError:
            pass


class MacLoginItem:
    """A per-user LaunchAgent; ``path`` is replaceable so the writer runs in tests on any OS."""

    def __init__(self, path: Path | None = None, label: str = LABEL) -> None:
        self.path = path if path is not None else Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        self.label = label

    def is_enabled(self) -> bool:
        return self.path.is_file()

    def enable(self, command: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            plistlib.dump({"Label": self.label, "ProgramArguments": list(command), "RunAtLoad": True}, f)

    def disable(self) -> None:
        self.path.unlink(missing_ok=True)


def login_item() -> LoginItem:
    """This platform's login item."""
    return WindowsLoginItem() if sys.platform == "win32" else MacLoginItem()
