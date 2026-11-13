"""Stamp the built Windows exe's LegalCopyright with the copyright line from LICENSE.

    set_copyright.py "build/clean-voice/windows/app/src/Clean Voice.exe"

Briefcase's build sets the exe's version strings with rcedit (briefcase/platforms/windows/app.py, the build's
"Setting stub app details"): company, description, versions, names and the icon. It doesn't set LegalCopyright, so
the stub's own "Copyright (C) 2022" stays. This runs the same rcedit, the one Briefcase downloaded into its tool cache,
after `briefcase build windows` and before `briefcase package windows`, which packages the exe as it is. A rebuild
leaves the line alone, because the build doesn't touch it. Prints the copyright read back from the exe; exits 1 with
the reason when it can't.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LICENSE = ROOT / "LICENSE"


def copyright_line(license_file: Path = LICENSE) -> str:
    """LICENSE's "Copyright (c) <year> <holder>" line."""
    for line in license_file.read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith("copyright"):
            return line.strip()
    raise ValueError(f"{license_file} has no Copyright line")


def rcedit_path() -> Path:
    """Where Briefcase keeps rcedit: BRIEFCASE_HOME, or its user cache, then tools/ (briefcase/commands/base.py
    validate_data_path, briefcase/integrations/rcedit.py rcedit_path)."""
    home = os.environ.get("BRIEFCASE_HOME")
    if home:
        base = Path(home).resolve()
    else:
        from platformdirs import PlatformDirs  # Briefcase's own dependency

        name = "org.beeware.briefcase" if platform.system() == "Darwin" else "briefcase"
        base = PlatformDirs(appname=name, appauthor="BeeWare").user_cache_path
    return base / "tools" / "rcedit-x64.exe"


def legal_copyright(exe: Path) -> str | None:
    """The exe's LegalCopyright string, read with version.dll; None if it has none."""
    version = ctypes.WinDLL("version")  # type: ignore[attr-defined]
    version.GetFileVersionInfoSizeW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p]
    version.GetFileVersionInfoSizeW.restype = ctypes.c_uint32
    version.GetFileVersionInfoW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    version.VerQueryValueW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    size = version.GetFileVersionInfoSizeW(str(exe), None)
    if not size:
        return None
    data = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(str(exe), 0, size, data):
        return None
    value, length = ctypes.c_void_p(), ctypes.c_uint32()
    # The string table is named by the resource's first language and code page (Briefcase's stub: 040904b0).
    if not version.VerQueryValueW(data, "\\VarFileInfo\\Translation", ctypes.byref(value), ctypes.byref(length)):
        return None
    language, codepage = ctypes.cast(value, ctypes.POINTER(ctypes.c_uint16))[0:2]
    key = f"\\StringFileInfo\\{language:04x}{codepage:04x}\\LegalCopyright"
    if not version.VerQueryValueW(data, key, ctypes.byref(value), ctypes.byref(length)):
        return None
    return ctypes.wstring_at(value.value, length.value).rstrip("\0") if value.value else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exe", type=Path, help="the built Clean Voice.exe")
    args = parser.parse_args(argv)
    exe: Path = args.exe
    if not exe.is_file():
        print(f"set_copyright: no such file: {exe}", file=sys.stderr)
        return 1
    rcedit = rcedit_path()
    if not rcedit.is_file():
        print(f"set_copyright: no rcedit at {rcedit}; run `briefcase build windows` first", file=sys.stderr)
        return 1
    line = copyright_line()
    subprocess.run([str(rcedit), str(exe), "--set-version-string", "LegalCopyright", line], check=True)
    written = legal_copyright(exe)
    if written != line:
        print(f"set_copyright: {exe} reads back {written!r}, not {line!r}", file=sys.stderr)
        return 1
    print(f"{exe}: LegalCopyright = {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
