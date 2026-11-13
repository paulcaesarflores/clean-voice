"""tools/ci/check_icons.py (J10 T38): CI's check that the built exe, MSI and .app carry Clean Voice's icon. Here it
runs against stand-ins built in the test: an exe whose icon is replaced the way rcedit does it (UpdateResourceW), an
MSI made with msilib, and an .app folder. CI runs it on the real build (the ci.yml lines are in the hand-off)."""

from __future__ import annotations

import ctypes
import importlib.util
import plistlib
import shutil
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ICO = ROOT / "icons" / "clean-voice.ico"
ICNS = ROOT / "icons" / "clean-voice.icns"


def _tool():
    spec = importlib.util.spec_from_file_location("check_icons", ROOT / "tools" / "ci" / "check_icons.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _app(tmp_path: Path, icns: bytes, name: str = "Clean Voice") -> Path:
    app = tmp_path / f"{name}.app"
    resources = app / "Contents" / "Resources"
    resources.mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps({"CFBundleIconFile": "clean-voice"}))
    (resources / "clean-voice.icns").write_bytes(icns)
    return app


def test_the_app_check_wants_our_icns_byte_for_byte(tmp_path: Path) -> None:
    tool = _tool()
    assert "is icons/clean-voice.icns" in tool.check_app(_app(tmp_path / "ours", ICNS.read_bytes()))
    with pytest.raises(tool.IconError, match="is not icons/clean-voice.icns"):
        tool.check_app(_app(tmp_path / "bee", ICNS.read_bytes()[:-1] + b"\0"))
    plain = _app(tmp_path / "none", ICNS.read_bytes())
    (plain / "Contents" / "Info.plist").write_bytes(plistlib.dumps({}))
    with pytest.raises(tool.IconError, match="CFBundleIconFile"):
        tool.check_app(plain)


def _with_our_icon(exe: Path) -> None:
    """Put icons/clean-voice.ico into ``exe`` as RT_GROUP_ICON 1 with RT_ICON 1..n, as rcedit --set-icon does."""
    members = _tool().ico_members(ICO.read_bytes())
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel.BeginUpdateResourceW.restype = ctypes.c_void_p
    kernel.BeginUpdateResourceW.argtypes = [ctypes.c_wchar_p, ctypes.c_bool]
    kernel.UpdateResourceW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint16,
        ctypes.c_char_p,
        ctypes.c_uint32,
    ]
    kernel.EndUpdateResourceW.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    handle = kernel.BeginUpdateResourceW(str(exe), True)  # True: drop the exe's own resources
    assert handle
    group = struct.pack("<HHH", 0, 1, len(members))
    for ident, (size, data) in enumerate(sorted(members.items()), start=1):
        assert kernel.UpdateResourceW(handle, ctypes.c_void_p(3), ctypes.c_void_p(ident), 0, data, len(data))
        side = 0 if size >= 256 else size
        group += struct.pack("<BBBBHHIH", side, side, 0, 0, 1, 32, len(data), ident)
    assert kernel.UpdateResourceW(handle, ctypes.c_void_p(14), ctypes.c_void_p(1), 0, group, len(group))
    assert kernel.EndUpdateResourceW(handle, False)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows exe resources")
def test_the_exe_check_reads_the_icon_group_and_wants_our_32_and_256(qapp, tmp_path: Path) -> None:
    tool = _tool()
    other = tmp_path / "python.exe"
    shutil.copy(sys.executable, other)  # Python's own icon: not ours
    with pytest.raises(tool.IconError):
        tool.check_exe(other)
    ours = tmp_path / "Clean Voice.exe"
    shutil.copy(sys.executable, ours)
    _with_our_icon(ours)
    assert "the 32 and 256 px icons are Clean Voice's" in tool.check_exe(ours)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Installer")
def test_the_msi_check_reads_the_icon_table(tmp_path: Path) -> None:
    msilib = pytest.importorskip("msilib")  # Python 3.12 on Windows
    schema = pytest.importorskip("msilib.schema")
    tool = _tool()

    def msi(path: Path, icon: Path) -> Path:
        database = msilib.init_database(str(path), schema, "Test", msilib.gen_uuid(), "1.0.0", "Test")
        msilib.add_data(database, "Icon", [("ProductIcon", msilib.Binary(str(icon)))])
        database.Commit()
        database.Close()
        return path

    assert "holds icons/clean-voice.ico" in tool.check_msi(msi(tmp_path / "ours.msi", ICO))
    bee = tmp_path / "bee.ico"
    bee.write_bytes(ICO.read_bytes()[:-1] + b"\0")
    with pytest.raises(tool.IconError, match="ProductIcon"):
        tool.check_msi(msi(tmp_path / "bee.msi", bee))


def test_the_command_line_says_why_it_failed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tool = _tool()
    app = _app(tmp_path, b"icns-but-not-ours")
    assert tool.main(["app", str(app)]) == 1
    assert "icon check failed" in capsys.readouterr().err
    assert tool.main(["app", str(_app(tmp_path / "ok", ICNS.read_bytes()))]) == 0
