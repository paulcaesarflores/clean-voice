"""CI's check that the built app carries Clean Voice's icon, not Briefcase's default (J10 Step 11, T38).

    check_icons.py exe PATH   Windows: the exe's RT_GROUP_ICON decodes to our 32 and 256 px images
    check_icons.py msi PATH   Windows: the MSI's Icon table holds our .ico, byte for byte
    check_icons.py app PATH   macOS: the .app's Info.plist CFBundleIconFile names an .icns byte-equal to ours

Ours are icons/clean-voice.ico and .icns (written from src/clean_voice/mark.py by tools/make_icons.py). Prints what it
checked; exits 1 with the reason when the icon is not ours.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import plistlib
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ICO = ROOT / "icons" / "clean-voice.ico"
ICNS = ROOT / "icons" / "clean-voice.icns"
RT_ICON, RT_GROUP_ICON = 3, 14
LOAD_AS_DATAFILE = 0x00000002 | 0x00000020  # LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE


class IconError(Exception):
    """The built app's icon is not Clean Voice's."""


def ico_members(data: bytes) -> dict[int, bytes]:
    """An .ico's members by pixel size (ICONDIR, then 16-byte ICONDIRENTRYs; a 0 size means 256)."""
    reserved, kind, count = struct.unpack_from("<HHH", data)
    if (reserved, kind) != (0, 1):
        raise IconError("not an .ico file")
    out = {}
    for i in range(count):
        width, _, _, _, _, _, length, offset = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
        out[width or 256] = data[offset : offset + length]
    return out


def group_members(group: bytes, icon: dict[int, bytes]) -> dict[int, bytes]:
    """A RT_GROUP_ICON (GRPICONDIR: 6 bytes, then 14-byte entries whose last field is the RT_ICON id) resolved against
    the exe's RT_ICON resources, by pixel size."""
    _, _, count = struct.unpack_from("<HHH", group)
    out = {}
    for i in range(count):
        width, _, _, _, _, _, _, ident = struct.unpack_from("<BBBBHHIH", group, 6 + 14 * i)
        if ident not in icon:
            raise IconError(f"the icon group names RT_ICON {ident}, which the exe does not hold")
        out[width or 256] = icon[ident]
    return out


def exe_resources(path: Path) -> tuple[bytes, dict[int, bytes]]:
    """The exe's first RT_GROUP_ICON and all its RT_ICON data, read without running it (LoadLibraryExW as a datafile)."""
    if sys.platform != "win32":
        raise IconError("reading an exe's resources needs Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel.LoadLibraryExW.restype = ctypes.c_void_p
    kernel.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
    kernel.FindResourceW.restype = ctypes.c_void_p
    kernel.FindResourceW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    kernel.LoadResource.restype = ctypes.c_void_p
    kernel.LoadResource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.LockResource.restype = ctypes.c_void_p
    kernel.LockResource.argtypes = [ctypes.c_void_p]
    kernel.SizeofResource.restype = ctypes.c_uint32
    kernel.SizeofResource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
    names: list[int | str] = []
    callback = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)  # type: ignore[attr-defined]

    def found(module: int, kind: int, name: int, param: int) -> bool:
        names.append(name if name < 0x10000 else ctypes.wstring_at(name))
        return True

    kernel.EnumResourceNamesW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, callback, ctypes.c_void_p]
    module = kernel.LoadLibraryExW(str(path), None, LOAD_AS_DATAFILE)
    if not module:
        raise IconError(f"could not open {path} (error {ctypes.get_last_error()})")  # type: ignore[attr-defined]

    def read(kind: int, name: int | str) -> bytes:
        handle = kernel.FindResourceW(module, name if isinstance(name, str) else ctypes.c_void_p(name), kind)
        if not handle:
            raise IconError(f"no resource {name} of type {kind}")
        size = kernel.SizeofResource(module, handle)
        return ctypes.string_at(kernel.LockResource(kernel.LoadResource(module, handle)), size)

    try:
        kernel.EnumResourceNamesW(module, ctypes.c_void_p(RT_GROUP_ICON), callback(found), None)
        if not names:
            raise IconError(f"{path.name} holds no icon group")
        group = read(RT_GROUP_ICON, names[0])
        _, _, count = struct.unpack_from("<HHH", group)
        ids = [struct.unpack_from("<H", group, 6 + 14 * i + 12)[0] for i in range(count)]
        return group, {i: read(RT_ICON, i) for i in ids}
    finally:
        kernel.FreeLibrary(module)


def decoded(data: bytes) -> object:
    """A PNG member's pixels (QImage, straight alpha) so two encodings of one image compare equal."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication, QImage

    _app = QGuiApplication.instance() or QGuiApplication([sys.argv[0]])
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise IconError("an icon member is not a PNG, so it is not one of ours")
    image = QImage.fromData(data)
    if image.isNull():
        raise IconError("an icon member does not decode")
    return image.convertToFormat(QImage.Format.Format_ARGB32)


def check_exe(path: Path, ours: Path = ICO) -> str:
    group, icons = exe_resources(path)
    built, mine = group_members(group, icons), ico_members(ours.read_bytes())
    for size in (32, 256):
        if size not in built:
            raise IconError(f"{path.name} has no {size} px icon")
        if decoded(built[size]) != decoded(mine[size]):
            raise IconError(f"{path.name}'s {size} px icon is not Clean Voice's")
    return f"{path.name}: the 32 and 256 px icons are Clean Voice's ({len(built)} sizes)"


def msi_icons(path: Path) -> dict[str, bytes]:
    """The MSI's Icon table, Name to Data, read through msi.dll (MsiOpenDatabaseW read-only, MsiRecordReadStream)."""
    if sys.platform != "win32":
        raise IconError("reading an MSI needs Windows")
    msi = ctypes.WinDLL("msi")  # type: ignore[attr-defined]
    handle = ctypes.c_ulong
    database, view = handle(), handle()
    if msi.MsiOpenDatabaseW(str(path), ctypes.c_wchar_p(None), ctypes.byref(database)):  # MSIDBOPEN_READONLY is 0
        raise IconError(f"could not open {path}")
    out: dict[str, bytes] = {}
    try:
        if msi.MsiDatabaseOpenViewW(database, "SELECT `Name`, `Data` FROM `Icon`", ctypes.byref(view)):
            raise IconError(f"{path.name} has no Icon table")
        msi.MsiViewExecute(view, handle(0))
        while True:
            record = handle()
            if msi.MsiViewFetch(view, ctypes.byref(record)):  # ERROR_NO_MORE_ITEMS
                break
            try:
                size = ctypes.c_uint32(512)
                name = ctypes.create_unicode_buffer(size.value)
                msi.MsiRecordGetStringW(record, 1, name, ctypes.byref(size))
                length = msi.MsiRecordDataSize(record, 2)
                data = ctypes.create_string_buffer(max(1, length))
                got = ctypes.c_uint32(length)
                if msi.MsiRecordReadStream(record, 2, data, ctypes.byref(got)):
                    raise IconError(f"could not read the icon {name.value}")
                out[name.value] = data.raw[: got.value]
            finally:
                msi.MsiCloseHandle(record)
    finally:
        if view.value:
            msi.MsiCloseHandle(view)
        msi.MsiCloseHandle(database)
    return out


def check_msi(path: Path, ours: Path = ICO) -> str:
    found = msi_icons(path)
    if not found:
        raise IconError(f"{path.name} has an empty Icon table")
    if ours.read_bytes() not in found.values():
        raise IconError(f"no icon in {path.name}'s Icon table is icons/clean-voice.ico ({', '.join(found)})")
    return f"{path.name}: the Icon table holds icons/clean-voice.ico ({', '.join(found)})"


def check_app(path: Path, ours: Path = ICNS) -> str:
    info = plistlib.loads((path / "Contents" / "Info.plist").read_bytes())
    name = info.get("CFBundleIconFile")
    if not isinstance(name, str) or not name:
        raise IconError(f"{path.name}'s Info.plist names no CFBundleIconFile")
    resources = path / "Contents" / "Resources"
    icns = resources / name if name.endswith(".icns") else resources / f"{name}.icns"
    if not icns.is_file():
        raise IconError(f"{path.name} has no {icns.name}")
    if icns.read_bytes() != ours.read_bytes():
        raise IconError(f"{path.name}'s {icns.name} is not icons/clean-voice.icns")
    return f"{path.name}: {icns.name} is icons/clean-voice.icns"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("exe", "msi", "app"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    check = {"exe": check_exe, "msi": check_msi, "app": check_app}[args.what]
    try:
        print(check(args.path))
    except (IconError, OSError) as exc:
        print(f"icon check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
