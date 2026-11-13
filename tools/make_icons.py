"""Write Clean Voice's app icon files from the mark (src/clean_voice/mark.py, the one source; J10 Step 11).

    .venv/Scripts/python tools/make_icons.py [--out icons] [--sheet docs/ui/rounds/<N>/icon-sheet.png]

- ``icons/clean-voice.ico``: 16, 20, 24, 32, 40, 48, 64 and 256 px (Briefcase needs 16/32/48/64/256).
- ``icons/clean-voice.icns``: 16 to 1024 px, with the @2x members.
- ``--sheet``: every size side by side at 1:1 and enlarged, with the two-bar 16 px fallback beside the three-bar one.

Both files are written here with PNG members, by a small writer of their container formats, so no image plugin is
needed. ``pyproject.toml`` points Briefcase at them (``icon = "icons/clean-voice"``).
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 256)
# The icns members with PNG data: (type, pixel size). ic11-ic14 are the @2x members of 16, 32, 128 and 256.
ICNS_MEMBERS = (
    ("icp4", 16),
    ("ic11", 32),
    ("icp5", 32),
    ("ic12", 64),
    ("icp6", 64),
    ("ic07", 128),
    ("ic13", 256),
    ("ic08", 256),
    ("ic14", 512),
    ("ic09", 512),
    ("ic10", 1024),
)


def png(size: int, two_bars: bool = False) -> bytes:
    """The mark at ``size`` px as PNG bytes (straight alpha)."""
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice
    from PySide6.QtGui import QImage, QImageWriter, QPainter

    from clean_voice import mark

    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    p = QPainter(image)
    mark.paint_mark(p, size, mark.TWO_BARS if two_bars else mark.BARS)
    p.end()
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not QImageWriter(buffer, QByteArray(b"png")).write(image.convertToFormat(QImage.Format.Format_ARGB32)):
        raise OSError(f"could not encode the {size} px icon")
    buffer.close()
    return bytes(data.data())


def ico(members: dict[int, bytes]) -> bytes:
    """An .ico of PNG members: ICONDIR, one ICONDIRENTRY each (a 256 px size is written as 0), then the PNGs."""
    sizes = sorted(members)
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = len(header) + 16 * len(sizes)
    entries, blobs = b"", b""
    for size in sizes:
        data = members[size]
        side = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", side, side, 0, 0, 1, 32, len(data), offset + len(blobs))
        blobs += data
    return header + entries + blobs


def icns(members: list[tuple[str, bytes]]) -> bytes:
    """An .icns of PNG members: "icns" and the file's length, then each member's type, length and data."""
    body = b"".join(kind.encode("ascii") + struct.pack(">I", 8 + len(data)) + data for kind, data in members)
    return b"icns" + struct.pack(">I", 8 + len(body)) + body


def read_ico(data: bytes) -> dict[int, bytes]:
    """The PNG members of an .ico, by pixel size."""
    reserved, kind, count = struct.unpack_from("<HHH", data)
    assert (reserved, kind) == (0, 1), "not an .ico"
    out = {}
    for i in range(count):
        width, _, _, _, _, _, length, offset = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
        out[width or 256] = data[offset : offset + length]
    return out


def read_icns(data: bytes) -> list[tuple[str, bytes]]:
    """The members of an .icns, in file order."""
    assert data[:4] == b"icns" and struct.unpack_from(">I", data, 4)[0] == len(data), "not an .icns"
    out, at = [], 8
    while at < len(data):
        kind = data[at : at + 4].decode("ascii")
        (length,) = struct.unpack_from(">I", data, at + 4)
        out.append((kind, data[at + 8 : at + length]))
        at += length
    return out


def sheet(path: Path) -> None:
    """Every size at 1:1 on a light and a dark strip, then 16 and 32 px enlarged 8x (three bars, and two at 16)."""
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter

    sizes = (16, 20, 24, 32, 40, 48, 64, 128, 256)
    big = [(16, False), (16, True), (32, False)]
    width = sum(sizes) + 24 * (len(sizes) + 1)
    height = 2 * (256 + 48) + 32 * 8 + 80
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(QColor("#F7F9FA"))
    p = QPainter(image)
    font = QFont()
    font.setPixelSize(13)
    p.setFont(font)
    for row, ground in enumerate(("#F7F9FA", "#1A232B")):
        top = row * (256 + 48)
        p.fillRect(QRect(0, top, width, 256 + 48), QColor(ground))
        x = 24
        for size in sizes:
            icon = QImage.fromData(png(size))
            p.drawImage(x, top + 24 + 256 - size, icon)
            p.setPen(QColor("#56636F" if row == 0 else "#97A4AF"))
            p.drawText(QRect(x - 12, top + 24 + 256 + 4, size + 24, 18), Qt.AlignmentFlag.AlignHCenter, str(size))
            x += size + 24
    top = 2 * (256 + 48) + 24
    x = 24
    for size, two in big:
        icon = QImage.fromData(png(size, two)).scaled(size * 8, size * 8, mode=Qt.TransformationMode.FastTransformation)
        p.drawImage(x, top, icon)
        p.setPen(QColor("#15212B"))
        label = f"{size} px x8" + (", two bars" if two else "")
        p.drawText(QRect(x, top + size * 8 + 4, 260, 18), Qt.AlignmentFlag.AlignLeft, label)
        x += size * 8 + 48
    p.end()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(path)):
        raise OSError(f"could not write {path}")


def make(out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    ico_path, icns_path = out / "clean-voice.ico", out / "clean-voice.icns"
    ico_path.write_bytes(ico({size: png(size) for size in ICO_SIZES}))
    icns_path.write_bytes(icns([(kind, png(size)) for kind, size in ICNS_MEMBERS]))
    return ico_path, icns_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "icons", help="folder for the .ico and .icns")
    parser.add_argument("--sheet", type=Path, help="also write the icon sheet here (a PNG)")
    args = parser.parse_args(argv)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # The offscreen platform has no font list of its own (the sheet's labels): point it at the system fonts.
    for fonts in (Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts", Path("/System/Library/Fonts")):
        if fonts.is_dir():
            os.environ.setdefault("QT_QPA_FONTDIR", str(fonts))
            break
    from PySide6.QtGui import QGuiApplication

    _app = QGuiApplication.instance() or QGuiApplication([sys.argv[0]])
    for path in make(args.out):
        print(path)
    if args.sheet is not None:
        sheet(args.sheet)
        print(args.sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
