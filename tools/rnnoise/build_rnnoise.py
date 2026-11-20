"""Build the RNNoise shared libraries that Clean Voice loads with ctypes.

Clean Voice ships the RNNoise that OBS Studio bundles (BSD-3-Clause, Xiph.Org), built from OBS's
``plugins/obs-filters/rnnoise`` folder (``reference/obs-studio``) and bound with ``clean_voice.rnnoise`` (ctypes).

Run from the project folder after ``pip install --group native``:

    .venv/Scripts/python tools/rnnoise/build_rnnoise.py

It cross-compiles OBS's copy with Zig for Windows x64 and for macOS 13 arm64 and x86_64, merges the two macOS builds
into one universal dylib, and writes the results and the RNNoise license to ``src/clean_voice/native/``.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "src" / "clean_voice" / "native"
CACHE_DIR = ROOT / ".cache" / "zig"
SRC = ROOT / "reference" / "obs-studio" / "plugins" / "obs-filters" / "rnnoise"
TOP = "rnnoise"

# The source list OBS builds: plugins/obs-filters/cmake/rnnoise.cmake:13-34.
SOURCES = [
    "src/denoise.c",
    "src/rnn.c",
    "src/pitch.c",
    "src/kiss_fft.c",
    "src/celt_lpc.c",
    "src/rnn_data.c",
    "src/rnn_reader.c",
]
# -DCOMPILE_OPUS as rnnoise.cmake:40; x86_64 builds stay at the SSE2 baseline every x64 CPU has.
COMMON_FLAGS = [
    "-shared",
    "-O2",
    "-fvisibility=hidden",
    "-Wno-#warnings",
    "-DRNNOISE_BUILD",
    "-DCOMPILE_OPUS",
    "-Iinclude",
    "-Isrc",
]

MACHO_MAGIC_64 = 0xFEEDFACF
FAT_MAGIC = 0xCAFEBABE
FAT_ALIGN = 14  # 16 KiB, the arm64 page size


def zig_cc(src: Path, target: str, output: Path, extra: list[str]) -> None:
    env = dict(os.environ, ZIG_GLOBAL_CACHE_DIR=str(CACHE_DIR), ZIG_LOCAL_CACHE_DIR=str(CACHE_DIR / "local"))
    # No debug info and no build folder in the output: the libraries are published, and the paths name the builder's
    # user folder (release r1).
    private = ["-g0", f"-ffile-prefix-map={src}={TOP}", f"-ffile-prefix-map={src.as_posix()}={TOP}"]
    cmd = [sys.executable, "-m", "ziglang", "cc", "-target", target, *COMMON_FLAGS, *private, *extra, *SOURCES]
    cmd += ["-o", str(output)]
    subprocess.run(cmd, cwd=src, env=env, check=True)


def merge_universal(slices: list[Path], output: Path) -> None:
    """Write a fat Mach-O holding each thin 64-bit slice (what ``lipo -create`` does)."""
    blobs = [path.read_bytes() for path in slices]
    headers: list[tuple[int, int]] = []
    for path, blob in zip(slices, blobs, strict=True):
        magic, cputype, cpusubtype = struct.unpack_from("<III", blob)
        if magic != MACHO_MAGIC_64:
            raise SystemExit(f"{path} is not a thin 64-bit Mach-O file")
        headers.append((cputype, cpusubtype))
    align = 1 << FAT_ALIGN
    offset = align  # the fat header and arch table fit in the first page
    table = bytearray(struct.pack(">II", FAT_MAGIC, len(blobs)))
    body = bytearray()
    for (cputype, cpusubtype), blob in zip(headers, blobs, strict=True):
        body += b"\0" * (offset - align - len(body))  # pad up to this slice's aligned offset, as lipo does
        table += struct.pack(">IIIII", cputype, cpusubtype, offset, len(blob), FAT_ALIGN)
        body += blob
        offset += len(blob) + (-len(blob) % align)
    # The file ends with the last slice: codesign's strict validation rejects any data after it.
    output.write_bytes(bytes(table) + b"\0" * (align - len(table)) + bytes(body))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    scratch = CACHE_DIR / "thin"
    scratch.mkdir(parents=True, exist_ok=True)
    zig_cc(SRC, "x86_64-windows-gnu", OUT_DIR / "rnnoise.dll", ["-DWIN32", "-DDLL_EXPORT"])
    thin: list[Path] = []
    for arch in ("aarch64", "x86_64"):
        path = scratch / f"librnnoise-{arch}.dylib"
        zig_cc(SRC, f"{arch}-macos.13.0", path, ["-Wl,-install_name,@loader_path/librnnoise.dylib"])
        thin.append(path)
    merge_universal(thin, OUT_DIR / "librnnoise.dylib")
    shutil.copyfile(SRC / "COPYING", OUT_DIR / "RNNOISE-COPYING.txt")
    for leftover in OUT_DIR.glob("*.lib"):
        leftover.unlink()
    for leftover in OUT_DIR.glob("*.pdb"):
        leftover.unlink()
    for path in sorted(OUT_DIR.iterdir()):
        print(f"{path.name}: {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
