"""Build the RNNoise shared libraries that Clean Voice loads with ctypes.

No RNNoise binding on PyPI installs from wheels on Windows and macOS 13+ with MIT-compatible dependencies, so Clean
Voice ships librnnoise itself (BSD-3-Clause, Xiph.Org) and binds it with ``clean_voice.rnnoise`` (ctypes).

Run from the project folder after ``pip install --group native``:

    .venv/Scripts/python tools/rnnoise/build_rnnoise.py

It downloads the RNNoise 0.2 release tarball (checked against a pinned SHA-256), cross-compiles it with Zig for
Windows x64 and for macOS 13 arm64 and x86_64, merges the two macOS builds into one universal dylib, and writes
the results and the RNNoise license to ``src/clean_voice/native/``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "src" / "clean_voice" / "native"
CACHE_DIR = ROOT / ".cache" / "zig"

URL = "https://github.com/xiph/rnnoise/releases/download/v0.2/rnnoise-0.2.tar.gz"
SHA256 = "90fce4b00b9ff24c08dbfe31b82ffd43bae383d85c5535676d28b0a2b11c0d37"
TOP = "rnnoise-0.2"

# The library sources, as listed in RNNoise's Makefile.am (no x86 run-time dispatch: plain C runs everywhere).
SOURCES = [
    "src/denoise.c",
    "src/rnn.c",
    "src/pitch.c",
    "src/kiss_fft.c",
    "src/celt_lpc.c",
    "src/nnet.c",
    "src/nnet_default.c",
    "src/parse_lpcnet_weights.c",
    "src/rnnoise_data.c",
    "src/rnnoise_tables.c",
]
# x86_64 builds stay at the SSE2 baseline every x64 CPU has; RNNoise's #warning about AVX is silenced.
COMMON_FLAGS = ["-shared", "-O2", "-fvisibility=hidden", "-Wno-#warnings", "-DRNNOISE_BUILD", "-Iinclude", "-Isrc"]

OS_SUPPORT_SHIM = """#ifndef OS_SUPPORT_H
#define OS_SUPPORT_H
#include "common.h"
#define OPUS_CLEAR(dst, n) RNN_CLEAR(dst, n)
#define OPUS_COPY(dst, src, n) RNN_COPY(dst, src, n)
#define OPUS_MOVE(dst, src, n) RNN_MOVE(dst, src, n)
#endif
"""

MACHO_MAGIC_64 = 0xFEEDFACF
FAT_MAGIC = 0xCAFEBABE
FAT_ALIGN = 14  # 16 KiB, the arm64 page size


def download(dest: Path) -> Path:
    archive = dest / "rnnoise-0.2.tar.gz"
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive.write_bytes(response.read())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"rnnoise-0.2.tar.gz SHA-256 mismatch: {digest}")
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    src = dest / TOP
    # The 0.2 release tarball lacks src/os_support.h, which the NEON (arm64) code includes for OPUS_CLEAR.
    (src / "src" / "os_support.h").write_text(OS_SUPPORT_SHIM, encoding="ascii")
    return src


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
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
        tmp = Path(tmp_name)
        src = download(tmp)
        zig_cc(src, "x86_64-windows-gnu", OUT_DIR / "rnnoise.dll", ["-DWIN32", "-DDLL_EXPORT"])
        thin: list[Path] = []
        for arch in ("aarch64", "x86_64"):
            path = tmp / f"librnnoise-{arch}.dylib"
            zig_cc(src, f"{arch}-macos.13.0", path, ["-Wl,-install_name,@loader_path/librnnoise.dylib"])
            thin.append(path)
        merge_universal(thin, OUT_DIR / "librnnoise.dylib")
        shutil.copyfile(src / "COPYING", OUT_DIR / "RNNOISE-COPYING.txt")
    for leftover in OUT_DIR.glob("*.lib"):
        leftover.unlink()
    for leftover in OUT_DIR.glob("*.pdb"):
        leftover.unlink()
    for path in sorted(OUT_DIR.iterdir()):
        print(f"{path.name}: {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
