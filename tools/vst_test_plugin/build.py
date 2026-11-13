"""Build Clean Voice's VST 2 test plug-in (``gain_plugin.c``) for the host's tests (J10 plan Step 6).

Run from the project folder after ``pip install --group native``:

    .venv/Scripts/python tools/vst_test_plugin/build.py

Like ``tools/rnnoise/build_rnnoise.py`` it compiles with Zig, on any OS, for Windows x64 and for macOS 13 arm64 and
x86_64 (merged into one universal binary). It writes ``.cache/vst_test_plugin/CleanVoiceTestGain.dll`` and the
bundle ``.cache/vst_test_plugin/CleanVoiceTestGain.vst``. The tests (``tests/test_vst_plugin.py``,
``tests/test_vst_editor.py``) load them, and skip when they are not built. The plug-in is never shipped.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT_DIR = ROOT / ".cache" / "vst_test_plugin"
CACHE_DIR = ROOT / ".cache" / "zig"
NAME = "CleanVoiceTestGain"
DLL = OUT_DIR / f"{NAME}.dll"
BUNDLE = OUT_DIR / f"{NAME}.vst"

sys.path.insert(0, str(ROOT / "tools" / "rnnoise"))
from build_rnnoise import merge_universal  # pyright: ignore[reportMissingImports]  # the same fat Mach-O writer

FLAGS = ["-shared", "-O2", "-g0", "-fvisibility=hidden", "-std=c11", "-Wall", "-Werror", f"-I{HERE}"]


def zig_cc(target: str, output: Path, extra: list[str]) -> None:
    env = dict(os.environ, ZIG_GLOBAL_CACHE_DIR=str(CACHE_DIR), ZIG_LOCAL_CACHE_DIR=str(CACHE_DIR / "local"))
    cmd = [sys.executable, "-m", "ziglang", "cc", "-target", target, *FLAGS, *extra, str(HERE / "gain_plugin.c")]
    subprocess.run([*cmd, "-o", str(output)], cwd=HERE, env=env, check=True)


def build_windows() -> None:
    zig_cc("x86_64-windows-gnu", DLL, ["-luser32"])
    for leftover in (*OUT_DIR.glob("*.lib"), *OUT_DIR.glob("*.pdb")):
        leftover.unlink()


def build_macos() -> None:
    macos = BUNDLE / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
        thin: list[Path] = []
        for arch in ("aarch64", "x86_64"):
            path = Path(tmp_name) / f"{NAME}-{arch}"
            zig_cc(f"{arch}-macos.13.0", path, [f"-Wl,-install_name,@loader_path/{NAME}"])
            thin.append(path)
        merge_universal(thin, macos / NAME)
    info = {
        "CFBundleExecutable": NAME,
        "CFBundleIdentifier": "org.cleanvoice.tests.gain",
        "CFBundleName": NAME,
        "CFBundlePackageType": "BNDL",
        "CFBundleSignature": "????",
        "CFBundleVersion": "1",
    }
    with open(BUNDLE / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(info, f)
    (BUNDLE / "Contents" / "PkgInfo").write_bytes(b"BNDL????")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    build_windows()
    build_macos()
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(OUT_DIR)}: {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
