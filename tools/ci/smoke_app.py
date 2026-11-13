"""Launch-check the packaged Clean Voice app (CI, J04). Standard library only, so it runs before any install.

Subcommands:
  exe COMMAND... [--require-line] [-- APP_ARGS...]
      Run COMMAND --quit-after-ms 3000 APP_ARGS. Passes on exit code 0 (and, with --require-line, the
      "window shown" line in stdout). COMMAND is the program, optionally followed by its own arguments
      (e.g. `python -m clean_voice`).
  msi DIR
      Silent per-user install of the one *.msi in DIR, launch check of the installed exe, silent uninstall.
  dmg DIR --arch arm64|x86_64
      Check the machine, mount the one *.dmg in DIR, check every binary is universal, launch check.

Exit code 0 means every check passed; 1 means a check failed (the reason is printed).
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

# Must match clean_voice.app.SHOWN_MESSAGE; the app's code is not importable where this runs (the Intel job).
SHOWN_LINE = "Clean Voice window shown"
QUIT_AFTER_MS = 3000
RUN_TIMEOUT_S = 120
REQUIRED_ARCHS = ("arm64", "x86_64")
APP_DIR = "Clean Voice"
EXE_NAME = "Clean Voice.exe"
MAC_BINARY = "Clean Voice"
LOG_TAIL_LINES = 60
# The msi check also needs the "window shown" line in stdout: J04 Step 6 confirmed the GUI-subsystem stub
# (build/.../Clean Voice.exe) writes it to a piped stdout.
MSI_REQUIRE_LINE = True


class SmokeError(Exception):
    """A launch check failed; the message says why."""


def find_single(directory: Path, pattern: str) -> Path:
    """The one file in ``directory`` matching ``pattern``; anything else is an error."""
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        found = ", ".join(m.name for m in matches) or "no files"
        raise SmokeError(f"expected exactly one {pattern} in {directory}, found {len(matches)}: {found}")
    return matches[0]


def find_installed_exe(roots: Iterable[Path]) -> Path | None:
    """The installed ``Clean Voice/Clean Voice.exe`` under the first root that has it (no author folder)."""
    for root in roots:
        exe = root / APP_DIR / EXE_NAME
        if exe.is_file():
            return exe
    return None


def parse_lipo_archs(output: str) -> set[str]:
    """The architectures ``lipo -archs`` printed."""
    return set(output.split())


def missing_archs(archs: Iterable[str]) -> list[str]:
    """The required architectures a binary lacks, sorted."""
    have = set(archs)
    return sorted(arch for arch in REQUIRED_ARCHS if arch not in have)


def check_arch(expected: str, machine: str | None = None) -> None:
    """Refuse to run on the wrong CPU, so the Intel job proves it really ran on Intel."""
    actual = platform.machine() if machine is None else machine
    if actual != expected:
        raise SmokeError(f"this machine is {actual}, expected {expected}")


def judge_run(returncode: int, stdout: str, require_line: bool) -> str | None:
    """Why a launch failed, or None when it passed."""
    if returncode != 0:
        return f"the app exited with exit code {returncode}"
    if require_line and SHOWN_LINE not in (line.strip() for line in stdout.splitlines()):
        return f"{SHOWN_LINE!r} is not in the app's output"
    return None


def check_app(
    command: Sequence[str], *, require_line: bool, app_args: Sequence[str] = (), timeout: float = RUN_TIMEOUT_S
) -> None:
    """Run the app with --quit-after-ms and judge the run; raises SmokeError when it fails."""
    program = Path(command[0])
    # An absolute program path: CreateProcess does not resolve a relative `dir/app.exe` with forward slashes.
    first = str(program.resolve()) if program.is_file() else command[0]
    full = [first, *command[1:], "--quit-after-ms", str(QUIT_AFTER_MS), *app_args]
    print(f"run: {full}", flush=True)
    try:
        proc = subprocess.run(full, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise SmokeError(f"the app did not exit within {timeout} s") from exc
    except OSError as exc:
        raise SmokeError(f"the app could not start: {exc}") from exc
    print(
        f"exit code {proc.returncode}\n--- app stdout ---\n{proc.stdout}--- app stderr ---\n{proc.stderr}", flush=True
    )
    problem = judge_run(proc.returncode, proc.stdout, require_line)
    if problem:
        raise SmokeError(problem)


def read_log_tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    """The last lines of an msiexec log (UTF-16 with a BOM, or 8-bit)."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"(no log at {path}: {exc})"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16", "replace")
    else:
        text = data.decode("utf-8", "replace")
    return "\n".join(text.splitlines()[-lines:])


def _run(command: Sequence[str], timeout: float = 600) -> subprocess.CompletedProcess[str]:
    print(f"run: {list(command)}", flush=True)
    return subprocess.run(list(command), capture_output=True, text=True, timeout=timeout, check=False)


def smoke_msi(directory: Path) -> None:
    msi = find_single(directory, "*.msi")
    log = directory / "msi-install.log"
    install = _run(["msiexec", "/i", str(msi), "/qn", "/norestart", "/l*v", str(log)])
    if install.returncode != 0:
        raise SmokeError(f"msiexec install exited {install.returncode}; log tail:\n{read_log_tail(log)}")
    # Per-user installs land in %LOCALAPPDATA%\Programs, per-machine ones in %ProgramFiles% (Briefcase's .wxs).
    roots: list[Path] = []
    if local := os.environ.get("LOCALAPPDATA"):
        roots.append(Path(local) / "Programs")
    if program_files := os.environ.get("ProgramFiles"):
        roots.append(Path(program_files))
    exe = find_installed_exe(roots)
    failure: SmokeError | None = None
    try:
        if exe is None:
            searched = ", ".join(str(r) for r in roots)
            raise SmokeError(f"{APP_DIR}\\{EXE_NAME} not found under {searched}; log tail:\n{read_log_tail(log)}")
        print(f"installed: {exe}", flush=True)
        check_app([str(exe)], require_line=MSI_REQUIRE_LINE)
    except SmokeError as exc:
        failure = exc
    uninstall = _run(["msiexec", "/x", str(msi), "/qn", "/norestart"])
    if failure is not None:
        raise failure
    if uninstall.returncode != 0:
        raise SmokeError(f"msiexec uninstall exited {uninstall.returncode}")
    if exe is not None and exe.exists():
        raise SmokeError(f"{exe} is still there after the uninstall")


def smoke_dmg(directory: Path, arch: str) -> None:
    check_arch(arch)
    dmg = find_single(directory, "*.dmg")
    mount = Path(tempfile.mkdtemp(prefix="clean-voice-dmg-"))
    attach = _run(["hdiutil", "attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint", str(mount), str(dmg)])
    if attach.returncode != 0:
        raise SmokeError(f"hdiutil attach exited {attach.returncode}: {attach.stderr}")
    try:
        app = find_single(mount, "*.app")
        binary = app / "Contents" / "MacOS" / MAC_BINARY
        binaries = [binary, *sorted(p for p in app.rglob("*") if p.suffix in (".so", ".dylib") and p.is_file())]
        problems: list[str] = []
        for path in binaries:
            lipo = _run(["lipo", "-archs", str(path)], timeout=60)
            if lipo.returncode != 0:
                problems.append(f"{path}: lipo exited {lipo.returncode}: {lipo.stderr.strip()}")
                continue
            if missing := missing_archs(parse_lipo_archs(lipo.stdout)):
                problems.append(f"{path}: missing {', '.join(missing)}")
        print(f"universal check: {len(binaries)} binaries, {len(problems)} problems", flush=True)
        if problems:
            raise SmokeError("not universal:\n" + "\n".join(problems))
        # stdout goes to the system log on macOS, so the exit code is the proof.
        check_app([str(binary)], require_line=False)
    finally:
        _run(["hdiutil", "detach", str(mount)], timeout=120)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smoke_app", description="Launch-check the packaged Clean Voice app.")
    sub = parser.add_subparsers(dest="mode", required=True)
    # The exe words are split off by hand in main(): its command may hold words that start with a dash.
    sub.add_parser("exe", help="launch-check a program: exe COMMAND... [--require-line] [-- APP_ARGS...]")
    msi = sub.add_parser("msi", help="install, launch-check and uninstall the one MSI in DIR")
    msi.add_argument("dir", type=Path)
    dmg = sub.add_parser("dmg", help="mount the one DMG in DIR, check it is universal, launch-check it")
    dmg.add_argument("dir", type=Path)
    dmg.add_argument("--arch", required=True, choices=REQUIRED_ARCHS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    words = list(sys.argv[1:] if argv is None else argv)
    app_args: list[str] = []
    if "--" in words:
        cut = words.index("--")
        words, app_args = words[:cut], words[cut + 1 :]
    parser = _parser()
    if words[:1] == ["exe"]:
        command = [w for w in words[1:] if w != "--require-line"]
        if not command:
            parser.error("exe needs a COMMAND")
        return _report(lambda: check_app(command, require_line="--require-line" in words, app_args=app_args))
    args = parser.parse_args(words)
    if app_args:
        parser.error("app arguments after -- are for the exe subcommand only")
    if args.mode == "msi":
        return _report(lambda: smoke_msi(args.dir))
    return _report(lambda: smoke_dmg(args.dir, args.arch))


def _report(check: Callable[[], None]) -> int:
    try:
        check()
    except SmokeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
