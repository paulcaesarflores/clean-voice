"""CI's end-to-end check of the install command (B019). Standard library only; run from the project folder.

    smoke_install.py windows|macos [--arch arm64|x86_64] [--from-old machine|user]

It builds the release folder exactly as the release workflow does (``git archive HEAD`` -> ``release_assets.py``), serves
it from a local HTTP server that answers every file as ``application/octet-stream`` (as GitHub does), then runs the guest's
real command with a stripped PATH (no Python, no git): Windows PowerShell 5.1 ``irm <url> | iex``, a Mac's
``curl -fsSL <url> | bash``, with CLEAN_VOICE_BASE pointing at the same server. Then it checks the install, runs the
command a second time (one install, one entry, still launches) and, on Windows, removes it with the registered
UninstallString. ``--from-old`` first installs v0.2.0's real MSI (Windows; ``machine`` = all users, ``user`` = this user) or
DMG (Mac) from the public release, writes a prefs file and a "Start with Windows" value, runs the command and checks that
one Clean Voice is left, the prefs are untouched and the Run value points at an exe that exists.

Exit code 0 means every check passed; 1 means a check failed (the reason is printed).
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import importlib.util
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from types import ModuleType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = "paulcaesarflores/clean-voice"
OLD_TAG = "v0.2.0"  # the last release that shipped an MSI and a DMG: the one allowed mention (the migration run)
OLD_MSI = f"https://github.com/{REPO}/releases/download/{OLD_TAG}/Clean-Voice-Windows.msi"
OLD_DMG = f"https://github.com/{REPO}/releases/download/{OLD_TAG}/Clean-Voice-macOS.dmg"
RUN_TIMEOUT_S = 3600
UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
WINDOWS_PATH = r"C:\Windows\System32;C:\Windows;C:\Windows\System32\WindowsPowerShell\v1.0"
MAC_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
EXE_NAME = "Clean Voice.exe"
APP_BUNDLE = "Clean Voice.app"
PREFS = '{"smoke": "these must survive the install"}\n'


class SmokeError(Exception):
    """A check failed; the message says why."""


def sibling(name: str) -> ModuleType:
    """A tools/ci script loaded from its file (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    if spec is None or spec.loader is None:
        raise SmokeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ the release folder and the local server


def build_release(out: Path, root: Path = ROOT) -> str:
    """The release files for the current commit, built the way the release workflow builds them. Returns the tag."""
    assets = sibling("release_assets")
    archive = out.parent / f"{out.name}-source.zip"
    subprocess.run(["git", "archive", "--format=zip", "-o", str(archive), "HEAD"], cwd=root, check=True)
    versions = assets.app_versions(root)
    tag = "v" + min(set(versions.values()))
    assets.prepare(out, archive, root / "tools" / "install", tag, versions)
    return tag


class OctetStreamHandler(http.server.SimpleHTTPRequestHandler):
    """Every file as application/octet-stream, as GitHub serves release files (so ``irm`` returns bytes, not text)."""

    def guess_type(self, path: str | os.PathLike[str]) -> str:
        return "application/octet-stream"

    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def serve(directory: Path) -> Iterator[str]:
    """Serve ``directory`` on 127.0.0.1; yields the base URL. The thread is stopped on exit."""

    def handler(*args: object, **kwargs: object) -> http.server.SimpleHTTPRequestHandler:
        return OctetStreamHandler(*args, directory=str(directory), **kwargs)  # type: ignore[arg-type]

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)  # type: ignore[arg-type]
    thread = threading.Thread(target=server.serve_forever, name="smoke-install-server", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def local_command(platform: str, base: str) -> str:
    """The guest's command with the release files served from ``base``. With the real latest-download link in place of
    ``base`` this is exactly ``clean_voice.updates.install_command_text`` (tests/test_ci_smoke_install.py)."""
    if platform == "windows":
        return f"irm {base}/install.ps1 | iex"
    return f"curl -fsSL {base}/install.sh | bash"


def stripped_env(platform: str, base: str, environ: Mapping[str, str]) -> dict[str, str]:
    """The environment the command runs in: no Python and no git on PATH, no update variables, the release files at
    ``base``. Everything else (SystemRoot, TEMP, APPDATA, HOME...) is kept: a guest has them."""
    env = {k: v for k, v in environ.items() if k.upper() != "PATH" and not k.upper().startswith("CLEAN_VOICE_")}
    env["PATH"] = WINDOWS_PATH if platform == "windows" else MAC_PATH
    env["CLEAN_VOICE_BASE"] = base
    return env


def powershell_exe() -> str:
    return str(
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )


def command_argv(platform: str, base: str) -> list[str]:
    command = local_command(platform, base)
    if platform == "windows":
        return [powershell_exe(), "-NoProfile", "-Command", command]
    return ["/bin/bash", "-c", command]


def run_command(platform: str, base: str, log: Path) -> None:
    """Run the guest's command; its output goes to the CI log and ``log``."""
    argv = command_argv(platform, base)
    print(f"run: {argv}", flush=True)
    done = subprocess.run(
        argv,
        env=stripped_env(platform, base, os.environ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )
    log.write_text(f"{done.stdout}\n--- stderr ---\n{done.stderr}", encoding="utf-8")
    print(f"{done.stdout}\n--- stderr ---\n{done.stderr}\nexit code {done.returncode}", flush=True)
    if done.returncode != 0:
        raise SmokeError(f"the install command exited {done.returncode}")


# ------------------------------------------------------------------ pure judgements


def registry_problems(values: Mapping[str, object], folder: str, version: str) -> list[str]:
    """What is wrong with Clean Voice's Installed-apps entry (an empty list when it is right)."""
    problems = []
    for name, want in (("DisplayName", "Clean Voice"), ("Publisher", "Clean Voice"), ("DisplayVersion", version)):
        if values.get(name) != want:
            problems.append(f"{name} is {values.get(name)!r}, expected {want!r}")
    if values.get("InstallLocation") != folder:
        problems.append(f"InstallLocation is {values.get('InstallLocation')!r}, expected {folder!r}")
    if values.get("DisplayIcon") != str(PureWindowsPath(folder) / EXE_NAME):  # a Windows path, whichever OS judges it
        problems.append(f"DisplayIcon is {values.get('DisplayIcon')!r}")
    uninstall = str(values.get("UninstallString", ""))
    if not uninstall.startswith("powershell.exe") or "Uninstall Clean Voice.ps1" not in uninstall:
        problems.append(f"UninstallString is {uninstall!r}")
    return problems


def plist_problems(info: Mapping[str, object]) -> list[str]:
    """App Nap must be off (DESIGN: as OBS's Info.plist), and the bundle must name its icon."""
    problems = []
    if info.get("LSAppNapIsDisabled") is not True:
        problems.append("Info.plist LSAppNapIsDisabled is not true")
    if not info.get("CFBundleIconFile"):
        problems.append("Info.plist names no CFBundleIconFile")
    return problems


def run_value_problem(value: str | None, exists: Callable[[str], bool]) -> str | None:
    """Why a "Start with Windows" value is wrong, or None. It must name an exe that exists (first word, quotes allowed)."""
    if not value:
        return "the Start with Windows value is gone"
    path = value.split('"')[1] if value.startswith('"') else value.split(" --")[0]
    return None if exists(path) else f"the Start with Windows value points at {path}, which does not exist"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ Windows


def windows_paths() -> dict[str, Path]:
    local, roaming = Path(os.environ["LOCALAPPDATA"]), Path(os.environ["APPDATA"])
    folder = local / "Programs" / "Clean Voice"
    return {
        "folder": folder,
        "exe": folder / EXE_NAME,
        "setup": local / "CVSetup",
        "shortcut": roaming / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Clean Voice" / "Clean Voice.lnk",
        "prefs": local / "Clean Voice" / "settings.json",
    }


def windows_entries() -> list[str]:
    """Every Uninstall entry named Clean Voice, in HKCU and HKLM (64 and 32-bit views): "hive\\key"."""
    assert sys.platform == "win32"
    import winreg

    found = []
    for name, hive, path in (
        ("HKCU", winreg.HKEY_CURRENT_USER, UNINSTALL_KEY),
        ("HKLM", winreg.HKEY_LOCAL_MACHINE, UNINSTALL_KEY),
        ("HKLM32", winreg.HKEY_LOCAL_MACHINE, r"Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
    ):
        try:
            with winreg.OpenKey(hive, path) as root:
                count = winreg.QueryInfoKey(root)[0]
                for i in range(count):
                    sub = winreg.EnumKey(root, i)
                    try:
                        with winreg.OpenKey(root, sub) as key:
                            if winreg.QueryValueEx(key, "DisplayName")[0] == "Clean Voice":
                                found.append(f"{name}\\{sub}")
                    except OSError:
                        continue
        except OSError:
            continue
    return found


def windows_key_values() -> dict[str, object]:
    assert sys.platform == "win32"
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY + r"\CleanVoice") as key:
        count = winreg.QueryInfoKey(key)[1]
        return {winreg.EnumValue(key, i)[0]: winreg.EnumValue(key, i)[1] for i in range(count)}


def windows_run_value() -> str | None:
    assert sys.platform == "win32"
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            return str(winreg.QueryValueEx(key, "Clean Voice")[0])
    except OSError:
        return None


def windows_check_install(version: str, expect_one: bool = True) -> None:
    smoke_app, paths = sibling("smoke_app"), windows_paths()
    for name in ("exe", "shortcut"):
        if not paths[name].is_file():
            raise SmokeError(f"{paths[name]} is missing")
    if not (paths["folder"] / "installed-by-command").is_file():
        raise SmokeError("the installed-by-command marker is missing (Install update needs it)")
    if paths["setup"].exists():
        raise SmokeError(f"the setup folder {paths['setup']} was not removed")
    problems = registry_problems(windows_key_values(), str(paths["folder"]), version)
    if problems:
        raise SmokeError("Installed apps entry: " + "; ".join(problems))
    entries = windows_entries()
    if expect_one and entries != ["HKCU\\CleanVoice"]:
        raise SmokeError(f"expected one Clean Voice entry, found {entries}")
    try:
        smoke_app.check_app([str(paths["exe"])], require_line=True)
    except smoke_app.SmokeError as exc:
        raise SmokeError(str(exc)) from exc
    icons = subprocess.run([sys.executable, str(HERE / "check_icons.py"), "exe", str(paths["exe"])], check=False)
    if icons.returncode != 0:
        raise SmokeError("the installed exe does not carry Clean Voice's icon")


def windows_uninstall() -> None:
    paths = windows_paths()
    command = str(windows_key_values()["UninstallString"])
    print(f"run: cmd /c {command}", flush=True)
    done = subprocess.run(["cmd", "/c", command], stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
    print(done.stdout, done.stderr, flush=True)
    leftovers = [str(paths[n]) for n in ("folder", "shortcut") if paths[n].exists()]
    if leftovers or windows_entries():
        raise SmokeError(f"after the registered uninstall: left {leftovers} {windows_entries()}")


def windows_old_install(scope: str, work: Path) -> None:
    """v0.2.0's real MSI, installed silently for all users (``machine``) or this user (``user``)."""
    msi = work / "old.msi"
    urllib.request.urlretrieve(OLD_MSI, msi)
    properties = ["ALLUSERS=1"] if scope == "machine" else ["ALLUSERS=2", "MSIINSTALLPERUSER=1"]
    print(f"old MSI: {properties}", flush=True)
    done = subprocess.run(["msiexec", "/i", str(msi), "/qn", "/norestart", *properties], check=False)
    if done.returncode not in (0, 3010):
        raise SmokeError(f"the old MSI did not install (msiexec exit {done.returncode})")


def windows_from_old(base: str, scope: str, work: Path, version: str) -> None:
    assert sys.platform == "win32"
    import winreg

    paths = windows_paths()
    windows_old_install(scope, work)
    before = windows_entries()
    if not before:
        raise SmokeError("the old MSI left no Installed apps entry")
    old_exe = next(
        (p for p in (Path(os.environ.get("ProgramFiles", "")) / "Clean Voice" / EXE_NAME, paths["exe"]) if p.is_file()),
        None,
    )
    if old_exe is None:
        raise SmokeError("the old MSI's exe was not found")
    old_hash = sha256(old_exe)
    paths["prefs"].parent.mkdir(parents=True, exist_ok=True)
    paths["prefs"].write_text(PREFS, encoding="utf-8")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
        winreg.SetValueEx(key, "Clean Voice", 0, winreg.REG_SZ, f'"{old_exe}" --minimized')
    run_command("windows", base, work / "install-from-old.log")
    windows_check_install(version, expect_one=False)
    entries = windows_entries()
    if entries != ["HKCU\\CleanVoice"]:
        raise SmokeError(f"expected one Clean Voice after replacing the old MSI, found {entries} (was {before})")
    if sha256(paths["exe"]) == old_hash:
        raise SmokeError("the exe is still the old MSI's")
    if paths["prefs"].read_text(encoding="utf-8") != PREFS:
        raise SmokeError("the prefs file changed")
    problem = run_value_problem(windows_run_value(), lambda p: Path(p).is_file())
    if problem:
        raise SmokeError(problem)
    print("from-old: PASS", flush=True)


# ------------------------------------------------------------------ macOS


def mac_app_paths() -> list[Path]:
    return [Path("/Applications") / APP_BUNDLE, Path.home() / "Applications" / APP_BUNDLE]


def mac_installed() -> Path:
    found = [p for p in mac_app_paths() if p.is_dir()]
    if len(found) != 1:
        raise SmokeError(f"expected one {APP_BUNDLE} in Applications or ~/Applications, found {found}")
    return found[0]


def _run(command: Sequence[str], timeout: float = 600) -> subprocess.CompletedProcess[str]:
    print(f"run: {list(command)}", flush=True)
    return subprocess.run(list(command), capture_output=True, text=True, timeout=timeout, check=False)


def mac_check_install(arch: str | None) -> None:
    smoke_app = sibling("smoke_app")
    if arch:
        try:
            smoke_app.check_arch(arch)
        except smoke_app.SmokeError as exc:
            raise SmokeError(str(exc)) from exc
    app = mac_installed()
    binary = app / "Contents" / "MacOS" / "Clean Voice"
    marker = Path.home() / "Library" / "Application Support" / "Clean Voice" / "installed-by-command"
    if not marker.is_file() or marker.read_text().strip() != sha256(binary):
        raise SmokeError("the installed-by-command marker is missing or does not match the app")
    if Path.home().joinpath("Library", "Caches", "CVSetup").exists():
        raise SmokeError("the setup folder was not removed")
    binaries = [binary, *sorted(p for p in app.rglob("*") if p.suffix in (".so", ".dylib") and p.is_file())]
    problems: list[str] = []
    for path in binaries:
        lipo = _run(["lipo", "-archs", str(path)], timeout=60)
        if lipo.returncode != 0:
            problems.append(f"{path}: lipo exited {lipo.returncode}: {lipo.stderr.strip()}")
        elif missing := smoke_app.missing_archs(smoke_app.parse_lipo_archs(lipo.stdout)):
            problems.append(f"{path}: missing {', '.join(missing)}")
    print(f"universal check: {len(binaries)} binaries, {len(problems)} problems", flush=True)
    if problems:
        raise SmokeError("not universal:\n" + "\n".join(problems))
    sign = _run(["codesign", "--verify", "--deep", "--strict", str(app)])
    if sign.returncode != 0:
        raise SmokeError(f"codesign --verify failed: {sign.stderr}")
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    if plist_problems(info):
        raise SmokeError("; ".join(plist_problems(info)))
    icons = subprocess.run([sys.executable, str(HERE / "check_icons.py"), "app", str(app)], check=False)
    if icons.returncode != 0:
        raise SmokeError("the installed app does not carry Clean Voice's icon")
    try:
        smoke_app.check_app(
            [str(binary)], require_line=False
        )  # stdout goes to the system log: the exit code is the proof
    except smoke_app.SmokeError as exc:
        raise SmokeError(str(exc)) from exc


def mac_from_old(base: str, work: Path, arch: str | None) -> None:
    dmg = work / "old.dmg"
    urllib.request.urlretrieve(OLD_DMG, dmg)
    mount = Path(tempfile.mkdtemp(prefix="clean-voice-old-dmg-"))
    attach = _run(["hdiutil", "attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint", str(mount), str(dmg)])
    if attach.returncode != 0:
        raise SmokeError(f"hdiutil attach exited {attach.returncode}: {attach.stderr}")
    old = Path("/Applications") / APP_BUNDLE
    try:
        copied = _run(["ditto", str(next(mount.glob("*.app"))), str(old)])
        if copied.returncode != 0:
            raise SmokeError(f"could not copy the old app to Applications: {copied.stderr}")
    finally:
        _run(["hdiutil", "detach", str(mount)], timeout=120)
    old_hash = sha256(old / "Contents" / "MacOS" / "Clean Voice")
    prefs = Path.home() / "Library" / "Preferences" / "Clean Voice" / "settings.json"
    prefs.parent.mkdir(parents=True, exist_ok=True)
    prefs.write_text(PREFS, encoding="utf-8")
    run_command("macos", base, work / "install-from-old.log")
    mac_check_install(arch)
    if sha256(mac_installed() / "Contents" / "MacOS" / "Clean Voice") == old_hash:
        raise SmokeError("the app is still the old DMG's")
    if prefs.read_text(encoding="utf-8") != PREFS:
        raise SmokeError("the prefs file changed")
    print("from-old: PASS", flush=True)


# ------------------------------------------------------------------ the run


def smoke(platform: str, arch: str | None, from_old: str | None) -> None:
    work = Path(tempfile.mkdtemp(prefix="clean-voice-smoke-"))
    release = work / "release"
    try:
        tag = build_release(release)
        version = tag.removeprefix("v")
        print(f"release files: {sorted(p.name for p in release.iterdir())}", flush=True)
        with serve(release) as base:
            if from_old:
                if platform == "windows":
                    windows_from_old(base, from_old, work, version)
                else:
                    mac_from_old(base, work, arch)
            else:
                run_command(platform, base, work / "install.log")
                if platform == "windows":
                    windows_check_install(version)
                else:
                    mac_check_install(arch)
            run_command(platform, base, work / "install-again.log")  # a second run: one install, still launches
            if platform == "windows":
                windows_check_install(version)
                windows_uninstall()
            else:
                mac_check_install(arch)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="smoke_install", description="Run the install command against a local server."
    )
    parser.add_argument("platform", choices=("windows", "macos"))
    parser.add_argument("--arch", choices=("arm64", "x86_64"), help="refuse to run on another CPU (the Intel job)")
    parser.add_argument(
        "--from-old", choices=("machine", "user"), help="first install v0.2.0's MSI (Windows) or DMG (Mac)"
    )
    args = parser.parse_args(argv)
    try:
        smoke(args.platform, args.arch, args.from_old)
    except (SmokeError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
