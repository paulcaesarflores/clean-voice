"""Check for updates (DESIGN "Updates", user decision 2026-11-11; the original had it too: main.rs:82-93,
main.js:642-662).

It asks GitHub's Releases API for the project's latest release, over HTTPS with the standard library and a generic
User-Agent: no personal data, no telemetry. The repo comes from the app's Homepage URL (pyproject.toml, carried into
the app's metadata by Briefcase); while its owner is still the placeholder (J04), there is nothing to ask. Installing
needs the guest's click: it downloads the release's installer for the OS into a temporary folder, checks it against
the release's ``SHA256SUMS``, then starts it (``msiexec /i`` on Windows, ``open`` on macOS) and the app quits.

Every address is HTTPS on GitHub's own hosts, redirects included, verified with certifi's CA bundle (a Mac bundle's
Python may have no certificates of its own). Asset names are matched after GitHub's renaming of spaces to dots, so a
``SHA256SUMS`` made before the upload still matches. Every failure, whatever raised it, ends in a plain message.

B019 (2026-11-21): a guest installs one of two ways, the installer or the install command (the page's "Way 2").
Install update follows how this copy was installed. An app the command installed (it leaves ``MARKER_NAME``) runs the
same command in a visible terminal window, which rebuilds the new version while Clean Voice keeps running; the
command's script reports through a one-word status file that the app reads (``swap``: the app may quit for the swap;
``failed``). There is no OBS source for that handshake: the user approved it. Windows opens a new PowerShell console
(``subprocess`` ``CREATE_NEW_CONSOLE``); a Mac opens a ``.command`` file in Terminal with Apple's ``open -a Terminal``.
An app the installer installed keeps the download path above.

Everything here runs off the UI thread; ``opener`` defaults to ``open_url``, and the tests replace it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import ssl
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import certifi

USER_AGENT = "Clean-Voice-updater"
API = "https://api.github.com/repos/{repo}/releases/latest"
TIMEOUT_S = 15.0
CHUNK = 64 * 1024
SUMS_NAME = "SHA256SUMS"
INSTALLERS = {"win32": ".msi", "darwin": ".dmg"}
PLACEHOLDER_OWNER = "REPLACE_WITH_GITHUB_OWNER"
# Where a release's files may come from: github.com, and the hosts its download links redirect to.
ALLOWED_HOSTS = frozenset(
    {"api.github.com", "github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}
)
CA_FILE = certifi.where()

READY = "Clean Voice {version} is ready to install"
LATEST = "You have the latest version"
FAILED = "Couldn't check for updates right now"
DOWNLOAD_FAILED = "Couldn't download the update. Try again later."
MISMATCH = "The download didn't match its published checksum, so nothing was installed."
NO_INSTALLER = "This release has no installer for this computer, so nothing was installed."
UNSAFE = "That update didn't come from Clean Voice's release page, so nothing was installed."
CONFIRM = "Clean Voice will close to install the update and you can open it again afterwards."
CONFIRM_COMMAND = (
    "Clean Voice will rebuild the new version in a terminal window (a few minutes), keep working meanwhile, "
    "then close and open again."
)
INSTALLING = "Updating in the terminal window. Clean Voice stays open until the last moment."
UPDATE_FAILED = "The update didn't work, so you still have the old Clean Voice. The window shows why."
LAUNCH_FAILED = "Couldn't open the update window. Try again, or use the command on the download page."
# The permanent-link form of a release file: the latest release's file of that name (docs/site/download.js links()).
LATEST_DOWNLOAD = "https://github.com/{repo}/releases/latest/download/{name}"
SCRIPTS = {"win32": "install.ps1", "darwin": "install.sh"}
CREATE_NEW_CONSOLE = 0x00000010  # subprocess.CREATE_NEW_CONSOLE (Windows only; the constant is absent elsewhere)
STATUS_SWAP = "swap"
STATUS_FAILED = "failed"
# What the install scripts leave so the app knows the command installed it. Windows: this file in the app's folder.
# macOS: this file in Application Support holding the SHA-256 of the app's main binary (a file added inside the signed
# bundle would break its signature), so a DMG copy installed over it no longer matches.
MARKER_NAME = "installed-by-command"

Opener = Callable[..., Any]


class UpdateError(Exception):
    """A failed install step; its text is what the guest reads."""


@dataclass(frozen=True)
class CheckResult:
    state: str  # "newer", "latest" or "failed"
    message: str
    version: str | None = None
    assets: tuple[dict[str, str], ...] = field(default=())

    @property
    def installable(self) -> bool:
        return self.state == "newer"


def homepage() -> str | None:
    """The app's Homepage URL from its own metadata (Briefcase writes Home-page; an editable install, Project-URL)."""
    try:
        meta = metadata.metadata("clean-voice")
    except metadata.PackageNotFoundError:
        return None
    if meta.get("Home-page"):
        return str(meta["Home-page"])
    for entry in meta.get_all("Project-URL") or []:
        label, _, url = str(entry).partition(",")
        if label.strip().lower() == "homepage":
            return url.strip()
    return None


def repo_of(url: str | None) -> str | None:
    """``owner/name`` of a GitHub URL, or None (not GitHub, or the owner not set yet)."""
    match = re.fullmatch(r"https://github\.com/([A-Za-z0-9-]+)/([A-Za-z0-9._-]+?)/?", url or "")
    if match is None or match.group(1) == PLACEHOLDER_OWNER:
        return None
    return f"{match.group(1)}/{match.group(2)}"


def parse_version(tag: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", tag.strip())
    return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None


def is_newer(tag: str, current: str) -> bool:
    new, now = parse_version(tag), parse_version(current)
    return new is not None and now is not None and new > now


def is_safe_url(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    return parts.scheme == "https" and parts.hostname in ALLOWED_HOSTS and not parts.username and not parts.password


def tls_context() -> ssl.SSLContext:
    """Certificate and host name checks on, with certifi's CA bundle."""
    return ssl.create_default_context(cafile=CA_FILE)


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to HTTPS on GitHub's hosts (urllib would follow https to http)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        if not is_safe_url(newurl):
            raise urllib.error.HTTPError(newurl, code, "redirect refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_url(request: urllib.request.Request, timeout: float) -> Any:
    """``urlopen`` with certifi's certificates and the safe redirects."""
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=tls_context()), SafeRedirect())
    return opener.open(request, timeout=timeout)


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})


def check(current: str, repo: str | None, opener: Opener | None = None) -> CheckResult:
    """Ask for the latest release. Never raises: any failure is ``FAILED``."""
    if repo is None:
        return CheckResult("failed", FAILED)
    opener = opener or open_url
    try:
        with opener(_request(API.format(repo=repo)), timeout=TIMEOUT_S) as response:
            data = json.loads(response.read().decode("utf-8"))
        tag = data["tag_name"]
        assets = tuple({"name": str(a["name"]), "url": str(a["browser_download_url"])} for a in data.get("assets", []))
    except Exception:  # noqa: BLE001 - "never raises": a stuck "Checking…" is worse than any message (r1 n1)
        return CheckResult("failed", FAILED)
    if not isinstance(tag, str) or parse_version(tag) is None:
        return CheckResult("failed", FAILED)
    version = ".".join(map(str, parse_version(tag) or ()))
    if is_newer(tag, current):
        return CheckResult("newer", READY.format(version=version), version, assets)
    return CheckResult("latest", LATEST, version, assets)


def asset_for(assets: Any, platform: str) -> tuple[str, str] | None:
    """(name, url) of the installer for ``platform``: the .msi on Windows, the .dmg on macOS."""
    suffix = INSTALLERS.get(platform)
    for asset in assets if suffix else ():
        name = asset["name"]
        url = asset.get("url") or asset.get("browser_download_url")
        if name.lower().endswith(suffix) and url:
            return name, url
    return None


def _github_name(name: str) -> str:
    """An asset's name as GitHub stores it: spaces become dots ("Clean Voice-1.0.msi" -> "Clean.Voice-1.0.msi")."""
    return name.replace(" ", ".")


def parse_sums(text: str) -> dict[str, str]:
    """``sha256sum`` output (``<hex>  <name>`` or ``<hex> *<name>``) as name -> lower-case hex."""
    sums: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9A-Fa-f]+) [ *](.+)", line.strip())
        if match:
            sums[match.group(2)] = match.group(1).lower()
    return sums


def _get(url: str, opener: Opener) -> bytes:
    with opener(_request(url), timeout=TIMEOUT_S) as response:
        return response.read()


def download_verified(
    result: CheckResult,
    platform: str,
    folder: Path,
    progress: Callable[[float], None],
    opener: Opener | None = None,
) -> Path:
    """Download the installer into ``folder`` and check its SHA-256 against the release's ``SHA256SUMS``. Raises
    UpdateError, with the guest's words, on any failure; a file that fails the check is deleted."""
    installer = asset_for(result.assets, platform)
    sums_asset = next((a for a in result.assets if a["name"] == SUMS_NAME), None)
    if installer is None:
        raise UpdateError(NO_INSTALLER)
    if sums_asset is None:
        raise UpdateError(MISMATCH)
    name, url = installer
    if not (is_safe_url(url) and is_safe_url(sums_asset["url"])):
        raise UpdateError(UNSAFE)
    opener = opener or open_url
    path = folder / Path(name).name  # never a path from the server
    try:
        sums = parse_sums(_get(sums_asset["url"], opener).decode("utf-8", "replace"))
        # Matched after GitHub's space-to-dot renaming, whichever side has it.
        expected = {_github_name(k): v for k, v in sums.items()}.get(_github_name(name))
        if expected is None:
            raise UpdateError(MISMATCH)
        digest = hashlib.sha256()
        with opener(_request(url), timeout=TIMEOUT_S) as response, path.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(CHUNK):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if total:
                    progress(min(1.0, done / total))
        if total and done != total:
            raise UpdateError(DOWNLOAD_FAILED)  # cut short: not a checksum problem
        progress(1.0)
    except UpdateError:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:  # any failure is "couldn't download", never a stuck "Downloading…" (r1 n1)
        path.unlink(missing_ok=True)
        raise UpdateError(DOWNLOAD_FAILED) from exc
    if digest.hexdigest() != expected:
        path.unlink(missing_ok=True)
        raise UpdateError(MISMATCH)
    return path


def install_command(path: Path, platform: str) -> list[str]:
    """The command that starts the installer: an argument list, never a shell (and never os.startfile)."""
    if platform == "win32":
        return ["msiexec", "/i", str(path)]
    if platform == "darwin":
        return ["open", str(path)]
    raise UpdateError(NO_INSTALLER)


# ------------------------------------------------------------------ the install command (B019)


def installed_by_command(platform: str, exe: str | Path, home: Path | None = None) -> bool:
    """True when the install command installed this copy (see ``MARKER_NAME``). ``exe`` is the running app's file."""
    try:
        path = Path(exe)
        if platform == "win32":
            return (path.parent / MARKER_NAME).is_file()
        if platform == "darwin":
            marker = (home or Path.home()) / "Library" / "Application Support" / "Clean Voice" / MARKER_NAME
            return marker.read_text(encoding="ascii").strip() == hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError):
        return False
    return False


def install_url(repo: str, platform: str) -> str:
    """The release's install script for ``platform``, the permanent latest-download link."""
    return LATEST_DOWNLOAD.format(repo=repo, name=SCRIPTS[platform])


def install_command_text(repo: str, platform: str) -> str:
    """The command the download page shows, and the one Install update runs: PowerShell's ``irm | iex`` on Windows,
    ``curl | bash`` on a Mac. docs/site/download.js ``commands`` builds the same text (tests/test_download_page.py)."""
    url = install_url(repo, platform)
    if platform == "win32":
        return f"irm {url} | iex"
    return f"curl -fsSL {url} | bash"


@dataclass(frozen=True)
class LaunchSpec:
    """What starts the terminal: the program and its arguments, its environment, the Windows creation flags."""

    argv: list[str]
    env: dict[str, str] | None
    creationflags: int
    script: Path | None  # the Mac's .command file (None on Windows)
    cwd: Path | None = None  # Windows: the child's working directory, outside the install folder (B019-15)


def powershell_exe() -> Path:
    """Windows PowerShell 5.1 by its full System32 path, never a PATH lookup (what the install command is written for)."""
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"


def update_env(pid: int, status_file: Path) -> dict[str, str]:
    """The three variables the install scripts read in update mode (tools/install/install.ps1, install.sh)."""
    return {
        "CLEAN_VOICE_UPDATE": "1",
        "CLEAN_VOICE_WAIT_PID": str(pid),
        "CLEAN_VOICE_STATUS_FILE": str(status_file),
    }


def launch_spec(repo: str, platform: str, pid: int, folder: Path) -> LaunchSpec:
    """The terminal launch for ``platform``; ``folder`` is a fresh folder that holds the status file (and, on a Mac,
    the .command file). A Mac's file is mode 0700 so only this user can read or change what Terminal will run."""
    if platform not in SCRIPTS:
        raise OSError(f"Install update does not support {platform}")
    status = folder / "status"
    command = install_command_text(repo, platform)
    variables = update_env(pid, status)
    if platform == "win32":
        argv = [str(powershell_exe()), "-NoProfile", "-Command", command]
        # The child would inherit the app's working directory, which a Start menu launch sets to the install folder; Windows
        # refuses to rename a folder that is a process's working directory, so the swap would fail. The attempt's own
        # folder is outside it. (A Mac's Terminal starts in the home folder: nothing to do there.)
        return LaunchSpec(argv, {**os.environ, **variables}, CREATE_NEW_CONSOLE, None, folder)
    script = folder / "Clean Voice Update.command"
    lines = ["#!/bin/bash", *(f"export {k}={shlex.quote(v)}" for k, v in variables.items()), command, ""]
    script.write_text("\n".join(lines), encoding="utf-8")
    script.chmod(0o700)
    return LaunchSpec(["open", "-a", "Terminal", str(script)], None, 0, script)


def start_terminal(spec: LaunchSpec) -> subprocess.Popen[bytes] | None:
    """Start the terminal. Windows: the new console's process, which the app watches. A Mac: ``open`` returns at once,
    so a non-zero exit is the failure; there is no process to watch."""
    if spec.script is None:
        return subprocess.Popen(
            spec.argv, env=spec.env, creationflags=spec.creationflags, cwd=spec.cwd
        )  # an argument list, no shell
    done = subprocess.run(spec.argv, check=False)
    if done.returncode != 0:
        raise OSError(f"open exited {done.returncode}")
    return None


def fresh_folder() -> Path:
    """A new folder for one attempt, so an old ``failed`` is never read by the next attempt."""
    return Path(tempfile.mkdtemp(prefix="clean-voice-update-"))


def read_status(folder: Path) -> str:
    """The script's one-word status, or "" when there is none yet (or it can't be read)."""
    try:
        return (folder / "status").read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return ""
