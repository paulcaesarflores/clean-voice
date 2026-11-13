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

Everything here runs off the UI thread; ``opener`` defaults to ``open_url``, and the tests replace it.
"""

from __future__ import annotations

import hashlib
import json
import re
import ssl
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
