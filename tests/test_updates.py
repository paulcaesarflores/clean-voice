"""Check for updates (DESIGN "Updates", user decision 2026-11-11; the original's main.rs:82-93, main.js:642-662). The
network is always faked here: no test reaches GitHub, and no installer or terminal is ever started. An app the
installer installed downloads, verifies and starts the installer; an app the install command installed (B019) runs that
command in a terminal and keeps running until the script says ``swap``."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import stat
import sys
import urllib.error
import urllib.request
import urllib.response
from email.message import Message
from pathlib import Path

import pytest

from clean_voice import updates
from clean_voice.updates import (
    CheckResult,
    UpdateError,
    asset_for,
    check,
    download_verified,
    install_command,
    is_newer,
    parse_sums,
    parse_version,
    repo_of,
)

REPO = "someone/clean-voice"
MSI = b"MSI installer bytes" * 1000
DMG = b"DMG disk image bytes" * 1000


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def release(tag: str = "v0.2.0", sums: bytes | None = None) -> dict:
    sums = (
        sums
        if sums is not None
        else (
            f"{hashlib.sha256(MSI).hexdigest()}  Clean Voice-0.2.0.msi\n"
            f"{hashlib.sha256(DMG).hexdigest()} *Clean Voice-0.2.0.dmg\n"
        ).encode()
    )
    files = {"Clean Voice-0.2.0.msi": MSI, "Clean Voice-0.2.0.dmg": DMG, "SHA256SUMS": sums}
    return {
        "tag_name": tag,
        "assets": [
            {"name": name, "browser_download_url": f"https://github.com/{REPO}/releases/download/{tag}/{name}"}
            for name in files
        ],
        "_files": files,
    }


class FakeNet:
    """Answers the release API and the asset downloads from memory; records every request."""

    def __init__(self, rel: dict | None = None, error: Exception | None = None) -> None:
        self.rel = rel if rel is not None else release()
        self.error = error
        self.requests: list[tuple[str, dict[str, str]]] = []

    def __call__(self, request, timeout: float):
        self.requests.append((request.full_url, dict(request.header_items())))
        if self.error is not None:
            raise self.error
        if request.full_url == f"https://api.github.com/repos/{REPO}/releases/latest":
            body = {k: v for k, v in self.rel.items() if k != "_files"}
            return FakeResponse(json.dumps(body).encode())
        name = request.full_url.rsplit("/", 1)[1].replace("%20", " ")
        return FakeResponse(self.rel["_files"][name])


# ------------------------------------------------------------------ versions and the repo


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v1.2.3", (1, 2, 3)),
        ("1.2.3", (1, 2, 3)),
        ("v0.10.0", (0, 10, 0)),
        ("v1.2", None),
        ("latest", None),
        ("", None),
    ],
)
def test_tags_are_read_as_versions(tag: str, expected) -> None:
    assert parse_version(tag) == expected


def test_newer_same_and_older() -> None:
    assert is_newer("v0.2.0", "0.1.0")
    assert is_newer("v0.10.0", "0.9.9")  # numbers, not text
    assert not is_newer("v0.1.0", "0.1.0")
    assert not is_newer("v0.0.9", "0.1.0")
    assert not is_newer("nightly", "0.1.0")  # a malformed tag is never an update


def test_the_repo_comes_from_the_homepage_url() -> None:
    assert repo_of("https://github.com/someone/clean-voice") == "someone/clean-voice"
    assert repo_of("https://github.com/someone/clean-voice/") == "someone/clean-voice"
    assert repo_of("https://github.com/REPLACE_WITH_GITHUB_OWNER/clean-voice") is None  # not set up yet (J04)
    assert repo_of("https://example.com/clean-voice") is None
    assert repo_of(None) is None
    # And the app reads it from its own metadata: pyproject's Homepage, through Briefcase or the editable install.
    assert updates.homepage() == "https://github.com/paulcaesarflores/clean-voice"
    assert repo_of(updates.homepage()) == "paulcaesarflores/clean-voice"


# ------------------------------------------------------------------ the check


def test_a_newer_release_is_ready_to_install() -> None:
    net = FakeNet(release("v0.2.0"))
    result = check("0.1.0", REPO, opener=net)
    assert result.state == "newer" and result.version == "0.2.0"
    assert result.message == "Clean Voice 0.2.0 is ready to install"
    url, headers = net.requests[0]
    assert url == f"https://api.github.com/repos/{REPO}/releases/latest" and url.startswith("https://")
    assert headers["User-agent"] == "Clean-Voice-updater"  # generic: no name, no machine, nothing personal
    assert set(headers) <= {"User-agent", "Accept"}


@pytest.mark.parametrize("tag", ["v0.1.0", "v0.0.5"])
def test_the_same_or_an_older_release_is_the_latest(tag: str) -> None:
    result = check("0.1.0", REPO, opener=FakeNet(release(tag)))
    assert result.state == "latest" and result.message == "You have the latest version"


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("offline"),
        urllib.error.HTTPError("u", 403, "rate limited", {}, None),  # type: ignore[arg-type]
        urllib.error.HTTPError("u", 404, "no release yet", {}, None),  # type: ignore[arg-type]
        TimeoutError("slow"),
        OSError("socket"),
    ],
)
def test_any_network_error_says_it_couldnt_check(error: Exception) -> None:
    result = check("0.1.0", REPO, opener=FakeNet(error=error))
    assert result.state == "failed" and result.message == "Couldn't check for updates right now"


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"tag_name": 7}', b'{"assets": []}'])
def test_a_malformed_answer_says_it_couldnt_check(body: bytes) -> None:
    def opener(request, timeout):
        return FakeResponse(body)

    assert check("0.1.0", REPO, opener=opener).state == "failed"


def test_with_no_repo_yet_it_says_it_couldnt_check_without_asking_anyone() -> None:
    net = FakeNet()
    assert check("0.1.0", None, opener=net).state == "failed" and not net.requests


def test_a_malformed_tag_is_not_an_update() -> None:
    assert check("0.1.0", REPO, opener=FakeNet(release("v2"))).state == "failed"


# ------------------------------------------------------------------ the install


def test_the_right_asset_for_each_os() -> None:
    rel = release()
    win, mac = asset_for(rel["assets"], "win32"), asset_for(rel["assets"], "darwin")
    assert win is not None and win[0] == "Clean Voice-0.2.0.msi"
    assert mac is not None and mac[0] == "Clean Voice-0.2.0.dmg"
    assert asset_for(rel["assets"], "linux") is None
    assert asset_for([a for a in rel["assets"] if not a["name"].endswith(".msi")], "win32") is None


def test_the_checksum_list_is_read_in_both_sha256sum_forms() -> None:
    sums = parse_sums("abc123  a file.msi\nDEF456 *b.dmg\n\n# a comment\n")
    assert sums == {"a file.msi": "abc123", "b.dmg": "def456"}


def test_the_download_is_verified_with_progress(tmp_path: Path) -> None:
    net = FakeNet()
    result = check("0.1.0", REPO, opener=net)
    seen: list[float] = []
    path = download_verified(result, "win32", tmp_path, seen.append, opener=net)
    assert path.read_bytes() == MSI and path.parent == tmp_path and path.name == "Clean Voice-0.2.0.msi"
    assert seen[-1] == 1.0 and seen == sorted(seen) and len(seen) > 1
    assert all(url.startswith("https://") for url, _ in net.requests)


def test_a_checksum_mismatch_is_refused(tmp_path: Path) -> None:
    wrong = f"{hashlib.sha256(b'something else').hexdigest()}  Clean Voice-0.2.0.msi\n".encode()
    net = FakeNet(release(sums=wrong))
    result = check("0.1.0", REPO, opener=net)
    with pytest.raises(UpdateError) as caught:
        download_verified(result, "win32", tmp_path, lambda f: None, opener=net)
    assert "didn't match" in str(caught.value) and "nothing was installed" in str(caught.value)
    assert not list(tmp_path.glob("*.msi"))  # the bad file is not left behind


def test_a_release_without_checksums_or_its_installer_is_refused(tmp_path: Path) -> None:
    rel = release()
    rel["assets"] = [a for a in rel["assets"] if a["name"] != "SHA256SUMS"]
    net = FakeNet(rel)
    with pytest.raises(UpdateError):
        download_verified(check("0.1.0", REPO, opener=net), "win32", tmp_path, lambda f: None, opener=net)
    unlisted = f"{hashlib.sha256(DMG).hexdigest()}  Clean Voice-0.2.0.dmg\n".encode()
    net = FakeNet(release(sums=unlisted))
    with pytest.raises(UpdateError):
        download_verified(check("0.1.0", REPO, opener=net), "win32", tmp_path, lambda f: None, opener=net)


def test_a_failed_download_is_a_plain_error(tmp_path: Path) -> None:
    net = FakeNet()
    result = check("0.1.0", REPO, opener=net)
    net.error = urllib.error.URLError("offline")
    with pytest.raises(UpdateError) as caught:
        download_verified(result, "win32", tmp_path, lambda f: None, opener=net)
    assert str(caught.value) == "Couldn't download the update. Try again later."


def test_the_install_command_for_each_os() -> None:
    msi = Path("C:/Temp/cv/Clean Voice-0.2.0.msi")
    assert install_command(msi, "win32") == ["msiexec", "/i", str(msi)]  # a list: no shell, the space is safe
    dmg = Path("/tmp/cv/Clean Voice-0.2.0.dmg")
    assert install_command(dmg, "darwin") == ["open", str(dmg)]
    with pytest.raises(UpdateError):
        install_command(dmg, "linux")


def test_check_result_is_only_installable_when_newer() -> None:
    assert not CheckResult("latest", "You have the latest version").installable
    assert not CheckResult("failed", "Couldn't check for updates right now").installable


# ------------------------------------------------------------------ the buttons (sidebar and Settings)


def controller(net: FakeNet, launched: list[list[str]], quits: list[int], platform: str = "win32"):
    from clean_voice.app import UpdateController

    return UpdateController(
        "0.1.0",
        REPO,
        platform,
        quit_app=lambda: quits.append(1),
        opener=net,
        run=lambda job: job(),  # the worker thread, run inline here
        launch=launched.append,  # recorded, never started
        via_command=False,  # an app the installer installed: the download path
    )


@pytest.fixture
def window_with(qtbot, tmp_path: Path):
    from test_main_screen import MICS, ON, OUTPUTS, FakeLoginItem

    from clean_voice.app import MainWindow
    from clean_voice.preview import PreviewEngine

    def make(ctl):
        window = MainWindow(
            PreviewEngine(ON, MICS, tmp_path, OUTPUTS), tmp_path / "s.json", FakeLoginItem(), updater=ctl
        )
        qtbot.addWidget(window)
        window.show()
        return window

    return make


def views(window) -> list:
    return [window.updates_view]


def test_check_for_updates_sits_beside_the_version_in_the_footer(window_with) -> None:
    window = window_with(controller(FakeNet(), [], []))
    view = window.updates_view
    assert view.check_button.text() == "Check for updates" and view.check_button.isEnabled()
    assert not view.install_button.isVisibleTo(window) and not view.status.isVisibleTo(window)
    assert window.footer.isAncestorOf(view.check_button) and window.footer.isAncestorOf(window.version_label)
    version = window.version_label.mapTo(window, window.version_label.rect().topRight())
    check = view.check_button.mapTo(window, view.check_button.rect().topLeft())
    assert check.x() > version.x() and abs(check.y() - version.y()) < 12  # beside it, on its line


def test_a_newer_release_offers_install_and_a_click_installs_and_quits(window_with, tmp_path: Path) -> None:
    launched: list[list[str]] = []
    quits: list[int] = []
    ctl = controller(FakeNet(release("v0.2.0")), launched, quits)
    window = window_with(ctl)
    window.updates_view.check_button.click()
    for view in views(window):
        assert view.status.text() == "Clean Voice 0.2.0 is ready to install"
        assert view.install_button.isVisibleTo(view) and view.install_button.isEnabled()
    assert not launched and not quits  # nothing installs without the click
    window.updates_view.install_button.click()  # Install update: it asks first
    assert not launched
    window.updates_view.install_button.click()  # Install
    assert len(launched) == 1 and launched[0][:2] == ["msiexec", "/i"]
    installer = Path(launched[0][2])
    assert installer.read_bytes() == MSI and installer.name == "Clean Voice-0.2.0.msi"
    assert quits == [1]


def test_the_latest_version_and_a_failure_say_so(window_with) -> None:
    window = window_with(controller(FakeNet(release("v0.1.0")), [], []))
    window.updates_view.check_button.click()
    assert window.updates_view.status.text() == "You have the latest version"
    assert not window.updates_view.install_button.isVisibleTo(window.updates_view)
    window = window_with(controller(FakeNet(error=urllib.error.URLError("offline")), [], []))
    window.updates_view.check_button.click()
    assert window.updates_view.status.text() == "Couldn't check for updates right now"
    assert window.updates_view.check_button.isEnabled()  # it can be tried again


def test_the_quiet_launch_check_shows_only_an_update() -> None:
    for net, shown in ((FakeNet(release("v0.1.0")), ""), (FakeNet(error=OSError("x")), "")):
        ctl = controller(net, [], [])
        ctl.check(quiet=True)
        assert ctl.state == "idle" and ctl.message == shown
    ctl = controller(FakeNet(release("v0.2.0")), [], [])
    ctl.check(quiet=True)
    assert ctl.state == "newer" and ctl.message == "Clean Voice 0.2.0 is ready to install"


def test_buttons_are_disabled_while_checking_and_downloading(window_with) -> None:
    jobs: list = []
    from clean_voice.app import UpdateController

    ctl = UpdateController(
        "0.1.0", REPO, "win32", lambda: None, opener=FakeNet(), run=jobs.append, launch=lambda c: None
    )
    window = window_with(ctl)
    view = window.updates_view
    view.check_button.click()
    assert not view.check_button.isEnabled() and view.status.text() == "Checking for updates…"
    jobs.pop()()  # the check finishes
    assert view.check_button.isEnabled() and view.install_button.isEnabled()
    view.install_button.click()  # the question
    assert not view.check_button.isEnabled() and view.cancel_button.isEnabled()
    view.install_button.click()  # Install
    assert not view.check_button.isEnabled() and not view.install_button.isEnabled()
    assert view.install_button.text() == "Downloading 0 %"
    jobs.pop()()
    assert ctl.state == "idle" or ctl.fraction == 1.0


def test_a_mismatched_download_is_refused_on_screen(window_with) -> None:
    launched: list[list[str]] = []
    quits: list[int] = []
    wrong = f"{hashlib.sha256(b'other').hexdigest()}  Clean Voice-0.2.0.msi\n".encode()
    window = window_with(controller(FakeNet(release(sums=wrong)), launched, quits))
    window.updates_view.check_button.click()
    window.updates_view.install_button.click()
    window.updates_view.install_button.click()
    assert window.updates_view.status.text() == updates.MISMATCH
    assert not launched and not quits
    assert window.updates_view.install_button.isEnabled()  # it may be tried again


def test_on_macos_the_dmg_is_opened(tmp_path: Path) -> None:
    launched: list[list[str]] = []
    ctl = controller(FakeNet(), launched, [], platform="darwin")
    ctl.check()
    ctl.confirm()
    ctl.install()
    assert launched[0][0] == "open" and launched[0][1].endswith("Clean Voice-0.2.0.dmg")


@pytest.mark.parametrize("size", ["default", "minimum"])
@pytest.mark.parametrize("fraction", [0.0, 0.37, 1.0])
def test_the_download_label_fits_its_button(window_with, size: str, fraction: float) -> None:
    # fine-tune-updates r1, F2: the sidebar's button read "ownloading... 37 '"; now it sits in the footer.
    from PySide6.QtWidgets import QApplication

    from clean_voice.app import MIN_SIZE, WINDOW_SIZE, UpdateController

    jobs: list = []
    ctl = UpdateController(
        "0.1.0", REPO, "win32", lambda: None, opener=FakeNet(release("v0.2.0")), run=jobs.append, launch=lambda c: None
    )
    window = window_with(ctl)
    window.resize(*(WINDOW_SIZE if size == "default" else MIN_SIZE))
    ctl.check()
    jobs.pop()()
    ctl.confirm()
    ctl.install()
    ctl._on_progress(fraction)
    QApplication.processEvents()
    for view in views(window):
        button = view.install_button
        assert button.text().startswith("Downloading")
        assert button.width() >= button.sizeHint().width(), (button.text(), button.width(), button.sizeHint().width())


# ------------------------------------------------------------------ hardening (fine-tune-updates r1)


@pytest.mark.parametrize(
    "error",
    [
        http.client.IncompleteRead(b"part"),
        http.client.BadStatusLine("junk"),
        http.client.LineTooLong("header"),
        RuntimeError("anything else"),
    ],
)
def test_no_network_error_escapes_the_check_or_the_download(error: Exception, tmp_path: Path) -> None:
    # n1: http.client's errors are not OSErrors; they escaped, killed the worker, and left "Checking…" on screen.
    assert check("0.1.0", REPO, opener=FakeNet(error=error)).state == "failed"
    net = FakeNet()
    result = check("0.1.0", REPO, opener=net)
    net.error = error
    with pytest.raises(UpdateError) as caught:
        download_verified(result, "win32", tmp_path, lambda f: None, opener=net)
    assert str(caught.value) == updates.DOWNLOAD_FAILED


def test_the_view_never_sticks_when_a_job_fails(window_with) -> None:
    from clean_voice.app import UpdateController

    def broken(request, timeout):
        raise RuntimeError("boom")

    ctl = UpdateController("0.1.0", REPO, "win32", lambda: None, opener=broken, run=lambda job: job())
    ctl.check()
    assert ctl.state == "failed" and ctl.message == updates.FAILED
    ctl.result = check("0.1.0", REPO, opener=FakeNet())
    ctl.state = "newer"
    ctl.confirm()
    ctl.install()
    assert ctl.state == "install-failed" and ctl.message == updates.DOWNLOAD_FAILED


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/someone/clean-voice/releases/download/v0.2.0/x.msi",
        "file:///C:/Windows/x.msi",
        "ftp://github.com/x.msi",
        "https://evil.example/x.msi",
        "https://github.com.evil.example/x.msi",
    ],
)
def test_only_https_github_downloads_are_accepted(url: str, tmp_path: Path) -> None:
    # n2: the asset URLs come from the release JSON; anything but HTTPS on GitHub's hosts is refused.
    rel = release()
    for asset in rel["assets"]:
        asset["browser_download_url"] = url
    net = FakeNet(rel)
    result = check("0.1.0", REPO, opener=net)
    with pytest.raises(UpdateError):
        download_verified(result, "win32", tmp_path, lambda f: None, opener=net)
    assert len(net.requests) == 1  # only the API was asked: nothing fetched from the bad URL


def test_redirects_may_not_leave_https_github() -> None:
    handler = updates.SafeRedirect()
    request = urllib.request.Request("https://github.com/someone/clean-voice/releases/download/v1/x.msi")
    headers = http.client.HTTPMessage()
    good = "https://objects.githubusercontent.com/github-production-release-asset/x"
    ok = handler.redirect_request(request, io.BytesIO(), 302, "Found", headers, good)
    assert ok is not None and ok.full_url == good
    for bad in ("http://objects.githubusercontent.com/x", "https://evil.example/x"):
        with pytest.raises(urllib.error.HTTPError):
            handler.redirect_request(request, io.BytesIO(), 302, "Found", headers, bad)


def test_the_tls_context_verifies_with_certifi() -> None:
    # n4: the Mac bundle's Python may have no CA certificates of its own, so the app ships certifi's.
    import ssl

    import certifi

    context = updates.tls_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert len(context.get_ca_certs()) > 50
    assert updates.CA_FILE == certifi.where()


def test_assets_are_matched_after_githubs_space_to_dot_renaming(tmp_path: Path) -> None:
    # n3: GitHub stores "Clean Voice-0.2.0.msi" as "Clean.Voice-0.2.0.msi"; SHA256SUMS made before the upload lists
    # the original name. Either spelling on either side matches.
    rel = release()
    for asset in rel["assets"]:
        asset["name"] = asset["name"].replace(" ", ".")
        asset["browser_download_url"] = asset["browser_download_url"].replace(" ", ".")
    rel["_files"] = {k.replace(" ", "."): v for k, v in rel["_files"].items()}
    net = FakeNet(rel)
    path = download_verified(check("0.1.0", REPO, opener=net), "win32", tmp_path, lambda f: None, opener=net)
    assert path.read_bytes() == MSI


def test_install_asks_first_that_clean_voice_will_close(window_with) -> None:
    launched: list[list[str]] = []
    quits: list[int] = []
    ctl = controller(FakeNet(release("v0.2.0")), launched, quits)
    window = window_with(ctl)
    view = window.updates_view
    view.check_button.click()
    view.install_button.click()
    assert view.status.text() == updates.CONFIRM and not launched
    assert view.install_button.text() == "Install" and view.cancel_button.isVisibleTo(view)
    for button in (view.install_button, view.cancel_button):  # both fit the sidebar
        assert button.width() >= button.sizeHint().width(), button.text()
    view.cancel_button.click()  # Cancel: back to the offer, nothing downloaded
    assert ctl.state == "newer" and view.status.text() == "Clean Voice 0.2.0 is ready to install" and not launched
    assert not view.cancel_button.isVisibleTo(view)
    view.install_button.click()
    view.install_button.click()  # Install
    assert len(launched) == 1 and quits == [1]


# ------------------------------------------------------------------ through the real opener (fine-tune-updates r2, T1)


@pytest.fixture
def fake_https(monkeypatch: pytest.MonkeyPatch):
    """The real ``open_url`` (its handlers, its redirect check), with HTTPS and HTTP answered from a table."""

    table: dict[str, tuple[int, bytes, str | None]] = {}
    seen: list[str] = []

    def answer(self, req):  # type: ignore[no-untyped-def]
        seen.append(req.full_url)
        code, body, location = table.get(req.full_url, (404, b"", None))
        if code == 404:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", Message(), io.BytesIO())
        headers = Message()
        if location:
            headers["Location"] = location
        headers["Content-Length"] = str(len(body))
        response = urllib.response.addinfourl(io.BytesIO(body), headers, req.full_url, code)
        response.msg = "Found" if location else "OK"  # type: ignore[attr-defined]
        return response

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", answer)
    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", answer)
    return table, seen


def release_body(rel: dict) -> bytes:
    return json.dumps({k: v for k, v in rel.items() if k != "_files"}).encode()


@pytest.mark.parametrize("target", ["http://api.github.com/plain", "https://evil.example/latest"])
def test_the_real_opener_refuses_a_redirect_off_https_github(fake_https, target: str) -> None:
    # Removing SafeRedirect from open_url turns this red: urllib alone follows https -> http and any host.
    table, seen = fake_https
    api = updates.API.format(repo=REPO)
    table[api] = (302, b"", target)
    table[target] = (200, release_body(release("v0.2.0")), None)
    assert check("0.1.0", REPO).state == "failed" and seen == [api]  # the default opener: open_url


def test_the_real_opener_follows_githubs_own_redirect(fake_https, tmp_path: Path) -> None:
    table, _ = fake_https
    rel = release("v0.2.0")
    table[updates.API.format(repo=REPO)] = (200, release_body(rel), None)
    for asset in rel["assets"]:
        target = "https://release-assets.githubusercontent.com/x/" + asset["name"].replace(" ", ".")
        table[asset["browser_download_url"]] = (302, b"", target)
        table[target] = (200, rel["_files"][asset["name"]], None)
    result = check("0.1.0", REPO)
    assert result.state == "newer"
    assert download_verified(result, "win32", tmp_path, lambda f: None).read_bytes() == MSI


# ------------------------------------------------------------------ the install command


def test_the_install_command_text_for_each_os() -> None:
    base = f"https://github.com/{REPO}/releases/latest/download/"
    assert updates.install_url(REPO, "win32") == base + "install.ps1"
    assert updates.install_url(REPO, "darwin") == base + "install.sh"
    assert updates.install_command_text(REPO, "win32") == f"irm {base}install.ps1 | iex"
    assert updates.install_command_text(REPO, "darwin") == f"curl -fsSL {base}install.sh | bash"
    assert updates.is_safe_url(updates.install_url(REPO, "win32"))  # GitHub's host, HTTPS


def test_the_windows_launch_spec(tmp_path: Path) -> None:
    spec = updates.launch_spec(REPO, "win32", 4242, tmp_path)
    expected = (
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    assert spec.argv == [str(expected), "-NoProfile", "-Command", updates.install_command_text(REPO, "win32")]
    assert spec.creationflags == 0x10  # CREATE_NEW_CONSOLE: a visible window of its own
    assert spec.env is not None
    assert spec.env["CLEAN_VOICE_UPDATE"] == "1"
    assert spec.env["CLEAN_VOICE_WAIT_PID"] == "4242"
    assert spec.env["CLEAN_VOICE_STATUS_FILE"] == str(tmp_path / "status")
    assert spec.script is None


def test_the_mac_launch_writes_a_private_command_file(tmp_path: Path) -> None:
    spec = updates.launch_spec(REPO, "darwin", 77, tmp_path)
    assert spec.script is not None and spec.script.parent == tmp_path
    assert spec.argv == ["open", "-a", "Terminal", str(spec.script)]
    lines = spec.script.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "export CLEAN_VOICE_UPDATE=1" in lines and "export CLEAN_VOICE_WAIT_PID=77" in lines
    assert f"export CLEAN_VOICE_STATUS_FILE={tmp_path / 'status'}" in lines or any(
        line.startswith("export CLEAN_VOICE_STATUS_FILE=") for line in lines
    )
    assert lines[-1] == updates.install_command_text(REPO, "darwin")
    if sys.platform != "win32":
        assert stat.S_IMODE(spec.script.stat().st_mode) == 0o700


def test_an_unsupported_platform_cannot_launch(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        updates.launch_spec(REPO, "linux", 1, tmp_path)


def test_the_status_is_read_from_the_attempts_folder(tmp_path: Path) -> None:
    assert updates.read_status(tmp_path) == ""
    (tmp_path / "status").write_text("swap")
    assert updates.read_status(tmp_path) == "swap"


class FakeProcess:
    """The Windows console's process: running until ``code`` is set."""

    def __init__(self) -> None:
        self.code: int | None = None

    def poll(self) -> int | None:
        return self.code


class Recorder:
    """The launcher: records each launch spec; can raise or hand back a process."""

    def __init__(self, error: Exception | None = None) -> None:
        self.specs: list[updates.LaunchSpec] = []
        self.error = error
        self.process = FakeProcess()

    def __call__(self, spec: updates.LaunchSpec) -> object:
        if self.error is not None:
            raise self.error
        self.specs.append(spec)
        return self.process


def command_controller(
    tmp_path: Path,
    launcher: Recorder,
    quits: list[int],
    platform: str = "win32",
    net: FakeNet | None = None,
):
    from clean_voice.app import UpdateController

    count = iter(range(1, 100))

    def new_folder() -> Path:
        folder = tmp_path / f"attempt{next(count)}"
        folder.mkdir()
        return folder

    return UpdateController(
        "0.1.0",
        REPO,
        platform,
        quit_app=lambda: quits.append(1),
        opener=net or FakeNet(),
        run=lambda job: job(),  # the worker thread, run inline here
        pid=4242,
        via_command=True,
        terminal=launcher,
        new_folder=new_folder,
    )


def offered(tmp_path: Path, launcher: Recorder | None = None, quits: list[int] | None = None, platform: str = "win32"):
    ctl = command_controller(tmp_path, launcher or Recorder(), [] if quits is None else quits, platform)
    ctl.check()
    assert ctl.state == "newer"
    return ctl


def status_file(ctl) -> Path:
    return ctl._folder / "status"


def test_confirm_then_install_launches_once_and_the_app_keeps_running(tmp_path: Path) -> None:
    launcher, quits = Recorder(), []
    ctl = offered(tmp_path, launcher, quits)
    ctl.install()  # without the question first: nothing
    assert not launcher.specs and ctl.state == "newer"
    ctl.confirm()
    assert ctl.state == "confirm" and ctl.message == updates.CONFIRM_COMMAND
    ctl.install()
    assert len(launcher.specs) == 1
    assert ctl.state == "installing" and ctl.message == updates.INSTALLING
    assert not quits  # Clean Voice stays open during the rebuild
    assert launcher.specs[0].env is not None and launcher.specs[0].env["CLEAN_VOICE_STATUS_FILE"] == str(
        status_file(ctl)
    )


def test_each_attempt_gets_a_new_status_folder_and_never_reads_an_old_failure(tmp_path: Path) -> None:
    launcher = Recorder()
    ctl = offered(tmp_path, launcher)
    ctl.confirm()
    ctl.install()
    first = status_file(ctl)
    first.write_text("failed")
    ctl.poll()
    assert ctl.state == "newer"
    ctl.confirm()
    ctl.install()
    second = status_file(ctl)
    assert second != first and second.parent != first.parent
    assert first.read_text() == "failed" and not second.exists()
    ctl.poll()
    assert ctl.state == "installing"  # the old failure is not read


def test_swap_quits_the_app_once(tmp_path: Path) -> None:
    quits: list[int] = []
    ctl = offered(tmp_path, quits=quits)
    ctl.confirm()
    ctl.install()
    ctl.poll()
    assert quits == []  # nothing yet
    status_file(ctl).write_text("swap")
    ctl.poll()
    ctl.poll()
    assert quits == [1]


def test_failed_returns_to_the_offer_with_the_plain_line_and_no_quit(tmp_path: Path) -> None:
    quits: list[int] = []
    ctl = offered(tmp_path, quits=quits)
    ctl.confirm()
    ctl.install()
    status_file(ctl).write_text("failed")
    ctl.poll()
    assert ctl.state == "newer"
    assert ctl.message == "The update didn't work, so you still have the old Clean Voice. The window shows why."
    assert not quits


def test_a_windows_console_closed_without_a_status_gives_the_same_line(tmp_path: Path) -> None:
    launcher = Recorder()
    ctl = offered(tmp_path, launcher)
    ctl.confirm()
    ctl.install()
    ctl.poll()
    assert ctl.state == "installing"  # the window is still open
    launcher.process.code = 1  # the guest closed it
    ctl.poll()
    assert ctl.state == "newer" and ctl.message == updates.UPDATE_FAILED


def test_a_mac_has_no_process_to_watch(tmp_path: Path) -> None:
    launcher = Recorder()
    ctl = offered(tmp_path, launcher, platform="darwin")
    ctl.confirm()
    ctl.install()
    launcher.process.code = 0
    ctl.poll()
    assert ctl.state == "installing"  # `open` returned at once; only the status file speaks


@pytest.mark.parametrize("error", [OSError("no powershell"), PermissionError("blocked")])
def test_a_launch_that_raises_is_install_failed_and_the_app_stays(tmp_path: Path, error: Exception) -> None:
    quits: list[int] = []
    ctl = offered(tmp_path, Recorder(error), quits)
    ctl.confirm()
    ctl.install()
    assert ctl.state == "install-failed" and ctl.message == updates.LAUNCH_FAILED
    assert not quits
    assert (
        updates.LAUNCH_FAILED == "Couldn't open the update window. Try again, or use the command on the download page."
    )


def test_a_second_click_after_a_failed_launch_works(tmp_path: Path) -> None:
    launcher = Recorder(OSError("first"))
    ctl = offered(tmp_path, launcher)
    ctl.confirm()
    ctl.install()
    assert ctl.state == "install-failed"
    launcher.error = None
    ctl.confirm()
    ctl.install()
    assert ctl.state == "installing" and len(launcher.specs) == 1


def test_cancel_returns_to_the_offer_and_launches_nothing(tmp_path: Path) -> None:
    launcher = Recorder()
    ctl = offered(tmp_path, launcher)
    ctl.confirm()
    ctl.cancel()
    assert ctl.state == "newer" and ctl.message == "Clean Voice 0.2.0 is ready to install" and not launcher.specs


def test_check_for_updates_clears_installing(tmp_path: Path) -> None:
    ctl = offered(tmp_path)
    ctl.confirm()
    ctl.install()
    assert ctl.state == "installing"
    ctl.check()  # a Mac terminal closed by hand leaves no status; the guest's own check returns the footer to normal
    assert ctl.state == "newer" and ctl.message == "Clean Voice 0.2.0 is ready to install"
    ctl.poll()  # and the old attempt is no longer watched
    assert ctl.state == "newer"


def test_the_quiet_check_leaves_an_update_in_progress_alone(tmp_path: Path) -> None:
    ctl = offered(tmp_path)
    ctl.confirm()
    ctl.install()
    ctl.check(quiet=True)
    assert ctl.state == "installing"


def test_the_view_shows_the_offer_and_the_installing_line(window_with, tmp_path: Path) -> None:
    launcher = Recorder()
    ctl = command_controller(tmp_path, launcher, [])
    window = window_with(ctl)
    view = window.updates_view
    view.check_button.click()
    assert view.status.text() == "Clean Voice 0.2.0 is ready to install"
    assert view.install_button.isVisibleTo(view) and view.install_button.isEnabled()
    view.install_button.click()  # Install update: it asks first
    assert view.status.text() == updates.CONFIRM_COMMAND and not launcher.specs
    assert view.install_button.text() == "Install" and view.cancel_button.isVisibleTo(view)
    for button in (view.install_button, view.cancel_button):
        assert button.width() >= button.sizeHint().width(), button.text()
    view.install_button.click()  # Install
    assert len(launcher.specs) == 1
    assert view.status.text() == updates.INSTALLING
    assert not view.install_button.isVisibleTo(view) and view.check_button.isEnabled()


# ------------------------------------------------------------------ which update path an app takes (B019)


def test_an_app_installed_by_the_installer_downloads_and_an_app_installed_by_the_command_does_not(
    tmp_path: Path,
) -> None:
    from clean_voice.app import UpdateController

    launched: list[list[str]] = []
    terminals: list[object] = []
    net = FakeNet()
    for via_command, expect_terminal in ((False, False), (True, True)):
        ctl = UpdateController(
            "0.1.0",
            REPO,
            "win32",
            lambda: None,
            opener=net,
            run=lambda job: job(),
            launch=launched.append,
            terminal=lambda spec: terminals.append(spec),
            via_command=via_command,
            new_folder=lambda: tmp_path,
        )
        ctl.check()
        ctl.confirm()
        assert ctl.message == (updates.CONFIRM_COMMAND if via_command else updates.CONFIRM)
        ctl.install()
        assert (ctl.state == "installing") == expect_terminal
    assert len(launched) == 1 and launched[0][:2] == ["msiexec", "/i"]  # the installer app's path, as before
    assert len(terminals) == 1


def test_the_marker_decides_how_an_app_was_installed(tmp_path: Path) -> None:
    exe = tmp_path / "app" / "Clean Voice.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"stub")
    assert not updates.installed_by_command("win32", exe)
    (exe.parent / updates.MARKER_NAME).write_text("0.3.0")
    assert updates.installed_by_command("win32", exe)
    assert updates.MARKER_NAME == "installed-by-command"


def test_on_a_mac_the_marker_holds_the_binarys_hash_so_a_dmg_copy_does_not_match(tmp_path: Path) -> None:
    import hashlib

    binary = tmp_path / "Clean Voice.app" / "Contents" / "MacOS" / "Clean Voice"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"built by the command")
    home = tmp_path / "home"
    marker = home / "Library" / "Application Support" / "Clean Voice" / updates.MARKER_NAME
    marker.parent.mkdir(parents=True)
    assert not updates.installed_by_command("darwin", binary, home)
    marker.write_text(hashlib.sha256(b"built by the command").hexdigest() + "\n")
    assert updates.installed_by_command("darwin", binary, home)
    binary.write_bytes(b"a DMG copied over it")
    assert not updates.installed_by_command("darwin", binary, home)
    assert not updates.installed_by_command("linux", binary, home)
