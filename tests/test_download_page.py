"""The download page (``docs/site/``): its OS detection and links, run in Qt's JavaScript engine (QJSEngine), and the
HTML's offline promises (no outside scripts, working links without JavaScript)."""

from __future__ import annotations

import json
import re
import tomllib
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import cast

import pytest
from PySide6.QtQml import QJSEngine, QJSValue

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"

WINDOWS_CHROME = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
WINDOWS_FIREFOX = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0"
# Safari and Chrome on a Mac say "Intel Mac OS X" on Apple Silicon too.
MAC_SAFARI = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/18.1 Safari/605.1.15"
)
MAC_CHROME = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
LINUX_FIREFOX = "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0"
IPHONE = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 "
    "Mobile/15E148 Safari/604.1"
)
ANDROID = (
    "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile "
    "Safari/537.36"
)


def evaluate(engine: QJSEngine, program: str, file_name: str = "") -> QJSValue:
    # PySide6's stub also lists the stack-trace overload's tuple; this call returns the value.
    result = cast(QJSValue, engine.evaluate(program, file_name))
    assert not result.isError(), result.toString()
    return result


@pytest.fixture(scope="module")
def engine(qapp: object) -> QJSEngine:
    js = QJSEngine()  # needs the Qt application first: without one, QJSEngine ends the process
    evaluate(js, (SITE / "download.js").read_text(encoding="utf-8"), "download.js")
    return js


def call(engine: QJSEngine, expression: str) -> object:
    return evaluate(engine, expression).toVariant()


@pytest.mark.parametrize(
    ("ua_data", "user_agent", "touch", "expected"),
    [
        (None, WINDOWS_CHROME, 0, "windows"),
        (None, WINDOWS_FIREFOX, 0, "windows"),
        ({"platform": "Windows", "mobile": False}, WINDOWS_CHROME, 0, "windows"),
        (None, MAC_SAFARI, 0, "mac"),  # Mac, Intel or Apple Silicon (Safari cannot tell them apart)
        ({"platform": "macOS", "mobile": False}, MAC_CHROME, 0, "mac"),  # Mac Apple Silicon, Chrome
        (None, MAC_CHROME, 0, "mac"),  # Mac Intel, Chrome without userAgentData
        (None, LINUX_FIREFOX, 0, "other"),
        ({"platform": "Linux", "mobile": False}, LINUX_FIREFOX, 0, "other"),
        (None, IPHONE, 5, "other"),
        (None, IPHONE, 0, "other"),  # older Safari reports no touch points: the user agent alone must say phone
        (None, ANDROID, 5, "other"),
        ({"platform": "Android", "mobile": True}, ANDROID, 5, "other"),
        (None, MAC_SAFARI, 5, "other"),  # an iPad asking for the desktop site
        ({"platform": "", "mobile": False}, WINDOWS_CHROME, 0, "windows"),  # empty platform: read the user agent
        (None, "", 0, "other"),
    ],
)
def test_the_button_matches_the_computer(
    engine: QJSEngine, ua_data: dict | None, user_agent: str, touch: int, expected: str
) -> None:
    expr = f"CleanVoiceDownload.detectOS({json.dumps(ua_data)}, {json.dumps(user_agent)}, {touch})"
    assert call(engine, expr) == expected


def homepage_repo() -> str:
    homepage = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["urls"]["Homepage"]
    return homepage.removeprefix("https://github.com/")


@pytest.mark.parametrize(
    ("hostname", "pathname", "expected"),
    [
        ("someone.github.io", "/clean-voice/", "someone/clean-voice"),
        ("Someone.GitHub.io", "/clean-voice/index.html", "someone/clean-voice"),
        ("someone.github.io", "/my-fork", "someone/my-fork"),
        ("someone.github.io", "/", None),  # a user site, not a repo's page: the fallback
        ("", "/C:/Users/x/docs/site/index.html", None),  # a local file
        ("localhost", "/", None),
        ("evil-github.io", "/clean-voice/", None),
    ],
)
def test_the_repo_comes_from_the_page_address(
    engine: QJSEngine, hostname: str, pathname: str, expected: str | None
) -> None:
    repo = call(engine, f"CleanVoiceDownload.repoFromLocation({json.dumps(hostname)}, {json.dumps(pathname)})")
    assert repo == (expected or homepage_repo())


def test_links_use_the_release_stable_names(engine: QJSEngine) -> None:
    links = call(engine, 'CleanVoiceDownload.links("someone/clean-voice")')
    assert links == {
        "windows": "https://github.com/someone/clean-voice/releases/latest/download/Clean-Voice-Windows.msi",
        "mac": "https://github.com/someone/clean-voice/releases/latest/download/Clean-Voice-macOS.dmg",
        "source": "https://github.com/someone/clean-voice",
        "licence": "https://github.com/someone/clean-voice/blob/main/LICENSE",
    }
    # The same names the release workflow uploads.
    script = (ROOT / "tools" / "ci" / "release_assets.py").read_text(encoding="utf-8")
    assert 'WINDOWS_NAME = "Clean-Voice-Windows.msi"' in script
    assert 'MAC_NAME = "Clean-Voice-macOS.dmg"' in script


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


@pytest.fixture(scope="module")
def page() -> Page:
    parser = Page()
    parser.feed((SITE / "index.html").read_text(encoding="utf-8"))
    return parser


def test_without_javascript_both_buttons_work(engine: QJSEngine, page: Page) -> None:
    # The HTML's own addresses are the fallback repo's, so a page without JavaScript still downloads.
    links = call(engine, f"CleanVoiceDownload.links({json.dumps(homepage_repo())})")
    assert isinstance(links, dict)
    tagged = [attrs for tag, attrs in page.tags if tag == "a" and attrs.get("data-link")]
    assert {attrs["data-link"] for attrs in tagged} == set(links)
    for attrs in tagged:
        assert attrs["href"] == links[str(attrs["data-link"])], attrs
    buttons = [attrs for tag, attrs in page.tags if tag == "a" and "download" in str(attrs.get("class", "")).split()]
    assert {attrs["data-link"] for attrs in buttons} == {"windows", "mac"}


class CableStep(HTMLParser):
    """Step 3 (``#step-cable``) as pieces: plain text, or the default text of a ``data-cable`` element."""

    def __init__(self) -> None:
        super().__init__()
        self.pieces: list[list[str]] = []  # [kind, text], kind "text" or "cable"
        self.depth = 0  # inside #step-cable
        self.cable_depth = 0  # inside a data-cable element

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        found = dict(attrs)
        if self.depth:
            self.depth += 1
            if self.cable_depth:
                self.cable_depth += 1
            elif "data-cable" in found:
                self.cable_depth = 1
                self.pieces.append(["cable", ""])
        elif found.get("id") == "step-cable":
            self.depth = 1

    def handle_endtag(self, tag: str) -> None:
        if self.depth:
            self.depth -= 1
            if self.cable_depth:
                self.cable_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.depth:
            return
        if self.cable_depth:
            self.pieces[-1][1] += data
        else:
            self.pieces.append(["text", data])


# Runs the page's own apply() on a minimal document: only the step's data-cable elements, holding their HTML text.
FAKE_APPLY = """(function (texts, uaData, ua, touch) {
  var els = texts.map(function (t) { return { textContent: t }; });
  var doc = {
    documentElement: { setAttribute: function () {} },
    querySelectorAll: function (sel) { return sel === "[data-cable]" ? els : []; }
  };
  CleanVoiceDownload.apply(doc, { userAgentData: uaData, userAgent: ua, maxTouchPoints: touch },
                           { hostname: "someone.github.io", pathname: "/clean-voice/" });
  return els.map(function (e) { return e.textContent; });
})"""


def step_three(engine: QJSEngine, ua_data: dict | None, user_agent: str | None, touch: int = 0) -> str:
    """Step 3's text as the visitor reads it; ``user_agent=None`` means no JavaScript (the HTML as written)."""
    parser = CableStep()
    parser.feed((SITE / "index.html").read_text(encoding="utf-8"))
    assert parser.pieces, "no #step-cable in index.html"
    cables = [text for kind, text in parser.pieces if kind == "cable"]
    if user_agent is not None:
        args = f"{json.dumps(cables)}, {json.dumps(ua_data)}, {json.dumps(user_agent)}, {touch}"
        filled = call(engine, f"{FAKE_APPLY}({args})")
        assert isinstance(filled, list) and len(filled) == len(cables)
        cables = [str(text) for text in filled]
    out, it = [], iter(cables)
    for kind, text in parser.pieces:
        out.append(next(it) if kind == "cable" else text)
    return " ".join("".join(out).split())


WINDOWS_NAME, MAC_NAME = "CABLE Output", "VB-Cable"  # app.py cable_mic_name: win32, and every other OS


@pytest.mark.parametrize(
    ("ua_data", "user_agent", "touch", "names"),
    [
        (None, WINDOWS_CHROME, 0, {WINDOWS_NAME}),
        ({"platform": "Windows", "mobile": False}, WINDOWS_CHROME, 0, {WINDOWS_NAME}),
        (None, MAC_SAFARI, 0, {MAC_NAME}),
        ({"platform": "macOS", "mobile": False}, MAC_CHROME, 0, {MAC_NAME}),
        (None, LINUX_FIREFOX, 0, {WINDOWS_NAME, MAC_NAME}),
        (None, IPHONE, 5, {WINDOWS_NAME, MAC_NAME}),
        (None, None, 0, {WINDOWS_NAME, MAC_NAME}),  # no JavaScript
    ],
)
def test_step_three_names_the_cable_for_the_computer(
    engine: QJSEngine, ua_data: dict | None, user_agent: str | None, touch: int, names: set[str]
) -> None:
    text = step_three(engine, ua_data, user_agent, touch)
    shown = {name for name in (WINDOWS_NAME, MAC_NAME) if name in text}
    assert shown == names, text
    if names == {WINDOWS_NAME, MAC_NAME}:
        assert f"{WINDOWS_NAME}” on Windows" in text and f"{MAC_NAME}” on Mac" in text, text


class Shots(HTMLParser):
    """The screenshot's ``<source>`` and ``<img>`` (``data-shot``), with the address each has in the HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.found: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        got = dict(attrs)
        if "data-shot" in got:
            attr = "srcset" if tag == "source" else "src"
            self.found.append({"tag": tag, "shot": str(got["data-shot"]), "attr": attr, "url": str(got.get(attr))})


# Runs apply() on a document holding only the screenshot's elements; returns each one's address afterwards.
FAKE_SHOTS = """(function (shots, uaData, ua, touch) {
  var els = shots.map(function (s) {
    var attrs = { "data-shot": s.shot };
    attrs[s.attr] = s.url;
    return {
      tagName: s.tag.toUpperCase(), attrs: attrs,
      getAttribute: function (k) { return k in this.attrs ? this.attrs[k] : null; },
      setAttribute: function (k, v) { this.attrs[k] = v; }
    };
  });
  var doc = {
    documentElement: { setAttribute: function () {} },
    querySelectorAll: function (sel) { return sel === "[data-shot]" ? els : []; }
  };
  CleanVoiceDownload.apply(doc, { userAgentData: uaData, userAgent: ua, maxTouchPoints: touch },
                           { hostname: "someone.github.io", pathname: "/clean-voice/" });
  return els.map(function (e, i) { return e.attrs[shots[i].attr]; });
})"""

WINDOWS_SHOTS = {"light": "running.png", "dark": "running-dark.png"}
MAC_SHOTS = {"light": "running-mac.png", "dark": "running-mac-dark.png"}


@pytest.mark.parametrize(
    ("ua_data", "user_agent", "touch", "expected"),
    [
        (None, WINDOWS_CHROME, 0, WINDOWS_SHOTS),
        (None, MAC_SAFARI, 0, MAC_SHOTS),
        ({"platform": "macOS", "mobile": False}, MAC_CHROME, 0, MAC_SHOTS),
        (None, LINUX_FIREFOX, 0, WINDOWS_SHOTS),
        (None, IPHONE, 5, WINDOWS_SHOTS),
        (None, None, 0, WINDOWS_SHOTS),  # no JavaScript: the HTML's own
    ],
)
def test_the_screenshot_shows_the_cable_name_of_the_computer(
    engine: QJSEngine, ua_data: dict | None, user_agent: str | None, touch: int, expected: dict[str, str]
) -> None:
    # Release r1, m4: the Windows screenshot says "Pick CABLE Output", which a Mac does not have.
    parser = Shots()
    parser.feed((SITE / "index.html").read_text(encoding="utf-8"))
    shots = parser.found
    assert sorted((s["tag"], s["shot"]) for s in shots) == [("img", "light"), ("source", "dark")]
    urls = [s["url"] for s in shots]
    if user_agent is not None:
        args = f"{json.dumps(shots)}, {json.dumps(ua_data)}, {json.dumps(user_agent)}, {touch}"
        filled = call(engine, f"{FAKE_SHOTS}({args})")
        assert isinstance(filled, list)
        urls = [str(url) for url in filled]
    assert {s["shot"]: url for s, url in zip(shots, urls, strict=True)} == expected


def test_the_mac_screenshots_match_the_windows_ones(page: Page) -> None:
    from PySide6.QtGui import QImage

    img = next(attrs for tag, attrs in page.tags if tag == "img")
    size = (int(str(img["width"])), int(str(img["height"])))
    assert size == (760, 684), size  # the redesigned window at 1x (docs/ui/rounds/20/running.png)
    for name in (*WINDOWS_SHOTS.values(), *MAC_SHOTS.values()):
        image = QImage(str(SITE / name))
        assert (image.width(), image.height()) == size, name  # the <img> width and height
    alt = img["alt"]
    assert alt and "CABLE" not in alt and "VB-Cable" not in alt  # true of both screenshots


def _luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_the_step_numbers_are_readable_in_both_themes() -> None:
    # Release r1, m7: WCAG AA for normal text (4.5:1); white on the light green was 3.57:1.
    css = (SITE / "style.css").read_text(encoding="utf-8")
    step = re.search(r"\.steps li::before \{(.*?)\}", css, re.DOTALL)
    assert step and "background: var(--signal)" in step[1] and "color: var(--on-signal)" in step[1]
    light, dark = css.split("@media (prefers-color-scheme: dark)", 1)
    for theme in (light, dark.split("\n}\n", 1)[0]):
        tokens = dict(re.findall(r"--([\w-]+):\s*(#[0-9A-Fa-f]{6})", theme))
        assert _contrast(tokens["on-signal"], tokens["signal"]) >= 4.5, tokens


def test_the_guest_guide_is_not_linked() -> None:
    # The host emails the guide to guests; it is not published (decision 2026-11-12).
    for name in ("index.html", "download.js"):
        assert "guide" not in (SITE / name).read_text(encoding="utf-8").lower(), name


def test_nothing_loads_from_elsewhere(page: Page) -> None:
    # No outside scripts, styles, fonts or images (no trackers): only files beside the page.
    for tag, attrs in page.tags:
        for key in ("src", "srcset") + (("href",) if tag == "link" else ()):
            value = attrs.get(key)
            if value:
                assert not re.match(r"(https?:)?//", value), (tag, attrs)
                assert (SITE / value.split()[0]).is_file() or value.startswith("data:"), (tag, attrs)
    css = (SITE / "style.css").read_text(encoding="utf-8")
    assert "@import" not in css
    for url in re.findall(r"url\(([^)]*)\)", css):
        url = url.strip("'\"")
        assert (SITE / url).is_file(), url


# ---- B019: Windows and Mac tabs above one card, each with two ways as numbered steps -----------------------------------------


def test_the_commands_are_the_ones_the_app_runs(engine: QJSEngine) -> None:
    from clean_voice import updates

    commands = call(engine, 'CleanVoiceDownload.commands("someone/clean-voice")')
    assert commands == {
        "windows": updates.install_command_text("someone/clean-voice", "win32"),
        "mac": updates.install_command_text("someone/clean-voice", "darwin"),
    }
    # The scripts the commands fetch are the ones the release workflow uploads (tools/ci/release_assets.py).
    script = (ROOT / "tools" / "ci" / "release_assets.py").read_text(encoding="utf-8")
    assert 'SCRIPT_NAMES = ("install.ps1", "install.sh")' in script
    assert call(engine, "CleanVoiceDownload.SCRIPTS") == {"windows": "install.ps1", "mac": "install.sh"}


def test_without_javascript_both_computers_show_and_the_commands_are_right(engine: QJSEngine, page: Page) -> None:
    commands = call(engine, f"CleanVoiceDownload.commands({json.dumps(homepage_repo())})")
    assert isinstance(commands, dict)
    html = (SITE / "index.html").read_text(encoding="utf-8")
    for os_name, command in commands.items():
        assert f'<code class="command" data-command="{os_name}">{command}</code>' in html
    # Both panels are in the HTML itself: nothing hides one of them, and no link swaps them.
    panels = [attrs for tag, attrs in page.tags if tag == "section" and attrs.get("data-os-target")]
    assert [attrs["data-os-target"] for attrs in panels] == ["windows", "mac"]  # Windows first: on top on a phone
    assert "data-show" not in html and "Show the Mac options" not in html and "Show the Windows" not in html
    copies = [attrs for tag, attrs in page.tags if tag == "button" and "data-copy" in attrs]
    assert {attrs["data-copy"] for attrs in copies} == {"windows", "mac"}


def _page_text() -> str:
    return " ".join((SITE / "index.html").read_text(encoding="utf-8").split())


def _panel(html: str, os_name: str) -> str:
    start = html.index(f'data-os-target="{os_name}"')
    ends = [i for i in (html.find('data-os-target="', start + 1), html.find("</header>", start)) if i > start]
    return html[start : min(ends)]


def test_each_panel_has_way_1_then_way_2_and_no_badge() -> None:
    html = _page_text()
    for os_name in ("windows", "mac"):
        panel = _panel(html, os_name)
        way1, way2 = panel.index("Way 1: the installer"), panel.index("Way 2: one command")
        assert way1 < panel.index("Download for") < way2 < panel.index("data-command=")
    assert "recommended" not in html.lower()


def test_smart_app_control_note_comes_before_the_windows_download_button() -> None:
    html = _page_text()
    windows = _panel(html, "windows")
    note = windows.index("First, turn off Smart App Control.")
    assert windows.index('class="note note-warn"') <= note < windows.index("Click the <strong>Start</strong> button")
    assert note < windows.index('class="download"')
    assert "Download for Mac" not in windows
    assert "Smart App Control" not in _panel(html, "mac")


def _steps(panel: str, way: str) -> list[str]:
    """The numbered steps of one way as plain text (tags dropped, spaces collapsed)."""
    start = panel.index(f"{way}</h3>")
    end = panel.find("<h3", start)
    block = panel[start : end if end > 0 else len(panel)]
    return [" ".join(_text(item).split()) for item in re.findall(r"<li>(.*?)</li>", block, re.DOTALL)]


def _text(fragment: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", re.sub(r"</?(span|a|div|button|code)[^>]*>", " ", fragment)))


def test_the_steps_read_as_the_mockup_says() -> None:
    html = _page_text()
    windows, mac = _panel(html, "windows"), _panel(html, "mac")
    base = "https://github.com/paulcaesarflores/clean-voice/releases/latest/download/"
    assert _steps(windows, "Way 1: the installer") == [
        "Click the Start button, type Smart App Control, and open it.",
        "Choose Off. (Nothing found, or it's already off? Go to step 3. Windows 10 doesn't have it.)",
        "Click Download for Windows (the button, right here). Download for Windows Windows 10 or 11",
        "Open the file you downloaded and follow the installer.",
        "If Windows says “Windows protected your PC”, click More info, then Run anyway.",
    ]
    assert _steps(windows, "Way 2: one command") == [
        f"Click Copy. irm {base}install.ps1 | iex Copy",
        "Right-click the Start button and choose Terminal (on Windows 10: Windows PowerShell).",
        "Press Ctrl+V, then Enter.",
        "Wait about 10 minutes. Clean Voice opens by itself. If it stops, do it again.",
    ]
    assert _steps(mac, "Way 1: the installer") == [
        "Click Download for Mac (the button, right here). Download for Mac macOS 13 or later, Apple Silicon and Intel",
        "Open the file you downloaded and drag Clean Voice into Applications.",
        (
            "Open Clean Voice. If your Mac says it can't open it: System Settings → Privacy & Security, "
            "scroll down, click Open Anyway."
        ),
    ]
    assert _steps(mac, "Way 2: one command") == [
        f"Click Copy. curl -fsSL {base}install.sh | bash Copy",
        "Open Terminal: press Cmd+Space, type Terminal, press Return.",
        "Press Cmd+V, then Return.",
        "If your Mac asks to install developer tools, click Install.",
        "Wait about 10 minutes. Clean Voice opens by itself. If it stops, do it again.",
    ]


def test_the_mockup_texts_are_on_the_page() -> None:
    html = _page_text()
    for text in (
        "Find your computer below. Pick Way 1 or Way 2. Both give you the same app.",
        "Clean Voice is safe, free, and its code is public. Windows blocks it only because it isn't signed.",
        "Signing costs a monthly fee to Microsoft, which this free app doesn't pay.",
        "Install Clean Voice (above), then open it and pick your microphone.",
    ):
        assert text in html, text
    # B019-4: "Windows or Mac warns you?" stays until a real PC run shows no warning, unchanged.
    assert '<h2 id="warn-title">Windows or Mac warns you?</h2>' in html
    assert "click <em>More info</em>, then <em>Run anyway</em>" in html
    # B019-7 (ledger decision): what the command downloads is a Note for the orchestrator, not guest-facing jargon.
    assert "PyPI" not in html
    assert "MinGit" not in html


def test_the_tabs_are_an_accessible_tablist(page: Page) -> None:
    lists = [attrs for tag, attrs in page.tags if attrs.get("role") == "tablist"]
    assert len(lists) == 1 and lists[0].get("aria-label")
    tabs = [attrs for tag, attrs in page.tags if attrs.get("role") == "tab"]
    assert [t["data-tab"] for t in tabs] == ["windows", "mac"]  # Windows first
    assert [t["aria-selected"] for t in tabs] == ["true", "false"]
    assert [t.get("tabindex") for t in tabs] == [None, "-1"]  # one tab stop; arrow keys move
    panels = [attrs for tag, attrs in page.tags if attrs.get("role") == "tabpanel"]
    assert [p["data-os-target"] for p in panels] == ["windows", "mac"]
    for tab, panel in zip(tabs, panels, strict=True):
        assert tab["aria-controls"] == panel["id"] and panel["aria-labelledby"] == tab["id"]
        assert "hidden" not in panel  # without JavaScript both show
    js = (SITE / "download.js").read_text(encoding="utf-8")
    for key in ("ArrowLeft", "ArrowRight", "Home", "End"):
        assert key in js


def test_without_javascript_the_tabs_are_hidden_and_both_cards_show() -> None:
    css = (SITE / "style.css").read_text(encoding="utf-8")
    assert re.search(r"\.tabs \{[^}]*display: none;", css)
    assert re.search(r"html\[data-os\] \.tabs \{[^}]*display: flex;", css)
    assert re.search(r"\.tab\[aria-selected=\"true\"\] \{[^}]*background: var\(--desk\);", css)
    assert (
        "grid-template-columns"
        not in css.split("/* ---------------------------------------------------------------- the three steps")[0]
    )
    assert "os-block" not in css and "other-os" not in css  # the one-computer swap is gone


# Runs apply() on a document of two tabs and two panels (and their keys); returns [tab states, panel hidden flags].
FAKE_TABS = """(function (uaData, ua, touch, then) {
  function el(attrs) {
    return { attrs: attrs, hidden: false,
             getAttribute: function (k) { return k in this.attrs ? this.attrs[k] : null; },
             setAttribute: function (k, v) { this.attrs[k] = v; } };
  }
  var tabs = [el({ "data-tab": "windows" }), el({ "data-tab": "mac" })];
  var panels = [el({ "data-os-target": "windows" }), el({ "data-os-target": "mac" })];
  var doc = { documentElement: { setAttribute: function () {} },
              querySelectorAll: function (sel) {
                return sel === "[data-tab]" ? tabs : sel === "[data-os-target]" ? panels : []; } };
  CleanVoiceDownload.apply(doc, { userAgentData: uaData, userAgent: ua, maxTouchPoints: touch },
                           { hostname: "someone.github.io", pathname: "/clean-voice/" });
  if (then) CleanVoiceDownload.selectTab(doc, then);
  return [tabs.map(function (t) { return t.attrs["aria-selected"]; }),
          panels.map(function (p) { return p.hidden; })];
})"""


@pytest.mark.parametrize(
    ("ua_data", "user_agent", "touch", "selected"),
    [
        (None, WINDOWS_CHROME, 0, "windows"),
        (None, MAC_SAFARI, 0, "mac"),
        ({"platform": "macOS", "mobile": False}, MAC_CHROME, 0, "mac"),
        (None, LINUX_FIREFOX, 0, "windows"),  # an unknown computer: Windows first, as without JavaScript
        (None, IPHONE, 5, "windows"),
    ],
)
def test_the_tab_for_the_visitors_computer_is_selected(
    engine: QJSEngine, ua_data: dict | None, user_agent: str, touch: int, selected: str
) -> None:
    args = f"{json.dumps(ua_data)}, {json.dumps(user_agent)}, {touch}"
    states, hidden = call(engine, f"{FAKE_TABS}({args})")  # type: ignore[misc]
    assert states == (["true", "false"] if selected == "windows" else ["false", "true"])
    assert hidden == ([False, True] if selected == "windows" else [True, False])


def test_choosing_the_other_tab_shows_the_other_panel(engine: QJSEngine) -> None:
    args = f"null, {json.dumps(WINDOWS_CHROME)}, 0"
    assert call(engine, f"{FAKE_TABS}({args}, 'mac')") == [["false", "true"], [True, False]]
    assert call(engine, f"{FAKE_TABS}(null, {json.dumps(MAC_SAFARI)}, 0, 'windows')") == [
        ["true", "false"],
        [False, True],
    ]


def test_apply_fills_each_command_from_the_pages_own_address(engine: QJSEngine) -> None:
    program = """(function () {
      var boxes = [{ textContent: "", attrs: { "data-command": "windows" } },
                   { textContent: "", attrs: { "data-command": "mac" } }];
      boxes.forEach(function (b) { b.getAttribute = function (k) { return this.attrs[k]; }; });
      var doc = { documentElement: { setAttribute: function () {} },
                  querySelectorAll: function (sel) { return sel === "[data-command]" ? boxes : []; } };
      CleanVoiceDownload.apply(doc, { userAgentData: null, userAgent: "", maxTouchPoints: 0 },
                               { hostname: "someone.github.io", pathname: "/clean-voice/" });
      return boxes.map(function (b) { return b.textContent; });
    })()"""
    assert call(engine, program) == [
        "irm https://github.com/someone/clean-voice/releases/latest/download/install.ps1 | iex",
        "curl -fsSL https://github.com/someone/clean-voice/releases/latest/download/install.sh | bash",
    ]


def test_copy_puts_the_command_on_the_clipboard(engine: QJSEngine) -> None:
    program = """(function () {
      var written = [], said = [];
      var nav = { clipboard: { writeText: function (t) { written.push(t); return { then: function (ok) { ok(); } }; } } };
      var ok = function () { said.push("copied"); }, no = function () { said.push("failed"); };
      CleanVoiceDownload.copyCommand(nav, "irm x | iex", ok, no);
      CleanVoiceDownload.copyCommand({}, "irm x | iex", ok, no);
      return written.concat(said);
    })()"""
    assert call(engine, program) == ["irm x | iex", "copied", "failed"]


def test_the_warning_note_is_readable_in_both_themes() -> None:
    css = (SITE / "style.css").read_text(encoding="utf-8")
    light, dark = css.split("@media (prefers-color-scheme: dark)", 1)
    for theme in (light, dark.split("\n}\n", 1)[0]):
        tokens = dict(re.findall(r"--([\w-]+):\s*(#[0-9A-Fa-f]{6})", theme))
        assert _contrast(tokens["warn-ink"], tokens["warn-bg"]) >= 4.5, tokens
        assert _contrast(tokens["ink"], tokens["warn-bg"]) >= 4.5, tokens
