"""Render the original Clean Mic UI (the J07 baseline stub in scratch/ui-baseline/) to PNG files at 520 x 800.

    .venv/Scripts/python tools/ui_baseline.py [--src scratch/ui-baseline] [--out docs/ui/baseline]

PySide6 QtWebEngine is not installed (the project uses PySide6-Essentials), so the pages are rendered by the
system's Edge or Chrome in headless mode, straight from a temporary copy of the files: no web server runs. Browsers
refuse ``fetch()`` on ``file://`` pages, so the copy gets one extra script, ahead of the stub, that answers the
stub's ``fetch("devices.json")`` with that file's contents (the machine's real device names). It also turns CSS
transitions off: headless virtual time does not run them, so a grab would catch e.g. the selected tab mid-fade.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 520, 800  # the original window (tauri.conf.json)

# Output file -> query string of scratch/ui-baseline/tauri-stub.js.
SHOTS = {
    "voice": "tab=voice",
    "cleaning": "tab=clean",
    "fine-tune": "tab=tune",
    "settings": "tab=settings",
    "cable-missing": "state=cable-missing",
    "error": "state=error",
    "cleaning-off": "state=off",
    "mic-menu-open": "menu=1",
}

INLINE_DEVICES = """// Added by tools/ui_baseline.py: file:// pages cannot fetch(), so devices.json is answered from here.
(function () {
  const devices = %s;
  const realFetch = window.fetch.bind(window);
  window.fetch = (url, opts) =>
    String(url).endsWith("devices.json")
      ? Promise.resolve(new Response(JSON.stringify(devices), { headers: { "Content-Type": "application/json" } }))
      : realFetch(url, opts);
  // Show each control in its end state at once (headless virtual time does not run CSS transitions).
  const style = document.createElement("style");
  style.textContent = "*, *::before, *::after { transition: none !important; }";
  document.head.appendChild(style);
})();
"""


def find_browser() -> Path | None:
    """The installed Edge or Chrome, or None."""
    roots = [os.environ.get(k) for k in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
    candidates = [
        Path(r) / sub
        for r in roots
        if r
        for sub in (
            Path("Microsoft/Edge/Application/msedge.exe"),
            Path("Google/Chrome/Application/chrome.exe"),
        )
    ]
    candidates += [
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    ]
    return next((c for c in candidates if c.is_file()), None)


def prepare(src: Path, work: Path) -> Path:
    """Copy the stub to ``work`` with devices.json inlined; return the copied index.html."""
    site = work / "site"
    shutil.copytree(src, site)
    devices = json.loads((src / "devices.json").read_text(encoding="utf-8"))
    (site / "devices-inline.js").write_text(INLINE_DEVICES % json.dumps(devices, ensure_ascii=False), encoding="utf-8")
    index = site / "index.html"
    html = index.read_text(encoding="utf-8")
    stub = '<script src="tauri-stub.js"></script>'
    if stub not in html:
        raise SystemExit(f"{src / 'index.html'} does not load tauri-stub.js")
    index.write_text(html.replace(stub, '<script src="devices-inline.js"></script>\n  ' + stub), encoding="utf-8")
    return index


def wait_for_png(path: Path, timeout: float) -> bool:
    """The browser may hand the work to a detached process: wait until the file exists and stops growing."""
    deadline = time.monotonic() + timeout
    last = -1
    while time.monotonic() < deadline:
        size = path.stat().st_size if path.is_file() else -1
        if size > 0 and size == last:
            return True
        last = size
        time.sleep(0.25)
    return False


def shoot(browser: Path, url: str, out: Path, profile: Path, settle_ms: int) -> bool:
    out.unlink(missing_ok=True)
    cmd = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-extensions",
        "--disable-sync",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--user-data-dir={profile}",
        f"--window-size={WIDTH},{HEIGHT}",
        f"--virtual-time-budget={settle_ms}",  # let the meters, fonts and the stub's button presses settle
        f"--screenshot={out}",
        url,
    ]
    subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    return wait_for_png(out, timeout=60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=ROOT / "scratch" / "ui-baseline", help="the baseline stub")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "ui" / "baseline", help="folder for the PNGs")
    parser.add_argument("--browser", type=Path, help="Edge or Chrome executable (found automatically otherwise)")
    parser.add_argument("--settle-ms", type=int, default=500, help="time the page runs before the grab")
    args = parser.parse_args(argv)

    browser = args.browser or find_browser()
    if browser is None:
        print("no Edge or Chrome found; pass --browser", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ui-baseline-", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        index = prepare(args.src, work)
        for n, (name, query) in enumerate(SHOTS.items()):
            path = (args.out / f"{name}.png").resolve()
            # A fresh profile per shot, so no shot is handed to a browser still finishing the previous one.
            if not shoot(browser, f"{index.as_uri()}?{query}", path, work / f"profile-{n}", args.settle_ms):
                print(f"could not write {path}", file=sys.stderr)
                return 1
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
