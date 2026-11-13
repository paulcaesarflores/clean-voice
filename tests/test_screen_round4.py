"""Round 4 of the main screen, kept for the one window (J10): the bundled Barlow, Clean Voice's own words, no middle
dots and nothing cut off, in every state at the opening and the minimum size. Its page checks (the sidebar's banner,
the quick bar, the styles row, the plain rows) went with the pages; the headline and the mic row moved to
test_one_window.py."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QPushButton, QWidget
from pytestqt.qtbot import QtBot
from test_main_screen import DEVICES, ON, FakeLoginItem, make  # noqa: F401

from clean_voice import fonts, prefs
from clean_voice.app import MIN_SIZE, WINDOW_SIZE, MainWindow
from clean_voice.preview import sample_states

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_UI = ROOT / "reference" / "simple-clean-mic-0.1.0" / "ui"


@pytest.fixture
def screen(request: pytest.FixtureRequest):
    """test_main_screen's window factory, under this file's own name."""
    return request.getfixturevalue("make")


# ------------------------------------------------------------------ type: Barlow, bundled


def test_barlow_is_bundled_with_its_licence_and_used(qapp: QApplication) -> None:
    for name in (*fonts.FILES, "OFL.txt"):
        assert (fonts.FOLDER / name).is_file(), name
    assert "SIL Open Font License" in (fonts.FOLDER / "OFL.txt").read_text(encoding="utf-8")
    assert {"Barlow", "Barlow Semi Condensed"} <= set(fonts.load())
    assert {"Barlow", "Barlow Semi Condensed"} <= set(QFontDatabase.families())
    # Briefcase packages the whole package folder, so the fonts ship in the MSI and DMG.
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "src/clean_voice" in config["tool"]["briefcase"]["app"]["clean-voice"]["sources"]
    assert fonts.FOLDER == ROOT / "src" / "clean_voice" / "fonts"
    # The system fonts stay behind Barlow for Japanese.
    assert fonts.body()[0] == "Barlow" and "Yu Gothic UI" in fonts.body() and "Hiragino Sans" in fonts.body()


def test_the_window_uses_barlow(screen) -> None:
    window, _ = screen(ON)
    assert window.font().families()[0] == "Barlow"
    assert window.headline.font().families()[0] == "Barlow Semi Condensed"


# ------------------------------------------------------------------ our own words, no middle dots


def _texts(window: MainWindow) -> list[str]:
    """Every string the window can show the guest: labels, buttons, tooltips and the painted widgets' words."""
    out: list[str] = []
    for w in window.findChildren(QWidget):
        if isinstance(w, QLabel | QAbstractButton):
            out.append(w.text())
        out.append(w.toolTip())
        label = getattr(w, "label", None)
        if isinstance(label, str):
            out.append(label)
    return [t for t in out if t]


def _words(text: str) -> list[str]:
    text = re.sub(r"<[^>]+>", " ", text).replace("&nbsp;", " ").replace("’", "'")
    return re.findall(r"[a-z0-9']+", text.lower())


def _runs(text: str, n: int = 5) -> set[str]:
    w = _words(text)
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def _original_runs() -> set[str]:
    if not ORIGINAL_UI.is_dir():
        pytest.skip("the original app is not in reference/ (it is gitignored)")
    html = (ORIGINAL_UI / "index.html").read_text(encoding="utf-8")
    js = (ORIGINAL_UI / "main.js").read_text(encoding="utf-8")
    strings = re.findall(r">([^<>]+)<", html) + re.findall(r"title=\"([^\"]+)\"", html)
    strings += [a or b for a, b in re.findall(r"\"([^\"\n]{12,})\"|`([^`]{12,})`", js)]
    found: set[str] = set()
    for s in strings:
        found |= _runs(s)
    return found


def _every_window_text(qtbot: QtBot, tmp_path: Path) -> list[str]:
    from clean_voice import engine as engine_module
    from clean_voice import tray

    texts = [v for k, v in vars(engine_module).items() if k.isupper() and isinstance(v, str) and " " in v]
    for platform in ("win32", "darwin"):
        for state, sample in sample_states(DEVICES).items():
            path = tmp_path / f"{state}-{platform}.json"
            prefs.save(path, sample.saved())
            window = MainWindow(sample.engine(tmp_path), path, FakeLoginItem(), platform)
            qtbot.addWidget(window)
            window.tick()
            texts += _texts(window)
            if state == "running":
                icon = tray.Tray(window)
                texts += [*icon.actions(), icon.tooltip()]
                icon.hide()
            window.close()
    return texts


def test_no_sentence_of_the_original_survives(qtbot: QtBot, tmp_path: Path) -> None:
    # Every user-visible string, in every state and on both OSes, against main.js and index.html (5-word runs).
    original = _original_runs()
    assert len(original) > 100  # the scan really read the original
    copied = set()
    for text in set(_every_window_text(qtbot, tmp_path)):
        copied |= _runs(text) & original
    assert not copied, "copied from the original: " + " | ".join(sorted(copied))


def test_no_middle_dot_strings_anywhere(qtbot: QtBot, tmp_path: Path) -> None:
    # DIRECTION.md v2, principle 4.
    dotted = [t for t in _every_window_text(qtbot, tmp_path) if "·" in t]
    assert not dotted, dotted


# ------------------------------------------------------------------ nothing cut off


@pytest.mark.parametrize("size", [WINDOW_SIZE, MIN_SIZE], ids=["default", "minimum"])  # J07 r1 m3: the minimum too
def test_nothing_is_cut_off(qtbot: QtBot, tmp_path: Path, size: tuple[int, int]) -> None:
    # No visible words are elided or clipped in any state, except a long device name, which then carries the full
    # name in its tooltip.
    problems = []
    for state, sample in sample_states(DEVICES).items():
        path = tmp_path / f"{state}.json"
        prefs.save(path, sample.saved())
        window = MainWindow(sample.engine(tmp_path), path, FakeLoginItem(), "win32")
        qtbot.addWidget(window)
        window.resize(*size)
        window.show()
        window.tick()
        QApplication.processEvents()
        for w in window.findChildren(QWidget):
            if not w.isVisibleTo(window):
                continue
            cut = getattr(w, "cut_off", None)
            if callable(cut):
                device = getattr(w, "device_name", False) and w.toolTip() == w.text()  # type: ignore[attr-defined]
                if cut() and not device:
                    problems.append((state, type(w).__name__, getattr(w, "text", lambda: "")()))
            elif isinstance(w, QLabel) and not w.wordWrap() and w.textFormat() != Qt.TextFormat.RichText:
                if w.text() and w.fontMetrics().horizontalAdvance(w.text()) > w.contentsRect().width() + 1:
                    problems.append((state, "QLabel", w.text()))
            elif isinstance(w, QPushButton) and w.text() and w.width() < w.minimumSizeHint().width():
                # squeezed below the size its words and padding need
                problems.append((state, "QPushButton", w.text()))
        window.close()
    assert not problems, problems[:10]


# ------------------------------------------------------------------ round 5 polish (docs/ui/rounds/4/CRITIQUE.md)


def test_test_your_mic_is_the_filled_desk_button_and_the_instruction_reads_in_ink(screen) -> None:
    from PySide6.QtGui import QColor

    from clean_voice import theme

    window, _ = screen(ON)
    window.tick()
    desk = QColor(theme.current().desk)
    image = window.test_button.grab().toImage()
    corner = image.pixelColor(8, image.height() // 2)  # inside the button, clear of its words
    assert abs(corner.red() - desk.red()) + abs(corner.green() - desk.green()) + abs(corner.blue() - desk.blue()) < 30
    assert window.use_hint.property("role") == "instruction" and "<b>CABLE Output</b>" in window.use_hint.text()
    sheet = window.style_text()
    assert f'QLabel[role="instruction"] {{ color: {theme.current().ink}; font-size: 13px; }}' in sheet
