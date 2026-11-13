"""Do the title bars and the tray menu follow the Theme choice on the real Windows platform (theme-choice r1 m2)?

test_theme_choice.py runs this script with ``QT_QPA_PLATFORM=windows``, once with Qt's own style for this Windows and
once with ``QT_STYLE_OVERRIDE=windowsvista``, Windows 10's default. It sets the app up as ``app.main`` does, opens the
main window, a dialog, a parentless top-level (as the VST editor's frame) and a parentless menu (as the tray's), then
chooses Dark, Light and Dark again. Each window's dark title bar is read back from DWM; the menu's colour from a grab.
Only this process's colour scheme changes, never the computer's. Prints each problem and exits 1 if there is any."""

from __future__ import annotations

import ctypes
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QProxyStyle, QWidget

DWMWA_USE_IMMERSIVE_DARK_MODE = 20


def dark_title_bar(widget: QWidget) -> int:
    value = ctypes.c_int(-1)
    ctypes.windll.dwmapi.DwmGetWindowAttribute(  # type: ignore[attr-defined]
        ctypes.c_void_p(int(widget.winId())), DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value)
    )
    return value.value


def main() -> int:
    app = QApplication([sys.argv[0]])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_main_screen import FakeLoginItem

    from clean_voice import prefs
    from clean_voice.app import MainWindow, follow_scheme_style
    from clean_voice.engine import State, Status
    from clean_voice.preview import PreviewEngine, PreviewMicLevel

    start_style = app.style().name()
    follow_scheme_style(app)
    tmp = Path(tempfile.mkdtemp(prefix="cv-theme-follow-"))
    prefs.save(tmp / "s.json", prefs.Prefs(theme="light"))
    window = MainWindow(
        PreviewEngine(Status(state=State.ON), ["Mic A"], tmp),
        tmp / "s.json",
        FakeLoginItem(),
        "win32",
        mic_levels=PreviewMicLevel(),
    )
    window.show()
    box = QMessageBox(window)
    box.setText("A question")
    box.show()
    editor = QWidget(None)  # as vst_editor's frame: a parentless top-level with the fixed-size dialog hint
    editor.setWindowFlags(editor.windowFlags() | Qt.WindowType.MSWindowsFixedSizeDialogHint)
    editor.resize(300, 200)
    editor.show()
    menu = QMenu()  # as tray.Tray builds it
    menu.addAction("Open Clean Voice")
    menu.addAction("Quit")
    found: list[str] = []
    for mode in ("dark", "light", "dark"):
        window.choose_theme(mode)
        for _ in range(5):
            app.processEvents()
        want = int(mode == "dark")
        for name, widget in (("main", window), ("question", box), ("editor", editor)):
            if dark_title_bar(widget) != want:
                found.append(f"{mode}: the {name} window's title bar is {'light' if want else 'dark'}")
        menu_dark = menu.grab().toImage().pixelColor(4, 4).lightness() < 128
        if menu_dark != bool(want):
            found.append(f"{mode}: the tray menu is {'dark' if menu_dark else 'light'}")
    style = app.style()
    name = (style.baseStyle() if isinstance(style, QProxyStyle) else style).name()  # through WindowsFusion
    print("style", start_style, "->", name, "all follow" if not found else "")
    print(*found, sep="\n")
    for widget in (box, editor):
        widget.close()
    window.close()
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
