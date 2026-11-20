"""Render the Clean Voice screen in every state to PNG files, offscreen (the ui-snapshot skill's render step).

    .venv/Scripts/python tools/ui_snapshot.py --out docs/ui/rounds/1 [--state running ...] [--platform darwin]

Each state comes from ``clean_voice.preview.sample_states()``, shown in the one window through a stand-in engine, in
the light and the dark theme (``<state>.png``, ``<state>-dark.png``) at 1x and 2x (``...@2x.png``), at the window's
opening size unless the state sets its own. The device names are this computer's own (listed, never opened); with
no sound system the lists are empty. The saved choices each state starts from go to a temporary file, never to the
guest's own settings. ``--platform`` renders the text and folders another system's app shows (``darwin``: the Mac's
"VB-Cable"); the default is this computer's. A state with the footer's Theme list open draws the list over the window
where it opens.

The filter window's states (``clean_voice.preview.filter_samples()``, ``filters-*``) are rendered the same way, with
stand-in VST plug-ins; a state that opens the + menu or a question draws it over the window where it opens.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

SCALES = (1, 2)
THEMES = ("light", "dark")  # <state>.png is light, <state>-dark.png dark


class _RoomyScreen:
    """Stands in for QScreen with a big available area (the filter window asks only for ``availableGeometry``)."""

    def __init__(self, area: object) -> None:
        self.area = area

    def availableGeometry(self) -> object:  # QScreen's name
        return self.area


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="folder for the PNG files")
    parser.add_argument("--state", action="append", help="render only this state (repeatable)")
    parser.add_argument("--platform", choices=("win32", "darwin"), default=sys.platform, help="whose text to show")
    args = parser.parse_args(argv)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # The offscreen platform has no font list of its own: point it at the system fonts the real app uses.
    for fonts in (Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts", Path("/System/Library/Fonts")):
        if fonts.is_dir():
            os.environ.setdefault("QT_QPA_FONTDIR", str(fonts))
            break

    from PySide6.QtCore import QPoint, QRect, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter
    from PySide6.QtWidgets import QApplication, QWidget

    from clean_voice import filter_window, prefs, theme
    from clean_voice.app import APP_NAME, UI_FONTS, MainWindow
    from clean_voice.preview import PREVIEW_RUNTIME, filter_samples, machine_devices, sample_states

    class NoLoginItem:
        def is_enabled(self) -> bool:
            return False

        def enable(self, command: list[str]) -> None: ...

        def disable(self) -> None: ...

    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName(APP_NAME)
    # Offscreen Qt defaults to the first font file it finds; use the font the native platform would give the app.
    font = QFont()
    font.setFamilies(UI_FONTS)
    QApplication.setFont(font)
    devices = machine_devices()
    print(f"microphones: {list(devices.mics)!a}")
    print(f"outputs: {list(devices.outputs)!a}")
    states = sample_states(devices, args.platform)
    filters = filter_samples(args.platform)
    unknown = sorted(set(args.state or []) - set(states) - set(filters))
    if unknown:
        print(f"unknown state(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    windows = []  # kept alive until the end, so Python never collects a window Qt still owns
    with tempfile.TemporaryDirectory() as tmp:
        for (name, sample), theme_name in ((item, t) for item in states.items() for t in THEMES):
            if args.state and name not in args.state:
                continue
            prefs_path = Path(tmp) / f"{name}-{theme_name}.json"
            prefs.save(prefs_path, sample.saved())
            window = MainWindow(
                sample.engine(),
                prefs_path,
                login_item=NoLoginItem(),
                theme_name=theme_name,
                platform=args.platform,
                mic_levels=sample.mic_levels(),  # never this computer's own mic level
            )
            windows.append(window)
            if sample.size is not None:
                window.resize(*sample.size)
            window.show()
            for _ in range(20):  # let the meters settle on the sample levels
                window.tick()
                app.processEvents()
            settle = time.monotonic() + 0.25  # and the power ring finishes its 150 ms fade
            while time.monotonic() < settle:
                app.processEvents()
            if sample.focus:
                # As after a Tab press: the window marks the change as the keyboard's, and the control draws its ring.
                window.activateWindow()
                settle = time.monotonic() + 1.0
                while not window.isActiveWindow() and time.monotonic() < settle:
                    app.processEvents()
                target = getattr(window, sample.focus)
                target.setFocus(Qt.FocusReason.TabFocusReason)
                app.processEvents()  # the activation's own focus events clear the mark: set it after them
                window.setAttribute(Qt.WidgetAttribute.WA_KeyboardFocusChange)
                target.update()
            if sample.menu_open:
                window.open_mic_menu()
            popups: list[QWidget] = []
            if sample.theme_menu_open:
                # As on a laptop, where the window fills the screen's height: no room below, so the list opens upward.
                screen = window.screen().availableGeometry()
                window.move(window.x(), screen.bottom() + 1 - window.frameGeometry().height())
                app.processEvents()
                window.theme_combo.showPopup()  # a window of its own: drawn over the main one where it opens
                popups.append(window.theme_combo.view().window())
            app.processEvents()
            for scale in SCALES:
                image = QImage(window.size() * scale, QImage.Format.Format_ARGB32_Premultiplied)
                image.setDevicePixelRatio(scale)
                image.fill(0)
                window.render(image)
                if popups:
                    painter = QPainter(image)
                    origin = window.mapToGlobal(QPoint(0, 0))
                    for popup in popups:
                        popup.render(painter, popup.mapToGlobal(QPoint(0, 0)) - origin)
                    painter.end()
                stem = name if theme_name == "light" else f"{name}-dark"
                path = args.out / (f"{stem}.png" if scale == 1 else f"{stem}@{scale}x.png")
                if not image.save(str(path)):
                    print(f"could not write {path}", file=sys.stderr)
                    return 1
                print(path)
            if popups:
                window.theme_combo.hidePopup()
            window.close()
        for (name, fsample), theme_name in ((item, t) for item in filters.items() for t in THEMES):
            if args.state and name not in args.state:
                continue
            pal = theme.for_name(theme_name)
            theme.use(pal)
            fwindow = filter_window.FilterWindow(fsample.model(), pal, args.platform, runtime=lambda: PREVIEW_RUNTIME)
            windows.append(fwindow)
            # The EQ keeps to its screen's available area (B015); the offscreen screen is 800 x 800, so show it a roomy one.
            fwindow.screen = lambda: _RoomyScreen(QRect(0, 0, 1920, 1080))  # type: ignore[method-assign]
            fwindow.show()
            fwindow.list.setCurrentRow(fsample.select)
            app.processEvents()
            on_top: list[QWidget] = []
            if fsample.menu_open:
                fwindow.add_menu.popup(fwindow.add_menu_pos())
                on_top.append(fwindow.add_menu)
            if fsample.confirm_remove:
                spec = fwindow.selected()
                box = fwindow.confirm_box(
                    filter_window.REMOVE_TITLE, filter_window.remove_text(spec.name if spec else "")
                )
                box.show()
                app.processEvents()
                box.move(fwindow.geometry().center() - box.rect().center() + QPoint(0, 15))
                on_top.append(box)
                windows.append(box)
            if fsample.rename:
                fwindow.list.setFocus()
                fwindow.rename_selected()
            settle = time.monotonic() + 0.1
            while time.monotonic() < settle:
                app.processEvents()
            for scale in SCALES:
                image = QImage(fwindow.size() * scale, QImage.Format.Format_ARGB32_Premultiplied)
                image.setDevicePixelRatio(scale)
                image.fill(0)
                fwindow.render(image)
                painter = QPainter(image)
                origin = fwindow.geometry().topLeft()
                for widget in on_top:
                    offset = widget.geometry().topLeft() - origin
                    if widget.isWindow() and not isinstance(widget, type(fwindow.add_menu)):
                        # A dialog is a window of its own: its frame and title, as the system draws them roughly.
                        frame = QRect(offset.x() - 1, offset.y() - 29, widget.width() + 2, widget.height() + 30)
                        painter.fillRect(frame, QColor(pal.band))
                        painter.setPen(QColor(pal.line))
                        painter.drawRect(frame.adjusted(0, 0, -1, -1))
                        painter.setPen(QColor(pal.ink))
                        painter.setFont(fwindow.font())
                        painter.drawText(
                            frame.adjusted(10, 0, 0, -widget.height() - 1),
                            int(Qt.AlignmentFlag.AlignVCenter),
                            widget.windowTitle(),
                        )
                    widget.render(painter, QPoint(offset.x(), offset.y()))
                painter.end()
                stem = name if theme_name == "light" else f"{name}-dark"
                path = args.out / (f"{stem}.png" if scale == 1 else f"{stem}@{scale}x.png")
                if not image.save(str(path)):
                    print(f"could not write {path}", file=sys.stderr)
                    return 1
                print(path)
            for widget in on_top:
                widget.hide()
            fwindow.close()
            fwindow.model.registry.close_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
