"""Does the main window's footer fit (DESIGN "Theme choice", 2026-11-13; theme-choice r1 F1)? One row, nothing cut,
the close note on one line at the opening size and never one word alone, and no taller than the switch row it always
had.

test_theme_choice.py checks this in its own (offscreen) process, and runs this file as a script on the real Windows
platform at several display scales: offscreen keeps 100 % text metrics at every ``QT_SCALE_FACTOR``, while Windows'
own text at 125 % and 150 % is wider and taller, which once wrapped the note with "tray." alone.

As a script: ``python tests/footer_fit.py`` with ``QT_QPA_PLATFORM`` and ``QT_SCALE_FACTOR`` set by the caller; prints
each problem and exits 1 if there is any."""

from __future__ import annotations

import itertools
import math
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QFontMetricsF, QImage, QRegion, QTextLayout, QTextOption
from PySide6.QtWidgets import QApplication, QLabel, QStyle, QStyleOptionComboBox, QWidget

from clean_voice.app import MIN_SIZE, WINDOW_SIZE, MainWindow

# The note's room beyond its one-line width at the opening size, so the platform's rounding cannot take it: at 125 %
# the row's other widgets are 1 px wider each and the text 3.5 px wider than at 100 % (review r1, probe_footer_widths).
NOTE_SLACK = 8
FOOTER_H = 49  # the footer with its switch row (the fold test counts on it)


def note_lines(label: QLabel) -> list[str]:
    """The label's text as Qt breaks it at the label's width and font, one string per line."""
    layout = QTextLayout(label.text(), label.font())
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WordWrap)
    layout.setTextOption(option)
    lines: list[str] = []
    layout.beginLayout()
    while (line := layout.createLine()).isValid():
        line.setLineWidth(label.contentsRect().width())
        lines.append(label.text()[line.textStart() : line.textStart() + line.textLength()].strip())
    layout.endLayout()
    return lines


def inked_edges(label: QLabel) -> list[str]:
    """The label's top and bottom device-pixel rows that its drawn text reaches: the widget clips there, so ink in
    either row is a glyph that may be cut (theme-choice r2 F2: the last line's descenders at 125 % and 175 %). Drawn
    as the screen draws it, at the label's device pixel ratio, over a transparent image."""
    dpr = label.devicePixelRatioF()
    image = QImage(
        math.ceil(label.width() * dpr), math.ceil(label.height() * dpr), QImage.Format.Format_ARGB32_Premultiplied
    )
    image.setDevicePixelRatio(dpr)
    image.fill(Qt.GlobalColor.transparent)
    label.render(image, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)  # the text only, no background
    found: list[str] = []
    for name, y in (("top", 0), ("bottom", image.height() - 1)):
        inked = sum(1 for x in range(image.width()) if image.pixelColor(x, y).alpha() > 0)
        if inked:
            found.append(f"text ink in its {name} device row ({inked} px, dpr {dpr}, {label.height()} px tall)")
    return found


def problems(window: MainWindow, opening: bool) -> list[str]:
    """What does not fit in ``window``'s footer; ``opening``: the window is at its opening size, else its minimum."""
    found: list[str] = []
    footer = window.footer
    row = [
        window.close_note,
        window.autostart_label,
        window.autostart_switch,
        window.theme_label,
        window.theme_combo,
        window.version_label,
        window.updates_view.check_button,
    ]
    rects = [QRect(w.mapTo(footer, w.rect().topLeft()), w.size()) for w in row]
    for widget, rect in zip(row, rects, strict=True):
        if not (widget.isVisibleTo(window) and footer.rect().contains(rect)):
            found.append(f"{type(widget).__name__} outside the footer")
    for left, right in itertools.pairwise(rects):
        if left.right() >= right.left():
            found.append("two footer widgets overlap")
    note = window.close_note
    found += [f"close note cut: {edge}" for edge in inked_edges(note)]
    lines = note_lines(note)
    if opening:
        if len(lines) > 1:
            found.append(f"close note on two lines at the opening size: {lines}")
        need = math.ceil(QFontMetricsF(note.font()).horizontalAdvance(note.text())) + NOTE_SLACK
        if note.contentsRect().width() < need:
            found.append(f"close note has {note.contentsRect().width()} px, needs {need} (its line + slack)")
    if len(lines) > 2:
        found.append(f"close note on {len(lines)} lines: {lines}")
    if len(lines) > 1 and len(lines[-1].split()) <= 1:
        found.append(f"one word alone: {lines}")
    if window.theme_label.fontMetrics().horizontalAdvance("Theme") > window.theme_label.width():
        found.append("Theme label cut")
    combo = window.theme_combo
    current = combo.currentIndex()
    for i in range(combo.count()):  # each choice fits the box the style really draws its text in
        combo.setCurrentIndex(i)
        option = QStyleOptionComboBox()
        combo.initStyleOption(option)
        field = combo.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox, option, QStyle.SubControl.SC_ComboBoxEditField, combo
        )
        if combo.fontMetrics().horizontalAdvance(combo.itemText(i)) > field.width():
            found.append(f"{combo.itemText(i)!r} cut in the Theme box")
        if field.right() >= combo.width() - 26:
            found.append(f"{combo.itemText(i)!r} runs into the chevron")
    combo.setCurrentIndex(current)
    if footer.height() > FOOTER_H:
        found.append(f"footer {footer.height()} px, taller than {FOOTER_H}")
    return found


def main() -> int:
    app = QApplication([sys.argv[0]])
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_main_screen import DEVICES, MICS, OUTPUTS, FakeLoginItem

    from clean_voice import prefs, theme
    from clean_voice.preview import PreviewEngine, sample_states

    theme.request_scheme = lambda mode: None  # never ask the platform for a scheme: this is a layout check
    tmp = Path(tempfile.mkdtemp(prefix="cv-footer-fit-"))
    found: list[str] = []
    for platform, size in itertools.product(("win32", "darwin"), (WINDOW_SIZE, MIN_SIZE)):
        sample = sample_states(DEVICES, platform)["running"]
        prefs.save(tmp / "s.json", sample.saved())
        engine = PreviewEngine(sample.status, MICS, tmp, OUTPUTS)
        window = MainWindow(engine, tmp / "s.json", FakeLoginItem(), platform)
        window.resize(*size)
        window.show()
        window.tick()
        for _ in range(5):
            app.processEvents()
        found += [f"{platform} {size}: {p}" for p in problems(window, size == WINDOW_SIZE)]
        window.close()
    print(app.platformName(), "dpr", app.primaryScreen().devicePixelRatio(), "fits" if not found else "")
    print(*found, sep="\n")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
