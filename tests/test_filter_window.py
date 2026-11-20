"""J10 Step 10, the filter window (docs/plans/J10-one-window.md §6, Tests T31-T34), against OBS's Filters dialog
(frontend/dialogs/OBSBasicFilters.cpp), its properties view (shared/properties-view/properties-view.cpp) and obs-vst's
properties (plugins/obs-vst/obs-vst.cpp)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QPushButton, QSpinBox
from pytestqt.qtbot import QtBot
from test_main_screen import MICS, ON, OUTPUTS, FakeLoginItem
from vst_fake import FakeLibrary, FakePlugin

from clean_voice import filter_window as fw
from clean_voice import fine_tune, prefs, vst, vst_editor
from clean_voice.app import MainWindow
from clean_voice.filter_window import FilterModel, FilterWindow
from clean_voice.fine_tune import BUTTON, CHECKBOX, HIDDEN, INT, KINDS, LIST, SLIDER, FilterSpec
from clean_voice.preview import PreviewEngine, PreviewMicLevel
from clean_voice.settings import Settings, replace

# OBS's ten audio filters as its + menu lists them: sorted by display name (CreateAddFilterPopupMenu :458-516), the
# names from plugins/obs-filters/data/locale/en-US.ini and plugins/obs-vst/data/locale/en-US.ini:3. The user's
# screenshot docs/ui/mockups/J10/5-obs-add-menu.png shows exactly this list.
OBS_MENU = [
    "3-Band Equalizer",
    "Compressor",
    "Expander",
    "Gain",
    "Invert Polarity",
    "Limiter",
    "Noise Gate",
    "Noise Suppression",
    "Upward Compressor",
    "VST 2.x Plug-in",
]

PLUGIN_PATH = "C:/Program Files/VSTPlugins/Fake Gain.dll"
OTHER_PATH = "C:/Program Files/VSTPlugins/Other Gain.dll"
FOUND = [("Fake Gain", PLUGIN_PATH), ("Other Gain", OTHER_PATH)]


class Host:
    """What the main window gives the model: the current Settings, and every publish recorded."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.published: list[Settings] = []
        self.registry = vst.VstRegistry()
        self.unlisted_at_publish: list[int] = []  # VST uids with a path published before their plug-in was added

    def publish(self, settings: Settings) -> None:
        for f in settings.filters:
            if f.kind == vst.KIND and f.settings.get("plugin_path") and self.registry.plugin(f.uid) is None:
                self.unlisted_at_publish.append(f.uid)
        self.settings = settings
        self.published.append(settings)


class Fakes:
    """A loader that loads a new FakePlugin for each path (kept alive here), or raises ``fail`` for a path."""

    def __init__(self) -> None:
        self.loaded: list[tuple[str, FakePlugin, vst.VstPlugin]] = []
        self.fail: dict[str, vst.VstError] = {}
        self.flags = vst.FLAG_CAN_REPLACING | vst.FLAG_HAS_EDITOR

    def __call__(self, settings: Mapping[str, object]) -> vst.VstPlugin | None:
        path = str(settings.get("plugin_path") or "")
        if path in self.fail:
            raise self.fail[path]
        fake = FakePlugin(flags=self.flags, name=Path(path).stem)
        plugin = vst.create(settings, opener=lambda p: FakeLibrary({"VSTPluginMain": fake}))
        if plugin is not None:
            self.loaded.append((path, fake, plugin))
        return plugin

    def fake(self, path: str) -> FakePlugin:
        return [f for p, f, _ in self.loaded if p == path][-1]


@pytest.fixture
def make(qtbot: QtBot):
    windows: list[FilterWindow] = []

    def make(
        settings: Settings | None = None,
        found: list[tuple[str, str]] | None = None,
        loader: Callable[[Mapping[str, object]], vst.VstPlugin | None] | None = None,
    ) -> tuple[FilterWindow, Host]:
        host = Host(settings)
        model = FilterModel(
            lambda: host.settings,
            host.publish,
            host.registry,
            loader or Fakes(),
            lambda: list(FOUND if found is None else found),
            lambda: [Path("C:/Program Files/VSTPlugins")],
        )
        model.load_saved()
        window = FilterWindow(model)
        window.asked: list[str] = []  # type: ignore[attr-defined]
        window.informed: list[tuple[str, str]] = []  # type: ignore[attr-defined]
        window.confirmed: list[tuple[str, str]] = []  # type: ignore[attr-defined]
        window.answers: list[str | None] = []  # type: ignore[attr-defined]
        window.yes = True  # type: ignore[attr-defined]

        def ask_name(default: str) -> str | None:
            window.asked.append(default)  # type: ignore[attr-defined]
            answers = window.answers  # type: ignore[attr-defined]
            return answers.pop(0) if answers else default

        def confirm(title: str, text: str) -> bool:
            window.confirmed.append((title, text))  # type: ignore[attr-defined]
            return window.yes  # type: ignore[attr-defined]

        window.ask_name = ask_name  # type: ignore[method-assign]
        window.confirm = confirm  # type: ignore[method-assign]
        window.inform = lambda title, text: window.informed.append((title, text))  # type: ignore[method-assign,attr-defined]
        qtbot.addWidget(window)
        window.show()
        qtbot.waitExposed(window)
        window.activateWindow()
        qtbot.waitActive(window)  # its shortcuts work in the active window only, as OBS's
        windows.append(window)
        return window, host

    yield make
    for window in windows:
        window.model.registry.close_all()


def names(host: Host) -> list[str]:
    return [f.name for f in host.settings.filters]


def focus_list(window: FilterWindow) -> None:
    window.list.setFocus()
    QApplication.processEvents()
    assert window.list.hasFocus()


def row_point(window: FilterWindow, row: int, x: int) -> QPoint:
    rect = window.list.visualItemRect(window.list.item(row))
    return QPoint(x, rect.center().y())


# ------------------------------------------------------------------ T31 the list


def test_the_add_menu_lists_clean_voices_two_then_obss_ten_kinds_in_obss_order(make) -> None:
    window, _ = make()
    actions = window.add_menu.actions()
    assert actions[2].isSeparator()  # J13: Denoise, Voice Style, a separator, then OBS's
    filters = [a for a in actions if not a.isSeparator()]
    assert [a.text() for a in filters] == ["Denoise", "Voice Style", *OBS_MENU]
    assert [a.data() for a in filters] == list(fine_tune.MENU_ORDER)
    assert sorted(a.data() for a in filters) == sorted(KINDS)  # every kind, each once


def test_t31_adding_asks_for_a_name_with_obss_rule_and_puts_it_at_the_bottom_selected(make) -> None:
    window, host = make()
    window.add_filter("compressor_filter")
    window.add_filter("gain_filter")
    window.add_filter("compressor_filter")
    assert window.asked == ["Compressor", "Gain", "Compressor 2"]  # type: ignore[attr-defined]
    assert names(host) == ["Compressor", "Gain", "Compressor 2"]
    assert window.list.currentRow() == 2 and window.selected() is host.settings.filters[2]
    assert [window.list.item(i).text() for i in range(3)] == names(host)
    assert host.settings.filters[0].settings == fine_tune.defaults("compressor_filter")


def test_t31_an_empty_or_taken_name_is_refused_with_a_message_and_asked_again(make) -> None:
    window, host = make()
    window.add_filter("gain_filter")
    window.answers = ["   ", "Gain", "My gain"]  # type: ignore[attr-defined]
    window.add_filter("gain_filter")
    assert window.informed == [fw.NAME_EMPTY, fw.NAME_TAKEN]  # type: ignore[attr-defined]
    assert window.asked == ["Gain", "Gain 2", "Gain 2", "Gain 2"]  # type: ignore[attr-defined]
    assert names(host) == ["Gain", "My gain"]
    window.answers = [None]  # type: ignore[attr-defined]  # Cancel: nothing added
    assert window.add_filter("limiter_filter") is None
    assert names(host) == ["Gain", "My gain"]


def test_adding_a_filter_turns_fine_tune_on(make) -> None:
    window, host = make()
    assert not host.settings.fine_tune
    window.add_filter("gain_filter")
    assert host.settings.fine_tune  # the plan's question 6
    window.model.reset(host.settings.filters[0].uid)
    assert host.settings.fine_tune


def test_t31_the_eye_switches_a_filter_and_says_so(make, qtbot: QtBot) -> None:
    window, host = make()
    window.add_filter("gain_filter")
    window.add_filter("limiter_filter")
    window.list.setCurrentRow(1)
    QTest.mouseClick(window.list.viewport(), Qt.MouseButton.LeftButton, pos=row_point(window, 0, 14))
    assert [f.enabled for f in host.settings.filters] == [False, True]
    assert window.list.currentRow() == 1  # the eye doesn't select its row
    assert window.list.item(0).data(fw.ENABLED_ROLE) is False
    focus_list(window)
    window.list.setCurrentRow(1)
    QTest.keyClick(window.list, Qt.Key.Key_Space)
    assert [f.enabled for f in host.settings.filters] == [False, False]


def test_t31_up_down_and_ctrl_arrows_reorder(make) -> None:
    window, host = make()
    for kind in ("gain_filter", "limiter_filter", "compressor_filter"):
        window.add_filter(kind)
    window.list.setCurrentRow(2)
    window.up_button.click()
    assert names(host) == ["Gain", "Compressor", "Limiter"] and window.list.currentRow() == 1
    window.up_button.click()
    window.up_button.click()  # already at the top: nothing
    assert names(host) == ["Compressor", "Gain", "Limiter"] and window.list.currentRow() == 0
    window.down_button.click()
    assert names(host) == ["Gain", "Compressor", "Limiter"] and window.list.currentRow() == 1
    focus_list(window)
    QTest.keyClick(window.list, Qt.Key.Key_Down, Qt.KeyboardModifier.ControlModifier)
    assert names(host) == ["Gain", "Limiter", "Compressor"] and window.list.currentRow() == 2
    QTest.keyClick(window.list, Qt.Key.Key_Up, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(window.list, Qt.Key.Key_Up, Qt.KeyboardModifier.ControlModifier)
    assert names(host) == ["Compressor", "Gain", "Limiter"] and window.list.currentRow() == 0


def test_t31_del_and_the_bin_ask_then_remove(make) -> None:
    window, host = make()
    window.add_filter("gain_filter")
    window.add_filter("limiter_filter")
    window.list.setCurrentRow(0)
    window.yes = False  # type: ignore[attr-defined]
    window.remove_button.click()
    assert window.confirmed == [(fw.REMOVE_TITLE, 'Remove "Gain"?')]  # type: ignore[attr-defined]
    assert names(host) == ["Gain", "Limiter"]
    window.yes = True  # type: ignore[attr-defined]
    window.remove_button.click()
    assert names(host) == ["Limiter"] and window.list.currentRow() == 0
    focus_list(window)
    QTest.keyClick(window.list, Qt.Key.Key_Delete)
    assert names(host) == [] and window.list.count() == 0


def test_the_empty_list_says_how_to_start(make) -> None:
    from PySide6.QtWidgets import QLabel

    window, _ = make()
    texts = [label.text() for label in window.pane_scroll.widget().findChildren(QLabel)]
    assert texts == [fw.EMPTY_TEXT]
    assert not window.defaults_button.isEnabled() and not window.remove_button.isEnabled()


def test_t31_a_double_click_renames_and_refuses_a_taken_or_empty_name(make, qtbot: QtBot) -> None:
    window, host = make()
    window.add_filter("gain_filter")
    window.add_filter("limiter_filter")
    viewport = window.list.viewport()

    def rename(row: int, text: str) -> None:
        point = row_point(window, row, 80)
        QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=point)
        editors = [e for e in window.list.findChildren(QLineEdit, "renameEdit") if e.isVisible()]
        assert len(editors) == 1  # the rename editor, in the row (an earlier one may wait for deleteLater)
        editor = editors[0]
        editor.selectAll()
        QTest.keyClick(editor, Qt.Key.Key_Backspace)
        QTest.keyClicks(editor, text)
        QTest.keyClick(editor, Qt.Key.Key_Return)
        QApplication.processEvents()

    rename(0, "  Make-up gain  ")
    assert names(host) == ["Make-up gain", "Limiter"]
    assert window.list.item(0).text() == "Make-up gain"
    rename(1, "Make-up gain")
    assert window.informed == [fw.NAME_TAKEN]  # type: ignore[attr-defined]
    assert names(host) == ["Make-up gain", "Limiter"] and window.list.item(1).text() == "Limiter"
    rename(1, "")
    assert window.informed[-1] == fw.NAME_EMPTY  # type: ignore[attr-defined]
    assert names(host) == ["Make-up gain", "Limiter"]


# ------------------------------------------------------------------ T32 the properties pane


@pytest.mark.parametrize("kind", list(KINDS))
def test_t32_each_kinds_pane_has_obss_controls(make, kind: str) -> None:
    window, host = make()
    window.add_filter(kind)
    pane = window.pane
    assert pane is not None
    spec = host.settings.filters[0]
    shown = [p for p in KINDS[kind].props if p.widget != HIDDEN]
    assert list(pane.controls) == [p.key for p in shown]
    for prop in shown:
        control = pane.controls[prop.key]
        if prop.widget in (SLIDER, INT):
            assert isinstance(control, QSpinBox if prop.is_int else QDoubleSpinBox), prop.key
            assert (prop.key in pane.sliders) is (prop.widget == SLIDER), prop.key
            assert control.suffix() == prop.suffix
            assert (control.minimum(), control.maximum()) == (prop.lo, prop.hi)
            assert control.singleStep() == pytest.approx(prop.step)
            assert control.value() == pytest.approx(float(spec.settings[prop.key]))  # type: ignore[arg-type]
            if isinstance(control, QDoubleSpinBox):
                assert control.decimals() == 2  # AddFloat :495-501: QDoubleSpinBox's two for every OBS step here
            assert pane.labels[prop.key].text() == prop.label
        elif prop.widget == LIST:
            assert isinstance(control, QComboBox)
            assert pane.labels[prop.key].text() == prop.label
            items = [(control.itemText(i), control.itemData(i)) for i in range(control.count())]
            if prop.key == "plugin_path":
                assert items == [(fw.CHOOSE_PLUGIN, ""), *FOUND]
            else:
                assert items == [(label, value) for value, label in prop.options]
                assert control.currentData() == spec.settings[prop.key]
        elif prop.widget == CHECKBOX:
            assert isinstance(control, QCheckBox) and control.text() == prop.label
            assert prop.key not in pane.labels  # AddCheckbox: the box carries its words
        elif prop.widget == BUTTON:
            assert isinstance(control, QPushButton) and control.text() == prop.label
    texts = [w.text() for w in pane.findChildren(type(pane.status)) if w.property("role") == "paneNote"]
    if kind == "noise_suppress_filter":
        assert texts == ["RNNoise (good quality, more CPU usage)"]
    elif kind == "invert_polarity_filter":
        assert texts == [fw.NO_PROPERTIES]
    else:
        assert texts == []


def test_t32_obss_number_boxes_read_as_in_the_screenshot(make) -> None:
    window, _ = make()
    window.add_filter("noise_gate_filter")
    pane = window.pane
    assert pane is not None
    shown = {k: c.text() for k, c in pane.controls.items() if isinstance(c, QSpinBox | QDoubleSpinBox)}
    assert shown == {
        "close_threshold": "-32.00 dB",
        "open_threshold": "-26.00 dB",
        "attack_time": "25 ms",
        "hold_time": "200 ms",
        "release_time": "150 ms",
    }
    assert set(pane.sliders) == {"close_threshold", "open_threshold"}  # the times are boxes only


def test_t32_an_edit_reaches_the_settings_at_once(make) -> None:
    window, host = make()
    window.add_filter("compressor_filter")
    pane = window.pane
    assert pane is not None
    ratio = pane.controls["ratio"]
    assert isinstance(ratio, QDoubleSpinBox)
    ratio.setValue(4.5)
    assert host.settings.filters[0].settings["ratio"] == 4.5
    pane.sliders["threshold"].setValue(pane.sliders["threshold"].maximum())  # the slider drives the box
    assert host.settings.filters[0].settings["threshold"] == pytest.approx(0.0)
    assert pane.controls["threshold"].value() == pytest.approx(0.0)  # type: ignore[attr-defined]
    attack = pane.controls["attack_time"]
    assert isinstance(attack, QSpinBox)
    attack.setValue(30)
    assert host.settings.filters[0].settings["attack_time"] == 30
    assert pane.sliders["attack_time"].value() == 30
    assert window.pane is pane  # an edit doesn't rebuild the pane


def test_t32_the_expanders_preset_switch_resets_that_presets_defaults(make) -> None:
    window, host = make()
    window.add_filter("expander_filter")
    pane = window.pane
    assert pane is not None
    pane.controls["threshold"].setValue(-20.0)  # type: ignore[attr-defined]
    presets = pane.controls["presets"]
    assert isinstance(presets, QComboBox)
    presets.setCurrentIndex(presets.findData("gate"))
    # expander_update :190-195: obs_data_clear, the preset, then expander_defaults for the gate.
    assert dict(host.settings.filters[0].settings) == fine_tune.defaults("expander_filter", {"presets": "gate"})
    assert host.settings.filters[0].settings["ratio"] == 10.0
    assert host.settings.filters[0].settings["release_time"] == 125
    pane = window.pane
    assert pane is not None
    assert pane.controls["ratio"].value() == 10.0  # type: ignore[attr-defined]  # the pane shows them
    assert pane.controls["threshold"].value() == -40.0  # type: ignore[attr-defined]
    presets = pane.controls["presets"]
    assert isinstance(presets, QComboBox)
    presets.setCurrentIndex(presets.findData("expander"))
    assert dict(host.settings.filters[0].settings) == fine_tune.defaults("expander_filter")


def test_t32_defaults_asks_then_resets(make) -> None:
    window, host = make()
    window.add_filter("gain_filter")
    pane = window.pane
    assert pane is not None
    pane.controls["db"].setValue(6.0)  # type: ignore[attr-defined]
    window.yes = False  # type: ignore[attr-defined]
    window.defaults_button.click()
    assert window.confirmed == [(fw.RESET_TITLE, fw.RESET_TEXT)]  # type: ignore[attr-defined]
    assert host.settings.filters[0].settings["db"] == 6.0
    window.yes = True  # type: ignore[attr-defined]
    window.defaults_button.click()
    assert host.settings.filters[0].settings["db"] == 0.0
    assert window.pane is not None and window.pane.controls["db"].value() == 0.0  # type: ignore[attr-defined]


def test_t32_an_edit_in_the_window_reaches_the_engine(qtbot: QtBot, tmp_path: Path) -> None:
    engine = PreviewEngine(ON, MICS, tmp_path, OUTPUTS)
    window = MainWindow(engine, tmp_path / "s.json", FakeLoginItem(), "win32", mic_levels=PreviewMicLevel())
    qtbot.addWidget(window)
    window.show_plugin_button.click()
    filters = window.filter_window
    assert filters is not None and filters.isVisible()
    filters.ask_name = lambda default: default  # type: ignore[method-assign]
    filters.add_filter("gain_filter")
    assert engine.settings().fine_tune and [f.kind for f in engine.settings().filters] == ["gain_filter"]
    assert window.fine_tune_toggle.isChecked()  # Fine-tune turned on, and the switch shows it
    pane = filters.pane
    assert pane is not None
    pane.controls["db"].setValue(-3.0)  # type: ignore[attr-defined]
    assert engine.settings().filters[0].settings["db"] == -3.0
    assert window.filters_label.text() == "1 filter"
    window.show_plugin_button.click()  # a second click raises the open window
    assert window.filter_window is filters
    filters.close()
    engine.close()


# ------------------------------------------------------------------ T33 the VST pane


def vst_window(make, found: list[tuple[str, str]] | None = None, fakes: Fakes | None = None):
    fakes = fakes or Fakes()
    window, host = make(found=found, loader=fakes)
    window.add_filter("vst_filter")
    return window, host, fakes


def combo(window: FilterWindow) -> QComboBox:
    assert window.pane is not None
    control = window.pane.controls["plugin_path"]
    assert isinstance(control, QComboBox)
    return control


def choose(window: FilterWindow, path: str) -> None:
    box = combo(window)
    box.setCurrentIndex(box.findData(path))


def shown(window: FilterWindow, key: str) -> bool:
    assert window.pane is not None
    return window.pane.form.isRowVisible(window.pane.controls[key])


def test_t33_choosing_a_plugin_loads_it_into_the_registry_first(make) -> None:
    window, host, fakes = vst_window(make)
    first_uid = host.settings.filters[0].uid
    assert not shown(window, "open_vst_settings") and not shown(window, "close_vst_settings")  # nothing loaded
    choose(window, PLUGIN_PATH)
    spec = host.settings.filters[0]
    assert spec.settings["plugin_path"] == PLUGIN_PATH
    assert spec.uid != first_uid  # one uid, one plug-in
    plugin = window.model.plugin(spec.uid)
    assert plugin is not None and plugin is fakes.loaded[0][2]
    assert host.unlisted_at_publish == []  # in the registry before any Settings named it
    assert window.pane is not None and window.pane.status.text() == "Fake Gain by Tests Inc"
    assert shown(window, "open_vst_settings") and not shown(window, "close_vst_settings")
    assert combo(window).currentData() == PLUGIN_PATH
    # Another plug-in: a new uid, and the first one is unloaded (no chain runs here).
    choose(window, OTHER_PATH)
    assert host.settings.filters[0].uid not in (first_uid, spec.uid)
    assert fakes.fake(PLUGIN_PATH).closed and not plugin.loaded
    assert window.pane.status.text() == "Other Gain by Tests Inc"  # type: ignore[union-attr]
    # "Choose a plug-in": none, and the plug-in goes.
    choose(window, "")
    assert fakes.fake(OTHER_PATH).closed
    assert window.model.plugin(host.settings.filters[0].uid) is None
    assert not shown(window, "open_vst_settings")


def test_t33_a_plugin_that_fails_to_load_says_so_and_stays_in_the_list(make) -> None:
    fakes = Fakes()
    fakes.fail[PLUGIN_PATH] = vst.VstError(vst.WRONG_ARCHITECTURE)
    window, host, _ = vst_window(make, fakes=fakes)
    choose(window, PLUGIN_PATH)
    assert names(host) == ["VST 2.x Plug-in"]
    assert host.settings.filters[0].settings["plugin_path"] == PLUGIN_PATH
    assert window.pane is not None
    assert window.pane.status.text() == (
        "Clean Voice couldn't load this plug-in: it is made for a different kind of computer (a 32-bit plug-in, or "
        "one for another processor). Try another plug-in or another version of it."
    )
    assert window.pane.status.property("warn") is True and window.pane.status.isVisibleTo(window)
    assert not shown(window, "open_vst_settings") and not shown(window, "close_vst_settings")

    def crash(settings: Mapping[str, object]) -> vst.VstPlugin | None:
        raise OSError("access violation")

    window.model._loader = crash
    choose(window, OTHER_PATH)  # anything else a plug-in throws: plain words too, no crash
    assert fw.STARTUP_ERROR in window.pane.status.text()  # type: ignore[union-attr]


def test_t33_no_plugins_found_names_the_folders(make) -> None:
    window, _, _ = vst_window(make, found=[])
    assert combo(window).count() == 1
    assert window.pane is not None
    assert window.pane.status.text().splitlines() == [
        "No VST 2 plug-ins were found in the usual folders.",
        str(Path("C:/Program Files/VSTPlugins")),
    ]


@pytest.mark.parametrize("found", [None, []], ids=["plugins-found", "none-found"])
def test_m5_the_vst_pane_ends_with_steinbergs_trademark_line(make, found: list[tuple[str, str]] | None) -> None:
    # Review J10 r1 m5 (plan question 10): there is no About text, so the line sits where a VST user sees it.
    window, _, _ = vst_window(make, found=found)
    pane = window.pane
    assert pane is not None
    assert fw.VST_TRADEMARK == "VST is a trademark of Steinberg Media Technologies GmbH."
    line = pane.trademark
    assert line is not None and line.text() == fw.VST_TRADEMARK and line.isVisibleTo(window)
    assert pane.form.getWidgetPosition(line)[0] == pane.form.rowCount() - 1  # the pane's last row
    below = [w for w in (pane.status, *pane.controls.values()) if w.isVisibleTo(window)]
    assert all(w.mapTo(pane, w.rect().bottomLeft()).y() < line.mapTo(pane, line.rect().topLeft()).y() for w in below)
    # J10 r2 m3: plan §7's type scale, 12 px the minimum; muted like the pane's status line.
    line.ensurePolished()
    assert line.font().pixelSize() == 12
    assert line.palette().color(line.foregroundRole()).name() == QColor(window.palette_now.muted).name()
    window.add_filter("gain_filter")  # no other kind's pane carries it
    assert window.pane is not None and window.pane.trademark is None


def test_t33_a_saved_plugin_no_longer_found_is_cleared_as_obs_does(make) -> None:
    # AddList :700-708: the saved path isn't in the list, so the list's first entry becomes the choice.
    fakes = Fakes()
    saved = FilterSpec("vst_filter", "VST 2.x Plug-in", settings={"plugin_path": "C:/gone/Old.dll"})
    window, host = make(Settings(fine_tune=True, filters=(saved,)), loader=fakes)
    assert fakes.loaded and fakes.loaded[0][0] == "C:/gone/Old.dll"  # loaded at launch
    assert host.settings.filters[0].settings["plugin_path"] == ""
    assert fakes.fake("C:/gone/Old.dll").closed
    assert combo(window).currentIndex() == 0


def test_t33_open_and_close_interface_swap_as_vst_changed_does(make, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[tuple[str, str]] = []

    class Editor:
        def __init__(self, plugin: vst.VstPlugin) -> None:
            self.plugin = plugin

        def request_resize(self, width: int, height: int) -> None: ...

        def close(self) -> bool:
            self.plugin.editor = None
            return True

    def open_editor(plugin: vst.VstPlugin, filter_name: str = "") -> None:
        opened.append((plugin.effect_name, filter_name))
        plugin.editor = Editor(plugin)

    monkeypatch.setattr(vst_editor, "open_editor", open_editor)
    window, host, _ = vst_window(make)
    choose(window, PLUGIN_PATH)
    assert window.pane is not None
    window.pane.controls["open_vst_settings"].click()  # type: ignore[attr-defined]
    assert opened == [("Fake Gain", "VST 2.x Plug-in")]
    assert not shown(window, "open_vst_settings") and shown(window, "close_vst_settings")
    window.pane.controls["close_vst_settings"].click()  # type: ignore[attr-defined]
    assert window.model.plugin(host.settings.filters[0].uid).editor is None  # type: ignore[union-attr]
    assert shown(window, "open_vst_settings") and not shown(window, "close_vst_settings")
    # Open interface when active: saved, handed to the plug-in, and the editor opens when a plug-in loads.
    box = window.pane.controls["open_when_active_vst_settings"]
    assert isinstance(box, QCheckBox)
    box.setChecked(True)
    assert host.settings.filters[0].settings["open_when_active_vst_settings"] is True
    assert window.model.plugin(host.settings.filters[0].uid).open_when_active  # type: ignore[union-attr]
    choose(window, OTHER_PATH)
    assert opened[-1] == ("Other Gain", "VST 2.x Plug-in")


def test_t33_removing_a_vst_filter_unloads_its_plugin_after_publishing(make) -> None:
    window, host, fakes = vst_window(make)
    choose(window, PLUGIN_PATH)
    fake = fakes.fake(PLUGIN_PATH)
    order: list[str] = []
    publish = window.model._publish

    def watch(settings: Settings) -> None:
        order.append(f"publish closed={fake.closed}")
        publish(settings)

    window.model._publish = watch
    window.remove_button.click()
    assert order == ["publish closed=False"] and fake.closed
    assert names(host) == []


def test_t33_defaults_on_a_vst_filter_clears_its_plugin_as_obs_does(make) -> None:
    window, host, fakes = vst_window(make)
    choose(window, PLUGIN_PATH)
    window.defaults_button.click()
    assert host.settings.filters[0].settings["plugin_path"] == ""
    assert fakes.fake(PLUGIN_PATH).closed


# ------------------------------------------------------------------ T34 saving


def test_t34_closing_saves_every_filter_and_the_plugin_state_and_a_relaunch_restores_them(
    qtbot: QtBot, tmp_path: Path
) -> None:
    path = tmp_path / "settings.json"
    fakes = Fakes()
    engine = PreviewEngine(ON, MICS, tmp_path, OUTPUTS)
    window = MainWindow(
        engine, path, FakeLoginItem(), "win32", mic_levels=PreviewMicLevel(), vst_loader=fakes, vst_scan=lambda: FOUND
    )
    qtbot.addWidget(window)
    window.show_filters()
    filters = window.filter_window
    assert filters is not None
    filters.ask_name = lambda default: default  # type: ignore[method-assign]
    for kind in ("noise_gate_filter", "vst_filter", "compressor_filter"):
        filters.add_filter(kind)
    filters.list.setCurrentRow(1)
    choose(filters, PLUGIN_PATH)
    fakes.fake(PLUGIN_PATH).params[0] = 0.25  # changed in the plug-in's own editor: no Settings change
    filters.list.setCurrentRow(2)
    filters.up_button.click()  # Noise Gate, Compressor, VST
    filters.model.rename(filters.model.filters[0].uid, "Gate")
    filters.model.set_enabled(filters.model.filters[1].uid, False)
    filters.close()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [(f["id"], f["name"], f["enabled"]) for f in saved["filters"]] == [
        ("noise_gate_filter", "Gate", True),
        ("compressor_filter", "Compressor", False),
        ("vst_filter", "VST 2.x Plug-in", True),
    ]
    assert saved["fine_tune"] is True
    vst_saved = saved["filters"][2]["settings"]
    assert vst_saved["plugin_path"] == PLUGIN_PATH
    assert vst_saved["chunk_data"]  # the plug-in's parameters (it keeps no chunk), base64
    window.quit_app()
    assert not fakes.loaded[0][2].loaded  # unloaded at quit

    again_fakes = Fakes()
    engine2 = PreviewEngine(ON, MICS, tmp_path, OUTPUTS)
    again = MainWindow(
        engine2,
        path,
        FakeLoginItem(),
        "win32",
        mic_levels=PreviewMicLevel(),
        vst_loader=again_fakes,
        vst_scan=lambda: FOUND,
    )
    qtbot.addWidget(again)
    restored = engine2.settings().filters
    assert [(f.kind, f.name, f.enabled) for f in restored] == [
        ("noise_gate_filter", "Gate", True),
        ("compressor_filter", "Compressor", False),
        ("vst_filter", "VST 2.x Plug-in", True),
    ]
    plugin = engine2.vst.plugin(restored[2].uid)
    assert plugin is not None and again_fakes.fake(PLUGIN_PATH).params[0] == 0.25  # the state came back
    again.show_filters()
    assert again.filter_window is not None and again.filter_window.list.count() == 3
    again.quit_app()


def test_t34_quit_saves_the_plugin_state_with_no_window_open(qtbot: QtBot, tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    fakes = Fakes()
    spec = FilterSpec("vst_filter", "Mine", settings={"plugin_path": PLUGIN_PATH})
    prefs.save(path, prefs.Prefs(settings=Settings(fine_tune=True, filters=(spec,))))
    engine = PreviewEngine(ON, MICS, tmp_path, OUTPUTS)
    window = MainWindow(
        engine, path, FakeLoginItem(), "win32", mic_levels=PreviewMicLevel(), vst_loader=fakes, vst_scan=lambda: FOUND
    )
    qtbot.addWidget(window)
    assert fakes.loaded, "the saved plug-in loads at launch"
    fakes.fake(PLUGIN_PATH).params[0] = 0.5
    window.quit_app()
    saved = prefs.load(path).settings.filters[0].settings
    assert saved["chunk_data"]
    fresh = FakePlugin()
    restored = vst.create(saved, opener=lambda p: FakeLibrary({"VSTPluginMain": fresh}))
    assert restored is not None and fresh.params[0] == 0.5
    restored.close()


def test_with_state_leaves_unchanged_settings_alone(make) -> None:
    window, host, _ = vst_window(make)
    choose(window, PLUGIN_PATH)
    once = window.model.with_state(host.settings)
    assert once is not host.settings  # the first read adds the state
    assert window.model.with_state(once) is once  # nothing changed since: the same Settings
    plain = replace(host.settings, filters=())
    assert window.model.with_state(plain) is plain
