"""B013: Fine-tune presets. The whole filter chain saved under a name, loaded back, deleted. Not in OBS: the user asked
for it and approved it on 2026-11-20 (docs/fixes.md). Stored in settings.json beside the chain (``prefs``)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from PySide6.QtWidgets import QPushButton
from pytestqt.qtbot import QtBot
from test_filter_window import FOUND, PLUGIN_PATH, Fakes, Host

from clean_voice import fine_tune, prefs, vst
from clean_voice.filter_window import FilterModel, FilterWindow
from clean_voice.fine_tune import FilterSpec
from clean_voice.settings import Settings


def chain() -> tuple[FilterSpec, ...]:
    return (
        FilterSpec("gain_filter", "Gain", True, {"db": 6.5}),
        FilterSpec("compressor_filter", "Squash", False, {"ratio": 4.0, "threshold": -30.0}),
        FilterSpec(fine_tune.EQ_KIND, "10-Band EQ", True, {"b3_on": True, "b3_gain": 3.0}),
        FilterSpec("vst_filter", "VST 2.x Plug-in", True, {"plugin_path": PLUGIN_PATH, "chunk_data": "QUJD"}),
    )


def spec_data(filters: tuple[FilterSpec, ...]) -> list[dict]:
    return fine_tune.filters_to_json(filters)


def new_model(host: Host, loader=None) -> FilterModel:
    model = FilterModel(
        lambda: host.settings,
        host.publish,
        host.registry,
        loader or Fakes(),
        lambda: list(FOUND),
        lambda: [Path("C:/Program Files/VSTPlugins")],
    )
    return model


# ------------------------------------------------------------------ the model


def test_save_then_load_gives_the_identical_chain() -> None:
    host = Host(Settings(filters=chain(), fine_tune=True))
    model = new_model(host)
    model.load_saved()
    before = spec_data(model.with_state(host.settings).filters)  # the chain as saved: the plug-in's state read in
    model.save_preset("Podcast")
    for spec in list(host.settings.filters):  # change the chain: remove everything, add one filter
        model.remove(spec.uid)
    model.add("gain_filter", "Other")
    assert [f.name for f in host.settings.filters] == ["Other"]
    model.load_preset("Podcast")
    assert spec_data(host.settings.filters) == before  # every filter, order, enabled state and setting
    assert [f.enabled for f in host.settings.filters] == [True, False, True, True]
    assert model.registry.plugin(host.settings.filters[3].uid) is not None  # the VST plug-in is loaded again
    model.registry.close_all()


def test_load_switches_fine_tune_on_as_adding_a_filter_does() -> None:
    host = Host(Settings(filters=chain()[:2], fine_tune=False))
    model = new_model(host)
    model.save_preset("A")
    assert not host.settings.fine_tune
    model.load_preset("A")
    assert host.settings.fine_tune


def test_load_gives_filters_new_uids_so_loading_twice_is_safe() -> None:
    host = Host(Settings(filters=chain()[:2]))
    model = new_model(host)
    model.save_preset("A")
    model.load_preset("A")
    first = {f.uid for f in host.settings.filters}
    model.load_preset("A")
    assert not first & {f.uid for f in host.settings.filters}
    assert len(host.settings.filters) == 2


def test_save_keeps_a_plug_ins_current_state() -> None:
    host = Host(Settings(filters=chain()[3:]))
    model = new_model(host)
    model.load_saved()
    plugin = model.registry.plugin(host.settings.filters[0].uid)
    assert plugin is not None
    model.save_preset("V")
    saved = model.presets()["V"][0]
    assert saved.settings["plugin_path"] == PLUGIN_PATH
    model.registry.close_all()


def test_a_preset_whose_plugin_is_missing_loads_what_it_can() -> None:
    host = Host(Settings(filters=chain()))
    loader = Fakes()
    model = new_model(host, loader)
    model.load_saved()
    model.save_preset("P")
    loader.fail[PLUGIN_PATH] = vst.VstError("not found")
    model.load_preset("P")
    assert [f.name for f in host.settings.filters] == [f.name for f in chain()]  # the whole chain, plug-in's slot kept
    last = host.settings.filters[3]
    assert model.registry.plugin(last.uid) is None
    assert model.errors[last.uid] == "not found"
    assert host.settings.filters[0].settings["db"] == 6.5
    model.registry.close_all()


def test_delete_removes_the_preset() -> None:
    host = Host(Settings(filters=chain()[:1]))
    model = new_model(host)
    model.save_preset("A")
    model.save_preset("B")
    assert model.preset_names() == ["A", "B"]
    model.delete_preset("A")
    assert model.preset_names() == ["B"]
    model.load_preset("A")  # an unknown name changes nothing
    assert [f.name for f in host.settings.filters] == ["Gain"]


def test_saving_the_same_name_replaces_it() -> None:
    host = Host(Settings(filters=chain()[:1]))
    model = new_model(host)
    model.save_preset("A")
    host.settings = Settings(filters=chain()[1:2])
    model.save_preset("A")
    assert model.preset_names() == ["A"]
    assert [f.name for f in model.presets()["A"]] == ["Squash"]


# ------------------------------------------------------------------ the saved choices


def test_presets_survive_a_prefs_reload(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    p = prefs.Prefs(presets={"Podcast": chain(), "Empty": ()})
    prefs.save(path, p)
    back = prefs.load(path)
    assert set(back.presets) == {"Podcast", "Empty"}
    assert spec_data(back.presets["Podcast"]) == spec_data(chain())
    assert back.presets["Empty"] == ()


def test_a_file_without_presets_or_with_bad_ones_loads() -> None:
    assert prefs.from_json({}).presets == {}
    assert prefs.from_json({"presets": [1, 2]}).presets == {}
    bad = {"presets": {"": [], "ok": [{"id": "gain_filter"}], "x": "nope", "y": [{"id": "nonsense"}]}}
    got = prefs.from_json(bad).presets
    assert set(got) == {"ok", "y"}  # no empty name, no non-list chain
    assert [f.kind for f in got["ok"]] == ["gain_filter"]


# ------------------------------------------------------------------ the window


@pytest.fixture
def window(qtbot: QtBot):
    made: list[FilterWindow] = []

    def make(filters: tuple[FilterSpec, ...] = ()) -> tuple[FilterWindow, Host]:
        host = Host(Settings(filters=filters))
        w = FilterWindow(new_model(host), runtime=lambda: None)
        w.asked = []  # type: ignore[attr-defined]
        w.confirmed = []  # type: ignore[attr-defined]
        w.informed = []  # type: ignore[attr-defined]
        w.name_answers = []  # type: ignore[attr-defined]
        w.yes = True  # type: ignore[attr-defined]

        def ask(default: str) -> str | None:
            w.asked.append(default)  # type: ignore[attr-defined]
            answers = w.name_answers  # type: ignore[attr-defined]
            return answers.pop(0) if answers else default

        def confirm(title: str, text: str) -> bool:
            w.confirmed.append((title, text))  # type: ignore[attr-defined]
            return w.yes  # type: ignore[attr-defined]

        w.ask_preset_name = ask  # type: ignore[method-assign]
        w.confirm = confirm  # type: ignore[method-assign]
        w.inform = lambda title, text: w.informed.append((title, text))  # type: ignore[method-assign,attr-defined]
        qtbot.addWidget(w)
        made.append(w)
        return w, host

    yield make
    for w in made:
        w.model.registry.close_all()


def test_with_no_presets_load_and_delete_are_greyed_and_save_is_not(window) -> None:
    w, _ = window()
    assert w.preset_save_button.isEnabled()
    assert not w.preset_load_button.isEnabled()
    assert not w.preset_delete_button.isEnabled()
    assert w.preset_combo.count() == 0
    assert [b.text() for b in (w.preset_save_button, w.preset_load_button, w.preset_delete_button)] == [
        "Save",
        "Load",
        "Delete",
    ]
    assert all(
        isinstance(b, QPushButton) and b.property("role") == "btn"
        for b in w.findChildren(QPushButton)
        if b.text() in ("Save", "Load", "Delete")
    )


def test_saving_adds_it_to_the_list_selects_it_and_enables_load_and_delete(window) -> None:
    w, _ = window(chain()[:2])
    w.name_answers = ["Podcast"]
    w.preset_save_button.click()
    assert w.asked == [""]  # nothing selected: no prefill
    assert [w.preset_combo.itemText(i) for i in range(w.preset_combo.count())] == ["Podcast"]
    assert w.preset_combo.currentText() == "Podcast"
    assert w.preset_load_button.isEnabled()
    assert w.preset_delete_button.isEnabled()


def test_save_is_prefilled_with_the_selected_name_and_replacing_asks_first(window) -> None:
    w, host = window(chain()[:1])
    w.name_answers = ["A"]
    w.preset_save_button.click()
    host.settings = Settings(filters=chain()[1:2])
    w.confirmed.clear()
    w.yes = False
    w.preset_save_button.click()  # prefilled "A", the answer is the default: the name exists
    assert w.asked[-1] == "A"
    assert len(w.confirmed) == 1
    assert "A" in w.confirmed[0][1]
    assert [f.name for f in w.model.presets()["A"]] == ["Gain"]  # No: kept
    w.yes = True
    w.preset_save_button.click()
    assert len(w.confirmed) == 2
    assert [f.name for f in w.model.presets()["A"]] == ["Squash"]  # Yes: replaced


def test_a_new_name_is_saved_without_a_question_and_an_empty_one_is_refused(window) -> None:
    w, _ = window(chain()[:1])
    w.name_answers = ["   ", "New"]
    w.preset_save_button.click()
    assert len(w.informed) == 1
    assert w.confirmed == []
    assert w.model.preset_names() == ["New"]


def test_cancelling_the_name_saves_nothing(window) -> None:
    w, _ = window(chain()[:1])
    w.ask_preset_name = lambda default: None  # type: ignore[method-assign]
    w.preset_save_button.click()
    assert w.model.preset_names() == []


def test_load_replaces_the_chain_and_the_list(window) -> None:
    w, host = window(chain()[:2])
    w.name_answers = ["A"]
    w.preset_save_button.click()
    w.model.remove(host.settings.filters[0].uid)
    w.refresh(select=0)
    assert w.list.count() == 1
    w.preset_load_button.click()
    assert w.list.count() == 2
    assert spec_data(host.settings.filters) == spec_data(chain()[:2])


def test_delete_asks_then_removes_and_greys_the_buttons(window) -> None:
    w, _ = window(chain()[:1])
    w.name_answers = ["A"]
    w.preset_save_button.click()
    w.yes = False
    w.preset_delete_button.click()
    assert w.model.preset_names() == ["A"]  # No: kept
    w.yes = True
    w.preset_delete_button.click()
    assert w.model.preset_names() == []
    assert w.preset_combo.count() == 0
    assert not w.preset_load_button.isEnabled()
    assert not w.preset_delete_button.isEnabled()


def test_a_window_opened_later_lists_the_stored_presets(window) -> None:
    host = Host(Settings(filters=chain()[:1]))
    stored: dict[str, tuple[FilterSpec, ...]] = {"Kept": chain()[:2]}

    def keep(p: Mapping[str, tuple[FilterSpec, ...]]) -> None:
        stored.clear()
        stored.update(p)

    model = FilterModel(
        lambda: host.settings,
        host.publish,
        host.registry,
        Fakes(),
        list,
        list,
        presets=lambda: stored,
        store_presets=keep,
    )
    w = FilterWindow(model, runtime=lambda: None)
    assert [w.preset_combo.itemText(i) for i in range(w.preset_combo.count())] == ["Kept"]
    assert w.preset_combo.currentIndex() == -1  # nothing selected yet
    assert not w.preset_load_button.isEnabled()
    w.preset_combo.setCurrentIndex(0)
    assert w.preset_load_button.isEnabled()
    w.close()
