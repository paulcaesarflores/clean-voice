"""Fine-tune's data model and saved choices (J10 plan §2, Tests T10-T12): ``FilterSpec``, the new ``Settings`` switches
and filter list, their JSON form in settings.json, and old files."""

import json
from pathlib import Path

import pytest

from clean_voice import fine_tune, prefs
from clean_voice.fine_tune import KINDS, FilterSpec
from clean_voice.prefs import Prefs
from clean_voice.settings import Settings, style_bands


def one_of_each() -> tuple[FilterSpec, ...]:
    specs: list[FilterSpec] = []
    for kind in fine_tune.MENU_ORDER:
        settings = dict(fine_tune.defaults(kind))
        if kind == "vst_filter":
            settings.update(
                plugin_path="C:/Program Files/VSTPlugins/ReaComp (x64).dll",
                open_when_active_vst_settings=True,
                chunk_data="VkNDaHVuaw==",
                chunk_hash="0f343b0931126a20f133d67c2b018a3b",
            )
        elif kind == "compressor_filter":
            settings.update(ratio=4.5, attack_time=12)
        specs.append(fine_tune.new_filter(kind, specs, settings=settings, enabled=kind != "gain_filter"))
    return tuple(specs)


def test_settings_defaults_and_switches() -> None:
    s = Settings()
    assert (s.denoise, s.style, s.fine_tune, s.filters) == (True, True, False, ())
    with pytest.raises(ValueError):
        Settings(filters=("compressor_filter",))  # type: ignore[arg-type]


def test_filter_spec_is_frozen_resolved_and_has_an_unsaved_uid() -> None:
    a = FilterSpec("compressor_filter", "Compressor", settings={"ratio": 99})
    assert a.settings["ratio"] == 32.0 and a.settings["threshold"] == -18.0  # clamped, defaults filled in
    b = FilterSpec("compressor_filter", "Compressor", settings={"ratio": 32.0})
    assert a == b and a.uid != b.uid  # the uid is internal: never compared, never saved
    with pytest.raises(TypeError):
        a.settings["ratio"] = 2.0  # type: ignore[index]
    with pytest.raises(ValueError):
        FilterSpec("wobble_filter", "Wobble")
    with pytest.raises(ValueError):
        FilterSpec("gain_filter", "")
    assert "uid" not in fine_tune.filter_to_json(a)


def test_t10_round_trip_of_one_filter_of_each_kind(tmp_path: Path) -> None:
    filters = one_of_each()
    assert [f.kind for f in filters] == list(fine_tune.MENU_ORDER)
    p = Prefs(settings=Settings(denoise=False, style=False, fine_tune=True, filters=filters))
    path = tmp_path / "settings.json"
    prefs.save(path, p)
    back = prefs.load(path)
    assert back == p
    assert [f.uid for f in back.settings.filters] != [f.uid for f in filters]  # new uids on load
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert (saved["denoise"], saved["style"], saved["fine_tune"]) == (False, False, True)
    vst = saved["filters"][-1]
    assert set(vst) == {"id", "name", "enabled", "settings"}  # the four fields OBS keeps for a filter
    assert vst["id"] == "vst_filter" and vst["name"] == "VST 2.x Plug-in" and vst["enabled"] is True
    assert vst["settings"] == {
        "plugin_path": "C:/Program Files/VSTPlugins/ReaComp (x64).dll",
        "open_when_active_vst_settings": True,
        "chunk_data": "VkNDaHVuaw==",
        "chunk_hash": "0f343b0931126a20f133d67c2b018a3b",
    }
    gain = next(f for f in saved["filters"] if f["id"] == "gain_filter")
    assert gain == {"id": "gain_filter", "name": "Gain", "enabled": False, "settings": {"db": 0.0}}


OLD_FILE = {  # a J07/J09 settings.json: the 7-band EQ and the sidebar page, no Fine-tune filters
    "preset": "warm",
    "style_strength": 0.8,
    "user_eq": [  # as the deleted user_eq.py saved them (J10 Step 9 deleted the 7-band EQ)
        {"kind": "lowcut", "freq": 80.0, "gain": 0.0, "q": 0.7071067811865475, "on": False, "slope": 12},
        {"kind": "lowshelf", "freq": 120.0, "gain": 0.0, "q": 0.7071067811865475, "on": False, "slope": 12},
        {"kind": "bell", "freq": 250.0, "gain": 0.0, "q": 1.0, "on": False, "slope": 12},
        {"kind": "bell", "freq": 1000.0, "gain": -4.5, "q": 1.0, "on": True, "slope": 12},
        {"kind": "bell", "freq": 3000.0, "gain": 0.0, "q": 1.0, "on": False, "slope": 12},
        {"kind": "highshelf", "freq": 8000.0, "gain": 0.0, "q": 0.7071067811865475, "on": False, "slope": 12},
        {"kind": "highcut", "freq": 16000.0, "gain": 0.0, "q": 0.7071067811865475, "on": False, "slope": 12},
    ],
    "loudness": 62,
    "strength": 35,
    "cleaning": False,
    "echo": True,
    "recording": True,
    "mic": "Mic X",
    "output": "CABLE Input (VB-Audio Virtual Cable)",
    "tab": "tune",
    "monitor_volume": 40,
}


def test_t11_an_old_file_loads_its_choices_with_fine_tune_off_and_empty(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(OLD_FILE), encoding="utf-8")
    p = prefs.load(path)
    s = p.settings
    assert (p.preset, p.style_strength, s.bands) == ("warm", 0.8, style_bands("warm", 0.8))
    assert (s.loudness, s.strength, s.cleaning) == (62.0, 35.0, False)
    assert (p.echo, p.recording, p.mic, p.output, p.monitor_volume) == (
        True,
        True,
        "Mic X",
        "CABLE Input (VB-Audio Virtual Cable)",
        40.0,
    )
    assert (s.denoise, s.style, s.fine_tune, s.filters) == (True, True, False, ())


def test_t11_garbage_filters_load_the_valid_ones_clamped_and_renamed(tmp_path: Path) -> None:
    data = {
        "fine_tune": True,
        "filters": [
            {"id": "wobble_filter", "name": "Wobble", "enabled": True, "settings": {}},  # unknown id: dropped
            "Compressor",  # not an object: dropped
            42,
            {"id": "compressor_filter", "name": "Comp", "enabled": True, "settings": {"ratio": 400, "threshold": -99}},
            {"id": "compressor_filter", "name": "Comp", "settings": {"attack_time": "fast"}},  # duplicate name
            {"id": "gain_filter", "enabled": "yes", "settings": "loud"},  # no name, bad enabled, bad settings
            {"id": "gain_filter", "name": 7, "enabled": False, "settings": {"db": -45.0, "extra": 1}},
            {"id": "noise_gate_filter", "name": "", "settings": {"hold_time": 1e9}},
        ],
    }
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    f = prefs.load(path).settings.filters
    assert [(x.kind, x.name, x.enabled) for x in f] == [
        ("compressor_filter", "Comp", True),
        ("compressor_filter", "Comp 2", True),
        ("gain_filter", "Gain", True),
        ("gain_filter", "Gain 2", False),
        ("noise_gate_filter", "Noise Gate", True),
    ]
    assert (f[0].settings["ratio"], f[0].settings["threshold"]) == (32.0, -60.0)
    assert f[1].settings == fine_tune.defaults("compressor_filter")
    assert f[2].settings == {"db": 0.0}
    assert f[3].settings == {"db": -30.0}
    assert f[4].settings["hold_time"] == 10000
    assert prefs.load(path).settings.fine_tune is True
    for bad in ("filters", {"x": 1}, None, 3):
        path.write_text(json.dumps({"filters": bad, "fine_tune": "on"}), encoding="utf-8")
        assert prefs.load(path) == Prefs()


# Every JSON type a hand-edited file can hold (review J10 r1 m1: a list or object "id" crashed the launch).
JSON_TYPES = {"null": None, "true": True, "int": 3, "float": 2.5, "nan": float("nan"), "string": "x"}
JSON_TYPES |= {"list": ["gain_filter"], "object": {"id": "gain_filter"}}


@pytest.mark.parametrize("value", JSON_TYPES.values(), ids=list(JSON_TYPES))
@pytest.mark.parametrize("field", ["id", "name", "enabled", "settings"])
def test_t11_each_wrong_json_type_in_a_filter_field_is_dropped_or_defaulted(
    tmp_path: Path, field: str, value: object
) -> None:
    entry: dict[str, object] = {"id": "gain_filter", "name": "Loud", "enabled": False, "settings": {"db": 6.0}}
    entry[field] = value
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"fine_tune": True, "filters": [entry]}), encoding="utf-8")
    f = prefs.load(path).settings.filters
    if field == "id":
        assert f == ()  # no kind: the entry is dropped
        return
    (x,) = f
    assert x.kind == "gain_filter"
    want = {"name": ("Loud", "Gain"), "enabled": (False, True), "settings": ({"db": 6.0}, {"db": 0.0})}
    good = {k: v[0] for k, v in want.items()}
    good[field] = want[field][1]
    if field == "name" and value == "x":
        good["name"] = "x"  # a string is a valid name
    assert (x.name, x.enabled, dict(x.settings)) == (good["name"], good["enabled"], good["settings"])


# One setting of each widget kind: slider, int, checkbox, list (Expander's presets), free text (VST's chunk).
SETTING_CASES = [
    ("gain_filter", "db"),
    ("noise_gate_filter", "hold_time"),
    ("vst_filter", "open_when_active_vst_settings"),
    ("expander_filter", "presets"),
    ("vst_filter", "chunk_data"),
    ("vst_filter", "plugin_path"),
]


@pytest.mark.parametrize("value", JSON_TYPES.values(), ids=list(JSON_TYPES))
@pytest.mark.parametrize(("kind", "key"), SETTING_CASES, ids=[f"{k}.{s}" for k, s in SETTING_CASES])
def test_t11_each_wrong_json_type_in_a_setting_gives_its_default(
    tmp_path: Path, kind: str, key: str, value: object
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"filters": [{"id": kind, "settings": {key: value}}]}), encoding="utf-8")
    (x,) = prefs.load(path).settings.filters
    prop = next(p for p in KINDS[kind].props if p.key == key)
    got = x.settings[key]
    ok = fine_tune._valid(prop, value)  # pyright: ignore[reportPrivateUsage]
    if ok is None:
        assert got == fine_tune.defaults(kind)[key]
    else:  # the right type: kept (clamped)
        assert got == ok and type(got) is type(fine_tune.defaults(kind)[key])
    assert all(isinstance(v, bool | int | float | str) for v in x.settings.values())


def test_t12_the_name_rule() -> None:
    # OBSBasicFilters::AddNewFilter :522-529: the kind's name, then "Name 2", "Name 3"...
    first = fine_tune.new_filter("compressor_filter", ())
    second = fine_tune.new_filter("compressor_filter", (first,))
    third = fine_tune.new_filter("compressor_filter", (first, second))
    assert (first.name, second.name, third.name) == ("Compressor", "Compressor 2", "Compressor 3")
    assert first.settings == fine_tune.defaults("compressor_filter") and first.enabled
    assert fine_tune.unique_name("Gain", ["Gain", "Gain 3"]) == "Gain 2"
    assert fine_tune.unique_name("Gain", ["Gain 2"]) == "Gain"
    assert all(fine_tune.new_filter(k, ()).name == KINDS[k].name for k in KINDS)
