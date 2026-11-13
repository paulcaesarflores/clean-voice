"""Saved choices: the guest's settings survive a restart, and a bad file never blocks the launch (J07 P1-P3, with the
user decisions of 2026-11-11: the backup off by default, the style Strength, the Hear myself volume; J10: no sidebar
page and no 7-band EQ are saved, and files that hold them still load)."""

import dataclasses
import json
import math
from pathlib import Path

import pytest

from clean_voice.prefs import Prefs, load, save
from clean_voice.settings import Settings, bands_for, style_bands


def custom_prefs() -> Prefs:
    return Prefs(
        preset="warm",
        style_strength=0.8,
        settings=Settings(strength=35.0, loudness=62.0, bands=style_bands("warm", 0.8), cleaning=False),
        echo=True,
        recording=True,
        mic="マイク (USB オーディオ デバイス)",
        output="Speakers (Focusrite USB Audio)",
        monitor_volume=40.0,
    )


def test_p1_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "Clean Voice" / "settings.json"  # the folder does not exist yet
    prefs = custom_prefs()
    save(path, prefs)
    assert load(path) == prefs
    # Saved atomically: only the file itself is left behind, no temp file.
    assert [p.name for p in path.parent.iterdir()] == ["settings.json"]


def test_p1_round_trip_of_each_preset(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    for preset in ("warm", "flat", "radio"):
        prefs = Prefs(preset=preset, settings=Settings(bands=bands_for(preset)))
        save(path, prefs)
        assert load(path) == prefs


def test_p1_save_replaces_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    save(path, Prefs())
    save(path, custom_prefs())
    assert load(path) == custom_prefs()


DEFAULTS = Prefs()


def test_p2_defaults() -> None:
    assert DEFAULTS.settings.strength == 70.0
    assert DEFAULTS.preset == "natural" and DEFAULTS.style_strength == 0.5
    assert DEFAULTS.settings.bands == bands_for("natural")  # strength 0.5 is the style as designed
    assert DEFAULTS.echo is False
    assert DEFAULTS.recording is False  # user decision, 2026-11-11
    assert DEFAULTS.monitor_volume == 100.0
    assert DEFAULTS.settings.cleaning is True
    assert DEFAULTS.mic is None and DEFAULTS.output is None


@pytest.mark.parametrize(
    "content",
    [None, b"", b"   ", b"{not json", b"\xff\xfe\x00garbage", b"[1, 2, 3]", b'"natural"', b"null", b"{}"],
    ids=["missing", "empty", "blank", "broken", "binary", "list", "string", "null", "empty-object"],
)
def test_p2_missing_empty_or_corrupt_file_gives_defaults(tmp_path: Path, content: bytes | None) -> None:
    path = tmp_path / "settings.json"
    if content is not None:
        path.write_bytes(content)
    assert load(path) == DEFAULTS


def test_p2_a_folder_in_place_of_the_file_gives_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.mkdir()
    assert load(path) == DEFAULTS


def write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_p3_out_of_range_values_are_clamped(tmp_path: Path) -> None:
    prefs = load(write(tmp_path, {"strength": 150, "loudness": -20, "style_strength": 3, "monitor_volume": 250}))
    assert prefs.settings.strength == 100.0 and prefs.settings.loudness == 0.0
    assert prefs.style_strength == 1.0 and prefs.settings.bands == style_bands("natural", 1.0)
    assert prefs.monitor_volume == 100.0
    assert load(write(tmp_path, {"monitor_volume": -5, "style_strength": -1})).monitor_volume == 0.0


def test_p3_old_saved_files_still_load(tmp_path: Path) -> None:
    # Saved before the 2026-11-11 decisions: a custom curve from the old Fine-tune, and Bass/Mid/Treble. They are
    # ignored without an error; the rest is kept.
    old = {
        "preset": "custom",
        "bands": [{"freq": 120.0, "gain": 3.5, "q": 0.7}] * 5,
        "bass": 3,
        "mid": -2,
        "treble": 4,
        "loudness": 60,
        "echo": True,
        "recording": True,
    }
    prefs = load(write(tmp_path, old))
    assert prefs.preset == "natural" and prefs.settings.bands == bands_for("natural")
    assert prefs.settings.loudness == 60.0 and prefs.echo is True and prefs.recording is True
    assert load(write(tmp_path, {"preset": "deep", "bands": "broken"})).settings.bands == bands_for("deep")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("strength", "loud"),
        ("strength", None),
        ("strength", True),
        ("strength", [70]),
        ("loudness", {"v": 1}),
        ("cleaning", "yes"),
        ("cleaning", 0),
        ("echo", 1),
        ("echo", "true"),
        ("recording", None),
        ("mic", 42),
        ("output", ["CABLE Input"]),
        ("tab", "advanced"),
        ("tab", 3),
        ("tab", "tune"),  # J10: the sidebar is gone, a saved page is ignored
        ("preset", "jazz"),
        ("preset", None),
        ("style_strength", "half"),
        ("monitor_volume", True),
        ("user_eq", "flat"),
        ("user_eq", [{"kind": "bell"}] * 7),
        ("user_eq", [{"kind": "wobble", "freq": 100, "gain": 0, "q": 1, "on": True}] * 7),
        ("user_eq", [{"kind": "bell", "freq": 1000, "gain": -4.5, "q": 1, "on": True, "slope": 12}] * 7),  # a good one
        ("denoise", 0),
        ("style", "off"),
        ("fine_tune", 1),
        ("filters", "gain_filter"),
        ("filters", [{"id": "wobble_filter"}]),
    ],
)
def test_p3_wrong_type_values_fall_back_to_defaults(tmp_path: Path, key: str, value: object) -> None:
    prefs = load(write(tmp_path, {key: value}))
    assert prefs == DEFAULTS


def test_p3_one_bad_value_keeps_the_good_ones(tmp_path: Path) -> None:
    data = {"strength": "loud", "loudness": 60, "echo": True, "mic": "Mic X", "tab": "settings"}
    prefs = load(write(tmp_path, data))
    assert prefs.settings.strength == 70.0
    assert prefs.settings.loudness == 60.0
    assert prefs.echo is True
    assert prefs.mic == "Mic X"


def test_p3_non_finite_numbers_fall_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"strength": Infinity, "loudness": NaN, "style_strength": NaN}', encoding="utf-8")
    prefs = load(path)
    assert prefs == DEFAULTS
    assert all(math.isfinite(b.gain) for b in prefs.settings.bands)


def test_t29_no_page_and_no_7_band_eq_are_kept_or_saved(tmp_path: Path) -> None:
    # J10 Step 9: Prefs.tab, prefs.TABS and Settings.user_eq are gone; an old file's "tab" and "user_eq" are ignored
    # and are not written back.
    from clean_voice import prefs as prefs_module

    assert not hasattr(Prefs(), "tab") and not hasattr(prefs_module, "TABS")
    assert not hasattr(Settings(), "user_eq")
    path = write(tmp_path, {"tab": "tune", "user_eq": [{"kind": "bell"}] * 7, "loudness": 61})
    loaded = load(path)
    assert loaded.settings.loudness == 61.0
    save(path, loaded)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert "tab" not in saved and "user_eq" not in saved


def test_p1_the_strength_is_not_lost_when_the_style_changes() -> None:
    prefs = custom_prefs()
    changed = dataclasses.replace(prefs, preset="deep")
    assert changed.style_strength == prefs.style_strength and changed.settings == prefs.settings


# ------------------------------------------------------------------ the theme choice (DESIGN "Theme choice", 2026-11-13)


@pytest.mark.parametrize("choice", ["auto", "light", "dark"])
def test_the_theme_choice_round_trips(tmp_path: Path, choice: str) -> None:
    path = tmp_path / "settings.json"
    save(path, dataclasses.replace(custom_prefs(), theme=choice))
    assert json.loads(path.read_text(encoding="utf-8"))["theme"] == choice
    assert load(path).theme == choice


def test_the_theme_defaults_to_automatic() -> None:
    assert DEFAULTS.theme == "auto"


@pytest.mark.parametrize("value", ["Dark", "system", "", None, 1, True, ["dark"]])
def test_an_old_or_bad_theme_loads_as_automatic(tmp_path: Path, value: object) -> None:
    old = load(write(tmp_path, {"loudness": 61}))  # a file from before the choice existed
    assert old.theme == "auto" and old.settings.loudness == 61.0
    bad = load(write(tmp_path, {"theme": value, "loudness": 61}))
    assert bad.theme == "auto" and bad.settings.loudness == 61.0
