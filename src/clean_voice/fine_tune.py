"""Fine-tune's filters as data: OBS Studio's ten audio filters, their properties and defaults (J10 plan §4).

``KINDS`` is the one table of each filter kind's display name, properties (key, label, widget, range, step, suffix,
default) and the OBS file it follows. The DSP (``obs_filters``), the saved choices, the filter window and the parity
tests all read it, so every number lives here once. Keys, ids, labels and numbers are OBS's: the ids and properties
come from each filter's ``obs_source_info``, ``*_defaults`` and ``*_properties``
(``reference/obs-studio/plugins/obs-filters/``, ``plugins/obs-vst/obs-vst.cpp``), the names and labels from
``plugins/obs-filters/data/locale/en-US.ini`` and ``plugins/obs-vst/data/locale/en-US.ini``.

``resolve`` turns any saved or edited settings into a full, valid set: a missing or wrong-type value gives OBS's
default, a number out of range is clamped to OBS's range, an int property is rounded (OBS stores it with
``obs_data_set_int``), and an unknown key is dropped.

A ``FilterSpec`` is one filter in the guest's list, immutable like ``Settings``, which holds the list. Its saved form
is OBS's four fields for a filter, ``{"id", "name", "enabled", "settings"}``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

Value = float | int | str | bool

SLIDER = "slider"  # obs_properties_add_float_slider / add_int_slider: a slider and a number box
INT = "int"  # obs_properties_add_int: a number box only
LIST = "list"  # obs_properties_add_list
CHECKBOX = "checkbox"  # obs_properties_add_bool
BUTTON = "button"  # obs_properties_add_button2: not a setting
HIDDEN = "hidden"  # saved with the filter, never shown (obs-vst's chunk)


@dataclass(frozen=True)
class Prop:
    """One OBS property. ``is_int``: an int property (``obs_data_get_int``); otherwise a float, string or bool."""

    key: str
    label: str
    widget: str
    default: Value = 0.0
    lo: float = 0.0
    hi: float = 0.0
    step: float = 0.0
    suffix: str = ""
    is_int: bool = False
    options: tuple[tuple[str, str], ...] = ()  # a list's (value, label) pairs, in OBS's order

    @property
    def is_setting(self) -> bool:
        return self.widget != BUTTON


@dataclass(frozen=True)
class Kind:
    id: str  # OBS's obs_source_info id
    name: str  # OBS's display name (get_name)
    props: tuple[Prop, ...]
    cite: str  # the OBS file and lines it follows
    note: str = ""  # a line the properties pane shows instead of controls
    # Defaults that depend on a list's value: {list key: {value: {key: default}}} (expander_defaults :153-166).
    preset_defaults: Mapping[str, Mapping[str, Mapping[str, Value]]] = field(default_factory=dict)

    def prop(self, key: str) -> Prop:
        for p in self.props:
            if p.key == key:
                return p
        raise KeyError(key)


def _db(key: str, label: str, lo: float, hi: float, step: float, default: float) -> Prop:
    return Prop(key, label, SLIDER, default, lo, hi, step, " dB")


def _ms(key: str, label: str, lo: int, hi: int, default: int, widget: str = SLIDER) -> Prop:
    return Prop(key, label, widget, default, lo, hi, 1, " ms", is_int=True)


def _ratio(lo: float, hi: float, step: float, default: float) -> Prop:
    return Prop("ratio", "Ratio", SLIDER, default, lo, hi, step, ":1")


KINDS: dict[str, Kind] = {
    k.id: k
    for k in (
        Kind(
            "basic_eq_filter",
            "3-Band Equalizer",
            (  # shown High, Mid, Low (eq_properties :68-70)
                _db("high", "High", -20.0, 20.0, 0.1, 0.0),
                _db("mid", "Mid", -20.0, 20.0, 0.1, 0.0),
                _db("low", "Low", -20.0, 20.0, 0.1, 0.0),
            ),
            "plugins/obs-filters/eq-filter.c: eq_defaults :52-57, eq_properties :59-75",
        ),
        Kind(
            "compressor_filter",
            "Compressor",
            (  # the Sidechain/Ducking Source list is left out (J10 plan, question 7)
                _ratio(1.0, 32.0, 0.5, 10.0),
                _db("threshold", "Threshold", -60.0, 0.0, 0.1, -18.0),
                _ms("attack_time", "Attack", 1, 500, 6),
                _ms("release_time", "Release", 1, 1000, 60),
                _db("output_gain", "Output Gain", -32.0, 32.0, 0.1, 0.0),
            ),
            "plugins/obs-filters/compressor-filter.c: :44-53, compressor_defaults :438-446, "
            "compressor_properties :468-500",
        ),
        Kind(
            "expander_filter",
            "Expander",
            (
                Prop("presets", "Presets", LIST, "expander", options=(("expander", "Expander"), ("gate", "Gate"))),
                _ratio(1.0, 20.0, 0.1, 2.0),
                _db("threshold", "Threshold", -60.0, 0.0, 0.1, -40.0),
                _ms("attack_time", "Attack", 1, 100, 10),
                _ms("release_time", "Release", 1, 1000, 50),
                _db("output_gain", "Output Gain", -32.0, 32.0, 0.1, 0.0),
                Prop("detector", "Detection", LIST, "RMS", options=(("RMS", "RMS"), ("peak", "Peak"))),
            ),
            "plugins/obs-filters/expander-filter.c: :52-63, expander_defaults :153-166, expander_properties :432-468",
            preset_defaults={"presets": {"gate": {"ratio": 10.0, "release_time": 125}}},
        ),
        Kind(
            "gain_filter",
            "Gain",
            (_db("db", "Gain", -30.0, 30.0, 0.1, 0.0),),
            "plugins/obs-filters/gain-filter.c: gain_defaults :68-71, gain_properties :73-82",
        ),
        Kind(
            "invert_polarity_filter",
            "Invert Polarity",
            (),
            "plugins/obs-filters/invert-audio-polarity.c :20-38",
        ),
        Kind(
            "limiter_filter",
            "Limiter",
            (
                _db("threshold", "Threshold", -60.0, 0.0, 0.1, -6.0),
                _ms("release_time", "Release", 1, 1000, 60),
            ),
            "plugins/obs-filters/limiter-filter.c: :34-39, limiter_defaults :182-186, limiter_properties :188-201",
        ),
        Kind(
            "noise_gate_filter",
            "Noise Gate",
            (  # listed Close then Open; the times are number boxes (noise_gate_properties :170-179)
                Prop("close_threshold", "Close Threshold", SLIDER, -32.0, -96.0, 0.0, 1.0, " dB"),
                Prop("open_threshold", "Open Threshold", SLIDER, -26.0, -96.0, 0.0, 1.0, " dB"),
                _ms("attack_time", "Attack Time", 0, 10000, 25, INT),
                _ms("hold_time", "Hold Time", 0, 10000, 200, INT),
                _ms("release_time", "Release Time", 0, 10000, 150, INT),
            ),
            "plugins/obs-filters/noise-gate-filter.c: :47-48, noise_gate_defaults :156-163, "
            "noise_gate_properties :165-183",
        ),
        Kind(
            "noise_suppress_filter",
            "Noise Suppression",
            (),  # RNNoise only: an OBS build without Speex shows no properties (noise_suppress_properties :503-526)
            "plugins/obs-filters/noise-suppress-filter.c: noise_suppress_defaults_v2 :493-501, "
            "noise_suppress_properties :503-526",
            note="RNNoise (good quality, more CPU usage)",  # NoiseSuppress.Method.RNNoise
        ),
        Kind(
            "upward_compressor_filter",
            "Upward Compressor",
            (  # the detector is RMS, with no list (expander_properties :458-466)
                _ratio(0.0, 1.0, 0.1, 0.5),
                _db("threshold", "Threshold", -60.0, 0.0, 0.1, -20.0),
                _ms("attack_time", "Attack", 1, 100, 10),
                _ms("release_time", "Release", 1, 1000, 50),
                _db("output_gain", "Output Gain", -32.0, 32.0, 0.1, 0.0),
                # No suffix: :465 sets a float suffix on this int slider, which libobs ignores
                # (libobs/obs-properties.c obs_property_float_set_suffix :1040-1048), so OBS shows a bare number.
                Prop("knee_width", "Knee Width", SLIDER, 10, 0, 20, 1, "", is_int=True),
            ),
            "plugins/obs-filters/expander-filter.c: :54-55, upward_compressor_defaults :168-177, "
            "expander_properties :445-466",
        ),
        Kind(
            "vst_filter",
            "VST 2.x Plug-in",
            (  # vst_properties :296-330; the keys are obs-vst's own (vst_update :98-130, vst_save :140-145)
                Prop("plugin_path", "VST 2.x Plug-in", LIST, ""),
                Prop("open_vst_settings", "Open Plug-in Interface", BUTTON),
                Prop("close_vst_settings", "Close Plug-in Interface", BUTTON),
                Prop("open_when_active_vst_settings", "Open interface when active", CHECKBOX, False),
                Prop("chunk_data", "", HIDDEN, ""),
                Prop("chunk_hash", "", HIDDEN, ""),
            ),
            "plugins/obs-vst/obs-vst.cpp: vst_update :98-130, vst_save :140-145, vst_properties :296-330",
        ),
    )
}

# The + menu: sorted by display name, as OBSBasicFilters::CreateAddFilterPopupMenu does (:458-516).
MENU_ORDER: tuple[str, ...] = tuple(sorted(KINDS, key=lambda k: KINDS[k].name))


def defaults(kind: str, settings: Mapping[str, object] | None = None) -> dict[str, Value]:
    """Every setting's default for ``kind``. A preset in ``settings`` picks its own defaults, as expander_defaults
    reads the preset before it sets them (expander-filter.c:155-163)."""
    k = KINDS[kind]
    out: dict[str, Value] = {p.key: p.default for p in k.props if p.is_setting}
    for list_key, by_value in k.preset_defaults.items():
        chosen = (settings or {}).get(list_key)
        if isinstance(chosen, str) and chosen in by_value:
            out[list_key] = chosen
            out.update(by_value[chosen])
    return out


def _valid(p: Prop, v: object) -> Value | None:
    """``v`` as a valid value for ``p``, clamped and rounded, or None when it can't be one."""
    if p.widget in (SLIDER, INT):
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            return None
        x = min(p.hi, max(p.lo, float(v)))
        return round(x) if p.is_int else x
    if p.widget == CHECKBOX:
        return v if isinstance(v, bool) else None
    if not isinstance(v, str):
        return None
    if p.options and v not in (value for value, _ in p.options):
        return None
    return v


def resolve(kind: str, settings: Mapping[str, object] | None = None) -> dict[str, Value]:
    """A full, valid settings dict for ``kind`` from any ``settings``: OBS's defaults for what is missing or wrong,
    numbers clamped to OBS's ranges, unknown keys dropped."""
    given = settings or {}
    out = defaults(kind, given)
    for p in KINDS[kind].props:
        if p.is_setting and p.key in given:
            v = _valid(p, given[p.key])
            if v is not None:
                out[p.key] = v
    return out


# ------------------------------------------------------------------ the guest's list

_uids = itertools.count(1)  # next() on a count is atomic in CPython: any thread may make a FilterSpec


def new_uid() -> int:
    """A uid no filter has had. A VST filter given a different plug-in gets a new one, so that the chain treats it as
    a new filter and one uid always means one loaded plug-in (``vst.VstRegistry``)."""
    return next(_uids)


@dataclass(frozen=True)
class FilterSpec:
    """One filter in Fine-tune's list. ``settings`` is resolved (``resolve``) and read-only. ``uid`` is internal: it
    is assigned at creation or load, never saved and never compared; the chain keeps a filter's running instance for as
    long as its uid stays in the list."""

    kind: str  # an OBS filter id, a KINDS key
    name: str  # unique in the list, as OBS requires (OBSBasicFilters.cpp AddNewFilter :518-548)
    enabled: bool = True  # the eye (obs_source_set_enabled)
    settings: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))  # resolved: Values only
    uid: int = field(default_factory=new_uid, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown filter kind {self.kind!r}")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a filter needs a name")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "settings", MappingProxyType(resolve(self.kind, self.settings)))


def unique_name(base: str, taken: Iterable[str]) -> str:
    """``base``, or "base 2", "base 3"... : the first one not taken (OBSBasicFilters.cpp AddNewFilter :522-529)."""
    names = set(taken)
    text, i = base, 2
    while text in names:
        text = f"{base} {i}"
        i += 1
    return text


def new_filter(
    kind: str, existing: Iterable[FilterSpec], settings: Mapping[str, object] | None = None, enabled: bool = True
) -> FilterSpec:
    """A new filter of ``kind`` named by OBS's rule against the ``existing`` list, with OBS's defaults."""
    return FilterSpec(kind, unique_name(KINDS[kind].name, (f.name for f in existing)), enabled, settings or {})


def filter_to_json(spec: FilterSpec) -> dict[str, Any]:
    return {"id": spec.kind, "name": spec.name, "enabled": spec.enabled, "settings": dict(spec.settings)}


def filters_to_json(filters: Iterable[FilterSpec]) -> list[dict[str, Any]]:
    return [filter_to_json(f) for f in filters]


def filters_from_json(value: object) -> tuple[FilterSpec, ...]:
    """The list from settings.json. An entry that isn't an object or has an unknown id is dropped; a missing or bad
    name gives the kind's name; a duplicate name gets OBS's " 2" rule; a bad ``enabled`` is on; bad settings give
    OBS's defaults, and numbers out of range are clamped. Anything but a list is no filters."""
    if not isinstance(value, list):
        return ()
    out: list[FilterSpec] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        kind = entry.get("id")
        if not isinstance(kind, str) or kind not in KINDS:  # a list or object id is unhashable: check the type first
            continue
        name = entry.get("name")
        base = name if isinstance(name, str) and name else KINDS[kind].name
        enabled = entry.get("enabled")
        settings = entry.get("settings")
        out.append(
            FilterSpec(
                kind,
                unique_name(base, (f.name for f in out)),
                enabled if isinstance(enabled, bool) else True,
                settings if isinstance(settings, dict) else {},
            )
        )
    return tuple(out)
