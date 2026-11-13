"""The guest's sound settings: voice style (5-band EQ), noise-removal strength, voice volume, the on/off bypass, and
the Step 2 switches (Denoise, Voice style, Fine-tune) with Fine-tune's filter list (J10).

The values match the original app exactly (reference/simple-clean-mic-0.1.0/ui/main.js:126-219 and
src-tauri/src/dsp.rs:106-133). A ``Settings`` is immutable; the UI hands a new one to the engine, which stores it in a
``SettingsCell``. That is one reference assignment, atomic in CPython, with no lock and no queue. The worker reads the
cell once per block and retargets the chain when the reference changed, so a block never sees half a change.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

from clean_voice.fine_tune import FilterSpec

FREQ_RANGE = (20.0, 20_000.0)  # dsp.rs:117
GAIN_RANGE = (-18.0, 18.0)  # dsp.rs:118
Q_RANGE = (0.2, 8.0)  # dsp.rs:122
TONE_RANGE_DB = 12.0  # the Bass/Mid/Treble bars span -12..+12 dB (main.js:192-194)
TONE_BANDS = {"bass": 0, "mid": 2, "treble": 4}  # main.js:207


def _clamp(value: float, lo: float, hi: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, not {value!r}")
    return min(hi, max(lo, v))


@dataclass(frozen=True)
class Band:
    """One EQ band. Band 0 is a low shelf, band 4 a high shelf, the others peaking (bell) filters."""

    freq: float
    gain: float
    q: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "freq", _clamp(self.freq, *FREQ_RANGE, "freq"))
        object.__setattr__(self, "gain", _clamp(self.gain, *GAIN_RANGE, "gain"))
        object.__setattr__(self, "q", _clamp(self.q, *Q_RANGE, "q"))


@dataclass(frozen=True)
class Preset:
    name: str
    gains: tuple[float, float, float, float, float]


# main.js:128-134, equal to dsp.rs:127-133 DEFAULT_EQ's frequencies and Q.
BASE: tuple[tuple[float, float], ...] = ((100.0, 0.7), (300.0, 1.0), (1000.0, 1.0), (3500.0, 0.8), (8000.0, 0.7))

# main.js:136-143: the names and gains. The styles carry no description (user decision, J07 round 4).
PRESETS: dict[str, Preset] = {
    "natural": Preset("Natural", (0.0, -2.0, 0.0, 2.5, 1.5)),
    "warm": Preset("Warm", (3.0, 0.5, -1.0, 1.0, -1.0)),
    "crisp": Preset("Crisp", (-1.5, -3.0, 0.0, 4.0, 4.0)),
    "radio": Preset("Radio", (4.0, -3.0, 1.0, 4.0, 2.5)),
    "deep": Preset("Deep", (6.0, 1.5, -1.0, 1.0, 0.0)),
    "flat": Preset("Flat", (0.0, 0.0, 0.0, 0.0, 0.0)),
}


def bands_for(preset: str) -> tuple[Band, ...]:
    """The five bands of a preset: BASE frequencies and Q with the preset's gains (main.js:148-150)."""
    gains = PRESETS[preset].gains
    return tuple(Band(f, g, q) for (f, q), g in zip(BASE, gains, strict=True))


STYLE_STRENGTH = 0.5  # the Presets page's Strength: 0.5 is the style as designed (user decision, 2026-11-11)


def style_bands(preset: str, strength: float) -> tuple[Band, ...]:
    """The style's five bands at ``strength`` (0..1): each gain times 2 x strength, clamped to the EQ range by
    ``Band``. 0 bypasses the style's EQ, 0.5 is the style itself, 1 doubles it."""
    k = 2.0 * _clamp(strength, 0.0, 1.0, "strength")
    return tuple(Band(f, g * k, q) for (f, q), g in zip(BASE, PRESETS[preset].gains, strict=True))


def tone_to_gain(value: float) -> float:
    """A Bass/Mid/Treble bar value (0..100, 50 = no change) to its band gain in 0.5 dB steps (main.js:194)."""
    return round((value - 50.0) / 50.0 * TONE_RANGE_DB * 2.0) / 2.0


def gain_to_tone(gain: float) -> float:
    """The inverse: a band gain to its bar value (main.js:193)."""
    return 50.0 + gain / TONE_RANGE_DB * 50.0


@dataclass(frozen=True)
class Settings:
    strength: float = 70.0  # noise removal, 0..100 (main.js:407)
    loudness: float = 50.0  # voice volume, 0..100, 50 = 0 dB (main.js:200)
    bands: tuple[Band, ...] = field(default_factory=lambda: bands_for("natural"))
    cleaning: bool = True  # False: the raw mic goes to the cable, as the original's Off did (main.js:622)
    denoise: bool = True  # Step 2's Denoise switch: off is the delayed dry path (J10)
    style: bool = True  # Voice style: off is a flat EQ
    fine_tune: bool = False  # Fine-tune (advanced): off bypasses the whole filter list and keeps it
    filters: tuple[FilterSpec, ...] = ()  # Fine-tune's list, top first: the top filter runs first

    def __post_init__(self) -> None:
        object.__setattr__(self, "strength", _clamp(self.strength, 0.0, 100.0, "strength"))
        object.__setattr__(self, "loudness", _clamp(self.loudness, 0.0, 100.0, "loudness"))
        bands = tuple(self.bands)
        if len(bands) != len(BASE) or not all(isinstance(b, Band) for b in bands):
            raise ValueError(f"bands must be {len(BASE)} Band values")
        object.__setattr__(self, "bands", bands)
        object.__setattr__(self, "cleaning", bool(self.cleaning))
        for name in ("denoise", "style", "fine_tune"):
            object.__setattr__(self, name, bool(getattr(self, name)))
        filters = tuple(self.filters)
        if not all(isinstance(f, FilterSpec) for f in filters):
            raise ValueError("filters must be FilterSpec values")
        object.__setattr__(self, "filters", filters)


def replace(settings: Settings, **changes: object) -> Settings:
    return dataclasses.replace(settings, **changes)  # type: ignore[arg-type]


def with_tone(settings: Settings, bar: str, value: float) -> Settings:
    """Settings with the Bass, Mid or Treble bar moved: the gain of band 0, 2 or 4."""
    i = TONE_BANDS[bar]
    bands = list(settings.bands)
    bands[i] = dataclasses.replace(bands[i], gain=tone_to_gain(value))
    return replace(settings, bands=tuple(bands))


class SettingsCell:
    """One reference to the current ``Settings``: the UI assigns ``value``, the worker reads it once per block."""

    def __init__(self, value: Settings | None = None) -> None:
        self.value: Settings = value if value is not None else Settings()
