"""Settings: the original's presets and tone mapping, clamping, and the lock-free swap into the worker (P1-P5)."""

from __future__ import annotations

import dataclasses
import math
import time

import pytest
from test_engine import CABLE, MIC, wait_for

from clean_voice.engine import State
from clean_voice.settings import (
    BASE,
    PRESETS,
    Band,
    Settings,
    SettingsCell,
    bands_for,
    gain_to_tone,
    tone_to_gain,
    with_tone,
)

# reference/simple-clean-mic-0.1.0/ui/main.js:128-143: the names and gains (the styles carry no description since
# J07 round 4, a user decision).
ORIGINAL = {
    "natural": ("Natural", (0, -2, 0, 2.5, 1.5)),
    "warm": ("Warm", (3, 0.5, -1, 1, -1)),
    "crisp": ("Crisp", (-1.5, -3, 0, 4, 4)),
    "radio": ("Radio", (4, -3, 1, 4, 2.5)),
    "deep": ("Deep", (6, 1.5, -1, 1, 0)),
    "flat": ("Flat", (0, 0, 0, 0, 0)),
}


def test_presets_and_base_are_the_originals() -> None:
    # P1
    assert BASE == ((100.0, 0.7), (300.0, 1.0), (1000.0, 1.0), (3500.0, 0.8), (8000.0, 0.7))
    assert list(PRESETS) == list(ORIGINAL)
    for key, (name, gains) in ORIGINAL.items():
        p = PRESETS[key]
        assert (p.name, p.gains) == (name, tuple(float(g) for g in gains))
        assert bands_for(key) == tuple(Band(f, g, q) for (f, q), g in zip(BASE, gains, strict=True))
    defaults = Settings()
    assert defaults.bands == bands_for("natural")
    assert (defaults.strength, defaults.loudness, defaults.cleaning) == (70.0, 50.0, True)


def test_bass_mid_treble_map_to_bands_0_2_4() -> None:
    # P2 (main.js:192-194, 207-219)
    assert tone_to_gain(50) == 0.0 and tone_to_gain(0) == -12.0 and tone_to_gain(100) == 12.0
    assert tone_to_gain(52) == 0.5 and tone_to_gain(51) == 0.0  # 0.5 dB steps
    for k in range(-24, 25):
        g = k / 2
        assert tone_to_gain(gain_to_tone(g)) == g
    s = with_tone(Settings(bands=bands_for("flat")), "bass", 100)
    s = with_tone(s, "mid", 0)
    s = with_tone(s, "treble", 75)
    assert [b.gain for b in s.bands] == [12.0, 0.0, -12.0, 0.0, 6.0]


def test_values_are_clamped_to_the_dsp_ranges() -> None:
    # P3 (dsp.rs:116-124)
    b = Band(5.0, 30.0, 20.0)
    assert (b.freq, b.gain, b.q) == (20.0, 18.0, 8.0)
    b = Band(50_000.0, -30.0, 0.01)
    assert (b.freq, b.gain, b.q) == (20_000.0, -18.0, 0.2)
    s = Settings(strength=150, loudness=-5)
    assert (s.strength, s.loudness) == (100.0, 0.0)
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError):
            Band(bad, 0.0, 1.0)
        with pytest.raises(ValueError):
            Settings(strength=bad)
    with pytest.raises(ValueError):
        Settings(bands=bands_for("flat")[:4])


def test_settings_reach_the_worker_lock_free(make_engine, monkeypatch: pytest.MonkeyPatch) -> None:
    # P4: the UI thread only swaps a reference; the worker's chain picks it up at its next block.
    engine, _ = make_engine([MIC, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    path = engine._path
    assert path is not None
    chain = path.chain
    new = Settings(strength=20, loudness=80, bands=bands_for("radio"))

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("set_settings must not queue or lock")

    with monkeypatch.context() as m:
        m.setattr(engine._commands, "put", refuse)
        m.setattr(engine, "_lock", type("NoLock", (), {"acquire": refuse, "__enter__": refuse}))
        engine.set_settings(new)
        assert engine.settings() is new
    blocks = path.blocks_processed
    wait_for(lambda: path.blocks_processed >= blocks + 2)
    assert chain.applied is new  # type: ignore[attr-defined]


def test_settings_survive_a_reconnect(make_engine) -> None:
    # P5
    engine, backend = make_engine([MIC, CABLE])
    engine.start(MIC.name)
    wait_for(lambda: engine.status().state is State.ON)
    new = Settings(strength=0, bands=bands_for("deep"))
    engine.set_settings(new)
    first = engine._path
    backend.present.remove(MIC)
    wait_for(lambda: engine.status().state is State.MIC_LOST)
    backend.present.append(MIC)
    wait_for(lambda: engine.status().state is State.ON and engine._path is not first)
    path = engine._path
    assert path is not None
    assert path.chain.cell is engine._cell  # type: ignore[attr-defined]
    time.sleep(0.1)
    assert path.chain.applied is new  # type: ignore[attr-defined]


def test_cell_holds_one_reference() -> None:
    cell = SettingsCell()
    assert cell.value == Settings()
    s = Settings(loudness=10)
    cell.value = s
    assert cell.value is s


# ------------------------------------------------------------------ the style's Strength (user decision, 2026-11-11)


def test_style_strength_scales_the_style_gains() -> None:
    from clean_voice.settings import STYLE_STRENGTH, style_bands

    assert STYLE_STRENGTH == 0.5
    for key in PRESETS:
        assert style_bands(key, 0.5) == bands_for(key)  # 0.5 is the style as designed
        assert all(b.gain == 0.0 for b in style_bands(key, 0.0))  # 0 bypasses the style's EQ
        doubled = style_bands(key, 1.0)
        assert [b.gain for b in doubled] == [min(18.0, max(-18.0, 2 * g)) for g in PRESETS[key].gains]
        assert [(b.freq, b.q) for b in doubled] == [(b.freq, b.q) for b in bands_for(key)]
    assert style_bands("radio", 0.25) == tuple(dataclasses.replace(b, gain=b.gain / 2) for b in bands_for("radio"))


def test_style_strength_clamps_to_the_eq_range(monkeypatch) -> None:
    from clean_voice import settings
    from clean_voice.settings import Preset, style_bands

    monkeypatch.setitem(settings.PRESETS, "loud", Preset("Loud", (12.0, -14.0, 0.0, 0.0, 0.0)))
    assert [b.gain for b in style_bands("loud", 1.0)][:2] == [18.0, -18.0]
    assert style_bands("natural", 7.0) == style_bands("natural", 1.0)  # strength itself is 0..1


def test_settings_no_longer_carry_the_7_band_eq() -> None:
    # J10 Step 9: the 7-band Fine-tune EQ is deleted; Fine-tune is the filter list.
    assert not hasattr(Settings(), "user_eq")
    with pytest.raises(TypeError):
        Settings(user_eq=())  # type: ignore[call-arg]
    assert Settings().filters == () and Settings().fine_tune is False
