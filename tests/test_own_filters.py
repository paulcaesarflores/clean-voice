"""Clean Voice's own Fine-tune filters, Denoise and Voice Style (J13): each equals its Step 2 stage, changes are
click-free, both save and restore, and a chain of the two in either order runs speech with no dropout."""

import sys
from pathlib import Path

import numpy as np
import pytest
from speech import RATE, looped_noise, read_wav, speech

from clean_voice import dsp, fine_tune, obs_filters, prefs
from clean_voice.dsp import Chain, Equalizer, FineTune, RumbleFilter
from clean_voice.fine_tune import DENOISE_KIND, STYLE_KIND, FilterSpec, new_filter
from clean_voice.prefs import Prefs
from clean_voice.settings import STYLE_STRENGTH, Settings, SettingsCell, replace, style_bands
from clean_voice.styles import PRESETS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import audio_check

BLOCK = 480
SPEECH = Path(__file__).resolve().parent / "data" / "speech.wav"
DROPOUTS = ("discontinuity", "long gap", "zero run")


def blocks(f: obs_filters.Filter | FineTune | Chain | Equalizer, x: np.ndarray, size: int = BLOCK) -> np.ndarray:
    return np.concatenate([f.process(x[i : i + size]) for i in range(0, len(x), size)])


def speech_in(seconds: float = 8.0) -> np.ndarray:
    x = speech()[: int(seconds * RATE)] * 0.1
    fan = looped_noise(len(x), seed=1, name="fan.wav") * 0.02
    return (x + fan).astype(np.float32)


# ------------------------------------------------------------------ the kinds' data


def test_step_2_wording_defaults_and_ranges() -> None:
    denoise, style = fine_tune.KINDS[DENOISE_KIND], fine_tune.KINDS[STYLE_KIND]
    assert (denoise.name, style.name) == ("Denoise", "Voice Style")
    p = denoise.prop("strength")
    assert (p.label, p.default, p.lo, p.hi, p.step, p.is_int) == ("Strength", Settings().strength, 0, 100, 1, True)
    assert [p.key for p in style.props] == ["style", "strength"]
    names = style.prop("style")
    assert names.options == tuple((k, v.name) for k, v in PRESETS.items())  # Step 2's names, order
    assert [label for _, label in names.options] == ["Natural", "Warm", "Crisp", "Radio", "Deep", "Flat"]
    assert names.default == "natural"
    s = style.prop("strength")
    assert (s.label, s.default, s.lo, s.hi) == ("Strength", STYLE_STRENGTH, 0.0, 1.0)
    assert fine_tune.resolve(STYLE_KIND, {"strength": 7, "style": "loud"}) == {"style": "natural", "strength": 1.0}
    assert fine_tune.resolve(DENOISE_KIND, {"strength": 250}) == {"strength": 100}


def test_menu_order_puts_clean_voices_two_first() -> None:
    assert fine_tune.MENU_ORDER[:3] == (DENOISE_KIND, STYLE_KIND, "basic_eq_filter")
    assert len(fine_tune.MENU_ORDER) == 12 and len(set(fine_tune.MENU_ORDER)) == 12


# ------------------------------------------------------------------ Denoise equals Step 2's Denoise stage


@pytest.mark.parametrize("strength", [0, 35, 70, 100])
def test_denoise_filter_equals_the_chains_denoise_stage(strength: int) -> None:
    x = speech_in(6.0)
    chain = Chain(SettingsCell(Settings(strength=float(strength))), block=BLOCK)
    stage = []
    for i in range(0, len(x), BLOCK):
        chain.process(x[i : i + BLOCK])
        # the chain's own stage: RNNoise's wet output and the delayed dry path, mixed by the strength's ramp value
        stage.append(chain.denoise.wet + chain._mix.value * (chain.dry - chain.denoise.wet))
    want = np.concatenate(stage).astype(np.float32)
    rumble = RumbleFilter()  # the stage's input is the chain's rumble-filtered mic
    rumbled = np.concatenate([rumble.process(x[i : i + BLOCK]) for i in range(0, len(x), BLOCK)])
    got = blocks(obs_filters.make(DENOISE_KIND, {"strength": strength}), rumbled)
    assert np.array_equal(got, want)


def test_denoise_filter_reports_the_chains_latency() -> None:
    f = obs_filters.make(DENOISE_KIND, None, block=BLOCK)
    assert isinstance(f, obs_filters.Denoise) and f.latency == Chain(block=BLOCK).latency
    assert obs_filters.make(DENOISE_KIND, None, block=None).latency == Chain(block=None).latency  # type: ignore[attr-defined]


# ------------------------------------------------------------------ Voice Style equals Step 2's style EQ


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("style", list(PRESETS))
def test_style_filter_equals_step_2s_style_eq(style: str, strength: float) -> None:
    x = speech_in(3.0)
    chain = Chain(SettingsCell(Settings(bands=style_bands(style, strength))), block=BLOCK)
    want = blocks(chain.eq, x)  # Step 2's Equalizer, built as the chain builds it
    got = blocks(obs_filters.make(STYLE_KIND, {"style": style, "strength": strength}), x)
    assert np.array_equal(got, want)
    assert np.array_equal(want, blocks(Equalizer(style_bands(style, strength)), x))


# ------------------------------------------------------------------ changes mid-stream are click-free


def retarget_run(kind: str, first: dict, second: dict, x: np.ndarray) -> tuple[np.ndarray, int]:
    f = obs_filters.make(kind, first)
    at = len(x) // 2 // BLOCK * BLOCK
    out = [f.process(x[i : i + BLOCK]) for i in range(0, at, BLOCK)]
    f.update(second)
    out += [f.process(x[i : i + BLOCK]) for i in range(at, len(x) - BLOCK + 1, BLOCK)]
    return np.concatenate(out), at


def test_changing_the_style_or_strength_is_click_free() -> None:
    from test_dsp import energy_above_db

    t = np.arange(6 * RATE) / RATE
    tone = (0.1 * np.sin(2 * np.pi * 317 * t)).astype(np.float32)
    for first, second in (
        ({"style": "natural", "strength": 0.0}, {"style": "radio", "strength": 1.0}),
        ({"style": "deep", "strength": 1.0}, {"style": "crisp", "strength": 0.2}),
    ):
        y, at = retarget_run(STYLE_KIND, first, second, tone)
        assert energy_above_db(y, at) <= -60.0
    x = speech_in(8.0)
    y, _ = retarget_run(STYLE_KIND, {"style": "warm", "strength": 0.0}, {"style": "crisp", "strength": 1.0}, x)
    assert [f for f in audio_check.analyze(y.astype(np.float64)) if f.kind in DROPOUTS] == []


def test_changing_the_denoise_strength_is_click_free() -> None:
    x = speech_in(8.0)
    y, at = retarget_run(DENOISE_KIND, {"strength": 0}, {"strength": 100}, x)
    assert [f for f in audio_check.analyze(y.astype(np.float64)) if f.kind in DROPOUTS] == []
    # the mix ramps over 10 ms: no sample step at the change is bigger than the speech makes on its own elsewhere
    steps = np.abs(np.diff(y.astype(np.float64)))
    assert steps[at - 2 : at + 2 * BLOCK].max() <= steps[: at - 2].max() * 1.5 + 1e-3


def test_a_style_change_reaches_the_sound_as_step_2s_eq_does() -> None:
    x = speech_in(4.0)
    first, second = {"style": "warm", "strength": 1.0}, {"style": "crisp", "strength": 0.6}
    y, at = retarget_run(STYLE_KIND, first, second, x)
    # Step 2's Equalizer, retargeted at the same block with the same bands: sample for sample the same
    eq = Equalizer(style_bands("warm", 1.0))
    want = [eq.process(x[i : i + BLOCK]) for i in range(0, at, BLOCK)]
    eq.set_bands(style_bands("crisp", 0.6))
    want += [eq.process(x[i : i + BLOCK]) for i in range(at, len(x) - BLOCK + 1, BLOCK)]
    assert np.array_equal(y, np.concatenate(want))
    # and the change is audible: not what the unchanged filter would have played
    kept = blocks(obs_filters.make(STYLE_KIND, first), x)[: len(y)]
    assert np.array_equal(y[:at], kept[:at])
    assert not np.allclose(y[at:], kept[at:], atol=1e-3)


def test_a_denoise_change_reaches_the_sound_as_the_chains_ramped_stage_does() -> None:
    x = speech_in(4.0)
    at = len(x) // 2 // BLOCK * BLOCK  # where retarget_run changes a setting
    m0, m1 = dsp.strength_mix(0.0), dsp.strength_mix(100.0)
    assert m0 != m1
    chain = Chain(SettingsCell(Settings(strength=0.0)), block=BLOCK)
    stage = []
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        if i == at:
            chain.cell.value = replace(chain.cell.value, strength=100.0)
        chain.process(x[i : i + BLOCK])
        # the chain's wet and dry paths, mixed by a plain linear 10 ms ramp from the old mix to the new
        k = np.arange(1, BLOCK + 1) + (i - at)
        d = m0 + (m1 - m0) * np.clip(k, 0, dsp.RAMP_SAMPLES) / dsp.RAMP_SAMPLES if i >= at else m0
        stage.append(chain.denoise.wet + d * (chain.dry - chain.denoise.wet))
    want = np.concatenate(stage)
    # the stage's input is the chain's rumble-filtered mic
    rumble = RumbleFilter()
    rumbled = np.concatenate([rumble.process(x[i : i + BLOCK]) for i in range(0, len(x), BLOCK)])
    f = obs_filters.make(DENOISE_KIND, {"strength": 0})
    parts = []
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        if i == at:
            f.update({"strength": 100})
        parts.append(f.process(rumbled[i : i + BLOCK]))
    got = np.concatenate(parts)
    assert np.allclose(got, want, rtol=0, atol=1e-6)
    kept = blocks(obs_filters.make(DENOISE_KIND, {"strength": 0}), rumbled)[: len(got)]
    assert np.array_equal(got[:at], kept[:at])
    assert not np.allclose(got[at:], kept[at:], atol=1e-3)  # the change is audible


# ------------------------------------------------------------------ in Fine-tune: save, restore, order


def test_save_and_restore_round_trip(tmp_path: Path) -> None:
    filters: list[FilterSpec] = []
    for kind, settings in (
        (DENOISE_KIND, {"strength": 40}),
        (STYLE_KIND, {"style": "radio", "strength": 0.8}),
        (DENOISE_KIND, {}),
        (STYLE_KIND, {}),
    ):
        filters.append(new_filter(kind, filters, settings, enabled=len(filters) != 2))
    assert [f.name for f in filters] == ["Denoise", "Voice Style", "Denoise 2", "Voice Style 2"]
    p = Prefs(settings=Settings(fine_tune=True, filters=tuple(filters)))
    path = tmp_path / "settings.json"
    prefs.save(path, p)
    back = prefs.load(path)
    assert back == p
    assert back.settings.filters[1].settings == {"style": "radio", "strength": 0.8}
    assert back.settings.filters[0].settings == {"strength": 40}


def test_the_filter_list_runs_them_in_order_and_toggles_with_the_eye() -> None:
    x = speech_in(4.0)
    a = new_filter(STYLE_KIND, (), {"style": "radio", "strength": 1.0})
    b = new_filter(DENOISE_KIND, (a,), {"strength": 100})
    cell = SettingsCell(Settings(fine_tune=True, filters=(a, b)))
    ft = FineTune(cell.value, BLOCK)
    two = blocks(ft, x)
    assert isinstance(ft.instances[a.uid], obs_filters.VoiceStyle) and isinstance(
        ft.instances[b.uid], obs_filters.Denoise
    )
    cell.value = replace(cell.value, filters=(b, a))  # reordered: the list's top runs first
    ft.retarget(cell.value)
    for _ in range(4):
        ft.process(x[:BLOCK])  # the swap, with its ramps
    swapped = blocks(ft, x)
    assert not np.allclose(two[: len(swapped)], swapped, atol=1e-3)
    cell.value = replace(
        cell.value,
        filters=tuple(
            f if f.uid != b.uid else FilterSpec(f.kind, f.name, False, f.settings, f.uid) for f in cell.value.filters
        ),
    )
    ft.retarget(cell.value)
    for _ in range(4):
        ft.process(x[:BLOCK])
    assert ft._slots[0].eye.value == 0.0  # the eye: Denoise is skipped


# ------------------------------------------------------------------ a chain of the two, either way round


@pytest.mark.parametrize("order", [(STYLE_KIND, DENOISE_KIND), (DENOISE_KIND, STYLE_KIND)])
def test_both_orders_process_speech_with_no_dropout(order: tuple[str, str]) -> None:
    x = speech()[: 20 * RATE].astype(np.float32)
    filters: list[FilterSpec] = []
    for kind in order:
        filters.append(new_filter(kind, filters, {"strength": 100} if kind == DENOISE_KIND else {"style": "warm"}))
    chain = Chain(
        SettingsCell(Settings(denoise=False, style=False, fine_tune=True, filters=tuple(filters))), block=BLOCK
    )
    y = blocks(chain, x)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0.01
    q = np.round(y.astype(np.float64) * (1 << 23)) / (1 << 23)
    faults = [f for f in audio_check.analyze(q, RATE, read_wav(SPEECH)) if f.kind in DROPOUTS]
    assert faults == []


def test_the_harness_takes_them(tmp_path: Path) -> None:
    import harness

    out = tmp_path / "out.wav"
    args = [str(SPEECH), str(out), "--settings", "denoise=0", "style=0", "fine_tune=1", "--filters"]
    assert harness.main([*args, f"{STYLE_KIND},{DENOISE_KIND}"]) == 0
    assert len(read_wav(out)) == len(read_wav(SPEECH))
