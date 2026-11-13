"""Fine-tune in the clean-up chain (J10 plan §1 and §3, Tests T13-T16): the Denoise, Voice style and Fine-tune
switches, the filter list's order, click-free changes, and the applied-settings handshake."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
from speech import RATE, mix, read_wav

from clean_voice import dsp, fine_tune, obs_filters
from clean_voice.dsp import (
    Chain,
    Compressor,
    Delay,
    Equalizer,
    FineTune,
    Limiter,
    NoiseSuppressor,
    Ramp,
    RumbleFilter,
    VoiceVolume,
    db_to_lin,
    strength_mix,
)
from clean_voice.fine_tune import FilterSpec, new_filter
from clean_voice.settings import Settings, SettingsCell, bands_for, replace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import audio_check
import harness

BLOCK = 480
CEILING = db_to_lin(-1.0)
FOUR = "noise_gate_filter,basic_eq_filter,compressor_filter,gain_filter"  # the done-when's test chain


def tone(freq: float, seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def run_chain(chain: Chain, x: np.ndarray, size: int = BLOCK) -> np.ndarray:
    return np.concatenate([chain.process(x[i : i + size]) for i in range(0, len(x), size)])


def specs(*kinds: str) -> tuple[FilterSpec, ...]:
    """A list of ``kinds`` at OBS's defaults, named by OBS's rule."""
    out: list[FilterSpec] = []
    for k in kinds:
        out.append(new_filter(k, out))
    return tuple(out)


def old_chain(x: np.ndarray, s: Settings, mix_gain: float | None = None, block: int = BLOCK) -> np.ndarray:
    """The chain as it was before J10 with the Fine-tune EQ flat (it skipped off bands): rumble, RNNoise and the
    strength mix, the style EQ, the compressor, Voice volume and the limiter. ``mix_gain``: a fixed mix instead of the
    strength's (1.0 is all dry)."""
    rumble, ns = RumbleFilter(), NoiseSuppressor(block)
    lat = ns.latency + Chain.RNNOISE_LATENCY
    dry, mix = Delay(lat), Ramp(strength_mix(s.strength) if mix_gain is None else mix_gain)
    eq, comp, vol, lim = Equalizer(s.bands), Compressor(), VoiceVolume(s.loudness), Limiter()
    out = []
    for i in range(0, len(x), block):
        r = rumble.process(x[i : i + block])
        d = dry.process(r)
        w = ns.process(r)
        y = (w + mix.gains(len(r)) * (d - w)).astype(np.float32)
        out.append(lim.process(vol.process(comp.process(eq.process(y)))))
    return np.concatenate(out)


# ------------------------------------------------------------------ T14 only what is switched changes the sound


def speech_in(seconds: float = 6.0, seed: int = 3) -> np.ndarray:
    x, _, _ = mix(-26.0, 15.0, seed=seed)
    return x[: int(seconds * RATE)]


def test_t14_all_on_with_no_filters_is_the_chain_before_j10() -> None:
    x = speech_in()
    s = Settings(bands=bands_for("warm"), strength=60.0, loudness=55.0)
    assert (s.denoise, s.style, s.fine_tune) == (True, True, False)
    for on in (s, replace(s, fine_tune=True)):  # Fine-tune on with an empty list changes nothing
        assert np.array_equal(run_chain(Chain(SettingsCell(on), block=BLOCK), x), old_chain(x, s))


def test_t14_denoise_off_is_the_delayed_dry_path() -> None:
    x = speech_in()
    s = Settings(bands=bands_for("radio"), denoise=False)
    assert np.array_equal(run_chain(Chain(SettingsCell(s), block=BLOCK), x), old_chain(x, s, mix_gain=1.0))


def test_t14_voice_style_off_is_flat() -> None:
    x = speech_in()
    off = run_chain(Chain(SettingsCell(Settings(bands=bands_for("deep"), style=False)), block=BLOCK), x)
    flat = run_chain(Chain(SettingsCell(Settings(bands=bands_for("flat"))), block=BLOCK), x)
    assert np.array_equal(off, flat)


def test_t14_filters_only_change_the_sound_while_fine_tune_is_on() -> None:
    x = speech_in(3.0)
    filters = (FilterSpec("gain_filter", "Gain", settings={"db": -12.0}),)
    plain = run_chain(Chain(SettingsCell(Settings()), block=BLOCK), x)
    off = run_chain(Chain(SettingsCell(Settings(filters=filters)), block=BLOCK), x)
    on = run_chain(Chain(SettingsCell(Settings(filters=filters, fine_tune=True)), block=BLOCK), x)
    assert np.array_equal(off, plain)
    assert not np.allclose(on, plain, atol=1e-3)


@pytest.mark.parametrize("size", [480, 333])
def test_t14_the_ceiling_holds_with_a_30_db_gain_filter(size: int) -> None:
    from test_dsp import clip_inputs

    hot = specs("gain_filter", "basic_eq_filter")
    hot = (
        dataclasses.replace(hot[0], settings={"db": 30.0}),
        dataclasses.replace(hot[1], settings={"low": 20.0, "mid": 20.0, "high": 20.0}),
    )
    worst = 0.0
    for name, x in clip_inputs().items():
        for s in (
            Settings(fine_tune=True, filters=hot, loudness=100),
            Settings(fine_tune=True, filters=hot, denoise=False, style=False),
        ):
            out = run_chain(Chain(SettingsCell(s)), x, size)
            assert np.all(np.isfinite(out)), name
            worst = max(worst, float(np.max(np.abs(out))))
            assert worst <= CEILING, (name, 20 * np.log10(worst))
    assert worst > db_to_lin(-1.5)


# ------------------------------------------------------------------ T15 order


def test_t15_the_list_runs_from_the_top() -> None:
    gain = FilterSpec("gain_filter", "Gain", settings={"db": -6.0})
    lim = FilterSpec("limiter_filter", "Limiter", settings={"threshold": -6.0})
    x = tone(300, 1.0, 1.0)

    def peak_db(filters: tuple[FilterSpec, ...]) -> float:
        ft = FineTune(Settings(fine_tune=True, filters=filters), block=BLOCK)
        y = np.concatenate([ft.process(x[i : i + BLOCK]) for i in range(0, len(x), BLOCK)])
        return 20 * np.log10(float(np.max(np.abs(y[RATE // 2 :]))))

    assert peak_db((gain, lim)) == pytest.approx(-6.0, abs=0.05)  # -6 dB in, under the limiter's -6 dB
    assert peak_db((lim, gain)) == pytest.approx(-12.0, abs=0.05)  # limited to -6, then 6 dB down
    whole = [
        run_chain(Chain(SettingsCell(Settings(fine_tune=True, filters=f)), block=BLOCK), x)
        for f in ((gain, lim), (lim, gain))
    ]
    assert not np.allclose(whole[0], whole[1], atol=1e-3)


# ------------------------------------------------------------------ the section: retargeting and the handshake


def test_a_disabled_filter_or_section_is_not_called(monkeypatch: pytest.MonkeyPatch) -> None:
    # OBS skips a disabled filter (libobs/obs-source.c:3914): once its eye is fully off, it is not run.
    calls = {"n": 0}
    from clean_voice import obs_filters

    real = obs_filters.Gain.process

    def counted(self, x):
        calls["n"] += 1
        return real(self, x)

    monkeypatch.setattr(obs_filters.Gain, "process", counted)
    g = FilterSpec("gain_filter", "Gain", settings={"db": -6.0})
    cell = SettingsCell(Settings(fine_tune=True, filters=(g,)))
    chain = Chain(cell, block=BLOCK)
    x = tone(300, 0.5, 0.1)
    run_chain(chain, x[:4800])
    assert calls["n"] == 10
    cell.value = replace(cell.value, filters=(dataclasses.replace(g, enabled=False),))
    run_chain(chain, x[:4800])
    assert calls["n"] == 11  # one block of the 10 ms fade, then skipped
    cell.value = replace(cell.value, filters=(g,), fine_tune=False)
    run_chain(chain, x[:4800])
    assert calls["n"] == 12  # the section fades out over one block, then nothing in it runs


def test_instances_keep_their_state_across_edits_and_reorders() -> None:
    comp = FilterSpec("compressor_filter", "Compressor")
    gain = FilterSpec("gain_filter", "Gain")
    cell = SettingsCell(Settings(fine_tune=True, filters=(comp, gain)))
    chain = Chain(cell, block=BLOCK)
    first = {uid: inst for uid, inst in chain.fine_tune.instances.items()}
    run_chain(chain, speech_in(1.0))
    cell.value = replace(cell.value, filters=(gain, dataclasses.replace(comp, settings={"ratio": 4.0})))
    run_chain(chain, speech_in(1.0))
    assert chain.fine_tune.instances == first  # the same objects: reordered, updated, never rebuilt
    kept = first[comp.uid]
    assert isinstance(kept, obs_filters.Compressor) and kept.settings["ratio"] == 4.0
    assert [s.uid for s in chain.fine_tune.applied.filters] == [gain.uid, comp.uid]


def test_the_applied_settings_drop_a_removed_filter_only_after_its_fade() -> None:
    a = FilterSpec("gain_filter", "Gain", settings={"db": -3.0})
    b = FilterSpec("limiter_filter", "Limiter")
    cell = SettingsCell(Settings(fine_tune=True, filters=(a, b)))
    chain = Chain(cell, block=BLOCK)
    x = tone(300, 0.2, 0.1)
    chain.process(x[:BLOCK])
    removed = replace(cell.value, filters=(a,))
    cell.value = removed
    chain.process(x[:BLOCK])  # the section fades to its input over this block: b still runs
    assert b.uid in {f.uid for f in chain.fine_tune.applied.filters}
    assert b.uid in chain.fine_tune.instances
    chain.process(x[:BLOCK])  # swapped at the block boundary, fading back in
    assert chain.fine_tune.applied is removed and b.uid not in chain.fine_tune.instances
    # An in-place change (a setting, the eye, a name) is applied at once.
    renamed = replace(removed, filters=(dataclasses.replace(a, name="Quieter", settings={"db": -9.0}),))
    cell.value = renamed
    chain.process(x[:BLOCK])
    assert chain.fine_tune.applied is renamed


# ------------------------------------------------------------------ T13 every switch is click-free


def t13_run(x: np.ndarray) -> tuple[np.ndarray, list[int], int]:
    """``x`` through the chain while, every 200 ms, the next change in turn is made: Denoise, Voice style and
    Fine-tune flip, the top filter's eye toggles, a filter is added, the top two swap, and the added one goes.
    Returns the output, the sample of each change and the chain's latency."""
    base = specs("gain_filter", "basic_eq_filter", "compressor_filter", "limiter_filter")
    base = (
        dataclasses.replace(base[0], settings={"db": 3.0}),
        dataclasses.replace(base[1], settings={"low": 4.0, "high": -3.0}),
        base[2],
        base[3],
    )
    extra = FilterSpec("gain_filter", "Gain 2", settings={"db": -4.0})
    cell = SettingsCell(Settings(fine_tune=True, filters=base))
    chain = Chain(cell, block=BLOCK)

    def flip(field: str):
        return lambda s: replace(s, **{field: not getattr(s, field)})

    def eye(s: Settings) -> Settings:
        f = list(s.filters)
        f[0] = dataclasses.replace(f[0], enabled=not f[0].enabled)
        return replace(s, filters=tuple(f))

    actions = [
        flip("denoise"),
        flip("style"),
        flip("fine_tune"),
        eye,
        lambda s: replace(s, filters=(*s.filters, extra)),  # add
        lambda s: replace(s, filters=(s.filters[1], s.filters[0], *s.filters[2:])),  # reorder
        lambda s: replace(s, filters=tuple(f for f in s.filters if f.uid != extra.uid)),  # remove
    ]
    out, changes = [], []
    for k, i in enumerate(range(0, len(x), BLOCK)):
        if k and k % 20 == 0:  # every 200 ms
            cell.value = actions[(k // 20) % len(actions)](cell.value)
            changes.append(i)
        out.append(chain.process(x[i : i + BLOCK]))
    return np.concatenate(out).astype(np.float64), changes, chain.latency


def test_t13_switches_are_click_free() -> None:
    from test_dsp import energy_above_db

    # On a steady tone first: audio_check's discontinuity rule is for dropouts and cannot see a hard gain switch on
    # speech, so each change is also held to the click check of test_dsp (E3, V2, B1): the energy above 3 kHz in the
    # 30 ms around it. A hard switch measured -27.6 dB (bypass) there.
    y, changes, _ = t13_run(tone(317, 9.0, 0.1))
    worst = max(energy_above_db(y, c) for c in changes)
    assert worst <= -60.0, worst
    speech_x = speech_in(9.0, seed=4)
    x = (speech_x + tone(220, 9.0, db_to_lin(-30))).astype(np.float32)
    y, changes, lat = t13_run(x)
    faults = [f for f in audio_check.analyze(y) if f.kind in ("discontinuity", "zero run")]
    assert faults == []
    # The timing never shifts: in every 200 ms window the output lines up with the expected path, the input through
    # the chain's first fixed stage (the 80 Hz rumble filter, whose phase moves the 220 Hz tone), at the chain's
    # latency. Up to half a millisecond either way is the filters' own phase on the tone: the 3-Band EQ's crossover
    # low-passes and its x[n-3] (eq-filter.c:103-115, 9 samples measured with Low +4 dB), and RNNoise's wet path
    # against the dry one (3). A dropped or repeated block would move it by 480.
    expected = np.concatenate([RumbleFilter().process(x[i : i + BLOCK]) for i in range(0, len(x), BLOCK)])
    expected = expected.astype(np.float64)
    for start in range(RATE, len(y) - RATE // 5, RATE // 5):
        seg = y[start : start + RATE // 5]
        lags = np.arange(lat - 100, lat + 101)  # under half the tone's period, so it can't lock on a wrong cycle
        corr = [np.dot(seg, expected[start - k : start - k + len(seg)]) for k in lags]
        assert abs(int(lags[int(np.argmax(corr))]) - lat) <= 24, start / RATE


# ------------------------------------------------------------------ T16 the done-when's speech test chain


SPEECH = Path(__file__).resolve().parent / "data" / "speech.wav"


def faults_24bit(y: np.ndarray) -> list:
    """audio_check on ``y`` as the harness writes it (24-bit)."""
    q = np.round(y.astype(np.float64) * (1 << 23)) / (1 << 23)
    return audio_check.analyze(q, RATE, read_wav(SPEECH))


DROPOUTS = ("discontinuity", "long gap", "zero run")  # audio_check's dropout signatures; gate dips are level, not loss


def gate_record(monkeypatch: pytest.MonkeyPatch) -> tuple[list[int], dict[int, np.ndarray]]:
    """Record, per chain block, what the Noise Gate let through: its output for every sample it ran (its input times
    its own attenuation, noise-gate-filter.c:139-151). A block where it did not run (its eye fully off) has no entry."""
    from clean_voice import obs_filters

    block = [-1]
    outs: dict[int, np.ndarray] = {}
    real_ft, real_gate = FineTune.process, obs_filters.NoiseGate.process

    def counted(self, x):
        block[0] += 1
        return real_ft(self, x)

    def recorded(self, x):
        y = real_gate(self, x)
        outs[block[0]] = y.astype(np.float64)
        return y

    monkeypatch.setattr(FineTune, "process", counted)
    monkeypatch.setattr(obs_filters.NoiseGate, "process", recorded)
    return block, outs


EQ_DELAY = 3  # the 3-Band EQ's output is built on x[n-3] (eq-filter.c:103-115): a gate before it shows 3 samples late


def gate_silenced(outs: dict[int, np.ndarray], n: int) -> np.ndarray:
    """Per output sample: the Noise Gate silenced it. Closed, opening, or at the very end of its release, its own
    output is under half a 24-bit step, so the file holds an exact 0 there (the chain after it only cuts or keeps the
    level at OBS's defaults). Reset to closed by every settings change, as OBS does (noise-gate-filter.c:96-99)."""
    out = np.full(n + BLOCK, np.nan)
    for k, y in outs.items():
        out[k * BLOCK : k * BLOCK + len(y)] = y
    silent = np.abs(out[:n]) < 0.5 / (1 << 23)
    late = np.concatenate((np.zeros(EQ_DELAY, bool), silent[:-EQ_DELAY]))
    return silent | late


def retuned_blocks(moving: bool, n_blocks: int) -> set[int]:
    """The blocks at whose first sample --moving-filters changed an enabled filter's setting: OBS applies a new setting
    at once (gain-filter.c gain_update :34-40 and every filter's update), so the level can step there by design."""
    if not moving:
        return set()
    s = harness.parse_settings(["denoise=0", "style=0", "fine_tune=1"], FOUR)
    base, presets = s.filters, list(harness.st.PRESETS)
    out, prev = set(), s
    for k in range(n_blocks):
        cur = harness.moving_filters(prev, base, k, presets)
        before = {f.uid: f for f in prev.filters}
        if any(f.enabled and before[f.uid].enabled and f.settings != before[f.uid].settings for f in cur.filters):
            out.add(k)
        prev = cur
    return out


@pytest.mark.parametrize("moving", [False, True])
def test_t16_speech_through_gate_eq_compressor_gain(
    moving: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The done-when's test chain on speech: harness.py speech.wav out.wav --settings denoise=0 style=0 fine_tune=1
    --filters noise_gate_filter,basic_eq_filter,compressor_filter,gain_filter (also with --moving-filters), then
    audio_check.py out.wav --speech speech.wav. No dropout (orchestrator decision, option b): no long gap, no
    discontinuity, and no zero run except where the Noise Gate itself silenced the audio (closed or opening: its own
    output, recorded as it runs; a closed OBS gate's pauses are exact zeros in the 24-bit file). The Compressor's dips
    (OBS's 10:1 default) are level, not loss, and are not judged. With --moving-filters, a jump at the first sample of
    a block where a filter's setting changed is OBS's instant update, not a lost sample (no smoothing: OBS has none)."""
    block, outs = gate_record(monkeypatch)
    out = tmp_path / "out.wav"
    args = [str(SPEECH), str(out), "--settings", "denoise=0", "style=0", "fine_tune=1", "--filters", FOUR]
    assert harness.main([*args, "--moving-filters"] if moving else args) == 0
    faults = [f for f in audio_check.check_file(out, SPEECH) if f.kind in DROPOUTS]
    n = len(read_wav(out))
    assert block[0] + 1 == -(-n // BLOCK)  # one record per block: aligned with the output
    gated = gate_silenced(outs, n)
    steps = retuned_blocks(moving, block[0] + 1)
    # The zero runs' exact sample ranges, as audio_check's analyze finds them (inside long gaps they are not reported).
    y = read_wav(out)
    runs = dict(audio_check.zero_runs(y, RATE, skip=audio_check.long_gaps(y, RATE)))
    assert runs  # the gate does close in the pauses: the exception is exercised
    unexplained = []
    for f in faults:
        i = round(f.time * RATE)
        if f.kind == "zero run":
            if gated[i : runs[i]].all():
                continue
        elif f.kind == "discontinuity" and i % BLOCK == 0 and i // BLOCK in steps:
            continue  # an instant OBS setting change at this block's start, not a lost sample
        unexplained.append(f)
    assert unexplained == [], "\n".join(map(str, unexplained))


def test_t16_the_test_chain_has_no_dropouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """What does hold, and is the done-when's "no dropouts": every fault on the four-filter chain is one of OBS's
    filters doing what OBS does, never the engine. Each zero run lies where the Noise Gate's gain is exactly 0 (a
    closed or just opening gate), there is no discontinuity, and every dip is also there with OBS's Compressor alone."""
    from clean_voice import obs_filters

    gains: list[np.ndarray] = []
    real = obs_filters.NoiseGate.process

    def recorded(self, x):
        y = real(self, x)
        gains.append(self._scratch["gain"][: len(x)].copy())
        return y

    monkeypatch.setattr(obs_filters.NoiseGate, "process", recorded)
    x = read_wav(SPEECH).astype(np.float32)
    switches = ["denoise=0", "style=0", "fine_tune=1"]
    y = harness.run(x, harness.parse_settings(switches, FOUR))
    gate = np.concatenate(gains)[: len(y)]  # the gate runs every block (no eye changes): aligned with the output
    faults = faults_24bit(y)
    assert not [f for f in faults if f.kind in ("discontinuity", "long gap")]
    zero_runs = [f for f in faults if f.kind == "zero run"]
    assert zero_runs  # the gate does close in the pauses
    for f in zero_runs:
        i = round(f.time * RATE)
        n = round(float(f.detail.split()[0]) * RATE / 1000)
        # Closed, or just opening: the first 25 ms of the attack ramp on a quiet onset rounds to 0 in 24 bits.
        assert np.mean(gate[i : i + n] == 0.0) > 0.5 and gate[i : i + n].max() < 0.05, str(f)
    alone = faults_24bit(harness.run(x, harness.parse_settings(switches, "compressor_filter")))
    assert {round(f.time, 2) for f in faults if f.kind == "gate dip"} <= {
        round(f.time, 2) for f in alone if f.kind == "gate dip"
    }


def test_t16_moving_eyes_reorders_and_styles_are_fault_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """--moving-filters without its parameter retunes (OBS's instant updates): every block a new style and an eye
    toggled, every 10th block a reorder. Without the gate nothing at all is found; with it, only its closed pauses."""
    monkeypatch.setattr(harness, "_retuned", lambda base, now, k: now)
    x = read_wav(SPEECH).astype(np.float32)
    switches = ["denoise=0", "style=0", "fine_tune=1"]
    assert faults_24bit(harness.run(x, harness.parse_settings(switches, FOUR[18:]), moving=True)) == []
    faults = faults_24bit(harness.run(x, harness.parse_settings(switches, FOUR), moving=True))
    assert {f.kind for f in faults} <= {"zero run"}


def test_harness_filters_and_overrides() -> None:
    s = harness.parse_settings(
        ["fine_tune=1", "denoise=0"], "compressor_filter:ratio=4:attack_time=2,gain_filter:db=-3"
    )
    assert (s.fine_tune, s.denoise, s.style) == (True, False, True)
    assert [f.kind for f in s.filters] == ["compressor_filter", "gain_filter"]
    assert (s.filters[0].settings["ratio"], s.filters[0].settings["attack_time"]) == (4.0, 2)
    assert s.filters[1].settings["db"] == -3.0
    assert [f.name for f in harness.parse_settings([], "gain_filter,gain_filter").filters] == ["Gain", "Gain 2"]
    with pytest.raises(SystemExit):
        harness.parse_settings([], "wobble_filter")


def test_dsp_has_no_guest_eq_stage() -> None:
    assert "user_eq" not in Chain.STAGES and "fine_tune" in Chain.STAGES
    assert not hasattr(Chain(), "user_eq")
    assert dsp.FLAT_BANDS == bands_for("flat")
    assert fine_tune.KINDS  # the chain reads the same table
