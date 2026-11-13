"""The clean-up chain: rumble filter, RNNoise framing, compressor, limiter, EQ, strength, bypass and levels."""

import dataclasses
from itertools import pairwise

import numpy as np
import pytest
from speech import mix

from clean_voice import dsp
from clean_voice.dsp import (
    SAMPLE_RATE,
    Chain,
    Compressor,
    Limiter,
    NoiseSuppressor,
    RumbleFilter,
    db_to_lin,
)
from clean_voice.fine_tune import FilterSpec
from clean_voice.settings import Settings, SettingsCell, bands_for

CEILING = db_to_lin(-1.0)


def blocks(x: np.ndarray, size: int):
    for i in range(0, len(x), size):
        yield x[i : i + size]


def run(stage, x: np.ndarray, size: int = 480) -> np.ndarray:
    return np.concatenate([stage.process(b) for b in blocks(x.astype(np.float32), size)])


def rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-20)


def tone(freq: float, seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def speech_like(seed: int = 3, level_db: float = -20.0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Words of a voiced, syllable-modulated signal with stop-consonant gaps and soft endings, between pauses.

    Returns the signal and the (start, end) sample index of every word.
    """
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    words: list[tuple[int, int]] = []
    pos = 0
    for _ in range(12):
        pause = np.zeros(int(rng.uniform(0.4, 0.7) * SAMPLE_RATE), np.float32)
        parts.append(pause)
        pos += len(pause)
        dur = rng.uniform(0.5, 1.3)
        n = int(dur * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE
        f0 = 110 + 25 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6))
        phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
        voiced = sum(np.sin(k * phase) / k for k in range(1, 12))
        breath = rng.standard_normal(n) * 0.15
        # Syllables at about 4 per second; between syllables the level falls by about 20 dB.
        syllables = 0.1 + 0.9 * np.sin(np.pi * np.clip(t * 4.0 % 1.0, 0, 1)) ** 2
        env = syllables.copy()
        # Two stop consonants per word: 60-90 ms of near-silence mid-word.
        for _ in range(2):
            gap_start = rng.uniform(0.15, dur - 0.25)
            gap = (t >= gap_start) & (t < gap_start + rng.uniform(0.06, 0.09))
            env[gap] = 0.003
        # A soft ending: the last 150 ms fades by 25 dB.
        tail = t > dur - 0.15
        env[tail] *= np.linspace(1.0, 0.056, int(tail.sum()))
        word = (voiced + breath) * env
        word *= db_to_lin(level_db) / np.max(np.abs(word))
        parts.append(word.astype(np.float32))
        words.append((pos, pos + n))
        pos += n
    parts.append(np.zeros(int(0.5 * SAMPLE_RATE), np.float32))
    signal = np.concatenate(parts)
    noise = (rng.standard_normal(len(signal)) * db_to_lin(-70)).astype(np.float32)
    return signal + noise, words


# ------------------------------------------------------------------ the whole chain


def test_chain_processes_a_block() -> None:
    chain = Chain()
    block = (np.random.default_rng(0).standard_normal(480) * 0.1).astype(np.float32)
    out = chain.process(block)
    assert out.shape == (480,)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("size", [480, 333, 1024])
def test_clipping_input_never_exceeds_minus_1_dbfs(size: int) -> None:
    rng = np.random.default_rng(7)
    # A hot, clipped input: a loud tone driven past full scale, square-ish, with bursts of noise.
    x = np.clip(tone(180, 4.0, 1.6) + rng.standard_normal(4 * SAMPLE_RATE).astype(np.float32) * 0.4, -1.0, 1.0)
    x[SAMPLE_RATE : SAMPLE_RATE + 4800] = 1.0  # 100 ms stuck at full scale
    x[2 * SAMPLE_RATE : 2 * SAMPLE_RATE + 9600 : 2] = -1.0  # a full-scale buzz
    out = run(Chain(), x, size)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= CEILING


def test_background_noise_is_clearly_reduced() -> None:
    noise = (np.random.default_rng(1).standard_normal(3 * SAMPLE_RATE) * 0.03).astype(np.float32)
    out = run(Chain(), noise)
    second_half = slice(len(out) // 2, None)
    assert rms_db(out[second_half]) < rms_db(noise[second_half]) - 30


def test_silence_stays_silent() -> None:
    out = run(Chain(), np.zeros(SAMPLE_RATE, np.float32))
    assert np.max(np.abs(out)) < 1e-6


# ------------------------------------------------------------------ stages


def test_rumble_filter_cuts_low_rumble_and_keeps_voice() -> None:
    low = run(RumbleFilter(), tone(30, 2.0, 0.5))
    voice = run(RumbleFilter(), tone(1000, 2.0, 0.5))
    settled = slice(SAMPLE_RATE, None)
    assert rms_db(low[settled]) < rms_db(tone(30, 2.0, 0.5)[settled]) - 30
    assert abs(rms_db(voice[settled]) - rms_db(tone(1000, 2.0, 0.5)[settled])) < 0.5


def test_noise_suppressor_framing_is_independent_of_block_size() -> None:
    # OBS noise-suppress-filter.c buffering: any block size gives the same output with a fixed 480-sample delay.
    x = (np.random.default_rng(2).standard_normal(SAMPLE_RATE) * 0.05).astype(np.float32)
    in_frames = run(NoiseSuppressor(), x, 480)
    odd = run(NoiseSuppressor(), x, 137)
    assert np.allclose(in_frames, odd, atol=1e-7)
    assert np.all(in_frames[: NoiseSuppressor.LATENCY] == 0.0)


@pytest.mark.parametrize("level_db", [-20.0, -36.0])
def test_the_chain_never_cuts_a_word(level_db: float) -> None:
    # The "never cuts a word" half of the old gate test, on the whole chain (there is no gate since 2026-11-11): words
    # with stop-consonant gaps and soft endings, at a normal and a quiet (-36 dBFS peak) level, all come through.
    from speech import word_drops_db

    signal, words = speech_like(seed=5 if level_db < -30 else 3, level_db=level_db)
    out = run(Chain(), signal)
    drops = word_drops_db(out, signal, words, Chain.LATENCY)
    assert drops.min() >= -3.0, drops


def test_compressor_reduces_loud_peaks_by_the_ratio() -> None:
    loud = run(Compressor(), tone(500, 1.0, db_to_lin(-6)))
    quiet = run(Compressor(), tone(500, 1.0, db_to_lin(-36)))
    settled = slice(SAMPLE_RATE // 2, None)
    # -6 dBFS peak is 18 dB over the -24 dB threshold: 3:1 leaves 6 dB over, then +6 dB makeup.
    peak_db = 20 * np.log10(np.max(np.abs(loud[settled])))
    assert peak_db == pytest.approx(-24 + 6 + 6, abs=0.5)
    quiet_db = 20 * np.log10(np.max(np.abs(quiet[settled])))
    assert quiet_db == pytest.approx(-36 + 6, abs=0.2)


def obs_compressor(x: np.ndarray, threshold_db: float, ratio: float, attack_s: float, release_s: float, gain_db: float):
    """OBS plugins/obs-filters/compressor-filter.c, sample by sample: the |x| follower (:310-318, coefficients
    exp(-1 / (rate * t)), :133-136) and the gain (1 - 1/ratio)(threshold - env dB), reduction only, times the output
    gain (:199, :358-367). The reference the audit's probe_compressor.py compared against."""
    a = np.exp(-1.0 / (SAMPLE_RATE * attack_s))
    r = np.exp(-1.0 / (SAMPLE_RATE * release_s))
    slope = 1.0 - 1.0 / ratio
    env = 0.0
    out = np.empty(len(x))
    for i, v in enumerate(np.abs(x.astype(np.float64)).tolist()):
        env = v + (a if env < v else r) * (env - v)
        reduction = min(0.0, slope * (threshold_db - 20 * np.log10(max(env, 1e-30))))
        out[i] = x[i] * 10 ** (reduction / 20) * 10 ** (gain_db / 20)
    return out


@pytest.mark.parametrize("size", [480, 333])
def test_compressor_follows_obs_sample_by_sample(size: int) -> None:
    # OBS audit I-2: the follower runs on |x| every sample, not on 1 ms block peaks (which measured 1.89 dB more
    # reduction on speech at -18 dBFS than OBS at the same settings).
    from speech import speech

    x = (speech()[5 * SAMPLE_RATE : 9 * SAMPLE_RATE] * db_to_lin(8.0)).astype(np.float32)  # loud speech: it compresses
    got = run(Compressor(), x, size).astype(np.float64)
    want = obs_compressor(
        x,
        dsp.COMP_THRESHOLD_DB,
        dsp.COMP_RATIO,
        dsp.COMP_ATTACK_MS / 1000,
        dsp.COMP_RELEASE_MS / 1000,
        dsp.COMP_MAKEUP_DB,
    )
    assert np.max(np.abs(got - want)) < 1e-5


def test_limiter_holds_the_ceiling_and_releases() -> None:
    x = np.concatenate([tone(300, 0.5, 2.0), tone(300, 1.0, 0.3)])
    out = run(Limiter(), x, 256)
    assert np.max(np.abs(out)) <= CEILING
    tail = out[-SAMPLE_RATE // 4 :]
    assert np.max(np.abs(tail)) == pytest.approx(0.3, rel=1e-3)  # fully released after the burst


@pytest.mark.parametrize("stage_type", [Compressor, Limiter, dsp.RumbleFilter])
def test_state_carries_across_blocks(stage_type) -> None:
    x, _ = speech_like(seed=9)
    x = x[: 2 * SAMPLE_RATE] * 8
    one = run(stage_type(), x, 960)
    many = run(stage_type(), x, 480)
    assert np.allclose(one, many, atol=1e-6)


# ------------------------------------------------------------------ J06 Step 5: EQ, voice volume, bypass

BLOCK = 480


def rbj_oracle(i: int, freq: float, gain: float, q: float) -> tuple[float, ...]:
    """A port of the original's curve formulas (simple-clean-mic ui/main.js:234-251)."""
    w = 2 * np.pi * freq / SAMPLE_RATE
    s, c = np.sin(w), np.cos(w)
    a = 10 ** (gain / 40)
    if i in (0, 4):
        alpha = s / 2 * np.sqrt(2)
        sa = 2 * np.sqrt(a) * alpha
        sign = -1 if i == 0 else 1
        b0 = a * ((a + 1) + sign * (a - 1) * c + sa)
        b1 = -sign * 2 * a * ((a - 1) + sign * (a + 1) * c)
        b2 = a * ((a + 1) + sign * (a - 1) * c - sa)
        a0 = (a + 1) - sign * (a - 1) * c + sa
        a1 = sign * 2 * ((a - 1) - sign * (a + 1) * c)
        a2 = (a + 1) - sign * (a - 1) * c - sa
        return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
    alpha = s / (2 * q)
    a0 = 1 + alpha / a
    return (1 + alpha * a) / a0, -2 * c / a0, (1 - alpha * a) / a0, -2 * c / a0, (1 - alpha / a) / a0


def oracle_db(coefs: tuple[float, ...], f: float) -> float:
    """main.js:253-258."""
    b0, b1, b2, a1, a2 = coefs
    w = 2 * np.pi * f / SAMPLE_RATE
    c1, s1, c2, s2 = np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)
    nr, ni = b0 + b1 * c1 + b2 * c2, -(b1 * s1 + b2 * s2)
    dr, di = 1 + a1 * c1 + a2 * c2, -(a1 * s1 + a2 * s2)
    return float(10 * np.log10((nr * nr + ni * ni) / (dr * dr + di * di)))


def response_db(eq, n: int = 1 << 16) -> tuple[np.ndarray, np.ndarray]:
    imp = np.zeros(n, np.float32)
    imp[0] = 1.0
    h = run(eq, imp, BLOCK).astype(np.float64)
    return np.fft.rfftfreq(n, 1 / SAMPLE_RATE), 20 * np.log10(np.abs(np.fft.rfft(h)))


@pytest.mark.parametrize("preset", ["natural", "warm", "crisp", "radio", "deep", "flat"])
def test_eq_matches_the_original_curve(preset: str) -> None:
    # E1
    from clean_voice.dsp import Equalizer

    bands = bands_for(preset)
    freqs, db = response_db(Equalizer(bands))
    for f in np.geomspace(30, 18_000, 20):
        k = int(np.argmin(np.abs(freqs - f)))
        expected = sum(oracle_db(rbj_oracle(i, b.freq, b.gain, b.q), freqs[k]) for i, b in enumerate(bands))
        assert db[k] == pytest.approx(expected, abs=0.05), f"{preset} at {freqs[k]:.0f} Hz"
    for i in (1, 2, 3):  # a peaking band gives exactly its gain at its own frequency
        b = bands[i]
        assert oracle_db(rbj_oracle(i, b.freq, b.gain, b.q), b.freq) == pytest.approx(b.gain, abs=1e-9)
        k = int(np.argmin(np.abs(freqs - b.freq)))
        alone = list(bands_for("flat"))
        alone[i] = b
        _, db_alone = response_db(Equalizer(tuple(alone)))
        assert db_alone[k] == pytest.approx(oracle_db(rbj_oracle(i, b.freq, b.gain, b.q), freqs[k]), abs=0.01)


def test_flat_eq_is_transparent() -> None:
    # E2
    from clean_voice.dsp import Equalizer

    x = (np.random.default_rng(4).standard_normal(SAMPLE_RATE) * 0.1).astype(np.float32)
    assert np.max(np.abs(run(Equalizer(bands_for("flat")), x) - x)) < 1e-6


def energy_above_db(y: np.ndarray, center: int, cut_hz: float = 3000.0) -> float:
    """Energy above cut_hz in the 30 ms around ``center``, against the window's whole energy (Blackman-Harris)."""
    from scipy.signal.windows import blackmanharris

    half = int(0.015 * SAMPLE_RATE)
    seg = y[center - half : center + half].astype(np.float64)
    spec = np.abs(np.fft.rfft(seg * blackmanharris(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / SAMPLE_RATE)
    return float(10 * np.log10(spec[f > cut_hz].sum() / spec.sum() + 1e-30))


def max_step(y: np.ndarray) -> float:
    return float(np.max(np.abs(np.diff(y.astype(np.float64)))))


def test_eq_retune_is_click_free() -> None:
    # E3: the original swaps the coefficients in one step (dsp.rs:70-76); Clean Voice glides over 20 ms.
    from clean_voice.dsp import Equalizer

    start = list(bands_for("natural"))
    start[2] = dataclasses.replace(start[2], gain=-12.0)
    end = list(start)
    end[2] = dataclasses.replace(start[2], gain=12.0)
    end[4] = dataclasses.replace(start[4], freq=3000.0)
    eq = Equalizer(tuple(start))
    x = tone(1000, 1.0, db_to_lin(-12))
    change = SAMPLE_RATE // 2
    out = [eq.process(x[i : i + BLOCK]) for i in range(0, change, BLOCK)]
    eq.set_bands(tuple(end))
    out += [eq.process(x[i : i + BLOCK]) for i in range(change, len(x), BLOCK)]
    y = np.concatenate(out)
    assert energy_above_db(y, change) <= -60.0
    assert max_step(y[change - 480 : change + 4800]) <= 1.2 * max_step(y[-SAMPLE_RATE // 4 :])


@pytest.mark.parametrize("size", [960, 333])
def test_eq_state_and_glide_carry_across_blocks(size: int) -> None:
    # E4: the same output whatever the block size, also while the bands are moving.
    from clean_voice.dsp import Equalizer

    x, _ = speech_like(seed=9)
    x = x[: 2 * SAMPLE_RATE] * 8
    outs = []
    change = SAMPLE_RATE // 2
    for bs in (BLOCK, size):
        eq = Equalizer(bands_for("natural"))
        parts = [eq.process(x[i : min(i + bs, change)]) for i in range(0, change, bs)]
        eq.set_bands(bands_for("radio"))
        parts += [eq.process(x[i : i + bs]) for i in range(change, len(x), bs)]
        outs.append(np.concatenate(parts))
    assert np.allclose(outs[0], outs[1], atol=1e-6)


def test_voice_volume_mapping() -> None:
    # V1: 24 * (v - 50) / 100 dB: 50 is 0 dB, the range +-12 dB. Measured through the whole chain on a steady tone
    # below the compressor threshold, against Voice volume 50.
    x = tone(300, 2.0, db_to_lin(-40))
    level = {}
    for v in (0, 50, 100):
        y = run(Chain(SettingsCell(Settings(loudness=v, strength=0))), x)
        level[v] = rms_db(y[SAMPLE_RATE:])
    assert level[0] - level[50] == pytest.approx(-12.0, abs=0.05)
    assert level[100] - level[50] == pytest.approx(12.0, abs=0.05)


def test_voice_volume_change_is_click_free() -> None:
    # V2
    from clean_voice.dsp import VoiceVolume

    vol = VoiceVolume(0)
    x = tone(1000, 1.0, db_to_lin(-30))
    change = SAMPLE_RATE // 2
    out = [vol.process(x[i : i + BLOCK]) for i in range(0, change, BLOCK)]
    vol.set_loudness(100)
    out += [vol.process(x[i : i + BLOCK]) for i in range(change, len(x), BLOCK)]
    y = np.concatenate(out)
    gain = np.abs(y) / np.maximum(np.abs(x), 1e-6)
    peaks = np.array([gain[i : i + 48].max() for i in range(change - 480, change + 1440, 48)])
    assert np.all(np.diff(peaks) >= -1e-4)  # monotonic
    done = int(np.argmax(peaks >= db_to_lin(12) * 0.999))
    assert 0 < done * 48 - 480 <= 480  # the whole move takes at most 10 ms
    assert energy_above_db(y, change) <= -60.0


def tone_switch_hf_db(change: dict, start: Settings) -> float:
    """A settings change mid-stream on a steady 317 Hz tone (not locked to the 480-sample blocks): the energy above
    3 kHz in the 30 ms around the switch, the click check E3 and V2 use. A hard switch measured -27.6 dB (bypass) and
    -47.0 dB (strength) against -75.9 and -73.8 dB ramped (J06 review r1, F2)."""
    x = tone(317, 4.0, 0.1)
    cell = SettingsCell(start)
    chain = Chain(cell)
    at = 2 * SAMPLE_RATE
    out = []
    for i in range(0, len(x), BLOCK):
        if i == at:
            cell.value = dataclasses.replace(cell.value, **change)
        out.append(chain.process(x[i : i + BLOCK]))
    return energy_above_db(np.concatenate(out), at)


def test_bypass_crossfades_and_then_is_the_raw_mic() -> None:
    # B1: Off sends the raw mic to the cable, as the original's on/off did (main.js:622, audio.rs:278-287), delayed by
    # the chain's latency so the crossfade lines up. First both ways on a tone: speech's own steps hide a hard switch
    # from the step check further down (review r1, F2).
    assert tone_switch_hf_db({"cleaning": False}, Settings(strength=0)) <= -60.0
    assert tone_switch_hf_db({"cleaning": True}, Settings(strength=0, cleaning=False)) <= -60.0
    x, _, _ = mix(-26.0, 20.0, seed=3)
    cell = SettingsCell(Settings())
    chain = Chain(cell)
    marks = {int(8.0 * SAMPLE_RATE): False, int(16.0 * SAMPLE_RATE): True, int(24.0 * SAMPLE_RATE): False}
    out = []
    for i in range(0, len(x), BLOCK):
        if i in marks:
            cell.value = dataclasses.replace(cell.value, cleaning=marks[i])
        out.append(chain.process(x[i : i + BLOCK]))
    y = np.concatenate(out)
    lat = Chain.LATENCY
    raw = slice(9 * SAMPLE_RATE, 16 * SAMPLE_RATE)
    assert np.max(np.abs(y[raw] - x[raw.start - lat : raw.stop - lat])) < 1e-6
    steady = max(max_step(y[SAMPLE_RATE : 8 * SAMPLE_RATE]), max_step(y[raw]))
    for m in marks:
        assert max_step(y[m - 480 : m + 1440]) <= 1.2 * steady, f"a click at {m / SAMPLE_RATE:.1f} s"


def run_chain(chain: Chain, x: np.ndarray, size: int = BLOCK) -> np.ndarray:
    return np.concatenate([chain.process(x[i : i + size]) for i in range(0, len(x), size)])


# ------------------------------------------------------------------ J06 Step 6: strength, levels, clipping


def test_dry_path_is_aligned_with_rnnoise_output() -> None:
    # S1: y = wet + d * (dry - wet) only works if dry and wet line up; measure RNNoise's whole latency.
    x, _, _ = mix(-26.0, None, seed=2)
    chain = Chain(SettingsCell(Settings(strength=0)))
    wet, dry = [], []
    for i in range(0, len(x), BLOCK):
        chain.process(x[i : i + BLOCK])
        wet.append(chain.denoise.wet.copy())
        dry.append(chain.dry.copy())
    w, d = np.concatenate(wet).astype(np.float64), np.concatenate(dry).astype(np.float64)
    lags = np.arange(-960, 961)
    corr = [np.dot(w[2000 + k : len(w) - 2000 + k], d[2000 : len(d) - 2000]) for k in lags]
    assert lags[int(np.argmax(corr))] == 0
    # ...and the delay used is the measured one: RNNoise's output against its own input.
    ns = NoiseSuppressor()
    wet_only = run(ns, x).astype(np.float64)
    xin = x.astype(np.float64)
    corr_in = [np.dot(wet_only[k:], xin[: len(xin) - k]) for k in range(2000)]
    assert int(np.argmax(corr_in)) == Chain.LATENCY == 1440  # 480 queued + 960 inside RNNoise 0.2


def test_whole_frame_blocks_add_no_queue_latency() -> None:
    # OBS audit m-1: OBS hands RNNoise's output on as soon as a whole packet exists (noise-suppress-filter.c:451-452). The
    # worker always runs 480-sample blocks, so the 480 zeros pre-loaded for odd block sizes were 10 ms OBS does not add.
    x, _, _ = mix(-26.0, None, seed=2)
    chain = Chain(SettingsCell(Settings(strength=100)), block=480)
    assert chain.latency == 2 * 480  # RNNoise 0.2's own 960 samples, nothing queued on top
    out = np.concatenate([chain.process(x[i : i + 480]) for i in range(0, len(x), 480)])
    corr = [np.dot(out[k : k + 10 * SAMPLE_RATE].astype(np.float64), x[: 10 * SAMPLE_RATE]) for k in range(0, 2000, 48)]
    assert int(np.argmax(corr)) * 48 == pytest.approx(chain.latency, abs=48)
    # The raw (Off) path and the dry path are delayed by the same amount, so the bypass crossfade still lines up.
    off = Chain(SettingsCell(Settings(cleaning=False)), block=480)
    y = np.concatenate([off.process(x[i : i + 480]) for i in range(0, 20 * 480, 480)])
    assert np.max(np.abs(y[off.latency :] - x[: 20 * 480 - off.latency])) < 1e-6
    with pytest.raises(ValueError):
        Chain(block=480).process(np.zeros(333, np.float32))  # a block that breaks the promise is refused, not padded


def noise_only(name: str, seconds: float = 6.0, level_db: float = -40.0) -> np.ndarray:
    from speech import looped_noise

    n = looped_noise(int(seconds * SAMPLE_RATE), seed=5, name=name)
    return (n * db_to_lin(level_db) / np.sqrt(np.mean(n**2))).astype(np.float32)


# Noise alone at strength 70 is turned down at least 10 dB on the tonal hum and at least 12 dB on the broadband fan, and
# more strength never removes less (user decision, 2026-11-11, with the gate removed: RNNoise alone measured hum 10.6 dB,
# fan 12.9 dB; the gate had given 11.2 and 36.5 dB).
NOISE_ALONE_MIN_DB = {"noise.wav": 10.0, "fan.wav": 12.0}
STRENGTH_SWEEP = (0, 20, 50, 70, 100)


@pytest.mark.parametrize("name", ["noise.wav", "fan.wav"])
def test_strength_sets_how_much_noise_is_removed(name: str) -> None:
    # S2 (and N1 part a)
    x = noise_only(name)
    half = slice(len(x) // 2, None)
    reduction = {}
    for s in STRENGTH_SWEEP:
        y = run_chain(Chain(SettingsCell(Settings(strength=s))), x)
        reduction[s] = rms_db(x[half]) - rms_db(y[half])
    steps = [reduction[b] - reduction[a] for a, b in pairwise(STRENGTH_SWEEP)]
    assert min(steps) >= -0.05, reduction  # more strength never removes less
    assert reduction[0] < reduction[70], reduction
    assert reduction[70] >= NOISE_ALONE_MIN_DB[name], reduction


def test_strength_change_is_click_free() -> None:
    # S4. First on a tone: the step check on noisy speech below cannot see a hard switch (review r1, F2).
    assert tone_switch_hf_db({"strength": 100}, Settings(strength=0)) <= -60.0
    x, _, _ = mix(-26.0, 10.0, seed=4)
    cell = SettingsCell(Settings(strength=0))
    chain = Chain(cell)
    change = 10 * SAMPLE_RATE
    out = []
    for i in range(0, len(x), BLOCK):
        if i == change:
            cell.value = dataclasses.replace(cell.value, strength=100)
        out.append(chain.process(x[i : i + BLOCK]))
    y = np.concatenate(out)
    steady = max(max_step(y[2 * SAMPLE_RATE : change]), max_step(y[change + SAMPLE_RATE :]))
    assert max_step(y[change - 480 : change + 1440]) <= 1.2 * steady


def clip_inputs() -> dict[str, np.ndarray]:
    from speech import speech

    rng = np.random.default_rng(7)
    x = np.clip(tone(180, 2.0, 1.6) + rng.standard_normal(2 * SAMPLE_RATE).astype(np.float32) * 0.4, -1.0, 1.0)
    x[SAMPLE_RATE // 2 : SAMPLE_RATE // 2 + 4800] = 1.0  # 100 ms stuck at full scale
    x[SAMPLE_RATE : SAMPLE_RATE + 9600 : 2] = -1.0  # a full-scale buzz
    s = speech()[6 * SAMPLE_RATE : 10 * SAMPLE_RATE] * db_to_lin(30)  # speech driven +30 dB, clipped
    return {"tone-noise-buzz": x, "hot speech": np.clip(s, -1.0, 1.0).astype(np.float32)}


CLIP_SETTINGS = [
    Settings(bands=bands_for(p), loudness=v, strength=s)
    for p in ("natural", "warm", "crisp", "radio", "deep", "flat")
    for v in (0, 50, 100)
    for s in (0, 70, 100)
] + [Settings(bands=tuple(dataclasses.replace(b, gain=18.0) for b in bands_for("flat")), loudness=100)]
# Fine-tune at its extremes (J10): a 3-Band EQ with every band +20 dB and a +30 dB Gain, on top of the style EQ at
# +18 dB and the loudest Voice volume, with Denoise on and off.
EXTREME_FILTERS = (
    FilterSpec("basic_eq_filter", "3-Band Equalizer", settings={"low": 20.0, "mid": 20.0, "high": 20.0}),
    FilterSpec("gain_filter", "Gain", settings={"db": 30.0}),
)
CLIP_SETTINGS += [
    Settings(
        bands=tuple(dataclasses.replace(b, gain=18.0) for b in bands_for("flat")),
        loudness=100,
        fine_tune=True,
        filters=EXTREME_FILTERS,
        denoise=d,
    )
    for d in (True, False)
]


@pytest.mark.parametrize("size", [480, 333])
def test_nothing_exceeds_minus_1_dbfs_at_any_setting(size: int) -> None:
    # C1 (DESIGN Features: the output never exceeds -1 dBFS)
    worst = 0.0
    for name, x in clip_inputs().items():
        for s in CLIP_SETTINGS:
            out = run_chain(Chain(SettingsCell(s)), x, size)
            assert np.all(np.isfinite(out)), (name, s)
            peak = float(np.max(np.abs(out)))
            worst = max(worst, peak)
            assert peak <= CEILING, (name, s, 20 * np.log10(peak))
    assert worst > db_to_lin(-1.5)  # the inputs really reached the limiter


def test_bypass_still_holds_minus_1_dbfs() -> None:
    # B2
    rng = np.random.default_rng(8)
    x = np.clip(tone(180, 3.0, 1.6) + rng.standard_normal(3 * SAMPLE_RATE).astype(np.float32) * 0.4, -1.0, 1.0)
    out = run(Chain(SettingsCell(Settings(cleaning=False))), x)
    assert np.all(np.isfinite(out)) and np.max(np.abs(out)) <= CEILING
