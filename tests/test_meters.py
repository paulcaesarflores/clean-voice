"""J10 Step 9a (DESIGN "OBS-style meters and faders"): the level meters follow OBS's VolumeMeter.cpp, the volume
sliders read in dB as OBS's fader does (VolumeSlider.cpp, VolumeControl.cpp, libobs/obs-audio-controls.c), and Mic
level shows the dB the OS reports. The expected numbers sit here with their OBS file and line."""

from __future__ import annotations

import ctypes
import math
import sys
from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot
from test_main_screen import MICS, ON, OUTPUTS, FakeLoginItem

from clean_voice import prefs
from clean_voice.app import MainWindow, monitor_percent, monitor_position, monitor_text, voice_volume_text
from clean_voice.preview import PreviewEngine
from clean_voice.settings import Settings
from clean_voice.widgets import LevelMeter, db_text, fader_db, fader_deflection

# ------------------------------------------------------------------ the meter (VolumeMeter.cpp)


def test_the_meters_levels_are_obss() -> None:
    assert LevelMeter.MIN_DB == -60.0  # minimumLevel, VolumeMeter.cpp:314
    assert (LevelMeter.WARNING_DB, LevelMeter.ERROR_DB) == (-20.0, -9.0)  # :315-316, the sample-peak meter :269-277
    assert LevelMeter.CLIP_DB == 0.0  # clipLevel :317
    assert LevelMeter.MIN_INPUT_DB == -50.0  # minimumInputLevel :318
    assert LevelMeter.PEAK_DECAY_DB_S == 11.76  # peakDecayRate :319
    assert LevelMeter.MAGNITUDE_S == 0.3  # magnitudeIntegrationTime :320
    assert (LevelMeter.PEAK_HOLD_S, LevelMeter.INPUT_HOLD_S) == (20.0, 1.0)  # :321-322
    assert (LevelMeter.TICK_DB, LevelMeter.TICK_SIZE, LevelMeter.INDICATOR) == (6, 2, 3)  # :15, :17, :18
    assert LevelMeter.CLIP_FLASH_S == 1.0  # CLIP_FLASH_DURATION_MS :16
    assert LevelMeter.THICKNESS == min(max(22 // 1, 3), 6)  # doLayout :496-497, one channel


@pytest.mark.parametrize(
    ("db", "length", "position"),
    [
        (-60.0, 300, 0),
        (0.0, 300, 300),
        (-30.0, 300, 150),
        (-20.0, 300, 200),  # the warning zone starts two thirds along
        (-9.0, 300, 255),  # length - int(-9 * 300 / -60) = 300 - 45
        (-9.5, 301, 254),  # int() truncates, as OBS's convertToInt: 301 - int(47.66)
        (-80.0, 300, 0),  # below the scale: its start
        (-math.inf, 300, 0),
        (3.0, 300, 300),  # above 0 dB: its end
    ],
)
def test_the_db_to_position_mapping_is_obss(db: float, length: int, position: int) -> None:
    # updateBackgroundCache :707-710 and paintEvent :784: position = length - int(dB * length / minimumLevel).
    assert LevelMeter.position(db, length) == position


def make_meter(qtbot: QtBot, width: int = 305) -> LevelMeter:
    meter = LevelMeter()
    qtbot.addWidget(meter)
    meter.resize(width, meter.height())
    return meter


def test_the_zones_and_their_colours(qtbot: QtBot) -> None:
    from clean_voice import theme

    meter = make_meter(qtbot)
    pal = theme.current()
    assert [meter.zone_color(db) for db in (-40.0, -20.1, -20.0, -9.1, -9.0, 0.0)] == [
        pal.signal,
        pal.signal,
        pal.hot,
        pal.hot,
        pal.clip,
        pal.clip,
    ]
    meter.set_muted(True)  # the call can't hear it: OBS's disabled (grey) colours
    assert {meter.zone_color(db) for db in (-40.0, -15.0, -3.0)} == {pal.muted}


def test_the_peak_rises_at_once_and_falls_11_76_db_a_second(qtbot: QtBot) -> None:
    # calculateBallisticsForChannel :519-529.
    meter = make_meter(qtbot)
    meter.set_level(0.5, 0.033)
    assert meter.peak == pytest.approx(-6.0206, abs=1e-3)
    meter.set_level(0.0, 1.0)
    assert meter.peak == pytest.approx(-6.0206 - 11.76, abs=1e-3)
    meter.set_level(0.1, 0.1)  # -20 dB is below the falling peak, so it keeps falling
    assert meter.peak == pytest.approx(-6.0206 - 11.76 * 1.1, abs=1e-3)
    meter.set_level(0.9, 0.033)
    assert meter.peak == pytest.approx(20 * math.log10(0.9))


def test_the_peak_hold_stays_20_seconds_then_falls_back(qtbot: QtBot) -> None:
    # :531-544.
    meter = make_meter(qtbot)
    meter.set_level(0.5, 0.033)
    held = meter.peak_hold
    meter.set_level(0.01, 19.0)
    assert meter.peak_hold == held
    meter.set_level(0.01, 0.9)
    assert meter.peak_hold == held  # 19.9 s: still held
    meter.set_level(0.01, 0.2)
    assert meter.peak_hold == pytest.approx(-40.0)  # past 20 s: the current peak


def test_the_input_indicator_holds_one_second_and_takes_obss_colours(qtbot: QtBot) -> None:
    # :546-559 and getPeakColor :585-602.
    from clean_voice import theme

    pal = theme.current()
    meter = make_meter(qtbot)
    meter.set_level(0.5, 0.033)  # -6 dB: the error zone
    assert meter.input_color().name().upper() == pal.clip.upper()
    meter.set_level(0.05, 0.9)
    assert meter.input_hold == pytest.approx(-6.0206, abs=1e-3)
    meter.set_level(0.05, 0.2)
    assert meter.input_hold == pytest.approx(-26.02, abs=1e-2)  # past 1 s: -26 dB, nominal
    assert meter.input_color().name().upper() == pal.signal.upper()
    meter.set_level(0.001, 1.1)  # -60 dB, under minimumInputLevel: the dim nominal colour
    assert meter.input_color().name().upper() not in (pal.signal.upper(), pal.hot.upper(), pal.clip.upper())
    meter.set_level(1.0, 0.033)  # 0 dB: the clip colour
    assert meter.input_color().name().upper() == pal.ink.upper()


def test_a_peak_at_0_db_flashes_the_whole_bar_for_one_second(qtbot: QtBot) -> None:
    # :804-810.
    meter = make_meter(qtbot)
    meter.set_level(1.0, 0.033)
    assert meter.clipping
    meter.set_level(0.1, 0.5)
    assert meter.clipping
    meter.set_level(0.1, 0.6)
    assert not meter.clipping


def test_the_magnitude_integrates_to_99_percent_in_300_ms(qtbot: QtBot) -> None:
    # :561-573: the first value is taken as it is, then each tick moves (target - shown) * dt / 0.3 * 0.99.
    meter = make_meter(qtbot)
    meter.set_level(0.1, 0.033, magnitude=0.1)
    assert meter.magnitude == pytest.approx(-20.0)
    meter.set_level(1.0, 0.3, magnitude=1.0)
    assert meter.magnitude == pytest.approx(-20.0 + 20.0 * 0.99)
    meter.set_level(0.0, 0.3, magnitude=0.0)  # silence: clamped to the scale's bottom
    assert meter.magnitude == -60.0


def test_no_audio_resets_the_meter(qtbot: QtBot) -> None:
    # detectIdle/resetLevels :397-412, :506-515: no stream, nothing shown.
    meter = make_meter(qtbot)
    meter.set_level(0.5, 0.033)
    meter.active = False
    meter.set_level(0.5, 0.033)
    assert meter.peak == meter.peak_hold == meter.input_hold == -math.inf and meter.fraction == 0.0


def test_the_scale_has_a_tick_and_a_number_every_6_db(qtbot: QtBot) -> None:
    # paintHTicks :604-632: 0, -6 ... -60, a 2 px tick each; the numbers left out every other one when too narrow.
    wide = make_meter(qtbot, 460)
    ticks = wide.tick_labels()
    assert [label for _, label in ticks] == [str(-6 * k) for k in range(11)]
    xs = [x for x, _ in ticks]
    assert xs == sorted(xs, reverse=True) and xs[0] == wide.width() - 1  # 0 dB at the right end
    narrow = make_meter(qtbot, 160)
    labels = [label for _, label in narrow.tick_labels()]
    assert labels[0::2] == [str(-12 * k) for k in range(6)] and not any(labels[1::2])
    for meter in (wide, narrow):
        assert not meter.grab().isNull()


def test_the_meter_repaints_only_when_a_position_changes(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    meter = make_meter(qtbot)
    meter.set_level(0.5, 0.033)
    updates: list[int] = []
    monkeypatch.setattr(meter, "update", lambda *a: updates.append(1))
    meter.set_level(0.5, 0.0)  # the same peak, no time: nothing moved
    assert updates == []
    meter.set_level(0.05, 0.5)
    assert updates == [1]


# ------------------------------------------------------------------ the fader and its readout


def test_the_fader_mapping_is_obss_log_fader() -> None:
    # libobs/obs-audio-controls.c :159-185: LOG_OFFSET_DB 6, LOG_RANGE_DB 96, LOG_OFFSET_VAL -0.778..., LOG_RANGE_VAL
    # -2.0086...; the fader reads 0 dB at the top and -inf at the bottom.
    assert -math.log10(6.0) == pytest.approx(-0.77815125038364363)
    assert -math.log10(96.0 + 6.0) == pytest.approx(-2.00860017176191756)
    assert fader_db(1.0) == 0.0 and fader_db(0.0) == -math.inf
    assert fader_db(0.5) == pytest.approx(-102.0 * (102.0 / 6.0) ** -0.5 + 6.0)  # about -18.7 dB
    assert fader_deflection(0.0) == 1.0 and fader_deflection(-96.0) == 0.0 and fader_deflection(-120.0) == 0.0
    for d in (0.05, 0.25, 0.5, 0.75, 0.99):
        assert fader_deflection(fader_db(d)) == pytest.approx(d)


@pytest.mark.parametrize(
    ("db", "text"),
    [
        (0.0, "0.0 dB"),
        (-0.04, "0.0 dB"),  # never "-0.0 dB"
        (-18.739, "-18.7 dB"),
        (-96.0, "-96.0 dB"),
        (-96.01, "-inf dB"),  # VolumeControl::updateText :854: below -96 dB
        (-math.inf, "-inf dB"),
        (12.0, "12.0 dB"),
    ],
)
def test_the_readout_is_obss(db: float, text: str) -> None:
    assert db_text(db) == text  # VolumeControl::updateText :849-860


def test_voice_volume_reads_in_db_at_its_ends_and_middle() -> None:
    # Voice volume is -12..+12 dB (dsp.loudness_db), 50 = 0 dB.
    assert [voice_volume_text(v) for v in (0, 25, 50, 75, 100)] == [
        "-12.0 dB",
        "-6.0 dB",
        "0.0 dB",
        "6.0 dB",
        "12.0 dB",
    ]


def test_hear_myself_is_an_obs_fader_and_keeps_the_saved_percent() -> None:
    # Its position is the fader's deflection; the engine still takes a percent (gain = (percent / 100)^2), so old
    # saved values mean what they meant.
    assert [monitor_text(p) for p in (0, 50, 100)] == ["-inf dB", "-18.7 dB", "0.0 dB"]
    assert monitor_percent(0) == 0.0 and monitor_percent(100) == 100.0
    assert monitor_position(0.0) == 0.0 and monitor_position(100.0) == 100.0
    for percent in (10.0, 30.0, 50.0, 70.0):
        gain_db = 40.0 * math.log10(percent / 100.0)
        assert fader_db(monitor_position(percent) / 100.0) == pytest.approx(gain_db)
        assert monitor_percent(round(monitor_position(percent))) == pytest.approx(percent, rel=0.06)


# ------------------------------------------------------------------ the window's sliders


@pytest.fixture
def window(qtbot: QtBot, tmp_path: Path):
    def make(saved: prefs.Prefs | None = None, levels=None) -> tuple[MainWindow, PreviewEngine]:
        path = tmp_path / "settings.json"
        prefs.save(path, saved or prefs.Prefs())
        engine = PreviewEngine(ON, MICS, tmp_path, OUTPUTS)
        w = MainWindow(engine, path, FakeLoginItem(), "win32", mic_levels=levels)
        qtbot.addWidget(w)
        return w, engine

    return make


class DbMicLevel:
    def __init__(self, level: float | None, db: float | None) -> None:
        self.value, self.db = level, db

    def level(self, mic: str) -> float | None:
        return self.value

    def level_db(self, mic: str) -> float | None:
        return self.db

    def set_level(self, mic: str, level: float) -> bool:
        return True

    def close(self) -> None:
        pass


def test_the_window_sliders_read_in_db(window) -> None:
    w, engine = window(prefs.Prefs(settings=Settings(loudness=75.0), monitor_volume=50.0), DbMicLevel(0.6, -7.25))
    assert w.call_volume.value_text() == "6.0 dB"
    assert w.monitor_volume.value() == round(monitor_position(50.0))
    assert w.monitor_volume.value_text() == monitor_text(w.monitor_volume.value())
    assert ("monitor_volume", "50.0") in engine.commands  # the saved percent, sent as it was
    w.monitor_volume.set_value(0)
    w.monitor_volume.valueChanged.emit(0)
    assert w.monitor_volume.value_text() == "-inf dB" and engine.monitor_volume == 0.0
    w.monitor_volume.set_value(100)
    w.monitor_volume.valueChanged.emit(100)
    assert w.monitor_volume.value_text() == "0.0 dB" and engine.monitor_volume == 100.0
    w.save_prefs()
    assert prefs.load(w.prefs_path).monitor_volume == 100.0
    # Mic level: the OS's dB beside the OS's own slider position.
    assert w.mic_gain.value() == 60 and w.mic_gain.value_text() == "-7.2 dB"


def test_a_mic_with_no_db_shows_its_percent(window) -> None:
    w, _ = window(levels=DbMicLevel(0.6, None))
    assert w.mic_gain.value_text() == "60 %"
    w.mic_levels.db = -3.0  # type: ignore[attr-defined]
    w.tick()
    assert w.mic_gain.value_text() == "-3.0 dB" and w.mic_gain.accessibleDescription() == "-3.0 dB"
    w.mic_levels.value = None  # type: ignore[attr-defined]
    w.tick()
    assert w.mic_gain.value_text() == "–"


# ------------------------------------------------------------------ Windows: IAudioEndpointVolume's dB


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows endpoint volume")
def test_the_endpoint_db_comes_from_get_master_volume_level_within_its_range() -> None:
    """A fake IAudioEndpointVolume built in ctypes: CaptureVolume reads GetMasterVolumeLevel (vtable 8) and keeps it
    inside GetVolumeRange (vtable 20); it never calls a setter."""
    from clean_voice.endpoints import CaptureVolume

    calls: list[str] = []
    state = {"db": -10.5, "range": (-30.0, 12.0)}
    hresult = ctypes.c_long
    level_proto = ctypes.WINFUNCTYPE(hresult, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float))  # type: ignore[attr-defined]
    range_proto = ctypes.WINFUNCTYPE(  # type: ignore[attr-defined]
        hresult, ctypes.c_void_p, *(ctypes.POINTER(ctypes.c_float),) * 3
    )
    release_proto = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)  # type: ignore[attr-defined]

    def get_level(this, out):
        calls.append("level")
        out[0] = state["db"]
        return 0

    def get_range(this, lo, hi, step):
        calls.append("range")
        lo[0], hi[0] = state["range"]
        step[0] = 0.5
        return 0

    def release(this):
        calls.append("release")
        return 0

    def forbidden(this):
        raise AssertionError("a method other than the three reads was called")

    functions = {2: release_proto(release), 8: level_proto(get_level), 20: range_proto(get_range)}
    fallback = release_proto(forbidden)
    vtable = (ctypes.c_void_p * 21)()
    for i in range(21):
        vtable[i] = ctypes.cast(functions.get(i, fallback), ctypes.c_void_p).value
    obj = (ctypes.c_void_p * 1)(ctypes.addressof(vtable))
    volume = CaptureVolume(None, ctypes.c_void_p(ctypes.addressof(obj)), initialised=False)
    assert volume.get_db() == pytest.approx(-10.5)
    assert volume.volume_range() == (-30.0, 12.0)
    state["db"] = -55.0  # outside its range: kept inside it
    assert volume.get_db() == -30.0
    assert calls.count("range") == 1  # the range is read once
    volume.close()
    assert calls[-1] == "release" and volume.get_db() is None


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows endpoint volume")
def test_the_real_default_mic_db_reads_without_changing_it() -> None:
    """Read only: the default capture endpoint's dB, inside its own range. Never sets the real mic level."""
    sd = pytest.importorskip("sounddevice")
    from clean_voice.endpoints import open_capture_volume

    wasapi = next((api for api in sd.query_hostapis() if api["name"] == "Windows WASAPI"), None)
    if wasapi is None or wasapi["default_input_device"] < 0:
        pytest.skip("no default capture device")
    name = sd.query_devices(wasapi["default_input_device"])["name"]
    volume = open_capture_volume(name)
    assert volume is not None, f"no endpoint volume for {name!a}"
    try:
        before = volume.get()
        span = volume.volume_range()
        db = volume.get_db()
        assert span is not None and db is not None and span[0] <= db <= span[1]
        assert volume.get() == before  # nothing was changed
    finally:
        volume.close()
