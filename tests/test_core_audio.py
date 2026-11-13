"""Step 1's Mic level on macOS: mic_level's Core Audio path. The logic runs on every OS against a fake Core Audio that
answers as Apple's C API does (AudioHardware.h); on a Mac the real default input is read, never set."""

from __future__ import annotations

import ctypes
import sys
import types
from typing import Any

import pytest

from clean_voice import mic_level as ml

WHO = 0x77686F3F  # kAudioHardwareUnknownPropertyError, 'who?'


class FakeCoreAudio:
    """Both frameworks in one: devices by id, each {"name", "inputs", "volumes": {element: scalar}, "settable"}."""

    def __init__(self, devices: dict[int, dict[str, Any]]) -> None:
        self.devices = devices
        self.strings: dict[int, str] = {}
        self.names_read: list[int] = []
        self.released: list[int] = []
        self.sets: list[tuple[int, int, float]] = []

    def AudioObjectGetPropertyDataSize(self, obj, addr, qs, q, size) -> int:
        a = addr._obj
        if obj == ml._SYSTEM_OBJECT and a.selector == ml._DEVICES:
            size._obj.value = 4 * len(self.devices)
            return 0
        if a.selector == ml._STREAMS and a.scope == ml._INPUT:
            size._obj.value = 4 * self.devices[obj]["inputs"]
            return 0
        return WHO

    def AudioObjectGetPropertyData(self, obj, addr, qs, q, size, out) -> int:
        a = addr._obj
        if obj == ml._SYSTEM_OBJECT and a.selector == ml._DEVICES:
            assert a.scope == ml._GLOBAL and a.element == ml._MAIN
            assert size._obj.value == 4 * len(self.devices)
            for i, device in enumerate(self.devices):
                out[i] = device
            return 0
        device = self.devices[obj]
        if a.selector == ml._NAME:
            assert size._obj.value == ctypes.sizeof(ctypes.c_void_p)
            ref = 1000 + obj  # a new retained CFString, which the caller must release
            self.strings[ref] = device["name"]
            self.names_read.append(obj)
            out._obj.value = ref
            return 0
        if a.selector == ml._VOLUME and a.scope == ml._INPUT and a.element in device["volumes"]:
            assert size._obj.value == ctypes.sizeof(ctypes.c_float)
            out._obj.value = device["volumes"][a.element]
            return 0
        if a.selector == ml._DECIBELS and a.scope == ml._INPUT and a.element in device.get("decibels", {}):
            assert size._obj.value == ctypes.sizeof(ctypes.c_float)
            out._obj.value = device["decibels"][a.element]
            return 0
        if a.selector == ml._DECIBEL_RANGE and a.scope == ml._INPUT and "db_range" in device:
            assert size._obj.value == ctypes.sizeof(ml._ValueRange)
            out._obj.minimum, out._obj.maximum = device["db_range"]
            return 0
        return WHO

    def AudioObjectHasProperty(self, obj, addr) -> int:
        a = addr._obj
        return int(a.selector == ml._VOLUME and a.scope == ml._INPUT and a.element in self.devices[obj]["volumes"])

    def AudioObjectIsPropertySettable(self, obj, addr, flag) -> int:
        flag._obj.value = int(addr._obj.element in self.devices[obj]["settable"])
        return 0

    def AudioObjectSetPropertyData(self, obj, addr, qs, q, size, data) -> int:
        a = addr._obj
        assert size == ctypes.sizeof(ctypes.c_float)
        self.devices[obj]["volumes"][a.element] = data._obj.value
        self.sets.append((obj, a.element, data._obj.value))
        return 0

    def CFStringGetCString(self, ref, buf, n, encoding) -> int:
        assert encoding == ml._UTF8
        raw = self.strings[ref.value].encode("utf-8") + b"\0"
        if len(raw) > n:
            return 0
        ctypes.memmove(buf, raw, len(raw))
        return 1

    def CFRelease(self, ref) -> None:
        self.released.append(ref.value)


@pytest.fixture
def core_audio(monkeypatch: pytest.MonkeyPatch):
    """Put a fake Core Audio behind mic_level, as if on a Mac. It never reaches this computer's own sound system."""

    def make(devices: dict[int, dict[str, Any]]) -> FakeCoreAudio:
        fake = FakeCoreAudio(devices)
        monkeypatch.setattr(ml, "_core_audio", lambda: (fake, fake))
        monkeypatch.setattr(ml, "sys", types.SimpleNamespace(platform="darwin"))
        return fake

    return make


def test_the_main_element_volume_is_read_and_set(core_audio) -> None:
    fake = core_audio(
        {
            40: {"name": "MacBook Pro Speakers", "inputs": 0, "volumes": {0: 0.5}, "settable": {0}},
            41: {"name": "MacBook Pro Microphone", "inputs": 1, "volumes": {0: 0.62}, "settable": {0}},
        }
    )
    volume = ml.open_core_audio_volume("MacBook Pro Microphone")
    assert volume is not None and volume.get() == pytest.approx(0.62)
    assert volume.set(1.4) and fake.sets == [(41, 0, pytest.approx(1.0))]  # clamped
    assert fake.names_read == [41]  # the output-only device is skipped before its name is read
    assert fake.released == [1041]  # and the name's CFString released once


def test_a_mic_with_no_main_volume_uses_each_channel(core_audio) -> None:
    fake = core_audio({7: {"name": "マイク USB", "inputs": 2, "volumes": {1: 0.4, 2: 0.6}, "settable": {1, 2}}})
    volume = ml.open_core_audio_volume("マイク USB")  # a non-ASCII name, as PortAudio lists it
    assert volume is not None and volume.get() == pytest.approx(0.5)
    assert volume.set(0.8) and [(e, v) for _, e, v in fake.sets] == [(1, pytest.approx(0.8)), (2, pytest.approx(0.8))]
    assert fake.released == [1007]


def test_a_main_volume_that_cannot_be_set_falls_back_to_the_channels(core_audio) -> None:
    core_audio({7: {"name": "Interface", "inputs": 1, "volumes": {0: 1.0, 1: 0.3}, "settable": {1}}})
    volume = ml.open_core_audio_volume("Interface")
    assert volume is not None and volume.get() == pytest.approx(0.3)


@pytest.mark.parametrize(
    "device",
    [
        {"name": "Fixed", "inputs": 1, "volumes": {0: 1.0}, "settable": set()},  # a level it will not let us set
        {"name": "Fixed", "inputs": 1, "volumes": {}, "settable": set()},  # no volume at all
    ],
    ids=["not-settable", "no-volume"],
)
def test_a_mic_with_no_level_that_can_be_set_has_none(core_audio, device: dict[str, Any]) -> None:
    fake = core_audio({7: device})
    assert ml.open_core_audio_volume("Fixed") is None
    assert fake.released == [1007]


def test_a_name_that_matches_nothing_has_none_and_releases_every_name(core_audio) -> None:
    fake = core_audio(
        {
            3: {"name": "One", "inputs": 1, "volumes": {0: 0.5}, "settable": {0}},
            4: {"name": "Two", "inputs": 1, "volumes": {0: 0.5}, "settable": {0}},
        }
    )
    assert ml.open_core_audio_volume("Three") is None
    assert fake.released == [1003, 1004] and fake.sets == []


def test_a_volume_read_that_fails_reads_none(core_audio) -> None:
    fake = core_audio({7: {"name": "USB", "inputs": 1, "volumes": {0: 0.5}, "settable": {0}}})
    volume = ml.open_core_audio_volume("USB")
    assert volume is not None
    del fake.devices[7]["volumes"][0]  # unplugged: Core Audio answers with an error
    assert volume.get() is None


def test_the_level_in_db_is_the_devices_own(core_audio) -> None:
    # DESIGN "OBS-style meters and faders": Mic level reads kAudioDevicePropertyVolumeDecibels, inside the device's
    # kAudioDevicePropertyVolumeRangeDecibels, averaged over the channels as the scalar is.
    core_audio(
        {
            7: {
                "name": "USB",
                "inputs": 2,
                "volumes": {1: 0.4, 2: 0.6},
                "settable": {1, 2},
                "decibels": {1: -12.0, 2: -6.0},
                "db_range": (-40.0, 12.0),
            }
        }
    )
    volume = ml.open_core_audio_volume("USB")
    assert volume is not None and volume.get_db() == pytest.approx(-9.0)


def test_a_device_with_no_db_reads_none_so_the_slider_shows_its_percent(core_audio) -> None:
    fake = core_audio({7: {"name": "USB", "inputs": 1, "volumes": {0: 0.5}, "settable": {0}}})
    volume = ml.open_core_audio_volume("USB")
    assert volume is not None and volume.get_db() is None
    fake.devices[7].update(decibels={0: -3.0}, db_range=(0.0, 0.0))  # an empty range: no usable dB either
    assert volume.get_db() is None
    fake.devices[7].update(db_range=(-30.0, -10.0))  # a dB outside its own range is kept inside it
    assert volume.get_db() == -10.0


def test_not_on_a_mac_there_is_no_core_audio_volume() -> None:
    if sys.platform == "darwin":
        pytest.skip("this is a Mac")
    assert ml.open_core_audio_volume("anything") is None


@pytest.mark.skipif(sys.platform != "darwin", reason="the macOS Core Audio volume")
def test_the_real_default_mic_level_reads_without_changing_it() -> None:
    """The default Core Audio input, found by the name PortAudio lists it under: its level reads, and is left as it
    is. Never sets."""
    assert ml.open_core_audio_volume("no such mic") is None  # the frameworks load, every device's name is read
    sd = pytest.importorskip("sounddevice")
    core = next((api for api in sd.query_hostapis() if api["name"] == "Core Audio"), None)
    if core is None or core["default_input_device"] < 0:
        pytest.skip("no default input device")
    name: str = sd.query_devices(core["default_input_device"])["name"]
    ca, cf = ml._core_audio()
    count = ml._size(ca, ml._SYSTEM_OBJECT, ml._Address(ml._DEVICES, ml._GLOBAL, ml._MAIN)) // 4
    ids = (ctypes.c_uint32 * count)()
    got = ctypes.c_uint32(4 * count)
    address = ml._Address(ml._DEVICES, ml._GLOBAL, ml._MAIN)
    assert ca.AudioObjectGetPropertyData(ml._SYSTEM_OBJECT, ctypes.byref(address), 0, None, ctypes.byref(got), ids) == 0
    names = [ml._device_name(ca, cf, device) for device in ids]
    assert name in names, f"{name!a} not among Core Audio's devices {names!a}"
    volume = ml.open_core_audio_volume(name)
    if volume is None:  # a mic whose level the OS keeps fixed: the slider is greyed, nothing more to read
        return
    try:
        level = volume.get()
        assert level is not None and 0.0 <= level <= 1.0
        db = volume.get_db()  # read only; a device may report none
        assert db is None or -200.0 < db < 100.0
    finally:
        volume.close()
