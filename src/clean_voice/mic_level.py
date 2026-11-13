"""A microphone's own input level in the OS, for Step 1's Mic level slider (DESIGN "Home gain sliders", J10): the
same control as the system Sound settings, so lowering it stops clipping before Clean Voice gets the sound.

Windows: the capture endpoint's IAudioEndpointVolume (clean_voice.endpoints). macOS: Core Audio's
kAudioDevicePropertyVolumeScalar on the input scope, on the main element or else on each channel. The slider's readout
is the dB the OS gives that level (DESIGN "OBS-style meters and faders"): Windows' GetMasterVolumeLevel, macOS'
kAudioDevicePropertyVolumeDecibels, each kept inside the device's own dB range; a device with none shows its percent.
Everything here runs on the UI thread; the audio never waits on it. The level is the OS's to keep, never saved.
"""

from __future__ import annotations

import ctypes
import logging
import math
import sys
import time
from collections.abc import Callable
from typing import Any, Protocol

log = logging.getLogger(__name__)

RETRY_S = 3.0  # a mic with no level (or one that just vanished) is asked again this often


class MicLevel(Protocol):
    """What the window needs; tests substitute a fake."""

    def level(self, mic: str) -> float | None:
        """``mic``'s level now, 0 to 1, or None when it has none that can be set from here."""
        ...

    def set_level(self, mic: str, level: float) -> bool: ...

    def level_db(self, mic: str) -> float | None:
        """The dB the OS reports for ``mic``'s level, or None when it reports none."""
        ...

    def close(self) -> None: ...


class Endpoint(Protocol):
    """One open mic's level control."""

    def get(self) -> float | None: ...

    def get_db(self) -> float | None: ...

    def set(self, level: float) -> bool: ...

    def close(self) -> None: ...


class SystemMicLevel:
    """Keeps the selected mic's control open between reads, opens it again when the mic changes, and asks again at
    most every RETRY_S for a mic with none (or one that failed)."""

    def __init__(self, opener: Callable[[str], Endpoint | None], clock: Callable[[], float] = time.monotonic) -> None:
        self._open = opener
        self._clock = clock
        self._name: str | None = None
        self._endpoint: Endpoint | None = None
        self._retry_at = 0.0

    def _drop(self) -> None:
        endpoint, self._endpoint = self._endpoint, None
        if endpoint is not None:
            try:
                endpoint.close()
            except Exception:  # a vanished device
                log.debug("closing the mic level failed", exc_info=True)

    def _failed(self) -> None:
        self._drop()
        self._retry_at = self._clock() + RETRY_S

    def _endpoint_for(self, mic: str) -> Endpoint | None:
        if mic != self._name:
            self._drop()
            self._name, self._retry_at = mic, 0.0
        if self._endpoint is None and self._clock() >= self._retry_at:
            try:
                self._endpoint = self._open(mic)
            except Exception:  # never crash the window over a level
                log.debug("opening the mic level failed", exc_info=True)
            if self._endpoint is None:
                self._failed()
        return self._endpoint

    def level(self, mic: str) -> float | None:
        endpoint = self._endpoint_for(mic)
        if endpoint is None:
            return None
        value = endpoint.get()
        if value is None:
            self._failed()
        return value

    def set_level(self, mic: str, level: float) -> bool:
        endpoint = self._endpoint_for(mic)
        return endpoint is not None and endpoint.set(min(1.0, max(0.0, level)))

    def level_db(self, mic: str) -> float | None:
        """The open control's dB, never opening one: ``level`` (read first on every tick) keeps it open."""
        endpoint = self._endpoint if mic == self._name else None
        if endpoint is None:
            return None
        try:
            return endpoint.get_db()
        except Exception:  # a vanished device: the next level() read notices it
            log.debug("reading the mic's dB failed", exc_info=True)
            return None

    def close(self) -> None:
        self._drop()


class NoMicLevel:
    """A platform with no mic level control."""

    def level(self, mic: str) -> float | None:
        return None

    def set_level(self, mic: str, level: float) -> bool:
        return False

    def level_db(self, mic: str) -> float | None:
        return None

    def close(self) -> None:
        pass


def system() -> MicLevel:
    """This computer's mic level control."""
    if sys.platform == "win32":
        from clean_voice.endpoints import open_capture_volume

        return SystemMicLevel(open_capture_volume)
    if sys.platform == "darwin":
        return SystemMicLevel(open_core_audio_volume)
    return NoMicLevel()


# ------------------------------------------------------------------ macOS: Core Audio


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


_SYSTEM_OBJECT = 1  # kAudioObjectSystemObject
_DEVICES = _fourcc("dev#")  # kAudioHardwarePropertyDevices
_NAME = _fourcc("lnam")  # kAudioObjectPropertyName, the name PortAudio lists the device under
_STREAMS = _fourcc("stm#")  # kAudioDevicePropertyStreams
_VOLUME = _fourcc("volm")  # kAudioDevicePropertyVolumeScalar
_DECIBELS = _fourcc("vold")  # kAudioDevicePropertyVolumeDecibels
_DECIBEL_RANGE = _fourcc("vdb#")  # kAudioDevicePropertyVolumeRangeDecibels, an AudioValueRange
_GLOBAL = _fourcc("glob")  # kAudioObjectPropertyScopeGlobal
_INPUT = _fourcc("inpt")  # kAudioDevicePropertyScopeInput
_MAIN = 0  # kAudioObjectPropertyElementMain
_MAX_CHANNEL = 32  # the per-channel elements looked for when the main element has no volume
_UTF8 = 0x08000100  # kCFStringEncodingUTF8


class _Address(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]


class _ValueRange(ctypes.Structure):  # AudioValueRange
    _fields_ = [("minimum", ctypes.c_double), ("maximum", ctypes.c_double)]


_libs: tuple[Any, Any] | None = None


def _core_audio() -> tuple[Any, Any]:
    global _libs
    if _libs is None:
        ca = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        addr = ctypes.POINTER(_Address)
        u32p = ctypes.POINTER(ctypes.c_uint32)
        ca.AudioObjectHasProperty.argtypes = [ctypes.c_uint32, addr]
        ca.AudioObjectHasProperty.restype = ctypes.c_ubyte
        ca.AudioObjectIsPropertySettable.argtypes = [ctypes.c_uint32, addr, ctypes.POINTER(ctypes.c_ubyte)]
        ca.AudioObjectIsPropertySettable.restype = ctypes.c_int32
        ca.AudioObjectGetPropertyDataSize.argtypes = [ctypes.c_uint32, addr, ctypes.c_uint32, ctypes.c_void_p, u32p]
        ca.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
        ca.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32,
            addr,
            ctypes.c_uint32,
            ctypes.c_void_p,
            u32p,
            ctypes.c_void_p,
        ]
        ca.AudioObjectGetPropertyData.restype = ctypes.c_int32
        ca.AudioObjectSetPropertyData.argtypes = [
            ctypes.c_uint32,
            addr,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        ca.AudioObjectSetPropertyData.restype = ctypes.c_int32
        cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        cf.CFStringGetCString.restype = ctypes.c_ubyte
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        cf.CFRelease.restype = None
        _libs = (ca, cf)
    return _libs


def _size(ca: Any, obj: int, address: _Address) -> int:
    size = ctypes.c_uint32()
    if ca.AudioObjectGetPropertyDataSize(obj, ctypes.byref(address), 0, None, ctypes.byref(size)) != 0:
        return 0
    return size.value


def _device_name(ca: Any, cf: Any, device: int) -> str | None:
    ref = ctypes.c_void_p()
    size = ctypes.c_uint32(ctypes.sizeof(ref))
    address = _Address(_NAME, _GLOBAL, _MAIN)
    if ca.AudioObjectGetPropertyData(device, ctypes.byref(address), 0, None, ctypes.byref(size), ctypes.byref(ref)):
        return None
    if not ref.value:
        return None
    try:
        buf = ctypes.create_string_buffer(1024)
        return buf.value.decode("utf-8") if cf.CFStringGetCString(ref, buf, len(buf), _UTF8) else None
    finally:
        cf.CFRelease(ref)


def _settable(ca: Any, device: int, address: _Address) -> bool:
    if not ca.AudioObjectHasProperty(device, ctypes.byref(address)):
        return False
    flag = ctypes.c_ubyte()
    return ca.AudioObjectIsPropertySettable(device, ctypes.byref(address), ctypes.byref(flag)) == 0 and bool(flag.value)


class CoreAudioVolume:
    """An input device's volume scalar on its main element, or on each of its channels together."""

    def __init__(self, ca: Any, device: int, elements: tuple[int, ...]) -> None:
        self._ca = ca
        self._device = device
        self._elements = elements

    def get(self) -> float | None:
        values = []
        for element in self._elements:
            value = ctypes.c_float()
            size = ctypes.c_uint32(ctypes.sizeof(value))
            address = _Address(_VOLUME, _INPUT, element)
            if self._ca.AudioObjectGetPropertyData(
                self._device, ctypes.byref(address), 0, None, ctypes.byref(size), ctypes.byref(value)
            ):
                return None
            values.append(float(value.value))
        return min(1.0, max(0.0, sum(values) / len(values))) if values else None

    def _read(self, selector: int, element: int, value: Any) -> bool:
        size = ctypes.c_uint32(ctypes.sizeof(value))
        address = _Address(selector, _INPUT, element)
        return not self._ca.AudioObjectGetPropertyData(
            self._device, ctypes.byref(address), 0, None, ctypes.byref(size), ctypes.byref(value)
        )

    def get_db(self) -> float | None:
        """kAudioDevicePropertyVolumeDecibels, averaged over the elements as ``get`` does, inside the device's
        kAudioDevicePropertyVolumeRangeDecibels; None when the device reports no dB."""
        values = []
        lo, hi = math.inf, -math.inf
        for element in self._elements:
            value, span = ctypes.c_float(), _ValueRange()
            if not (self._read(_DECIBELS, element, value) and self._read(_DECIBEL_RANGE, element, span)):
                return None
            if not (math.isfinite(value.value) and span.minimum < span.maximum):
                return None
            values.append(float(value.value))
            lo, hi = min(lo, span.minimum), max(hi, span.maximum)
        return min(hi, max(lo, sum(values) / len(values))) if values else None

    def set(self, level: float) -> bool:
        value = ctypes.c_float(min(1.0, max(0.0, level)))
        ok = True
        for element in self._elements:
            address = _Address(_VOLUME, _INPUT, element)
            status = self._ca.AudioObjectSetPropertyData(
                self._device, ctypes.byref(address), 0, None, ctypes.sizeof(value), ctypes.byref(value)
            )
            ok = ok and status == 0
        return ok

    def close(self) -> None:
        pass  # nothing is held: the device is addressed by its id


def open_core_audio_volume(name: str) -> CoreAudioVolume | None:
    """The input volume of the Core Audio device named ``name`` that has inputs, or None: not macOS, no such device,
    or no settable volume on its input."""
    if sys.platform != "darwin":
        return None
    try:
        ca, cf = _core_audio()
        devices_at = _Address(_DEVICES, _GLOBAL, _MAIN)
        size = _size(ca, _SYSTEM_OBJECT, devices_at)
        count = size // ctypes.sizeof(ctypes.c_uint32)
        if count == 0:
            return None
        ids = (ctypes.c_uint32 * count)()
        got = ctypes.c_uint32(size)
        if ca.AudioObjectGetPropertyData(_SYSTEM_OBJECT, ctypes.byref(devices_at), 0, None, ctypes.byref(got), ids):
            return None
        for device in ids:
            if _size(ca, device, _Address(_STREAMS, _INPUT, _MAIN)) == 0 or _device_name(ca, cf, device) != name:
                continue
            if _settable(ca, device, _Address(_VOLUME, _INPUT, _MAIN)):
                return CoreAudioVolume(ca, device, (_MAIN,))
            channels = tuple(
                e for e in range(1, _MAX_CHANNEL + 1) if _settable(ca, device, _Address(_VOLUME, _INPUT, e))
            )
            return CoreAudioVolume(ca, device, channels) if channels else None
        return None
    except OSError:
        log.debug("opening the mic's Core Audio volume failed", exc_info=True)
        return None
