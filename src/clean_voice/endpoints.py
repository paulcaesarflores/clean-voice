"""The Windows default speakers' name, and a mic's own level, asked of WASAPI directly, without listing every device.

OBS's WASAPISource::InitDevice asks the device enumerator for just the default endpoint,
IMMDeviceEnumerator::GetDefaultAudioEndpoint(eRender, eConsole) (plugins/win-wasapi/win-wasapi.cpp:607-618). Listing
devices again means a whole PortAudio initialisation, which enumerates every endpoint and can disturb USB and
Bluetooth drivers while audio streams (docs/simple-clean-mic-audio-report.md, cause 5). This asks for one endpoint and
reads its friendly name, which is the name PortAudio lists it under, through raw COM calls (no extra package).

Step 1's Mic level slider reads and sets a microphone's own Windows level the same way: the capture endpoint with the
mic's friendly name, and its IAudioEndpointVolume master volume; its readout is the dB Windows reports for that level
(GetMasterVolumeLevel, within GetVolumeRange; DESIGN "OBS-style meters and faders").
"""

from __future__ import annotations

import ctypes
import logging
import math
import sys
from typing import Any

log = logging.getLogger(__name__)

_E_RENDER = 0  # EDataFlow eRender
_E_CONSOLE = 0  # ERole eConsole (as OBS asks for render devices, win-wasapi.cpp:614-615)
_CLSCTX_ALL = 0x17
_STGM_READ = 0
_VT_LPWSTR = 31
_RPC_E_CHANGED_MODE = -2147417850  # 0x80010106: COM already set up in another mode on this thread


class _GUID(ctypes.Structure):
    _fields_ = [
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def of(cls, text: str) -> _GUID:
        h = text.strip("{}").replace("-", "")
        return cls(int(h[0:8], 16), int(h[8:12], 16), int(h[12:16], 16), (ctypes.c_ubyte * 8)(*bytes.fromhex(h[16:])))


class _PropertyKey(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", ctypes.c_uint32)]


class _PropVariant(ctypes.Structure):
    # vt, three reserved words, then the value union (24 bytes in all on 64-bit Windows).
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("reserved", ctypes.c_ushort * 3),
        ("value", ctypes.c_void_p),
        ("pad", ctypes.c_void_p),
    ]


_CLSID_ENUMERATOR = _GUID.of("BCDE0395-E52F-467C-8E3D-C4579291692E")  # MMDeviceEnumerator
_IID_ENUMERATOR = _GUID.of("A95664D2-9614-4F35-A746-DE8DB63617E6")  # IMMDeviceEnumerator
_PKEY_FRIENDLY_NAME = _PropertyKey(_GUID.of("A45C254E-DF1C-4EFD-8020-67D146A850E0"), 14)  # PKEY_Device_FriendlyName


def _method(obj: ctypes.c_void_p, index: int, *argtypes: Any) -> Any:
    """The ``index``-th method of a COM object's vtable, as a callable taking the object first."""
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    proto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)  # type: ignore[attr-defined]
    return proto(vtable[index])


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _method(obj, 2)(obj)  # IUnknown::Release


def _com_init(ole32: Any) -> bool | None:
    """Set COM up on this thread: True when the call must be balanced by CoUninitialize, False when COM was already
    set up in another mode (usable as it is), None when it failed."""
    hr = ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED, as the device and UI threads already are
    if hr in (0, 1):  # S_OK, S_FALSE: this call must be balanced
        return True
    if hr == _RPC_E_CHANGED_MODE:
        return False
    log.debug("COM initialisation failed: 0x%08x", hr & 0xFFFFFFFF)
    return None


def _enumerator(ole32: Any) -> ctypes.c_void_p | None:
    enumerator = ctypes.c_void_p()
    hr = ole32.CoCreateInstance(
        ctypes.byref(_CLSID_ENUMERATOR), None, _CLSCTX_ALL, ctypes.byref(_IID_ENUMERATOR), ctypes.byref(enumerator)
    )
    return enumerator if hr == 0 else None


def _friendly_name(ole32: Any, device: ctypes.c_void_p) -> str | None:
    """An endpoint's PKEY_Device_FriendlyName: the name PortAudio lists it under."""
    store = ctypes.c_void_p()
    try:
        open_store = _method(device, 4, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
        if open_store(device, _STGM_READ, ctypes.byref(store)) != 0:  # IMMDevice::OpenPropertyStore
            return None
        value = _PropVariant()
        get_value = _method(store, 5, ctypes.POINTER(_PropertyKey), ctypes.POINTER(_PropVariant))
        if get_value(store, ctypes.byref(_PKEY_FRIENDLY_NAME), ctypes.byref(value)) != 0:  # IPropertyStore::GetValue
            return None
        try:
            return ctypes.wstring_at(value.value) if value.vt == _VT_LPWSTR and value.value else None
        finally:
            ole32.PropVariantClear(ctypes.byref(value))
    finally:
        _release(store)


def default_render_name() -> str | None:
    """The friendly name of Windows' default output now, or None (no default output, or not Windows)."""
    if sys.platform != "win32":
        return None
    ole32 = ctypes.WinDLL("ole32")  # type: ignore[attr-defined]
    initialised = _com_init(ole32)
    if initialised is None:
        return None
    enumerator, device = None, ctypes.c_void_p()
    try:
        enumerator = _enumerator(ole32)
        if enumerator is None:
            return None
        get_default = _method(enumerator, 4, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))
        if get_default(enumerator, _E_RENDER, _E_CONSOLE, ctypes.byref(device)) != 0:  # GetDefaultAudioEndpoint
            return None  # E_NOTFOUND: no output device at all
        return _friendly_name(ole32, device)
    except OSError:
        log.debug("asking WASAPI for the default output failed", exc_info=True)
        return None
    finally:
        for obj in (device, enumerator):
            if obj is not None:
                _release(obj)
        if initialised:
            ole32.CoUninitialize()


# ------------------------------------------------------------------ a microphone's own level (Home's mic slider)

_E_CAPTURE = 1  # EDataFlow eCapture
_DEVICE_STATE_ACTIVE = 1
_IID_ENDPOINT_VOLUME = _GUID.of("5CDF2C82-841E-4546-9722-0CF74078229A")  # IAudioEndpointVolume


# IAudioEndpointVolume's vtable, after IUnknown's three: RegisterControlChangeNotify 3, UnregisterControlChangeNotify 4,
# GetChannelCount 5, SetMasterVolumeLevel 6, SetMasterVolumeLevelScalar 7, GetMasterVolumeLevel 8,
# GetMasterVolumeLevelScalar 9, ... GetVolumeRange 20 (endpointvolume.h).
_SET_SCALAR, _GET_LEVEL_DB, _GET_SCALAR, _GET_RANGE = 7, 8, 9, 20


class CaptureVolume:
    """One capture endpoint's master volume (IAudioEndpointVolume, the level slider in Windows' Sound settings), as a
    scalar from 0 to 1, and the dB Windows gives that level. Used on the UI thread only; ``close`` releases it."""

    def __init__(self, ole32: Any, volume: ctypes.c_void_p, initialised: bool) -> None:
        self._ole32 = ole32
        self._volume: ctypes.c_void_p | None = volume
        self._initialised = initialised
        self._range: tuple[float, float] | None = None  # GetVolumeRange's minimum and maximum, read once

    def get(self) -> float | None:
        """The level now, or None once the endpoint is gone (unplugged, disabled)."""
        if self._volume is None:
            return None
        value = ctypes.c_float()
        try:
            get_scalar = _method(self._volume, _GET_SCALAR, ctypes.POINTER(ctypes.c_float))
            hr = get_scalar(self._volume, ctypes.byref(value))
        except OSError:
            return None
        return min(1.0, max(0.0, float(value.value))) if hr == 0 else None

    def set(self, level: float) -> bool:
        if self._volume is None:
            return False
        try:
            set_scalar = _method(self._volume, _SET_SCALAR, ctypes.c_float, ctypes.c_void_p)
            return set_scalar(self._volume, min(1.0, max(0.0, level)), None) == 0
        except OSError:
            return False

    def volume_range(self) -> tuple[float, float] | None:
        """The endpoint's dB range (GetVolumeRange), or None when it gives none that makes sense."""
        if self._range is None and self._volume is not None:
            lo, hi, step = ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
            floats = (ctypes.POINTER(ctypes.c_float),) * 3
            try:
                hr = _method(self._volume, _GET_RANGE, *floats)(
                    self._volume, ctypes.byref(lo), ctypes.byref(hi), ctypes.byref(step)
                )
            except OSError:
                return None
            if hr == 0 and math.isfinite(lo.value) and math.isfinite(hi.value) and lo.value < hi.value:
                self._range = (float(lo.value), float(hi.value))
        return self._range

    def get_db(self) -> float | None:
        """The level in dB as Windows reports it (GetMasterVolumeLevel), kept inside its range; None once the
        endpoint is gone or when it reports no usable dB (the slider then shows its percent)."""
        span = self.volume_range()
        if self._volume is None or span is None:
            return None
        value = ctypes.c_float()
        try:
            hr = _method(self._volume, _GET_LEVEL_DB, ctypes.POINTER(ctypes.c_float))(self._volume, ctypes.byref(value))
        except OSError:
            return None
        if hr != 0 or not math.isfinite(value.value):
            return None
        return min(span[1], max(span[0], float(value.value)))

    def close(self) -> None:
        volume, self._volume = self._volume, None
        if volume is not None:
            _release(volume)
            if self._initialised:
                self._ole32.CoUninitialize()


def open_capture_volume(name: str) -> CaptureVolume | None:
    """The level of the active capture endpoint whose friendly name is ``name`` (the name PortAudio lists the mic
    under), or None: not Windows, no such endpoint, or no volume control on it."""
    if sys.platform != "win32":
        return None
    ole32 = ctypes.WinDLL("ole32")  # type: ignore[attr-defined]
    initialised = _com_init(ole32)
    if initialised is None:
        return None
    enumerator, collection, volume = None, ctypes.c_void_p(), ctypes.c_void_p()
    try:
        enumerator = _enumerator(ole32)
        if enumerator is None:
            return None
        enum_endpoints = _method(enumerator, 3, ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
        if enum_endpoints(enumerator, _E_CAPTURE, _DEVICE_STATE_ACTIVE, ctypes.byref(collection)) != 0:
            return None
        count = ctypes.c_uint32()
        if _method(collection, 3, ctypes.POINTER(ctypes.c_uint32))(collection, ctypes.byref(count)) != 0:  # GetCount
            return None
        item = _method(collection, 4, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))  # IMMDeviceCollection::Item
        for i in range(count.value):
            device = ctypes.c_void_p()
            if item(collection, i, ctypes.byref(device)) != 0:
                continue
            try:
                if _friendly_name(ole32, device) != name:
                    continue
                activate = _method(  # IMMDevice::Activate
                    device, 3, ctypes.POINTER(_GUID), ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
                )
                if activate(device, ctypes.byref(_IID_ENDPOINT_VOLUME), _CLSCTX_ALL, None, ctypes.byref(volume)) != 0:
                    return None
                found = CaptureVolume(ole32, volume, initialised)
                initialised = False  # the CaptureVolume balances it on close
                volume = ctypes.c_void_p()  # and owns the reference
                return found
            finally:
                _release(device)
        return None
    except OSError:
        log.debug("opening the mic's endpoint volume failed", exc_info=True)
        return None
    finally:
        for obj in (volume, collection, enumerator):
            if obj is not None:
                _release(obj)
        if initialised:
            ole32.CoUninitialize()
