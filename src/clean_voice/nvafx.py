"""ctypes binding to NVIDIA's Audio Effects SDK runtime, and one effect on it (J12), as OBS Studio's NVIDIA Audio Effects
filter uses it. OBS is GPL and Clean Voice is MIT: this follows the approach in our own Python, no OBS code is copied.

Follows ``reference/obs-studio/plugins/nv-filters/``: ``nvafx-load.h`` (where the runtime is, how it is loaded),
``nv_sdk_versions.h`` (the version numbers) and ``nvidia-audiofx-filter.c`` (the effect). Windows only: on macOS, or
without the runtime, ``runtime()`` is None and the filter is not offered (``nv-filters.c`` obs_module_load :23-28
registers the filter only if ``load_nvidia_afx`` succeeded).

The runtime is NVIDIA's, installed by the guest; nothing of it is bundled. Loading it and loading a model are slow
(OBS: "about 2 seconds", :322), so both happen on a thread of their own, never in the audio path (:102-107, :465).
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

log = logging.getLogger("clean_voice.nvafx")

Frame = npt.NDArray[np.float32]


def pack_version(major: int, minor: int, build: int, revision: int) -> int:
    """nv_sdk_versions.h: major << 24 | minor << 16 | build << 8 | revision."""
    return major << 24 | minor << 16 | build << 8 | revision


MIN_SDK_VERSION = pack_version(1, 6, 1, 2)  # nv_sdk_versions.h MIN_AFX_SDK_VERSION :5
MIN_LOGGER_VERSION = pack_version(1, 6, 1, 2)  # MIN_AFX_LOGGER_VERSION :6, which also gates the VAD option
MIN_ARM_VERSION = pack_version(3, 0, 0, 49)  # MIN_ARM_AFX_SDK_VERSION :7: the per-effect DLLs and renamed parameters

# nvafx-load.h :20-27
DENOISER = "denoiser"
DEREVERB = "dereverb"
DEREVERB_DENOISER = "dereverb_denoiser"
MODELS = {
    DENOISER: "\\models\\denoiser_48k.trtpkg",
    DEREVERB: "\\models\\dereverb_48k.trtpkg",
    DEREVERB_DENOISER: "\\models\\dereverb_denoiser_48k.trtpkg",
}

# nvafx-load.h :36-51
PARAM_MODEL_PATH = b"model_path"
PARAM_INPUT_SAMPLE_RATE = b"input_sample_rate"
PARAM_NUM_INPUT_CHANNELS = b"num_input_channels"
PARAM_INTENSITY_RATIO = b"intensity_ratio"
PARAM_ENABLE_VAD = b"enable_vad"
PARAM_SAMPLES_PER_FRAME = b"num_input_samples_per_frame"  # before SDK 3.0.0.49
PARAM_SAMPLES_PER_FRAME_ARM = b"num_samples_per_input_frame"  # from SDK 3.0.0.49

SAMPLE_RATE = 48_000  # nvidia-audiofx-filter.c NVAFX_SAMPLE_RATE :44: the SDK's, "can't be changed"
FRAME_SIZE = 480  # NVAFX_FRAME_SIZE :46: 10 ms

STATUS_SUCCESS = 0  # nvafx-load.h NvAFX_Status :75
STATUS_FAILED = 1
STATUS_GPU_UNSUPPORTED = 11

MAX_PATH = 260  # the buffer OBS finds the folder in (nvafx_get_sdk_path callers :263, :299)

# LoadLibraryExA's flags, nvafx-load.h load_afx_module :293
LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000

LOG_LEVEL_ERROR = 1  # NVAFX_LOG_LEVEL_ERROR, nvafx-load.h :129
LOG_TARGET_STDERR = 0  # LOG_TARGET_STDERR :138

_SYMBOLS = (  # every symbol OBS loads (LOAD_SYM, nvidia-audiofx-filter.c :217-230); one missing unloads everything
    "NvAFX_CreateEffect",
    "NvAFX_DestroyEffect",
    "NvAFX_SetU32",
    "NvAFX_SetString",
    "NvAFX_SetFloat",
    "NvAFX_GetU32",
    "NvAFX_GetString",
    "NvAFX_GetFloat",
    "NvAFX_Load",
    "NvAFX_GetSupportedDevices",
    "NvAFX_Run",
    "NvAFX_Reset",
    "NvAFX_InitializeLogger",
    "NvAFX_UninitializeLogger",
)


# ------------------------------------------------------------------ where the runtime is


def sdk_dir(environ: Mapping[str, str] | None = None) -> str:
    """The runtime's folder, as nvafx_get_sdk_path :229-251 finds it: the environment variable NVAFX_SDK_DIR, else
    AFX_SDK_DIR (each used only if set, non-empty and shorter than the 260-character buffer), else
    ``%ProgramFiles%\\NVIDIA Corporation\\NVIDIA Audio Effects SDK``. "" when even ProgramFiles is not set."""
    env = os.environ if environ is None else environ
    for name in ("NVAFX_SDK_DIR", "AFX_SDK_DIR"):
        value = env.get(name, "")
        if value and len(value) < MAX_PATH - 1:
            return value
    program_files = env.get("ProgramFiles", "")
    if not program_files:
        return ""
    return f"{program_files}\\NVIDIA Corporation\\NVIDIA Audio Effects SDK"


def dll_version(path: str) -> int:
    """The file version of a DLL as nv_sdk_versions.h packs it, 0 when it can't be read (libobs/util/platform-windows.c
    get_dll_ver :920-963, used by get_lib_version :253-279)."""
    if sys.platform != "win32":
        return 0
    from ctypes import wintypes

    version = ctypes.WinDLL("version", use_last_error=True)
    size_of = version.GetFileVersionInfoSizeW
    size_of.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    size_of.restype = wintypes.DWORD
    get_info = version.GetFileVersionInfoW
    get_info.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    get_info.restype = wintypes.BOOL
    query = version.VerQueryValueW
    query.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.UINT)]
    query.restype = wintypes.BOOL
    size = size_of(path, None)
    if not size:
        return 0
    data = ctypes.create_string_buffer(size)
    if not get_info(path, 0, size, data):
        return 0
    fixed, length = ctypes.c_void_p(), wintypes.UINT()
    if not query(data, "\\", ctypes.byref(fixed), ctypes.byref(length)) or not fixed.value or not length.value:
        return 0
    words = (wintypes.DWORD * 13).from_address(fixed.value)  # VS_FIXEDFILEINFO: dwFileVersionMS, LS are words 2 and 3
    ms, ls = words[2], words[3]
    return pack_version(ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)


def version_text(version: int) -> str:
    return f"{version >> 24 & 0xFF}.{version >> 16 & 0xFF}.{version >> 8 & 0xFF}.{version & 0xFF}"


# ------------------------------------------------------------------ the runtime's C API


class Api:
    """The 14 NVAFX functions OBS loads, over a loaded ``NVAudioEffects.dll``. Each returns the SDK's status (0 is
    success); a value comes back with it. ``channel_pointers`` and ``run`` take the frame OBS gives to NvAFX_Run
    (:658): one pointer to one channel's samples, read and written in place."""

    def __init__(self, lib: Any) -> None:
        self._lib = lib
        c = ctypes
        sig: dict[str, tuple[Any, list[Any]]] = {
            "NvAFX_CreateEffect": (c.c_int, [c.c_char_p, c.POINTER(c.c_void_p)]),
            "NvAFX_DestroyEffect": (c.c_int, [c.c_void_p]),
            "NvAFX_SetU32": (c.c_int, [c.c_void_p, c.c_char_p, c.c_uint]),
            "NvAFX_SetString": (c.c_int, [c.c_void_p, c.c_char_p, c.c_char_p]),
            "NvAFX_SetFloat": (c.c_int, [c.c_void_p, c.c_char_p, c.c_float]),
            "NvAFX_GetU32": (c.c_int, [c.c_void_p, c.c_char_p, c.POINTER(c.c_uint)]),
            "NvAFX_GetString": (c.c_int, [c.c_void_p, c.c_char_p, c.c_char_p, c.c_int]),
            "NvAFX_GetFloat": (c.c_int, [c.c_void_p, c.c_char_p, c.POINTER(c.c_float)]),
            "NvAFX_Load": (c.c_int, [c.c_void_p]),
            "NvAFX_GetSupportedDevices": (c.c_int, [c.c_void_p, c.POINTER(c.c_int), c.POINTER(c.c_int)]),
            "NvAFX_Run": (c.c_int, [c.c_void_p, c.c_void_p, c.c_void_p, c.c_uint, c.c_uint]),
            "NvAFX_Reset": (c.c_int, [c.c_void_p]),
            "NvAFX_InitializeLogger": (c.c_int, [c.c_int, c.c_int, c.c_char_p, c.c_void_p, c.c_void_p]),
            "NvAFX_UninitializeLogger": (c.c_int, []),
        }
        for name, (restype, argtypes) in sig.items():
            fn = getattr(lib, name)  # AttributeError: a symbol is missing, which ``load`` turns into "not available"
            fn.restype, fn.argtypes = restype, argtypes

    def create_effect(self, code: str) -> tuple[int, int | None]:
        handle = ctypes.c_void_p()
        status = self._lib.NvAFX_CreateEffect(code.encode(), ctypes.byref(handle))
        return status, handle.value

    def destroy_effect(self, handle: int) -> int:
        return int(self._lib.NvAFX_DestroyEffect(ctypes.c_void_p(handle)))

    def set_u32(self, handle: int, name: bytes, value: int) -> int:
        return int(self._lib.NvAFX_SetU32(ctypes.c_void_p(handle), name, value))

    def set_float(self, handle: int, name: bytes, value: float) -> int:
        return int(self._lib.NvAFX_SetFloat(ctypes.c_void_p(handle), name, value))

    def set_string(self, handle: int, name: bytes, value: str) -> int:
        return int(self._lib.NvAFX_SetString(ctypes.c_void_p(handle), name, value.encode()))

    def get_u32(self, handle: int, name: bytes) -> tuple[int, int]:
        value = ctypes.c_uint()
        status = self._lib.NvAFX_GetU32(ctypes.c_void_p(handle), name, ctypes.byref(value))
        return int(status), value.value

    def load_effect(self, handle: int) -> int:
        return int(self._lib.NvAFX_Load(ctypes.c_void_p(handle)))

    def channel_pointers(self, frame: Frame) -> Any:
        """The ``float *[1]`` NvAFX_Run takes for one channel; keep it as long as ``frame`` is used."""
        return (ctypes.c_void_p * 1)(frame.ctypes.data)

    def run(self, handle: int, pointers: Any, samples: int, channels: int) -> int:
        """NvAFX_Run :658-660: the same pointer array is input and output, so the frame is processed in place."""
        return int(self._lib.NvAFX_Run(ctypes.c_void_p(handle), pointers, pointers, samples, channels))

    def init_logger(self) -> int:
        """NvAFX_InitializeLogger :615: errors only (SDK >= 1.6.1.2). OBS sends them to a callback (:529-533, target
        LOG_TARGET_CALLBACK); here they go to stderr (LOG_TARGET_STDERR, nvafx-load.h :138). A Python callback
        crashed the process every time it exited after an effect had loaded (measured on SDK 1.6.1.2: the SDK
        logs while it tears down, after Python's callback is gone), and stderr exits cleanly."""
        return int(self._lib.NvAFX_InitializeLogger(LOG_LEVEL_ERROR, LOG_TARGET_STDERR, None, None, None))

    def uninit_logger(self) -> None:
        self._lib.NvAFX_UninitializeLogger()


@dataclass(frozen=True)
class Runtime:
    """A runtime that loaded and passed OBS's GPU check. ``version``: its packed file version (0 if unreadable)."""

    sdk_dir: str
    version: int
    api: Api | None = None  # None only for a stand-in the UI tests and the preview use

    @property
    def vad_supported(self) -> bool:
        """The Voice Optimized option exists from SDK 1.6.1.2 (MIN_AFX_LOGGER_VERSION :6; :306, :487, :821)."""
        return self.version >= MIN_LOGGER_VERSION

    @property
    def outdated(self) -> bool:
        """nvidia_audio_properties :846: a readable version below the minimum shows OBS's warning."""
        return 0 < self.version < MIN_SDK_VERSION


def _load_module(path: str) -> ctypes.CDLL:
    """load_afx_module :293: LoadLibraryExA with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS."""
    flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
    return ctypes.CDLL(path, winmode=flags)


def load(folder: str) -> Runtime | None:
    """load_nvidia_afx :183-260: read the version (:185-199), load ``NVAudioEffects.dll`` and, from SDK 3.0.0.49, the
    three per-effect DLLs (load_lib :297-317), bind all 14 symbols (:209-231), then create and destroy one denoiser
    to prove the GPU can run it (:233-250). Any failure unloads everything and returns None: the filter is not offered."""
    if sys.platform != "win32" or not folder:
        return None
    version = dll_version(f"{folder}\\NVAudioEffects.dll")
    if version:
        log.info("[NVIDIA Audio Effects:] version: %s", version_text(version))
        if version < MIN_SDK_VERSION:
            log.info("[NVIDIA Audio Effects:]: SDK is outdated. Please update both audio & video SDK.")
    modules: list[ctypes.CDLL] = []  # kept alive for as long as the runtime is (nv_audiofx, nv_denoiserfx, ... :14-17)
    try:
        modules.append(_load_module(f"{folder}\\NVAudioEffects.dll"))
        if version >= MIN_ARM_VERSION:
            for name in ("nvafxdenoiser.dll", "nvafxdereverb.dll", "nvafxdereverbdenoiser.dll"):
                modules.append(_load_module(f"{folder}\\{name}"))
        api = Api(modules[0])
    except (OSError, AttributeError) as exc:
        log.info(
            "[NVIDIA Audio Effects:] NVIDIA denoiser disabled, redistributable not found or could not be loaded: %s",
            exc,
        )
        return None
    status, handle = api.create_effect(DENOISER)
    if status != STATUS_SUCCESS or handle is None:
        if status == STATUS_GPU_UNSUPPORTED:
            log.info("[NVIDIA Audio Effects:] disabled: unsupported GPU")
        else:
            log.error("[NVIDIA Audio Effects:] disabled, error %i", status)
        return None
    status = api.destroy_effect(handle)
    if status != STATUS_SUCCESS:
        log.error("[NVIDIA Audio Effects:]: disabled, error %i", status)
        return None
    log.info("[NVIDIA Audio Effects:] enabled")
    _keep.append(modules)
    return Runtime(folder, version, api)


_keep: list[list[ctypes.CDLL]] = []  # the loaded DLLs stay loaded for the life of the process, as OBS's module's do
_probe_lock = threading.Lock()
_probed = sys.platform != "win32"  # nothing to look for on macOS: the answer is already "no runtime"
_probe_started = False
_runtime: Runtime | None = None


def runtime() -> Runtime | None:
    """The runtime if it is installed and usable, found once (OBS looks once, when its module loads, :23-28). The first
    call does the whole check (loading the DLLs takes a while) and waits for it, so only the window's menu uses this;
    the app starts the check early on a thread (``probe_async``) and the audio side uses ``peek``."""
    global _probed, _runtime
    with _probe_lock:
        if not _probed:
            try:
                _runtime = load(sdk_dir())
            except Exception:  # a broken install must never break the app: the filter is just not offered
                log.exception("[NVIDIA Audio Effects:] the runtime check failed")
                _runtime = None
            _probed = True
        return _runtime


def probe_async() -> None:
    """Start the one-time check on a thread, as OBS does it at module load and not when a filter is added."""
    global _probe_started
    if not _probed and not _probe_started:
        _probe_started = True
        threading.Thread(target=runtime, name="clean-voice-nvafx-probe", daemon=True).start()


def peek() -> tuple[bool, Runtime | None]:
    """(checked, runtime) without waiting: the audio side never blocks on the check. Starts it if nobody has."""
    if not _probed:
        probe_async()
        return False, None
    return True, _runtime


# ------------------------------------------------------------------ one effect


class Effect:
    """One NVAFX effect on one mono channel, which is all Clean Voice's chain has (OBS: ``handle[i]`` per channel).

    Follows nvidia-audiofx-filter.c. Creating it starts the load on a thread (nvidia_audio_create :544-566 and
    nvidia_audio_update :461-466 -> nvidia_audio_initialize :347-398, nvidia_audio_initialize_internal :275-336: create
    the effect, set the 48 kHz rate, the intensity, Voice Optimized (SDK >= 1.6.1.2, not for room echo removal alone)
    and the model path, then Load; the number of channels must be 1 and the frame 480 samples). Until that is done
    ``run`` leaves the frame as it is, as OBS passes the audio on while ``nvafx_initialized`` and ``reinit_done`` are
    false (:710, :655). A change of method needs a new effect (:504-525): it is loaded on the thread again, and the
    old one destroyed first. A change of intensity or Voice Optimized is a set call on the loaded effect (:472-502).

    One difference from OBS, which locks a mutex in the audio thread (:656): ``run`` only tries the lock, and while
    the load thread holds it the frame passes unchanged, so the audio path never waits on a 2-second model load.
    ``failed`` is set when the effect cannot be created, loaded or set (OBS then switches the filter off, :333-345).
    """

    _init_lock = threading.Lock()  # nvidia_afx_initializer_mutex :176: one effect initialises at a time

    def __init__(self, rt: Runtime, method: str, intensity: float, vad: bool) -> None:
        if rt.api is None:
            raise ValueError("this runtime has no API")
        self._rt = rt
        self._api = rt.api
        self._lock = threading.Lock()  # nvafx_mutex :107: held by the load thread, tried by ``run``
        self._handle: int | None = None
        self._loaded_method = ""
        self._method = method
        self._wanted: tuple[float, bool] = (intensity, vad)
        self._applied: tuple[float, bool] | None = None
        self._ready = False
        self.failed = False
        self._closed = False
        self._io = np.zeros(FRAME_SIZE, np.float32)  # nvafx_segment_buffers :111, :578
        self._pointers = self._api.channel_pointers(self._io)
        self._logger = rt.version >= MIN_LOGGER_VERSION and rt.api.init_logger() == STATUS_SUCCESS
        self._start_load(method)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def method(self) -> str:
        return self._method

    def set(self, method: str, intensity: float, vad: bool) -> None:
        """The filter's settings changed (nvidia_audio_update :444-527), from the audio thread, between blocks: cheap.
        A new method starts a load; a new intensity or VAD is applied by the next ``run``."""
        self._wanted = (intensity, vad)
        if method != self._method and not self._closed:
            self._method = method
            self._ready = False  # reinit_done = false :511
            self._start_load(method)

    def _start_load(self, method: str) -> None:
        threading.Thread(target=self._load, args=(method,), name="clean-voice-nvafx-load", daemon=True).start()

    def _model(self, method: str) -> str:
        """set_nv_model :424-442: the runtime's folder plus the model file of the method (the denoiser's for any other)."""
        return self._rt.sdk_dir + MODELS.get(method, MODELS[DENOISER])

    def _set_parameters(self, handle: int, method: str, intensity: float, vad: bool) -> bool:
        """The intensity and Voice Optimized sets of :297-313 and :472-502 (VAD: SDK >= 1.6.1.2, not for dereverb)."""
        status = self._api.set_float(handle, PARAM_INTENSITY_RATIO, intensity)
        if status != STATUS_SUCCESS:
            log.error("NvAFX_SetFloat(Intensity Ratio: %f) failed, error %i", intensity, status)
            return False
        if self._rt.version >= MIN_LOGGER_VERSION and method != DEREVERB:
            status = self._api.set_u32(handle, PARAM_ENABLE_VAD, int(vad))
            if status != STATUS_SUCCESS:
                log.error("NvAFX_SetU32(VAD: %i) failed, error %i", vad, status)
                return False
        return True

    def _load(self, method: str) -> None:
        with Effect._init_lock, self._lock:
            if method != self._method or self._closed:
                self._finish_if_closed()
                return  # superseded by a newer change or closed while it waited
            self._destroy_handle()
            ok = self._create(method)
            if not ok:
                self.failed = True  # use_nvafx = false :334; OBS also switches the filter off :343
                self._ready = False
            self._finish_if_closed()

    def _create(self, method: str) -> bool:
        api = self._api
        status, handle = api.create_effect(method)
        if status != STATUS_SUCCESS or handle is None:
            log.error("%s FX creation failed, error %i", method, status)
            return False
        self._handle = handle
        intensity, vad = self._wanted
        status = api.set_u32(handle, PARAM_INPUT_SAMPLE_RATE, SAMPLE_RATE)
        if status != STATUS_SUCCESS:
            log.error("NvAFX_SetU32(Sample Rate: %u) failed, error %i", SAMPLE_RATE, status)
            return False
        if not self._set_parameters(handle, method, intensity, vad):
            return False
        status = api.set_string(handle, PARAM_MODEL_PATH, self._model(method))
        if status != STATUS_SUCCESS:
            log.error("NvAFX_SetString() failed, error %i", status)
            return False
        status = api.load_effect(handle)  # the long step :322-327
        if status != STATUS_SUCCESS:
            log.error("NvAFX_Load() failed with error %i", status)
            return False
        status, channels = api.get_u32(handle, PARAM_NUM_INPUT_CHANNELS)
        if status != STATUS_SUCCESS or channels != 1:
            log.error("NvAFX channels: error %i, %i channels (expected 1)", status, channels)  # :361-369
            return False
        name = PARAM_SAMPLES_PER_FRAME_ARM if self._rt.version >= MIN_ARM_VERSION else PARAM_SAMPLES_PER_FRAME
        status, size = api.get_u32(handle, name)
        if status != STATUS_SUCCESS or size != FRAME_SIZE:
            log.error("NvAFX frame: error %i, %i samples (expected 480)", status, size)  # :370-384
            return False
        self._applied = (intensity, vad)
        self._loaded_method = method
        self._ready = True
        return True

    def _destroy_handle(self) -> None:
        if self._handle is not None:
            status = self._api.destroy_effect(self._handle)
            if status != STATUS_SUCCESS:
                log.error("FX failed to be destroyed (error %i).", status)  # :514-516
            self._handle = None
        self._ready = False

    def _finish_if_closed(self) -> None:
        if self._closed:
            self._destroy_handle()

    def run(self, frame: Frame) -> bool:
        """Denoise one 480-sample frame of 48 kHz mono in place (nvidia_audio_filter_audio :739-805 -> process_fx
        :625-702). False, with the frame untouched, when the effect is not ready (loading, failed, closed)."""
        if not self._ready or not self._lock.acquire(blocking=False):
            return False
        try:
            handle = self._handle
            if not self._ready or handle is None:
                return False
            if self._wanted != self._applied:  # :472-502: an intensity or VAD change, between two frames
                intensity, vad = self._wanted
                if not self._set_parameters(handle, self._loaded_method, intensity, vad):
                    self.failed, self._ready = True, False
                    return False
                self._applied = (intensity, vad)
            self._io[:] = (
                frame  # the SDK works in place on the effect's own buffer: the caller's frame is kept on a failure
            )
            status = self._api.run(handle, self._pointers, FRAME_SIZE, 1)
            if status == STATUS_SUCCESS:
                frame[:] = self._io
                return True
            if status == STATUS_FAILED:
                # :663-667: "can occur when changing the FX and is not consequential": stop until the next load
                log.debug("NvAFX_Run() failed, error NVAFX_STATUS_FAILED.")
                self._ready = False
            else:
                log.error("NvAFX_Run() failed, error %i.", status)
            return False
        finally:
            self._lock.release()

    def close(self) -> None:
        """nvidia_audio_destroy :128-173: destroy the effect and the logger. The destroy takes the lock on a thread of its
        own, so a load still in progress finishes first and the caller never waits."""
        if self._closed:
            return
        self._closed = True
        self._ready = False
        threading.Thread(target=self._close, name="clean-voice-nvafx-close", daemon=True).start()

    def _close(self) -> None:
        with self._lock:
            self._destroy_handle()
        if self._logger:
            self._logger = False
            self._api.uninit_logger()
        # nothing else holds the effect: OBS frees its buffers and the resamplers here (:154-171), ours are Python's

    def __del__(self) -> None:
        with contextlib.suppress(Exception):  # at interpreter exit the thread machinery may be gone
            self.close()
