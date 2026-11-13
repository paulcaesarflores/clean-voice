"""The VST 2.x host: Fine-tune's "VST 2.x Plug-in" filter (J10 plan §5).

It follows OBS Studio's obs-vst (``reference/obs-studio/plugins/obs-vst/``): the folders it scans, how it loads a
plug-in, processes audio, answers the plug-in's calls, saves its state and unloads it. Each step cites its OBS file;
the code is our own (OBS is GPL, Clean Voice is MIT).

**The binary interface** (the ``AEffect`` layout, opcode and flag numbers) is declared here in ctypes from Xaymar's
vst2sdk v0.4.0, ``include/vst.h`` (BSD-3-Clause, a clean-room reimplementation; its licence is bundled as
``native/VST2SDK-LICENSE.txt``, and the pinned header is ``tools/vst_test_plugin/vst2sdk/vst.h``). No Steinberg SDK,
VeSTige or FST source is used. vst2sdk documents no time-info structure, so ``TimeInfo`` (the block a host hands out
for ``audioMasterGetTime``, host opcode 0x07) is declared here from the published binary layout itself; the host
answers it as OBS does, with a fixed 120 BPM, 4/4 transport (VSTPlugin.cpp:53-57, :71-75, :194-201).

**Threads.** Loading, the editor, chunk get/set and unloading run on the UI thread; the worker only calls
``VstPlugin.process``. A ctypes call releases the GIL for the whole foreign call, so a plug-in's processing never
holds up the UI. OBS guards ``process`` and ``unloadEffect`` with one mutex (VSTPlugin.cpp:235, :272-284); the worker
takes no lock here, so a plug-in is unloaded only once no running chain can still call it (``VstRegistry``).
"""

from __future__ import annotations

import base64
import binascii
import collections
import ctypes
import hashlib
import logging
import os
import plistlib
import re
import sys
import time
import weakref
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from clean_voice.dsp import FineTune

log = logging.getLogger(__name__)

KIND = "vst_filter"  # obs-vst.cpp:335
SAMPLE_RATE = 48_000
BLOCK_SIZE = 512  # obs-vst headers/VSTPlugin.h:22: the most a plug-in is handed per call
MAX_CHANNELS = 256  # VSTPlugin.cpp:169: the sanity check on a plug-in's channel count
VENDOR = b"Clean Voice\0"  # OBS answers "OBS Studio" (VSTPlugin.cpp:49-51)
NAME_BUFFER = 256  # OBS passes char[64] (headers/VSTPlugin.h:66-68); a bigger buffer only adds room for long names

Audio = npt.NDArray[np.float32]

# ------------------------------------------------------------------ the ABI (vst2sdk v0.4.0 include/vst.h)

MAGIC = 0x56737450  # VST_MAGICNUMBER, the FourCC 'VstP' (:60, :1597)

# Host to plug-in (VST_EFFECT_OPCODE, :1899-2434).
EFF_OPEN = 0x00  # VST_EFFECT_OPCODE_CREATE :1901
EFF_CLOSE = 0x01  # VST_EFFECT_OPCODE_DESTROY :1913
EFF_SET_SAMPLE_RATE = 0x0A  # :2014, the rate in p_float
EFF_SET_BLOCK_SIZE = 0x0B  # :2024, the size in p_int2
EFF_MAINS_CHANGED = 0x0C  # VST_EFFECT_OPCODE_SUSPEND_RESUME :2038, p_int2 1 resumes, 0 suspends
EFF_EDIT_GET_RECT = 0x0D  # VST_EFFECT_OPCODE_EDITOR_GET_RECT :2054, p_ptr a vst_rect_t**
EFF_EDIT_OPEN = 0x0E  # VST_EFFECT_OPCODE_EDITOR_OPEN :2065, p_ptr the parent window
EFF_EDIT_CLOSE = 0x0F  # VST_EFFECT_OPCODE_EDITOR_CLOSE :2075
EFF_IDENTIFY = 0x16  # VST_EFFECT_OPCODE_FOURCC :2161
EFF_GET_CHUNK = 0x17  # VST_EFFECT_OPCODE_GET_CHUNK_DATA :2175, p_int1 1 = program, p_ptr a void**
EFF_SET_CHUNK = 0x18  # VST_EFFECT_OPCODE_SET_CHUNK_DATA :2189, p_int2 the size, p_ptr the data
EFF_GET_EFFECT_NAME = 0x2D  # VST_EFFECT_OPCODE_EFFECT_NAME :2409
EFF_GET_VENDOR_STRING = 0x2F  # VST_EFFECT_OPCODE_VENDOR_NAME :2434

# Plug-in to host (VST_HOST_OPCODE, :1053-1455): the ones OBS answers (VSTPlugin.cpp hostCallback_static :32-68).
HOST_VERSION = 0x01  # VST_HOST_OPCODE_VST_VERSION :1065
HOST_GET_TIME = 0x07  # VST_HOST_OPCODE_07 :1099 (undocumented in vst2sdk): answered with a TimeInfo pointer
HOST_SIZE_WINDOW = 0x0F  # VST_HOST_OPCODE_EDITOR_RESIZE :1162, p_int1 width, p_int2 height
HOST_GET_SAMPLE_RATE = 0x10  # VST_HOST_OPCODE_GET_SAMPLE_RATE :1173
HOST_WILL_REPLACE = 0x16  # VST_HOST_OPCODE_16 :1233; OBS: "we always replace, never accumulate" (:39-41)
HOST_PROCESS_LEVEL = 0x17  # VST_HOST_OPCODE_GET_ACTIVE_THREAD :1244
HOST_VENDOR_STRING = 0x20  # VST_HOST_OPCODE_VENDOR_NAME :1290
HOST_VERSION_2400 = 2400  # VST_VERSION_2_4_0_0 :164

# VST_EFFECT_FLAG (:1828-1888).
FLAG_HAS_EDITOR = 1 << 0  # VST_EFFECT_FLAG_EDITOR
FLAG_CAN_REPLACING = 1 << 4  # VST_EFFECT_FLAG_SUPPORTS_FLOAT: process_float is there
FLAG_PROGRAM_CHUNKS = 1 << 5  # VST_EFFECT_FLAG_CHUNKS
FLAG_IS_SYNTH = 1 << 8  # VST_EFFECT_FLAG_INSTRUMENT


class AEffect(ctypes.Structure):
    """struct vst_effect_t (:2956-3120), 64-bit (``#pragma pack(push, 8)``, :32: natural alignment)."""


FloatPtrs = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
EffectPtr = ctypes.POINTER(AEffect)
# vst_effect_control_t :2899 and vst_effect_process_float_t :2939 (VST_FUNCTION_INTERFACE is __cdecl, the only x64
# convention). CFUNCTYPE, never PYFUNCTYPE: the GIL is released while the plug-in runs.
Dispatcher = ctypes.CFUNCTYPE(
    ctypes.c_ssize_t, EffectPtr, ctypes.c_int32, ctypes.c_int32, ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_float
)
ProcessFn = ctypes.CFUNCTYPE(None, EffectPtr, FloatPtrs, FloatPtrs, ctypes.c_int32)
SetParameterFn = ctypes.CFUNCTYPE(None, EffectPtr, ctypes.c_uint32, ctypes.c_float)  # :2918
GetParameterFn = ctypes.CFUNCTYPE(ctypes.c_float, EffectPtr, ctypes.c_uint32)  # :2926
# vst_host_callback_t :1587.
HostCallback = ctypes.CFUNCTYPE(
    ctypes.c_ssize_t, EffectPtr, ctypes.c_int32, ctypes.c_int32, ctypes.c_int64, ctypes.c_void_p, ctypes.c_float
)
# VST_ENTRYPOINT :3127: vst_effect_t* VSTPluginMain(vst_host_callback_t).
EntryFn = ctypes.CFUNCTYPE(ctypes.c_void_p, HostCallback)

AEffect._fields_ = [
    ("magic", ctypes.c_int32),
    ("dispatcher", Dispatcher),
    ("process", ProcessFn),  # the accumulating process of VST 1, never called (OBS uses processReplacing)
    ("set_parameter", SetParameterFn),
    ("get_parameter", GetParameterFn),
    ("num_programs", ctypes.c_int32),
    ("num_params", ctypes.c_int32),
    ("num_inputs", ctypes.c_int32),
    ("num_outputs", ctypes.c_int32),
    ("flags", ctypes.c_int32),
    ("_unknown_00", ctypes.c_void_p),
    ("_unknown_01", ctypes.c_void_p),
    ("delay", ctypes.c_int32),
    ("_unknown_02", ctypes.c_int32),
    ("_unknown_03", ctypes.c_int32),
    ("input_output_ratio", ctypes.c_float),
    ("effect_internal", ctypes.c_void_p),
    ("host_internal", ctypes.c_void_p),
    ("unique_id", ctypes.c_int32),
    ("version", ctypes.c_int32),
    ("process_replacing", ProcessFn),  # process_float
    ("process_double", ctypes.c_void_p),
    ("_unknown", ctypes.c_uint8 * 56),
]


class Rect(ctypes.Structure):
    """struct vst_rect_t (:173-178)."""

    _fields_ = [
        ("top", ctypes.c_int16),
        ("left", ctypes.c_int16),
        ("bottom", ctypes.c_int16),
        ("right", ctypes.c_int16),
    ]


class TimeInfo(ctypes.Structure):
    """The time-info block a VST 2.4 host returns for ``audioMasterGetTime``, declared here as an interoperability
    fact of the binary interface (no SDK, VeSTige or FST source used): eight 64-bit floats, then six 32-bit ints, in
    this order, natural alignment, 88 bytes.

    - ``sample_pos``: the transport's position in samples; ``sample_rate``: in Hz; ``nano_seconds``: the system time;
      ``ppq_pos``: the position in quarter notes; ``tempo``: in BPM; ``bar_start_pos``: the last bar's start in
      quarter notes; ``cycle_start_pos`` and ``cycle_end_pos``: the loop, in quarter notes.
    - ``time_sig_numerator`` and ``time_sig_denominator``: the time signature; ``smpte_offset`` and
      ``smpte_frame_rate``: SMPTE; ``samples_to_next_clock``: MIDI clock; ``flags``: which fields are valid (the
      ``TIME_*`` bits).
    """

    _fields_ = [
        ("sample_pos", ctypes.c_double),
        ("sample_rate", ctypes.c_double),
        ("nano_seconds", ctypes.c_double),
        ("ppq_pos", ctypes.c_double),
        ("tempo", ctypes.c_double),
        ("bar_start_pos", ctypes.c_double),
        ("cycle_start_pos", ctypes.c_double),
        ("cycle_end_pos", ctypes.c_double),
        ("time_sig_numerator", ctypes.c_int32),
        ("time_sig_denominator", ctypes.c_int32),
        ("smpte_offset", ctypes.c_int32),
        ("smpte_frame_rate", ctypes.c_int32),
        ("samples_to_next_clock", ctypes.c_int32),
        ("flags", ctypes.c_int32),
    ]


# TimeInfo.flags bits (the binary interface's numbers): the three OBS sets (VSTPlugin.cpp:201).
TIME_TRANSPORT_PLAYING = 1 << 1
TIME_NANOS_VALID = 1 << 8
TIME_TEMPO_VALID = 1 << 10


def _clock_ms() -> float:
    """OBS's GetTimeInfo clock (VSTPlugin.cpp:73): ``os_gettime_ns() / 1000000``, an integer count of milliseconds
    of the monotonic clock, stored in the nanoseconds field as OBS stores it."""
    return float(time.perf_counter_ns() // 1_000_000)


# ------------------------------------------------------------------ plain words for a plug-in that won't load

WRONG_ARCHITECTURE = "it is made for a different kind of computer (a 32-bit plug-in, or one for another processor)"
NO_ENTRY_POINT = "it is not a VST 2 plug-in (it has no VST 2 starting point)"
NO_EFFECT = "the plug-in did not start"
BAD_MAGIC = "it is not a VST 2 plug-in"
BAD_CHANNELS = "it reports an impossible number of audio channels"
IS_SYNTH = "it is an instrument, not an effect"
NO_REPLACING = "it is too old: it can't process audio the way VST 2.4 hosts need"


class VstError(Exception):
    """A plug-in Clean Voice can't load. ``reason`` completes "Clean Voice couldn't load this plug-in: ..."."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ------------------------------------------------------------------ the folder scan (obs-vst.cpp fill_out_plugins)

# The 64-bit Windows folders, in OBS's order (obs-vst.cpp:178-183): (environment variable, folder under it).
WINDOWS_FOLDERS = (
    ("ProgramFiles", "Steinberg/VstPlugins/"),
    ("CommonProgramFiles", "Steinberg/Shared Components/"),
    ("CommonProgramFiles", "VST2"),
    ("CommonProgramFiles", "Steinberg/VST2"),
    ("CommonProgramFiles", "VSTPlugins/"),
    ("ProgramFiles", "VSTPlugins/"),
)
MAC_SYSTEM_FOLDER = "/Library/Audio/Plug-Ins/VST/"  # obs-vst.cpp:167
MAC_USER_FOLDER = "Library/Audio/Plug-ins/VST/"  # obs-vst.cpp:168, under the home folder


def search_folders(platform: str = sys.platform) -> list[Path]:
    """The folders OBS scans for VST 2 plug-ins (obs-vst.cpp:162-192). On macOS the user folder's ``~`` is expanded:
    OBS hands "~/Library/..." to QDir as it is, which never expands it, so OBS misses the folder it clearly means."""
    if platform == "darwin":
        return [Path(MAC_SYSTEM_FOLDER), Path.home() / MAC_USER_FOLDER]
    if platform == "win32":
        # qEnvironmentVariable gives "" for an unset variable, and OBS appends "/..." all the same.
        return [Path(os.environ.get(var, "") + "/" + sub) for var, sub in WINDOWS_FOLDERS]
    return []


def _hidden(entry: os.DirEntry[str]) -> bool:
    """QDir lists no hidden entries unless asked (QDir::Hidden): dot-names, and on Windows the hidden attribute."""
    if entry.name.startswith("."):
        return True
    attrs = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attrs & 0x2)  # FILE_ATTRIBUTE_HIDDEN


def _walk(folder: Path, suffix: str, seen: set[str]) -> Iterable[Path]:
    """Every entry under ``folder`` whose name ends in ``suffix`` (any case, as QDir's name filters), files and folders
    alike (a macOS .vst is a folder), recursing into every subfolder and following links, as QDirIterator with
    Subdirectories | FollowSymlinks does. ``seen`` stops a link loop."""
    try:
        real = os.path.realpath(folder)
        if real in seen:
            return
        seen.add(real)
        entries = sorted(os.scandir(folder), key=lambda e: e.name)
    except OSError:
        return
    for entry in entries:
        try:
            if _hidden(entry):
                continue
            if entry.name.lower().endswith(suffix):
                yield Path(entry.path)
            if entry.is_dir():  # follows links
                yield from _walk(Path(entry.path), suffix, seen)
        except OSError:
            continue


def scan(folders: Iterable[Path] | None = None, platform: str = sys.platform) -> list[tuple[str, str]]:
    """The plug-in list, (name, path) per plug-in, as obs-vst.cpp fill_out_plugins builds it (:222-267): ``*.vst`` on
    macOS and ``*.dll`` on Windows; the name is the file name with every ".vst"/".dll" taken out (QString::remove,
    any case); sorted case-sensitively by "name=path" (std::stable_sort of QStrings, :259). Nothing is loaded."""
    suffix = ".vst" if platform == "darwin" else ".dll"
    found: list[str] = []
    seen: set[str] = set()
    for folder in search_folders(platform) if folders is None else folders:
        for path in _walk(Path(folder), suffix, seen):
            name = re.sub(re.escape(suffix), "", path.name, flags=re.IGNORECASE)
            found.append(f"{name}={path}")
    found.sort(key=lambda s: s.encode("utf-16-be"))  # QString's operator< compares UTF-16 code units
    return [(s[: s.index("=")], s[s.index("=") + 1 :]) for s in found]


# ------------------------------------------------------------------ opening the library (win/, mac/ VSTPlugin-*.cpp)


class Library(Protocol):
    """An opened plug-in binary."""

    def symbol(self, name: str) -> int | None:
        """The address of an exported function, or None."""
        ...

    def free(self) -> None: ...


ERROR_BAD_EXE_FORMAT = 193
RTLD_NOW, RTLD_LOCAL = 0x2, 0x4  # macOS <dlfcn.h>

# The entry points, tried in order: win/VSTPlugin-win.cpp:49-57 and mac/VSTPlugin-osx.mm:45-51.
ENTRY_POINTS = {"win32": ("VSTPluginMain", "VstPluginMain()", "main"), "darwin": ("VSTPluginMain", "main_macho")}


if sys.platform == "win32":

    class _WindowsLibrary:
        """LoadLibraryW / GetProcAddress / FreeLibrary, as win/VSTPlugin-win.cpp loadEffect :28-57 and unloadLibrary
        :81-87."""

        def __init__(self, path: str) -> None:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # its own instance: argtypes stay ours
            k32.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
            k32.LoadLibraryW.restype = ctypes.c_void_p
            k32.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            k32.GetProcAddress.restype = ctypes.c_void_p
            k32.FreeLibrary.argtypes = [ctypes.c_void_p]
            self._k32 = k32
            self._handle: int | None = k32.LoadLibraryW(path)
            if not self._handle:
                error = ctypes.get_last_error()
                if error == ERROR_BAD_EXE_FORMAT:  # :37-39
                    raise VstError(WRONG_ARCHITECTURE)
                raise VstError(f"Windows couldn't open the file ({ctypes.FormatError(error).strip()})")

        def symbol(self, name: str) -> int | None:
            return self._k32.GetProcAddress(self._handle, name.encode("ascii")) or None

        def free(self) -> None:
            if self._handle:
                self._k32.FreeLibrary(self._handle)
                self._handle = None


class _MacBundle:
    """The bundle's executable, opened with dlopen. OBS opens the .vst with CFBundleCreate and looks its entry points
    up with CFBundleGetFunctionPointerForName (mac/VSTPlugin-osx.mm:36-51); CFBundle loads the executable that
    Info.plist's CFBundleExecutable names from Contents/MacOS, which is what this does directly."""

    def __init__(self, path: str) -> None:
        bundle = Path(path)
        executable = bundle.stem
        try:
            with open(bundle / "Contents" / "Info.plist", "rb") as f:
                executable = str(plistlib.load(f).get("CFBundleExecutable") or executable)
        except (OSError, plistlib.InvalidFileException, ValueError):
            pass  # CFBundle also falls back to the bundle's name
        libc = ctypes.CDLL(None)
        libc.dlopen.argtypes = [ctypes.c_char_p, ctypes.c_int]
        libc.dlopen.restype = ctypes.c_void_p
        libc.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        libc.dlsym.restype = ctypes.c_void_p
        libc.dlclose.argtypes = [ctypes.c_void_p]
        libc.dlerror.restype = ctypes.c_char_p
        self._libc = libc
        self._handle: int | None = libc.dlopen(
            os.fsencode(bundle / "Contents" / "MacOS" / executable), RTLD_NOW | RTLD_LOCAL
        )
        if not self._handle:
            detail = (libc.dlerror() or b"").decode("utf-8", "replace")
            # dyld: "...mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64')".
            if "architecture" in detail:
                raise VstError(WRONG_ARCHITECTURE)
            raise VstError(f"macOS couldn't open the plug-in ({detail or 'no reason given'})")

    def symbol(self, name: str) -> int | None:
        return self._libc.dlsym(self._handle, name.encode("ascii")) or None

    def free(self) -> None:
        if self._handle:
            self._libc.dlclose(self._handle)
            self._handle = None


def open_library(path: str) -> Library:
    if sys.platform == "win32":
        return _WindowsLibrary(path)
    if sys.platform == "darwin":
        return _MacBundle(path)
    raise VstError("VST 2 plug-ins work on Windows and macOS")


# ------------------------------------------------------------------ the host callback (VSTPlugin.cpp :22-69)

_PLUGINS: dict[int, VstPlugin] = {}  # by the address of the plug-in's AEffect (OBS uses effect->user)


def _address(effect: object) -> int:
    return ctypes.cast(effect, ctypes.c_void_p).value or 0  # type: ignore[arg-type]


def _host_callback(effect: object, opcode: int, index: int, value: int, ptr: int | None, opt: float) -> int:
    """The plug-in's calls to the host: OBS's answers (hostCallback_static :32-68), everything else 0. A plug-in may
    call it from the audio thread, so it does little and never raises."""
    try:
        plugin = _PLUGINS.get(_address(effect)) if effect else None
        if opcode == HOST_VERSION:
            return HOST_VERSION_2400
        if opcode == HOST_PROCESS_LEVEL:
            return 1
        if opcode == HOST_WILL_REPLACE:
            return 1
        if opcode == HOST_GET_SAMPLE_RATE:
            return SAMPLE_RATE if plugin is not None else 0
        if opcode == HOST_VENDOR_STRING:
            if ptr:
                ctypes.memmove(ptr, VENDOR, len(VENDOR))
            return 1
        if opcode == HOST_GET_TIME:  # :53-57: the plug-in's own block, its clock brought up to date (:71-75)
            if plugin is None:
                return 0
            plugin.time_info.nano_seconds = _clock_ms()
            return plugin.time_info_address
        if opcode == HOST_SIZE_WINDOW:  # index: width, value: height (:59-64)
            editor = plugin.editor if plugin is not None else None
            if editor is not None:
                editor.request_resize(index, value)
            return 1
        return 0
    except Exception:
        log.exception("VST host callback (opcode %d)", opcode)
        return 0


HOST = HostCallback(_host_callback)  # one for every plug-in, alive as long as the module


# ------------------------------------------------------------------ one loaded plug-in


class Editor(Protocol):
    """The plug-in's editor window (``vst_editor.EditorWindow``)."""

    def request_resize(self, width: int, height: int) -> None: ...

    def close(self) -> bool: ...


class VstPlugin:
    """One loaded VST 2 effect: OBS's VSTPlugin (VSTPlugin.cpp). Made on the UI thread by ``load``; the worker calls
    only ``process`` (and ``update``, which a plug-in doesn't need: its state is its chunk)."""

    def __init__(self, path: str, library: Library, entry_points: Iterable[str]) -> None:
        self.path = path
        self._library: Library | None = library
        self._ready = False
        self._opened = False  # True once unloading must call effMainsChanged(0) and effClose, as unloadEffect does
        self.editor: Editor | None = None
        self.open_when_active = False
        self.effect_name = ""
        self.vendor = ""
        self.num_inputs = self.num_outputs = self.flags = 0
        # mTimeInfo (headers/VSTPlugin.h:70): one block for the plug-in's lifetime, so the pointer GetTime hands out
        # never dangles and the audio thread's GetTime allocates nothing.
        self.time_info = TimeInfo()
        self.time_info_address = ctypes.addressof(self.time_info)
        try:
            self._load(entry_points)
        except BaseException:
            self.close()  # a refused plug-in is unloaded at once (OBS keeps it until the filter goes)
            raise

    def _load(self, entry_points: Iterable[str]) -> None:
        assert self._library is not None
        entry = next((a for a in (self._library.symbol(name) for name in entry_points) if a), None)
        if entry is None:
            raise VstError(NO_ENTRY_POINT)  # win :59-62, osx :53-60
        address = EntryFn(entry)(HOST)  # win :65-70, osx :62
        if not address:
            raise VstError(NO_EFFECT)  # win :72-75, osx :63-70
        self._effect = ctypes.cast(address, EffectPtr)
        eff = self._effect.contents
        # loadEffectFromPath :159-165: a bad magic number means the file is not a VST plug-in, or is corrupt. Its
        # struct can't be trusted, so it is not called at all (OBS would still dispatch effClose to it on unload).
        if eff.magic != MAGIC:
            raise VstError(BAD_MAGIC)
        self._dispatch = eff.dispatcher
        self._key = address
        _PLUGINS[address] = self
        self._opened = True
        self._start(eff)

    def _start(self, eff: AEffect) -> None:
        """loadEffectFromPath :167-209, in OBS's order."""
        channels = max(eff.num_inputs, eff.num_outputs)
        if channels < 0 or channels > MAX_CHANNELS:  # :167-172
            raise VstError(BAD_CHANNELS)
        self.num_inputs, self.num_outputs, self.flags = eff.num_inputs, eff.num_outputs, eff.flags
        # createChannelBuffers :91-107: max(in, out) channels of BLOCK_SIZE, made once. Mono: every input gets the
        # mic's block (OBS hands its stereo mix's planes, where a mono mic is the same on both).
        self._ins = np.zeros((channels, BLOCK_SIZE), np.float32)
        self._outs = np.zeros((channels, BLOCK_SIZE), np.float32)
        ptr = ctypes.POINTER(ctypes.c_float)
        self._in_ptrs = (ptr * max(channels, 1))(*(row.ctypes.data_as(ptr) for row in self._ins))
        self._out_ptrs = (ptr * max(channels, 1))(*(row.ctypes.data_as(ptr) for row in self._outs))
        self._channels = channels
        self.effect_name = self._string(EFF_GET_EFFECT_NAME)  # :177-178
        self.vendor = self._string(EFF_GET_VENDOR_STRING)
        if self.flags & FLAG_IS_SYNTH:  # :180-184 (the check OBS takes from Audacity)
            raise VstError(IS_SYNTH)
        if not self.flags & FLAG_CAN_REPLACING:
            raise VstError(NO_REPLACING)
        self._process = eff.process_replacing
        self.dispatch(EFF_IDENTIFY)  # :187
        self.dispatch(EFF_OPEN)  # :189
        t = self.time_info  # :194-201, after effOpen as OBS does: zeroed (sample position 0), then the fixed transport
        ctypes.memset(self.time_info_address, 0, ctypes.sizeof(t))
        t.sample_rate = float(SAMPLE_RATE)
        t.nano_seconds = _clock_ms()
        t.tempo = 120.0
        t.time_sig_numerator, t.time_sig_denominator = 4, 4
        t.flags = TIME_TEMPO_VALID | TIME_NANOS_VALID | TIME_TRANSPORT_PLAYING
        self.dispatch(EFF_SET_SAMPLE_RATE, opt=float(SAMPLE_RATE))  # :203
        self.dispatch(EFF_SET_BLOCK_SIZE, value=BLOCK_SIZE)  # :204-205
        self.dispatch(EFF_MAINS_CHANGED, value=1)  # :207
        self._ready = True  # effectReady :209

    # -------------------------------------------------------------- calls into the plug-in

    def dispatch(self, opcode: int, index: int = 0, value: int = 0, ptr: int | None = None, opt: float = 0.0) -> int:
        return int(self._dispatch(self._effect, opcode, index, value, ptr, opt))

    def _string(self, opcode: int) -> str:
        buf = ctypes.create_string_buffer(NAME_BUFFER)
        self.dispatch(opcode, ptr=ctypes.addressof(buf))
        return buf.value.decode("utf-8", "replace")

    @property
    def loaded(self) -> bool:
        """vstLoaded (win :89-92): the library is still open."""
        return self._library is not None

    @property
    def has_editor(self) -> bool:
        return self.loaded and bool(self.flags & FLAG_HAS_EDITOR)

    def editor_rect(self) -> tuple[int, int] | None:
        """effEditGetRect: the editor's (width, height), or None when the plug-in gives no rect."""
        rect = ctypes.POINTER(Rect)()
        self.dispatch(EFF_EDIT_GET_RECT, ptr=ctypes.addressof(rect))
        if not rect:
            return None
        r = rect.contents
        return r.right - r.left, r.bottom - r.top

    # -------------------------------------------------------------- the worker

    def update(self, settings: Mapping[str, object]) -> None:
        """A changed filter setting: nothing for the running instance (the guest's choices live in the plug-in)."""

    def process(self, x: Audio) -> Audio:
        """process :226-266: passes of at most BLOCK_SIZE, every output zeroed before the call, and only the plug-in's
        own outputs read back. Mono: the block goes to every input, and the output is the mean of the first two
        outputs (output 0 alone if it has one; the input unchanged if it has none, as OBS copies nothing back)."""
        n = len(x)
        if not self._ready or self._channels == 0:  # :230-233
            return x.astype(np.float32, copy=True)
        out = np.empty(n, np.float32)
        outs, ins, process, effect = self._outs, self._ins, self._process, self._effect
        n_out = self.num_outputs
        pos = 0
        while pos < n:
            frames = min(BLOCK_SIZE, n - pos)
            outs.fill(0.0)  # silenceChannel :242
            ins[:, :frames] = x[pos : pos + frames]
            process(effect, self._in_ptrs, self._out_ptrs, frames)  # :252, the GIL released
            dest = out[pos : pos + frames]
            if n_out >= 2:
                np.add(outs[0, :frames], outs[1, :frames], out=dest)
                dest *= 0.5
            elif n_out == 1:
                dest[:] = outs[0, :frames]
            else:
                dest[:] = x[pos : pos + frames]
            pos += frames
        return out

    # -------------------------------------------------------------- state (VSTPlugin.cpp getChunk / setChunk)

    def get_chunk(self) -> str:
        """getChunk :350-375: the program chunk (effGetChunk, index 1) when the plug-in keeps chunks, else every
        parameter as float32; base64 either way."""
        if not self.loaded:
            return ""
        if self.flags & FLAG_PROGRAM_CHUNKS:
            buf = ctypes.c_void_p()
            size = self.dispatch(EFF_GET_CHUNK, index=1, ptr=ctypes.addressof(buf))
            data = ctypes.string_at(buf.value, size) if buf.value and size > 0 else b""
            return base64.b64encode(data).decode("ascii")
        eff = self._effect.contents
        values = np.array([eff.get_parameter(self._effect, i) for i in range(eff.num_params)], np.float32)
        return base64.b64encode(values.tobytes()).decode("ascii")

    def set_chunk(self, data: str) -> bool:
        """setChunk :377-408: effSetChunk index 1 for a chunk plug-in; otherwise the floats, restored only if there is
        one for every parameter. False when nothing was restored."""
        if not self.loaded:
            return False
        try:
            raw = base64.b64decode(data)
        except (binascii.Error, ValueError):
            return False  # Qt's fromBase64 skips bad input instead; either way nothing usable reaches the plug-in
        if self.flags & FLAG_PROGRAM_CHUNKS:
            buf = ctypes.create_string_buffer(raw, len(raw))
            self.dispatch(EFF_SET_CHUNK, index=1, value=len(raw), ptr=ctypes.addressof(buf))
            return True
        eff = self._effect.contents
        values = np.frombuffer(raw[: len(raw) // 4 * 4], np.float32)
        if len(values) != eff.num_params:  # :400-402
            return False
        for i, v in enumerate(values):
            eff.set_parameter(self._effect, i, float(v))
        return True

    # -------------------------------------------------------------- unloading (VSTPlugin.cpp unloadEffect)

    def close(self) -> None:
        """unloadEffect :268-289: close the editor, effMainsChanged(0), effClose, free the library. Only once no chain
        can call ``process`` (``VstRegistry``). Safe to call twice."""
        editor, self.editor = self.editor, None
        if editor is not None:
            editor.close()  # closeEditor :270
        self._ready = False  # :276
        if self._opened:
            self._opened = False
            self.dispatch(EFF_MAINS_CHANGED, value=0)  # :279
            self.dispatch(EFF_CLOSE)  # :280
            _PLUGINS.pop(self._key, None)
        library, self._library = self._library, None
        if library is not None:
            library.free()  # unloadLibrary :286

    def __repr__(self) -> str:
        return f"VstPlugin({self.effect_name or self.path!r})"


def load(path: str, opener: Callable[[str], Library] = open_library, platform: str = sys.platform) -> VstPlugin:
    """Load the plug-in at ``path``, on the UI thread (loadEffectFromPath :136-215). Raises ``VstError`` in plain
    words; OBS only logs a failure, the screen shows it."""
    return VstPlugin(path, opener(path), ENTRY_POINTS.get(platform, ENTRY_POINTS["win32"]))


def file_md5(path: str) -> str:
    """getFileMD5 (obs-vst.cpp:72-83): the file's MD5 in lower-case hex, or "" when it can't be read (a macOS .vst
    is a folder, so its hash is always "")."""
    try:
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "md5").hexdigest()
    except OSError:
        return ""


def restore(plugin: VstPlugin, chunk_data: str, chunk_hash: str) -> bool:
    """vst_update's restore (obs-vst.cpp:122-129): the saved chunk goes back only if the plug-in file's hash matches
    the saved one, or none was saved."""
    if not chunk_data:
        return False
    if chunk_hash and chunk_hash != file_md5(plugin.path):
        return False
    return plugin.set_chunk(chunk_data)


def saved_state(plugin: VstPlugin) -> dict[str, str]:
    """vst_save (obs-vst.cpp:140-145): the settings keys a VST filter saves."""
    return {"chunk_data": plugin.get_chunk(), "chunk_hash": file_md5(plugin.path)}


def create(settings: Mapping[str, object], opener: Callable[[str], Library] = open_library) -> VstPlugin | None:
    """vst_create / vst_update (obs-vst.cpp:98-138) for a filter's settings: load ``plugin_path`` (None when no
    plug-in is chosen), restore its saved chunk, and remember "Open interface when active", which the editor reads
    once it is loaded (VSTPlugin.cpp:211-213)."""
    path = settings.get("plugin_path")
    if not isinstance(path, str) or not path:
        return None
    plugin = load(path, opener)
    chunk, digest = settings.get("chunk_data"), settings.get("chunk_hash")
    restore(plugin, chunk if isinstance(chunk, str) else "", digest if isinstance(digest, str) else "")
    plugin.open_when_active = settings.get("open_when_active_vst_settings") is True
    return plugin


# ------------------------------------------------------------------ the engine's registry and the disposal handshake


class VstRegistry:
    """The loaded plug-ins, by filter uid, shared by the UI thread and the engine's chains.

    The UI thread loads a plug-in, ``add``s it, then publishes the ``Settings`` naming its filter; the worker's
    ``FineTune`` takes it with ``get``. To remove one, the UI publishes ``Settings`` without it, then calls
    ``remove``: the plug-in is unloaded (``VstPlugin.close``) only once no running chain may still call it, at once
    when none does, or later from ``collect`` on the UI's tick. A different plug-in for a filter gets a new uid
    (``fine_tune.new_uid``), so one uid always means one instance.

    No lock: the worker never waits. Each ``FineTune`` registers itself (``attach``) before it takes any instance and
    lists every uid it may still call in ``held``, adding a uid there before it calls ``get``. ``remove`` takes the uid
    out of ``get``'s reach first, so once ``collect`` sees a uid in no live chain's ``held``, no chain has it or can
    get it. A chain that is closed (``FineTune.close``) or gone holds nothing.
    """

    def __init__(self) -> None:
        self._active: dict[int, VstPlugin] = {}
        self._retired: dict[int, VstPlugin] = {}
        self._new: collections.deque[weakref.ref[FineTune]] = collections.deque()  # the worker appends
        self._holders: list[weakref.ref[FineTune]] = []  # the UI thread's own list

    # UI thread
    def add(self, uid: int, plugin: VstPlugin) -> None:
        if uid in self._active or uid in self._retired:
            raise ValueError(f"filter {uid} already has a plug-in: a new plug-in needs a new uid")
        self._active[uid] = plugin

    def plugin(self, uid: int) -> VstPlugin | None:
        return self._active.get(uid)

    def remove(self, uid: int) -> None:
        plugin = self._active.pop(uid, None)
        if plugin is not None:
            self._retired[uid] = plugin
        self.collect()

    def in_use(self) -> frozenset[int]:
        """The uids a live chain may still call."""
        while self._new:
            self._holders.append(self._new.popleft())
        live = [h for h in (ref() for ref in self._holders) if h is not None]
        self._holders = [weakref.ref(h) for h in live]
        return frozenset().union(*(h.held for h in live))

    def collect(self) -> int:
        """Unload every removed plug-in no live chain holds; how many were unloaded."""
        if not self._retired:
            return 0
        held = self.in_use()
        done = [uid for uid in self._retired if uid not in held]
        for uid in done:
            plugin = self._retired.pop(uid)
            try:
                plugin.close()
            except Exception:
                log.exception("unloading %r failed", plugin)
        return len(done)

    @property
    def pending(self) -> int:
        """Removed plug-ins still waiting for the chain to let go."""
        return len(self._retired)

    def close_all(self) -> None:
        """At quit, once the engine's worker has stopped (nothing can call a plug-in any more): unload everything."""
        self._retired.update(self._active)
        self._active.clear()
        self._holders.clear()
        self._new.clear()
        self.collect()

    # worker
    def attach(self, holder: FineTune) -> None:
        self._new.append(weakref.ref(holder))

    def get(self, uid: int) -> VstPlugin | None:
        return self._active.get(uid)
