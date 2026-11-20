"""The bundled RNNoise library and its ctypes binding."""

import ctypes
import struct

import numpy as np
from scipy import signal
from speech import FAN, RATE, noise

from clean_voice import rnnoise
from clean_voice.rnnoise import FRAME_SIZE, NATIVE_DIR, RNNoise

FAT_MAGIC = 0xCAFEBABE
LC_BUILD_VERSION = 0x32
LC_SEGMENT_64 = 0x19
LC_CODE_SIGNATURE = 0x1D
CPU_TYPES = {0x0100000C: "arm64", 0x01000007: "x86_64"}


def test_loads_and_denoises() -> None:
    rnnoise.load()
    denoiser = RNNoise()
    fan = noise(FAN)  # real fan noise: OBS's RNNoise, trained on such noise, leaves white noise almost untouched
    out = np.zeros(FRAME_SIZE, np.float32)
    energy_in = energy_out = 0.0
    for i in range(200):
        frame = fan[i * FRAME_SIZE : (i + 1) * FRAME_SIZE].astype(np.float32)
        vad = denoiser.process(frame, out)
        assert 0.0 <= vad <= 1.0
        if i >= 50:
            energy_in += float(np.sum(frame.astype(np.float64) ** 2))
            energy_out += float(np.sum(out.astype(np.float64) ** 2))
    denoiser.close()
    assert 10 * np.log10(energy_out / energy_in) < -20


def _denoise(x: np.ndarray) -> np.ndarray:
    denoiser = RNNoise()
    out = np.zeros(len(x) // FRAME_SIZE * FRAME_SIZE, np.float32)
    for i in range(0, len(out), FRAME_SIZE):
        denoiser.process(x[i : i + FRAME_SIZE].astype(np.float32), out[i : i + FRAME_SIZE])
    denoiser.close()
    return out


def test_fan_only_pauses_are_as_quiet_as_obs() -> None:
    # B005: fan.wav through OBS's RNNoise leaves about -76 dB, RNNoise 0.2 about -57 dB (audit p9: the fan after the 80 Hz
    # high-pass, as RNNoise sees it in the chain; whole-output RMS).
    fan = noise(FAN)[: 10 * RATE]
    x = np.asarray(signal.sosfilt(signal.butter(4, 80, "highpass", fs=RATE, output="sos"), fan), np.float32)
    out = _denoise(x)
    assert 20 * np.log10(np.sqrt(np.mean(out.astype(np.float64) ** 2)) + 1e-12) < -70.0


def test_lag_is_one_frame() -> None:
    # B005: OBS's RNNoise delays its output by 480 samples (10 ms), 0.2 by 960.
    x = np.zeros(20 * FRAME_SIZE, np.float32)
    x[5 * FRAME_SIZE + 100] = 0.5
    out = _denoise(x)
    assert int(np.argmax(np.abs(out))) - (5 * FRAME_SIZE + 100) == FRAME_SIZE


def test_exports_obs_api() -> None:
    # B005: OBS's rnnoise.h has rnnoise_get_size and no rnnoise_get_frame_size.
    lib = ctypes.CDLL(str(rnnoise.library_path()))
    assert hasattr(lib, "rnnoise_get_size")
    assert not hasattr(lib, "rnnoise_get_frame_size")


def _macho_min_os(blob: bytes) -> tuple[int, int]:
    magic, _cpu, _sub, _filetype, ncmds, _size, _flags, _reserved = struct.unpack_from("<8I", blob)
    assert magic == 0xFEEDFACF
    offset = 32
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", blob, offset)
        if cmd == LC_BUILD_VERSION:
            _platform, minos = struct.unpack_from("<II", blob, offset + 8)
            return minos >> 16, (minos >> 8) & 0xFF
        offset += cmdsize
    raise AssertionError("no LC_BUILD_VERSION")


def test_macos_library_is_universal_for_macos_13() -> None:
    # Checked here because no Mac is needed to read the file; CI also runs `lipo -archs` on the built DMG.
    blob = (NATIVE_DIR / "librnnoise.dylib").read_bytes()
    magic, count = struct.unpack_from(">II", blob)
    assert magic == FAT_MAGIC
    archs = {}
    for i in range(count):
        cputype, _sub, offset, size, _align = struct.unpack_from(">5I", blob, 8 + 20 * i)
        archs[CPU_TYPES[cputype]] = _macho_min_os(blob[offset : offset + size])
    assert archs == {"arm64": (13, 0), "x86_64": (13, 0)}


def _linkedit_end_and_signature_end(blob: bytes) -> tuple[int, int, int | None]:
    """The highest file end of any segment, __LINKEDIT's end, and LC_CODE_SIGNATURE's end (None when unsigned)."""
    ncmds = struct.unpack_from("<I", blob, 16)[0]
    offset, highest, linkedit, signature = 32, 0, -1, None
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", blob, offset)
        if cmd == LC_SEGMENT_64:
            name = blob[offset + 8 : offset + 24].rstrip(b"\0")
            fileoff, filesize = struct.unpack_from("<QQ", blob, offset + 40)
            highest = max(highest, fileoff + filesize)
            if name == b"__LINKEDIT":
                linkedit = fileoff + filesize
        elif cmd == LC_CODE_SIGNATURE:
            dataoff, datasize = struct.unpack_from("<II", blob, offset + 8)
            signature = dataoff + datasize
        offset += cmdsize
    return highest, linkedit, signature


def test_macos_library_passes_codesign_strict_layout() -> None:
    # Briefcase signs the dylib with codesign, which refuses a "main executable" that fails strict validation: data
    # after the last slice (the first CI run padded the last slice to 16 KiB), overlapping or misaligned slices, or a
    # slice whose __LINKEDIT (with the signature at its end) does not end the slice.
    blob = (NATIVE_DIR / "librnnoise.dylib").read_bytes()
    magic, count = struct.unpack_from(">II", blob)
    assert magic == FAT_MAGIC
    slices = sorted(struct.unpack_from(">5I", blob, 8 + 20 * i)[2:] for i in range(count))
    end = 8 + 20 * count
    for offset, size, align in slices:
        assert offset % (1 << align) == 0, "slice not aligned to its declared alignment"
        assert end <= offset < end + (1 << align), "slices overlap or leave more than one alignment of gap"
        assert not any(blob[end:offset]), "non-zero bytes between slices"
        thin = blob[offset : offset + size]
        highest, linkedit, signature = _linkedit_end_and_signature_end(thin)
        assert linkedit == highest == size, "__LINKEDIT must be the last segment and end the slice"
        assert signature in (None, size), "the code signature must end the slice"
        end = offset + size
    trailing = len(blob) - end
    assert trailing == 0, f"{trailing} bytes after the last slice"
