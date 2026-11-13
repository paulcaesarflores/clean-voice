"""An in-test VST 2 plug-in: an ``AEffect`` built in ctypes whose dispatcher, process and parameter calls are Python
``CFUNCTYPE`` callbacks (J10 plan Step 5), and a fake library that exports its entry point."""

from __future__ import annotations

import ctypes
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from clean_voice import vst

# The compiled test plug-in (tools/vst_test_plugin/build.py).
TEST_PLUGIN = (
    Path(__file__).resolve().parents[1]
    / ".cache"
    / "vst_test_plugin"
    / ("CleanVoiceTestGain.vst" if sys.platform == "darwin" else "CleanVoiceTestGain.dll")
)


def require_test_plugin() -> None:
    """Skip when the test plug-in isn't built, except on GitHub Actions, which builds it before the tests (ci.yml):
    there a missing plug-in fails the test."""
    if sys.platform not in ("win32", "darwin"):
        pytest.skip("VST 2 hosting is Windows and macOS only")
    if not TEST_PLUGIN.exists():
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail(f"CI builds the test plug-in before the tests, but {TEST_PLUGIN} is missing")
        pytest.skip("the test plug-in is not built (tools/vst_test_plugin/build.py)")


class FakePlugin:
    """A gain effect that records every dispatcher call. ``chunks``: it keeps a program chunk (its parameters'
    bytes); otherwise the host saves its parameters. ``process`` multiplies by parameter 0 on every channel, and
    raises a flag (``after_close``) if it is ever called after effClose."""

    def __init__(
        self,
        inputs: int = 2,
        outputs: int = 2,
        flags: int = vst.FLAG_CAN_REPLACING,
        params: int = 2,
        magic: int = vst.MAGIC,
        sleep_s: float = 0.0,
        rect: tuple[int, int, int, int] | None = (0, 0, 300, 400),  # top, left, bottom, right
        name: str = "Fake Gain",
    ) -> None:
        self.calls: list[tuple[int, int, int, float]] = []  # (opcode, index, value, opt)
        self.params = [1.0] + [0.25] * (params - 1)
        self.closed = False
        self.after_close = 0
        self.processed = 0
        self.sleep_s = sleep_s
        self.rect = vst.Rect(*rect) if rect else None
        self.name = name
        self.chunk = ctypes.create_string_buffer(4 * params)
        self.received_chunk = b""
        self.editor_parent: int | None = None
        self.host: Callable[..., int] | None = None
        self.seen_outputs: list[np.ndarray] = []  # the outputs as handed in (zeroed?), when ``keep_outputs``
        self.keep_outputs = False
        self.inputs_seen: list[np.ndarray] = []
        self._dispatcher = vst.Dispatcher(self._dispatch)
        self._process = vst.ProcessFn(self._run)
        self._set = vst.SetParameterFn(self._set_param)
        self._get = vst.GetParameterFn(self._get_param)
        self.effect = vst.AEffect()
        e = self.effect
        e.magic = magic
        e.dispatcher = self._dispatcher
        e.process_replacing = self._process
        e.set_parameter = self._set
        e.get_parameter = self._get
        e.num_programs = 1
        e.num_params = params
        e.num_inputs = inputs
        e.num_outputs = outputs
        e.flags = flags
        self.entry = vst.EntryFn(self._main)

    @property
    def opcodes(self) -> list[int]:
        return [c[0] for c in self.calls]

    def _main(self, host: object) -> int:
        self.host = host  # type: ignore[assignment]
        return ctypes.addressof(self.effect)

    def _dispatch(self, effect: object, opcode: int, index: int, value: int, ptr: int | None, opt: float) -> int:
        self.calls.append((opcode, index, value, opt))
        if opcode == vst.EFF_CLOSE:
            self.closed = True
        elif opcode == vst.EFF_GET_EFFECT_NAME and ptr:
            ctypes.memmove(ptr, self.name.encode() + b"\0", len(self.name) + 1)
        elif opcode == vst.EFF_GET_VENDOR_STRING and ptr:
            ctypes.memmove(ptr, b"Tests Inc\0", 10)
        elif opcode == vst.EFF_GET_CHUNK and ptr:
            raw = np.array(self.params, np.float32).tobytes()
            ctypes.memmove(self.chunk, raw, len(raw))
            ctypes.c_void_p.from_address(ptr).value = ctypes.addressof(self.chunk)
            return len(raw)
        elif opcode == vst.EFF_SET_CHUNK and ptr:
            self.received_chunk = ctypes.string_at(ptr, value)
            vals = np.frombuffer(self.received_chunk, np.float32)
            self.params[: len(vals)] = [float(v) for v in vals]
            return 1
        elif opcode == vst.EFF_EDIT_GET_RECT and ptr:
            ctypes.c_void_p.from_address(ptr).value = ctypes.addressof(self.rect) if self.rect is not None else None
            return 1 if self.rect is not None else 0
        elif opcode == vst.EFF_EDIT_OPEN:
            self.editor_parent = ptr
            return 1
        return 0

    def _run(self, effect: object, ins: ctypes._Pointer, outs: ctypes._Pointer, frames: int) -> None:  # type: ignore[type-arg]
        if self.closed:
            self.after_close += 1
        self.processed += 1
        if self.sleep_s:
            time.sleep(self.sleep_s)
        e = self.effect
        channels = max(e.num_inputs, e.num_outputs)
        in_p = ctypes.cast(ins, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
        out_p = ctypes.cast(outs, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
        if self.keep_outputs:
            self.seen_outputs.append(
                np.array([np.ctypeslib.as_array(out_p[c], (frames,)).copy() for c in range(channels)])
            )
            self.inputs_seen.append(
                np.array([np.ctypeslib.as_array(in_p[c], (frames,)).copy() for c in range(channels)])
            )
        for c in range(e.num_outputs):
            src = np.ctypeslib.as_array(in_p[min(c, max(e.num_inputs - 1, 0))], (frames,))
            dst = np.ctypeslib.as_array(out_p[c], (frames,))
            dst[:] = src * self.params[0] * (1.0 + c)  # output c is (1 + c) x gain x input: the mean is checkable

    def _set_param(self, effect: object, index: int, value: float) -> None:
        self.params[index] = value

    def _get_param(self, effect: object, index: int) -> float:
        return self.params[index]


class FakeLibrary:
    """A plug-in binary that exports ``symbols``: name -> FakePlugin (its entry point)."""

    def __init__(self, symbols: dict[str, FakePlugin]) -> None:
        self.symbols = symbols
        self.freed = 0

    def symbol(self, name: str) -> int | None:
        plugin = self.symbols.get(name)
        return ctypes.cast(plugin.entry, ctypes.c_void_p).value if plugin is not None else None

    def free(self) -> None:
        self.freed += 1


def load_fake(plugin: FakePlugin, entry: str = "VSTPluginMain", path: str = "fake.dll") -> vst.VstPlugin:
    library = FakeLibrary({entry: plugin})
    loaded = vst.load(path, lambda p: library, platform="win32")
    loaded.fake_library = library  # type: ignore[attr-defined]
    return loaded
