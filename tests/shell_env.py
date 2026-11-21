"""When a test may start a shell (Windows PowerShell, bash): shared by the install-script tests (B019).

Inside a Claude session the Python safety net refuses to start shells, so those tests skip there and run in CI, as
tests/sandbox/test_prefix_transport.py does. CI sets CLEAN_VOICE_REQUIRE_SHELL_TESTS=1: there, a test that cannot run
on that OS fails instead of skipping, so a skip can never hide a missing proof.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

REQUIRE_VAR = "CLEAN_VOICE_REQUIRE_SHELL_TESTS"
NET_MARK = "[clean-voice safety net]"


def running_under_net() -> bool:
    """True when this process loaded the guard's audit hook (it refuses to start shells)."""
    mod = sys.modules.get("sitecustomize")
    return getattr(mod, "MARK", None) == NET_MARK


def powershell_path() -> Path | None:
    """Windows PowerShell 5.1, never pwsh: the guest's shell."""
    if sys.platform != "win32":
        return None
    exe = (
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    return exe if exe.is_file() else None


def bash_path() -> str | None:
    return shutil.which("bash") if sys.platform != "win32" else None


def require(reason_if_missing: str, available: bool) -> None:
    """Skip the test when it cannot run here; fail instead when CI says every shell test must run."""
    required = os.environ.get(REQUIRE_VAR) == "1"
    if running_under_net():
        # CI never runs under the net; a session does, and cannot prove these.
        if required:
            pytest.fail(f"shell tests cannot run under the safety net ({REQUIRE_VAR}=1)")
        pytest.skip("this session's safety net refuses to start shells; CI runs this test")
    if not available:
        if required:
            pytest.fail(f"{reason_if_missing} ({REQUIRE_VAR}=1)")
        pytest.skip(reason_if_missing)
