"""tools/ci/set_copyright.py: the built exe's LegalCopyright is LICENSE's copyright line. Briefcase's build sets the
exe's other version strings with rcedit but not this one, which leaves the stub's own "Copyright (C) 2022"."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("set_copyright", ROOT / "tools" / "ci" / "set_copyright.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_line_comes_from_the_license(tmp_path: Path) -> None:
    tool = _load()
    license_file = tmp_path / "LICENSE"
    license_file.write_text("MIT License\n\nCopyright (c) 2031 Someone Else\n\nPermission is hereby granted\n", "utf-8")
    assert tool.copyright_line(license_file) == "Copyright (c) 2031 Someone Else"
    project = tool.copyright_line(ROOT / "LICENSE")
    assert project.startswith("Copyright (c) 20") and project in (ROOT / "LICENSE").read_text("utf-8")


def test_a_license_without_a_copyright_line_is_refused(tmp_path: Path) -> None:
    tool = _load()
    license_file = tmp_path / "LICENSE"
    license_file.write_text("MIT License\n\nPermission is hereby granted\n", "utf-8")
    with pytest.raises(ValueError, match="no Copyright line"):
        tool.copyright_line(license_file)


def test_rcedit_is_the_one_briefcase_downloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load()
    monkeypatch.setenv("BRIEFCASE_HOME", str(tmp_path))
    assert tool.rcedit_path() == tmp_path.resolve() / "tools" / "rcedit-x64.exe"
    monkeypatch.delenv("BRIEFCASE_HOME")
    default = tool.rcedit_path()
    assert default.name == "rcedit-x64.exe" and default.parent.name == "tools"
    assert "briefcase" in str(default).lower()


@pytest.mark.skipif(sys.platform != "win32", reason="a Windows exe's version resource")
def test_stamps_an_exe_and_reads_it_back(tmp_path: Path) -> None:
    tool = _load()
    if not tool.rcedit_path().exists():
        pytest.skip("Briefcase has not downloaded rcedit yet (it does on the first `briefcase build windows`)")
    exe = tmp_path / "stub.exe"
    shutil.copyfile(getattr(sys, "_base_executable", sys.executable), exe)  # any exe with a version resource
    before = tool.legal_copyright(exe)
    line = tool.copyright_line(ROOT / "LICENSE")
    assert before != line
    assert tool.main([str(exe)]) == 0
    assert tool.legal_copyright(exe) == line


@pytest.mark.skipif(sys.platform != "win32", reason="a Windows exe's version resource")
def test_a_missing_exe_fails_plainly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tool = _load()
    assert tool.main([str(tmp_path / "missing.exe")]) == 1
    assert "no such file" in capsys.readouterr().err
