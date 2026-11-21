"""The CI launch-check script ``tools/ci/smoke_app.py`` (J04): its pure parts and the app-run judgement."""

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "ci" / "smoke_app.py"
SHOWN = "Clean Voice window shown"


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    # tools/ is not a package and not on the import path: load the script from its file.
    spec = importlib.util.spec_from_file_location("smoke_app", SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_app(tmp_path: Path, body: str, name: str = "fake_app.py") -> list[str]:
    """A command prefix that runs a small Python script standing in for the packaged app."""
    script = tmp_path / name
    script.write_text("import sys\n" + textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, str(script)]


# S1
def test_find_single_requires_exactly_one(smoke: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(smoke.SmokeError, match="no"):
        smoke.find_single(tmp_path, "*.msi")
    one = tmp_path / "Clean Voice-0.1.0.msi"
    one.write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    assert smoke.find_single(tmp_path, "*.msi") == one
    (tmp_path / "Clean Voice-0.0.9.msi").write_bytes(b"")
    with pytest.raises(smoke.SmokeError, match="2"):
        smoke.find_single(tmp_path, "*.msi")


# S2
def test_find_installed_exe_in_either_root(smoke: ModuleType, tmp_path: Path) -> None:
    root_a = tmp_path / "LocalAppData" / "Programs"
    root_b = tmp_path / "Program Files"
    root_a.mkdir(parents=True)
    root_b.mkdir(parents=True)
    missing_root = tmp_path / "not-there"
    assert smoke.find_installed_exe([missing_root, root_a, root_b]) is None
    for root in (root_a, root_b):
        # The install folder is named for the app, right under the root (DESIGN, Install folder).
        exe = root / "Clean Voice" / "Clean Voice.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"")
        assert smoke.find_installed_exe([missing_root, root_a, root_b]) == exe
        exe.unlink()
        exe.parent.rmdir()
    assert smoke.find_installed_exe([root_a, root_b]) is None
    # The old author-named parent folder is not accepted: the MSI must not install there.
    old = root_a / "Paul Caesar Flores" / "Clean Voice" / "Clean Voice.exe"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"")
    assert smoke.find_installed_exe([root_a, root_b]) is None


# S3
def test_universal_check(smoke: ModuleType) -> None:
    assert smoke.parse_lipo_archs("x86_64 arm64\n") == {"x86_64", "arm64"}
    assert smoke.missing_archs(smoke.parse_lipo_archs("x86_64 arm64\n")) == []
    assert smoke.missing_archs(smoke.parse_lipo_archs("arm64\n")) == ["x86_64"]
    assert smoke.missing_archs(set()) == ["arm64", "x86_64"]


# S4
def test_check_app_judges_exit_code_and_line(smoke: ModuleType, tmp_path: Path) -> None:
    shown = fake_app(tmp_path, f"print({SHOWN!r})\n", "shown.py")
    smoke.check_app(shown, require_line=True)
    smoke.check_app(shown, require_line=False)

    not_shown = fake_app(tmp_path, "print('Clean Voice window NOT shown', file=sys.stderr)\nsys.exit(2)\n", "exit2.py")
    with pytest.raises(smoke.SmokeError, match="exit code 2"):
        smoke.check_app(not_shown, require_line=False)

    silent = fake_app(tmp_path, "pass\n", "silent.py")
    with pytest.raises(smoke.SmokeError, match="not in the app's output"):
        smoke.check_app(silent, require_line=True)
    smoke.check_app(silent, require_line=False)


# S5
def test_exe_passes_app_arguments_through(
    smoke: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    echo = fake_app(tmp_path, f"print('ARGV ' + ' '.join(sys.argv[1:]))\nprint({SHOWN!r})\n", "echo.py")
    assert smoke.main(["exe", *echo, "--", "-platform", "x"]) == 0
    assert "ARGV --quit-after-ms 3000 -platform x" in capsys.readouterr().out

    # A command may hold dash words (`python -m clean_voice`); --require-line is the script's own flag.
    python, script = fake_app(tmp_path, f"print({SHOWN!r})\n", "dash.py")
    assert smoke.main(["exe", python, "-B", script, "--require-line"]) == 0

    failing = fake_app(tmp_path, "sys.exit(3)\n", "fail.py")
    assert smoke.main(["exe", *failing]) == 1


def test_exe_accepts_a_relative_program_path(
    smoke: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CI (Git Bash) and the local check name the built exe relative to the project folder with forward slashes;
    # Windows' CreateProcess does not find `dir/sub/app.exe` written that way (WinError 2).
    python, script = fake_app(tmp_path, f"print({SHOWN!r})\n", "rel.py")
    home = Path(python).parent.parent
    monkeypatch.chdir(home)
    relative = Path(python).relative_to(home).as_posix()
    assert smoke.main(["exe", relative, script, "--require-line"]) == 0


def test_msi_log_tail_reads_utf16_and_8bit(smoke: ModuleType, tmp_path: Path) -> None:
    lines = [f"line {i}" for i in range(100)]
    utf16 = tmp_path / "utf16.log"
    utf16.write_bytes("\r\n".join(lines).encode("utf-16"))  # with a BOM
    plain = tmp_path / "plain.log"
    plain.write_bytes("\r\n".join(lines).encode("utf-8"))
    for log in (utf16, plain):
        assert smoke.read_log_tail(log).splitlines() == lines[-60:]
    assert "no log" in smoke.read_log_tail(tmp_path / "missing.log")


def test_msi_check_requires_the_line(smoke: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # J04 Step 6: the built GUI exe prints the "window shown" line to a piped stdout, so the msi check gates on it.
    assert smoke.MSI_REQUIRE_LINE is True
    (tmp_path / "Clean Voice-0.1.0.msi").write_bytes(b"")
    exe = tmp_path / "Programs" / "Clean Voice" / "Clean Voice.exe"
    runs: list[list[str]] = []
    checks: list[tuple[list[str], bool]] = []

    def fake_run(command: list[str], timeout: float = 600) -> subprocess.CompletedProcess[str]:
        runs.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(smoke, "_run", fake_run)
    monkeypatch.setattr(smoke, "find_installed_exe", lambda roots: exe)
    monkeypatch.setattr(smoke, "check_app", lambda command, *, require_line: checks.append((command, require_line)))
    smoke.smoke_msi(tmp_path)
    assert checks == [([str(exe)], True)]
    assert [run[:2] for run in runs] == [["msiexec", "/i"], ["msiexec", "/x"]]


# S6
def test_wrong_machine_is_refused(smoke: ModuleType) -> None:
    smoke.check_arch("arm64", machine="arm64")
    with pytest.raises(smoke.SmokeError, match="x86_64"):
        smoke.check_arch("arm64", machine="x86_64")


# ---- tools/ci/smoke_install.py (B019): the pure parts; the real run is CI's ---------------------------------------------


@pytest.fixture(scope="module")
def install_smoke() -> ModuleType:
    ci = str(ROOT / "tools" / "ci")
    sys.path.insert(0, ci)  # smoke_install loads its siblings by file, but never hurts to have them importable
    try:
        spec = importlib.util.spec_from_file_location("smoke_install", ROOT / "tools" / "ci" / "smoke_install.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(ci)


def test_the_local_command_is_the_guests_command_with_another_base(install_smoke: ModuleType) -> None:
    from clean_voice import updates

    repo = "paulcaesarflores/clean-voice"
    real_base = f"https://github.com/{repo}/releases/latest/download"
    for platform, app_platform in (("windows", "win32"), ("macos", "darwin")):
        local = install_smoke.local_command(platform, "http://127.0.0.1:1234")
        assert local.replace("http://127.0.0.1:1234", real_base) == updates.install_command_text(repo, app_platform)


def test_the_command_runs_with_no_python_and_no_git_on_path(install_smoke: ModuleType) -> None:
    env = install_smoke.stripped_env(
        "windows", "http://127.0.0.1:1", {"Path": r"C:\Python312;C:\Git\cmd", "CLEAN_VOICE_UPDATE": "1", "TEMP": "x"}
    )
    assert env["PATH"] == install_smoke.WINDOWS_PATH and "Path" not in env
    assert "python" not in env["PATH"].lower() and "git" not in env["PATH"].lower()
    assert env["CLEAN_VOICE_BASE"] == "http://127.0.0.1:1" and "CLEAN_VOICE_UPDATE" not in env and env["TEMP"] == "x"
    assert install_smoke.stripped_env("macos", "b", {"PATH": "/opt/homebrew/bin"})["PATH"] == install_smoke.MAC_PATH
    argv = install_smoke.command_argv("windows", "http://h")
    assert argv[0].lower().endswith("powershell.exe") and argv[1:3] == ["-NoProfile", "-Command"]
    assert argv[3] == "irm http://h/install.ps1 | iex"
    assert install_smoke.command_argv("macos", "http://h")[1:] == ["-c", "curl -fsSL http://h/install.sh | bash"]


def test_the_local_server_answers_every_file_as_octet_stream(install_smoke: ModuleType, tmp_path: Path) -> None:
    import socket

    (tmp_path / "install.ps1").write_bytes(b"# a script" + bytes([10]))
    (tmp_path / "page.html").write_text("<p>x</p>", encoding="ascii")
    with install_smoke.serve(tmp_path) as base:
        port = int(base.rsplit(":", 1)[1])
        for name, body in (("install.ps1", b"# a script" + bytes([10])), ("page.html", b"<p>x</p>")):
            # A raw loopback socket: the suite's no-network guard refuses urllib and socket.create_connection.
            with socket.socket() as conn:
                conn.settimeout(30)
                conn.connect(("127.0.0.1", port))
                crlf = bytes([13, 10])
                conn.sendall(f"GET /{name} HTTP/1.0".encode("ascii") + crlf + b"Host: 127.0.0.1" + crlf + crlf)
                data = b""
                while chunk := conn.recv(4096):
                    data += chunk
            head, _, payload = data.partition(crlf + crlf)
            assert b"content-type: application/octet-stream" in head.lower()
            assert payload == body


def test_registry_entry_judgement(install_smoke: ModuleType) -> None:
    folder = r"C:\Users\x\AppData\Local\Programs\Clean Voice"
    good = {
        "DisplayName": "Clean Voice",
        "Publisher": "Clean Voice",
        "DisplayVersion": "0.3.0",
        "InstallLocation": folder,
        "DisplayIcon": folder + r"\Clean Voice.exe",
        "UninstallString": rf'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{folder}\Uninstall Clean Voice.ps1"',
    }
    assert install_smoke.registry_problems(good, folder, "0.3.0") == []
    assert len(install_smoke.registry_problems({**good, "Publisher": "Someone"}, folder, "0.3.0")) == 1
    assert install_smoke.registry_problems({**good, "UninstallString": "msiexec /x"}, folder, "0.3.0")
    assert install_smoke.registry_problems(good, folder, "0.4.0")


def test_plist_and_run_value_judgement(install_smoke: ModuleType) -> None:
    assert install_smoke.plist_problems({"LSAppNapIsDisabled": True, "CFBundleIconFile": "clean-voice"}) == []
    assert len(install_smoke.plist_problems({"CFBundleIconFile": "clean-voice"})) == 1
    exists = {r"C:\Apps\Clean Voice\Clean Voice.exe"}.__contains__
    assert install_smoke.run_value_problem(r'"C:\Apps\Clean Voice\Clean Voice.exe" --minimized', exists) is None
    assert "does not exist" in install_smoke.run_value_problem(r'"C:\Gone\Clean Voice.exe" --minimized', exists)
    assert "gone" in install_smoke.run_value_problem(None, exists)


def test_the_release_folder_is_built_from_git_archive(install_smoke: ModuleType, tmp_path: Path) -> None:
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is needed")
    project = tmp_path / "proj"
    (project / "src" / "clean_voice").mkdir(parents=True)
    (project / "tools" / "install").mkdir(parents=True)
    (project / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n[tool.briefcase]\nversion = "1.2.3"\n')
    (project / "src" / "clean_voice" / "__init__.py").write_text('__version__ = "1.2.3"\n')
    for name in ("install.ps1", "install.sh"):
        (project / "tools" / "install" / name).write_text("v=@@VERSION@@ h=@@SOURCE_SHA256@@\n", encoding="ascii")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=project, check=True
        )

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "x")
    out = tmp_path / "release"
    assert install_smoke.build_release(out, project) == "v1.2.3"
    assert sorted(p.name for p in out.iterdir()) == ["Clean-Voice-source.zip", "install.ps1", "install.sh"]
    assert (out / "install.ps1").read_text(encoding="ascii").startswith("v=1.2.3 h=")
