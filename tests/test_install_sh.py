"""The macOS install script ``tools/install/install.sh`` (B019, step 3).

Each scenario sources the script with CLEAN_VOICE_TEST_IMPORT set (it then only defines its functions), points HOME and
the two Applications folders at scratch places, puts fake ``xcode-select`` and ``open`` programs first on PATH, and
runs one function under bash. Nothing times anything: a test asserts on the exit code, the output and the state left
behind (docs/checklists/tests-timing.md). Inside a Claude session the safety net refuses to start bash, so the
scenario tests skip there and run on the macOS CI jobs (tests/shell_env.py); the byte checks always run.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import shell_env

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "install" / "install.sh"


# ---- The file itself (runs everywhere) ---------------------------------------------------------------------------


def test_ascii_only_with_unix_line_endings() -> None:
    data = SCRIPT.read_bytes()
    assert all(b < 128 for b in data), [i for i, b in enumerate(data) if b >= 128][:5]
    assert b"\r" not in data
    assert data.startswith(b"#!/bin/bash\n")


def test_placeholders_appear_once() -> None:
    text = SCRIPT.read_text(encoding="ascii")
    assert text.count("@@VERSION@@") == 1
    assert text.count("@@SOURCE_SHA256@@") == 1


def test_main_is_called_on_the_last_line() -> None:
    last = SCRIPT.read_text(encoding="ascii").rstrip().splitlines()[-1]
    assert last == 'if [ -z "${CLEAN_VOICE_TEST_IMPORT:-}" ]; then main "$@"; fi'


def test_pins_equal_pyproject_and_uv_pins_are_sha256() -> None:
    import re
    import tomllib

    dev = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["dependency-groups"]["dev"]
    (pin,) = [d.split("==")[1] for d in dev if d.startswith("briefcase==")]
    text = SCRIPT.read_text(encoding="ascii")
    assert f"BRIEFCASE_PIN='{pin}'" in text
    assert re.search(r"UV_SHA256_ARM64='[0-9a-f]{64}'", text)
    assert re.search(r"UV_SHA256_X86_64='[0-9a-f]{64}'", text)


# ---- Scenarios under bash (CI, macOS) ------------------------------------------------------------------------------


@pytest.fixture
def sh(tmp_path: Path):
    if sys.platform == "win32":
        pytest.skip("the macOS installer runs under bash on a Mac")
    shell_env.require("bash is needed", shell_env.bash_path() is not None)
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    system_apps = tmp_path / "Applications"
    for folder in (home, bin_dir, system_apps):
        folder.mkdir()

    def fake(name: str, body: str) -> None:
        path = bin_dir / name
        path.write_text("#!/bin/bash\n" + body, encoding="ascii")
        path.chmod(0o755)

    # xcode-select: -p succeeds once the counter file says "ready"; --install counts the requests.
    fake(
        "xcode-select",
        f'''case "$1" in
  -p) n=$(cat "{tmp_path}/xcode-ready" 2>/dev/null || echo 0)
      if [ "$n" -le 0 ]; then exit 1; fi
      echo "{tmp_path}/tools"; exit 0 ;;
  --install) echo x >>"{tmp_path}/xcode-installs"; exit 0 ;;
esac
''',
    )
    fake("open", f'echo "$@" >>"{tmp_path}/open.log"\n')

    def run(body: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        script = tmp_path / "scenario.sh"
        script.write_text(
            f"""#!/bin/bash
export HOME="{home}"
export CLEAN_VOICE_TEST_IMPORT=1
source "{SCRIPT}"
init_vars
APPS_SYSTEM="{system_apps}"
APPS_USER="{home}/Applications"
WORK="{tmp_path}/work"
LOG="{tmp_path}/install.log"
STATUS_FILE="{tmp_path}/status.txt"
{body}
""",
            encoding="ascii",
        )
        full_env = {k: v for k, v in os.environ.items() if not k.startswith("CLEAN_VOICE_")}
        full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env.get('PATH', '')}"
        full_env.update(env or {})
        return subprocess.run(
            [str(shell_env.bash_path()), str(script)],
            capture_output=True,
            text=True,
            timeout=120,
            env=full_env,
            check=False,
        )

    run.tmp = tmp_path  # type: ignore[attr-defined]
    run.system_apps = system_apps  # type: ignore[attr-defined]
    run.home = home  # type: ignore[attr-defined]
    run.fake = fake  # type: ignore[attr-defined]
    return run


def make_app(folder: Path, text: str) -> Path:
    app = folder / "Clean Voice.app" if folder.name != "Clean Voice.app" else folder
    (app / "Contents").mkdir(parents=True, exist_ok=True)
    (app / "Contents" / "marker").write_text(text, encoding="ascii")
    return app


def marker(app: Path) -> str:
    path = app / "Contents" / "marker"
    return path.read_text(encoding="ascii") if path.exists() else "none"


def test_bash_syntax_is_valid(sh) -> None:
    result = subprocess.run(
        [str(shell_env.bash_path()), "-n", str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert (result.returncode, result.stderr) == (0, "")


def test_checksum_good_and_bad(sh) -> None:
    file = sh.tmp / "f.bin"
    file.write_bytes(b"hello")
    good = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    ok = sh(f'check_sha256 "{file}" {good}; echo passed')
    assert (ok.returncode, ok.stdout.strip()) == (0, "passed")
    bad = sh(f'check_sha256 "{file}" {"0" * 64}; echo passed')
    assert bad.returncode == 1
    assert "The download didn't match what was published, so nothing was installed." in bad.stdout
    assert "passed" not in bad.stdout
    assert (sh.tmp / "status.txt").read_text() == "failed"


def test_the_archive_follows_the_machine(sh) -> None:
    sh.fake("uname", 'echo "$FAKE_ARCH"\n')
    arm = sh("uv_archive", env={"FAKE_ARCH": "arm64"})
    intel = sh("uv_archive", env={"FAKE_ARCH": "x86_64"})
    other = sh("uv_archive || echo unsupported", env={"FAKE_ARCH": "ppc"})
    assert arm.stdout.strip() == "uv-aarch64-apple-darwin"
    assert intel.stdout.strip() == "uv-x86_64-apple-darwin"
    assert other.stdout.strip() == "unsupported"


def test_macos_12_is_refused_and_13_passes(sh) -> None:
    old = sh("check_macos 12.7.4; echo passed")
    assert old.returncode == 1 and "Clean Voice needs macOS 13 or later." in old.stdout
    assert "passed" not in old.stdout
    for version in ("13.0", "14.5", "26.1"):
        assert sh(f"check_macos {version}; echo passed").stdout.strip() == "passed"


def test_the_download_failure_line(sh) -> None:
    sh.fake("curl", "exit 22\n")
    result = sh(f'fetch "http://127.0.0.1:9/x" "{sh.tmp}/x"')
    assert result.returncode == 1
    assert "Couldn't reach GitHub. Check your internet connection and run the command again." in result.stdout


def test_developer_tools_are_waited_for_and_a_timeout_says_to_run_again(sh) -> None:
    # Fails twice, then succeeds: the third poll sees the tools.
    sh.fake(
        "sleep",
        f'n=$(cat "{sh.tmp}/xcode-ready" 2>/dev/null || echo 0); echo $((n + 1)) >"{sh.tmp}/xcode-ready"\n'
        f'if [ "$n" -ge 1 ]; then echo 1 >"{sh.tmp}/xcode-ready"; fi\n',
    )
    result = sh("POLL_INTERVAL=0; POLL_TRIES=5; ensure_tools; echo continued")
    assert result.returncode == 0 and result.stdout.strip().endswith("continued")
    assert (
        "Apple is asking to install its developer tools. Click Install and wait; Clean Voice continues by itself."
        in result.stdout
    )
    assert len((sh.tmp / "xcode-installs").read_text().split()) == 1
    # Never succeeding.
    sh.fake("sleep", "exit 0\n")
    (sh.tmp / "xcode-ready").unlink()
    result = sh("POLL_INTERVAL=0; POLL_TRIES=3; ensure_tools; echo continued")
    assert result.returncode == 1 and "continued" not in result.stdout
    assert "Run the same command again once Apple's tools have finished installing." in result.stdout


def test_tools_already_installed_asks_nothing(sh) -> None:
    (sh.tmp / "xcode-ready").write_text("1")
    result = sh("ensure_tools; echo continued")
    assert result.stdout.strip() == "continued"
    assert not (sh.tmp / "xcode-installs").exists()


# ---- Replacing the old copy ---------------------------------------------------------------------------------------------


def test_an_old_applications_copy_is_replaced_in_place(sh) -> None:
    old = make_app(sh.system_apps, "old")
    new = make_app(sh.tmp / "built", "new")
    result = sh(f'swap_app "{new}" "{old}"; echo done')
    assert result.returncode == 0, result.stdout
    assert marker(old) == "new"
    assert sorted(p.name for p in sh.system_apps.iterdir()) == ["Clean Voice.app"]


def test_an_old_copy_in_the_users_applications_is_replaced_there(sh) -> None:
    (sh.home / "Applications").mkdir()
    old = make_app(sh.home / "Applications", "old")
    result = sh("pick_target")
    assert result.stdout.strip() == str(old)


def test_a_system_copy_wins_over_a_user_copy(sh) -> None:
    (sh.home / "Applications").mkdir()
    make_app(sh.home / "Applications", "user")
    system = make_app(sh.system_apps, "system")
    assert sh("pick_target").stdout.strip() == str(system)


def test_no_old_copy_goes_to_applications_when_writable(sh) -> None:
    assert sh("pick_target").stdout.strip() == str(sh.system_apps / "Clean Voice.app")


def test_no_old_copy_and_unwritable_applications_uses_the_users_folder(sh) -> None:
    sh.system_apps.chmod(0o555)
    try:
        if os.access(sh.system_apps, os.W_OK):  # running as root: permissions do not apply
            pytest.skip("this account can write anywhere")
        result = sh("pick_target")
    finally:
        sh.system_apps.chmod(stat.S_IRWXU)
    assert result.stdout.strip() == str(sh.home / "Applications" / "Clean Voice.app")
    assert (sh.home / "Applications").is_dir()


def test_an_old_copy_that_cannot_be_replaced_stops_and_changes_nothing(sh) -> None:
    old = make_app(sh.system_apps, "old")
    new = make_app(sh.tmp / "built", "new")
    sh.system_apps.chmod(0o555)
    try:
        if os.access(sh.system_apps, os.W_OK):
            pytest.skip("this account can write anywhere")
        result = sh(f'swap_app "{new}" "{old}"; echo done')
    finally:
        sh.system_apps.chmod(stat.S_IRWXU)
    assert result.returncode == 1 and "done" not in result.stdout
    assert (
        "The Clean Voice in Applications can't be replaced from your account. Drag it to the Bin, then run the command again."
        in result.stdout
    )
    assert marker(old) == "old"
    assert sorted(p.name for p in sh.system_apps.iterdir()) == ["Clean Voice.app"]


def test_a_failed_move_rolls_back_and_leaves_the_old_copy(sh) -> None:
    old = make_app(sh.system_apps, "old")
    new = make_app(sh.tmp / "built", "new")
    body = f'''mv() {{ case "$1" in *.new) return 1 ;; esac; command mv "$@"; }}
swap_app "{new}" "{old}"; echo done'''
    result = sh(body)
    assert result.returncode == 1 and "done" not in result.stdout
    assert marker(old) == "old"
    assert sorted(p.name for p in sh.system_apps.iterdir()) == ["Clean Voice.app"]
    assert not (sh.tmp / "open.log").exists()  # not update mode: nothing is reopened
    assert (sh.tmp / "status.txt").read_text() == "failed"


# ---- Update mode ------------------------------------------------------------------------------------------------------------


def test_update_mode_writes_swap_waits_for_the_pid_then_swaps(sh) -> None:
    old = make_app(sh.system_apps, "old")
    new = make_app(sh.tmp / "built", "new")
    body = f'''UPDATE=1
sleep 0 & WAIT_PID=$!
wait $WAIT_PID
swap_app "{new}" "{old}"
echo "status=$(cat "$STATUS_FILE")"'''
    result = sh(body)
    assert result.returncode == 0, result.stdout
    assert "status=swap" in result.stdout
    assert marker(old) == "new"


def test_update_mode_says_so_when_the_app_does_not_close(sh) -> None:
    old = make_app(sh.system_apps, "old")
    new = make_app(sh.tmp / "built", "new")
    result = sh(f'UPDATE=1; WAIT_PID=$$; WAIT_TRIES=2; WAIT_SLEEP=0; swap_app "{new}" "{old}"; echo done')
    assert result.returncode == 1 and "done" not in result.stdout
    assert "Clean Voice didn't close, so the old version is still in place." in result.stdout
    assert marker(old) == "old"
    assert not (sh.system_apps / "Clean Voice.app.new").exists()
    assert (sh.tmp / "status.txt").read_text() == "failed"


def test_a_forced_swap_failure_in_update_mode_reopens_the_old_app(sh) -> None:
    old = make_app(sh.system_apps, "old")
    new = make_app(sh.tmp / "built", "new")
    body = f'''mv() {{ case "$1" in *.new) return 1 ;; esac; command mv "$@"; }}
UPDATE=1
swap_app "{new}" "{old}"; echo done'''
    result = sh(body)
    assert result.returncode == 1 and "done" not in result.stdout
    assert "Couldn't replace the old Clean Voice. Your old version is back and was reopened." in result.stdout
    assert marker(old) == "old"
    assert (sh.tmp / "open.log").read_text().strip() == str(old)  # the fake `open` was given the restored old app
    assert (sh.tmp / "status.txt").read_text() == "failed"


def test_a_clean_install_runs_every_step(sh) -> None:
    # The whole of main with the downloads and Briefcase faked: curl copies, tar and unzip make the tree, python runs.
    served = sh.tmp / "served"
    served.mkdir()
    sh.fake("sw_vers", "echo 14.5\n")
    sh.fake("pgrep", "exit 1\n")
    (sh.tmp / "xcode-ready").write_text("1")
    body = """run_logged() { log "> $*"; case "$*" in
  *"unzip"*) mkdir -p "$WORK/source/build/clean-voice/macos/app/Clean Voice.app/Contents"; echo new >"$WORK/source/build/clean-voice/macos/app/Clean Voice.app/Contents/marker" ;;
esac; return 0; }
fetch() { log "fetch $1"; echo x >"$2"; }
check_sha256() { :; }
"""
    # main() runs init_vars again and resets WORK, LOG and the Applications folders in init_vars, so patch init_vars after it.
    body += f'''real_init=$(declare -f init_vars | tail -n +2)
init_vars() {{ eval "$real_init"; APPS_SYSTEM="{sh.system_apps}"; APPS_USER="{sh.home}/Applications"; WORK="{sh.tmp}/work"; LOG="{sh.tmp}/install.log"; STATUS_FILE=""; }}
main
'''
    result = sh(body)
    assert result.returncode == 0, result.stdout
    for number, name in enumerate(
        [
            "Checking your Mac",
            "Getting the build tools",
            "Getting Clean Voice",
            "Getting Python",
            "Building Clean Voice (a few minutes)",
            "Installing",
            "Finishing",
        ],
        start=1,
    ):
        assert f"Step {number} of 7: {name}" in result.stdout
    assert marker(sh.system_apps / "Clean Voice.app") == "new"
    assert not (sh.tmp / "work").exists()  # the setup folder is removed at the end


def test_the_marker_holds_the_hash_of_the_installed_binary(sh) -> None:
    import hashlib

    binary = sh.tmp / "Clean Voice.app" / "Contents" / "MacOS" / "Clean Voice"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"built by the command")
    result = sh(f'write_marker "{sh.tmp / "Clean Voice.app"}"; echo done')
    assert result.returncode == 0, result.stdout
    marker = sh.home / "Library" / "Application Support" / "Clean Voice" / "installed-by-command"
    assert marker.read_text().strip() == hashlib.sha256(b"built by the command").hexdigest()
