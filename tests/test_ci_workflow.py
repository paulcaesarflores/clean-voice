"""The GitHub Actions workflow ``.github/workflows/ci.yml`` (J04): its structure, pinned with PyYAML."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# A `uses:` line pinned to a full commit SHA with its version as a comment.
PINNED_USES = re.compile(r"^\s*(-\s*)?uses:\s*[\w.-]+/[\w.-]+@[0-9a-f]{40}\s+#\s*v\d+(\.\d+)*\s*$")


@pytest.fixture(scope="module")
def text() -> str:
    assert WORKFLOW.is_file(), f"missing {WORKFLOW}"
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow(text: str) -> dict[Any, Any]:
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def jobs(workflow: dict[Any, Any]) -> dict[str, dict[str, Any]]:
    return workflow["jobs"]


def triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # PyYAML reads the bare key `on` as the boolean True (YAML 1.1).
    on = workflow.get("on", workflow.get(True))
    assert isinstance(on, dict), f"no trigger table: {on!r}"
    return on


def steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return job["steps"]


def run_lines(job: dict[str, Any]) -> list[str]:
    """Every `run:` command of a job, one entry per step."""
    return [step["run"] for step in steps(job) if "run" in step]


def uses_steps(job: dict[str, Any], action: str) -> list[dict[str, Any]]:
    return [step for step in steps(job) if str(step.get("uses", "")).startswith(f"{action}@")]


def step_index(job: dict[str, Any], predicate: str) -> int:
    """Index of the first step whose `run` or `uses` contains ``predicate``."""
    for i, step in enumerate(steps(job)):
        if predicate in str(step.get("run", "")) or predicate in str(step.get("uses", "")):
            return i
    raise AssertionError(f"no step contains {predicate!r}")


# W1
def test_workflow_parses(workflow: dict[Any, Any]) -> None:
    assert workflow["name"] == "CI"
    triggers(workflow)


# W2
def test_triggers(workflow: dict[Any, Any], text: str) -> None:
    on = triggers(workflow)
    # workflow_call: release.yml runs this whole workflow before it publishes a release.
    assert set(on) == {"push", "pull_request", "workflow_dispatch", "workflow_call"}
    assert on["push"]["branches"] == ["main"]
    assert "pull_request_target" not in text


# W3
def test_least_privilege(workflow: dict[Any, Any], jobs: dict[str, dict[str, Any]]) -> None:
    assert workflow["permissions"] == {"contents": "read"}
    checkouts = 0
    for name, job in jobs.items():
        assert "permissions" not in job, f"job {name} widens permissions"
        for step in uses_steps(job, "actions/checkout"):
            assert step["with"]["persist-credentials"] is False, f"job {name}: checkout persists credentials"
            checkouts += 1
    assert checkouts == len(jobs)


# W4
def test_actions_pinned_to_commit_sha(text: str) -> None:
    uses = [line for line in text.splitlines() if re.match(r"^\s*(-\s*)?uses:", line)]
    assert uses, "no uses: lines"
    for line in uses:
        assert PINNED_USES.match(line), f"not pinned to a full SHA with a version comment: {line.strip()}"


# W5
def test_runners(jobs: dict[str, dict[str, Any]]) -> None:
    assert set(jobs) == {"checks", "package-windows", "package-macos", "smoke-macos-intel"}
    checks = jobs["checks"]
    assert checks["strategy"]["fail-fast"] is False
    assert {entry["os"] for entry in checks["strategy"]["matrix"]["include"]} == {"windows-latest", "macos-latest"}
    assert checks["runs-on"] == "${{ matrix.os }}"
    assert jobs["package-windows"]["runs-on"] == "windows-latest"
    assert jobs["package-macos"]["runs-on"] == "macos-latest"
    intel = jobs["smoke-macos-intel"]
    assert intel["runs-on"] == "macos-15-intel"
    assert intel["needs"] in ("package-macos", ["package-macos"])
    for name, job in jobs.items():
        # Windows and macOS only (DESIGN, Platforms).
        assert "ubuntu" not in str(job["runs-on"]), f"job {name} runs on Linux"


# W6
def test_every_job_is_bounded(workflow: dict[Any, Any], jobs: dict[str, dict[str, Any]]) -> None:
    for name, job in jobs.items():
        timeout = job.get("timeout-minutes")
        assert isinstance(timeout, int) and 0 < timeout <= 60, f"job {name}: timeout-minutes {timeout!r}"
    assert workflow["concurrency"]["group"]


# W7
def test_python_312_everywhere(jobs: dict[str, dict[str, Any]]) -> None:
    for name, job in jobs.items():
        setups = uses_steps(job, "actions/setup-python")
        assert len(setups) == 1, f"job {name}: {len(setups)} setup-python steps"
        version = setups[0]["with"]["python-version"]
        assert version == "3.12" and isinstance(version, str), f"job {name}: python-version {version!r}"


# W8
def test_checks_run_the_project_gate(jobs: dict[str, dict[str, Any]]) -> None:
    checks = jobs["checks"]
    joined = "\n".join(run_lines(checks))
    gate = [
        "-m pip install -e . --group dev",
        "-m ruff check .",
        "-m ruff format --check .",
        "-m pyright\n",
        "actionlint",
        "-m pytest",
    ]
    position = 0
    for command in gate:
        found = joined.find(command, position)
        assert found >= 0, f"checks job lacks {command!r} (in order)"
        position = found + len(command)


# Kept local only (release r1, user decision): the agent sandbox, its tests and the project's working docs.
NOT_PUBLIC = ("CLAUDE.md", ".claude", "tests/sandbox", "tests/typecheck", "docs/DESIGN", "docs/WORKPLAN", "docs/ui")


def test_ci_needs_no_local_only_file(text: str) -> None:
    # The public repo holds no agent sandbox: CI must not type-check its hooks or run its tests.
    for name in (*NOT_PUBLIC, "CLEAN_VOICE_REQUIRE_TRANSPORT"):
        assert name not in text, name


# W9
@pytest.mark.parametrize(
    ("job_name", "platform", "pattern", "artifact"),
    [
        ("package-windows", "windows", "dist/*.msi", "clean-voice-windows-msi"),
        ("package-macos", "macOS", "dist/*.dmg", "clean-voice-macos-dmg"),
    ],
)
def test_packaging(
    jobs: dict[str, dict[str, Any]], text: str, job_name: str, platform: str, pattern: str, artifact: str
) -> None:
    job = jobs[job_name]
    runs = run_lines(job)
    for command in (
        f"-m briefcase create {platform} --no-input",
        f"-m briefcase build {platform} --no-input",
        f"-m briefcase package {platform} --adhoc-sign --no-input",
    ):
        assert any(command in run for run in runs), f"{job_name} lacks {command!r}"
    uploads = uses_steps(job, "actions/upload-artifact")
    assert len(uploads) == 1
    upload = uploads[0]["with"]
    assert upload["path"] == pattern
    assert upload["name"] == artifact
    assert upload["if-no-files-found"] == "error"
    # The installer is uploaded before the launch check, so a failed launch still leaves it to inspect.
    assert step_index(job, "actions/upload-artifact@") < step_index(job, "tools/ci/smoke_app.py")
    assert step_index(job, "-m briefcase package") < step_index(job, "actions/upload-artifact@")
    # Briefcase only: PyInstaller is banned (DESIGN, Packaging).
    assert "pyinstaller" not in text.lower()


# J10 Step 13: Briefcase leaves the stub's "Copyright (C) 2022" in the exe; LICENSE's line goes in after the build and
# before the package, which takes the exe as it is.
def test_the_exe_gets_the_license_copyright_between_build_and_package(jobs: dict[str, dict[str, Any]]) -> None:
    job = jobs["package-windows"]
    stamp = step_index(job, 'tools/ci/set_copyright.py "build/clean-voice/windows/app/src/Clean Voice.exe"')
    assert step_index(job, "-m briefcase build windows") < stamp < step_index(job, "-m briefcase package windows")


# W10
def test_launch_checks_wired(jobs: dict[str, dict[str, Any]]) -> None:
    assert any("tools/ci/smoke_app.py msi dist" in run for run in run_lines(jobs["package-windows"]))
    assert any("tools/ci/smoke_app.py dmg dist --arch arm64" in run for run in run_lines(jobs["package-macos"]))
    intel = jobs["smoke-macos-intel"]
    downloads = uses_steps(intel, "actions/download-artifact")
    assert len(downloads) == 1
    assert downloads[0]["with"]["name"] == "clean-voice-macos-dmg"
    assert downloads[0]["with"]["path"] == "dist"
    assert any("tools/ci/smoke_app.py dmg dist --arch x86_64" in run for run in run_lines(intel))
    assert step_index(intel, "actions/download-artifact@") < step_index(intel, "tools/ci/smoke_app.py")


# J10 T23/T24: the VST host's tests load the project's own test plug-in, built before the tests on both OSes, and the
# editor test runs again on the Windows desktop (the Tests step is offscreen, where it skips).
def test_checks_build_the_vst_test_plugin_before_the_tests(jobs: dict[str, dict[str, Any]]) -> None:
    checks = jobs["checks"]
    build = step_index(checks, "tools/vst_test_plugin/build.py")
    assert "--group native" in steps(checks)[build]["run"]
    assert "if" not in steps(checks)[build]  # both OSes
    tests = next(i for i, s in enumerate(steps(checks)) if s.get("name") == "Tests")
    assert build < tests
    editor = step_index(checks, "tests/test_vst_editor.py")
    assert editor > tests
    assert steps(checks)[editor]["if"] == "runner.os == 'Windows'"
    assert steps(checks)[editor]["env"] == {"QT_QPA_PLATFORM": "windows"}


# The first CI run failed inside Briefcase's ad-hoc signing on librnnoise.dylib ("main executable failed strict
# validation"); the dylib is signed and strictly verified on its own before Briefcase, so that cause is named.
def test_macos_signs_and_strictly_verifies_the_rnnoise_dylib_before_briefcase(jobs: dict[str, dict[str, Any]]) -> None:
    job = jobs["package-macos"]
    check = step_index(job, "codesign --verify --strict")
    run = steps(job)[check]["run"]
    assert "src/clean_voice/native/librnnoise.dylib" in run
    assert "codesign --sign - --force" in run
    assert check < step_index(job, "-m briefcase create macOS")


# App Nap stays off in the built app, as in OBS (frontend/cmake/macos/Info.plist.in): the generated Info.plist is
# checked after the build, since only the macOS runner has Briefcase's template output.
def test_macos_checks_app_nap_is_disabled_in_the_built_info_plist(jobs: dict[str, dict[str, Any]]) -> None:
    job = jobs["package-macos"]
    check = step_index(job, ':LSAppNapIsDisabled"')
    run = steps(job)[check]["run"]
    assert '/usr/libexec/PlistBuddy -c "Print :LSAppNapIsDisabled"' in run
    assert '"build/clean-voice/macos/app/Clean Voice.app/Contents/Info.plist"' in run
    assert '= "true"' in run
    assert step_index(job, "-m briefcase build macOS") < check
