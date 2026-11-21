"""The release and Pages workflows (``.github/workflows/release.yml``, ``pages.yml``): their structure, pinned with
PyYAML. actionlint checks their syntax."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PINNED_USES = re.compile(r"^\s*(-\s*)?uses:\s*[\w.-]+/[\w.-]+@[0-9a-f]{40}\s+#\s*v\d+(\.\d+)*\s*$")


def load(name: str) -> tuple[str, dict[Any, Any]]:
    path = WORKFLOWS / name
    assert path.is_file(), f"missing {path}"
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert isinstance(data, dict)
    return text, data


def triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    on = workflow.get("on", workflow.get(True))  # PyYAML reads the bare key `on` as True
    assert isinstance(on, dict), f"no trigger table: {on!r}"
    return on


def steps_with(job: dict[str, Any], action: str) -> list[dict[str, Any]]:
    return [s for s in job.get("steps", []) if str(s.get("uses", "")).startswith(f"{action}@")]


def runs(job: dict[str, Any]) -> str:
    return "\n".join(s["run"] for s in job.get("steps", []) if "run" in s)


@pytest.mark.parametrize("name", ["release.yml", "pages.yml"])
def test_common_hygiene(name: str) -> None:
    text, workflow = load(name)
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is False  # never cut a release or a deploy in half
    assert "pull_request_target" not in text
    assert "pyinstaller" not in text.lower()
    for line in text.splitlines():
        if re.match(r"^\s*(-\s*)?uses:\s*\w", line):  # every remote action; ./ is this repo's own ci.yml
            assert PINNED_USES.match(line), f"{name}: not pinned to a full SHA with a version comment: {line.strip()}"
    for job_name, job in workflow["jobs"].items():
        if "runs-on" in job:
            assert job["runs-on"] in ("windows-latest", "macos-latest"), f"{name}/{job_name}: {job['runs-on']}"
            assert isinstance(job.get("timeout-minutes"), int) and job["timeout-minutes"] <= 60
        for step in steps_with(job, "actions/checkout"):
            assert step["with"]["persist-credentials"] is False
        for step in steps_with(job, "actions/setup-python"):
            assert step["with"]["python-version"] == "3.12"


def test_release_runs_all_of_ci_then_publishes() -> None:
    _, workflow = load("release.yml")
    assert triggers(workflow) == {"push": {"tags": ["v*"]}}
    jobs = workflow["jobs"]
    assert set(jobs) == {"tag", "ci", "release", "pages"}
    # Release r1, m5: the tag is checked against the app version first, so a mistyped tag fails in seconds.
    tag = jobs["tag"]
    assert "needs" not in tag
    assert "permissions" not in tag  # the workflow's contents: read
    assert 'tools/ci/release_assets.py --check-only --tag "$GITHUB_REF_NAME"' in runs(tag)
    # The whole CI workflow: checks, the MSI and the universal DMG, every launch check (Intel included).
    assert jobs["ci"]["uses"] == "./.github/workflows/ci.yml"
    assert jobs["ci"]["needs"] in ("tag", ["tag"])
    assert "permissions" not in jobs["ci"]
    release = jobs["release"]
    assert release["needs"] in ("ci", ["ci"])
    assert release["permissions"] == {"contents": "write"}  # the only job that may write
    downloads = {s["with"]["name"]: s["with"]["path"] for s in steps_with(release, "actions/download-artifact")}
    assert set(downloads) == {"clean-voice-windows-msi", "clean-voice-macos-dmg"}
    commands = runs(release)
    prepare = f"tools/ci/release_assets.py release --msi {downloads['clean-voice-windows-msi']}"
    assert prepare in commands
    assert f"--dmg {downloads['clean-voice-macos-dmg']}" in commands
    # B019: the source zip and the two stamped install scripts are published beside the installers.
    assert 'git archive --format=zip -o Clean-Voice-source.zip "$GITHUB_REF_NAME"' in commands
    assert "--source Clean-Voice-source.zip" in commands
    assert '--tag "$GITHUB_REF_NAME"' in commands
    assert re.search(r'gh release create "\$GITHUB_REF_NAME" release/\*', commands)
    assert "--verify-tag" in commands


def test_ci_is_callable() -> None:
    _, ci = load("ci.yml")
    assert "workflow_call" in triggers(ci)


def test_pages_is_deployed_only_after_the_release_exists() -> None:
    # B019-1: the page links files of the release. A release made with GITHUB_TOKEN does not trigger a
    # `release: published` workflow, so release.yml calls pages.yml itself, once the release job is done.
    _, release = load("release.yml")
    pages = release["jobs"]["pages"]
    assert pages["needs"] in ("release", ["release"])
    assert pages["uses"] == "./.github/workflows/pages.yml"
    _, workflow = load("pages.yml")
    on = triggers(workflow)
    assert set(on) == {"workflow_call", "workflow_dispatch"}  # no push trigger: never before the release
    (job,) = workflow["jobs"].values()
    assert job["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}
    assert job["environment"]["name"] == "github-pages"
    (upload,) = steps_with(job, "actions/upload-pages-artifact")
    assert upload["with"]["path"] == "docs/site"
    assert len(steps_with(job, "actions/deploy-pages")) == 1


def test_the_pages_caller_grants_every_permission_the_called_workflow_asks_for() -> None:
    # B019-10: a called workflow cannot hold more than its caller job; without these grants the release run fails at startup.
    _, release = load("release.yml")
    _, pages = load("pages.yml")
    granted = release["jobs"]["pages"]["permissions"]
    for job in pages["jobs"].values():
        for permission, level in job.get("permissions", {}).items():
            assert granted.get(permission) == level, (
                f"the pages job in release.yml does not grant {permission}: {level}"
            )
    assert set(granted) == {"contents", "pages", "id-token"}
