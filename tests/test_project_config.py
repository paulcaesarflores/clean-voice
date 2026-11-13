"""Project metadata checks for the Briefcase project in ``pyproject.toml`` (J03, J04)."""

import ast
import os
import re
import sys
import tomllib
import uuid
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
LICENSE = ROOT / "LICENSE"

# A requirement pinned to one exact version, e.g. "pyright[nodejs]==1.1.414".
PINNED = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<extras>\[[^\]]+\])?==\d[\w.]*$")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def briefcase(pyproject: dict[str, Any]) -> dict[str, Any]:
    return pyproject["tool"]["briefcase"]


@pytest.fixture(scope="module")
def app(briefcase: dict[str, Any]) -> dict[str, Any]:
    return briefcase["app"]["clean-voice"]


def test_project_name(pyproject: dict[str, Any], briefcase: dict[str, Any]) -> None:
    assert pyproject["project"]["name"] == "clean-voice"
    assert briefcase["project_name"] == "clean-voice"
    assert list(briefcase["app"]) == ["clean-voice"]


def test_formal_name(app: dict[str, Any]) -> None:
    assert app["formal_name"] == "Clean Voice"


def test_bundle_id(briefcase: dict[str, Any]) -> None:
    # Reverse DNS of the owner's GitHub Pages domain, paulcaesarflores.github.io (release r1, m3).
    assert briefcase["bundle"] == "io.github.paulcaesarflores.cleanvoice"


def test_mac_login_item_label_is_the_bundle_id(briefcase: dict[str, Any]) -> None:
    from clean_voice import autostart

    assert autostart.LABEL == briefcase["bundle"]


def test_one_version_everywhere(pyproject: dict[str, Any], briefcase: dict[str, Any]) -> None:
    # Release r1, F1: the release checks the tag, Briefcase builds the MSI/DMG from [tool.briefcase].version, and
    # the updater compares the tag with clean_voice.__version__. Any drift offers the same update forever.
    import clean_voice

    assert pyproject["project"]["version"] == briefcase["version"] == clean_voice.__version__


def test_author(pyproject: dict[str, Any], briefcase: dict[str, Any]) -> None:
    # The package's author is the person; Briefcase's own `author` overrides it for the bundles (DESIGN, Install
    # folder: the installer's publisher is "Clean Voice"). No email is published.
    assert pyproject["project"]["authors"] == [{"name": "Paul Caesar Flores"}]
    assert briefcase["author"] == "Clean Voice"
    assert "author_email" not in briefcase


def test_installer_publisher_is_the_app() -> None:
    # Briefcase fills the MSI Manufacturer, its `Software\<author>\<app>` registry key, the exe's CompanyName and the
    # bundle METADATA Author from the resolved app config, so no value there may carry the person's name. The
    # licence text (LICENSE) keeps the copyright holder.
    from briefcase.config import parse_config
    from briefcase.console import Console

    for platform in ("windows", "macOS"):
        _, apps = parse_config(PYPROJECT, platform=platform, output_format="app", console=Console(input_enabled=False))
        app = apps["clean-voice"]
        assert app["author"] == "Clean Voice", platform
        assert app.get("author_email") is None, platform
        leaks = [key for key, value in app.items() if "Paul Caesar Flores" in str(value)]
        assert not leaks, f"{platform}: {leaks}"


def test_windows_install_folder_is_named_for_the_app(app: dict[str, Any]) -> None:
    # DESIGN, Install folder (user, 2026-11-12): `…\Programs\Clean Voice`, not `…\Programs\<author>\Clean Voice`.
    # Briefcase's Windows template nests the install folder under the author unless this is false.
    assert app["windows"]["use_full_install_path"] is False


def test_homepage(pyproject: dict[str, Any], briefcase: dict[str, Any]) -> None:
    homepage = pyproject["project"]["urls"]["Homepage"]
    assert re.fullmatch(r"https://github\.com/[A-Za-z0-9_-]+/[A-Za-z0-9._-]+", homepage), homepage
    # One place per value: Briefcase reads `url` from [project.urls] Homepage.
    assert "url" not in briefcase


def test_briefcase_reads_homepage(pyproject: dict[str, Any]) -> None:
    from briefcase.config import parse_config
    from briefcase.console import Console

    _, apps = parse_config(PYPROJECT, platform="windows", output_format="app", console=Console(input_enabled=False))
    assert apps["clean-voice"]["url"] == pyproject["project"]["urls"]["Homepage"]


def test_windows_guid(app: dict[str, Any], briefcase: dict[str, Any]) -> None:
    # The MSI UpgradeCode. Pinned to the value Briefcase derived from the J03 bundle id (io.github.cleanvoice), so a
    # bundle-id change, like release r1's, cannot break Windows upgrades. Briefcase derives it from the reversed
    # bundle identifier (briefcase/platforms/windows/__init__.py).
    guid = app["windows"]["guid"]
    assert guid == "77f9ae70-b7d9-5c5d-a2a8-380c042d321d"
    assert guid == str(uuid.uuid5(uuid.NAMESPACE_DNS, "clean-voice.cleanvoice.github.io"))
    # Without the pin, the current bundle id would give another UpgradeCode.
    reversed_domain = ".".join(f"{briefcase['bundle']}.clean-voice".split(".")[::-1])
    assert guid != str(uuid.uuid5(uuid.NAMESPACE_DNS, reversed_domain))


def test_homepage_owner_is_set_in_ci(pyproject: dict[str, Any]) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        pytest.skip("runs on GitHub Actions only: there the Homepage owner placeholder must be replaced")
    homepage = pyproject["project"]["urls"]["Homepage"]
    assert "REPLACE_WITH_GITHUB_OWNER" not in homepage, (
        "replace the owner: in pyproject.toml, under [project.urls], set "
        'Homepage = "https://github.com/<owner>/clean-voice" to the repository\'s GitHub user or org'
    )


def test_description_fits_briefcase(pyproject: dict[str, Any], app: dict[str, Any]) -> None:
    # Briefcase warns above 80 characters; some platforms truncate longer descriptions.
    assert len(pyproject["project"]["description"]) <= 80
    assert len(app["description"]) <= 80


def test_sources(app: dict[str, Any]) -> None:
    assert app["sources"] == ["src/clean_voice"]


def test_license_is_mit(pyproject: dict[str, Any]) -> None:
    assert pyproject["project"]["license"] == "MIT"
    assert pyproject["project"]["license-files"] == ["LICENSE"]
    text = LICENSE.read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 Paul Caesar Flores" in text
    assert "Permission is hereby granted, free of charge, to any person obtaining a copy" in text
    assert 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND' in text


def test_requires_python_is_312(pyproject: dict[str, Any]) -> None:
    assert pyproject["project"]["requires-python"] == ">=3.12,<3.13"


def test_pyside6_pinned(pyproject: dict[str, Any]) -> None:
    deps = pyproject["project"]["dependencies"]
    pyside = [d for d in deps if _normalize(d).startswith("pyside6-essentials")]
    assert pyside == ["PySide6-Essentials==6.11.3"]


def test_no_pyinstaller() -> None:
    # Briefcase only: PyInstaller is banned (DESIGN, Packaging).
    assert "pyinstaller" not in PYPROJECT.read_text(encoding="utf-8").lower()


def test_dev_group_pinned(pyproject: dict[str, Any]) -> None:
    dev = pyproject["dependency-groups"]["dev"]
    found: dict[str, str] = {}
    for req in dev:
        match = PINNED.match(req)
        assert match, f"dev requirement not pinned with ==: {req!r}"
        found[_normalize(match["name"])] = match["extras"] or ""
    for tool in ("briefcase", "pytest", "pytest-qt", "ruff", "pyright"):
        assert tool in found, f"dev group lacks {tool}"
    assert found["pyright"] == "[nodejs]"


def test_platform_tables(app: dict[str, Any]) -> None:
    # Windows and macOS only; no Linux (DESIGN, Platforms) and no mobile or web targets.
    platforms = {key for key, value in app.items() if isinstance(value, dict) and key != "permission"}
    assert platforms == {"macOS", "windows"}
    mac = app["macOS"]
    assert mac["min_os_version"] == "13.0"
    assert mac["universal_build"] is True
    assert mac["requires"] == ["std-nslog~=2.0.0"]


def test_microphone_permission(app: dict[str, Any]) -> None:
    # macOS refuses microphone access to an app without a usage description (NSMicrophoneUsageDescription).
    assert app["permission"]["microphone"].strip()


def test_audio_dependencies_pinned(pyproject: dict[str, Any]) -> None:
    deps = pyproject["project"]["dependencies"]
    reqs = [req.split(";")[0].strip() for req in deps]  # the environment marker, if any, is not part of the pin
    for req in reqs:
        assert PINNED.match(req), f"dependency not pinned with ==: {req!r}"
    names = {_normalize(PINNED.match(req)["name"]) for req in reqs}  # type: ignore[index]
    assert {"numpy", "scipy", "samplerate", "sounddevice"} <= names
    # Echo removal (J06): Windows only, so the macOS build never needs these.
    markers = {_normalize(req.split(";")[0].split("==")[0]): req.partition(";")[2].strip() for req in deps}
    assert markers["pywebrtc-audio"] == markers["pyaudiowpatch"] == "sys_platform == 'win32'"
    # Numba has no macOS x86_64 wheels, which the universal macOS build needs.
    assert "numba" not in names


def test_rnnoise_libraries_bundled() -> None:
    native = ROOT / "src" / "clean_voice" / "native"
    assert (native / "rnnoise.dll").is_file()
    assert (native / "librnnoise.dylib").is_file()
    assert (native / "RNNOISE-COPYING.txt").read_text(encoding="utf-8").startswith("Copyright")


def test_interpreter_is_312() -> None:
    assert sys.version_info[:2] == (3, 12)


def test_tool_config(pyproject: dict[str, Any]) -> None:
    # The safety net always allows `.cache/`, so every role can run the tools without cache warnings.
    tool = pyproject["tool"]
    pytest_opts = tool["pytest"]["ini_options"]
    assert pytest_opts["cache_dir"] == ".cache/pytest"
    assert pytest_opts["testpaths"] == ["tests"]
    assert tool["ruff"]["cache-dir"].startswith(".cache/")
    pyright = tool["pyright"]
    for folder in ("src", "tests", "tools"):
        assert folder in pyright["include"], f"pyright include lacks {folder}"
    # tools/sandbox is the gitignored staging area for hook changes (checked through tests/typecheck/hooks).
    assert "tools/sandbox" in pyright["exclude"]
    # pyright 1.1.414 always adds its `**/.*` exclude below the config's own folder, so the root config cannot see
    # `.claude/hooks` (plan D1). Listing it there would mislead.
    assert not any(".claude" in entry for entry in pyright["include"])


def test_windows_bundle_leaves_out_the_unused_qml_folder(pyproject: dict[str, Any]) -> None:
    # PySide6's qml folder holds Qt Quick modules and two build-output folders (objects-Debug, objects-RelWithDebInfo):
    # its paths reach 164 characters inside the app folder, so the MSI failed (error 1320 / 1304, path too long) once
    # the install folder passed about 96 characters, as a per-user install under a long Windows user name can (J05).
    # Without it the deepest path is 93 characters.
    windows = pyproject["tool"]["briefcase"]["app"]["clean-voice"]["windows"]
    assert "**/app_packages/PySide6/qml" in windows["cleanup_paths"]
    assert "**/app_packages/*.whl" in windows["cleanup_paths"]  # an empty scipy wheel the install leaves


def test_the_app_uses_no_qml() -> None:
    # Why leaving the qml folder out is safe: Clean Voice is a Qt Widgets app.
    sources = (ROOT / "src" / "clean_voice").glob("*.py")
    assert not [p.name for p in sources if re.search(r"\bQt(Qml|Quick)", p.read_text(encoding="utf-8"))]


def _not_installed_everywhere(pyproject: dict[str, Any]) -> set[str]:
    """Import names of what CI's `pip install -e . --group dev` leaves out on some runner: platform-marked
    dependencies and every other dependency group."""
    reqs = [req for req in pyproject["project"]["dependencies"] if ";" in req]
    reqs += [req for group, items in pyproject["dependency-groups"].items() if group != "dev" for req in items]
    return {re.split(r"[\[=;<> ]", req, maxsplit=1)[0].lower().replace("-", "_") for req in reqs}


def test_imports_of_packages_ci_does_not_install_skip_pyright(pyproject: dict[str, Any]) -> None:
    # The first CI run's pyright failed on soundfile (tools group) on both OSes and on pywebrtc_audio (Windows only)
    # on macOS: installed here, absent there. Each such import carries pyright's targeted ignore.
    optional = _not_installed_everywhere(pyproject)
    assert {"soundfile", "pywebrtc_audio", "pyaudiowpatch", "ziglang"} <= optional
    files = [p for folder in ("src", "tests", "tools") for p in (ROOT / folder).rglob("*.py")]
    files = [p for p in files if "sandbox" not in p.relative_to(ROOT).parts[:2]]
    missing = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            ignored = "# pyright: ignore[reportMissingImports]" in lines[node.lineno - 1]
            if any(name.split(".")[0] in optional for name in names) and not ignored:
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not missing, f"imports pyright cannot resolve on every CI runner, without the ignore: {missing}"
