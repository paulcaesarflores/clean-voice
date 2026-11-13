"""The release-assets script ``tools/ci/release_assets.py``: stable installer names and checksum files that the
in-app updater (``clean_voice.updates``) downloads and verifies."""

import hashlib
import importlib.util
import io
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from clean_voice import updates
from clean_voice.updates import CheckResult, download_verified

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "ci" / "release_assets.py"
VERSION = "0.1.0"
MSI = b"msi bytes" * 500
DMG = b"dmg bytes" * 700


@pytest.fixture(scope="module")
def assets() -> ModuleType:
    # tools/ is not a package and not on the import path: load the script from its file.
    spec = importlib.util.spec_from_file_location("release_assets", SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def built(tmp_path: Path) -> tuple[Path, Path]:
    """The two CI artifacts as download-artifact leaves them: Briefcase's own file names."""
    msi_dir, dmg_dir = tmp_path / "msi", tmp_path / "dmg"
    msi_dir.mkdir()
    dmg_dir.mkdir()
    (msi_dir / f"Clean Voice-{VERSION}.msi").write_bytes(MSI)
    (dmg_dir / f"Clean Voice-{VERSION}.dmg").write_bytes(DMG)
    return msi_dir, dmg_dir


def run(assets: ModuleType, tmp_path: Path, tag: str = f"v{VERSION}") -> tuple[int, Path]:
    msi_dir, dmg_dir = built(tmp_path)
    out = tmp_path / "release"
    code = assets.main([str(out), "--msi", str(msi_dir), "--dmg", str(dmg_dir), "--tag", tag, "--version", VERSION])
    return code, out


def test_stable_names_and_checksum_files(assets: ModuleType, tmp_path: Path) -> None:
    code, out = run(assets, tmp_path)
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "Clean-Voice-Windows.msi",
        "Clean-Voice-Windows.msi.sha256",
        "Clean-Voice-macOS.dmg",
        "Clean-Voice-macOS.dmg.sha256",
        "SHA256SUMS",
    ]
    msi_sum, dmg_sum = hashlib.sha256(MSI).hexdigest(), hashlib.sha256(DMG).hexdigest()
    # sha256sum's text format, `<hex>  <name>`: what `sha256sum -c` and updates.parse_sums read.
    assert (out / "Clean-Voice-Windows.msi.sha256").read_text() == f"{msi_sum}  Clean-Voice-Windows.msi\n"
    assert (out / "Clean-Voice-macOS.dmg.sha256").read_text() == f"{dmg_sum}  Clean-Voice-macOS.dmg\n"
    assert updates.parse_sums((out / "SHA256SUMS").read_text()) == {
        "Clean-Voice-Windows.msi": msi_sum,
        "Clean-Voice-macOS.dmg": dmg_sum,
    }


class Served(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@pytest.mark.parametrize(
    ("platform", "name"), [("win32", "Clean-Voice-Windows.msi"), ("darwin", "Clean-Voice-macOS.dmg")]
)
def test_the_updater_installs_these_assets(assets: ModuleType, tmp_path: Path, platform: str, name: str) -> None:
    _, out = run(assets, tmp_path)
    base = "https://github.com/someone/clean-voice/releases/download/v0.2.0/"
    files = sorted(p.name for p in out.iterdir())
    result = CheckResult("newer", "", "0.2.0", tuple({"name": f, "url": base + f} for f in files))
    folder = tmp_path / "download"
    folder.mkdir()
    path = download_verified(
        result,
        platform,
        folder,
        lambda _: None,
        opener=lambda r, timeout: Served((out / r.full_url.rsplit("/", 1)[1]).read_bytes()),
    )
    assert path.name == name
    assert path.read_bytes() == (out / name).read_bytes()


def test_a_tag_that_is_not_the_app_version_is_refused(assets: ModuleType, tmp_path: Path) -> None:
    # The updater compares the tag with the installed version: a mismatch would offer the same update forever.
    code, out = run(assets, tmp_path, tag="v0.2.0")
    assert code == 1
    assert not out.exists()


def test_exactly_one_installer_per_artifact(assets: ModuleType, tmp_path: Path) -> None:
    msi_dir, dmg_dir = built(tmp_path)
    (msi_dir / "other.msi").write_bytes(b"x")
    code = assets.main([str(tmp_path / "out"), "--msi", str(msi_dir), "--dmg", str(dmg_dir), "--tag", "v0.1.0"])
    assert code == 1


def test_versions_default_to_all_three_sources(assets: ModuleType) -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    assert list(assets.app_versions().values()) == [version, version, version]


def fake_project(tmp_path: Path, project: str, briefcase: str, init: str) -> Path:
    root = tmp_path / "proj"
    (root / "src" / "clean_voice").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nversion = "{project}"\n\n[tool.briefcase]\nversion = "{briefcase}"\n', encoding="utf-8"
    )
    (root / "src" / "clean_voice" / "__init__.py").write_text(f'"""Doc."""\n\n__version__ = "{init}"\n')
    return root


@pytest.mark.parametrize(
    ("project", "briefcase", "init"),
    [
        ("0.2.0", "0.1.0", "0.1.0"),
        ("0.2.0", "0.2.0", "0.1.0"),
        ("0.2.0", "0.1.0", "0.2.0"),
        ("0.1.0", "0.2.0", "0.2.0"),
    ],
)
def test_a_tag_is_refused_unless_every_version_matches(
    assets: ModuleType, tmp_path: Path, project: str, briefcase: str, init: str
) -> None:
    # Release r1, F1: [project].version alone is not what the app reports (__version__) or what Briefcase builds
    # ([tool.briefcase].version). Bumping one of the three and tagging must fail.
    versions = assets.app_versions(fake_project(tmp_path, project, briefcase, init))
    with pytest.raises(assets.ReleaseError, match="v0.2.0"):
        assets.check_tag("v0.2.0", versions)


def test_a_tag_matching_every_version_passes(assets: ModuleType, tmp_path: Path) -> None:
    assets.check_tag("v0.2.0", assets.app_versions(fake_project(tmp_path, "0.2.0", "0.2.0", "0.2.0")))


def test_check_only_needs_no_installers(assets: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    # Release r1, m5: the release workflow runs this first, so a mistyped tag fails before the long CI.
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert assets.main(["--check-only", "--tag", f"v{version}"]) == 0
    assert assets.main(["--check-only", "--tag", "v99.0.0"]) == 1
    assert "v99.0.0" in capsys.readouterr().err
