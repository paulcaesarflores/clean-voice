"""Prepare a GitHub release's files (release workflow). Standard library only.

    release_assets.py OUT --source ZIP --msi DIR --dmg DIR --tag vX.Y.Z [--version X.Y.Z] [--scripts DIR]
    release_assets.py --check-only --tag vX.Y.Z

A release offers two ways to install (B019, user, 2026-11-21), under stable names so ``releases/latest/download/<name>``
is a permanent link:

* the installers: the one *.msi and the one *.dmg that CI built, plus the checksums the in-app updater verifies for an
  app installed by them (``clean_voice.updates``): ``SHA256SUMS`` for both, and one ``<name>.sha256`` per installer, each
  in sha256sum's text format (``<hex>  <name>``);
* the install command's files: ``Clean-Voice-source.zip`` (``git archive`` of the tag, which the scripts build from) and
  the two install scripts (``install.ps1``, ``install.sh``) from ``tools/install/``, each stamped with the version and
  the zip's SHA-256 (the ``@@VERSION@@`` and ``@@SOURCE_SHA256@@`` placeholders, each exactly once). The script checks
  the zip against the stamp: that catches a corrupt download, not a tampered release, since both come from one host.

``--msi`` and ``--dmg`` go together; without them only the command's files are written (the install smoke test).
The tag must be ``v`` + the app version in all three places it is kept: pyproject.toml's ``[project].version`` and
``[tool.briefcase].version`` (what Briefcase builds the MSI/DMG from), and ``clean_voice.__version__`` (what the updater
compares with the tag). A mismatch would offer the same update forever. ``--check-only`` checks only the tag, before CI
builds anything.

Exit code 0 when OUT is ready; 1 when a check failed (the reason is printed, and OUT is not created).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import shutil
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WINDOWS_NAME = "Clean-Voice-Windows.msi"
MAC_NAME = "Clean-Voice-macOS.dmg"
SUMS_NAME = "SHA256SUMS"  # must match clean_voice.updates.SUMS_NAME
SOURCE_NAME = "Clean-Voice-source.zip"
SCRIPT_NAMES = ("install.ps1", "install.sh")
SCRIPTS_DIR = ROOT / "tools" / "install"


class ReleaseError(Exception):
    """A release file check failed; the message says why."""


def init_version(path: Path) -> str:
    """The string literal assigned to ``__version__`` in ``path``."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and [getattr(target, "id", None) for target in node.targets] == ["__version__"]
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise ReleaseError(f"no __version__ = '...' in {path}")


def app_versions(root: Path = ROOT) -> dict[str, str]:
    """Every place the app version is kept, by where it is."""
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return {
        "pyproject.toml [project].version": data["project"]["version"],
        "pyproject.toml [tool.briefcase].version": data["tool"]["briefcase"]["version"],
        "src/clean_voice/__init__.py __version__": init_version(root / "src" / "clean_voice" / "__init__.py"),
    }


def check_tag(tag: str, versions: dict[str, str]) -> None:
    wrong = [f"{where} = {version!r}" for where, version in versions.items() if tag != f"v{version}"]
    if wrong:
        raise ReleaseError(f"tag {tag!r} is not 'v' + the app version in: {'; '.join(wrong)}")


def single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ReleaseError(f"expected exactly one {pattern} in {directory}, found {[m.name for m in matches]}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stamp(text: str, name: str, values: dict[str, str]) -> str:
    """``text`` with each placeholder replaced; each must occur exactly once."""
    for placeholder, value in values.items():
        count = text.count(placeholder)
        if count != 1:
            raise ReleaseError(f"{name} has {count} {placeholder}, expected exactly one")
        text = text.replace(placeholder, value)
    return text


def prepare(
    out: Path,
    source: Path,
    scripts: Path,
    tag: str,
    versions: dict[str, str],
    msi_dir: Path | None = None,
    dmg_dir: Path | None = None,
) -> None:
    check_tag(tag, versions)
    if source.suffix != ".zip":
        raise ReleaseError(f"{source.name} is not a .zip: the source archive is a zip")
    if not source.is_file():
        raise ReleaseError(f"source archive {source} not found")
    if (msi_dir is None) != (dmg_dir is None):
        raise ReleaseError("--msi and --dmg go together")
    installers = {}
    if msi_dir is not None and dmg_dir is not None:
        installers = {WINDOWS_NAME: single(msi_dir, "*.msi"), MAC_NAME: single(dmg_dir, "*.dmg")}
    digest = sha256(source)
    values = {"@@VERSION@@": tag.removeprefix("v"), "@@SOURCE_SHA256@@": digest}
    stamped = {}
    for name in SCRIPT_NAMES:  # all stamped before OUT exists, so a bad script leaves nothing behind
        path = scripts / name
        if not path.is_file():
            raise ReleaseError(f"{path} not found")
        stamped[name] = stamp(path.read_bytes().decode("ascii"), name, values)
    out.mkdir(parents=True)
    lines = []
    for name, installer in installers.items():
        shutil.copyfile(installer, out / name)
        line = f"{sha256(out / name)}  {name}\n"
        (out / f"{name}.sha256").write_text(line, encoding="utf-8", newline="\n")
        lines.append(line)
        print(line, end="", flush=True)
    if lines:
        (out / SUMS_NAME).write_text("".join(lines), encoding="utf-8", newline="\n")
    shutil.copyfile(source, out / SOURCE_NAME)
    for name, text in stamped.items():
        (out / name).write_bytes(text.encode("ascii"))
    print(f"{digest}  {SOURCE_NAME}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release_assets", description="Prepare Clean Voice's release files.")
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--source", type=Path, help="the source zip (git archive of the tag)")
    parser.add_argument("--scripts", type=Path, default=SCRIPTS_DIR, help="folder holding the install scripts")
    parser.add_argument("--msi", type=Path, help="folder holding the one built *.msi")
    parser.add_argument("--dmg", type=Path, help="folder holding the one built *.dmg")
    parser.add_argument("--tag", required=True, help="the pushed tag, vX.Y.Z")
    parser.add_argument("--version", default=None, help="the app version (default: all three places it is kept)")
    parser.add_argument("--check-only", action="store_true", help="only check the tag against the app version")
    args = parser.parse_args(argv)
    if not args.check_only and None in (args.out, args.source):
        parser.error("OUT and --source are required unless --check-only")
    try:
        versions = {"--version": args.version} if args.version else app_versions()
        if args.check_only:
            check_tag(args.tag, versions)
        else:
            prepare(args.out, args.source, args.scripts, args.tag, versions, args.msi, args.dmg)
    except ReleaseError as exc:
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
