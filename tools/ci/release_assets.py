"""Prepare a GitHub release's files (release workflow). Standard library only.

    release_assets.py OUT --msi DIR --dmg DIR --tag vX.Y.Z [--version X.Y.Z]
    release_assets.py --check-only --tag vX.Y.Z

Copies the one *.msi and the one *.dmg that CI built to OUT under stable names, so
``releases/latest/download/<name>`` is a permanent link, and writes the checksums the in-app updater verifies
(``clean_voice.updates``): ``SHA256SUMS`` for both installers, plus one ``<name>.sha256`` per installer, each in
sha256sum's text format (``<hex>  <name>``). The tag must be ``v`` + the app version in all three places it is kept:
pyproject.toml's ``[project].version`` and ``[tool.briefcase].version`` (what Briefcase builds the MSI/DMG from), and
``clean_voice.__version__`` (what the updater compares with the tag). A mismatch would offer the same update forever.
``--check-only`` checks only the tag, before CI builds anything.

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


def prepare(out: Path, msi_dir: Path, dmg_dir: Path, tag: str, versions: dict[str, str]) -> None:
    check_tag(tag, versions)
    sources = {WINDOWS_NAME: single(msi_dir, "*.msi"), MAC_NAME: single(dmg_dir, "*.dmg")}
    out.mkdir(parents=True)
    lines = []
    for name, source in sources.items():
        shutil.copyfile(source, out / name)
        line = f"{sha256(out / name)}  {name}\n"
        (out / f"{name}.sha256").write_text(line, encoding="utf-8", newline="\n")
        lines.append(line)
        print(line, end="", flush=True)
    (out / SUMS_NAME).write_text("".join(lines), encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release_assets", description="Prepare Clean Voice's release files.")
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--msi", type=Path, help="folder holding the one built *.msi")
    parser.add_argument("--dmg", type=Path, help="folder holding the one built *.dmg")
    parser.add_argument("--tag", required=True, help="the pushed tag, vX.Y.Z")
    parser.add_argument("--version", default=None, help="the app version (default: all three places it is kept)")
    parser.add_argument("--check-only", action="store_true", help="only check the tag against the app version")
    args = parser.parse_args(argv)
    if not args.check_only and None in (args.out, args.msi, args.dmg):
        parser.error("OUT, --msi and --dmg are required unless --check-only")
    try:
        versions = {"--version": args.version} if args.version else app_versions()
        if args.check_only:
            check_tag(args.tag, versions)
        else:
            prepare(args.out, args.msi, args.dmg, args.tag, versions)
    except ReleaseError as exc:
        print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
