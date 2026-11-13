"""Nothing publishable names the builder's own user folder (release r1): not in code, tests, tools or workflows, and
not in the bundled native libraries, whose build paths once did. The user name is read from this machine's home
folder at run time, so this test does not publish it either."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("src", "tests", "tools", ".github")
# Home folder names that are not a person's (CI runners, shared or default accounts): nothing to look for.
GENERIC = {"runner", "runneradmin", "user", "users", "admin", "administrator", "root", "guest", "default", "public"}


def _git_files(*options: str) -> set[str]:
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", *options, "--exclude-standard", "--", *FOLDERS],
        capture_output=True,
        check=True,
    ).stdout
    return {name for name in listed.decode("utf-8").split("\0") if name}


def publishable_files() -> list[Path]:
    """Files in FOLDERS that .gitignore does not keep local, tracked or not: what a commit would publish."""
    names = _git_files("--cached", "--others") - _git_files("--cached", "--ignored")
    return [ROOT / name for name in sorted(names)]


def test_no_file_names_the_user_folder() -> None:
    name = Path.home().name
    if len(name) < 3 or name.lower() in GENERIC:
        pytest.skip(f"the home folder name {name!r} is generic")
    pattern = re.compile(rb"Users[\\/]+" + re.escape(name.encode("utf-8")), re.IGNORECASE)
    files = [path for path in publishable_files() if path.is_file()]
    assert any(path.suffix == ".dylib" for path in files), "the native libraries were not scanned"
    leaks = [str(path.relative_to(ROOT)) for path in files if pattern.search(path.read_bytes())]
    assert not leaks, f"these files name this machine's user folder: {leaks}"
