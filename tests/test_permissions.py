"""The macOS microphone permission check, with the Objective-C call faked."""

import sys

import pytest


def _module():
    try:
        from clean_voice import permissions
    except ImportError:
        pytest.fail("clean_voice.permissions does not exist yet")
    return permissions


def test_mic_permission_maps_the_status_codes(monkeypatch) -> None:
    # B003 item 23: restricted (1) and denied (2) block; not decided (0) and allowed (3) do not; a failed call is
    # unknown. OBS: frontend/utility/platform-osx.mm CheckPermissionWithPrompt :208-231.
    permissions = _module()
    monkeypatch.setattr(permissions.sys, "platform", "darwin")
    for code, expected in ((0, "ok"), (3, "ok"), (1, "denied"), (2, "denied")):
        monkeypatch.setattr(permissions, "_authorization_status", lambda code=code: code)
        assert permissions.mic_permission() == expected, code

    def boom() -> int:
        raise OSError("no AVFoundation")

    monkeypatch.setattr(permissions, "_authorization_status", boom)
    assert permissions.mic_permission() is None
    monkeypatch.undo()
    if sys.platform != "darwin":
        assert permissions.mic_permission() is None
