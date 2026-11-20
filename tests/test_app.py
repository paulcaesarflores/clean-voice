"""The main screen's wording for each engine state."""

from clean_voice.app import HEAD_NO_MIC, describe
from clean_voice.engine import State, Status


def test_describe_mic_denied() -> None:
    # B003 item 23: macOS blocked the microphone; the notice says where to switch it on.
    mic_denied = getattr(State, "MIC_DENIED", None)
    assert mic_denied is not None, "State.MIC_DENIED does not exist yet"
    view = describe(Status(state=mic_denied))
    assert view.headline == HEAD_NO_MIC and view.tone == "hot" and view.power == "hot"
    assert view.notice is not None and view.notice.kind == "mic"
    assert view.notice.title == "Clean Voice isn't allowed to use your microphone"
    assert view.notice.body == (
        "Open System Settings, Privacy & Security, Microphone, and switch Clean Voice on. "
        "Clean Voice notices within a few seconds."
    )
