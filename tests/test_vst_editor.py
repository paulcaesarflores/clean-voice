"""The VST editor window (J10 plan §5 "Editor window", Test T24).

The fake plug-in's half runs everywhere. The compiled test plug-in's half needs a real window system: it skips under
Qt's offscreen platform (the normal test run) and runs on Windows with ``QT_QPA_PLATFORM=windows`` (CI's "VST editor
(Windows desktop)" step)."""

from __future__ import annotations

import ctypes
import sys

import pytest
from vst_fake import TEST_PLUGIN, FakePlugin, load_fake, require_test_plugin

from clean_voice import vst, vst_editor


def editor_plugin(**kwargs) -> tuple[FakePlugin, vst.VstPlugin]:
    fake = FakePlugin(flags=vst.FLAG_CAN_REPLACING | vst.FLAG_HAS_EDITOR, **kwargs)
    return fake, load_fake(fake)


def test_t24_open_hands_the_plugin_a_native_parent_and_sizes_from_its_rect(qtbot) -> None:
    fake, plugin = editor_plugin(rect=(0, 0, 300, 400))
    window = vst_editor.open_editor(plugin, "Warm EQ")
    assert window is not None
    qtbot.addWidget(window)
    assert plugin.editor is window and window.isVisible()
    assert window.windowTitle() == "Clean Voice: Warm EQ - Fake Gain"
    assert window.windowFlags() & vst_editor.Qt.WindowType.MSWindowsFixedSizeDialogHint
    assert fake.editor_parent == int(window.view.winId())  # effEditOpen got our native view
    assert window.view.testAttribute(vst_editor.Qt.WidgetAttribute.WA_NativeWindow)
    opens = fake.opcodes.index(vst.EFF_EDIT_OPEN)
    if sys.platform == "darwin":
        # mac/EditorWidget-osx.mm:37-49: the rect first, then open; the rect is in points, used as it is.
        assert vst.EFF_EDIT_GET_RECT in fake.opcodes[:opens]
        assert (window.width(), window.height()) == (400, 300)
    else:
        scale = window.devicePixelRatioF()  # win :51-58: the rect is in device pixels
        assert (window.width(), window.height()) == (int(400 / scale), int(300 / scale))
        assert vst.EFF_EDIT_GET_RECT in fake.opcodes[opens:]  # Windows: open, then ask the rect (win :46-49)
    assert vst_editor.open_editor(plugin, "Warm EQ") is window  # one editor at a time
    plugin.close()


def test_t24_a_size_request_resizes_from_the_plugins_rect(qtbot) -> None:
    fake, plugin = editor_plugin(rect=(0, 0, 300, 400))
    window = vst_editor.open_editor(plugin)
    assert window is not None
    qtbot.addWidget(window)
    assert window.windowTitle() == "Clean Voice - Fake Gain"
    fake.rect = vst.Rect(0, 0, 360, 520)
    assert fake.host is not None
    assert fake.host(ctypes.pointer(fake.effect), vst.HOST_SIZE_WINDOW, 520, 360, None, 0.0) == 1
    scale = window.devicePixelRatioF()
    qtbot.waitUntil(lambda: window.width() == int(520 / scale))
    assert window.height() == int(360 / scale)
    plugin.close()


def test_t24_closing_sends_edit_close_and_swaps_the_buttons(qtbot) -> None:
    fake, plugin = editor_plugin()
    assert vst_editor.interface_buttons(None) == (False, False)
    assert vst_editor.interface_buttons(plugin) == (True, False)
    window = vst_editor.open_editor(plugin)
    assert window is not None
    assert vst_editor.interface_buttons(plugin) == (False, True)
    vst_editor.close_editor(plugin)
    assert fake.opcodes.count(vst.EFF_EDIT_CLOSE) == 1
    assert plugin.editor is None
    assert vst_editor.interface_buttons(plugin) == (True, False)
    plugin.close()
    assert vst_editor.interface_buttons(plugin) == (False, False)


def test_t24_unloading_closes_the_editor_first(qtbot) -> None:
    fake, plugin = editor_plugin()
    window = vst_editor.open_editor(plugin)
    assert window is not None
    plugin.close()  # unloadEffect: closeEditor, then effMainsChanged(0) and effClose (VSTPlugin.cpp:268-289)
    tail = fake.opcodes[-3:]
    assert tail == [vst.EFF_EDIT_CLOSE, vst.EFF_MAINS_CHANGED, vst.EFF_CLOSE]


def test_t24_no_editor_no_window_and_no_rect_gives_300(qtbot) -> None:
    fake = FakePlugin()  # no effFlagsHasEditor (VSTPlugin.cpp:315-318)
    plugin = load_fake(fake)
    assert vst_editor.open_editor(plugin) is None and vst.EFF_EDIT_OPEN not in fake.opcodes
    plugin.close()
    fake, plugin = editor_plugin(rect=None)
    window = vst_editor.open_editor(plugin)
    assert window is not None
    qtbot.addWidget(window)
    assert (window.width(), window.height()) == (300, 300)
    plugin.close()


def test_t24_macos_takes_the_rect_first_as_a_fixed_size(qtbot) -> None:
    fake, plugin = editor_plugin(rect=(0, 0, 200, 250))
    window = vst_editor.EditorWindow(plugin, "EQ", platform="darwin")
    qtbot.addWidget(window)
    ops = fake.opcodes
    assert ops.index(vst.EFF_EDIT_GET_RECT) < ops.index(vst.EFF_EDIT_OPEN)  # mac :37-49
    assert (window.minimumWidth(), window.maximumWidth(), window.minimumHeight()) == (250, 250, 200)
    window._resize(640, 480)  # handleResizeRequest on macOS: the asked size, fixed
    assert (window.maximumWidth(), window.maximumHeight()) == (640, 480)
    window.close()
    plugin.close()


def test_open_when_active_opens_the_editor_on_load(qtbot) -> None:
    _, plugin = editor_plugin()
    assert vst_editor.open_if_wanted(plugin) is None
    plugin.open_when_active = True
    window = vst_editor.open_if_wanted(plugin, "EQ")
    assert window is not None and plugin.editor is window
    plugin.close()


# ------------------------------------------------------------------ the real window system, with the test plug-in


@pytest.fixture
def desktop(qapp) -> None:
    if qapp.platformName() == "offscreen":
        pytest.skip("needs a real window system (CI runs this file with QT_QPA_PLATFORM=windows)")
    if sys.platform != "win32":
        pytest.skip("the test plug-in has an editor on Windows only")
    require_test_plugin()


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 windows")
def test_t24_the_real_plugin_draws_in_a_native_child_of_our_window(desktop, qtbot) -> None:
    user32 = ctypes.WinDLL("user32")  # type: ignore[attr-defined]
    user32.GetWindow.restype = ctypes.c_void_p
    user32.GetWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    user32.GetParent.restype = ctypes.c_void_p
    user32.GetParent.argtypes = [ctypes.c_void_p]
    user32.IsWindow.argtypes = [ctypes.c_void_p]
    plugin = vst.load(str(TEST_PLUGIN))
    try:
        window = vst_editor.open_editor(plugin, "Test")
        assert window is not None
        qtbot.waitExposed(window)
        parent = int(window.view.winId())
        child = user32.GetWindow(parent, 5)  # GW_CHILD: the window the plug-in made in effEditOpen
        assert child and user32.GetParent(child) == parent
        scale = window.devicePixelRatioF()
        assert (window.width(), window.height()) == (int(400 / scale), int(300 / scale))
        # The plug-in grows its editor and calls audioMasterSizeWindow from setParameter.
        plugin._effect.contents.set_parameter(plugin._effect, 2, 1.0)
        qtbot.waitUntil(lambda: window.width() == int(520 / scale))
        assert window.height() == int(360 / scale)
        assert vst_editor.interface_buttons(plugin) == (False, True)
        vst_editor.close_editor(plugin)
        assert not user32.IsWindow(child)  # effEditClose: the plug-in destroyed its window
        assert vst_editor.interface_buttons(plugin) == (True, False)
    finally:
        plugin.close()


def test_the_filter_windows_interface_buttons_open_the_real_editor(desktop, qtbot) -> None:
    """The filter window's VST pane with the compiled plug-in: Open Plug-in Interface opens its editor, Close closes
    it, and closing the editor from its own title bar brings Open back (J10 plan T33 with T24's real window)."""
    from PySide6.QtWidgets import QComboBox

    from clean_voice import filter_window, fine_tune
    from clean_voice.settings import Settings

    held = {"settings": Settings()}
    registry = vst.VstRegistry()
    model = filter_window.FilterModel(
        lambda: held["settings"],
        lambda s: held.update(settings=s),
        registry,
        scanner=lambda: [("CleanVoiceTestGain", str(TEST_PLUGIN))],
    )
    window = filter_window.FilterWindow(model)
    qtbot.addWidget(window)
    window.ask_name = lambda default: default  # type: ignore[method-assign]
    window.show()
    window.add_filter(vst.KIND)
    try:
        pane = window.pane
        assert pane is not None
        box = pane.controls["plugin_path"]
        assert isinstance(box, QComboBox)
        box.setCurrentIndex(box.findData(str(TEST_PLUGIN)))
        spec = held["settings"].filters[0]
        plugin = model.plugin(spec.uid)
        assert plugin is not None and plugin.effect_name == "Test Gain"
        pane = window.pane
        assert pane is not None and pane.status.text() == "Test Gain by Clean Voice tests"
        pane.controls["open_vst_settings"].click()  # type: ignore[attr-defined]
        editor = plugin.editor
        assert isinstance(editor, vst_editor.EditorWindow)
        qtbot.waitExposed(editor)
        assert editor.windowTitle() == f"Clean Voice: {fine_tune.KINDS[vst.KIND].name} - Test Gain"
        assert not pane.form.isRowVisible(pane.controls["open_vst_settings"])
        assert pane.form.isRowVisible(pane.controls["close_vst_settings"])
        pane.controls["close_vst_settings"].click()  # type: ignore[attr-defined]
        assert plugin.editor is None and pane.form.isRowVisible(pane.controls["open_vst_settings"])
        pane.controls["open_vst_settings"].click()  # type: ignore[attr-defined]
        assert isinstance(plugin.editor, vst_editor.EditorWindow)
        plugin.editor.close()  # the editor's own title bar
        qtbot.waitUntil(lambda: pane.form.isRowVisible(pane.controls["open_vst_settings"]))
        assert not pane.form.isRowVisible(pane.controls["close_vst_settings"])
    finally:
        window.close()
        registry.close_all()
