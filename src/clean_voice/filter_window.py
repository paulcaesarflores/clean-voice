"""Fine-tune's filter window: OBS Studio's Filters dialog, audio part only (J10 plan §6).

It follows ``reference/obs-studio/frontend/dialogs/OBSBasicFilters.cpp`` (and ``frontend/forms/OBSBasicFilters.ui``):
the filter list with an eye per filter, the + menu of the ten kinds, a name asked for each new filter, rename by
double-click, remove with a question, up and down (Ctrl+Up / Ctrl+Down, Del removes), the selected filter's properties
on the right, built as OBS's properties view builds them (``shared/properties-view/properties-view.cpp``), Defaults
with a question, and Close. The VST 2.x filter's pane follows ``plugins/obs-vst/obs-vst.cpp`` ``vst_properties``.
Left out, as the plan's question 7 decided: undo/redo, copy/paste/duplicate, the right-click menu and the sidechain.

``FilterModel`` holds the edits and the VST plug-ins' life (no Qt widgets): every change is published as new
``Settings`` at once, as OBS updates a filter on every edit. A plug-in is loaded on the UI thread and added to the
engine's ``vst.VstRegistry`` before the ``Settings`` naming it are published; a plug-in leaves the registry only after
``Settings`` without it are published, and the registry unloads it once no chain can call it (``tick``). A filter
given another plug-in gets a new uid (``fine_tune.new_uid``), so one uid is always one loaded plug-in.

The code is our own (OBS is GPL, Clean Voice is MIT); each OBS behaviour cites its file and lines.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePath

from PySide6.QtCore import QModelIndex, QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QFont,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemDelegate,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from clean_voice import fine_tune, fonts, nvafx, theme, vst, vst_editor
from clean_voice.eq_panel import EqPanel
from clean_voice.fine_tune import BUTTON, CHECKBOX, HIDDEN, INT, KINDS, LIST, SLIDER, FilterSpec, Prop
from clean_voice.settings import Settings, replace
from clean_voice.theme import Palette
from clean_voice.widgets import IconButton, keyboard_focus, paint_icon, ui_font

log = logging.getLogger(__name__)

TITLE = "Fine-tune filters"  # OBS: "Filters for '%1'" (frontend/data/locale/en-US.ini Basic.Filters.Title)
LIST_TITLE = "Filters"  # OBS: "Audio Filters" (OBSBasicFilters.cpp:114-116)
SIZE = (720, 480)
MIN_SIZE = (600, 380)
EQ_SIZE = (1000, 700)  # what the 10-band EQ's panel is shown in (B012); never more than the screen's available area
LIST_W = 240
EMPTY_TEXT = "Add a filter with +. The filters run from the top of the list down."
NO_PROPERTIES = "This filter has no settings."  # OBS: "No properties available" (properties-view.cpp:160-163)
VST_TRADEMARK = "VST is a trademark of Steinberg Media Technologies GmbH."  # as in README
ADD_TITLE = "Filter name"  # Basic.Filters.AddFilter.Title
ADD_TEXT = "Name of the new filter:"  # OBS: "Please specify the name of the filter"
# OBS's two name messages (NoNameEntered, NameExists), in our words.
NAME_EMPTY = ("Type a name", "A filter needs a name.")
NAME_TAKEN = ("Choose another name", "Another filter already has this name.")
REMOVE_TITLE = "Remove filter"  # OBS: "Confirm Remove"
RESET_TITLE = "Reset filter"  # OBS: "Reset Properties"
RESET_TEXT = "Reset this filter's settings to their defaults?"
CHOOSE_PLUGIN = "Choose a plug-in"  # OBS: "{Please select a plug-in}" (obs-vst.cpp:262)
NO_PLUGINS = "No VST 2 plug-ins were found in the usual folders."
# OBS's own words (nv-filters data/locale/en-US.ini Nvafx.OutdatedSDK), shown as nvidia_audio_properties :846-851 does.
NVIDIA_OUTDATED = (
    "WARNING: Please upgrade both NVIDIA Video & Audio SDK. Your current version of Audio SDK is outdated. "
    "This can break Audio FX, Video FX, or both."
)
# OBS shows nothing for this (its filter is not registered without the runtime, so a saved one is a missing filter
# with no properties); the line tells a guest why the filter does nothing here.
NVIDIA_MISSING = (
    "NVIDIA's audio effects runtime was not found on this computer, so this filter passes your voice through "
    "unchanged. Your settings are kept."
)


PRESET_LABEL = "Preset"  # B013: not in OBS
PRESET_PLACEHOLDER = "Choose a preset"
PRESET_TITLE = "Save preset"
PRESET_TEXT = "Name of the preset:"
PRESET_EMPTY = ("Type a name", "A preset needs a name.")
PRESET_REPLACE_TITLE = "Replace preset"
PRESET_DELETE_TITLE = "Delete preset"


def replace_preset_text(name: str) -> str:
    return f'A preset named "{name}" already exists. Replace it?'


def delete_preset_text(name: str) -> str:
    return f'Delete the preset "{name}"?'


def remove_text(name: str) -> str:
    return f'Remove "{name}"?'  # OBS: "Are you sure you wish to remove '%1'?" (QueryRemove :707-722)


def load_failed_text(reason: str) -> str:
    return f"Clean Voice couldn't load this plug-in: {reason}. Try another plug-in or another version of it."


def plugin_text(plugin: vst.VstPlugin) -> str:
    return f"{plugin.effect_name} by {plugin.vendor}" if plugin.vendor else plugin.effect_name


def no_plugins_text(folders: Sequence[PurePath]) -> str:
    return NO_PLUGINS + "\n" + "\n".join(str(f) for f in folders)


STARTUP_ERROR = "it stopped with an error while starting"  # anything but a VstError from a plug-in's start


# ------------------------------------------------------------------ the model: edits, publishing, the plug-ins' life

Loader = Callable[[Mapping[str, object]], "vst.VstPlugin | None"]


class FilterModel:
    """The guest's filter list and its VST plug-ins. ``settings`` reads the current ``Settings``; ``publish`` sends
    new ones (the window: to the engine, and into the saved choices). ``loader`` is ``vst.create`` (obs-vst.cpp
    vst_create / vst_update :98-138); ``scanner`` lists the plug-ins found (``vst.scan``)."""

    def __init__(
        self,
        settings: Callable[[], Settings],
        publish: Callable[[Settings], None],
        registry: vst.VstRegistry,
        loader: Loader = vst.create,
        scanner: Callable[[], list[tuple[str, str]]] = vst.scan,
        folders: Callable[[], Sequence[PurePath]] = vst.search_folders,
        presets: Callable[[], Mapping[str, tuple[FilterSpec, ...]]] | None = None,
        store_presets: Callable[[Mapping[str, tuple[FilterSpec, ...]]], None] | None = None,
    ) -> None:
        """``presets`` reads the saved presets and ``store_presets`` keeps a new set (B013: the app gives the saved
        choices' ``Prefs.presets``); without them the model keeps its own."""
        self._own: dict[str, tuple[FilterSpec, ...]] = {}
        self.presets = presets or (lambda: self._own)
        self._store_presets = store_presets or self._keep_own
        self._settings = settings
        self._publish = publish
        self.registry = registry
        self._loader = loader
        self.scan = scanner
        self.folders = folders
        self.errors: dict[int, str] = {}  # a VST filter whose plug-in would not load: why, in plain words

    @property
    def filters(self) -> tuple[FilterSpec, ...]:
        return self._settings().filters

    def find(self, uid: int) -> tuple[int, FilterSpec] | None:
        return next(((i, f) for i, f in enumerate(self.filters) if f.uid == uid), None)

    def plugin(self, uid: int) -> vst.VstPlugin | None:
        return self.registry.plugin(uid)

    def _put(self, filters: Sequence[FilterSpec], **more: object) -> None:
        self._publish(replace(self._settings(), filters=tuple(filters), **more))

    def _replace(self, index: int, spec: FilterSpec) -> None:
        filters = list(self.filters)
        filters[index] = spec
        self._put(filters)

    # -------------------------------------------------------------- VST plug-ins

    def _load(self, spec: FilterSpec) -> vst.VstPlugin | None:
        """vst_update's load (obs-vst.cpp:104-130), on the UI thread, into the registry before any Settings name it.
        A plug-in that won't load leaves its reason in ``errors``; the filter stays in the list."""
        self.errors.pop(spec.uid, None)
        path = spec.settings.get("plugin_path")
        if spec.kind != vst.KIND or not isinstance(path, str) or not path:
            return None
        try:
            plugin = self._loader(spec.settings)
        except vst.VstError as exc:
            self.errors[spec.uid] = exc.reason
            return None
        except Exception:
            log.exception("loading the VST plug-in %s failed", path)
            self.errors[spec.uid] = STARTUP_ERROR
            return None
        if plugin is not None:
            self.registry.add(spec.uid, plugin)
        return plugin

    def _drop(self, uid: int) -> None:
        """After Settings without ``uid`` are published: close its editor (vst_destroy, obs-vst.cpp:91-96) and let the
        registry unload it once no chain can call it."""
        self.errors.pop(uid, None)
        plugin = self.registry.plugin(uid)
        if plugin is None:
            return
        vst_editor.close_editor(plugin)
        self.registry.remove(uid)

    def load_saved(self) -> None:
        """At launch, before the saved Settings reach the engine: load every saved VST filter's plug-in (vst_create)
        and open its editor if "Open interface when active" is on (VSTPlugin.cpp:211-213)."""
        for spec in self.filters:
            plugin = self._load(spec)
            if plugin is not None:
                vst_editor.open_if_wanted(plugin, spec.name)

    def _swap_plugin(self, index: int, spec: FilterSpec, settings: Mapping[str, object]) -> FilterSpec:
        """The filter's plug-in changes (another one chosen, or Defaults clearing it): a new uid with the new plug-in
        published in place of the old one, which is then dropped. OBS unloads the old effect before it loads the new
        (VSTPlugin.cpp loadEffectFromPath :136-145); here the old one runs until the chain has let go of it."""
        new = FilterSpec(spec.kind, spec.name, spec.enabled, settings, uid=fine_tune.new_uid())
        plugin = self._load(new)
        self._replace(index, new)
        self._drop(spec.uid)
        if plugin is not None:
            vst_editor.open_if_wanted(plugin, new.name)
        return new

    def choose_plugin(self, uid: int, path: str) -> FilterSpec | None:
        """The plug-in list's choice (``plugin_path``); "" chooses none (vst_update unloads, obs-vst.cpp:116-119)."""
        found = self.find(uid)
        if found is None:
            return None
        index, spec = found
        if spec.settings.get("plugin_path") == path:
            return spec
        return self._swap_plugin(index, spec, {**spec.settings, "plugin_path": path})

    def open_editor(self, uid: int) -> None:
        """Open Plug-in Interface (obs-vst.cpp open_editor_button_clicked :38-53)."""
        plugin = self.plugin(uid)
        found = self.find(uid)
        if plugin is not None and plugin.loaded and found is not None:
            vst_editor.open_editor(plugin, found[1].name)

    def close_editor(self, uid: int) -> None:
        """Close Plug-in Interface (close_editor_button_clicked :55-70)."""
        plugin = self.plugin(uid)
        if plugin is not None and plugin.loaded:
            vst_editor.close_editor(plugin)

    def with_state(self, settings: Settings) -> Settings:
        """``settings`` with every loaded plug-in's state read into its filter (vst_save, obs-vst.cpp:140-145): the
        saved choices take it on each save, when the filter window closes and at quit. The same Settings when no
        state changed. A plug-in that failed to load keeps the state it was saved with."""
        changed = False
        filters = []
        for spec in settings.filters:
            plugin = self.registry.plugin(spec.uid) if spec.kind == vst.KIND else None
            if plugin is not None and plugin.loaded:
                state = vst.saved_state(plugin)
                if any(spec.settings.get(k) != v for k, v in state.items()):
                    spec = dataclasses.replace(spec, settings={**spec.settings, **state})
                    changed = True
            filters.append(spec)
        return replace(settings, filters=tuple(filters)) if changed else settings

    def tick(self) -> None:
        """The UI's tick: unload the removed plug-ins no chain holds any more."""
        self.registry.collect()

    # -------------------------------------------------------------- the list (OBSBasicFilters.cpp)

    def add(self, kind: str, name: str) -> FilterSpec:
        """AddNewFilter :518-560: a new filter with OBS's defaults, at the bottom (AddFilter :282-300 appends its row).
        Fine-tune turns on, so the guest hears what was added (the plan's question 6)."""
        spec = FilterSpec(kind, name)
        self._put([*self.filters, spec], fine_tune=True)
        return spec

    def remove(self, uid: int) -> None:
        """delete_filter :1104-1139."""
        filters = [f for f in self.filters if f.uid != uid]
        if len(filters) == len(self.filters):
            return
        self._put(filters)
        self._drop(uid)

    def rename(self, uid: int, name: str) -> tuple[str, str] | None:
        """FilterNameEdited :977-1032: the name trimmed; empty or taken is refused with its message (title, text);
        the same name changes nothing."""
        found = self.find(uid)
        if found is None:
            return None
        index, spec = found
        name = name.strip()
        if name == spec.name:
            return None
        if not name:
            return NAME_EMPTY
        if any(f.name == name for f in self.filters):
            return NAME_TAKEN
        self._replace(index, dataclasses.replace(spec, name=name))
        return None

    def set_enabled(self, uid: int, on: bool) -> None:
        """The eye: obs_source_set_enabled (components/VisibilityItemWidget.cpp)."""
        found = self.find(uid)
        if found is not None and found[1].enabled != on:
            self._replace(found[0], dataclasses.replace(found[1], enabled=on))

    def move(self, uid: int, delta: int) -> bool:
        """OBS_ORDER_MOVE_UP / MOVE_DOWN (on_moveAsyncFilterUp/Down_clicked :743-757): one place, never past an end."""
        found = self.find(uid)
        if found is None:
            return False
        index = found[0]
        other = index + delta
        if not 0 <= other < len(self.filters):
            return False
        filters = list(self.filters)
        filters[index], filters[other] = filters[other], filters[index]
        self._put(filters)
        return True

    def set_setting(self, uid: int, key: str, value: object) -> bool:
        """One property edited (WidgetInfo::ControlChanged). True when the pane must be built again: the Expander's
        preset switch, which clears the settings to that preset's defaults (expander-filter.c expander_update
        :182-196), or another plug-in."""
        found = self.find(uid)
        if found is None:
            return False
        index, spec = found
        if spec.settings.get(key) == value:
            return False
        if spec.kind == vst.KIND and key == "plugin_path":
            self.choose_plugin(uid, str(value))
            return True
        presets = KINDS[spec.kind].preset_defaults
        if key in presets:
            self._replace(index, dataclasses.replace(spec, settings={key: value}))  # obs_data_clear, then the preset
            return True
        self._replace(index, dataclasses.replace(spec, settings={**spec.settings, key: value}))
        if spec.kind == vst.KIND and key == "open_when_active_vst_settings":
            plugin = self.plugin(uid)
            if plugin is not None:
                plugin.open_when_active = value is True  # vst_update :102
        return False

    # -------------------------------------------------------------- presets (B013: not in OBS, the user's request)

    def _keep_own(self, presets: Mapping[str, tuple[FilterSpec, ...]]) -> None:
        self._own = dict(presets)

    def preset_names(self) -> list[str]:
        return sorted(self.presets(), key=str.casefold)

    def save_preset(self, name: str) -> None:
        """The whole chain under ``name``: every filter, its order, enabled state and settings, a plug-in's state as
        the chain saves it (``with_state``). The same name is replaced."""
        chain = self.with_state(self._settings()).filters
        self._store_presets({**self.presets(), name: tuple(chain)})

    def delete_preset(self, name: str) -> None:
        self._store_presets({k: v for k, v in self.presets().items() if k != name})

    def load_preset(self, name: str) -> None:
        """The saved chain replaces the current one, published like any chain change. Each filter gets a new uid, so
        the chain builds it fresh and a loaded plug-in is its own instance; the old filters' plug-ins are dropped after
        the new list is published (as ``_swap_plugin``). A plug-in that won't load leaves its reason in ``errors``."""
        saved = self.presets().get(name)
        if saved is None:
            return
        fresh = [FilterSpec(f.kind, f.name, f.enabled, f.settings) for f in saved]
        old = self.filters
        loaded = [(spec, self._load(spec)) for spec in fresh]
        self._put(fresh, fine_tune=True)  # Fine-tune turns on, as in ``add``, so the guest hears the preset
        for spec in old:
            self._drop(spec.uid)
        for spec, plugin in loaded:
            if plugin is not None:
                vst_editor.open_if_wanted(plugin, spec.name)

    def reset(self, uid: int) -> None:
        """Defaults (ResetFilters :1044-1071): the filter's settings cleared, so every one is OBS's default. A VST
        filter's plug-in path is cleared too, which unloads the plug-in (vst_update :116-119)."""
        found = self.find(uid)
        if found is None:
            return
        index, spec = found
        if spec.kind == vst.KIND and spec.settings.get("plugin_path"):
            self._swap_plugin(index, spec, {})
            return
        self._replace(index, dataclasses.replace(spec, settings={}))


# ------------------------------------------------------------------ OBS's property controls


class _WheelOnlyWhenFocused:
    """spinbox-ignorewheel.cpp and SliderIgnoreScroll: the wheel scrolls the pane, not a control it passes over."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.hasFocus():  # type: ignore[attr-defined]
            event.ignore()
            return
        super().wheelEvent(event)  # type: ignore[misc]


class _Arrows:
    """The up and down arrows of OBS's number boxes, drawn as Clean Voice's chevrons over the style's buttons."""

    def paintEvent(self, event: object) -> None:
        super().paintEvent(event)  # type: ignore[misc]
        widget: QWidget = self  # type: ignore[assignment]
        pal = theme.current()
        p = QPainter(widget)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        tone = pal.muted if widget.isEnabled() else pal.line
        x, mid = widget.width() - 19.0, widget.height() / 2.0
        paint_icon(p, "chevron-up", QRectF(x, mid - 13.0, 14, 14), tone, 2.0)
        paint_icon(p, "chevron", QRectF(x, mid - 1.0, 14, 14), tone, 2.0)


class SpinBox(_Arrows, _WheelOnlyWhenFocused, QSpinBox):
    pass


class DoubleSpinBox(_Arrows, _WheelOnlyWhenFocused, QDoubleSpinBox):
    pass


class Combo(_WheelOnlyWhenFocused, QComboBox):
    """A property list (AddList), with Clean Voice's chevron. It never asks for the width of its longest item (NVIDIA's
    "Noise Removal + Room Echo Removal" would push the pane wider than the window): it fills the pane, and the popup
    is as wide as its items."""

    def __init__(self) -> None:
        super().__init__()
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(12)

    def paintEvent(self, event: object) -> None:
        super().paintEvent(event)  # type: ignore[arg-type]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_icon(p, "chevron", QRectF(self.width() - 26.0, self.height() / 2.0 - 8.0, 16, 16), theme.current().muted)


class CheckBox(QCheckBox):
    """A property checkbox (AddCheckbox), its tick drawn in Clean Voice's line style."""

    def paintEvent(self, event: object) -> None:
        super().paintEvent(event)  # type: ignore[arg-type]
        if not self.isChecked():
            return
        option = QStyleOptionButton()
        self.initStyleOption(option)
        box = self.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, self)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_icon(p, "tick", QRectF(box).adjusted(1, 1, -1, -1), theme.current().on_desk, 2.5)


class Slider(_WheelOnlyWhenFocused, QSlider):
    """OBS's property slider (SliderIgnoreScroll), drawn in Clean Voice's colours: a 4 px ``line`` track filled with
    ``desk`` up to a round handle. The handle sits where the style maps the value, so a click lands where it shows."""

    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # SliderIgnoreScroll's constructor
        self.setMinimumHeight(24)

    def handle_x(self) -> float:
        length = self.style().pixelMetric(QStyle.PixelMetric.PM_SliderLength, None, self)
        span = max(self.width() - length, 0)
        pos = QStyle.sliderPositionFromValue(self.minimum(), self.maximum(), self.value(), span)
        return pos + length / 2.0

    def paintEvent(self, event: object) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        mid = self.height() / 2.0
        x = self.handle_x()
        on = self.isEnabled()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(pal.line))
        p.drawRoundedRect(QRectF(2, mid - 2, self.width() - 4, 4), 2, 2)
        p.setBrush(QColor(pal.desk if on else pal.muted))
        p.drawRoundedRect(QRectF(2, mid - 2, max(x - 2, 0), 4), 2, 2)
        focused = keyboard_focus(self)
        if focused:
            ring = QColor(pal.desk)
            ring.setAlphaF(0.3)
            p.setBrush(ring)
            p.drawEllipse(QRectF(x - 11, mid - 11, 22, 22))
        p.setBrush(QColor(pal.surface))
        p.setPen(QPen(QColor(pal.desk if on else pal.muted), 2))
        p.drawEllipse(QRectF(x - 7, mid - 7, 14, 14))


class DoubleSlider(Slider):
    """properties-view/double-slider.cpp: an int slider over (max - min) / step positions."""

    double_changed = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.lo, self.step_size = 0.0, 1.0
        self.valueChanged.connect(lambda v: self.double_changed.emit((self.lo / self.step_size + v) * self.step_size))

    def set_constraints(self, lo: float, hi: float, step: float, value: float) -> None:
        self.lo, self.step_size = lo, step
        self.setMinimum(0)
        self.setMaximum(int((hi - lo) / step))
        self.setSingleStep(1)
        self.set_double(value)

    def set_double(self, value: float) -> None:
        self.setValue(int((value - self.lo) / self.step_size + 0.5))  # lround, for the non-negative positions


def float_decimals(step: float) -> int:
    """AddFloat :495-501: two decimals (QDoubleSpinBox's), more only for a step below 0.01."""
    if step >= 1.0:
        return 2
    return max(2, min(int(math.log10(1.0 / step) + 0.99), 8))


class Pane(QWidget):
    """The properties of one filter: OBS's properties view (RefreshProperties :115-166), a form with the labels on the
    right of their column. ``controls`` maps each property key to its main control; ``labels`` to its label."""

    changed = Signal(str, object)  # key, value
    clicked = Signal(str)  # a button's key

    def __init__(
        self,
        spec: FilterSpec,
        plugins: Sequence[tuple[str, str]] = (),
        runtime: nvafx.Runtime | None = None,
        pal: Palette | None = None,
    ) -> None:
        """``runtime``: NVIDIA's runtime, which only the NVIDIA filter's pane looks at."""
        super().__init__()
        self.setObjectName("pane")
        self.spec = spec
        self.runtime = runtime
        self.controls: dict[str, QWidget] = {}
        self.sliders: dict[str, QSlider] = {}
        self.labels: dict[str, QLabel] = {}
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)  # :129
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)  # :134
        form.setContentsMargins(4, 0, 8, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        self.form = form
        kind = KINDS[spec.kind]
        nvidia = spec.kind == fine_tune.NVIDIA_KIND
        shown = [p for p in kind.props if p.widget != HIDDEN]
        if nvidia and runtime is None:
            shown = []  # a missing module has no properties (nvidia_audio_properties :834 adds the list only if found)
        own_eq = spec.kind == fine_tune.EQ_KIND
        if own_eq:  # the 10-band EQ draws its own panel (B012) instead of one form row per property
            shown = []
            eq = EqPanel(spec.settings, pal)
            eq.changed.connect(self._emit)
            self.controls["eq"] = eq
            form.addRow(eq)
        for prop in shown:
            self._add(prop, plugins)
        if own_eq:
            pass
        elif nvidia and runtime is None:
            self._text(NVIDIA_MISSING)
        elif kind.note:
            self._text(kind.note)
        elif not shown:
            self._text(NO_PROPERTIES)  # :160-163
        self.status = QLabel("", self)  # the VST pane's line: the loaded effect, a load failure, no plug-ins found
        self.status.setProperty("role", "paneStatus")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.hide()
        if nvidia and runtime is not None:
            self._nvidia_vad(str(spec.settings.get("method", "denoiser")))
            if runtime.outdated:  # :846-851: OBS's warning under the controls
                form.addRow(self.status)
                self.set_status(NVIDIA_OUTDATED, warn=True)
        self.trademark: QLabel | None = None
        if spec.kind == vst.KIND:
            form.addRow(self.status)
            # Plan question 10: the app has no About text, so Steinberg's line ends the VST pane.
            self.trademark = QLabel(VST_TRADEMARK)
            self.trademark.setProperty("role", "paneTrademark")
            self.trademark.setWordWrap(True)
            self.trademark.setIndent(0)  # a styled padding would indent it by half an "x"
            form.addRow(self.trademark)

    def _text(self, text: str) -> None:
        label = QLabel(text)
        label.setProperty("role", "paneNote")
        label.setWordWrap(True)
        self.form.addRow(label)

    def set_status(self, text: str, warn: bool = False) -> None:
        self.status.setText(text)
        self.status.setProperty("warn", warn)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.status.setVisible(bool(text))

    def _label(self, prop: Prop) -> QLabel:
        label = QLabel(prop.label)
        label.setProperty("role", "paneLabel")
        self.labels[prop.key] = label
        return label

    def _add(self, prop: Prop, plugins: Sequence[tuple[str, str]]) -> None:
        """AddProperty :1523-1638."""
        value = self.spec.settings.get(prop.key, prop.default)
        key = prop.key
        if prop.widget in (SLIDER, INT):
            row = QHBoxLayout()
            row.setSpacing(10)
            if prop.is_int:  # AddInt :429-475
                spin: QSpinBox | QDoubleSpinBox = SpinBox()
                spin.setRange(int(prop.lo), int(prop.hi))
                spin.setSingleStep(int(prop.step))
                spin.setValue(int(value))  # type: ignore[arg-type]
            else:  # AddFloat :477-529
                spin = DoubleSpinBox()
                spin.setDecimals(float_decimals(prop.step))
                spin.setRange(prop.lo, prop.hi)
                spin.setSingleStep(prop.step)
                spin.setValue(float(value))  # type: ignore[arg-type]
            spin.setSuffix(prop.suffix)
            spin.setAccessibleName(prop.label)
            spin.lineEdit().setTextMargins(4, 0, 0, 0)
            if prop.widget == SLIDER:
                if prop.is_int:
                    slider: QSlider = Slider()
                    slider.setRange(int(prop.lo), int(prop.hi))
                    slider.setPageStep(int(prop.step))
                    slider.setValue(int(value))  # type: ignore[arg-type]
                    slider.valueChanged.connect(spin.setValue)
                    spin.valueChanged.connect(slider.setValue)  # type: ignore[arg-type]
                else:
                    double = DoubleSlider()
                    double.set_constraints(prop.lo, prop.hi, prop.step, float(value))  # type: ignore[arg-type]
                    double.double_changed.connect(spin.setValue)
                    spin.valueChanged.connect(double.set_double)  # type: ignore[arg-type]
                    slider = double
                slider.setAccessibleName(prop.label)
                self.sliders[key] = slider
                row.addWidget(slider, 1)
                spin.setFixedWidth(118)
            row.addWidget(spin, 0 if prop.widget == SLIDER else 1)
            spin.valueChanged.connect(lambda v, key=key, is_int=prop.is_int: self._emit(key, int(v) if is_int else v))
            self.controls[key] = spin
            self.form.addRow(self._label(prop), row)
        elif prop.widget == LIST:  # AddList :625-711
            combo = Combo()
            combo.setAccessibleName(prop.label)
            combo.setMaxVisibleItems(40)
            options = [(CHOOSE_PLUGIN, ""), *plugins] if key == "plugin_path" else [(lb, v) for v, lb in prop.options]
            for text, data in options:
                combo.addItem(text, data)
            index = combo.findData(value)
            if index != -1:
                combo.setCurrentIndex(index)
            combo.currentIndexChanged.connect(lambda i, key=key, combo=combo: self._emit(key, combo.itemData(i)))
            self.controls[key] = combo
            self.form.addRow(self._label(prop), combo)
        elif prop.widget == CHECKBOX:  # AddCheckbox :268-305: the box carries its own words, no label
            box = CheckBox(prop.label)
            box.setChecked(value is True)
            box.toggled.connect(lambda on, key=key: self._emit(key, on))
            self.controls[key] = box
            self.form.addRow(box)
        elif prop.widget == BUTTON:  # AddButton :778-785
            button = QPushButton(prop.label)
            button.setProperty("role", "btn")
            button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
            button.clicked.connect(lambda _=False, key=key: self.clicked.emit(key))
            self.controls[key] = button
            self.form.addRow(button)

    def _emit(self, key: str, value: object) -> None:
        if key == "method" and self.spec.kind == fine_tune.NVIDIA_KIND:
            self._nvidia_vad(str(value))
        self.changed.emit(key, value)

    def _nvidia_vad(self, method: str) -> None:
        """nvidia_audio_method_modified :815-826: Voice Optimized is shown from SDK 1.6.1.2, but not for room echo
        removal alone."""
        box = self.controls.get("vad")
        if box is not None and self.runtime is not None:
            self.form.setRowVisible(box, self.runtime.vad_supported and method != nvafx.DEREVERB)

    def show_buttons(self, open_shown: bool, close_shown: bool) -> None:
        """obs_property_set_visible for the two interface buttons (vst_changed :270-294)."""
        for key, shown in (("open_vst_settings", open_shown), ("close_vst_settings", close_shown)):
            widget = self.controls.get(key)
            if widget is not None:
                self.form.setRowVisible(widget, shown)


# ------------------------------------------------------------------ the list (VisibilityItemWidget / Delegate)

UID_ROLE = Qt.ItemDataRole.UserRole
ENABLED_ROLE = Qt.ItemDataRole.UserRole + 1
EYE_W = 34  # the eye's column on the left of each row
ROW_H = 34


class _Rows(QStyledItemDelegate):
    """Each row: the eye (crossed out when the filter is off), then the name, ``muted`` when off; the selected row
    is filled with ``desk`` at 15 %. The rename editor sits right of the eye."""

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex | object) -> QSize:
        return QSize(LIST_W - 4, ROW_H)

    def paint(self, p: QPainter, option: QStyleOptionViewItem, index: QModelIndex | object) -> None:  # type: ignore[override]
        assert isinstance(index, QModelIndex)
        pal = theme.current()
        rect = QRectF(option.rect).adjusted(3, 1, -3, -1)  # type: ignore[attr-defined]
        state = option.state  # type: ignore[attr-defined]
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = bool(state & QStyle.StateFlag.State_Selected)
        if selected or state & QStyle.StateFlag.State_MouseOver:
            fill = QColor(pal.desk)
            fill.setAlphaF(0.15)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(fill if selected else QColor(pal.hover))
            p.drawRoundedRect(rect, 4, 4)
        on = bool(index.data(ENABLED_ROLE))
        tone = pal.ink if on else pal.muted
        paint_icon(p, "eye" if on else "eye-off", QRectF(rect.left() + 8, rect.center().y() - 8, 16, 16), tone)
        widget = option.widget  # type: ignore[attr-defined]
        font = ui_font(widget, 13, QFont.Weight.Medium if selected else QFont.Weight.Normal)
        p.setFont(font)
        p.setPen(QColor(tone))
        text_rect = rect.adjusted(EYE_W, 0, -6, 0)
        text = p.fontMetrics().elidedText(str(index.data()), Qt.TextElideMode.ElideRight, int(text_rect.width()))
        p.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), text)
        if selected and isinstance(widget, QWidget) and keyboard_focus(widget):
            p.setPen(QColor(pal.desk))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
        p.restore()

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex | object) -> QWidget:
        edit = QLineEdit(parent)
        edit.setObjectName("renameEdit")
        return edit

    def updateEditorGeometry(self, editor: QWidget, option: QStyleOptionViewItem, index: QModelIndex | object) -> None:
        r: QRect = option.rect  # type: ignore[attr-defined]
        editor.setGeometry(r.adjusted(EYE_W, 3, -6, -3))

    def setModelData(self, editor: QWidget, model: object, index: QModelIndex | object) -> None:
        """Nothing here: the window takes the name when the editor closes (FilterNameEdited :977-1032)."""


class FilterList(QListWidget):
    """The filter list. A click on a row's eye switches the filter on or off without selecting the row; Space does it
    for the current row."""

    eye_clicked = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("filterList")
        self.setItemDelegate(_Rows(self))
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
        )  # double-click (EditItem :893-910); F2 / Return on macOS, as actionRenameFilter (:144-148)
        self.setMouseTracking(True)
        self.setUniformItemSizes(True)
        self.setFixedWidth(LIST_W)
        self.setAccessibleName("Filters")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        index = self.indexAt(event.position().toPoint())
        if index.isValid() and event.position().x() < EYE_W + 3 and event.button() == Qt.MouseButton.LeftButton:
            self.eye_clicked.emit(index.row())
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.position().x() < EYE_W + 3:
            return  # a quick second click on the eye switches it again, it doesn't rename
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and self.state() != QAbstractItemView.State.EditingState:
            if self.currentRow() >= 0:
                self.eye_clicked.emit(self.currentRow())
            return
        super().keyPressEvent(event)


# ------------------------------------------------------------------ the window


def style_sheet(pal: Palette) -> str:
    desk_15 = theme.rgba(pal.desk, 0.15)
    return f"""
QWidget#filterWindow, QWidget#pane, QScrollArea#paneScroll, QWidget#paneHolder {{ background: {pal.surface}; }}
QLabel {{ color: {pal.ink}; font-size: 13px; }}
QLabel[role="listTitle"] {{ font-size: 16px; font-weight: 600; }}
QLabel[role="paneLabel"] {{ color: {pal.ink}; }}
QLabel[role="paneNote"], QLabel[role="empty"] {{ color: {pal.muted}; }}
QLabel[role="paneStatus"] {{ color: {pal.muted}; font-size: 12px; }}
QLabel[role="paneTrademark"] {{ color: {pal.muted}; font-size: 12px; margin: 0; padding: 12px 0 0 0; }}
QLabel[role="paneStatus"][warn="true"] {{ color: {pal.hot_text}; font-size: 13px; }}
QListWidget#filterList {{ background: {pal.surface}; border: 1px solid {pal.line}; border-radius: 4px; padding: 3px 0;
    outline: none; }}
QLineEdit {{ background: {pal.surface}; color: {pal.ink}; border: 1px solid {pal.desk}; border-radius: 3px;
    padding: 2px 4px; selection-background-color: {desk_15}; selection-color: {pal.ink}; }}
QSpinBox, QDoubleSpinBox {{ background: {pal.surface}; color: {pal.ink}; border: 1px solid {pal.line};
    border-radius: 4px; padding: 5px 24px 5px 8px; font-size: 13px; }}
QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {pal.desk}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 16px; border: none; background: transparent; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow, QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: none; width: 0; height: 0; border: none; }}
QComboBox {{ background: {pal.surface}; color: {pal.ink}; border: 1px solid {pal.line}; border-radius: 4px;
    padding: 5px 30px 5px 10px; font-size: 13px; }}
QComboBox:focus {{ border-color: {pal.desk}; }}
QComboBox::drop-down {{ border: none; width: 28px; }}
QComboBox::down-arrow {{ image: none; width: 0; height: 0; border: none; }}
QComboBox QAbstractItemView {{ background: {pal.surface}; color: {pal.ink}; border: 1px solid {pal.line};
    selection-background-color: {pal.hover}; selection-color: {pal.desk}; outline: none; }}
QCheckBox {{ color: {pal.ink}; font-size: 13px; spacing: 8px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {pal.muted}; border-radius: 3px;
    background: {pal.surface}; }}
QCheckBox::indicator:checked {{ background: {pal.desk}; border-color: {pal.desk}; }}
QPushButton[role="btn"] {{ background: {pal.surface}; color: {pal.ink}; border: 1px solid {pal.line};
    border-radius: 6px; padding: 7px 14px; font-size: 13px; font-weight: 500; }}
QPushButton[role="btn"]:hover {{ border-color: {pal.muted}; background: {pal.hover}; }}
QPushButton[role="btn"]:focus {{ border: 2px solid {pal.desk}; padding: 6px 13px; }}
QPushButton[role="btn"]:disabled {{ color: {pal.muted}; }}
QPushButton[role="btn"][primary="true"] {{ background: {pal.desk}; border-color: {pal.desk}; color: {pal.on_desk}; }}
QMenu {{ background: {pal.surface}; color: {pal.ink}; border: 1px solid {pal.line}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 3px; }}
QMenu::item:selected {{ background: {pal.hover}; color: {pal.desk}; }}
QMessageBox, QInputDialog {{ background: {pal.surface}; }}
QMessageBox QPushButton, QInputDialog QPushButton {{ background: {pal.surface}; color: {pal.ink};
    border: 1px solid {pal.line}; border-radius: 6px; padding: 6px 18px; min-width: 56px; font-size: 13px; }}
QMessageBox QPushButton:hover, QInputDialog QPushButton:hover {{ border-color: {pal.muted}; background: {pal.hover}; }}
QMessageBox QPushButton:focus, QInputDialog QPushButton:focus {{ border: 2px solid {pal.desk}; padding: 5px 17px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {pal.line}; border-radius: 3px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QToolTip {{ background: {pal.surface}; color: {pal.ink}; border: 1px solid {pal.line}; }}
"""


class FilterWindow(QWidget):
    """The filter window: non-modal, its own top-level window. ``closed`` fires when it closes, so the app saves its
    choices (OBSBasicFilters::closeEvent :602-612 saves the project)."""

    closed = Signal()

    def __init__(
        self,
        model: FilterModel,
        palette: Palette | None = None,
        platform: str = sys.platform,
        runtime: Callable[[], nvafx.Runtime | None] = nvafx.runtime,
    ) -> None:
        """``runtime``: NVIDIA's audio effects runtime if this computer has it (``nvafx.runtime``); the tests and the
        preview give their own."""
        super().__init__(None, Qt.WindowType.Window)
        self.setObjectName("filterWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setWindowTitle(TITLE)
        self.model = model
        self.platform = platform
        self.runtime = runtime()  # found once, as OBS's module does at load (nv-filters.c :23-28)
        self.pane: Pane | None = None
        self._pane_uid: int | None = None
        self._plugins: list[tuple[str, str]] = []  # the plug-ins found when the VST pane was built
        self._watched: QWidget | None = None  # the plug-in editor whose closing the pane follows
        font = QFont()
        font.setFamilies(fonts.body())
        font.setPixelSize(13)
        self.setFont(font)
        self._build()
        self.set_theme(palette or theme.current())
        self.resize(*SIZE)
        self.setMinimumSize(*MIN_SIZE)
        self.refresh(select=0)
        self.show_presets()

    # -------------------------------------------------------------- layout

    def _build(self) -> None:
        grid = QGridLayout(self)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        title = QLabel(LIST_TITLE)
        title.setProperty("role", "listTitle")
        head = QFont(self.font())
        head.setFamilies(fonts.headline())
        head.setPixelSize(16)
        head.setWeight(QFont.Weight.DemiBold)
        title.setFont(head)
        grid.addWidget(title, 0, 0)
        # B013: the presets row, above the properties pane: a dropdown and Save, Load, Delete.
        presets = QHBoxLayout()
        presets.setSpacing(8)
        presets.addWidget(QLabel(PRESET_LABEL))
        self.preset_combo = Combo()
        self.preset_combo.setAccessibleName(PRESET_LABEL)
        self.preset_combo.setPlaceholderText(PRESET_PLACEHOLDER)
        presets.addWidget(self.preset_combo, 1)
        self.preset_save_button = QPushButton("Save")
        self.preset_load_button = QPushButton("Load")
        self.preset_delete_button = QPushButton("Delete")
        for button in (self.preset_save_button, self.preset_load_button, self.preset_delete_button):
            button.setProperty("role", "btn")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            presets.addWidget(button)
        self.preset_combo.currentIndexChanged.connect(self._show_preset_buttons)
        self.preset_save_button.clicked.connect(self.save_preset)
        self.preset_load_button.clicked.connect(self.load_preset)
        self.preset_delete_button.clicked.connect(self.delete_preset)
        grid.addLayout(presets, 0, 1)
        self.list = FilterList()
        self.list.currentRowChanged.connect(self._row_changed)
        self.list.eye_clicked.connect(self._eye_clicked)
        self.list.itemDelegate().closeEditor.connect(self._name_edited)
        grid.addWidget(self.list, 1, 0)
        self.pane_scroll = QScrollArea()
        self.pane_scroll.setObjectName("paneScroll")
        self.pane_scroll.setWidgetResizable(True)
        self.pane_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.pane_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        grid.addWidget(self.pane_scroll, 1, 1)
        tools = QHBoxLayout()
        tools.setSpacing(6)
        self.add_button = IconButton("plus", "Add a filter")
        self.remove_button = IconButton("bin", "Remove the selected filter")
        self.up_button = IconButton("chevron-up", "Move the selected filter up")
        self.down_button = IconButton("chevron", "Move the selected filter down")
        for button in (self.add_button, self.remove_button, self.up_button, self.down_button):
            button.setFixedWidth(36)
            tools.addWidget(button)
        tools.addStretch(1)
        grid.addLayout(tools, 2, 0)
        bottom = QHBoxLayout()
        self.defaults_button = QPushButton("Defaults")
        self.close_button = QPushButton("Close")
        for button in (self.defaults_button, self.close_button):
            button.setProperty("role", "btn")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        bottom.addWidget(self.defaults_button)
        bottom.addStretch(1)
        bottom.addWidget(self.close_button)
        grid.addLayout(bottom, 2, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(1, 1)
        # The + menu: Clean Voice's own Denoise and Voice Style, a separator, then OBS's audio filters (ten, and NVIDIA
        # Audio Effects when its runtime is installed), sorted by name (CreateAddFilterPopupMenu :458-516).
        self.add_menu = QMenu("Add", self)
        for position, kind in enumerate(fine_tune.menu_order(self.runtime is not None)):
            if position == len(fine_tune.OWN_KINDS):
                self.add_menu.addSeparator()
            action = self.add_menu.addAction(KINDS[kind].name)
            action.setData(kind)
            action.triggered.connect(lambda _=False, kind=kind: self.add_filter(kind))
        self.add_button.clicked.connect(self.show_add_menu)
        self.remove_button.clicked.connect(self.remove_selected)
        self.up_button.clicked.connect(lambda: self.move_selected(-1))
        self.down_button.clicked.connect(lambda: self.move_selected(1))
        self.defaults_button.clicked.connect(self.reset_selected)
        self.close_button.clicked.connect(self.close)
        self.close_button.setDefault(True)  # :87
        # OBS's actions, active while the list has the focus (on_action*_triggered :816-841; OBSBasicFilters.ui).
        for keys, slot in (
            (QKeySequence(Qt.Key.Key_Delete), self.remove_selected),
            (QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Up), lambda: self.move_selected(-1)),
            (QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_Down), lambda: self.move_selected(1)),
        ):
            action = QAction(self)
            action.setShortcut(keys)
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(lambda _=False, slot=slot: slot() if self.list.hasFocus() else None)
            self.addAction(action)

    def set_theme(self, pal: Palette) -> None:
        self.palette_now = pal
        self.setStyleSheet(style_sheet(pal))
        if self.pane is not None and isinstance(eq := self.pane.controls.get("eq"), EqPanel):
            eq.set_theme(pal)  # the EQ panel has its own style sheet and painted parts (B016)
        for widget in self.findChildren(QWidget):
            widget.update()

    # -------------------------------------------------------------- the list

    def selected(self) -> FilterSpec | None:
        row = self.list.currentRow()
        filters = self.model.filters
        return filters[row] if 0 <= row < len(filters) else None

    def refresh(self, select: int | None = None) -> None:
        """The list again from the model, the row ``select`` (else the current one) selected, and its pane."""
        filters = self.model.filters
        row = self.list.currentRow() if select is None else select
        self.list.blockSignals(True)
        self.list.clear()
        for spec in filters:
            item = QListWidgetItem(spec.name)
            item.setData(UID_ROLE, spec.uid)
            item.setData(ENABLED_ROLE, spec.enabled)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.list.addItem(item)
        if filters:
            self.list.setCurrentRow(min(max(row, 0), len(filters) - 1))
        self.list.blockSignals(False)
        self._show_pane()

    def _row_changed(self, row: int) -> None:
        self._show_pane()

    def _eye_clicked(self, row: int) -> None:
        filters = self.model.filters
        if 0 <= row < len(filters):
            spec = filters[row]
            self.model.set_enabled(spec.uid, not spec.enabled)
            item = self.list.item(row)
            if item is not None:
                item.setData(ENABLED_ROLE, not spec.enabled)

    def show_add_menu(self) -> None:
        """on_addAsyncFilter_clicked :724-731: the list takes the focus and the menu opens (here above the + button,
        where the guest is looking)."""
        self.list.setFocus()
        self.add_menu.exec(self.add_menu_pos())

    def add_menu_pos(self) -> QPoint:
        hint = self.add_menu.sizeHint()
        return self.add_button.mapToGlobal(QPoint(0, -hint.height() - 4))

    def add_filter(self, kind: str) -> FilterSpec | None:
        """AddNewFilter :518-560: ask for a name, prefilled with the kind's name or "Name 2", "Name 3"...; an empty
        or taken name is refused with its message and asked again."""
        while True:
            names = [f.name for f in self.model.filters]
            name = self.ask_name(fine_tune.unique_name(KINDS[kind].name, names))
            if name is None:
                return None
            name = name.strip()
            if not name:
                self.inform(*NAME_EMPTY)
                continue
            if name in names:
                self.inform(*NAME_TAKEN)
                continue
            break
        spec = self.model.add(kind, name)
        self.refresh(select=len(self.model.filters) - 1)
        return spec

    def remove_selected(self) -> None:
        """on_removeAsyncFilter_clicked :733-741, with QueryRemove's question."""
        spec = self.selected()
        if spec is None or not self.confirm(REMOVE_TITLE, remove_text(spec.name)):
            return
        row = self.list.currentRow()
        self.model.remove(spec.uid)
        self.refresh(select=row)

    def move_selected(self, delta: int) -> None:
        spec = self.selected()
        if spec is not None and self.model.move(spec.uid, delta):
            self.refresh(select=self.list.currentRow() + delta)

    def rename_selected(self) -> None:
        """EditItem :893-910: the name becomes editable in its row."""
        item = self.list.currentItem()
        if item is not None:
            self.list.editItem(item)

    def _name_edited(self, editor: QWidget, hint: QAbstractItemDelegate.EndEditHint) -> None:
        """FilterNameEdited :977-1032. Escape keeps the old name."""
        row = self.list.currentRow()
        filters = self.model.filters
        if not isinstance(editor, QLineEdit) or not 0 <= row < len(filters):
            return
        if hint == QAbstractItemDelegate.EndEditHint.RevertModelCache:
            return
        error = self.model.rename(filters[row].uid, editor.text())
        self.refresh(select=row)
        if error is not None:
            self.inform(*error)

    def reset_selected(self) -> None:
        """Defaults: ResetFilters :1044-1071, after ConfirmReset's question :1034-1042."""
        spec = self.selected()
        if spec is None or not self.confirm(RESET_TITLE, RESET_TEXT):
            return
        self.model.reset(spec.uid)
        self._show_pane(force=True)

    # -------------------------------------------------------------- presets (B013)

    def show_presets(self, select: str | None = None) -> None:
        """The dropdown again from the model, ``select`` chosen (else nothing)."""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        names = self.model.preset_names()
        self.preset_combo.addItems(names)
        self.preset_combo.setCurrentIndex(names.index(select) if select in names else -1)
        self.preset_combo.blockSignals(False)
        self._show_preset_buttons()

    def _show_preset_buttons(self) -> None:
        """Load and Delete need a preset chosen."""
        chosen = self.preset_combo.currentIndex() >= 0
        self.preset_load_button.setEnabled(chosen)
        self.preset_delete_button.setEnabled(chosen)

    def save_preset(self) -> None:
        """Ask for a name (the chosen preset's is prefilled), ask before replacing one that exists, store the chain."""
        taken = self.model.preset_names()
        while True:
            name = self.ask_preset_name(self.preset_combo.currentText())
            if name is None:
                return
            name = name.strip()
            if name:
                break
            self.inform(*PRESET_EMPTY)
        if name in taken and not self.confirm(PRESET_REPLACE_TITLE, replace_preset_text(name)):
            return
        self.model.save_preset(name)
        self.show_presets(select=name)

    def load_preset(self) -> None:
        name = self.preset_combo.currentText()
        if self.preset_combo.currentIndex() < 0:
            return
        self.model.load_preset(name)
        self.refresh(select=0)

    def delete_preset(self) -> None:
        name = self.preset_combo.currentText()
        if self.preset_combo.currentIndex() < 0 or not self.confirm(PRESET_DELETE_TITLE, delete_preset_text(name)):
            return
        self.model.delete_preset(name)
        self.show_presets()

    # -------------------------------------------------------------- the properties pane

    def _show_pane(self, force: bool = False) -> None:
        """UpdatePropertiesView :218-275: a new pane only for another filter (or after a change that rebuilds it)."""
        spec = self.selected()
        uid = spec.uid if spec is not None else None
        self.defaults_button.setEnabled(spec is not None)
        for button in (self.remove_button, self.up_button, self.down_button):
            button.setEnabled(spec is not None)
        if not force and uid == self._pane_uid and self.pane is not None:
            self.pane.spec = spec  # type: ignore[assignment]
            return
        self._pane_uid = uid
        old = self.pane_scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self.pane = None
        holder = QWidget()
        holder.setObjectName("paneHolder")
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 2, 0, 0)
        if spec is None:
            empty = QLabel(EMPTY_TEXT)
            empty.setProperty("role", "empty")
            empty.setWordWrap(True)
            column.addWidget(empty)
        else:
            plugins: list[tuple[str, str]] = []
            if spec.kind == vst.KIND:
                plugins = self.model.scan()  # fill_out_plugins runs each time the properties are built (:302)
                spec = self._unlisted_plugin(spec, plugins)
            self._plugins = plugins
            pane = Pane(spec, plugins, self.runtime, self.palette_now)
            pane.changed.connect(self._setting_changed)
            pane.clicked.connect(self._button_clicked)
            self.pane = pane
            column.addWidget(pane)
            if spec.kind == fine_tune.EQ_KIND:  # the EQ's graph and ten rows need room: the window grows, never shrinks
                self.fit_eq()
            if spec.kind == vst.KIND:
                self.show_vst_state()
        column.addStretch(1)
        self.pane_scroll.setWidget(holder)

    def fit_eq(self) -> None:
        """The EQ's room (B015): grow to EQ_SIZE, but never past the screen's available area (QScreen.availableGeometry:
        the taskbar and display scaling already taken out), frame included. Below that the pane scrolls."""
        if self.pane is None or self.pane.controls.get("eq") is None:
            return
        room = self.screen().availableGeometry()
        frame = self.frameGeometry()
        extra_w, extra_h = max(0, frame.width() - self.width()), max(0, frame.height() - self.height())
        w = min(max(self.width(), EQ_SIZE[0]), room.width() - extra_w)
        h = min(max(self.height(), EQ_SIZE[1]), room.height() - extra_h)
        self.resize(max(w, 1), max(h, 1))

    def _unlisted_plugin(self, spec: FilterSpec, plugins: list[tuple[str, str]]) -> FilterSpec:
        """AddList :700-708: a saved plug-in that isn't in the list found now makes the list's first entry the choice,
        which clears the plug-in, as OBS does."""
        path = spec.settings.get("plugin_path")
        if path and path not in [p for _, p in plugins]:
            new = self.model.choose_plugin(spec.uid, "")
            if new is not None:
                self._pane_uid = new.uid
                return new
        return spec

    def show_vst_state(self) -> None:
        """The VST pane's interface buttons (vst_changed :270-294) and its status line: the loaded effect, why it
        failed to load, or that no plug-ins were found and where Clean Voice looked."""
        pane, spec = self.pane, self.selected()
        if pane is None or spec is None:
            return
        plugin = self.model.plugin(spec.uid)
        pane.show_buttons(*vst_editor.interface_buttons(plugin))
        error = self.model.errors.get(spec.uid)
        if error is not None:
            pane.set_status(load_failed_text(error), warn=True)
        elif plugin is not None:
            pane.set_status(plugin_text(plugin))
        elif not self._plugins:
            pane.set_status(no_plugins_text(self.model.folders()))
        else:
            pane.set_status("")
        editor = plugin.editor if plugin is not None else None
        if isinstance(editor, QWidget) and editor is not self._watched:
            self._watched = editor
            editor.destroyed.connect(self._editor_gone)

    def _editor_gone(self, *_: object) -> None:
        """The plug-in's editor window was closed by its own title bar: the pane shows Open again."""
        self._watched = None
        self.show_vst_state()

    def _setting_changed(self, key: str, value: object) -> None:
        spec = self.selected()
        if spec is None:
            return
        if self.model.set_setting(spec.uid, key, value):
            self._show_pane(force=True)

    def _button_clicked(self, key: str) -> None:
        spec = self.selected()
        if spec is None:
            return
        if key == "open_vst_settings":
            self.model.open_editor(spec.uid)
        elif key == "close_vst_settings":
            self.model.close_editor(spec.uid)
        self.show_vst_state()

    # -------------------------------------------------------------- questions (replaced in the tests)

    def confirm_box(self, title: str, text: str) -> QMessageBox:
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(QMessageBox.Icon.Question)
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        box.setAutoFillBackground(True)
        return box

    def confirm(self, title: str, text: str) -> bool:
        return self.confirm_box(title, text).exec() == QMessageBox.StandardButton.Yes

    def ask_name(self, default: str) -> str | None:
        """NameDialog::AskForName."""
        text, ok = QInputDialog.getText(self, ADD_TITLE, ADD_TEXT, QLineEdit.EchoMode.Normal, default)
        return text if ok else None

    def ask_preset_name(self, default: str) -> str | None:
        text, ok = QInputDialog.getText(self, PRESET_TITLE, PRESET_TEXT, QLineEdit.EchoMode.Normal, default)
        return text if ok else None

    def inform(self, title: str, text: str) -> None:
        QMessageBox.information(self, title, text)

    # -------------------------------------------------------------- window life

    def showEvent(self, event: object) -> None:
        super().showEvent(event)  # type: ignore[arg-type]
        self.fit_eq()  # now the screen it opens on is known
        self.list.setFocus(Qt.FocusReason.OtherFocusReason)  # the list first, as OBS's dialog opens (:110-112)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        super().closeEvent(event)
        if event.isAccepted():
            self.closed.emit()
