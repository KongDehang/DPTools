from __future__ import annotations

import os
import random
import webbrowser
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import shutil
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from PyQt6.QtCore import QEvent, QSettings, QSignalBlocker, QSize, Qt, QThread, QTimer
from PyQt6.QtGui import QAction, QCloseEvent, QIcon, QImage, QKeyEvent, QKeySequence, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressDialog,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QStatusBar,
    QSlider,
    QTabWidget,
    QToolBar,
    QStyle,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..constants import (
    APP_NAME,
    DEFAULT_AUTOSAVE_SECONDS,
    DEFAULT_ZOOM_SCALE,
    FORMAT_LABELS,
    FORMAT_SUFFIX,
    IMAGE_EXTENSIONS,
    MAX_ZOOM_SCALE,
    MIN_ZOOM_SCALE,
    ZOOM_STEP,
)
from ..models import AnnotationDocument, Box
from ..services import (
    AnnotationIO,
    AutosaveController,
    ClassManager,
    DatasetService,
    DocumentHistory,
    default_shortcut_bindings,
    load_shortcut_bindings,
    save_shortcut_bindings,
    to_key_sequence,
)
from ..services.background_tasks import (
    DatasetScanResult,
    DatasetScanWorker,
    DatasetStatisticsResult,
    DatasetStatisticsWorker,
    ThumbnailLoadWorker,
)
from ..services.class_manager import ClassManagerState
from .class_selector_dialog import ClassSelectorDialog
from .class_checklist_dialog import ClassChecklistDialog
from .canvas import AnnotationCanvas
from .shortcut_editor_dialog import ShortcutEditorDialog
from .statistics_widget import DatasetStatisticsWidget
from .widgets import CollapsibleSection


@dataclass(slots=True)
class DeletedDatasetItem:
    image_path: Path
    image_bytes: bytes
    label_path: Path | None
    label_bytes: bytes | None
    source_index: int


@dataclass(slots=True)
class DatasetDeleteRecord:
    items: list[DeletedDatasetItem]
    preferred_after_delete: Path | None


@dataclass(slots=True)
class MergeImageTask:
    root_name: str
    source_dataset_tag: str
    source_image_path: Path
    source_label_path: Path | None
    source_manager_state: ClassManagerState
    source_class_id_to_target_id: dict[int, int]
    output_image_path: Path
    output_label_path: Path


@dataclass(slots=True)
class MergeDatasetLayout:
    mode: str
    image_templates: tuple[str, ...]
    label_templates: tuple[str, ...]
    split_names: tuple[str, ...]
    image_split_map: dict[Path, str]

    @property
    def signature(self) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        return (self.mode, self.image_templates, self.label_templates, self.split_names)


@dataclass(slots=True)
class MergeDatasetSource:
    root: Path
    service: DatasetService
    source_manager_state: ClassManagerState
    layout: MergeDatasetLayout
    dataset_tag: str
    class_id_map: dict[int, int]


@dataclass(slots=True)
class ExtractDatasetItem:
    image_path: Path
    label_path: Path | None
    relative_path: Path
    class_counts: dict[int, int]


class AnnotationMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)

        self.dataset_service = DatasetService()
        self.class_manager = ClassManager()
        self.annotation_io = AnnotationIO()
        self.history = DocumentHistory()
        self.autosave = AutosaveController(int(DEFAULT_AUTOSAVE_SECONDS * 1000), self)
        self.autosave.set_save_callback(self._autosave_flush)

        self.current_document: AnnotationDocument | None = None
        self.current_image: QImage | None = None
        self.current_image_index: int = -1
        self.current_zoom_scale = DEFAULT_ZOOM_SCALE
        self.draw_mode = False
        raw_draw_shape = str(QSettings().value("ui/draw_shape", "rectangle") or "rectangle").strip().lower()
        self.draw_shape = "polygon" if raw_draw_shape == "polygon" else "rectangle"
        self.edit_mode = False
        self._image_filter_text = ""
        self._pending_class_text = self._load_default_class_name()
        self._autosave_enabled_pref = self._read_bool_setting("ui/autosave_enabled", True)
        self._autosave_interval_seconds_pref = self._read_int_setting(
            "ui/autosave_interval_seconds",
            max(1, int(DEFAULT_AUTOSAVE_SECONDS)),
            minimum=1,
            maximum=120,
        )
        self._label_bg_alpha_pref = self._read_int_setting(
            "ui/label_bg_alpha",
            140,
            minimum=0,
            maximum=255,
        )
        self._label_show_name_pref = self._read_bool_setting("ui/label_show_name", True)
        self._label_show_id_pref = self._read_bool_setting("ui/label_show_id", False)
        self._shortcut_bindings = load_shortcut_bindings()
        self._shortcut_sequences: dict[str, QKeySequence] = {}
        self._image_index_map: dict[Path, int] = {}
        self._thumbnail_cache: dict[Path, tuple[tuple[int, int], QPixmap]] = {}
        self._dataset_stats_baseline_counts: dict[str, int] = {}
        self._dataset_stats_total_images = 0
        self._dataset_stats_annotated_images = 0
        self._dataset_stats_total_boxes = 0
        self._current_document_saved_counts: dict[str, int] = {}
        self._current_document_saved_box_count = 0
        self._current_document_saved_is_annotated = False
        self._dataset_loading = False
        self._dataset_statistics_loading = False
        self._dataset_job_generation = 0
        self._dataset_stats_job_generation = 0
        self._thumbnail_job_generation = 0
        self._thumbnail_visible_paths: list[Path] = []
        self._thumbnail_pending_paths: list[Path] = []
        self._thumbnail_item_index_map: dict[Path, int] = {}
        self._thumbnail_icon_width = 128
        self._thumbnail_min_icon_width = 84
        self._thumbnail_max_icon_width = 256
        self._thumbnail_zoom_step = 10
        self._thumbnail_refresh_pending = False
        self._statistics_refresh_pending = False
        self._dataset_scan_thread: QThread | None = None
        self._dataset_scan_worker: DatasetScanWorker | None = None
        self._dataset_statistics_thread: QThread | None = None
        self._dataset_statistics_worker: DatasetStatisticsWorker | None = None
        self._thumbnail_thread: QThread | None = None
        self._thumbnail_worker: ThumbnailLoadWorker | None = None
        self._icons_dir = Path(__file__).resolve().parent.parent / "assets" / "icons"
        self._pending_dataset_root: Path | None = None
        self._pending_dataset_preferred_path: Path | None = None
        self._dataset_delete_undo_stack: list[DatasetDeleteRecord] = []
        self._dataset_delete_redo_stack: list[DatasetDeleteRecord] = []
        self._dataset_delete_history_limit = 20
        self.sidebar_dock: QDockWidget | None = None
        self.thumbnail_dock: QDockWidget | None = None
        self.statistics_dock: QDockWidget | None = None
        self._sidebar_tab_order: tuple[str, ...] = ("project", "class", "annotation", "export", "settings")
        self._sidebar_tab_labels: dict[str, str] = {
            "project": "项目",
            "class": "类别",
            "annotation": "标注",
            "export": "导出",
            "settings": "设置",
        }
        self._sidebar_tab_tooltips: dict[str, str] = {
            "project": "项目与图像列表",
            "class": "类别管理与 YAML",
            "annotation": "标注编辑与快捷操作",
            "export": "导出与格式转换",
            "settings": "集中设置参数与默认值",
        }
        self._sidebar_tab_icons: dict[str, tuple[str, str, str]] = {
            "project": ("tab_project.svg", "SP_DirIcon", "SP_DialogOpenButton"),
            "class": ("tab_class.svg", "SP_FileDialogDetailedView", "SP_FileDialogContentsView"),
            "annotation": ("tab_annotation.svg", "SP_FileDialogContentsView", "SP_FileIcon"),
            "export": ("tab_export.svg", "SP_DialogSaveButton", "SP_DriveFDIcon"),
            "settings": ("tab_settings.svg", "SP_FileDialogDetailedView", "SP_FileDialogInfoView"),
        }
        self._sidebar_tab_index_by_id: dict[str, int] = {}
        self._sidebar_tab_toggle_actions: dict[str, QAction] = {}
        self._sidebar_tab_visibility: dict[str, bool] = self._load_sidebar_tab_visibility()

        self._apply_window_icon()

        self._build_ui()
        self._build_toolbar()
        self._build_status_bar()
        self._bind_signals()
        self._set_loaded_state(False)
        self._update_zoom_from_document()
        self._update_window_title()

    def _apply_window_icon(self) -> None:
        icon = self._asset_icon("app_logo.svg", "SP_ComputerIcon", "SP_DesktopIcon")
        if icon.isNull():
            return
        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def _asset_icon(self, asset_name: str, *fallback_standard_pixmaps: str) -> QIcon:
        candidate = self._icons_dir / asset_name
        if candidate.exists():
            icon = QIcon(str(candidate))
            if not icon.isNull():
                return icon

        style = self.style()
        for fallback_name in fallback_standard_pixmaps:
            standard_pixmap = getattr(QStyle.StandardPixmap, fallback_name, None)
            if standard_pixmap is None:
                continue
            icon = style.standardIcon(standard_pixmap)
            if not icon.isNull():
                return icon

        return QIcon()

    def _dialog_icon(self, level: str) -> QIcon:
        standard_map = {
            "info": QStyle.StandardPixmap.SP_MessageBoxInformation,
            "warning": QStyle.StandardPixmap.SP_MessageBoxWarning,
            "error": QStyle.StandardPixmap.SP_MessageBoxCritical,
            "question": QStyle.StandardPixmap.SP_MessageBoxQuestion,
        }
        standard_pixmap = standard_map.get(level, QStyle.StandardPixmap.SP_MessageBoxInformation)
        icon = self.style().standardIcon(standard_pixmap)
        if not icon.isNull():
            return icon
        return self._asset_icon("app_logo.svg", "SP_ComputerIcon", "SP_DesktopIcon")

    def _recommended_parallel_workers(self, task_count: int) -> int:
        if task_count <= 0:
            return 1
        import os
        logical_cores = os.cpu_count() or 4
        # I/O dominated tasks benefit from a slightly wider pool than CPU count.
        return max(2, min(12, task_count, logical_cores * 2))

    def _dialog_stylesheet(self) -> str:
        return """
QMessageBox {
    background: #161d26;
}

QMessageBox QLabel {
    color: #e7edf5;
}

QMessageBox QPushButton {
    min-width: 84px;
    padding: 6px 14px;
}
"""

    def _localize_dialog_buttons(
        self,
        message_box: QMessageBox,
        button_texts: dict[QMessageBox.StandardButton, str] | None,
    ) -> None:
        default_texts: dict[QMessageBox.StandardButton, str] = {
            QMessageBox.StandardButton.Ok: "确定",
            QMessageBox.StandardButton.Cancel: "取消",
            QMessageBox.StandardButton.Yes: "是",
            QMessageBox.StandardButton.No: "否",
            QMessageBox.StandardButton.Save: "保存",
            QMessageBox.StandardButton.Discard: "不保存",
            QMessageBox.StandardButton.Close: "关闭",
            QMessageBox.StandardButton.Retry: "重试",
            QMessageBox.StandardButton.Ignore: "忽略",
            QMessageBox.StandardButton.Abort: "中止",
            QMessageBox.StandardButton.Apply: "应用",
        }
        for standard_button, default_text in default_texts.items():
            button = message_box.button(standard_button)
            if button is None:
                continue
            text = default_text
            if button_texts is not None and standard_button in button_texts:
                text = button_texts[standard_button]
            button.setText(text)

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
        )

        self.canvas = AnnotationCanvas(self)
        self.canvas.drawBoxRequested.connect(self.on_draw_box_requested)
        self.canvas.drawPolygonRequested.connect(self.on_draw_polygon_requested)
        self.canvas.panRequested.connect(self.on_canvas_pan_requested)
        self.canvas.annotationChanged.connect(self.on_canvas_annotation_changed)
        self.canvas.selectionChanged.connect(self.on_canvas_selection_changed)
        self.canvas.cursorPositionChanged.connect(self.on_canvas_cursor_changed)
        self.canvas.setStyleSheet("background: #0f141a;")

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setStyleSheet("background: #0f141a; border: none;")
        viewport = self.scroll_area.viewport()
        assert viewport is not None
        viewport.installEventFilter(self)
        main_layout.addWidget(self.scroll_area, 1)

        self.sidebar_widget = QWidget(self)
        self.sidebar_widget.setMinimumWidth(360)
        sidebar_layout = QVBoxLayout(self.sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        self.sidebar_tabs = QTabWidget(self.sidebar_widget)
        self.sidebar_tabs.setDocumentMode(True)
        self.sidebar_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sidebar_tab_bar = self.sidebar_tabs.tabBar()
        assert sidebar_tab_bar is not None
        sidebar_tab_bar.setExpanding(True)
        sidebar_tab_bar.setUsesScrollButtons(False)
        sidebar_tab_bar.setElideMode(Qt.TextElideMode.ElideNone)
        self.sidebar_tabs.addTab(self._build_project_tab(), self._sidebar_tab_labels["project"])
        self.sidebar_tabs.addTab(self._build_class_tab(), self._sidebar_tab_labels["class"])
        self.sidebar_tabs.addTab(self._build_annotation_tab(), self._sidebar_tab_labels["annotation"])
        self.sidebar_tabs.addTab(self._build_export_tab(), self._sidebar_tab_labels["export"])
        self.sidebar_tabs.addTab(self._build_settings_tab(), self._sidebar_tab_labels["settings"])
        self._sidebar_tab_index_by_id = {tab_id: index for index, tab_id in enumerate(self._sidebar_tab_order)}
        self._apply_sidebar_tab_icons()
        sidebar_layout.addWidget(self.sidebar_tabs, 1)

        self.sidebar_dock = QDockWidget("标注控制台", self)
        self.sidebar_dock.setObjectName("sidebarDock")
        self.sidebar_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.sidebar_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.sidebar_dock.setWidget(self.sidebar_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.sidebar_dock)
        self.sidebar_dock.visibilityChanged.connect(self._on_dock_visibility_changed)

        self.thumbnail_dock = QDockWidget("缩略图浏览", self)
        self.thumbnail_dock.setObjectName("thumbnailDock")
        self.thumbnail_dock.setMinimumWidth(300)
        self.thumbnail_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.thumbnail_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.thumbnail_dock.setWidget(self._build_thumbnail_panel())
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.thumbnail_dock)
        self.thumbnail_dock.visibilityChanged.connect(self._on_dock_visibility_changed)

        self.statistics_dock = QDockWidget("样本统计", self)
        self.statistics_dock.setObjectName("statisticsDock")
        self.statistics_dock.setMinimumWidth(360)
        self.statistics_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.statistics_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.statistics_dock.setWidget(self._build_statistics_panel())
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.statistics_dock)
        self.splitDockWidget(self.thumbnail_dock, self.statistics_dock, Qt.Orientation.Vertical)
        self.statistics_dock.visibilityChanged.connect(self._on_dock_visibility_changed)

        menu_bar = self.menuBar()
        assert menu_bar is not None
        panel_menu = menu_bar.addMenu("面板")
        assert panel_menu is not None
        sidebar_toggle_action = self.sidebar_dock.toggleViewAction()
        thumbnail_toggle_action = self.thumbnail_dock.toggleViewAction()
        statistics_toggle_action = self.statistics_dock.toggleViewAction()
        assert sidebar_toggle_action is not None
        assert thumbnail_toggle_action is not None
        assert statistics_toggle_action is not None
        panel_menu.addAction(sidebar_toggle_action)
        panel_menu.addAction(thumbnail_toggle_action)
        panel_menu.addAction(statistics_toggle_action)
        sidebar_toggle_action.toggled.connect(lambda _checked: QTimer.singleShot(0, self._refresh_dock_layout))
        thumbnail_toggle_action.toggled.connect(lambda _checked: QTimer.singleShot(0, self._refresh_dock_layout))
        statistics_toggle_action.toggled.connect(lambda _checked: QTimer.singleShot(0, self._refresh_dock_layout))
        tab_visibility_menu = panel_menu.addMenu("选项卡显示")
        assert tab_visibility_menu is not None
        for tab_id in self._sidebar_tab_order:
            action = QAction(self._sidebar_tab_labels[tab_id], self)
            action.setCheckable(True)
            action.setChecked(bool(self._sidebar_tab_visibility.get(tab_id, True)))
            action.toggled.connect(lambda checked, target_id=tab_id: self._on_sidebar_tab_visibility_toggled(target_id, checked))
            tab_visibility_menu.addAction(action)
            self._sidebar_tab_toggle_actions[tab_id] = action
        self._apply_sidebar_tab_visibility()
        panel_menu.addSeparator()
        reset_layout_action = QAction("重置面板布局", self)
        reset_layout_action.triggered.connect(self.reset_panel_layout)
        panel_menu.addAction(reset_layout_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具栏", self)
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(toolbar)

        self.action_open = QAction("打开数据集", self)
        self.action_open.triggered.connect(self.load_dataset)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.setIcon(self._asset_icon("toolbar_open_dataset.svg", "SP_DialogOpenButton", "SP_DirOpenIcon"))
        self.action_open.setToolTip("打开数据集（Ctrl+O）")
        self.action_open.setStatusTip("打开数据集（Ctrl+O）")
        toolbar.addAction(self.action_open)

        self.action_refresh = QAction("刷新", self)
        self.action_refresh.triggered.connect(self.refresh_dataset)
        self.action_refresh.setShortcut(QKeySequence.StandardKey.Refresh)
        self.action_refresh.setIcon(self._asset_icon("toolbar_refresh.svg", "SP_BrowserReload"))
        self.action_refresh.setToolTip("刷新数据集（F5）")
        self.action_refresh.setStatusTip("刷新数据集（F5）")
        toolbar.addAction(self.action_refresh)

        self.action_save = QAction("保存", self)
        self.action_save.triggered.connect(lambda: self.save_current_annotation())
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.setIcon(self._asset_icon("toolbar_save.svg", "SP_DialogSaveButton"))
        self.action_save.setToolTip("保存当前标注（Ctrl+S）")
        self.action_save.setStatusTip("保存当前标注（Ctrl+S）")
        toolbar.addAction(self.action_save)

        self.action_undo = QAction("撤销", self)
        self.action_undo.triggered.connect(self.undo_last_action)
        self.action_undo.setIcon(self._asset_icon("toolbar_undo.svg", "SP_ArrowBack", "SP_ArrowLeft"))
        self.action_undo.setToolTip("撤销（Ctrl+Z）")
        self.action_undo.setStatusTip("撤销（Ctrl+Z）")
        toolbar.addAction(self.action_undo)

        self.action_redo = QAction("重做", self)
        self.action_redo.triggered.connect(self.redo_last_action)
        self.action_redo.setIcon(self._asset_icon("toolbar_redo.svg", "SP_ArrowForward", "SP_ArrowRight"))
        self.action_redo.setToolTip("重做（Ctrl+Y / Ctrl+Shift+Z）")
        self.action_redo.setStatusTip("重做（Ctrl+Y / Ctrl+Shift+Z）")
        toolbar.addAction(self.action_redo)

        self.action_export = QAction("导出当前", self)
        self.action_export.triggered.connect(self.export_current_annotation)
        self.action_export.setShortcut("Ctrl+Shift+S")
        self.action_export.setIcon(self._asset_icon("toolbar_export.svg", "SP_DriveFDIcon", "SP_DialogSaveButton"))
        self.action_export.setToolTip("导出当前标注（Ctrl+Shift+S）")
        self.action_export.setStatusTip("导出当前标注（Ctrl+Shift+S）")
        toolbar.addAction(self.action_export)

        toolbar.addSeparator()

        self.action_prev = QAction("上一张", self)
        self.action_prev.triggered.connect(self.show_previous_image)
        self.action_prev.setShortcut("Alt+Left")
        self.action_prev.setIcon(
            self._asset_icon("toolbar_prev.svg", "SP_MediaSeekBackward", "SP_ArrowBack", "SP_ArrowLeft")
        )
        self.action_prev.setToolTip("上一张（Alt+Left）")
        self.action_prev.setStatusTip("上一张（Alt+Left）")
        toolbar.addAction(self.action_prev)

        self.action_next = QAction("下一张", self)
        self.action_next.triggered.connect(self.show_next_image)
        self.action_next.setShortcut("Alt+Right")
        self.action_next.setIcon(
            self._asset_icon("toolbar_next.svg", "SP_MediaSeekForward", "SP_ArrowForward", "SP_ArrowRight")
        )
        self.action_next.setToolTip("下一张（Alt+Right）")
        self.action_next.setStatusTip("下一张（Alt+Right）")
        toolbar.addAction(self.action_next)

        toolbar.addSeparator()

        self.action_autosave = QAction("AUTO", self)
        self.action_autosave.setCheckable(True)
        self.action_autosave.toggled.connect(self.toggle_autosave)
        self.action_autosave.setIcon(self._asset_icon("toolbar_autosave.svg", "SP_BrowserReload"))
        self.action_autosave.setToolTip("自动保存")
        self.action_autosave.setStatusTip("自动保存")
        toolbar.addAction(self.action_autosave)

        self.action_toggle_sidebar = QAction("侧栏", self)
        self.action_toggle_sidebar.triggered.connect(self.toggle_sidebar)
        self.action_toggle_sidebar.setIcon(self._asset_icon("toolbar_sidebar.svg", "SP_FileDialogDetailedView", "SP_FileDialogListView"))
        self.action_toggle_sidebar.setToolTip("显示或隐藏侧栏")
        self.action_toggle_sidebar.setStatusTip("显示或隐藏侧栏")
        toolbar.addAction(self.action_toggle_sidebar)

        toolbar.addSeparator()

        self.action_draw = QAction("新建标注", self)
        self.action_draw.setCheckable(True)
        self.action_draw.toggled.connect(self.set_draw_mode)
        self.action_draw.setIcon(self._asset_icon("toolbar_draw.svg", "SP_DialogApplyButton", "SP_DialogYesButton"))
        self.action_draw.setToolTip("切换到绘制模式")
        self.action_draw.setStatusTip("切换到绘制模式")
        toolbar.addAction(self.action_draw)

        self.action_edit = QAction("编辑标注", self)
        self.action_edit.setCheckable(True)
        self.action_edit.toggled.connect(self.set_edit_mode)
        self.action_edit.setIcon(self._asset_icon("toolbar_edit.svg", "SP_FileDialogContentsView", "SP_FileIcon"))
        self.action_edit.setToolTip("切换到编辑模式")
        self.action_edit.setStatusTip("切换到编辑模式")
        toolbar.addAction(self.action_edit)

        toolbar.addSeparator()

        self.action_zoom_in = QAction("放大", self)
        self.action_zoom_in.triggered.connect(self.zoom_in)
        self.action_zoom_in.setIcon(self._asset_icon("toolbar_zoom_in.svg", "SP_ArrowUp", "SP_ArrowForward"))
        self.action_zoom_in.setToolTip("放大")
        self.action_zoom_in.setStatusTip("放大")
        toolbar.addAction(self.action_zoom_in)

        self.action_zoom_out = QAction("缩小", self)
        self.action_zoom_out.triggered.connect(self.zoom_out)
        self.action_zoom_out.setIcon(self._asset_icon("toolbar_zoom_out.svg", "SP_ArrowDown", "SP_ArrowBack"))
        self.action_zoom_out.setToolTip("缩小")
        self.action_zoom_out.setStatusTip("缩小")
        toolbar.addAction(self.action_zoom_out)

        self.action_fit = QAction("适应窗口", self)
        self.action_fit.triggered.connect(self.fit_to_window)
        self.action_fit.setIcon(self._asset_icon("toolbar_fit.svg", "SP_DialogResetButton", "SP_BrowserReload"))
        self.action_fit.setToolTip("适应窗口")
        self.action_fit.setStatusTip("适应窗口")
        toolbar.addAction(self.action_fit)

    def _show_dialog(
        self,
        level: str,
        title: str,
        text: str,
        *,
        informative_text: str | None = None,
        details: str | None = None,
        buttons: QMessageBox.StandardButton = QMessageBox.StandardButton.Ok,
        default_button: QMessageBox.StandardButton | None = None,
        button_texts: dict[QMessageBox.StandardButton, str] | None = None,
    ) -> QMessageBox.StandardButton:
        message_box = QMessageBox(self)
        message_box.setWindowTitle(title)
        message_box.setText(text)
        if informative_text:
            message_box.setInformativeText(informative_text)
        if details:
            message_box.setDetailedText(details)
        message_box.setStandardButtons(buttons)
        if default_button is not None:
            default_widget = message_box.button(default_button)
            if default_widget is not None:
                message_box.setDefaultButton(default_widget)

        message_box.setIcon(QMessageBox.Icon.NoIcon)
        icon = self._dialog_icon(level)
        if not icon.isNull():
            message_box.setIconPixmap(icon.pixmap(20, 20))

        try:
            message_box.setOption(QMessageBox.Option.DontUseNativeDialog, True)
        except Exception:
            pass

        message_box.setStyleSheet(self._dialog_stylesheet())
        layout = message_box.layout()
        if layout is not None:
            layout.setContentsMargins(16, 14, 16, 12)
            layout.setSpacing(8)

        self._localize_dialog_buttons(message_box, button_texts)
        return QMessageBox.StandardButton(message_box.exec())

    def _info_dialog(
        self,
        title: str,
        text: str,
        *,
        informative_text: str | None = None,
        details: str | None = None,
    ) -> None:
        self._show_dialog(
            "info",
            title,
            text,
            informative_text=informative_text,
            details=details,
        )

    def _warning_dialog(
        self,
        title: str,
        text: str,
        *,
        informative_text: str | None = None,
        details: str | None = None,
    ) -> None:
        self._show_dialog(
            "warning",
            title,
            text,
            informative_text=informative_text,
            details=details,
        )

    def _error_dialog(
        self,
        title: str,
        text: str,
        *,
        informative_text: str | None = None,
        details: str | None = None,
    ) -> None:
        self._show_dialog(
            "error",
            title,
            text,
            informative_text=informative_text,
            details=details,
        )

    def _ask_dialog(
        self,
        title: str,
        text: str,
        *,
        informative_text: str | None = None,
        details: str | None = None,
        buttons: QMessageBox.StandardButton = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        default_button: QMessageBox.StandardButton | None = None,
        button_texts: dict[QMessageBox.StandardButton, str] | None = None,
    ) -> QMessageBox.StandardButton:
        return self._show_dialog(
            "question",
            title,
            text,
            informative_text=informative_text,
            details=details,
            buttons=buttons,
            default_button=default_button,
            button_texts=button_texts,
        )

    def _read_bool_setting(self, key: str, default: bool) -> bool:
        raw = QSettings().value(key, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        text = str(raw).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def _read_int_setting(self, key: str, default: int, *, minimum: int, maximum: int) -> int:
        raw = QSettings().value(key, default)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = int(default)
        return max(int(minimum), min(int(maximum), value))

    def _shortcut_text(self, command_key: str) -> str:
        sequence = self._shortcut_sequences.get(command_key)
        if sequence is None or sequence.isEmpty():
            return ""
        return sequence.toString(QKeySequence.SequenceFormat.NativeText)

    def _shortcut_hint(self, *command_keys: str) -> str:
        hints: list[str] = []
        for command_key in command_keys:
            text = self._shortcut_text(command_key)
            if text and text not in hints:
                hints.append(text)
        return " / ".join(hints)

    def _event_matches_shortcut(self, event: QKeyEvent, command_key: str) -> bool:
        sequence = self._shortcut_sequences.get(command_key)
        if sequence is None or sequence.isEmpty():
            return False
        event_sequence = QKeySequence(event.keyCombination())
        return sequence.matches(event_sequence) == QKeySequence.SequenceMatch.ExactMatch

    def _apply_shortcuts_to_actions(self) -> None:
        defaults = default_shortcut_bindings()
        merged: dict[str, str] = {}
        for command_key, default_text in defaults.items():
            merged[command_key] = self._shortcut_bindings.get(command_key, default_text) or default_text
        self._shortcut_bindings = merged
        self._shortcut_sequences = {
            command_key: to_key_sequence(text)
            for command_key, text in self._shortcut_bindings.items()
        }

        def assign_shortcuts(action: QAction, *command_keys: str) -> None:
            sequences = [self._shortcut_sequences.get(command_key, QKeySequence()) for command_key in command_keys]
            valid = [sequence for sequence in sequences if not sequence.isEmpty()]
            if not valid:
                action.setShortcuts([])
            elif len(valid) == 1:
                action.setShortcut(valid[0])
            else:
                action.setShortcuts(valid)

        assign_shortcuts(self.action_refresh, "refresh_dataset")
        assign_shortcuts(self.action_save, "save_annotation")
        assign_shortcuts(self.action_export, "export_annotation")
        assign_shortcuts(self.action_undo, "undo")
        assign_shortcuts(self.action_redo, "redo", "redo_alt")
        assign_shortcuts(self.action_prev, "prev_image", "prev_image_alt")
        assign_shortcuts(self.action_next, "next_image", "next_image_alt")
        assign_shortcuts(self.action_draw, "toggle_draw_mode")
        assign_shortcuts(self.action_edit, "toggle_edit_mode")

        def caption(base_text: str, shortcut_hint: str) -> str:
            return f"{base_text}（{shortcut_hint}）" if shortcut_hint else base_text

        self.action_refresh.setToolTip(caption("刷新数据集", self._shortcut_hint("refresh_dataset")))
        self.action_refresh.setStatusTip(caption("刷新数据集", self._shortcut_hint("refresh_dataset")))
        self.action_save.setToolTip(caption("保存当前标注", self._shortcut_hint("save_annotation")))
        self.action_save.setStatusTip(caption("保存当前标注", self._shortcut_hint("save_annotation")))
        self.action_undo.setToolTip(caption("撤销", self._shortcut_hint("undo")))
        self.action_undo.setStatusTip(caption("撤销", self._shortcut_hint("undo")))
        self.action_redo.setToolTip(caption("重做", self._shortcut_hint("redo", "redo_alt")))
        self.action_redo.setStatusTip(caption("重做", self._shortcut_hint("redo", "redo_alt")))
        self.action_export.setToolTip(caption("导出当前标注", self._shortcut_hint("export_annotation")))
        self.action_export.setStatusTip(caption("导出当前标注", self._shortcut_hint("export_annotation")))
        self.action_prev.setToolTip(caption("上一张", self._shortcut_hint("prev_image", "prev_image_alt")))
        self.action_prev.setStatusTip(caption("上一张", self._shortcut_hint("prev_image", "prev_image_alt")))
        self.action_next.setToolTip(caption("下一张", self._shortcut_hint("next_image", "next_image_alt")))
        self.action_next.setStatusTip(caption("下一张", self._shortcut_hint("next_image", "next_image_alt")))
        self.action_draw.setToolTip(caption("新建标注", self._shortcut_hint("toggle_draw_mode")))
        self.action_draw.setStatusTip(caption("新建标注", self._shortcut_hint("toggle_draw_mode")))
        self.action_edit.setToolTip(caption("编辑标注", self._shortcut_hint("toggle_edit_mode")))
        self.action_edit.setStatusTip(caption("编辑标注", self._shortcut_hint("toggle_edit_mode")))

        self._update_shortcut_help_label()
        self._update_shortcut_summary_label()

    def _update_shortcut_help_label(self) -> None:
        if not hasattr(self, "shortcut_help_label"):
            return

        undo_hint = self._shortcut_hint("undo") or "未设置"
        redo_hint = self._shortcut_hint("redo", "redo_alt") or "未设置"
        rename_hint = self._shortcut_hint("rename_box", "rename_box_alt") or "未设置"
        delete_hint = self._shortcut_hint("delete_selection") or "未设置"

        self.shortcut_help_label.setText(
            f"快捷键：{undo_hint} 撤销，{redo_hint} 重做，方向键移动选中框，Ctrl+方向键快速移动，"
            f"{rename_hint} 重命名，{delete_hint} 删除，Esc 退出编辑"
        )

    def _update_shortcut_summary_label(self) -> None:
        if not hasattr(self, "shortcut_summary_label"):
            return

        summary_parts = [
            f"保存 {self._shortcut_hint('save_annotation') or '-'}",
            f"导出 {self._shortcut_hint('export_annotation') or '-'}",
            f"上一张 {self._shortcut_hint('prev_image', 'prev_image_alt') or '-'}",
            f"下一张 {self._shortcut_hint('next_image', 'next_image_alt') or '-'}",
            f"删除 {self._shortcut_hint('delete_selection') or '-'}",
        ]
        self.shortcut_summary_label.setText(" | ".join(summary_parts))

    def open_shortcut_editor(self) -> None:
        dialog = ShortcutEditorDialog(dict(self._shortcut_bindings), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._shortcut_bindings = dialog.shortcut_bindings()
        save_shortcut_bindings(self._shortcut_bindings)
        self._apply_shortcuts_to_actions()
        self._set_status_message("快捷键设置已更新")

    def _load_default_class_name(self) -> str:
        raw = QSettings().value("ui/default_class_name", "object")
        text = str(raw).strip()
        return text or "object"

    def _load_sidebar_tab_visibility(self) -> dict[str, bool]:
        visibility = {tab_id: True for tab_id in self._sidebar_tab_order}
        raw_value = QSettings().value("ui/sidebar_tabs_visible")

        ids: list[str] = []
        if isinstance(raw_value, str) and raw_value.strip():
            ids = [part.strip() for part in raw_value.split(",") if part.strip()]
        elif isinstance(raw_value, (list, tuple)):
            ids = [str(part).strip() for part in raw_value if str(part).strip()]

        if ids:
            id_set = set(ids)
            legacy_ids = {"project", "class", "annotation", "export"}
            for tab_id in self._sidebar_tab_order:
                if tab_id in id_set:
                    visibility[tab_id] = True
                elif tab_id in legacy_ids:
                    visibility[tab_id] = False
                else:
                    visibility[tab_id] = True

        if not any(visibility.values()):
            visibility[self._sidebar_tab_order[0]] = True
        return visibility

    def _save_sidebar_tab_visibility(self) -> None:
        visible_ids = [tab_id for tab_id in self._sidebar_tab_order if self._sidebar_tab_visibility.get(tab_id, False)]
        QSettings().setValue("ui/sidebar_tabs_visible", ",".join(visible_ids))

    def _visible_sidebar_tab_ids(self) -> list[str]:
        return [tab_id for tab_id in self._sidebar_tab_order if self._sidebar_tab_visibility.get(tab_id, False)]

    def _current_sidebar_tab_id(self) -> str | None:
        current_index = self.sidebar_tabs.currentIndex()
        for tab_id, index in self._sidebar_tab_index_by_id.items():
            if index == current_index:
                return tab_id
        return None

    def _sync_sidebar_tab_toggle_actions(self) -> None:
        for tab_id, action in self._sidebar_tab_toggle_actions.items():
            with QSignalBlocker(action):
                action.setChecked(bool(self._sidebar_tab_visibility.get(tab_id, False)))

    def _apply_sidebar_tab_icons(self) -> None:
        tab_bar = self.sidebar_tabs.tabBar()
        assert tab_bar is not None

        for tab_id, icon_spec in self._sidebar_tab_icons.items():
            tab_index = self._sidebar_tab_index_by_id.get(tab_id)
            if tab_index is None:
                continue
            asset_name, *fallback_standard_pixmaps = icon_spec
            icon = self._asset_icon(asset_name, *fallback_standard_pixmaps)
            if not icon.isNull():
                tab_bar.setTabIcon(tab_index, icon)

    def _apply_sidebar_tab_visibility(self, preferred_tab_id: str | None = None) -> None:
        tab_bar = self.sidebar_tabs.tabBar()
        assert tab_bar is not None

        visible_ids = self._visible_sidebar_tab_ids()
        if not visible_ids:
            fallback_id = self._sidebar_tab_order[0]
            self._sidebar_tab_visibility[fallback_id] = True
            visible_ids = [fallback_id]

        for tab_id in self._sidebar_tab_order:
            index = self._sidebar_tab_index_by_id.get(tab_id)
            if index is None:
                continue
            tab_bar.setTabVisible(index, bool(self._sidebar_tab_visibility.get(tab_id, False)))

        current_id = self._current_sidebar_tab_id()
        target_id = preferred_tab_id if preferred_tab_id in visible_ids else current_id
        if target_id not in visible_ids:
            target_id = visible_ids[0]

        target_index = self._sidebar_tab_index_by_id.get(target_id)
        if target_index is not None:
            self.sidebar_tabs.setCurrentIndex(target_index)

        self._sync_sidebar_tab_toggle_actions()

    def _on_sidebar_tab_visibility_toggled(self, tab_id: str, checked: bool) -> None:
        if tab_id not in self._sidebar_tab_visibility:
            return

        currently_visible = self._visible_sidebar_tab_ids()
        if not checked and tab_id in currently_visible and len(currently_visible) <= 1:
            self._info_dialog("保留一个功能页签", "至少需要保留一个可见选项卡。")
            action = self._sidebar_tab_toggle_actions.get(tab_id)
            if action is not None:
                with QSignalBlocker(action):
                    action.setChecked(True)
            return

        self._sidebar_tab_visibility[tab_id] = checked
        self._apply_sidebar_tab_visibility(preferred_tab_id=tab_id if checked else None)
        self._save_sidebar_tab_visibility()
        self._set_status_message("已更新面板选项卡显示")

    def _build_status_bar(self) -> None:
        status = QStatusBar(self)
        self.setStatusBar(status)

        self.status_message_label = QLabel("就绪")
        self.status_zoom_label = QLabel("缩放 100%")
        self.status_cursor_label = QLabel("坐标 --")
        self.status_box_label = QLabel("标注 0")
        self.status_box_size_label = QLabel("尺寸 --")
        self.status_mode_label = QLabel("模式 浏览")
        self.status_autosave_label = QLabel("自动保存 开")

        for label in (
            self.status_message_label,
            self.status_mode_label,
            self.status_zoom_label,
            self.status_cursor_label,
            self.status_box_label,
            self.status_box_size_label,
            self.status_autosave_label,
        ):
            status.addPermanentWidget(label)

    def _update_selected_box_size_status(self) -> None:
        if self.current_document is None:
            self.status_box_size_label.setText("尺寸 --")
            return

        index = self.canvas.selected_index
        if not (0 <= index < len(self.current_document.boxes)):
            self.status_box_size_label.setText("尺寸 --")
            return

        box = self.current_document.boxes[index].ordered()
        self.status_box_size_label.setText(f"尺寸 {box.width()}x{box.height()}")

    def _build_splitter_tab(self, sections: list[tuple[QWidget, int]]) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical, page)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        for index, (section, stretch) in enumerate(sections):
            section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            splitter.addWidget(section)
            splitter.setStretchFactor(index, max(1, int(stretch)))

        def rebalance_sizes() -> None:
            if splitter.count() == 0:
                return

            total_height = splitter.height()
            if total_height <= 0:
                total_height = sum(max(120, section.sizeHint().height()) for section, _ in sections)
            handle_space = max(0, splitter.count() - 1) * splitter.handleWidth()
            available = max(120, total_height - handle_space)

            collapsed_entries: list[tuple[int, int]] = []
            expanded_entries: list[tuple[int, int]] = []
            for index, (section, stretch) in enumerate(sections):
                if isinstance(section, CollapsibleSection) and not section.is_expanded():
                    collapsed_height = max(46, section.collapsed_hint_height())
                    collapsed_entries.append((index, collapsed_height))
                    available -= collapsed_height
                else:
                    expanded_entries.append((index, max(1, int(stretch))))

            minimum_expanded_total = 80 * len(expanded_entries)
            available = max(minimum_expanded_total, available)
            sizes = [0] * splitter.count()

            for index, collapsed_height in collapsed_entries:
                sizes[index] = collapsed_height

            weight_sum = sum(weight for _, weight in expanded_entries)
            assigned = 0
            for pos, (index, weight) in enumerate(expanded_entries):
                if pos == len(expanded_entries) - 1:
                    size = max(80, available - assigned)
                else:
                    size = max(80, int(round(available * weight / max(1, weight_sum))))
                sizes[index] = size
                assigned += size

            splitter.setSizes(sizes)

        for section, _ in sections:
            if isinstance(section, CollapsibleSection):
                section.toggled.connect(lambda _checked, fn=rebalance_sizes: fn())

        QTimer.singleShot(0, rebalance_sizes)
        layout.addWidget(splitter, 1)
        return page

    def _style_section_header(self, section: CollapsibleSection, accent: str) -> None:
        section.toggle_button.setStyleSheet(
            f"""
            QToolButton#sectionHeader {{
                background: #111b28;
                border: 1px solid #2f455c;
                border-left: 3px solid {accent};
                border-radius: 8px;
                padding: 7px 10px;
                color: #eaf3ff;
                font-weight: 700;
                text-align: left;
            }}
            QToolButton#sectionHeader:checked {{
                background: #162637;
                border-color: {accent};
                color: #f5f9ff;
            }}
            QToolButton#sectionHeader:!checked {{
                background: #0f1722;
                border-color: #33495f;
                color: #c8d8e9;
            }}
            QToolButton#sectionHeader:hover {{
                background: #172538;
                border-color: {accent};
                color: #ffffff;
            }}
            """
        )

    def _build_project_tab(self) -> QWidget:
        page = QWidget(self)

        project_section = CollapsibleSection("数据集", expanded=True, parent=page)
        self._style_section_header(project_section, "#4da3ff")
        project_form = QFormLayout()
        project_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        project_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        project_form.setVerticalSpacing(8)

        self.dataset_root_label = QLabel("未加载")
        self.dataset_root_label.setWordWrap(True)
        self.image_count_label = QLabel("0")
        self.label_count_label = QLabel("0")
        self.current_file_label = QLabel("--")
        self.current_file_label.setWordWrap(True)

        project_form.addRow("根目录", self.dataset_root_label)
        project_form.addRow("图片数量", self.image_count_label)
        project_form.addRow("标注文件", self.label_count_label)
        project_form.addRow("当前文件", self.current_file_label)
        project_section.content_layout_ref().addLayout(project_form)

        project_button_row = QHBoxLayout()
        self.btn_open_dataset = QPushButton("选择根目录")
        self.btn_open_dataset.clicked.connect(self.load_dataset)
        self.btn_refresh_dataset = QPushButton("重新扫描")
        self.btn_refresh_dataset.clicked.connect(self.refresh_dataset)
        self.btn_prev_image = QPushButton("上一张")
        self.btn_prev_image.clicked.connect(self.show_previous_image)
        self.btn_next_image = QPushButton("下一张")
        self.btn_next_image.clicked.connect(self.show_next_image)
        for button in (
            self.btn_open_dataset,
            self.btn_refresh_dataset,
            self.btn_prev_image,
            self.btn_next_image,
        ):
            project_button_row.addWidget(button)
            button.hide()

        image_section = CollapsibleSection("图像列表", expanded=True, parent=page)
        self._style_section_header(image_section, "#67d18a")
        filter_row = QHBoxLayout()
        self.image_filter_edit = QLineEdit()
        self.image_filter_edit.setPlaceholderText("过滤文件名...")
        self.image_filter_edit.textChanged.connect(self.on_image_filter_changed)
        self.btn_clear_filter = QPushButton("清空")
        self.btn_clear_filter.clicked.connect(lambda: self.image_filter_edit.setText(""))
        filter_row.addWidget(self.image_filter_edit, 1)
        filter_row.addWidget(self.btn_clear_filter)
        image_section.content_layout_ref().addLayout(filter_row)

        self.image_list_widget = QListWidget()
        self.image_list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.image_list_widget.setToolTip("图像列表：Shift+左键可区间选择，按 Delete 删除所选项目（图片+标注）")
        self.image_list_widget.itemClicked.connect(self.on_image_item_clicked)
        self.image_list_widget.currentItemChanged.connect(self.on_image_current_item_changed)
        image_section.content_layout_ref().addWidget(self.image_list_widget, 1)

        image_action_row = QHBoxLayout()
        self.btn_delete_dataset_items = QPushButton("删除所选项目")
        self.btn_delete_dataset_items.setObjectName("dangerButton")
        self.btn_delete_dataset_items.setToolTip("删除所选图片及对应标注文件（Delete）")
        self.btn_delete_dataset_items.clicked.connect(self.delete_selected_dataset_items)
        image_action_row.addWidget(self.btn_delete_dataset_items)
        image_action_row.addStretch(1)
        image_section.content_layout_ref().addLayout(image_action_row)

        return self._build_splitter_tab([
            (project_section, 2),
            (image_section, 3),
        ])

    def _build_thumbnail_panel(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        self.thumbnail_list_widget = QListWidget()
        self.thumbnail_list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.thumbnail_list_widget.setWrapping(True)
        self.thumbnail_list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbnail_list_widget.setSpacing(6)
        self.thumbnail_list_widget.setMovement(QListWidget.Movement.Static)
        self.thumbnail_list_widget.setFlow(QListWidget.Flow.LeftToRight)
        self.thumbnail_list_widget.setLayoutMode(QListWidget.LayoutMode.Batched)
        self.thumbnail_list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.thumbnail_list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.thumbnail_list_widget.setUniformItemSizes(True)
        self._apply_thumbnail_icon_size(self._thumbnail_icon_width)
        thumbnail_viewport = self.thumbnail_list_widget.viewport()
        assert thumbnail_viewport is not None
        thumbnail_viewport.installEventFilter(self)
        self.thumbnail_list_widget.itemClicked.connect(self.on_thumbnail_item_clicked)
        self.thumbnail_list_widget.currentItemChanged.connect(self.on_thumbnail_current_item_changed)
        layout.addWidget(self.thumbnail_list_widget, 1)
        return page

    def _thumbnail_dimensions(self) -> QSize:
        width = max(self._thumbnail_min_icon_width, min(self._thumbnail_max_icon_width, int(self._thumbnail_icon_width)))
        height = max(56, int(round(width * 88 / 128)))
        return QSize(width, height)

    def _apply_thumbnail_icon_size(self, width: int) -> None:
        self._thumbnail_icon_width = max(self._thumbnail_min_icon_width, min(self._thumbnail_max_icon_width, int(width)))
        icon_size = self._thumbnail_dimensions()
        self.thumbnail_list_widget.setIconSize(icon_size)
        self.thumbnail_list_widget.setGridSize(QSize(icon_size.width() + 34, icon_size.height() + 44))
        self.thumbnail_list_widget.setWordWrap(True)

    def _build_statistics_panel(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        self.dataset_statistics_widget = DatasetStatisticsWidget(page)
        layout.addWidget(self.dataset_statistics_widget, 1)
        return page

    def _build_class_tab(self) -> QWidget:
        page = QWidget(self)

        class_section = CollapsibleSection("类别管理", expanded=True, parent=page)
        self._style_section_header(class_section, "#7aa8ff")
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("输入新类别名称")
        self.class_input.returnPressed.connect(self.create_class)
        class_buttons = QHBoxLayout()
        self.btn_add_class = QPushButton("新建类别")
        self.btn_add_class.clicked.connect(self.create_class)
        self.btn_rename_class = QPushButton("重命名所选")
        self.btn_rename_class.clicked.connect(self.rename_selected_class)
        self.btn_delete_class = QPushButton("删除所选")
        self.btn_delete_class.setObjectName("dangerButton")
        self.btn_delete_class.clicked.connect(self.delete_selected_class)
        self.btn_sync_yaml = QPushButton("同步 YAML")
        self.btn_sync_yaml.clicked.connect(self.sync_yaml_file)
        for button in (self.btn_add_class, self.btn_rename_class, self.btn_delete_class, self.btn_sync_yaml):
            class_buttons.addWidget(button)
        class_section.content_layout_ref().addWidget(self.class_input)
        class_section.content_layout_ref().addLayout(class_buttons)

        remap_row = QHBoxLayout()
        remap_label = QLabel("目标索引")
        remap_label.setObjectName("mutedLabel")
        self.class_target_id_spin = QSpinBox()
        self.class_target_id_spin.setRange(0, 999999)
        self.class_target_id_spin.setValue(0)
        self.btn_remap_class_id = QPushButton("修改所选ID")
        self.btn_remap_class_id.clicked.connect(self.remap_selected_class_id)
        remap_row.addWidget(remap_label)
        remap_row.addWidget(self.class_target_id_spin)
        remap_row.addWidget(self.btn_remap_class_id)
        class_section.content_layout_ref().addLayout(remap_row)

        remap_option_row = QHBoxLayout()
        self.class_swap_checkbox = QCheckBox("目标ID冲突时交换")
        self.class_swap_checkbox.setChecked(True)
        self.class_rewrite_labels_checkbox = QCheckBox("同步改写数据集TXT标签")
        self.class_rewrite_labels_checkbox.setChecked(True)
        remap_option_row.addWidget(self.class_swap_checkbox)
        remap_option_row.addWidget(self.class_rewrite_labels_checkbox)
        remap_option_row.addStretch(1)
        class_section.content_layout_ref().addLayout(remap_option_row)

        self.class_list_widget = QListWidget()
        self.class_list_widget.itemSelectionChanged.connect(self.on_class_selection_changed)
        class_section.content_layout_ref().addWidget(self.class_list_widget, 1)

        yaml_section = CollapsibleSection("YAML 信息", expanded=False, parent=page)
        self._style_section_header(yaml_section, "#f3c969")
        self.yaml_path_label = QLabel("未检测到 YAML")
        self.yaml_path_label.setWordWrap(True)
        self.class_count_label = QLabel("0")
        yaml_form = QFormLayout()
        yaml_form.addRow("YAML 路径", self.yaml_path_label)
        yaml_form.addRow("类别数量", self.class_count_label)
        yaml_section.content_layout_ref().addLayout(yaml_form)

        self.yaml_content_view = QPlainTextEdit()
        self.yaml_content_view.setReadOnly(True)
        self.yaml_content_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.yaml_content_view.setMinimumHeight(150)
        yaml_section.content_layout_ref().addWidget(self.yaml_content_view, 1)

        return self._build_splitter_tab([
            (class_section, 3),
            (yaml_section, 1),
        ])

    def _build_annotation_tab(self) -> QWidget:
        page = QWidget(self)

        draw_section = CollapsibleSection("绘制设置", expanded=True, parent=page)
        self._style_section_header(draw_section, "#91d58b")
        draw_form = QFormLayout()
        draw_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        draw_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        self.draw_shape_combo = QComboBox()
        self.draw_shape_combo.addItem("矩形", "rectangle")
        self.draw_shape_combo.addItem("多边形", "polygon")
        draw_index = self.draw_shape_combo.findData(self.draw_shape)
        self.draw_shape_combo.setCurrentIndex(max(0, draw_index))
        self.draw_shape_combo.currentIndexChanged.connect(self.on_draw_shape_changed)
        draw_form.addRow("形状", self.draw_shape_combo)
        draw_section.content_layout_ref().addLayout(draw_form)

        selection_section = CollapsibleSection("选中标注", expanded=True, parent=page)
        self._style_section_header(selection_section, "#8ecbff")
        self.selected_info_label = QLabel("未选中")
        self.selected_info_label.setWordWrap(True)
        selection_section.content_layout_ref().addWidget(self.selected_info_label)

        apply_row = QHBoxLayout()
        self.box_class_combo = QComboBox()
        self.box_class_combo.setEditable(True)
        self.box_class_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.box_class_combo.setPlaceholderText("选择或输入类别")
        self.box_class_apply_button = QPushButton("应用类别")
        self.box_class_apply_button.clicked.connect(self.apply_class_to_selected_box)
        self.box_delete_button = QPushButton("删除选中")
        self.box_delete_button.setObjectName("dangerButton")
        self.box_delete_button.clicked.connect(self.delete_selected_box)
        apply_row.addWidget(self.box_class_combo, 1)
        apply_row.addWidget(self.box_class_apply_button)
        apply_row.addWidget(self.box_delete_button)
        selection_section.content_layout_ref().addLayout(apply_row)

        self.shortcut_help_label = QLabel(
            "快捷键：Ctrl+Z 撤销，Ctrl+Y / Ctrl+Shift+Z 重做，方向键移动选中框，Ctrl+方向键快速移动，R/F2 重命名，Del 删除，Esc 退出编辑"
        )
        self.shortcut_help_label.setObjectName("mutedLabel")
        self.shortcut_help_label.setWordWrap(True)
        selection_section.content_layout_ref().addWidget(self.shortcut_help_label)

        box_section = CollapsibleSection("当前标注列表", expanded=True, parent=page)
        self._style_section_header(box_section, "#63d0c9")
        self.box_list_widget = QListWidget()
        self.box_list_widget.itemSelectionChanged.connect(self.on_box_selection_changed)
        box_section.content_layout_ref().addWidget(self.box_list_widget, 1)

        return self._build_splitter_tab([
            (draw_section, 1),
            (selection_section, 2),
            (box_section, 3),
        ])

    def _build_export_tab(self) -> QWidget:
        page = QWidget(self)

        export_section = CollapsibleSection("目标设置", expanded=True, parent=page)
        self._style_section_header(export_section, "#9ac6ff")
        export_form = QFormLayout()
        export_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        export_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        export_form.setVerticalSpacing(8)

        self.export_format_combo = QComboBox()
        for key in ("json", "xml", "txt", "mask_png"):
            self.export_format_combo.addItem(FORMAT_LABELS[key], key)
        self.export_format_combo.setCurrentIndex(2)
        export_form.addRow("目标格式", self.export_format_combo)

        resize_row = QHBoxLayout()
        self.export_resize_width_spin = QSpinBox()
        self.export_resize_width_spin.setRange(64, 8192)
        self.export_resize_width_spin.setValue(640)
        self.export_resize_width_spin.setPrefix("W ")
        self.export_resize_height_spin = QSpinBox()
        self.export_resize_height_spin.setRange(64, 8192)
        self.export_resize_height_spin.setValue(640)
        self.export_resize_height_spin.setPrefix("H ")
        self.export_resize_letterbox_checkbox = QCheckBox("保持比例 Letterbox")
        self.export_resize_letterbox_checkbox.setChecked(True)
        self.export_resize_button = QPushButton("尺寸转换导出")
        self.export_resize_button.setObjectName("primaryButton")
        self.export_resize_button.clicked.connect(self.batch_resize_export_dataset)
        resize_row.addWidget(self.export_resize_width_spin)
        resize_row.addWidget(self.export_resize_height_spin)
        resize_row.addWidget(self.export_resize_letterbox_checkbox)
        resize_row.addWidget(self.export_resize_button)
        resize_row.addStretch(1)
        export_form.addRow("目标尺寸", resize_row)
        export_section.content_layout_ref().addLayout(export_form)

        export_buttons = QHBoxLayout()
        self.export_current_button = QPushButton("导出当前")
        self.export_current_button.setObjectName("primaryButton")
        self.export_current_button.clicked.connect(self.export_current_annotation)
        self.export_batch_button = QPushButton("批量转换")
        self.export_batch_button.clicked.connect(self.batch_convert_dataset)
        export_buttons.addWidget(self.export_current_button)
        export_buttons.addWidget(self.export_batch_button)
        export_section.content_layout_ref().addLayout(export_buttons)

        self.export_path_hint_label = QLabel("优先输出到标签目录；Mask PNG 将输出单通道语义索引图。")
        self.export_path_hint_label.setObjectName("mutedLabel")
        self.export_path_hint_label.setWordWrap(True)
        export_section.content_layout_ref().addWidget(self.export_path_hint_label)

        self.export_resize_hint_label = QLabel(
            "默认使用保持比例 Letterbox（补边）到目标尺寸，如 640x640，并同步缩放标注框。"
        )
        self.export_resize_hint_label.setObjectName("mutedLabel")
        self.export_resize_hint_label.setWordWrap(True)
        export_section.content_layout_ref().addWidget(self.export_resize_hint_label)

        dataset_tools_section = CollapsibleSection("数据集工具", expanded=True, parent=page)
        self._style_section_header(dataset_tools_section, "#7ec7a0")

        merge_row = QHBoxLayout()
        self.export_merge_button = QPushButton("合并数据集")
        self.export_merge_button.clicked.connect(self.merge_multiple_datasets)
        self.export_splitter_button = QPushButton("划分数据集")
        self.export_splitter_button.clicked.connect(self.open_dataset_splitter_tool)
        self.export_extract_button = QPushButton("抽取数据集(Mini)")
        self.export_extract_button.clicked.connect(self.extract_dataset_mini)
        merge_row.addWidget(self.export_merge_button)
        merge_row.addWidget(self.export_splitter_button)
        merge_row.addWidget(self.export_extract_button)
        dataset_tools_section.content_layout_ref().addLayout(merge_row)

        extract_form = QFormLayout()
        extract_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        extract_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        extract_form.setVerticalSpacing(8)
        self.export_extract_ratio_spin = QSpinBox()
        self.export_extract_ratio_spin.setRange(1, 100)
        self.export_extract_ratio_spin.setValue(20)
        self.export_extract_ratio_spin.setSuffix(" %")
        self.export_extract_seed_spin = QSpinBox()
        self.export_extract_seed_spin.setRange(0, 999999)
        self.export_extract_seed_spin.setValue(42)
        extract_form.addRow("抽取比例", self.export_extract_ratio_spin)
        extract_form.addRow("随机种子", self.export_extract_seed_spin)
        dataset_tools_section.content_layout_ref().addLayout(extract_form)

        dataset_tools_hint = QLabel(
            "抽取功能按 train/val/test 分别采样，并尽量保持各类别比例；抽取时可按 ID / 名称勾选需要保留的类别；划分功能会打开 tools/dataset_splitter.html。"
        )
        dataset_tools_hint.setObjectName("mutedLabel")
        dataset_tools_hint.setWordWrap(True)
        dataset_tools_section.content_layout_ref().addWidget(dataset_tools_hint)

        return self._build_splitter_tab([
            (export_section, 2),
            (dataset_tools_section, 3),
        ])

    def _build_settings_tab(self) -> QWidget:
        page = QWidget(self)

        autosave_section = CollapsibleSection("自动保存", expanded=True, parent=page)
        self._style_section_header(autosave_section, "#f59fb3")
        self.autosave_checkbox = QCheckBox("启用自动保存")
        self.autosave_checkbox.setChecked(self._autosave_enabled_pref)
        self.autosave_checkbox.toggled.connect(self.toggle_autosave)
        self.autosave_interval_spin = QSpinBox()
        self.autosave_interval_spin.setRange(1, 120)
        self.autosave_interval_spin.setSuffix(" 秒")
        self.autosave_interval_spin.setValue(self._autosave_interval_seconds_pref)
        self.autosave_interval_spin.valueChanged.connect(self.on_autosave_interval_changed)
        autosave_form = QFormLayout()
        autosave_form.addRow(self.autosave_checkbox)
        autosave_form.addRow("保存间隔", self.autosave_interval_spin)
        autosave_hint = QLabel("提示：工具栏上的 AUTO 图标可快速开关自动保存。")
        autosave_hint.setObjectName("mutedLabel")
        autosave_hint.setWordWrap(True)
        autosave_section.content_layout_ref().addLayout(autosave_form)
        autosave_section.content_layout_ref().addWidget(autosave_hint)

        defaults_section = CollapsibleSection("默认设置", expanded=True, parent=page)
        self._style_section_header(defaults_section, "#8ecbff")
        defaults_form = QFormLayout()
        self.default_class_input = QLineEdit(self._pending_class_text)
        self.default_class_input.setPlaceholderText("新建标注默认类别，例如 object")
        self.default_class_input.editingFinished.connect(self.apply_default_class_setting)
        defaults_form.addRow("默认类别", self.default_class_input)
        defaults_section.content_layout_ref().addLayout(defaults_form)

        defaults_button_row = QHBoxLayout()
        self.btn_apply_defaults = QPushButton("应用默认设置")
        self.btn_apply_defaults.clicked.connect(self.apply_default_class_setting)
        self.btn_reset_defaults = QPushButton("恢复默认")
        self.btn_reset_defaults.clicked.connect(self.reset_general_settings)
        defaults_button_row.addWidget(self.btn_apply_defaults)
        defaults_button_row.addWidget(self.btn_reset_defaults)
        defaults_button_row.addStretch(1)
        defaults_section.content_layout_ref().addLayout(defaults_button_row)

        defaults_hint = QLabel("可在此集中调整常用参数与默认值。")
        defaults_hint.setObjectName("mutedLabel")
        defaults_hint.setWordWrap(True)
        defaults_section.content_layout_ref().addWidget(defaults_hint)

        visual_section = CollapsibleSection("可视化设置", expanded=True, parent=page)
        self._style_section_header(visual_section, "#66c0ff")

        visual_form = QFormLayout()
        self.label_bg_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.label_bg_alpha_slider.setRange(0, 255)
        self.label_bg_alpha_slider.setSingleStep(5)
        self.label_bg_alpha_slider.setPageStep(20)
        self.label_bg_alpha_slider.setValue(self._label_bg_alpha_pref)

        self.label_bg_alpha_spin = QSpinBox()
        self.label_bg_alpha_spin.setRange(0, 255)
        self.label_bg_alpha_spin.setSuffix(" /255")
        self.label_bg_alpha_spin.setValue(self._label_bg_alpha_pref)

        alpha_row = QHBoxLayout()
        alpha_row.addWidget(self.label_bg_alpha_slider, 1)
        alpha_row.addWidget(self.label_bg_alpha_spin)
        visual_form.addRow("标签背景透明度", alpha_row)

        self.label_show_name_checkbox = QCheckBox("显示标签名称")
        self.label_show_name_checkbox.setChecked(self._label_show_name_pref)
        visual_form.addRow(self.label_show_name_checkbox)

        self.label_show_id_checkbox = QCheckBox("显示标签 ID")
        self.label_show_id_checkbox.setChecked(self._label_show_id_pref)
        visual_form.addRow(self.label_show_id_checkbox)

        self.label_bg_alpha_slider.valueChanged.connect(self.on_label_bg_alpha_slider_changed)
        self.label_bg_alpha_spin.valueChanged.connect(self.on_label_bg_alpha_spin_changed)
        self.label_show_name_checkbox.toggled.connect(self.on_label_display_options_changed)
        self.label_show_id_checkbox.toggled.connect(self.on_label_display_options_changed)

        visual_hint = QLabel("提示：将透明度调低可减少标签背景遮挡；关闭名称/ID可隐藏对应文本。")
        visual_hint.setObjectName("mutedLabel")
        visual_hint.setWordWrap(True)

        visual_section.content_layout_ref().addLayout(visual_form)
        visual_section.content_layout_ref().addWidget(visual_hint)

        shortcut_section = CollapsibleSection("快捷键设置", expanded=False, parent=page)
        self._style_section_header(shortcut_section, "#8fd6d2")

        shortcut_button_row = QHBoxLayout()
        self.btn_open_shortcut_editor = QPushButton("编辑快捷键...")
        self.btn_open_shortcut_editor.clicked.connect(self.open_shortcut_editor)
        shortcut_button_row.addWidget(self.btn_open_shortcut_editor)
        shortcut_button_row.addStretch(1)

        self.shortcut_summary_label = QLabel("按下“编辑快捷键...”查看或修改当前按键映射。")
        self.shortcut_summary_label.setObjectName("mutedLabel")
        self.shortcut_summary_label.setWordWrap(True)

        shortcut_hint = QLabel("建议避免为多个功能设置同一快捷键。")
        shortcut_hint.setObjectName("mutedLabel")
        shortcut_hint.setWordWrap(True)

        shortcut_section.content_layout_ref().addLayout(shortcut_button_row)
        shortcut_section.content_layout_ref().addWidget(self.shortcut_summary_label)
        shortcut_section.content_layout_ref().addWidget(shortcut_hint)

        return self._build_splitter_tab([
            (autosave_section, 2),
            (defaults_section, 2),
            (visual_section, 2),
            (shortcut_section, 2),
        ])

    def _wrap_scrollable(self, widget: QWidget) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll.setStyleSheet("background: #161d26; border: none;")
        widget.setStyleSheet("background: #161d26;")
        return scroll

    def _bind_signals(self) -> None:
        self.autosave.set_enabled(self._autosave_enabled_pref)
        self.action_autosave.setChecked(self._autosave_enabled_pref)
        self.canvas.editOperationStarted.connect(self.on_canvas_edit_operation_started)
        self.canvas.set_draw_shape(self.draw_shape)
        self.on_autosave_interval_changed(self.autosave_interval_spin.value())
        self._apply_canvas_label_visual_settings()
        self.status_autosave_label.setText("自动保存 开" if self.autosave.enabled else "自动保存 关")
        self._sync_history_actions()

    def _apply_canvas_label_visual_settings(self) -> None:
        self.canvas.set_label_visual_options(
            background_alpha=self._label_bg_alpha_pref,
            show_name=self._label_show_name_pref,
            show_id=self._label_show_id_pref,
        )

    def on_label_bg_alpha_slider_changed(self, value: int) -> None:
        alpha = max(0, min(255, int(value)))
        if hasattr(self, "label_bg_alpha_spin") and self.label_bg_alpha_spin.value() != alpha:
            with QSignalBlocker(self.label_bg_alpha_spin):
                self.label_bg_alpha_spin.setValue(alpha)

        if alpha == self._label_bg_alpha_pref:
            return

        self._label_bg_alpha_pref = alpha
        QSettings().setValue("ui/label_bg_alpha", alpha)
        self._apply_canvas_label_visual_settings()
        self._set_status_message(f"标签背景透明度已调整为 {alpha}/255", timeout_ms=1400)

    def on_label_bg_alpha_spin_changed(self, value: int) -> None:
        alpha = max(0, min(255, int(value)))
        if hasattr(self, "label_bg_alpha_slider") and self.label_bg_alpha_slider.value() != alpha:
            with QSignalBlocker(self.label_bg_alpha_slider):
                self.label_bg_alpha_slider.setValue(alpha)
        self.on_label_bg_alpha_slider_changed(alpha)

    def on_label_display_options_changed(self) -> None:
        show_name = bool(self.label_show_name_checkbox.isChecked())
        show_id = bool(self.label_show_id_checkbox.isChecked())

        changed = False
        if show_name != self._label_show_name_pref:
            self._label_show_name_pref = show_name
            QSettings().setValue("ui/label_show_name", show_name)
            changed = True

        if show_id != self._label_show_id_pref:
            self._label_show_id_pref = show_id
            QSettings().setValue("ui/label_show_id", show_id)
            changed = True

        if changed:
            self._apply_canvas_label_visual_settings()
            parts: list[str] = []
            if show_id:
                parts.append("ID")
            if show_name:
                parts.append("名称")
            text = "标签文本显示：" + (" + ".join(parts) if parts else "关闭")
            self._set_status_message(text, timeout_ms=1400)

    def on_draw_shape_changed(self) -> None:
        if not hasattr(self, "draw_shape_combo"):
            return
        shape = str(self.draw_shape_combo.currentData() or "rectangle")
        self.draw_shape = "polygon" if shape == "polygon" else "rectangle"
        self.canvas.set_draw_shape(self.draw_shape)
        QSettings().setValue("ui/draw_shape", self.draw_shape)
        self._update_mode_widgets()
        self._update_status_bar_values()
        label = "多边形" if self.draw_shape == "polygon" else "矩形"
        self._set_status_message(f"绘制形状：{label}", timeout_ms=1400)

    def eventFilter(self, a0, a1) -> bool:
        if a0 == self.scroll_area.viewport() and isinstance(a1, QWheelEvent):
            if a1.type() == QEvent.Type.Wheel and a1.modifiers() & Qt.KeyboardModifier.ControlModifier:
                step = ZOOM_STEP if a1.angleDelta().y() > 0 else 1.0 / ZOOM_STEP
                self.set_zoom_scale(self.current_zoom_scale * step)
                return True

        if (
            hasattr(self, "thumbnail_list_widget")
            and a0 == self.thumbnail_list_widget.viewport()
            and isinstance(a1, QWheelEvent)
            and a1.type() == QEvent.Type.Wheel
            and a1.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            direction = 1 if a1.angleDelta().y() > 0 else -1
            next_width = self._thumbnail_icon_width + direction * self._thumbnail_zoom_step
            next_width = max(self._thumbnail_min_icon_width, min(self._thumbnail_max_icon_width, next_width))
            if next_width != self._thumbnail_icon_width:
                self._apply_thumbnail_icon_size(next_width)
                self._thumbnail_cache.clear()
                self._refresh_thumbnail_list()
                self._set_status_message(f"缩略图大小 {self._thumbnail_dimensions().width()}px", timeout_ms=1200)
            return True
        return super().eventFilter(a0, a1)

    def keyPressEvent(self, a0: QKeyEvent | None) -> None:
        if a0 is None:
            return
        if self.handle_key_press(a0):
            return
        super().keyPressEvent(a0)

    def handle_key_press(self, event: QKeyEvent) -> bool:
        if self._event_matches_shortcut(event, "undo"):
            self.undo_last_action()
            return True

        if self._event_matches_shortcut(event, "redo") or self._event_matches_shortcut(event, "redo_alt"):
            self.redo_last_action()
            return True

        if self._event_matches_shortcut(event, "save_annotation"):
            self.save_current_annotation()
            return True

        if self._event_matches_shortcut(event, "export_annotation"):
            self.export_current_annotation()
            return True

        if self._event_matches_shortcut(event, "prev_image") or self._event_matches_shortcut(event, "prev_image_alt"):
            self.show_previous_image()
            return True

        if self._event_matches_shortcut(event, "next_image") or self._event_matches_shortcut(event, "next_image_alt"):
            self.show_next_image()
            return True

        if self._event_matches_shortcut(event, "toggle_draw_mode"):
            self.set_draw_mode(not self.draw_mode)
            return True

        if self._event_matches_shortcut(event, "toggle_edit_mode"):
            self.set_edit_mode(not self.edit_mode)
            return True

        if event.key() == Qt.Key.Key_Escape:
            if self.draw_mode or self.edit_mode or self.canvas.selected_index != -1:
                self.set_draw_mode(False)
                self.set_edit_mode(False)
                self.canvas.clear_interaction()
                self.canvas.set_selected_index(-1)
                self.history.clear_pending()
                self._set_selected_index(-1)
                self._set_status_message("已退出编辑")
                return True
            return False
        if self._keyboard_editing_enabled():
            if event.key() in (
                Qt.Key.Key_Left,
                Qt.Key.Key_Right,
                Qt.Key.Key_Up,
                Qt.Key.Key_Down,
            ):
                step = 10 if event.modifiers() & Qt.KeyboardModifier.ControlModifier else 1
                if event.key() == Qt.Key.Key_Left:
                    return self._nudge_selected_box(-step, 0)
                if event.key() == Qt.Key.Key_Right:
                    return self._nudge_selected_box(step, 0)
                if event.key() == Qt.Key.Key_Up:
                    return self._nudge_selected_box(0, -step)
                if event.key() == Qt.Key.Key_Down:
                    return self._nudge_selected_box(0, step)

        if self._event_matches_shortcut(event, "rename_box") or self._event_matches_shortcut(event, "rename_box_alt"):
            self.rename_selected_box_class()
            return True

        if self._event_matches_shortcut(event, "delete_selection"):
            focused_widget = self.focusWidget()
            if (
                focused_widget is self.image_list_widget
                or focused_widget is self.thumbnail_list_widget
                or (
                    focused_widget is not None
                    and (
                        self.image_list_widget.isAncestorOf(focused_widget)
                        or self.thumbnail_list_widget.isAncestorOf(focused_widget)
                    )
                )
            ):
                self.delete_selected_dataset_items()
            else:
                self.delete_selected_box()
            return True

        if self._event_matches_shortcut(event, "refresh_dataset"):
            self.refresh_dataset()
            return True

        return False

    def load_dataset(self) -> None:
        root = QFileDialog.getExistingDirectory(self, "选择根目录")
        if not root:
            return
        if not self._prepare_for_context_change():
            return
        self._start_dataset_loading(Path(root), auto_open_first=True, preferred_path=None)

    def refresh_dataset(self) -> None:
        if self.dataset_service.root_dir is None:
            return
        if not self._prepare_for_context_change():
            return
        current_image = self.current_document.image_path if self.current_document else None
        self._start_dataset_loading(self.dataset_service.root_dir, auto_open_first=False, preferred_path=current_image)

    def _load_dataset_from_root(self, root: Path, auto_open_first: bool) -> None:
        self._start_dataset_loading(root, auto_open_first=auto_open_first, preferred_path=None)

    def _start_dataset_loading(
        self,
        root: Path,
        auto_open_first: bool,
        preferred_path: Path | None,
    ) -> None:
        if self._dataset_loading:
            self._set_status_message("数据集正在加载，请稍候...")
            return

        self._cancel_dataset_statistics_loading()
        self._cancel_thumbnail_loading()
        self._dataset_job_generation += 1
        job_id = self._dataset_job_generation
        self._pending_dataset_root = root
        self._pending_dataset_preferred_path = preferred_path
        self._dataset_loading = True
        self._cancel_dataset_statistics_loading()
        self._cancel_thumbnail_loading()
        self._dataset_statistics_loading = True
        self._set_loaded_state(False)
        self._set_statistics_loading_state("等待统计数据...")
        self._set_status_message(f"正在扫描数据集: {root}")

        worker = DatasetScanWorker(root, dict(self.class_manager.id_to_name))
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progressChanged.connect(lambda current, total, job=job_id: self._on_dataset_scan_progress(job, current, total))
        worker.finished.connect(lambda result, job=job_id: self._on_dataset_scan_finished(job, result))
        worker.failed.connect(lambda message, job=job_id: self._on_dataset_scan_failed(job, message))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._dataset_scan_worker = worker
        self._dataset_scan_thread = thread
        thread.start()

    def open_image_at_index(self, index: int, prompt_if_dirty: bool = True) -> None:
        if not self.dataset_service.image_paths:
            return
        if index < 0 or index >= len(self.dataset_service.image_paths):
            return
        if prompt_if_dirty and not self._prepare_for_context_change():
            return

        image_path = self.dataset_service.image_paths[index]
        qimage = self._load_image_as_qimage(image_path)
        if qimage is None:
            self._warning_dialog(
                "打开图片失败",
                "无法加载该图片文件。",
                informative_text=f"文件：{self.dataset_service.display_name(image_path)}",
                details=str(image_path),
            )
            return

        label_path = self.dataset_service.find_label_for_image(image_path)
        image_size = (qimage.width(), qimage.height())
        boxes = self.annotation_io.load_annotation(label_path, image_size, self.class_manager)
        source_format = label_path.suffix.lower().lstrip(".") if label_path else self.export_format_combo.currentData() or "txt"
        self.current_document = AnnotationDocument(image_path, label_path, image_size, boxes, source_format)
        self.current_image = qimage
        self.current_image_index = index
        self.current_zoom_scale = self._clamp_zoom(self.current_zoom_scale)
        self.canvas.set_document(qimage, self.current_document.boxes, self.current_zoom_scale)
        self.canvas.set_draw_shape(self.draw_shape)
        self.canvas.set_modes(edit_mode=self.edit_mode, draw_mode=self.draw_mode)
        self.canvas.set_selected_index(-1)
        self.autosave.clear()
        self.history.clear()
        self._store_current_document_saved_statistics(self.current_document.boxes)
        self._refresh_all_views()
        self._set_loaded_state(True)
        self._set_status_message(f"已加载 {self.dataset_service.display_name(image_path)}")

    def _load_image_as_qimage(self, image_path: Path) -> QImage | None:
        try:
            data = np.fromfile(str(image_path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                return None
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        except Exception:
            return None

    def _load_image_size(self, image_path: Path) -> tuple[int, int] | None:
        try:
            data = np.fromfile(str(image_path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                return None
            height, width = image.shape[:2]
            return int(width), int(height)
        except Exception:
            return None

    def _save_yolo_txt_with_id_mapping(
        self,
        label_path: Path,
        boxes: list[Box],
        image_size: tuple[int, int],
        source_manager: ClassManager,
        source_class_id_to_target_id: dict[int, int],
    ) -> None:
        width, height = image_size
        lines: list[str] = []
        for box in boxes:
            source_class_id = source_manager.get_id(box.class_name)
            if source_class_id is None:
                try:
                    source_class_id = source_manager.ensure_name(box.class_name)
                except Exception:
                    continue
            class_id = source_class_id_to_target_id.get(int(source_class_id))
            if class_id is None:
                continue
            if box.is_polygon:
                coords: list[str] = []
                for x, y in box.polygon_points():
                    coords.append(f"{max(0.0, min(1.0, x / max(1.0, float(width)))):.6f}")
                    coords.append(f"{max(0.0, min(1.0, y / max(1.0, float(height)))):.6f}")
                if len(coords) >= 6:
                    lines.append(f"{int(class_id)} {' '.join(coords)}")
                continue
            cx, cy, bw, bh = box.normalized(width, height)
            lines.append(
                f"{int(class_id)} "
                f"{max(0.0, min(1.0, cx)):.6f} {max(0.0, min(1.0, cy)):.6f} "
                f"{max(0.0, min(1.0, bw)):.6f} {max(0.0, min(1.0, bh)):.6f}"
            )
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines), encoding="utf-8")

    def _extract_merge_split_name(self, relative_path: Path) -> str | None:
        for part in relative_path.parts:
            lowered = part.strip().lower()
            if lowered in {"val", "valid"}:
                return "val"
            if lowered in {"train", "test"}:
                return lowered
        return None

    def _normalize_merge_template(self, relative_parent: Path) -> str:
        tokens: list[str] = []
        for part in relative_parent.parts:
            lowered = part.strip().lower()
            if lowered in {"train", "val", "valid", "test"}:
                tokens.append("{split}")
            else:
                tokens.append(lowered)
        return "/".join(tokens) if tokens else "."

    def _analyze_merge_dataset_layout(
        self,
        root: Path,
        service: DatasetService,
    ) -> tuple[MergeDatasetLayout | None, str | None]:
        image_templates: set[str] = set()
        label_templates: set[str] = set()
        split_names: set[str] = set()
        image_split_map: dict[Path, str] = {}
        has_split_images = False
        has_unsplit_images = False

        for image_path in service.image_paths:
            try:
                image_relative = image_path.relative_to(root)
            except Exception:
                image_relative = Path(image_path.name)

            split_name = self._extract_merge_split_name(image_relative)
            if split_name is None:
                has_unsplit_images = True
                image_split_map[image_path] = "all"
            else:
                has_split_images = True
                split_names.add(split_name)
                image_split_map[image_path] = split_name

            image_templates.add(self._normalize_merge_template(image_relative.parent))

            label_path = service.find_label_for_image(image_path)
            if label_path is None:
                continue

            try:
                label_relative = label_path.relative_to(root)
            except Exception:
                label_relative = Path(label_path.name)
            label_templates.add(self._normalize_merge_template(label_relative.parent))

        if has_split_images and has_unsplit_images:
            return None, "同一数据集中同时存在未划分与 train/val/test 划分目录，无法安全合并。"

        if has_split_images:
            required = {"train", "val", "test"}
            if split_names != required:
                missing = sorted(required - split_names)
                extra = sorted(split_names - required)
                detail_parts: list[str] = []
                if missing:
                    detail_parts.append(f"缺少分级: {', '.join(missing)}")
                if extra:
                    detail_parts.append(f"额外分级: {', '.join(extra)}")
                detail = "；".join(detail_parts) if detail_parts else "目录分级与 train/val/test 不一致"
                return None, f"已划分数据集必须包含 train/val/test。{detail}。"

        mode = "split" if has_split_images else "unsplit"
        layout = MergeDatasetLayout(
            mode=mode,
            image_templates=tuple(sorted(image_templates)),
            label_templates=tuple(sorted(label_templates)),
            split_names=tuple(sorted(split_names)),
            image_split_map=image_split_map,
        )
        return layout, None

    def _describe_merge_layout(self, layout: MergeDatasetLayout) -> str:
        image_templates = ", ".join(layout.image_templates) if layout.image_templates else "(无)"
        label_templates = ", ".join(layout.label_templates) if layout.label_templates else "(无标签)"
        if layout.mode == "split":
            split_names = ", ".join(layout.split_names) if layout.split_names else "(未识别)"
            return (
                f"类型: 已划分 ({split_names})\n"
                f"图片目录模板: {image_templates}\n"
                f"标签目录模板: {label_templates}"
            )
        return (
            "类型: 未划分\n"
            f"图片目录模板: {image_templates}\n"
            f"标签目录模板: {label_templates}"
        )

    def _build_merge_class_name(self, dataset_tag: str, source_name: str, global_manager: ClassManager) -> str:
        base_name = str(source_name).strip() or "object"
        candidate = f"{dataset_tag}/{base_name}"
        if candidate not in global_manager.name_to_id:
            return candidate

        suffix = 2
        while f"{candidate}_{suffix}" in global_manager.name_to_id:
            suffix += 1
        return f"{candidate}_{suffix}"

    def _ensure_merge_class_id_mapping(
        self,
        dataset_tag: str,
        source_manager: ClassManager,
        source_class_id_to_target_id: dict[int, int],
        global_manager: ClassManager,
    ) -> None:
        for source_id, source_name in source_manager.sorted_items():
            source_id = int(source_id)
            if source_id in source_class_id_to_target_id:
                continue
            target_id = global_manager.next_available_id()
            target_name = self._build_merge_class_name(dataset_tag, source_name, global_manager)
            global_manager.ensure_id(target_id, target_name)
            source_class_id_to_target_id[source_id] = target_id

    def _write_merge_data_yaml(
        self,
        output_root: Path,
        class_mapping: dict[int, str],
        split_mode: bool,
        *,
        train_path: str | None = None,
        val_path: str | None = None,
        test_path: str | None = None,
    ) -> None:
        lines: list[str] = []
        if split_mode:
            lines.extend([
                f"train: {train_path or 'images/train'}",
                f"val: {val_path or 'images/val'}",
                f"test: {test_path or 'images/test'}",
            ])
        else:
            lines.extend([
                f"train: {train_path or 'images'}",
                f"val: {val_path or 'images'}",
            ])

        lines.append("")
        lines.append(f"nc: {len(class_mapping)}")
        lines.append("names:")
        if class_mapping:
            for class_id, class_name in sorted(class_mapping.items(), key=lambda item: int(item[0])):
                escaped = str(class_name).replace("'", "''")
                lines.append(f"  {int(class_id)}: '{escaped}'")
        else:
            lines.append("  {}")

        (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _resolve_dataset_splitter_html(self) -> Path | None:
        module_root = Path(__file__).resolve().parents[2]
        candidates = [
            module_root / "tools" / "dataset_splitter.html",
            module_root / "dataset_splitter.html",
            module_root.parent.parent / "dataset_splitter.html",
            Path.cwd() / "tools" / "dataset_splitter.html",
            Path.cwd() / "dataset_splitter.html",
        ]
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved.exists() and resolved.is_file():
                return resolved
        return None

    def open_dataset_splitter_tool(self) -> None:
        html_path = self._resolve_dataset_splitter_html()
        if html_path is None:
            self._warning_dialog(
                "未找到划分工具",
                "未检测到 dataset_splitter.html。",
                informative_text="请确认文件位于 tools/dataset_splitter.html。",
            )
            return

        try:
            opened = webbrowser.open(html_path.as_uri())
        except Exception as exc:
            self._error_dialog(
                "打开失败",
                "无法启动数据集划分工具页面。",
                informative_text=str(exc),
                details=str(html_path),
            )
            return

        if opened:
            self._set_status_message(f"已打开划分工具：{html_path.name}")
        else:
            self._info_dialog(
                "请手动打开",
                "系统未能自动拉起浏览器，请手动打开该文件。",
                details=str(html_path),
            )

    def _parse_yolo_txt_class_counts(self, label_path: Path) -> dict[int, int]:
        counts: dict[int, int] = {}
        if not label_path.exists() or label_path.suffix.lower() != ".txt":
            return counts

        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
            except Exception:
                continue
            counts[class_id] = counts.get(class_id, 0) + 1
        return counts

    def _resolve_extract_split_layout(
        self,
        source_root: Path,
        splits: tuple[str, ...],
    ) -> tuple[str | None, dict[str, tuple[Path, Path]], str | None]:
        split_aliases: dict[str, tuple[str, ...]] = {}
        for split_name in splits:
            if split_name == "val":
                split_aliases[split_name] = ("val", "valid")
            else:
                split_aliases[split_name] = (split_name,)

        resolved_layouts: dict[str, dict[str, tuple[Path, Path]]] = {}
        for mode in ("images_split", "split_nested"):
            mapping: dict[str, tuple[Path, Path]] = {}
            mode_ok = True
            for split_name in splits:
                aliases = split_aliases.get(split_name, (split_name,))
                selected_pair: tuple[Path, Path] | None = None
                for alias in aliases:
                    if mode == "images_split":
                        image_dir = source_root / "images" / alias
                        label_dir = source_root / "labels" / alias
                    else:
                        image_dir = source_root / alias / "images"
                        label_dir = source_root / alias / "labels"
                    if image_dir.exists() and image_dir.is_dir() and label_dir.exists() and label_dir.is_dir():
                        selected_pair = (image_dir, label_dir)
                        break
                if selected_pair is None:
                    mode_ok = False
                    break
                mapping[split_name] = selected_pair
            if mode_ok:
                resolved_layouts[mode] = mapping

        if not resolved_layouts:
            missing_a = [
                (
                    "images/val(或valid) + labels/val(或valid)"
                    if split_name == "val"
                    else f"images/{split_name} + labels/{split_name}"
                )
                for split_name in splits
            ]
            missing_b = [
                (
                    "val(或valid)/images + val(或valid)/labels"
                    if split_name == "val"
                    else f"{split_name}/images + {split_name}/labels"
                )
                for split_name in splits
            ]
            detail = (
                "未匹配到以下任一结构：\n"
                f"1) {'；'.join(missing_a)}\n"
                f"2) {'；'.join(missing_b)}"
            )
            return None, {}, detail

        selected_mode = "images_split" if "images_split" in resolved_layouts else next(iter(resolved_layouts))
        return selected_mode, resolved_layouts[selected_mode], None

    def _extract_layout_caption(self, mode: str) -> str:
        if mode == "split_nested":
            return "train/images + train/labels"
        return "images/train + labels/train"

    def _build_extract_output_paths(
        self,
        output_root: Path,
        split_name: str,
        relative_path: Path,
        layout_mode: str,
    ) -> tuple[Path, Path]:
        if layout_mode == "split_nested":
            image_output = output_root / split_name / "images" / relative_path
            label_output = (output_root / split_name / "labels" / relative_path).with_suffix(".txt")
            return image_output, label_output

        image_output = output_root / "images" / split_name / relative_path
        label_output = (output_root / "labels" / split_name / relative_path).with_suffix(".txt")
        return image_output, label_output

    def _prepare_extract_output_dirs(self, output_root: Path, splits: tuple[str, ...], layout_mode: str) -> None:
        for split_name in splits:
            if layout_mode == "split_nested":
                (output_root / split_name / "images").mkdir(parents=True, exist_ok=True)
                (output_root / split_name / "labels").mkdir(parents=True, exist_ok=True)
            else:
                (output_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
                (output_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    def _collect_split_dataset_items(
        self,
        images_dir: Path,
        labels_dir: Path,
    ) -> tuple[list[ExtractDatasetItem], dict[int, int]]:
        items: list[ExtractDatasetItem] = []
        class_totals: dict[int, int] = {}

        if not images_dir.exists() or not labels_dir.exists():
            return items, class_totals

        image_paths = [
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        image_paths.sort(key=lambda item: str(item).lower())

        for image_path in image_paths:
            relative_path = image_path.relative_to(images_dir)
            expected_label_path = (labels_dir / relative_path).with_suffix(".txt")
            label_path = expected_label_path if expected_label_path.exists() else None
            class_counts = self._parse_yolo_txt_class_counts(label_path) if label_path is not None else {}
            for class_id, count in class_counts.items():
                class_totals[class_id] = class_totals.get(class_id, 0) + int(count)
            items.append(
                ExtractDatasetItem(
                    image_path=image_path,
                    label_path=label_path,
                    relative_path=relative_path,
                    class_counts=class_counts,
                )
            )

        return items, class_totals

    def _filter_extract_items_by_class_ids(
        self,
        items: list[ExtractDatasetItem],
        selected_class_ids: set[int],
    ) -> tuple[list[ExtractDatasetItem], dict[int, int]]:
        selected_ids = {int(class_id) for class_id in selected_class_ids}
        filtered_items: list[ExtractDatasetItem] = []
        filtered_totals: dict[int, int] = {}

        for item in items:
            class_counts = {
                class_id: int(count)
                for class_id, count in item.class_counts.items()
                if class_id in selected_ids
            }
            if not class_counts:
                continue

            filtered_items.append(
                ExtractDatasetItem(
                    image_path=item.image_path,
                    label_path=item.label_path,
                    relative_path=item.relative_path,
                    class_counts=class_counts,
                )
            )
            for class_id, count in class_counts.items():
                filtered_totals[class_id] = filtered_totals.get(class_id, 0) + int(count)

        return filtered_items, filtered_totals

    def _write_filtered_extract_label(
        self,
        source_label_path: Path,
        target_label_path: Path,
        selected_class_ids: set[int],
    ) -> None:
        selected_ids = {int(class_id) for class_id in selected_class_ids}
        filtered_lines: list[str] = []

        for raw_line in source_label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 5:
                continue

            try:
                class_id = int(float(parts[0]))
            except Exception:
                continue

            if class_id in selected_ids:
                filtered_lines.append(raw_line)

        target_label_path.write_text("\n".join(filtered_lines), encoding="utf-8")

    def _sample_extract_items_with_ratio(
        self,
        items: list[ExtractDatasetItem],
        class_totals: dict[int, int],
        sample_ratio: float,
        seed: int,
    ) -> list[ExtractDatasetItem]:
        if not items:
            return []

        target_count = int(round(len(items) * sample_ratio))
        if sample_ratio > 0:
            target_count = max(1, target_count)
        target_count = min(len(items), target_count)
        if target_count >= len(items):
            return list(items)

        target_class_totals = {class_id: count * sample_ratio for class_id, count in class_totals.items()}
        selected: list[ExtractDatasetItem] = []
        selected_counts: dict[int, int] = {}

        rng = random.Random(int(seed))
        remaining_indices = list(range(len(items)))
        rng.shuffle(remaining_indices)

        while remaining_indices and len(selected) < target_count:
            if len(remaining_indices) > 240:
                candidate_indices = rng.sample(remaining_indices, 240)
            else:
                candidate_indices = list(remaining_indices)

            progress_ratio = float(len(selected) + 1) / float(target_count)
            best_index = candidate_indices[0]
            best_score = float("inf")

            for index in candidate_indices:
                item = items[index]
                score = 0.0
                for class_id, target_total in target_class_totals.items():
                    desired = target_total * progress_ratio
                    projected = selected_counts.get(class_id, 0) + item.class_counts.get(class_id, 0)
                    score += abs(projected - desired) / max(1.0, target_total)

                if not item.class_counts:
                    score += 0.03

                score += rng.random() * 1e-6
                if score < best_score:
                    best_score = score
                    best_index = index

            selected_item = items[best_index]
            selected.append(selected_item)
            for class_id, count in selected_item.class_counts.items():
                selected_counts[class_id] = selected_counts.get(class_id, 0) + int(count)
            remaining_indices.remove(best_index)

        if len(selected) < target_count:
            for index in remaining_indices[: target_count - len(selected)]:
                selected.append(items[index])

        return selected

    def _aggregate_extract_class_counts(self, items: list[ExtractDatasetItem]) -> dict[int, int]:
        aggregated: dict[int, int] = {}
        for item in items:
            for class_id, count in item.class_counts.items():
                aggregated[class_id] = aggregated.get(class_id, 0) + int(count)
        return aggregated

    def extract_dataset_mini(self) -> None:
        default_root = str(self.dataset_service.root_dir) if self.dataset_service.root_dir else ""
        source_root_text = QFileDialog.getExistingDirectory(self, "选择待抽取的数据集根目录", default_root)
        if not source_root_text:
            return

        source_root = Path(source_root_text)
        splits = ("train", "val", "test")
        layout_mode, split_dirs, layout_error = self._resolve_extract_split_layout(source_root, splits)
        if layout_mode is None:
            self._warning_dialog(
                "结构不完整",
                "抽取功能仅支持已划分数据集。",
                informative_text=layout_error or "请检查目录结构。",
                details=str(source_root),
            )
            return

        output_dir_text = QFileDialog.getExistingDirectory(self, "选择抽取输出目录")
        if not output_dir_text:
            return

        output_root = Path(output_dir_text)
        if self._is_path_within(output_root, source_root):
            self._warning_dialog(
                "输出目录无效",
                "输出目录不能位于源数据集内部。",
                informative_text="请重新选择独立目录。",
            )
            return

        if output_root.exists() and any(output_root.iterdir()):
            reply = self._ask_dialog(
                "目录非空",
                "输出目录已有文件，继续可能覆盖同名文件。是否继续？",
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                default_button=QMessageBox.StandardButton.No,
                button_texts={
                    QMessageBox.StandardButton.Yes: "继续",
                    QMessageBox.StandardButton.No: "取消",
                },
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        ratio_percent = int(self.export_extract_ratio_spin.value()) if hasattr(self, "export_extract_ratio_spin") else 20
        sample_ratio = max(0.01, min(1.0, ratio_percent / 100.0))
        random_seed = int(self.export_extract_seed_spin.value()) if hasattr(self, "export_extract_seed_spin") else 42

        source_items: dict[str, list[ExtractDatasetItem]] = {}
        source_class_totals: dict[str, dict[int, int]] = {}
        for split_name in splits:
            image_dir, label_dir = split_dirs[split_name]
            items, class_totals = self._collect_split_dataset_items(image_dir, label_dir)
            if not items:
                self._warning_dialog(
                    "分级为空",
                    f"{split_name} 未发现可抽取图片。",
                    informative_text=(
                        f"请检查目录：{image_dir.relative_to(source_root)}"
                        if image_dir.is_relative_to(source_root)
                        else f"请检查目录：{image_dir}"
                    ),
                    details=str(source_root),
                )
                return
            source_items[split_name] = items
            source_class_totals[split_name] = class_totals

        source_manager = ClassManager()
        source_manager.load_from_root(source_root)

        observed_class_ids: set[int] = set()
        for class_totals in source_class_totals.values():
            observed_class_ids.update(class_totals.keys())

        selected_class_ids: set[int] | None = None
        available_class_ids = sorted(observed_class_ids)
        if available_class_ids:
            class_items = [
                (class_id, source_manager.get_name(class_id) or f"ID {class_id}")
                for class_id in available_class_ids
            ]
            selected_ids, accepted = ClassChecklistDialog.get_selected_class_ids(
                self,
                class_items,
                default_checked_ids=set(available_class_ids),
                title="选择抽取类别",
                prompt="勾选需要抽取的类别：",
            )
            if not accepted:
                return

            selected_class_ids = {int(class_id) for class_id in selected_ids}
            if not selected_class_ids:
                self._info_dialog("抽取结果", "未选择任何类别。")
                return

            if selected_class_ids == set(available_class_ids):
                selected_class_ids = None

        effective_items: dict[str, list[ExtractDatasetItem]] = {}
        effective_class_totals: dict[str, dict[int, int]] = {}
        for split_name in splits:
            if selected_class_ids is None:
                effective_items[split_name] = list(source_items[split_name])
                effective_class_totals[split_name] = dict(source_class_totals[split_name])
                continue

            filtered_items, filtered_totals = self._filter_extract_items_by_class_ids(
                source_items[split_name],
                selected_class_ids,
            )
            effective_items[split_name] = filtered_items
            effective_class_totals[split_name] = filtered_totals

        sampled_items: dict[str, list[ExtractDatasetItem]] = {}
        for split_index, split_name in enumerate(splits):
            sampled_items[split_name] = self._sample_extract_items_with_ratio(
                effective_items[split_name],
                effective_class_totals[split_name],
                sample_ratio,
                seed=random_seed + split_index * 1009,
            )

        total_to_copy = sum(len(items) for items in sampled_items.values())
        if total_to_copy <= 0:
            self._info_dialog("抽取结果", "没有可抽取的样本。")
            return

        self._prepare_extract_output_dirs(output_root, splits, layout_mode)

        copied = 0
        copied_items: dict[str, list[ExtractDatasetItem]] = {split_name: [] for split_name in splits}
        cancelled = False
        progress = QProgressDialog("正在抽取数据集...", "取消", 0, total_to_copy, self)
        progress.setWindowTitle("抽取数据集")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        for split_name in splits:
            for item in sampled_items[split_name]:
                QApplication.processEvents()
                if progress.wasCanceled():
                    cancelled = True
                    break

                output_image_path, output_label_path = self._build_extract_output_paths(
                    output_root,
                    split_name,
                    item.relative_path,
                    layout_mode,
                )
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.image_path, output_image_path)

                if item.label_path is not None and item.label_path.exists():
                    output_label_path.parent.mkdir(parents=True, exist_ok=True)
                    if selected_class_ids is None:
                        shutil.copy2(item.label_path, output_label_path)
                    else:
                        self._write_filtered_extract_label(item.label_path, output_label_path, selected_class_ids)

                copied_items[split_name].append(item)
                copied += 1
                progress.setValue(copied)
                progress.setLabelText(f"正在处理: {split_name}/{item.image_path.name}")

            if cancelled:
                break

        progress.setValue(copied)
        progress.close()

        export_class_ids = available_class_ids if selected_class_ids is None else sorted(selected_class_ids)
        if not export_class_ids:
            export_class_ids = sorted(source_manager.id_to_name.keys())
        class_mapping = {
            class_id: source_manager.get_name(class_id) or f"ID {class_id}"
            for class_id in export_class_ids
        }
        if not class_mapping:
            class_mapping = {class_id: f"ID {class_id}" for class_id in sorted(observed_class_ids)}

        if layout_mode == "split_nested":
            train_path, val_path, test_path = "train/images", "val/images", "test/images"
        else:
            train_path, val_path, test_path = "images/train", "images/val", "images/test"

        self._write_merge_data_yaml(
            output_root,
            class_mapping,
            split_mode=True,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
        )

        details_lines = [
            f"源目录：{source_root}",
            f"输出目录：{output_root}",
            f"识别结构：{self._extract_layout_caption(layout_mode)}",
            f"抽取比例：{ratio_percent}%",
            f"随机种子：{random_seed}",
            f"类别筛选：{'全部类别' if selected_class_ids is None else f'已选 {len(selected_class_ids)} 类'}",
            "",
        ]
        for split_name in splits:
            before_images = len(source_items[split_name])
            after_images = len(copied_items[split_name])
            before_counts = effective_class_totals[split_name]
            after_counts = self._aggregate_extract_class_counts(copied_items[split_name])
            before_total = sum(before_counts.values())
            after_total = sum(after_counts.values())
            details_lines.append(
                f"{split_name}: 图片 {after_images}/{before_images}，标注 {after_total}/{before_total}"
            )

        self._info_dialog(
            "抽取已取消" if cancelled else "抽取完成",
            f"已抽取 {copied} 张图片。",
            informative_text="已按 train/val/test 分级输出，并保持类别比例尽量一致。",
            details="\n".join(details_lines),
        )

    def _merge_single_image_worker(
        self,
        task: MergeImageTask,
        global_manager: ClassManager,
        mapping_lock: Lock,
    ) -> bool:
        image_size = self._load_image_size(task.source_image_path)
        if image_size is None:
            return False

        local_manager = ClassManager()
        local_manager.restore(task.source_manager_state)
        local_manager.root_dir = None
        local_manager.yaml_path = None

        if task.source_label_path is not None:
            boxes = self.annotation_io.load_annotation(task.source_label_path, image_size, local_manager)
        else:
            boxes = []

        try:
            task.output_image_path.parent.mkdir(parents=True, exist_ok=True)
            task.output_label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(task.source_image_path, task.output_image_path)

            with mapping_lock:
                self._ensure_merge_class_id_mapping(
                    task.source_dataset_tag,
                    local_manager,
                    task.source_class_id_to_target_id,
                    global_manager,
                )
                source_class_id_to_target_id = dict(task.source_class_id_to_target_id)

            self._save_yolo_txt_with_id_mapping(
                task.output_label_path,
                boxes,
                image_size,
                local_manager,
                source_class_id_to_target_id,
            )
            return True
        except Exception:
            try:
                if task.output_image_path.exists():
                    task.output_image_path.unlink()
            except Exception:
                pass
            try:
                if task.output_label_path.exists():
                    task.output_label_path.unlink()
            except Exception:
                pass
            return False

    def show_image(self, index: int) -> None:
        self.open_image_at_index(index, prompt_if_dirty=True)

    def show_previous_image(self) -> None:
        if self.current_image_index > 0:
            self.open_image_at_index(self.current_image_index - 1)

    def show_next_image(self) -> None:
        if self.current_image_index + 1 < len(self.dataset_service.image_paths):
            self.open_image_at_index(self.current_image_index + 1)

    def on_image_item_clicked(self, item: QListWidgetItem) -> None:
        if self._defer_image_open_for_multi_select(item):
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(path, Path):
            index = self._image_index_map.get(path)
            if index is not None:
                self.open_image_at_index(index)

    def _defer_image_open_for_multi_select(self, clicked_item: QListWidgetItem | None = None) -> bool:
        modifiers = QApplication.keyboardModifiers()
        if modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier):
            return True

        selected_items = self.image_list_widget.selectedItems()
        if len(selected_items) > 1:
            return True

        if clicked_item is not None and clicked_item not in selected_items and selected_items:
            return True

        return False

    def on_image_current_item_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        if self._defer_image_open_for_multi_select(current):
            return
        self.on_image_item_clicked(current)

    def _selected_dataset_image_paths(self) -> list[Path]:
        selected_paths: list[Path] = []
        for item in self.image_list_widget.selectedItems():
            path = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(path, Path):
                selected_paths.append(path)

        if not selected_paths:
            for item in self.thumbnail_list_widget.selectedItems():
                path = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(path, Path):
                    selected_paths.append(path)

        if not selected_paths:
            current_item = self.image_list_widget.currentItem()
            if current_item is not None:
                path = current_item.data(Qt.ItemDataRole.UserRole)
                if isinstance(path, Path):
                    selected_paths.append(path)

        unique_paths = list(dict.fromkeys(selected_paths))
        unique_paths.sort(key=lambda item: self._image_index_map.get(item, 10**9))
        return unique_paths

    def delete_selected_dataset_items(self) -> bool:
        if self.dataset_service.root_dir is None or not self.dataset_service.image_paths:
            return False

        selected_paths = self._selected_dataset_image_paths()
        if not selected_paths:
            self._set_status_message("请先在图像列表选择要删除的项目")
            return False

        label_count = 0
        for image_path in selected_paths:
            if self.dataset_service.find_label_for_image(image_path) is not None:
                label_count += 1

        preview_lines = [self.dataset_service.display_name(path) for path in selected_paths[:8]]
        if len(selected_paths) > 8:
            preview_lines.append(f"... 其余 {len(selected_paths) - 8} 项")

        reply = self._ask_dialog(
            "确认删除项目",
            f"将删除 {len(selected_paths)} 张图片，以及对应的 {label_count} 个标注文件。",
            informative_text="该操作支持通过 Ctrl+Z 撤销。",
            details="\n".join(preview_lines),
            buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            default_button=QMessageBox.StandardButton.Cancel,
            button_texts={
                QMessageBox.StandardButton.Yes: "删除",
                QMessageBox.StandardButton.Cancel: "取消",
            },
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False

        if not self._prepare_for_context_change():
            return False

        backups: list[DeletedDatasetItem] = []
        failed_reads: list[str] = []
        for image_path in selected_paths:
            image_index = self._image_index_map.get(image_path, len(self.dataset_service.image_paths))
            try:
                image_bytes = image_path.read_bytes()
            except Exception as exc:
                failed_reads.append(f"{self.dataset_service.display_name(image_path)}（图片读取失败：{exc}）")
                continue

            label_path = self.dataset_service.find_label_for_image(image_path)
            label_bytes: bytes | None = None
            if label_path is not None and label_path.exists():
                try:
                    label_bytes = label_path.read_bytes()
                except Exception as exc:
                    failed_reads.append(f"{self.dataset_service.display_name(image_path)}（标注读取失败：{exc}）")
                    continue

            backups.append(
                DeletedDatasetItem(
                    image_path=image_path,
                    image_bytes=image_bytes,
                    label_path=label_path,
                    label_bytes=label_bytes,
                    source_index=image_index,
                )
            )

        deleted_items, failed_deletes = self._delete_backed_up_items(backups)
        failed = failed_reads + failed_deletes
        if not deleted_items:
            if failed:
                self._warning_dialog(
                    "删除失败",
                    "未删除任何项目。",
                    informative_text="请检查文件占用状态或访问权限后重试。",
                    details="\n".join(failed[:8]),
                )
            return False

        deleted_set = {item.image_path for item in deleted_items}
        preferred_after_delete = self._resolve_preferred_after_deletion(deleted_set)

        record = DatasetDeleteRecord(items=deleted_items, preferred_after_delete=preferred_after_delete)
        self._dataset_delete_undo_stack.append(record)
        if len(self._dataset_delete_undo_stack) > self._dataset_delete_history_limit:
            self._dataset_delete_undo_stack = self._dataset_delete_undo_stack[-self._dataset_delete_history_limit :]
        self._dataset_delete_redo_stack.clear()

        self._reload_dataset_after_file_change(preferred_after_delete)
        self._sync_history_actions()

        summary = f"已删除 {len(deleted_items)} 项"
        if failed:
            summary += f"，{len(failed)} 项失败"
        self._set_status_message(summary)
        if failed:
            self._warning_dialog(
                "部分删除失败",
                f"共有 {len(failed)} 个项目删除失败。",
                informative_text="可先关闭占用这些文件的程序后重试。",
                details="\n".join(failed[:8]),
            )
        return True

    def _delete_backed_up_items(self, backups: list[DeletedDatasetItem]) -> tuple[list[DeletedDatasetItem], list[str]]:
        deleted_items: list[DeletedDatasetItem] = []
        failures: list[str] = []
        for backup in backups:
            image_deleted = False
            label_deleted = False
            try:
                if backup.image_path.exists():
                    backup.image_path.unlink()
                    image_deleted = True
                if backup.label_path is not None and backup.label_path.exists():
                    backup.label_path.unlink()
                    label_deleted = True
                deleted_items.append(backup)
            except Exception as exc:
                if image_deleted:
                    try:
                        backup.image_path.parent.mkdir(parents=True, exist_ok=True)
                        backup.image_path.write_bytes(backup.image_bytes)
                    except Exception:
                        pass
                if label_deleted and backup.label_path is not None and backup.label_bytes is not None:
                    try:
                        backup.label_path.parent.mkdir(parents=True, exist_ok=True)
                        backup.label_path.write_bytes(backup.label_bytes)
                    except Exception:
                        pass
                failures.append(f"{self.dataset_service.display_name(backup.image_path)}（{exc}）")
        return deleted_items, failures

    def _restore_deleted_items(self, backups: list[DeletedDatasetItem]) -> tuple[list[DeletedDatasetItem], list[str]]:
        restored_items: list[DeletedDatasetItem] = []
        failures: list[str] = []
        for backup in backups:
            try:
                backup.image_path.parent.mkdir(parents=True, exist_ok=True)
                backup.image_path.write_bytes(backup.image_bytes)
                if backup.label_path is not None and backup.label_bytes is not None:
                    backup.label_path.parent.mkdir(parents=True, exist_ok=True)
                    backup.label_path.write_bytes(backup.label_bytes)
                restored_items.append(backup)
            except Exception as exc:
                failures.append(f"{self.dataset_service.display_name(backup.image_path)}（{exc}）")
        return restored_items, failures

    def _resolve_preferred_after_deletion(self, deleted_paths: set[Path]) -> Path | None:
        if not self.dataset_service.image_paths:
            return None

        remaining = [path for path in self.dataset_service.image_paths if path not in deleted_paths]
        if not remaining:
            return None

        anchor_index = min((self._image_index_map.get(path, 10**9) for path in deleted_paths), default=0)
        if anchor_index >= len(remaining):
            anchor_index = len(remaining) - 1
        return remaining[max(0, anchor_index)]

    def _reload_dataset_after_file_change(self, preferred_path: Path | None) -> None:
        if self.dataset_service.root_dir is None:
            return
        self._start_dataset_loading(self.dataset_service.root_dir, auto_open_first=False, preferred_path=preferred_path)

    def on_image_filter_changed(self, text: str) -> None:
        self._image_filter_text = text.strip().lower()
        self._refresh_image_list()
        self._refresh_thumbnail_list()

    def _refresh_image_list(self) -> None:
        self.image_list_widget.setUpdatesEnabled(False)
        try:
            with QSignalBlocker(self.image_list_widget):
                self.image_list_widget.clear()
                for path in self.dataset_service.image_paths:
                    display = self.dataset_service.display_name(path)
                    if self._image_filter_text and self._image_filter_text not in display.lower():
                        continue
                    item = QListWidgetItem(display)
                    item.setData(Qt.ItemDataRole.UserRole, path)
                    item.setToolTip(str(path))
                    self.image_list_widget.addItem(item)

                if self.current_document is not None:
                    current_path = self.current_document.image_path
                    for row in range(self.image_list_widget.count()):
                        item = self.image_list_widget.item(row)
                        if item and item.data(Qt.ItemDataRole.UserRole) == current_path:
                            self.image_list_widget.setCurrentRow(row)
                            break
        finally:
            self.image_list_widget.setUpdatesEnabled(True)

    def _refresh_thumbnail_list(self, *, force: bool = False) -> None:
        if not hasattr(self, "thumbnail_list_widget"):
            return

        if not force and not self._is_thumbnail_panel_visible():
            self._thumbnail_refresh_pending = True
            return

        self._thumbnail_refresh_pending = False

        visible_paths: list[Path] = []
        pending_paths: list[Path] = []
        item_index_map: dict[Path, int] = {}
        self.thumbnail_list_widget.setUpdatesEnabled(False)
        try:
            with QSignalBlocker(self.thumbnail_list_widget):
                self.thumbnail_list_widget.clear()
                for path in self.dataset_service.image_paths:
                    display = self.dataset_service.display_name(path)
                    if self._image_filter_text and self._image_filter_text not in display.lower():
                        continue

                    visible_paths.append(path)
                    item_index = len(visible_paths) - 1
                    item_index_map[path] = item_index
                    item = QListWidgetItem(display)
                    item.setData(Qt.ItemDataRole.UserRole, path)
                    item.setToolTip(str(path))

                    signature = self._thumbnail_signature(path)
                    cached = self._thumbnail_cache.get(path)
                    if cached is not None and signature is not None and cached[0] == signature:
                        item.setIcon(QIcon(cached[1]))
                    else:
                        item.setIcon(QIcon(self._thumbnail_icon_placeholder()))
                        pending_paths.append(path)
                    self.thumbnail_list_widget.addItem(item)
        finally:
            self.thumbnail_list_widget.setUpdatesEnabled(True)

        self._thumbnail_visible_paths = visible_paths
        self._thumbnail_pending_paths = pending_paths
        self._thumbnail_item_index_map = item_index_map
        self._sync_thumbnail_selection()
        self._start_thumbnail_loading(pending_paths)

    def _cancel_thumbnail_loading(self) -> None:
        worker = self._thumbnail_worker
        thread = self._thumbnail_thread
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
        if thread is not None and thread.isRunning():
            thread.quit()

    def _thumbnail_icon_placeholder(self) -> QPixmap:
        size = self._thumbnail_dimensions()
        placeholder = QPixmap(size)
        placeholder.fill(Qt.GlobalColor.darkGray)
        return placeholder

    def _thumbnail_signature(self, path: Path) -> tuple[int, int] | None:
        try:
            stat_result = path.stat()
        except Exception:
            return None
        return stat_result.st_mtime_ns, stat_result.st_size

    def _start_thumbnail_loading(self, visible_paths: list[Path]) -> None:
        self._cancel_thumbnail_loading()
        self._thumbnail_job_generation += 1
        job_id = self._thumbnail_job_generation

        size = self._thumbnail_dimensions()
        worker = ThumbnailLoadWorker(visible_paths, thumbnail_size=(size.width(), size.height()))
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.thumbnailReady.connect(
            lambda index, image, signature, job=job_id: self._on_thumbnail_ready(job, index, image, signature)
        )
        worker.progressChanged.connect(lambda current, total, job=job_id: self._on_thumbnail_progress(job, current, total))
        worker.finished.connect(lambda job=job_id: self._on_thumbnail_finished(job))
        worker.failed.connect(lambda message, job=job_id: self._on_thumbnail_failed(job, message))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thumbnail_worker = worker
        self._thumbnail_thread = thread
        thread.start()

    def _on_thumbnail_ready(self, job_id: int, index: int, thumbnail_image, signature) -> None:
        if job_id != self._thumbnail_job_generation:
            return
        if index < 0 or index >= len(self._thumbnail_pending_paths):
            return
        path = self._thumbnail_pending_paths[index]
        pixmap = QPixmap.fromImage(thumbnail_image)
        if isinstance(signature, tuple) and len(signature) == 2:
            self._thumbnail_cache[path] = (signature, pixmap)
        else:
            self._thumbnail_cache[path] = ((0, 0), pixmap)

        if not self._is_thumbnail_panel_visible():
            return
        item_index = self._thumbnail_item_index_map.get(path)
        if item_index is None or item_index < 0 or item_index >= self.thumbnail_list_widget.count():
            return
        item = self.thumbnail_list_widget.item(item_index)
        if item is not None:
            item.setIcon(QIcon(pixmap))

    def _on_thumbnail_progress(self, job_id: int, current: int, total: int) -> None:
        if job_id != self._thumbnail_job_generation:
            return
        if not self._is_thumbnail_panel_visible():
            return
        self.status_zoom_label.setText(f"缩放 {int(round(self.current_zoom_scale * 100))}% | 缩略图 {current}/{total}")

    def _on_thumbnail_finished(self, job_id: int) -> None:
        if job_id != self._thumbnail_job_generation:
            return
        self._thumbnail_worker = None
        self._thumbnail_thread = None
        self._update_status_bar_values()

    def _on_thumbnail_failed(self, job_id: int, message: str) -> None:
        if job_id != self._thumbnail_job_generation:
            return
        self._thumbnail_worker = None
        self._thumbnail_thread = None
        self._set_status_message(f"缩略图加载失败：{message}")

    def on_thumbnail_item_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(path, Path):
            index = self._image_index_map.get(path)
            if index is not None:
                self.open_image_at_index(index)

    def on_thumbnail_current_item_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        self.on_thumbnail_item_clicked(current)

    def _sync_image_browser_selection(self) -> None:
        current_path = self.current_document.image_path if self.current_document else None
        if current_path is None:
            return

        with QSignalBlocker(self.image_list_widget):
            for row in range(self.image_list_widget.count()):
                item = self.image_list_widget.item(row)
                if item and item.data(Qt.ItemDataRole.UserRole) == current_path:
                    self.image_list_widget.setCurrentRow(row)
                    break

    def _sync_thumbnail_selection(self) -> None:
        current_path = self.current_document.image_path if self.current_document else None
        if current_path is None:
            return

        with QSignalBlocker(self.thumbnail_list_widget):
            for row in range(self.thumbnail_list_widget.count()):
                item = self.thumbnail_list_widget.item(row)
                if item and item.data(Qt.ItemDataRole.UserRole) == current_path:
                    self.thumbnail_list_widget.setCurrentRow(row)
                    break

    def _refresh_project_info(self) -> None:
        root_text = str(self.dataset_service.root_dir) if self.dataset_service.root_dir else "未加载"
        self.dataset_root_label.setText(root_text)
        self.image_count_label.setText(str(self.dataset_service.image_count()))
        self.label_count_label.setText(str(self.dataset_service.label_count()))
        self.current_file_label.setText(self.current_document.image_name if self.current_document else "--")

    def _refresh_class_widgets(self) -> None:
        self.canvas.set_class_mapping(dict(self.class_manager.name_to_id))
        selected_class_id: int | None = None
        selected_item = self.class_list_widget.currentItem()
        if selected_item is not None:
            candidate = selected_item.data(Qt.ItemDataRole.UserRole)
            if isinstance(candidate, int):
                selected_class_id = candidate

        with QSignalBlocker(self.class_list_widget), QSignalBlocker(self.box_class_combo):
            self.class_list_widget.clear()
            self.box_class_combo.clear()
            selected_row = -1
            for class_id, class_name in self.class_manager.sorted_items():
                item = QListWidgetItem(f"ID {class_id} | {class_name}")
                item.setData(Qt.ItemDataRole.UserRole, class_id)
                self.class_list_widget.addItem(item)
                self.box_class_combo.addItem(class_name, class_id)
                if selected_class_id is not None and class_id == selected_class_id:
                    selected_row = self.class_list_widget.count() - 1

            if selected_row >= 0:
                self.class_list_widget.setCurrentRow(selected_row)
            elif self.class_list_widget.count() > 0:
                self.class_list_widget.setCurrentRow(0)

            self.class_count_label.setText(str(len(self.class_manager.id_to_name)))
            self.yaml_path_label.setText(str(self.class_manager.yaml_path) if self.class_manager.yaml_path else "未检测到 YAML")
            self.yaml_content_view.setPlainText(self._build_yaml_content_preview())

            if self.current_document and self._selected_box_exists():
                class_name = self.current_selected_box.class_name
                combo_index = self.box_class_combo.findText(class_name)
                if combo_index >= 0:
                    self.box_class_combo.setCurrentIndex(combo_index)
                else:
                    self.box_class_combo.setEditText(class_name)
            elif self.box_class_combo.count() > 0:
                self.box_class_combo.setCurrentIndex(0)

        self.on_class_selection_changed()

    def _build_yaml_content_preview(self) -> str:
        path = self.class_manager.yaml_path
        train_value = "images/train"
        val_value = "images/val"
        test_value: str | None = None

        if path is not None and path.exists():
            try:
                for raw_line in path.read_text(encoding="utf-8").splitlines():
                    stripped = raw_line.strip()
                    if stripped.startswith("train:"):
                        train_value = stripped.partition(":")[2].strip() or train_value
                    elif stripped.startswith("val:"):
                        val_value = stripped.partition(":")[2].strip() or val_value
                    elif stripped.startswith("test:"):
                        value = stripped.partition(":")[2].strip()
                        test_value = value or test_value
            except Exception:
                pass

        lines: list[str] = [
            f"train: {train_value}",
            f"val: {val_value}",
        ]
        if test_value:
            lines.append(f"test: {test_value}")
        lines.append("nc: " + str(len(self.class_manager.id_to_name)))
        lines.append("names:")
        for class_id, class_name in self.class_manager.sorted_items():
            lines.append(f"  {class_id}: {class_name}")
        if not self.class_manager.id_to_name:
            lines.append("  {}")
        return "\n".join(lines)

    def create_class(self) -> None:
        text = self.class_input.text().strip()
        if not text:
            return

        existing_id = self.class_manager.get_id(text)
        if existing_id is not None:
            self._select_class_item_by_id(existing_id)
            self._set_status_message(f"类别已存在：ID {existing_id} | {text}")
            return

        if self.current_document is not None:
            self.history.begin(self.current_document.boxes, self.canvas.selected_index, self.class_manager)
        try:
            class_id = self.class_manager.ensure_name(text)
        except Exception as exc:
            self.history.clear_pending()
            self._warning_dialog("创建类别失败", str(exc))
            return

        self.class_manager.sync_to_yaml()
        if self.current_document is not None:
            self.history.commit(self.current_document.boxes, self.canvas.selected_index, self.class_manager)
        self._refresh_class_widgets()
        self._select_class_item_by_id(class_id)
        self._refresh_annotation_controls()
        self._set_status_message(f"已创建类别：ID {class_id} | {text}")

    def _select_class_item_by_id(self, class_id: int) -> None:
        for row in range(self.class_list_widget.count()):
            item = self.class_list_widget.item(row)
            if item and item.data(Qt.ItemDataRole.UserRole) == class_id:
                self.class_list_widget.setCurrentRow(row)
                break

    def rename_selected_class(self) -> None:
        item = self.class_list_widget.currentItem()
        if item is None:
            self.create_class()
            return
        text = self.class_input.text().strip()
        if not text:
            self._info_dialog("重命名类别", "请输入新的类别名称。")
            return
        class_id = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(class_id, int):
            if self.current_document is not None:
                self.history.begin(self.current_document.boxes, self.canvas.selected_index, self.class_manager)
            try:
                old_name = self.class_manager.rename_class(class_id, text)
            except Exception as exc:
                self.history.clear_pending()
                self._warning_dialog("重命名失败", str(exc))
                return
            if old_name == text:
                self._set_status_message("类别名称未变化")
                self.history.clear_pending()
                return
            if self.current_document is not None:
                for box in self.current_document.boxes:
                    if box.class_name == old_name:
                        box.class_name = text
                self._mark_document_dirty()
            self.class_manager.sync_to_yaml()
            if self.current_document is not None:
                self.history.commit(self.current_document.boxes, self.canvas.selected_index, self.class_manager)
            self._refresh_all_views()
            self._set_status_message(f"类别已重命名为 {text}")

    def delete_selected_class(self) -> None:
        item = self.class_list_widget.currentItem()
        if item is None:
            self._info_dialog("删除类别", "请先选择一个要删除的类别。")
            return

        class_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(class_id, int):
            self._warning_dialog("删除类别失败", "无效的类别选择。")
            return

        class_name = self.class_manager.get_name(class_id)
        if not class_name:
            self._warning_dialog("删除类别失败", "未找到对应类别。")
            return

        total_boxes, affected_files, current_boxes, has_unsaved_delta = self._collect_delete_preview(class_name)
        action = self._confirm_delete_class(
            class_id,
            class_name,
            total_boxes,
            affected_files,
            current_boxes,
            has_unsaved_delta,
        )
        if action == "cancel":
            return

        removed_boxes = 0
        cleaned_files = 0
        skipped_files = 0
        cleanup_canceled = False
        if action == "cleanup":
            if not self._prepare_for_context_change():
                return
            cleaned_files, removed_boxes, skipped_files, cleanup_canceled = self._purge_class_annotations(class_name)
            if cleanup_canceled:
                self._set_status_message("已取消清理，未删除类别")
                return
        elif current_boxes > 0:
            self._info_dialog(
                "当前图片仍包含该类别",
                f"当前图片中仍有 {current_boxes} 个“{class_name}”标注框。",
                informative_text="若仅删除类别定义，后续保存时该类别可能再次出现。",
                details="建议改用“删除类别并清理标注框”。",
            )
            return

        if self.current_document is not None:
            self.history.begin(self.current_document.boxes, self.canvas.selected_index, self.class_manager)
        try:
            deleted_name = self.class_manager.delete_class(class_id)
        except Exception as exc:
            self.history.clear_pending()
            self._warning_dialog("删除类别失败", str(exc))
            return

        self.class_manager.sync_to_yaml(force=True)
        if self.current_document is not None:
            self.history.commit(self.current_document.boxes, self.canvas.selected_index, self.class_manager)

        if action == "cleanup" and self.dataset_service.root_dir is not None:
            preferred = self.current_document.image_path if self.current_document else None
            self._start_dataset_loading(self.dataset_service.root_dir, auto_open_first=False, preferred_path=preferred)
            suffix = (
                f"，已清理 {removed_boxes} 个标注框（{cleaned_files} 个文件）"
                + (f"，跳过 {skipped_files} 个文件" if skipped_files > 0 else "")
            )
            self._set_status_message(f"已删除类别 {deleted_name}{suffix}")
            return

        self._refresh_all_views()
        self._set_status_message(f"已删除类别 {deleted_name}")

    def _dataset_live_class_count(self, class_name: str) -> int:
        counts = dict(self._dataset_stats_baseline_counts)
        if self.current_document is not None:
            live_counts, _, _ = self._capture_box_statistics(self.current_document.boxes)
            counts = self._apply_stats_delta(counts, self._current_document_saved_counts, live_counts)
        return int(counts.get(class_name, 0))

    def _label_contains_class(self, label_path: Path, class_name: str, class_mapping: dict[int, str]) -> int:
        counts, _ = self.annotation_io.count_annotation(label_path, class_mapping)
        return 1 if counts.get(class_name, 0) > 0 else 0

    def _collect_delete_preview(self, class_name: str) -> tuple[int, int, int, bool]:
        usage_count = self._dataset_live_class_count(class_name)

        affected_files = 0
        label_paths = list(self.dataset_service.label_paths)
        if label_paths:
            class_mapping = dict(self.class_manager.id_to_name)
            if len(label_paths) >= 24:
                workers = self._recommended_parallel_workers(len(label_paths))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(self._label_contains_class, label_path, class_name, class_mapping)
                        for label_path in label_paths
                    ]
                    for future in futures:
                        try:
                            affected_files += int(future.result())
                        except Exception:
                            continue
            else:
                for label_path in label_paths:
                    try:
                        affected_files += self._label_contains_class(label_path, class_name, class_mapping)
                    except Exception:
                        continue

        current_boxes = 0
        has_unsaved_delta = False
        if self.current_document is not None:
            current_boxes = sum(1 for box in self.current_document.boxes if box.class_name == class_name)
            saved_boxes = int(self._current_document_saved_counts.get(class_name, 0))
            has_unsaved_delta = self.autosave.dirty and saved_boxes != current_boxes
            if has_unsaved_delta:
                if saved_boxes == 0 and current_boxes > 0:
                    affected_files += 1
                elif saved_boxes > 0 and current_boxes == 0:
                    affected_files = max(0, affected_files - 1)

        return usage_count, affected_files, current_boxes, has_unsaved_delta

    def _purge_single_image_worker(
        self,
        image_path: Path,
        class_name: str,
        class_id: int | None,
        class_state,
    ) -> tuple[int, int, int]:
        label_path = self.dataset_service.find_label_for_image(image_path)
        if label_path is None:
            return 0, 0, 0

        if label_path.suffix.lower() == ".txt" and class_id is not None:
            removed = self.annotation_io.remove_class_id_from_txt_file(label_path, class_id)
            if removed > 0:
                return 1, removed, 0
            return 0, 0, 0

        image = self._load_image_as_qimage(image_path)
        if image is None:
            return 0, 0, 1

        local_manager = ClassManager()
        local_manager.restore(class_state)
        local_manager.root_dir = None
        local_manager.yaml_path = None

        boxes = self.annotation_io.load_annotation(
            label_path,
            (image.width(), image.height()),
            local_manager,
        )
        filtered_boxes = [box for box in boxes if box.class_name != class_name]
        removed = len(boxes) - len(filtered_boxes)
        if removed <= 0:
            return 0, 0, 0

        self.annotation_io.save_annotation(
            label_path,
            filtered_boxes,
            (image.width(), image.height()),
            local_manager,
            image_name=image_path.name,
        )
        return 1, removed, 0

    def _confirm_delete_class(
        self,
        class_id: int,
        class_name: str,
        usage_count: int,
        affected_files: int,
        current_boxes: int,
        has_unsaved_delta: bool,
    ) -> str:
        usage_hint = f"预计影响：{usage_count} 个标注框，约 {affected_files} 个标注文件。"
        if usage_count <= 0:
            usage_hint = "当前数据集中未检测到该类别标注框。"

        current_hint = (
            f"当前图片中有 {current_boxes} 个该类别标注。"
            if current_boxes > 0
            else "当前图片中没有该类别标注。"
        )
        if has_unsaved_delta:
            current_hint += "（包含未保存变更）"

        reply = self._show_dialog(
            "warning",
            "删除类别确认",
            f"删除类别：ID {class_id} | {class_name}",
            informative_text=f"{usage_hint}\n{current_hint}\n\n请选择删除方式。",
            details=(
                "删除行为说明：\n"
                "1. 仅删除类别定义：只更新类别映射与 YAML，不改动现有标注框。\n"
                "2. 删除类别并清理标注框：会遍历整个数据集并删除该类别所有标注框。\n"
                "3. 若存在未保存修改，系统会先要求保存。"
            ),
            buttons=(
                QMessageBox.StandardButton.Apply
                | QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel
            ),
            default_button=QMessageBox.StandardButton.Cancel,
            button_texts={
                QMessageBox.StandardButton.Apply: "仅删除类别定义",
                QMessageBox.StandardButton.Yes: "删除类别并清理标注框",
                QMessageBox.StandardButton.Cancel: "取消",
            },
        )
        if reply == QMessageBox.StandardButton.Apply:
            return "definition"
        if reply == QMessageBox.StandardButton.Yes:
            return "cleanup"
        return "cancel"

    def _purge_class_annotations(self, class_name: str) -> tuple[int, int, int, bool]:
        if not self.dataset_service.image_paths:
            return 0, 0, 0, False

        image_paths = list(self.dataset_service.image_paths)
        total = len(image_paths)
        progress = QProgressDialog("正在清理类别标注框...", "取消", 0, total, self)
        progress.setWindowTitle("删除类别标注")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        cleaned_files = 0
        removed_boxes = 0
        skipped_files = 0
        canceled = False
        class_state = self.class_manager.snapshot()
        class_id = self.class_manager.get_id(class_name)
        workers = self._recommended_parallel_workers(total)
        processed = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            image_iter = iter(image_paths)
            pending: dict[Future[tuple[int, int, int]], Path] = {}

            def submit_next() -> bool:
                try:
                    next_image = next(image_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    self._purge_single_image_worker,
                    next_image,
                    class_name,
                    class_id,
                    class_state,
                )
                pending[future] = next_image
                return True

            for _ in range(min(workers, total)):
                if not submit_next():
                    break

            while pending:
                done, _ = wait(set(pending.keys()), timeout=0.08, return_when=FIRST_COMPLETED)
                QApplication.processEvents()

                if progress.wasCanceled():
                    canceled = True

                if not done:
                    continue

                for future in done:
                    image_path = pending.pop(future)
                    processed += 1
                    progress.setLabelText(f"正在处理: {self.dataset_service.display_name(image_path)}")
                    try:
                        cleaned_delta, removed_delta, skipped_delta = future.result()
                    except Exception:
                        cleaned_delta, removed_delta, skipped_delta = 0, 0, 1

                    cleaned_files += cleaned_delta
                    removed_boxes += removed_delta
                    skipped_files += skipped_delta

                    progress.setValue(processed)

                    if not canceled:
                        submit_next()

                QApplication.processEvents()

        progress.setValue(processed if canceled else total)
        progress.close()
        return cleaned_files, removed_boxes, skipped_files, canceled

    def remap_selected_class_id(self) -> None:
        item = self.class_list_widget.currentItem()
        if item is None:
            self._info_dialog("修改类别 ID", "请先选择一个类别。")
            return

        old_id = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(old_id, int):
            self._warning_dialog("修改类别 ID 失败", "无效的类别选择。")
            return

        old_name = self.class_manager.get_name(old_id)
        if not old_name:
            self._warning_dialog("修改类别 ID 失败", "未找到对应类别。")
            return

        new_id = int(self.class_target_id_spin.value())
        if old_id == new_id:
            self._set_status_message("目标索引与当前索引一致")
            return

        if not self._prepare_for_context_change():
            return

        target_name = self.class_manager.get_name(new_id)
        swap = target_name is not None and self.class_swap_checkbox.isChecked()
        merge_into_existing = False
        if target_name is not None and not swap:
            reply = self._ask_dialog(
                "确认合并类别",
                f"目标索引 ID {new_id} 已被类别“{target_name}”使用。",
                informative_text=(
                    f"继续后会将“{old_name}”的标注迁移并合并为“{target_name}”，"
                    "并删除原类别定义。"
                ),
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                default_button=QMessageBox.StandardButton.Cancel,
                button_texts={
                    QMessageBox.StandardButton.Yes: "确认合并",
                    QMessageBox.StandardButton.Cancel: "取消",
                },
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            merge_into_existing = True

        id_mapping = {old_id: new_id}
        if swap and target_name is not None:
            id_mapping[new_id] = old_id

        rewritten_txt_files = 0
        scanned_txt_files = 0
        if self.class_rewrite_labels_checkbox.isChecked():
            rewritten_txt_files, scanned_txt_files = self._remap_dataset_txt_labels(id_mapping)
        elif any(path.suffix.lower() == ".txt" for path in self.dataset_service.label_paths):
            reply = self._ask_dialog(
                "确认继续",
                "当前数据集中存在 TXT 标注文件。",
                informative_text="若不改写标签，将导致类别索引与 YAML 映射不一致。",
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                default_button=QMessageBox.StandardButton.No,
                button_texts={
                    QMessageBox.StandardButton.Yes: "继续",
                    QMessageBox.StandardButton.No: "返回修改",
                },
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        moved_name = old_name
        swapped_name: str | None = None
        try:
            if merge_into_existing:
                self.class_manager.delete_class(old_id)
            else:
                moved_name, swapped_name = self.class_manager.remap_class_id(
                    old_id,
                    new_id,
                    swap_if_conflict=swap,
                )
        except Exception as exc:
            self._warning_dialog("修改类别 ID 失败", str(exc))
            return

        self.class_manager.sync_to_yaml(force=True)

        preferred_path = self.current_document.image_path if self.current_document is not None else None
        if self.dataset_service.root_dir is not None:
            # ID remap or merge can rewrite many labels; trigger dataset rescan so statistics stay in sync.
            self._reload_dataset_after_file_change(preferred_path)
        elif self.current_document is not None and self.current_image_index >= 0:
            self.open_image_at_index(self.current_image_index, prompt_if_dirty=False)
        else:
            self._refresh_all_views()

        if merge_into_existing and target_name is not None:
            if scanned_txt_files > 0:
                self._set_status_message(
                    f"已合并类别: {moved_name} (ID {old_id}) -> {target_name} (ID {new_id})，"
                    f"改写 TXT {rewritten_txt_files}/{scanned_txt_files}"
                )
            else:
                self._set_status_message(
                    f"已合并类别: {moved_name} (ID {old_id}) -> {target_name} (ID {new_id})"
                )
        elif swapped_name is not None:
            self._set_status_message(
                f"已交换索引: {moved_name} -> ID {new_id}, {swapped_name} -> ID {old_id}"
            )
        elif scanned_txt_files > 0:
            self._set_status_message(
                f"已更新索引: {moved_name} -> ID {new_id}，改写 TXT {rewritten_txt_files}/{scanned_txt_files}"
            )
        else:
            self._set_status_message(f"已更新索引: {moved_name} -> ID {new_id}")

    def _remap_dataset_txt_labels(self, id_mapping: dict[int, int]) -> tuple[int, int]:
        txt_paths = [path for path in self.dataset_service.label_paths if path.suffix.lower() == ".txt"]
        if not txt_paths:
            return 0, 0

        progress = QProgressDialog("正在改写 TXT 类别索引...", "", 0, len(txt_paths), self)
        progress.setWindowTitle("索引重映射")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setCancelButton(None)
        progress.show()

        rewritten = 0
        workers = self._recommended_parallel_workers(len(txt_paths))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self.annotation_io.remap_class_ids_in_file, path, id_mapping)
                for path in txt_paths
            ]
            for index, future in enumerate(futures, start=1):
                try:
                    if future.result():
                        rewritten += 1
                except Exception:
                    pass
                progress.setValue(index)
                QApplication.processEvents()

        progress.setValue(len(txt_paths))
        progress.close()
        return rewritten, len(txt_paths)

    def sync_yaml_file(self) -> None:
        path = self.class_manager.sync_to_yaml(force=True)
        if path is None:
            self._info_dialog("同步 YAML", "当前没有可同步的 YAML 文件。")
            return
        self._refresh_class_widgets()
        self._set_status_message(f"已同步 {path}")

    def on_canvas_annotation_changed(self) -> None:
        if self.current_document is not None:
            committed = self.history.commit(self.current_document.boxes, self.canvas.selected_index, self.class_manager)
            self._sync_history_actions()
            if committed:
                self._mark_document_dirty()
                self._refresh_box_widgets()
                self._refresh_project_info()
                self._update_window_title()

    def on_canvas_edit_operation_started(self) -> None:
        if self.current_document is None:
            return
        self.history.begin(self.current_document.boxes, self.canvas.selected_index, self.class_manager)
        self._sync_history_actions()

    def on_canvas_selection_changed(self, index: int) -> None:
        self._set_selected_index(index, sync_canvas=False)

    def on_canvas_cursor_changed(self, x: int, y: int) -> None:
        self.status_cursor_label.setText(f"坐标 {x}, {y}" if x >= 0 and y >= 0 else "坐标 --")

    def on_canvas_pan_requested(self, dx: int, dy: int) -> None:
        horizontal = self.scroll_area.horizontalScrollBar()
        vertical = self.scroll_area.verticalScrollBar()
        if horizontal is not None:
            horizontal.setValue(horizontal.value() - int(dx))
        if vertical is not None:
            vertical.setValue(vertical.value() - int(dy))

    def on_draw_box_requested(self, x1: int, y1: int, x2: int, y2: int) -> None:
        default_name = self.box_class_combo.currentText().strip() or self._pending_class_text
        class_names = [name for _, name in self.class_manager.sorted_items()]
        text, ok = ClassSelectorDialog.get_class_name(
            self,
            class_names,
            default_text=default_name,
            title="新建标注",
            prompt="选择或输入类别名称：",
        )
        if not ok:
            return
        class_name = self.class_manager.resolve_label_token(text)
        if self.current_document is None:
            return
        self.history.begin(self.current_document.boxes, self.canvas.selected_index, self.class_manager)
        new_box = Box(x1, y1, x2, y2, class_name).clamp(*self.current_document.image_size)
        self.current_document.boxes.append(new_box)
        self.class_manager.sync_to_yaml()
        self.canvas.set_selected_index(len(self.current_document.boxes) - 1)
        self._set_selected_index(len(self.current_document.boxes) - 1, sync_canvas=False)
        self.history.commit(self.current_document.boxes, self.canvas.selected_index, self.class_manager)
        self._mark_document_dirty()
        self._refresh_all_views()
        self._set_status_message(f"已创建标注 {class_name}")

    def on_draw_polygon_requested(self, points: object) -> None:
        if self.current_document is None:
            return

        point_list: list[tuple[int, int]] = []
        if isinstance(points, list):
            for raw_point in points:
                try:
                    x, y = raw_point
                    point_list.append((int(x), int(y)))
                except Exception:
                    continue
        if len(point_list) < 3:
            self._set_status_message("多边形至少需要 3 个点")
            return

        default_name = self.box_class_combo.currentText().strip() or self._pending_class_text
        class_names = [name for _, name in self.class_manager.sorted_items()]
        text, ok = ClassSelectorDialog.get_class_name(
            self,
            class_names,
            default_text=default_name,
            title="新建多边形标注",
            prompt="选择或输入类别名称：",
        )
        if not ok:
            return

        class_name = self.class_manager.resolve_label_token(text)
        self.history.begin(self.current_document.boxes, self.canvas.selected_index, self.class_manager)
        try:
            new_polygon = Box.from_polygon(point_list, class_name).clamp(*self.current_document.image_size)
        except Exception:
            self.history.clear_pending()
            self._set_status_message("多边形创建失败")
            return

        self.current_document.boxes.append(new_polygon)
        self.class_manager.sync_to_yaml()
        next_index = len(self.current_document.boxes) - 1
        self.canvas.set_selected_index(next_index)
        self._set_selected_index(next_index, sync_canvas=False)
        self.history.commit(self.current_document.boxes, self.canvas.selected_index, self.class_manager)
        self._mark_document_dirty()
        self._refresh_all_views()
        self._set_status_message(f"已创建多边形标注 {class_name}")

    def rename_selected_box_class(self) -> bool:
        index = self.canvas.selected_index
        if self.current_document is None or not (0 <= index < len(self.current_document.boxes)):
            self._set_status_message("请先选中一个标注")
            return False

        box = self.current_document.boxes[index]
        default_text = box.class_name
        self.history.begin(self.current_document.boxes, index, self.class_manager)
        class_names = [name for _, name in self.class_manager.sorted_items()]
        text, ok = ClassSelectorDialog.get_class_name(
            self,
            class_names,
            default_text=default_text,
            title="修改标注类别",
            prompt="选择或输入新的类别名称：",
        )
        if not ok:
            self.history.clear_pending()
            return False

        resolved = self.class_manager.resolve_label_token(text)
        if resolved == box.class_name:
            self._set_status_message("类别名称未变化")
            self.history.clear_pending()
            return True

        box.class_name = resolved
        self.class_manager.sync_to_yaml()
        self.history.commit(self.current_document.boxes, index, self.class_manager)
        self._mark_document_dirty()
        self._refresh_all_views()
        self._set_status_message(f"标注类别已改为 {resolved}")
        return True

    def _set_selected_index(self, index: int, sync_canvas: bool = True) -> None:
        self.canvas.set_selected_index(index if self.current_document is not None else -1)
        if self.current_document is None or not (0 <= index < len(self.current_document.boxes)):
            self.selected_info_label.setText("未选中")
            self._update_selected_box_size_status()
            if sync_canvas:
                self.canvas.set_selected_index(-1)
            return
        box = self.current_document.boxes[index]
        self.selected_info_label.setText(
            f"{box.summary(index)}\n类别：{box.class_name}"
        )
        with QSignalBlocker(self.box_class_combo):
            combo_index = self.box_class_combo.findText(box.class_name)
            if combo_index >= 0:
                self.box_class_combo.setCurrentIndex(combo_index)
            else:
                self.box_class_combo.setEditText(box.class_name)
        self._sync_box_list_selection(index)
        self._sync_canvas_selection(index, sync_canvas)
        self._update_selected_box_size_status()
        self.canvas.setFocus()

    def _sync_canvas_selection(self, index: int, sync_canvas: bool) -> None:
        if sync_canvas:
            self.canvas.set_selected_index(index)

    def _sync_box_list_selection(self, index: int) -> None:
        with QSignalBlocker(self.box_list_widget):
            for row in range(self.box_list_widget.count()):
                item = self.box_list_widget.item(row)
                if item and item.data(Qt.ItemDataRole.UserRole) == index:
                    self.box_list_widget.setCurrentRow(row)
                    break
            else:
                if index < 0:
                    self.box_list_widget.clearSelection()

    def on_box_selection_changed(self) -> None:
        item = self.box_list_widget.currentItem()
        if item is None:
            self._set_selected_index(-1)
            return
        box_index = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(box_index, int):
            self._set_selected_index(box_index)

    def _refresh_box_widgets(self) -> None:
        if self.current_document is None:
            with QSignalBlocker(self.box_list_widget):
                self.box_list_widget.clear()
            self.selected_info_label.setText("未选中")
            self.status_box_label.setText("标注 0")
            self._update_selected_box_size_status()
            return

        with QSignalBlocker(self.box_list_widget):
            self.box_list_widget.clear()
            for index, box in enumerate(self.current_document.boxes):
                item = QListWidgetItem(box.summary(index))
                item.setData(Qt.ItemDataRole.UserRole, index)
                item.setToolTip(box.summary(index))
                self.box_list_widget.addItem(item)
            self.status_box_label.setText(f"标注 {len(self.current_document.boxes)}")

        if self.canvas.selected_index >= 0:
            self._set_selected_index(self.canvas.selected_index)
        else:
            self.selected_info_label.setText("未选中")
            self.box_list_widget.clearSelection()
            self._update_selected_box_size_status()

    def _refresh_annotation_controls(self) -> None:
        if self.current_document is None:
            self.selected_info_label.setText("未选中")
            self.box_class_combo.setCurrentIndex(-1)
            return
        self._refresh_box_widgets()

    def _refresh_all_views(self) -> None:
        self._refresh_project_info()
        self._refresh_class_widgets()
        self._refresh_box_widgets()
        self._refresh_image_list()
        self._sync_image_browser_selection()
        self._sync_thumbnail_selection()
        self._update_dataset_statistics_widget()
        self._update_mode_widgets()
        self._update_status_bar_values()
        self._update_window_title()
        self._sync_history_actions()

    def _capture_box_statistics(self, boxes: list[Box]) -> tuple[dict[str, int], int, bool]:
        counts: dict[str, int] = {}
        for box in boxes:
            counts[box.class_name] = counts.get(box.class_name, 0) + 1
        box_count = len(boxes)
        annotated = box_count > 0
        return counts, box_count, annotated

    def _update_dataset_statistics_widget(self, *, force: bool = False) -> None:
        if not hasattr(self, "dataset_statistics_widget"):
            return

        if self._dataset_statistics_loading:
            self._set_statistics_loading_state("正在统计数据...")
            return

        if not force and not self._is_statistics_panel_visible():
            self._statistics_refresh_pending = True
            return

        self._statistics_refresh_pending = False

        self.dataset_statistics_widget.set_class_mapping(dict(self.class_manager.name_to_id))
        counts = dict(self._dataset_stats_baseline_counts)
        total_images = self._dataset_stats_total_images
        annotated_images = self._dataset_stats_annotated_images
        total_boxes = self._dataset_stats_total_boxes

        if self.current_document is not None:
            live_counts, live_box_count, live_annotated = self._capture_box_statistics(self.current_document.boxes)
            counts = self._apply_stats_delta(counts, self._current_document_saved_counts, live_counts)
            total_boxes = total_boxes - self._current_document_saved_box_count + live_box_count
            annotated_images = annotated_images - int(self._current_document_saved_is_annotated) + int(live_annotated)

        self.dataset_statistics_widget.set_statistics(counts, total_images, annotated_images, total_boxes)

    def _apply_stats_delta(
        self,
        base_counts: dict[str, int],
        old_counts: dict[str, int],
        new_counts: dict[str, int],
    ) -> dict[str, int]:
        updated = dict(base_counts)
        affected_names = set(old_counts) | set(new_counts)
        for class_name in affected_names:
            updated[class_name] = updated.get(class_name, 0) - old_counts.get(class_name, 0) + new_counts.get(class_name, 0)
            if updated[class_name] <= 0:
                updated.pop(class_name, None)
        return updated

    def _store_dataset_statistics_baseline(self, result: DatasetStatisticsResult) -> None:
        self._dataset_stats_baseline_counts = dict(result.class_counts)
        self._dataset_stats_total_images = int(result.total_images)
        self._dataset_stats_annotated_images = int(result.annotated_images)
        self._dataset_stats_total_boxes = int(result.total_boxes)

    def _store_current_document_saved_statistics(self, boxes: list[Box]) -> None:
        counts, box_count, annotated = self._capture_box_statistics(boxes)
        self._current_document_saved_counts = counts
        self._current_document_saved_box_count = box_count
        self._current_document_saved_is_annotated = annotated

    def _set_statistics_loading_state(self, text: str) -> None:
        if not hasattr(self, "dataset_statistics_widget"):
            return
        if not self._is_statistics_panel_visible():
            self._statistics_refresh_pending = True
            return
        self.dataset_statistics_widget.set_loading(text)

    def _on_dataset_scan_progress(self, job_id: int, current: int, total: int) -> None:
        if job_id != self._dataset_job_generation:
            return
        self._set_status_message(f"正在扫描数据集... {current}/{total}")

    def _on_dataset_scan_finished(self, job_id: int, result: DatasetScanResult) -> None:
        if job_id != self._dataset_job_generation:
            return

        self._dataset_loading = False
        previous_root = self.dataset_service.root_dir
        self.dataset_service.apply_scan_result(result.root_dir, result.image_paths, result.label_paths)
        if previous_root is None or previous_root != Path(result.root_dir):
            self._dataset_delete_undo_stack.clear()
            self._dataset_delete_redo_stack.clear()
            self._thumbnail_cache.clear()
        self.class_manager.load_from_root(result.root_dir)
        self.history.clear()
        self.history.clear_pending()
        self._image_index_map = {path: index for index, path in enumerate(self.dataset_service.image_paths)}
        self._image_filter_text = ""
        self.image_filter_edit.blockSignals(True)
        self.image_filter_edit.clear()
        self.image_filter_edit.blockSignals(False)

        self._refresh_project_info()
        self._refresh_class_widgets()
        self._refresh_image_list()

        if self.dataset_service.image_paths:
            if self._pending_dataset_preferred_path is not None and self._pending_dataset_preferred_path in self._image_index_map:
                self.open_image_at_index(self._image_index_map[self._pending_dataset_preferred_path], prompt_if_dirty=False)
            else:
                self.open_image_at_index(0, prompt_if_dirty=False)
        else:
            self._clear_document()

        self._refresh_thumbnail_list()
        self._start_dataset_statistics_loading(result)

        self._pending_dataset_root = None
        self._pending_dataset_preferred_path = None
        self._update_status_bar_values()
        self._set_status_message(f"数据集已就绪，共 {self.dataset_service.image_count()} 张图片；统计后台进行中")

    def _on_dataset_scan_failed(self, job_id: int, message: str) -> None:
        if job_id != self._dataset_job_generation:
            return
        self._dataset_loading = False
        self._dataset_statistics_loading = False
        self._dataset_statistics_worker = None
        self._dataset_statistics_thread = None
        self._set_loaded_state(bool(self.dataset_service.image_paths))
        self._set_statistics_loading_state("数据加载失败")
        self._set_status_message(f"数据集加载失败：{message}")

    def _start_dataset_statistics_loading(self, result: DatasetScanResult) -> None:
        self._cancel_dataset_statistics_loading()
        if not result.image_paths:
            self._dataset_statistics_loading = False
            self._set_statistics_loading_state("暂无统计数据")
            return

        self._dataset_stats_job_generation += 1
        job_id = self._dataset_stats_job_generation
        self._dataset_statistics_loading = True
        self._set_statistics_loading_state("正在统计数据...")

        worker = DatasetStatisticsWorker(result, dict(self.class_manager.id_to_name))
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progressChanged.connect(
            lambda current, total, job=job_id: self._on_dataset_statistics_progress(job, current, total)
        )
        worker.finished.connect(lambda stats, job=job_id: self._on_dataset_statistics_finished(job, stats))
        worker.failed.connect(lambda message, job=job_id: self._on_dataset_statistics_failed(job, message))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._dataset_statistics_worker = worker
        self._dataset_statistics_thread = thread
        thread.start()

    def _cancel_dataset_statistics_loading(self) -> None:
        worker = self._dataset_statistics_worker
        thread = self._dataset_statistics_thread
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
        if thread is not None and thread.isRunning():
            thread.quit()
        self._dataset_statistics_worker = None
        self._dataset_statistics_thread = None
        self._dataset_statistics_loading = False

    def _on_dataset_statistics_progress(self, job_id: int, current: int, total: int) -> None:
        if job_id != self._dataset_stats_job_generation:
            return
        if self._is_statistics_panel_visible():
            self.status_message_label.setText(f"正在统计标注... {current}/{total}")

    def _on_dataset_statistics_finished(self, job_id: int, result: DatasetStatisticsResult) -> None:
        if job_id != self._dataset_stats_job_generation:
            return
        if self.dataset_service.root_dir is None or Path(result.root_dir) != self.dataset_service.root_dir:
            return

        self._dataset_statistics_loading = False
        self._dataset_statistics_worker = None
        self._dataset_statistics_thread = None
        self._store_dataset_statistics_baseline(result)
        self._update_dataset_statistics_widget(force=True)
        self._set_status_message(
            f"统计完成：{result.annotated_images}/{result.total_images} 张有标注，{result.total_boxes} 个标注"
        )

    def _on_dataset_statistics_failed(self, job_id: int, message: str) -> None:
        if job_id != self._dataset_stats_job_generation:
            return
        self._dataset_statistics_loading = False
        self._dataset_statistics_worker = None
        self._dataset_statistics_thread = None
        self._set_statistics_loading_state("统计失败")
        self._set_status_message(f"统计失败：{message}")

    def _sync_history_actions(self) -> None:
        has_document = self.current_document is not None
        can_document_undo = has_document and self.history.can_undo()
        can_document_redo = has_document and self.history.can_redo()
        can_dataset_undo = bool(self._dataset_delete_undo_stack)
        can_dataset_redo = bool(self._dataset_delete_redo_stack)
        self.action_undo.setEnabled(can_document_undo or can_dataset_undo)
        self.action_redo.setEnabled(can_document_redo or can_dataset_redo)

    def undo_last_action(self) -> bool:
        if self.current_document is not None and self.history.can_undo():
            snapshot = self.history.undo()
            if snapshot is None:
                self._set_status_message("没有可撤销的操作")
                return False
            self._apply_snapshot(snapshot)
            self._set_status_message("已撤销")
            return True

        if not self._dataset_delete_undo_stack:
            self._set_status_message("没有可撤销的操作")
            return False

        if not self._prepare_for_context_change():
            return False

        record = self._dataset_delete_undo_stack.pop()
        restored_items, failures = self._restore_deleted_items(record.items)
        if not restored_items:
            self._set_status_message("撤销失败")
            if failures:
                self._warning_dialog(
                    "撤销失败",
                    "无法恢复已删除项目。",
                    informative_text="请检查文件权限或占用状态后重试。",
                    details="\n".join(failures[:8]),
                )
            self._sync_history_actions()
            return False

        self._dataset_delete_redo_stack.append(
            DatasetDeleteRecord(items=restored_items, preferred_after_delete=record.preferred_after_delete)
        )
        if len(self._dataset_delete_redo_stack) > self._dataset_delete_history_limit:
            self._dataset_delete_redo_stack = self._dataset_delete_redo_stack[-self._dataset_delete_history_limit :]

        restored_items_sorted = sorted(restored_items, key=lambda item: item.source_index)
        preferred = restored_items_sorted[0].image_path if restored_items_sorted else None
        self._reload_dataset_after_file_change(preferred)
        self._sync_history_actions()

        message = f"已撤销删除，恢复 {len(restored_items)} 项"
        if failures:
            message += f"，{len(failures)} 项失败"
        self._set_status_message(message)
        if failures:
            self._warning_dialog(
                "部分恢复失败",
                f"共有 {len(failures)} 个项目未成功恢复。",
                details="\n".join(failures[:8]),
            )
        return True

    def redo_last_action(self) -> bool:
        if self.current_document is not None and self.history.can_redo():
            snapshot = self.history.redo()
            if snapshot is None:
                self._set_status_message("没有可重做的操作")
                return False
            self._apply_snapshot(snapshot)
            self._set_status_message("已重做")
            return True

        if not self._dataset_delete_redo_stack:
            self._set_status_message("没有可重做的操作")
            return False

        if not self._prepare_for_context_change():
            return False

        record = self._dataset_delete_redo_stack.pop()
        deleted_items, failures = self._delete_backed_up_items(record.items)
        if not deleted_items:
            self._set_status_message("重做失败")
            if failures:
                self._warning_dialog(
                    "重做失败",
                    "无法再次删除这些项目。",
                    informative_text="请检查文件权限或占用状态后重试。",
                    details="\n".join(failures[:8]),
                )
            self._sync_history_actions()
            return False

        self._dataset_delete_undo_stack.append(
            DatasetDeleteRecord(items=deleted_items, preferred_after_delete=record.preferred_after_delete)
        )
        if len(self._dataset_delete_undo_stack) > self._dataset_delete_history_limit:
            self._dataset_delete_undo_stack = self._dataset_delete_undo_stack[-self._dataset_delete_history_limit :]

        deleted_set = {item.image_path for item in deleted_items}
        preferred = self._resolve_preferred_after_deletion(deleted_set)
        self._reload_dataset_after_file_change(preferred)
        self._sync_history_actions()

        message = f"已重做删除，删除 {len(deleted_items)} 项"
        if failures:
            message += f"，{len(failures)} 项失败"
        self._set_status_message(message)
        if failures:
            self._warning_dialog(
                "部分重做失败",
                f"共有 {len(failures)} 个项目未成功删除。",
                details="\n".join(failures[:8]),
            )
        return True

    def _apply_snapshot(self, snapshot) -> None:
        if self.current_document is None:
            return
        self.class_manager.restore(snapshot.class_state)
        self.class_manager.sync_to_yaml(force=True)
        self.current_document.boxes = [box.copy() for box in snapshot.boxes]
        self.canvas.set_boxes(self.current_document.boxes)
        self.canvas.set_selected_index(snapshot.selected_index)
        self._refresh_all_views()
        self._set_selected_index(snapshot.selected_index, sync_canvas=True)
        self._mark_document_dirty()

    def apply_class_to_selected_box(self) -> None:
        index = self.canvas.selected_index
        if self.current_document is None or not (0 <= index < len(self.current_document.boxes)):
            self._info_dialog("应用类别", "请先选中一个标注框。")
            return
        class_name = self.box_class_combo.currentText().strip()
        if not class_name:
            self._info_dialog("应用类别", "请输入类别名称。")
            return
        current_name = self.current_document.boxes[index].class_name
        resolved = self.class_manager.resolve_label_token(class_name)
        if resolved == current_name:
            self._set_status_message("类别名称未变化")
            return
        self.history.begin(self.current_document.boxes, index, self.class_manager)
        self.current_document.boxes[index].class_name = resolved
        self.class_manager.sync_to_yaml()
        self.history.commit(self.current_document.boxes, index, self.class_manager)
        self._mark_document_dirty()
        self._refresh_all_views()
        self._set_status_message(f"已应用类别 {resolved}")

    def _keyboard_editing_enabled(self) -> bool:
        focused_widget = self.focusWidget()
        return (
            self.current_document is not None
            and self.edit_mode
            and self.canvas.selected_index >= 0
            and focused_widget is self.canvas
        )

    def _nudge_selected_box(self, dx: int, dy: int) -> bool:
        if self.current_document is None:
            return False
        index = self.canvas.selected_index
        if not (0 <= index < len(self.current_document.boxes)):
            return False

        self.history.begin(self.current_document.boxes, index, self.class_manager)
        box = self.current_document.boxes[index].ordered()
        width = box.width()
        height = box.height()
        max_x = max(0, self.current_document.image_width - width)
        max_y = max(0, self.current_document.image_height - height)
        new_x1 = max(0, min(max_x, box.x1 + int(dx)))
        new_y1 = max(0, min(max_y, box.y1 + int(dy)))
        if self.current_document.boxes[index].is_polygon:
            self.current_document.boxes[index] = self.current_document.boxes[index].moved(
                new_x1 - box.x1,
                new_y1 - box.y1,
                self.current_document.image_width,
                self.current_document.image_height,
            )
        else:
            self.current_document.boxes[index] = Box(new_x1, new_y1, new_x1 + width, new_y1 + height, box.class_name)

        self.history.commit(self.current_document.boxes, index, self.class_manager)
        self._mark_document_dirty()
        self._refresh_box_widgets()
        self.canvas.set_selected_index(index)
        self.canvas.update()
        self._set_status_message(f"已移动选中标注 {box.class_name}")
        return True

    def delete_selected_box(self) -> None:
        index = self.canvas.selected_index
        if self.current_document is None or not (0 <= index < len(self.current_document.boxes)):
            return
        self.history.begin(self.current_document.boxes, index, self.class_manager)
        reply = self._ask_dialog(
            "确认删除",
            "确定删除当前选中的标注框吗？",
            details=self.current_document.boxes[index].summary(index),
            buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            default_button=QMessageBox.StandardButton.No,
            button_texts={
                QMessageBox.StandardButton.Yes: "删除",
                QMessageBox.StandardButton.No: "保留",
            },
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.history.clear_pending()
            return
        del self.current_document.boxes[index]
        self.canvas.set_selected_index(-1)
        self.history.commit(self.current_document.boxes, -1, self.class_manager)
        self._mark_document_dirty()
        self._refresh_all_views()
        self._set_status_message("标注已删除")

    def on_class_selection_changed(self) -> None:
        item = self.class_list_widget.currentItem()
        if item is None:
            self.btn_rename_class.setEnabled(False)
            self.btn_delete_class.setEnabled(False)
            self.btn_remap_class_id.setEnabled(False)
            return
        class_id = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(class_id, int):
            class_name = self.class_manager.get_name(class_id) or ""
            with QSignalBlocker(self.class_input):
                self.class_input.setText(class_name)
            with QSignalBlocker(self.class_target_id_spin):
                self.class_target_id_spin.setValue(class_id)
            self.btn_rename_class.setEnabled(True)
            self.btn_delete_class.setEnabled(True)
            self.btn_remap_class_id.setEnabled(True)

    def export_current_annotation(self) -> None:
        if self.current_document is None:
            return
        target_format = str(self.export_format_combo.currentData() or "txt")
        target_path = self._build_annotation_export_path(
            self.current_document.image_path,
            target_format,
            label_path=self.current_document.label_path,
        )
        adopt = self.current_document.label_path is None and target_format != "mask_png"
        self._save_to_path(target_path, adopt_if_new=adopt, show_message=True)

    def _build_annotation_export_path(
        self,
        image_path: Path,
        target_format: str,
        *,
        target_dir: Path | None = None,
        label_path: Path | None = None,
    ) -> Path:
        root_dir = self.dataset_service.root_dir

        if label_path is not None:
            if target_dir is not None and root_dir is not None:
                try:
                    return self._with_annotation_format(target_dir / label_path.relative_to(root_dir), target_format)
                except Exception:
                    pass
            return self._with_annotation_format(label_path, target_format)

        if target_dir is not None and root_dir is not None:
            try:
                relative_path = image_path.relative_to(root_dir)
            except Exception:
                relative_path = Path(image_path.name)
            label_relative = self._image_relative_to_label_relative(relative_path)
            return self._with_annotation_format(target_dir / label_relative, target_format)

        return self._with_annotation_format(image_path, target_format, from_image=True)

    def _image_relative_to_label_relative(self, relative_path: Path) -> Path:
        parts = list(relative_path.parts)
        for index, part in enumerate(parts):
            if part.lower() in {"image", "images", "img", "imgs"}:
                parts[index] = "labels"
                return Path(*parts)
        return relative_path

    def _with_annotation_format(self, path: Path, target_format: str, *, from_image: bool = False) -> Path:
        if target_format == "mask_png":
            if from_image:
                return path.with_name(f"{path.stem}_mask.png")
            return path.with_suffix(".png")
        return path.with_suffix(FORMAT_SUFFIX[target_format])

    def _warm_class_mapping_from_dataset_statistics(self) -> None:
        for class_name in self._dataset_stats_baseline_counts:
            name = str(class_name).strip()
            if not name:
                continue
            self.class_manager.ensure_name(name)
        if self.class_manager.is_dirty:
            self.class_manager.sync_to_yaml()

    def _save_cv_image_file(self, output_path: Path, image_bgr: np.ndarray) -> bool:
        suffix = output_path.suffix.lower() or ".jpg"
        try:
            ok, encoded = cv2.imencode(suffix, image_bgr)
            if not ok:
                return False
            output_path.parent.mkdir(parents=True, exist_ok=True)
            encoded.tofile(str(output_path))
            return True
        except Exception:
            return False

    def _scale_boxes_to_target(
        self,
        boxes: list[Box],
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> list[Box]:
        source_width, source_height = source_size
        target_width, target_height = target_size
        sx = target_width / max(1, float(source_width))
        sy = target_height / max(1, float(source_height))

        scaled: list[Box] = []
        for box in boxes:
            scaled.append(box.transformed(sx, sy, width=target_width, height=target_height))
        return scaled

    def _resize_image_letterbox(
        self,
        source_image: np.ndarray,
        target_size: tuple[int, int],
    ) -> tuple[np.ndarray, float, int, int]:
        source_height, source_width = source_image.shape[:2]
        target_width, target_height = target_size
        scale = min(
            target_width / max(1.0, float(source_width)),
            target_height / max(1.0, float(source_height)),
        )

        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(source_image, (resized_width, resized_height), interpolation=interpolation)

        canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        pad_x = (target_width - resized_width) // 2
        pad_y = (target_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        return canvas, scale, pad_x, pad_y

    def _resize_export_single_image_worker(
        self,
        image_path: Path,
        target_dir: Path,
        target_format: str,
        target_size: tuple[int, int],
        class_state: ClassManagerState,
        use_letterbox: bool,
    ) -> tuple[bool, bool, bool]:
        root_dir = self.dataset_service.root_dir
        if root_dir is None:
            return False, True, False

        try:
            relative_path = image_path.relative_to(root_dir)
        except Exception:
            relative_path = Path(image_path.name)

        output_image_path = target_dir / relative_path

        try:
            data = np.fromfile(str(image_path), dtype=np.uint8)
            source_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            source_image = None

        if source_image is None:
            return False, True, False

        source_height, source_width = source_image.shape[:2]
        target_width, target_height = target_size
        label_path = self.dataset_service.find_label_for_image(image_path)
        output_label_path = self._build_annotation_export_path(
            image_path,
            target_format,
            target_dir=target_dir,
            label_path=label_path,
        )
        boxes: list[Box] = []
        local_manager: ClassManager | None = None
        if label_path is not None:
            local_manager = ClassManager()
            local_manager.restore(class_state)
            local_manager.root_dir = None
            local_manager.yaml_path = None
            try:
                boxes = self.annotation_io.load_annotation(label_path, (source_width, source_height), local_manager)
            except Exception:
                boxes = []

        if use_letterbox:
            resized_image, scale, pad_x, pad_y = self._resize_image_letterbox(source_image, target_size)
            scaled_boxes: list[Box] = []
            for box in boxes:
                scaled_boxes.append(
                    box.transformed(
                        scale,
                        scale,
                        pad_x,
                        pad_y,
                        width=target_width,
                        height=target_height,
                    )
                )
        else:
            interpolation = cv2.INTER_AREA if (target_width < source_width or target_height < source_height) else cv2.INTER_LINEAR
            resized_image = cv2.resize(source_image, (target_width, target_height), interpolation=interpolation)
            scaled_boxes = self._scale_boxes_to_target(
                boxes,
                (source_width, source_height),
                (target_width, target_height),
            )

        if not self._save_cv_image_file(output_image_path, resized_image):
            return False, True, False

        if label_path is None:
            return True, False, False

        try:
            assert local_manager is not None
            self.annotation_io.save_annotation(
                output_label_path,
                scaled_boxes,
                (target_width, target_height),
                local_manager,
                image_name=output_image_path.name,
            )
            return True, False, True
        except Exception:
            try:
                if output_image_path.exists():
                    output_image_path.unlink()
            except Exception:
                pass
            try:
                if output_label_path.exists():
                    output_label_path.unlink()
            except Exception:
                pass
            return False, True, False

    def batch_resize_export_dataset(self) -> None:
        if self.dataset_service.root_dir is None or not self.dataset_service.image_paths:
            self._info_dialog("尺寸转换导出", "请先加载数据集。")
            return

        target_dir_text = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not target_dir_text:
            return

        if not self._prepare_for_context_change():
            return

        target_dir = Path(target_dir_text)
        target_format = str(self.export_format_combo.currentData() or "txt")
        target_width = int(self.export_resize_width_spin.value())
        target_height = int(self.export_resize_height_spin.value())
        target_size = (target_width, target_height)
        use_letterbox = self.export_resize_letterbox_checkbox.isChecked()

        self._warm_class_mapping_from_dataset_statistics()
        class_state = self.class_manager.snapshot()

        converted = 0
        skipped = 0
        labels_written = 0
        cancelled = False
        image_paths = list(self.dataset_service.image_paths)
        total = len(image_paths)
        workers = self._recommended_parallel_workers(total)

        progress = QProgressDialog("正在执行尺寸转换导出...", "取消", 0, total, self)
        progress.setWindowTitle("目标尺寸转换")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        processed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            image_iter = iter(image_paths)
            pending: dict[Future[tuple[bool, bool, bool]], Path] = {}

            def submit_next() -> bool:
                try:
                    next_image = next(image_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    self._resize_export_single_image_worker,
                    next_image,
                    target_dir,
                    target_format,
                    target_size,
                    class_state,
                    use_letterbox,
                )
                pending[future] = next_image
                return True

            for _ in range(min(workers, total)):
                if not submit_next():
                    break

            while pending:
                done, _ = wait(set(pending.keys()), timeout=0.08, return_when=FIRST_COMPLETED)
                QApplication.processEvents()

                if progress.wasCanceled():
                    cancelled = True

                if not done:
                    continue

                for future in done:
                    image_path = pending.pop(future)
                    processed += 1
                    progress.setLabelText(f"正在处理: {self.dataset_service.display_name(image_path)}")
                    try:
                        converted_ok, skipped_flag, wrote_label = future.result()
                    except Exception:
                        converted_ok, skipped_flag, wrote_label = False, True, False

                    if converted_ok:
                        converted += 1
                        if wrote_label:
                            labels_written += 1
                    elif skipped_flag:
                        skipped += 1

                    progress.setValue(processed)

                    if not cancelled:
                        submit_next()

                QApplication.processEvents()

        progress.setValue(processed if cancelled else total)
        progress.close()

        self.class_manager.sync_to_yaml()
        self._refresh_class_widgets()

        if cancelled:
            self._set_status_message(f"尺寸转换已取消，已导出 {converted} 张")

        self._info_dialog(
            "尺寸转换已取消" if cancelled else "尺寸转换完成",
            f"已导出 {converted} 张图片，标签输出 {labels_written} 个。",
            informative_text=f"目标尺寸：{target_width} x {target_height}，跳过 {skipped} 个文件。",
            details=(
                f"转换模式：{'保持比例 Letterbox' if use_letterbox else '直接缩放'}\n"
                f"导出目录：{target_dir}"
            ),
        )

    def _batch_convert_single_image_worker(
        self,
        image_path: Path,
        target_dir: Path,
        target_format: str,
        class_state,
    ) -> tuple[bool, bool]:
        label_path = self.dataset_service.find_label_for_image(image_path)
        if label_path is None:
            return False, True

        qimage = self._load_image_as_qimage(image_path)
        if qimage is None:
            return False, True

        local_manager = ClassManager()
        local_manager.restore(class_state)
        local_manager.root_dir = None
        local_manager.yaml_path = None

        boxes = self.annotation_io.load_annotation(label_path, (qimage.width(), qimage.height()), local_manager)

        root_dir = self.dataset_service.root_dir
        if root_dir is None:
            return False, True

        output_path = self._build_annotation_export_path(
            image_path,
            target_format,
            target_dir=target_dir,
            label_path=label_path,
        )
        self.annotation_io.save_annotation(
            output_path,
            boxes,
            (qimage.width(), qimage.height()),
            local_manager,
            image_name=image_path.name,
        )
        return True, False

    def batch_convert_dataset(self) -> None:
        if self.dataset_service.root_dir is None or not self.dataset_service.image_paths:
            self._info_dialog("批量转换", "请先加载数据集。")
            return

        target_dir_text = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not target_dir_text:
            return
        target_dir = Path(target_dir_text)
        target_format = str(self.export_format_combo.currentData() or "txt")
        if not self._prepare_for_context_change():
            return

        self._warm_class_mapping_from_dataset_statistics()
        class_state = self.class_manager.snapshot()

        converted = 0
        skipped = 0
        cancelled = False
        image_paths = list(self.dataset_service.image_paths)
        total = len(image_paths)
        workers = self._recommended_parallel_workers(total)

        progress = QProgressDialog("批量转换中...", "取消", 0, total, self)
        progress.setWindowTitle("批量转换")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        processed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            image_iter = iter(image_paths)
            pending: dict[Future[tuple[bool, bool]], Path] = {}

            def submit_next() -> bool:
                try:
                    next_image = next(image_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    self._batch_convert_single_image_worker,
                    next_image,
                    target_dir,
                    target_format,
                    class_state,
                )
                pending[future] = next_image
                return True

            for _ in range(min(workers, total)):
                if not submit_next():
                    break

            while pending:
                done, _ = wait(set(pending.keys()), timeout=0.08, return_when=FIRST_COMPLETED)
                QApplication.processEvents()

                if progress.wasCanceled():
                    cancelled = True

                if not done:
                    continue

                for future in done:
                    image_path = pending.pop(future)
                    processed += 1
                    progress.setLabelText(f"正在处理: {self.dataset_service.display_name(image_path)}")
                    try:
                        converted_ok, skipped_flag = future.result()
                    except Exception:
                        converted_ok, skipped_flag = False, True

                    if converted_ok:
                        converted += 1
                    elif skipped_flag:
                        skipped += 1

                    progress.setValue(processed)

                    if not cancelled:
                        submit_next()

                QApplication.processEvents()

        progress.setValue(processed if cancelled else total)
        progress.close()
        self.class_manager.sync_to_yaml()
        self._refresh_class_widgets()
        self._refresh_thumbnail_list()
        self._update_dataset_statistics_widget()
        if cancelled:
            self._set_status_message(f"批量转换已取消，已导出 {converted} 个")
        self._info_dialog(
            "批量转换已取消" if cancelled else "批量转换完成",
            f"已导出 {converted} 个标注文件。",
            informative_text=f"跳过 {skipped} 个无标注或加载失败的文件。",
            details=f"导出目录：{target_dir}",
        )

    def merge_multiple_datasets(self) -> None:
        dataset_roots = self._collect_merge_roots()
        if not dataset_roots:
            return

        output_dir_text = QFileDialog.getExistingDirectory(self, "选择合并后输出目录")
        if not output_dir_text:
            return

        output_root = Path(output_dir_text)
        for root in dataset_roots:
            if self._is_path_within(output_root, root):
                self._warning_dialog(
                    "合并目录无效",
                    "合并输出目录不能位于任一输入数据集内部。",
                    informative_text="请重新选择一个独立的输出目录。",
                )
                return

        merged, skipped, class_count, cancelled = self._merge_datasets_to_output(dataset_roots, output_root)

        if merged < 0:
            return

        if merged == 0:
            self._info_dialog("合并结果", "没有可合并的图片。")
            return

        title = "合并已取消" if cancelled else "合并完成"
        self._info_dialog(
            title,
            f"已处理 {len(dataset_roots)} 个数据集，合并图片 {merged} 张。",
            informative_text=f"跳过 {skipped} 张，输出类别数 {class_count}。",
            details=(
                f"输出目录：{output_root}\n"
                "输出为 YOLO TXT 标签，映射已写入 data.yaml。"
            ),
        )

        reply = self._ask_dialog(
            "打开合并结果",
            "是否立即打开合并后的数据集？",
            buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            default_button=QMessageBox.StandardButton.Yes,
            button_texts={
                QMessageBox.StandardButton.Yes: "立即打开",
                QMessageBox.StandardButton.No: "稍后",
            },
        )
        if reply == QMessageBox.StandardButton.Yes and self._prepare_for_context_change():
            self._start_dataset_loading(output_root, auto_open_first=True, preferred_path=None)

    def _collect_merge_roots(self) -> list[Path]:
        roots: list[Path] = []
        root_layouts: dict[Path, MergeDatasetLayout] = {}
        reference_root: Path | None = None
        reference_layout: MergeDatasetLayout | None = None

        while True:
            title = "选择要合并的第一个数据集目录" if not roots else "继续添加要合并的数据集目录"
            root_text = QFileDialog.getExistingDirectory(self, title)
            if not root_text:
                if roots:
                    break
                return []

            root = Path(root_text)
            if root in roots:
                self._info_dialog("已在列表中", "该目录已添加，无需重复添加。", details=str(root))
            else:
                service = DatasetService()
                service.scan(root)
                if not service.image_paths:
                    self._warning_dialog(
                        "目录不可用",
                        "该目录未发现可用图片，不能加入合并列表。",
                        details=str(root),
                    )
                    continue

                layout, layout_error = self._analyze_merge_dataset_layout(root, service)
                if layout is None:
                    self._warning_dialog(
                        "结构识别失败",
                        "该目录结构不满足合并要求。",
                        informative_text=layout_error or "无法识别数据集结构。",
                        details=str(root),
                    )
                    continue

                if reference_layout is None:
                    reference_root = root
                    reference_layout = layout
                elif layout.signature != reference_layout.signature:
                    self._warning_dialog(
                        "结构不一致",
                        "当前数据集与已选数据集目录分级不一致，不能加入。",
                        informative_text="请保证所有待合并数据集都为同一种结构（未划分或 train/val/test 划分，且目录模板一致）。",
                        details=(
                            f"已选基准：{reference_root}\n"
                            f"{self._describe_merge_layout(reference_layout)}\n\n"
                            f"当前目录：{root}\n"
                            f"{self._describe_merge_layout(layout)}"
                        ),
                    )
                    continue

                roots.append(root)
                root_layouts[root] = layout

            reply = self._ask_dialog(
                "继续添加",
                f"已选择 {len(roots)} 个数据集，是否继续添加？",
                informative_text="可在详情中查看当前已选数据集结构摘要。",
                details=self._build_merge_selection_summary(roots, root_layouts),
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                default_button=QMessageBox.StandardButton.No,
                button_texts={
                    QMessageBox.StandardButton.Yes: "继续添加",
                    QMessageBox.StandardButton.No: "开始合并",
                },
            )

            if reply == QMessageBox.StandardButton.Yes:
                continue

            confirm = self._ask_dialog(
                "确认开始合并",
                f"将合并 {len(roots)} 个数据集，是否开始？",
                informative_text="请再次确认结构摘要后再继续。",
                details=self._build_merge_selection_summary(roots, root_layouts),
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                default_button=QMessageBox.StandardButton.Yes,
                button_texts={
                    QMessageBox.StandardButton.Yes: "开始合并",
                    QMessageBox.StandardButton.No: "返回继续选择",
                },
            )
            if confirm == QMessageBox.StandardButton.Yes:
                break

        return roots

    def _build_merge_selection_summary(
        self,
        roots: list[Path],
        root_layouts: dict[Path, MergeDatasetLayout],
    ) -> str:
        if not roots:
            return "尚未选择数据集。"

        lines: list[str] = []
        first_layout = root_layouts.get(roots[0])
        if first_layout is not None:
            mode_text = "已划分（train/val/test）" if first_layout.mode == "split" else "未划分"
            lines.append(f"结构模式：{mode_text}")
            lines.append("")

        for index, root in enumerate(roots, start=1):
            layout = root_layouts.get(root)
            lines.append(f"[{index}] {root}")
            if layout is None:
                lines.append("  - 结构：未识别")
                lines.append("")
                continue

            mode_text = "已划分" if layout.mode == "split" else "未划分"
            image_templates = ", ".join(layout.image_templates) if layout.image_templates else "(无)"
            label_templates = ", ".join(layout.label_templates) if layout.label_templates else "(无标签)"
            split_names = ", ".join(layout.split_names) if layout.split_names else "-"
            lines.append(f"  - 类型：{mode_text}")
            lines.append(f"  - 分级：{split_names}")
            lines.append(f"  - 图片模板：{image_templates}")
            lines.append(f"  - 标签模板：{label_templates}")
            lines.append("")

        return "\n".join(lines).strip()

    def _merge_datasets_to_output(
        self,
        dataset_roots: list[Path],
        output_root: Path,
    ) -> tuple[int, int, int, bool]:
        sources: list[MergeDatasetSource] = []
        total_images = 0
        reference_layout: MergeDatasetLayout | None = None
        reference_root: Path | None = None
        used_dataset_tags: set[str] = set()

        for root in dataset_roots:
            service = DatasetService()
            service.scan(root)
            if not service.image_paths:
                continue

            layout, layout_error = self._analyze_merge_dataset_layout(root, service)
            if layout is None:
                self._warning_dialog(
                    "数据集结构不支持",
                    f"目录 {root.name or str(root)} 无法参与合并。",
                    informative_text=layout_error or "无法识别数据集目录结构。",
                    details=str(root),
                )
                return (-1, 0, 0, False)

            if reference_layout is None:
                reference_layout = layout
                reference_root = root
            elif layout.signature != reference_layout.signature:
                self._warning_dialog(
                    "数据集结构不一致",
                    "所选数据集目录分级不同，不能直接合并。",
                    informative_text="请确保所有数据集的图片/标签目录层级一致（同为未划分或同为 train/val/test）。",
                    details=(
                        f"基准数据集：{reference_root}\n"
                        f"{self._describe_merge_layout(reference_layout)}\n\n"
                        f"当前数据集：{root}\n"
                        f"{self._describe_merge_layout(layout)}"
                    ),
                )
                return (-1, 0, 0, False)

            source_manager = ClassManager()
            source_manager.load_from_root(root)
            source_manager.root_dir = None
            source_manager.yaml_path = None

            base_tag = self._safe_merge_token(root.name or "dataset")
            dataset_tag = self._unique_merge_stem(base_tag, used_dataset_tags)

            sources.append(
                MergeDatasetSource(
                    root=root,
                    service=service,
                    source_manager_state=source_manager.snapshot(),
                    layout=layout,
                    dataset_tag=dataset_tag,
                    class_id_map={},
                )
            )
            total_images += len(service.image_paths)

        if total_images == 0:
            return (0, 0, 0, False)

        assert reference_layout is not None
        split_mode = reference_layout.mode == "split"
        if split_mode:
            for split_name in ("train", "val", "test"):
                (output_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
                (output_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        else:
            (output_root / "images").mkdir(parents=True, exist_ok=True)
            (output_root / "labels").mkdir(parents=True, exist_ok=True)

        global_manager = ClassManager()
        global_manager.root_dir = None
        global_manager.yaml_path = None

        for source in sources:
            source_manager = ClassManager()
            source_manager.restore(source.source_manager_state)
            source_manager.root_dir = None
            source_manager.yaml_path = None
            self._ensure_merge_class_id_mapping(
                source.dataset_tag,
                source_manager,
                source.class_id_map,
                global_manager,
            )

        used_stems: set[str] = set()
        tasks: list[MergeImageTask] = []
        for source in sources:
            root = source.root
            root_name = root.name or str(root)
            source_state = source.source_manager_state
            for image_path in source.service.image_paths:
                stem = self._build_merge_stem(source.dataset_tag, root, image_path)
                unique_stem = self._unique_merge_stem(stem, used_stems)

                split_name = source.layout.image_split_map.get(image_path, "all")
                if split_mode:
                    output_image_path = output_root / "images" / split_name / f"{unique_stem}{image_path.suffix.lower()}"
                    output_label_path = output_root / "labels" / split_name / f"{unique_stem}.txt"
                else:
                    output_image_path = output_root / "images" / f"{unique_stem}{image_path.suffix.lower()}"
                    output_label_path = output_root / "labels" / f"{unique_stem}.txt"

                tasks.append(
                    MergeImageTask(
                        root_name=root_name,
                        source_dataset_tag=source.dataset_tag,
                        source_image_path=image_path,
                        source_label_path=source.service.find_label_for_image(image_path),
                        source_manager_state=source_state,
                        source_class_id_to_target_id=source.class_id_map,
                        output_image_path=output_image_path,
                        output_label_path=output_label_path,
                    )
                )

        if not tasks:
            return (0, 0, 0, False)

        merged = 0
        skipped = 0
        cancelled = False
        total_tasks = len(tasks)
        workers = self._recommended_parallel_workers(total_tasks)

        progress = QProgressDialog("正在合并数据集...", "取消", 0, total_tasks, self)
        progress.setWindowTitle("多数据集合并")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        processed = 0
        mapping_lock = Lock()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            task_iter = iter(tasks)
            pending: dict[Future[bool], MergeImageTask] = {}

            def submit_next() -> bool:
                try:
                    task = next(task_iter)
                except StopIteration:
                    return False
                future = executor.submit(self._merge_single_image_worker, task, global_manager, mapping_lock)
                pending[future] = task
                return True

            for _ in range(min(workers, total_tasks)):
                if not submit_next():
                    break

            while pending:
                done, _ = wait(set(pending.keys()), timeout=0.08, return_when=FIRST_COMPLETED)
                QApplication.processEvents()

                if progress.wasCanceled():
                    cancelled = True

                if not done:
                    continue

                for future in done:
                    task = pending.pop(future)
                    processed += 1
                    progress.setLabelText(f"正在处理: {task.root_name} / {task.source_image_path.name}")
                    try:
                        success = bool(future.result())
                    except Exception:
                        success = False

                    if success:
                        merged += 1
                    else:
                        skipped += 1

                    progress.setValue(processed)

                    if not cancelled:
                        submit_next()

                QApplication.processEvents()

        progress.setValue(processed if cancelled else total_tasks)
        progress.close()

        if merged > 0:
            self._write_merge_data_yaml(output_root, global_manager.id_to_name, split_mode)

        return (merged, skipped, len(global_manager.id_to_name), cancelled)

    def _build_merge_stem(self, dataset_tag: str, root: Path, image_path: Path) -> str:
        try:
            relative = image_path.relative_to(root).with_suffix("")
            relative_parts = relative.parts
        except Exception:
            relative_parts = (image_path.stem,)

        relative_token = "_".join(self._safe_merge_token(part) for part in relative_parts if part)
        if relative_token:
            return f"{dataset_tag}__{relative_token}"
        return dataset_tag

    def _unique_merge_stem(self, stem: str, used_stems: set[str]) -> str:
        candidate = stem
        suffix = 2
        while candidate in used_stems:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        used_stems.add(candidate)
        return candidate

    def _safe_merge_token(self, text: str) -> str:
        cleaned = "".join(char if char.isalnum() else "_" for char in str(text))
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        cleaned = cleaned.strip("_")
        return cleaned or "item"

    def _is_path_within(self, path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except Exception:
            return False

    def _save_to_path(self, target_path: Path, adopt_if_new: bool, show_message: bool) -> bool:
        if self.current_document is None:
            return False
        old_saved_counts = dict(self._current_document_saved_counts)
        old_saved_box_count = self._current_document_saved_box_count
        old_saved_is_annotated = self._current_document_saved_is_annotated
        try:
            saved_path = self.annotation_io.save_annotation(
                target_path,
                self.current_document.boxes,
                self.current_document.image_size,
                self.class_manager,
                image_name=self.current_document.image_name,
            )
        except Exception as exc:
            self._error_dialog(
                "保存失败",
                "无法保存当前标注。",
                informative_text=str(exc),
                details=str(target_path),
            )
            return False

        if adopt_if_new or self.current_document.label_path is None:
            self.current_document = AnnotationDocument(
                self.current_document.image_path,
                saved_path,
                self.current_document.image_size,
                self.current_document.boxes,
                self.current_document.source_format,
            )
        else:
            self.current_document.label_path = self.current_document.label_path

        live_counts, live_box_count, live_is_annotated = self._capture_box_statistics(self.current_document.boxes)
        self._dataset_stats_baseline_counts = self._apply_stats_delta(
            self._dataset_stats_baseline_counts,
            old_saved_counts,
            live_counts,
        )
        self._dataset_stats_total_boxes = self._dataset_stats_total_boxes - old_saved_box_count + live_box_count
        self._dataset_stats_annotated_images = (
            self._dataset_stats_annotated_images - int(old_saved_is_annotated) + int(live_is_annotated)
        )
        self._store_current_document_saved_statistics(self.current_document.boxes)

        self.autosave.clear()
        self.history.clear_pending()
        self._refresh_all_views()
        self._update_window_title()
        self.status_message_label.setText("已保存")
        if show_message:
            self._set_status_message(f"已保存到 {saved_path}")
        return True

    def save_current_annotation(self, *, silent: bool = False) -> bool:
        if self.current_document is None:
            return False
        if self.current_document.label_path is None:
            default_format = str(self.export_format_combo.currentData() or "txt")
            target_path = self.annotation_io.default_export_path(self.current_document.image_path, default_format)
            adopt = True
        else:
            target_path = self.current_document.label_path
            adopt = True
        result = self._save_to_path(target_path, adopt_if_new=adopt, show_message=not silent)
        if result and not silent:
            self._set_status_message("保存成功")
        return result

    def _autosave_flush(self) -> bool:
        return self.save_current_annotation(silent=True)

    def toggle_autosave(self, enabled: bool) -> None:
        self.autosave.set_enabled(enabled)
        self.status_autosave_label.setText("自动保存 开" if enabled else "自动保存 关")
        self.action_autosave.blockSignals(True)
        self.action_autosave.setChecked(enabled)
        self.action_autosave.blockSignals(False)
        if hasattr(self, "autosave_checkbox"):
            self.autosave_checkbox.blockSignals(True)
            self.autosave_checkbox.setChecked(enabled)
            self.autosave_checkbox.blockSignals(False)
        self._autosave_enabled_pref = bool(enabled)
        QSettings().setValue("ui/autosave_enabled", bool(enabled))
        self._set_status_message("已开启自动保存" if enabled else "已关闭自动保存")

    def on_autosave_interval_changed(self, value: int) -> None:
        interval_seconds = max(1, int(value))
        self.autosave.set_interval_ms(interval_seconds * 1000)
        self._autosave_interval_seconds_pref = interval_seconds
        QSettings().setValue("ui/autosave_interval_seconds", interval_seconds)
        if self.autosave.dirty:
            self.autosave.mark_dirty()

    def apply_default_class_setting(self) -> None:
        text = self.default_class_input.text().strip() if hasattr(self, "default_class_input") else ""
        value = text or "object"
        self._pending_class_text = value
        if hasattr(self, "default_class_input") and self.default_class_input.text().strip() != value:
            with QSignalBlocker(self.default_class_input):
                self.default_class_input.setText(value)
        QSettings().setValue("ui/default_class_name", value)
        self._set_status_message(f"默认类别已更新为 {value}")

    def reset_general_settings(self) -> None:
        default_interval = max(1, int(DEFAULT_AUTOSAVE_SECONDS))
        self.toggle_autosave(True)
        self.autosave_interval_spin.setValue(default_interval)
        self.default_class_input.setText("object")
        self.apply_default_class_setting()

        self._shortcut_bindings = default_shortcut_bindings()
        save_shortcut_bindings(self._shortcut_bindings)
        self._apply_shortcuts_to_actions()

        if hasattr(self, "label_bg_alpha_slider"):
            self.label_bg_alpha_slider.setValue(140)
        if hasattr(self, "label_show_name_checkbox"):
            self.label_show_name_checkbox.setChecked(True)
        if hasattr(self, "label_show_id_checkbox"):
            self.label_show_id_checkbox.setChecked(False)

        self._set_status_message("已恢复默认设置")

    def set_draw_mode(self, enabled: bool) -> None:
        self.draw_mode = bool(enabled)
        if self.draw_mode:
            self.edit_mode = False
        if not self.draw_mode and not self.edit_mode:
            self.canvas.clear_interaction()
            self.history.clear_pending()
        self._sync_mode_state()
        if self.draw_mode:
            self.canvas.setFocus()

    def set_edit_mode(self, enabled: bool) -> None:
        self.edit_mode = bool(enabled)
        if self.edit_mode:
            self.draw_mode = False
        if not self.draw_mode and not self.edit_mode:
            self.canvas.clear_interaction()
            self.history.clear_pending()
        self._sync_mode_state()
        if self.edit_mode:
            self.canvas.setFocus()

    def _sync_mode_state(self) -> None:
        self.canvas.set_draw_shape(self.draw_shape)
        self.canvas.set_modes(edit_mode=self.edit_mode, draw_mode=self.draw_mode)
        if self.current_document is not None:
            self.canvas.set_selected_index(-1)
        self._update_mode_widgets()
        self._update_status_bar_values()
        self._update_window_title()

    def _update_mode_widgets(self) -> None:
        with QSignalBlocker(self.action_draw), QSignalBlocker(self.action_edit):
            self.action_draw.setChecked(self.draw_mode)
            self.action_edit.setChecked(self.edit_mode)

        self.btn_add_class.setEnabled(True)
        shape_label = "多边形" if self.draw_shape == "polygon" else "标注"
        self.action_draw.setText("取消绘制" if self.draw_mode else f"新建{shape_label}")
        self.action_edit.setText("取消编辑" if self.edit_mode else "编辑标注")
        if hasattr(self, "draw_shape_combo"):
            self.draw_shape_combo.setEnabled(True)
        self.btn_prev_image.setEnabled(self.dataset_service.image_count() > 0)
        self.btn_next_image.setEnabled(self.dataset_service.image_count() > 0)
        self.btn_open_dataset.setEnabled(True)
        self.btn_refresh_dataset.setEnabled(self.dataset_service.root_dir is not None)
        self.btn_prev_image.setEnabled(self.dataset_service.image_count() > 0)
        self.btn_next_image.setEnabled(self.dataset_service.image_count() > 0)
        has_class_selection = self.class_list_widget.currentItem() is not None
        self.btn_rename_class.setEnabled(has_class_selection)
        self.btn_delete_class.setEnabled(has_class_selection)
        self.btn_remap_class_id.setEnabled(has_class_selection)
        self.box_class_apply_button.setEnabled(self.current_document is not None)
        self.box_delete_button.setEnabled(self.current_document is not None)
        self._sync_history_actions()

    def toggle_sidebar(self) -> None:
        if self.sidebar_dock is None:
            return
        visible = not self.sidebar_dock.isVisible()
        self.sidebar_dock.setVisible(visible)
        self._status_bar().showMessage("控制台已显示" if visible else "控制台已隐藏", 1200)

    def _is_thumbnail_panel_visible(self) -> bool:
        return self.thumbnail_dock is not None and self.thumbnail_dock.isVisible()

    def _is_statistics_panel_visible(self) -> bool:
        return self.statistics_dock is not None and self.statistics_dock.isVisible()

    def _flush_deferred_panel_refresh(self) -> None:
        if self._is_thumbnail_panel_visible() and self._thumbnail_refresh_pending:
            self._refresh_thumbnail_list(force=True)
        if self._is_statistics_panel_visible() and self._statistics_refresh_pending:
            self._update_dataset_statistics_widget(force=True)

    def _on_dock_visibility_changed(self, _visible: bool) -> None:
        if self.thumbnail_dock is not None and not self.thumbnail_dock.isVisible():
            self._thumbnail_refresh_pending = True
            self._cancel_thumbnail_loading()
        # Delay to next event loop turn so Qt finalizes dock visibility first.
        QTimer.singleShot(0, self._refresh_dock_layout)
        QTimer.singleShot(0, self._flush_deferred_panel_refresh)

    def _refresh_dock_layout(self) -> None:
        if self.sidebar_dock is None or self.thumbnail_dock is None or self.statistics_dock is None:
            return

        if self.thumbnail_dock.isVisible() and self.statistics_dock.isVisible():
            self.splitDockWidget(self.thumbnail_dock, self.statistics_dock, Qt.Orientation.Vertical)

        if self.sidebar_dock.isVisible():
            self.sidebar_dock.raise_()
        if self.thumbnail_dock.isVisible():
            self.thumbnail_dock.raise_()
        if self.statistics_dock.isVisible():
            self.statistics_dock.raise_()

    def reset_panel_layout(self) -> None:
        if self.sidebar_dock is None or self.thumbnail_dock is None or self.statistics_dock is None:
            return

        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.sidebar_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.thumbnail_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.statistics_dock)
        self.splitDockWidget(self.thumbnail_dock, self.statistics_dock, Qt.Orientation.Vertical)
        self.sidebar_dock.show()
        self.thumbnail_dock.show()
        self.statistics_dock.show()
        self._status_bar().showMessage("面板布局已重置", 1200)

    def zoom_in(self) -> None:
        self.set_zoom_scale(self.current_zoom_scale * ZOOM_STEP)

    def zoom_out(self) -> None:
        self.set_zoom_scale(self.current_zoom_scale / ZOOM_STEP)

    def fit_to_window(self) -> None:
        if self.current_document is None or self.current_image is None:
            return
        viewport_widget = self.scroll_area.viewport()
        assert viewport_widget is not None
        viewport = viewport_widget.size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        image_width = self.current_document.image_width
        image_height = self.current_document.image_height
        if image_width <= 0 or image_height <= 0:
            return
        fitted = min(viewport.width() / image_width, viewport.height() / image_height)
        self.set_zoom_scale(fitted)

    def set_zoom_scale(self, scale: float) -> None:
        scale = self._clamp_zoom(scale)
        if abs(scale - self.current_zoom_scale) < 1e-6:
            return
        self.current_zoom_scale = scale
        self.canvas.set_scale(scale)
        self.canvas.update()
        self._update_zoom_from_document()
        self._set_status_message(f"缩放已调整到 {int(round(scale * 100))}%")

    def _update_zoom_from_document(self) -> None:
        self.status_zoom_label.setText(f"缩放 {int(round(self.current_zoom_scale * 100))}%")
        self.canvas.set_scale(self.current_zoom_scale)

    def _clamp_zoom(self, scale: float) -> float:
        return max(MIN_ZOOM_SCALE, min(MAX_ZOOM_SCALE, float(scale)))

    def _mark_document_dirty(self) -> None:
        self.autosave.mark_dirty()
        self.status_message_label.setText("未保存更改")
        self._update_window_title()

    def _set_status_message(self, message: str, timeout_ms: int = 2500) -> None:
        self.status_message_label.setText(message)
        self._status_bar().showMessage(message, timeout_ms)

    def _status_bar(self) -> QStatusBar:
        status_bar = self.statusBar()
        assert status_bar is not None
        return status_bar

    def _update_status_bar_values(self) -> None:
        draw_shape_label = "多边形" if self.draw_shape == "polygon" else "矩形"
        self.status_mode_label.setText(
            f"模式 {'绘制' + draw_shape_label if self.draw_mode else '编辑' if self.edit_mode else '浏览'}"
        )
        self.status_autosave_label.setText(
            "自动保存 开" if self.autosave.enabled else "自动保存 关"
        )
        self.status_zoom_label.setText(f"缩放 {int(round(self.current_zoom_scale * 100))}%")
        if self.current_document is None:
            self.status_box_label.setText("标注 0")
        else:
            self.status_box_label.setText(f"标注 {len(self.current_document.boxes)}")
        self._update_selected_box_size_status()

    def _update_window_title(self) -> None:
        dirty_suffix = " *" if self.autosave.dirty else ""
        if self.current_document is None:
            title = f"{APP_NAME}{dirty_suffix}"
        else:
            title = f"{APP_NAME} - {self.current_document.image_name}{dirty_suffix}"
        self.setWindowTitle(title)

    def _set_loaded_state(self, loaded: bool) -> None:
        for widget in (
            self.action_save,
            self.action_export,
            self.action_prev,
            self.action_next,
            self.action_draw,
            self.action_edit,
            self.action_zoom_in,
            self.action_zoom_out,
            self.action_fit,
        ):
            widget.setEnabled(loaded)
        for button in (
            self.btn_prev_image,
            self.btn_next_image,
            self.btn_delete_dataset_items,
            self.btn_rename_class,
            self.box_class_apply_button,
            self.box_delete_button,
        ):
            button.setEnabled(loaded)
        self.btn_refresh_dataset.setEnabled(self.dataset_service.root_dir is not None)
        self.export_current_button.setEnabled(loaded)
        self.export_batch_button.setEnabled(loaded)
        self.export_resize_button.setEnabled(loaded)
        self.export_resize_width_spin.setEnabled(loaded)
        self.export_resize_height_spin.setEnabled(loaded)
        self.export_resize_letterbox_checkbox.setEnabled(loaded)
        self.export_merge_button.setEnabled(True)
        self.export_splitter_button.setEnabled(True)
        self.export_extract_button.setEnabled(True)
        self.export_extract_ratio_spin.setEnabled(True)
        self.export_extract_seed_spin.setEnabled(True)
        self.box_class_combo.setEnabled(loaded)
        self.box_list_widget.setEnabled(loaded)
        self.image_list_widget.setEnabled(loaded)
        self.thumbnail_list_widget.setEnabled(loaded)
        self.class_list_widget.setEnabled(True)
        self.action_autosave.setEnabled(True)
        self.action_toggle_sidebar.setEnabled(True)
        can_document_undo = loaded and self.history.can_undo()
        can_document_redo = loaded and self.history.can_redo()
        self.action_undo.setEnabled(can_document_undo or bool(self._dataset_delete_undo_stack))
        self.action_redo.setEnabled(can_document_redo or bool(self._dataset_delete_redo_stack))

    def _clear_document(self) -> None:
        self.current_document = None
        self.current_image = None
        self.current_image_index = -1
        self.canvas.set_document(None, [], self.current_zoom_scale)
        self.canvas.set_modes(edit_mode=False, draw_mode=False)
        self.canvas.set_selected_index(-1)
        self.autosave.clear()
        self.history.clear()
        self.draw_mode = False
        self.edit_mode = False
        self.selected_info_label.setText("未选中")
        self.current_file_label.setText("--")
        self.status_box_label.setText("标注 0")
        self.status_box_size_label.setText("尺寸 --")
        self._refresh_all_views()
        self._set_loaded_state(False)
        self._update_window_title()

    def _prepare_for_context_change(self) -> bool:
        if self.current_document is None or not self.autosave.dirty:
            return True
        if self.autosave.enabled:
            if self.autosave.flush():
                return True
            self._warning_dialog("自动保存失败", "自动保存失败，请先手动保存。")
            return False

        reply = self._ask_dialog(
            "未保存更改",
            "当前标注有未保存更改，离开前是否先保存？",
            buttons=(
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            ),
            default_button=QMessageBox.StandardButton.Save,
            button_texts={
                QMessageBox.StandardButton.Save: "保存并继续",
                QMessageBox.StandardButton.Discard: "不保存继续",
                QMessageBox.StandardButton.Cancel: "取消",
            },
        )
        if reply == QMessageBox.StandardButton.Save:
            return self.save_current_annotation()
        if reply == QMessageBox.StandardButton.Discard:
            self.autosave.clear()
            return True
        return False

    def _selected_box_exists(self) -> bool:
        return self.current_document is not None and 0 <= self.canvas.selected_index < len(self.current_document.boxes)

    @property
    def current_selected_box(self) -> Box:
        if not self._selected_box_exists():
            raise IndexError("No selected box")
        document = self.current_document
        assert document is not None
        return document.boxes[self.canvas.selected_index]

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        if a0 is None:
            return
        if not self._prepare_for_context_change():
            a0.ignore()
            return
        self._cancel_dataset_statistics_loading()
        self._cancel_thumbnail_loading()
        a0.accept()
