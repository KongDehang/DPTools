from __future__ import annotations

import numpy as np

APP_NAME = "LabelBox"
APP_ORG = "GitHubCopilot"
DEFAULT_WINDOW_TITLE = "LabelBox"

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

LABEL_EXTENSIONS = {".json", ".xml", ".txt"}
MASK_LABEL_EXTENSIONS = {".png"}
ALL_LABEL_EXTENSIONS = LABEL_EXTENSIONS | MASK_LABEL_EXTENSIONS
ANNOTATION_FORMATS = ("json", "xml", "txt", "mask_png")
FORMAT_SUFFIX = {"json": ".json", "xml": ".xml", "txt": ".txt", "mask_png": ".png"}
FORMAT_LABELS = {
    "json": "JSON / LabelMe",
    "xml": "XML / Pascal VOC",
    "txt": "TXT / YOLO",
    "mask_png": "Mask PNG / Semantic",
}
DEFAULT_AUTOSAVE_SECONDS = 2.5
DEFAULT_ZOOM_SCALE = 1.0
MIN_ZOOM_SCALE = 0.10
MAX_ZOOM_SCALE = 8.0
ZOOM_STEP = 1.15
HANDLE_SIZE = 8
CROSSHAIR_COLOR = (255, 255, 255, 140)
CANVAS_BACKGROUND = "#0f141a"
SIDEBAR_BACKGROUND = "#161d26"
PANEL_BACKGROUND = "#1c2530"
PANEL_BORDER = "#2b3644"
TEXT_PRIMARY = "#e7edf5"
TEXT_SECONDARY = "#9db0c4"
ACCENT = "#4da3ff"
ACCENT_HOVER = "#66b2ff"
SUCCESS = "#31c48d"
WARNING = "#f5a524"
ERROR = "#ef6b6b"


def build_palette(count: int = 128) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(42)
    colors: list[tuple[int, int, int]] = []
    for _ in range(count):
        channel_values = rng.integers(72, 248, size=3)
        colors.append(tuple(int(value) for value in channel_values))
    return colors


PALETTE = build_palette()

STYLE_SHEET = """
QMainWindow {
    background: #0f141a;
    color: #e7edf5;
}

QWidget {
    color: #e7edf5;
}

QMenuBar {
    background: #111821;
    color: #e7edf5;
    border-bottom: 1px solid #2b3644;
}

QMenuBar::item {
    padding: 6px 10px;
    background: transparent;
}

QMenuBar::item:selected {
    background: #1f2935;
    border-radius: 6px;
}

QMenu {
    background: #161d26;
    color: #e7edf5;
    border: 1px solid #2b3644;
}

QMenu::item {
    padding: 6px 28px 6px 18px;
}

QMenu::item:selected {
    background: #2a3b50;
}

QToolBar {
    background: #111821;
    border-bottom: 1px solid #2b3644;
    spacing: 6px;
    padding: 6px;
}

QDockWidget {
    border: 1px solid #33495f;
    background: #121a24;
}

QDockWidget::title {
    background: #101722;
    color: #e7edf5;
    text-align: left;
    padding: 7px 10px;
    border-bottom: 1px solid #33495f;
    font-weight: 600;
}

QToolButton,
QPushButton {
    background: #1e2834;
    color: #e7edf5;
    border: 1px solid #314052;
    border-radius: 8px;
    padding: 7px 12px;
    min-height: 18px;
}

QToolButton:hover,
QPushButton:hover {
    background: #263241;
    border-color: #4da3ff;
}

QToolButton:pressed,
QPushButton:pressed {
    background: #14202d;
}

QToolButton:checked,
QPushButton:checked {
    background: #22456a;
    border-color: #4da3ff;
    color: #ffffff;
}

QPushButton#primaryButton {
    background: #4da3ff;
    border-color: #4da3ff;
    color: #0f141a;
    font-weight: 600;
}

QPushButton#primaryButton:hover {
    background: #66b2ff;
    border-color: #66b2ff;
}

QPushButton#dangerButton {
    background: #3a2020;
    border-color: #5b2d2d;
    color: #ffdada;
}

QPushButton#dangerButton:hover {
    background: #4c2727;
    border-color: #ef6b6b;
}

QFrame#surfaceCard,
QFrame#sectionCard,
QFrame#canvasCard {
    background: #161d26;
    border: 1px solid #2b3644;
    border-radius: 12px;
}

QLabel {
    color: #e7edf5;
}

QLabel#mutedLabel {
    color: #9db0c4;
}

QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox,
QListWidget,
QTextEdit,
QPlainTextEdit {
    background: #121923;
    color: #e7edf5;
    border: 1px solid #314052;
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: #4da3ff;
    selection-color: #0f141a;
}

QLineEdit::placeholder {
    color: #7f93a9;
}

QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QListWidget:focus,
QTextEdit:focus,
QPlainTextEdit:focus {
    border: 1px solid #4da3ff;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox::down-arrow {
    width: 12px;
    height: 12px;
    image: url("__ICON_CHEVRON_DOWN__");
}

QComboBox::down-arrow:disabled {
    image: url("__ICON_CHEVRON_DOWN__");
}

QComboBox QAbstractItemView {
    background: #121923;
    color: #e7edf5;
    border: 1px solid #314052;
    selection-background-color: #2a4f74;
    selection-color: #ffffff;
    outline: 0;
}

QComboBox QAbstractItemView::item {
    min-height: 24px;
    padding: 4px 8px;
}

QAbstractItemView {
    background: #121923;
    color: #e7edf5;
    border: 1px solid #314052;
    selection-background-color: #2a4f74;
    selection-color: #ffffff;
}

QAbstractItemView::item {
    color: #e7edf5;
}

QAbstractItemView::item:selected {
    color: #ffffff;
}

QCheckBox {
    color: #e7edf5;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #4a637a;
    border-radius: 4px;
    background: #121923;
}

QCheckBox::indicator:checked {
    background: #3f84c4;
    border-color: #7fc0ff;
    image: url("__ICON_CHECK__");
}

QCheckBox::indicator:checked:hover {
    background: #4c90ce;
}

QComboBox:disabled,
QLineEdit:disabled,
QSpinBox:disabled,
QListWidget:disabled {
    color: #7d90a6;
    border-color: #273749;
    background: #0f151d;
}

QCheckBox:disabled,
QLabel:disabled {
    color: #7d90a6;
}

QListWidget::item {
    padding: 8px 10px;
    border-radius: 6px;
}

QListWidget::item:selected {
    background: #22456a;
    color: #ffffff;
}

QTabWidget::pane {
    border: 1px solid #3a5269;
    background: #161d26;
    border-radius: 12px;
    top: -1px;
}

QTabBar::tab {
    background: #101722;
    color: #c2d2e3;
    border: 1px solid #354c62;
    border-bottom: none;
    padding: 8px 14px;
    margin-right: 4px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    min-width: 80px;
    font-weight: 600;
}

QTabBar::tab:selected {
    background: #172233;
    color: #f4f8ff;
    border-color: #4da3ff;
}

QTabBar::tab:hover:!selected {
    background: #152131;
    color: #e2ecf7;
}

QTabBar::scroller {
    width: 0px;
}

QSplitter::handle {
    background: #182332;
}

QSplitter::handle:vertical {
    height: 8px;
    border-top: 1px solid #2d4459;
    border-bottom: 1px solid #2d4459;
}

QSplitter::handle:horizontal {
    width: 8px;
    border-left: 1px solid #2d4459;
    border-right: 1px solid #2d4459;
}

QSplitter::handle:hover {
    background: #24405c;
}

QScrollArea {
    border: none;
    background: transparent;
}

QStatusBar {
    background: #111821;
    color: #e7edf5;
    border-top: 1px solid #2b3644;
}

QStatusBar QLabel {
    color: #e7edf5;
    padding: 0 4px;
}

QMessageBox {
    background: #161d26;
}

QMessageBox QLabel {
    color: #e7edf5;
    min-width: 320px;
}

QMessageBox QPushButton {
    min-width: 106px;
    padding: 6px 12px;
}

QMessageBox QPlainTextEdit,
QMessageBox QTextEdit {
    background: #121923;
    color: #dce8f6;
    border: 1px solid #33495f;
    border-radius: 6px;
}

QToolButton#sectionHeader {
    background: transparent;
    border: none;
    padding: 4px 2px;
    color: #e7edf5;
    font-weight: 600;
}

QToolButton#sectionHeader:hover {
    color: #66b2ff;
    background: transparent;
}

QToolButton#sectionHeader:checked {
    color: #f2f8ff;
}

QToolButton#sectionHeader:!checked {
    color: #b9c9da;
}

QFrame#canvasCard {
    background: #0f141a;
}

QScrollBar:vertical {
    background: #111821;
    width: 12px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #314052;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #4da3ff;
}

QScrollBar:horizontal {
    background: #111821;
    height: 12px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #314052;
    border-radius: 5px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: #4da3ff;
}
"""
