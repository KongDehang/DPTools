from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QKeySequence


@dataclass(frozen=True, slots=True)
class ShortcutDefinition:
    key: str
    title: str
    default: str
    group: str


SHORTCUT_DEFINITIONS: tuple[ShortcutDefinition, ...] = (
    ShortcutDefinition("undo", "撤销", "Ctrl+Z", "编辑操作"),
    ShortcutDefinition("redo", "重做", "Ctrl+Y", "编辑操作"),
    ShortcutDefinition("redo_alt", "重做（备用）", "Ctrl+Shift+Z", "编辑操作"),
    ShortcutDefinition("save_annotation", "保存当前标注", "Ctrl+S", "文件操作"),
    ShortcutDefinition("export_annotation", "导出当前标注", "Ctrl+Shift+S", "文件操作"),
    ShortcutDefinition("refresh_dataset", "刷新数据集", "F5", "文件操作"),
    ShortcutDefinition("prev_image", "上一张图", "W", "导航"),
    ShortcutDefinition("next_image", "下一张图", "S", "导航"),
    ShortcutDefinition("prev_image_alt", "上一张图（备用）", "Alt+Left", "导航"),
    ShortcutDefinition("next_image_alt", "下一张图（备用）", "Alt+Right", "导航"),
    ShortcutDefinition("toggle_draw_mode", "切换绘制模式", "N", "标注操作"),
    ShortcutDefinition("toggle_edit_mode", "切换编辑模式", "E", "标注操作"),
    ShortcutDefinition("rename_box", "重命名选中框", "R", "标注操作"),
    ShortcutDefinition("rename_box_alt", "重命名选中框（备用）", "F2", "标注操作"),
    ShortcutDefinition("delete_selection", "删除（框/数据项）", "Delete", "标注操作"),
)

SHORTCUT_DEFINITIONS_BY_KEY: dict[str, ShortcutDefinition] = {
    item.key: item for item in SHORTCUT_DEFINITIONS
}


def normalize_shortcut(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    sequence = QKeySequence.fromString(text, QKeySequence.SequenceFormat.PortableText)
    if sequence.isEmpty():
        sequence = QKeySequence.fromString(text, QKeySequence.SequenceFormat.NativeText)
    if sequence.isEmpty():
        sequence = QKeySequence(text)
    if sequence.isEmpty():
        return ""

    return sequence.toString(QKeySequence.SequenceFormat.PortableText)


def to_key_sequence(value: str | None) -> QKeySequence:
    normalized = normalize_shortcut(value)
    if not normalized:
        return QKeySequence()
    return QKeySequence.fromString(normalized, QKeySequence.SequenceFormat.PortableText)


def format_shortcut_for_display(value: str | None) -> str:
    sequence = to_key_sequence(value)
    if sequence.isEmpty():
        return ""
    return sequence.toString(QKeySequence.SequenceFormat.NativeText)


def default_shortcut_bindings() -> dict[str, str]:
    return {item.key: normalize_shortcut(item.default) for item in SHORTCUT_DEFINITIONS}


def load_shortcut_bindings(settings: QSettings | None = None) -> dict[str, str]:
    store = settings or QSettings()
    bindings = default_shortcut_bindings()
    for item in SHORTCUT_DEFINITIONS:
        raw_value = store.value(f"ui/shortcuts/{item.key}", bindings[item.key])
        normalized = normalize_shortcut(str(raw_value) if raw_value is not None else "")
        bindings[item.key] = normalized if normalized else bindings[item.key]
    return bindings


def save_shortcut_bindings(bindings: dict[str, str], settings: QSettings | None = None) -> None:
    store = settings or QSettings()
    defaults = default_shortcut_bindings()
    for item in SHORTCUT_DEFINITIONS:
        value = normalize_shortcut(bindings.get(item.key, defaults[item.key]))
        store.setValue(f"ui/shortcuts/{item.key}", value)


def find_shortcut_conflicts(bindings: dict[str, str]) -> dict[str, list[str]]:
    sequence_to_keys: dict[str, list[str]] = {}
    defaults = default_shortcut_bindings()

    for item in SHORTCUT_DEFINITIONS:
        text = normalize_shortcut(bindings.get(item.key, defaults[item.key]))
        if not text:
            continue
        sequence_to_keys.setdefault(text, []).append(item.key)

    return {sequence: keys for sequence, keys in sequence_to_keys.items() if len(keys) > 1}
