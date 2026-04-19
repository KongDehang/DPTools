from __future__ import annotations

from collections import defaultdict

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..services.shortcut_manager import (
    SHORTCUT_DEFINITIONS,
    SHORTCUT_DEFINITIONS_BY_KEY,
    default_shortcut_bindings,
    find_shortcut_conflicts,
    format_shortcut_for_display,
    normalize_shortcut,
    to_key_sequence,
)


class ShortcutEditorDialog(QDialog):
    def __init__(self, bindings: dict[str, str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("shortcutEditorDialog")
        self.setWindowTitle("快捷键设置")
        self.setModal(True)
        self.resize(620, 620)
        self._apply_dialog_style()

        self._defaults = default_shortcut_bindings()
        self._bindings = {**self._defaults, **{key: normalize_shortcut(value) for key, value in bindings.items()}}
        self._editors: dict[str, QKeySequenceEdit] = {}

        self._build_ui()
        self._refresh_conflict_hint()

    def _apply_dialog_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog#shortcutEditorDialog {
                background: #edf2f7;
            }
            QDialog#shortcutEditorDialog QLabel {
                color: #2a3d52;
            }
            QDialog#shortcutEditorDialog QLabel#mutedLabel {
                color: #5a6e84;
            }
            QDialog#shortcutEditorDialog QGroupBox {
                color: #2a3d52;
                border: 1px solid #cad5e1;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QDialog#shortcutEditorDialog QGroupBox::title {
                color: #3a536d;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
            """
        )

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        intro = QLabel("可为常用功能自定义快捷键。若与其他功能冲突，需先调整后才能保存。")
        intro.setWordWrap(True)
        intro.setObjectName("mutedLabel")
        layout.addWidget(intro)

        groups: dict[str, list[str]] = defaultdict(list)
        for definition in SHORTCUT_DEFINITIONS:
            groups[definition.group].append(definition.key)

        for group_name in ("文件操作", "导航", "标注操作", "编辑操作"):
            keys = groups.get(group_name)
            if not keys:
                continue

            group_box = QGroupBox(group_name)
            form = QFormLayout(group_box)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
            form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
            form.setVerticalSpacing(8)

            for command_key in keys:
                definition = SHORTCUT_DEFINITIONS_BY_KEY[command_key]
                editor = QKeySequenceEdit()
                editor.setKeySequence(to_key_sequence(self._bindings.get(command_key, definition.default)))
                editor.keySequenceChanged.connect(lambda _seq, key=command_key: self._on_shortcut_changed(key))
                self._editors[command_key] = editor

                reset_button = QPushButton("默认")
                reset_button.clicked.connect(lambda _checked=False, key=command_key: self._reset_single(key))

                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(6)
                row.addWidget(editor, 1)
                row.addWidget(reset_button)
                form.addRow(definition.title, row)

            layout.addWidget(group_box)

        self.conflict_hint_label = QLabel("")
        self.conflict_hint_label.setWordWrap(True)
        self.conflict_hint_label.setObjectName("mutedLabel")
        layout.addWidget(self.conflict_hint_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.restore_all_button = self.button_box.addButton("恢复全部默认", QDialogButtonBox.ButtonRole.ResetRole)
        self.restore_all_button.clicked.connect(self._restore_all_defaults)
        self.button_box.accepted.connect(self._accept_if_valid)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _on_shortcut_changed(self, command_key: str) -> None:
        editor = self._editors[command_key]
        text = editor.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
        self._bindings[command_key] = normalize_shortcut(text)
        self._refresh_conflict_hint()

    def _reset_single(self, command_key: str) -> None:
        editor = self._editors[command_key]
        default_text = self._defaults.get(command_key, "")
        editor.setKeySequence(to_key_sequence(default_text))
        self._bindings[command_key] = normalize_shortcut(default_text)
        self._refresh_conflict_hint()

    def _restore_all_defaults(self) -> None:
        for definition in SHORTCUT_DEFINITIONS:
            self._reset_single(definition.key)

    def _collect_bindings(self) -> dict[str, str]:
        collected = dict(self._defaults)
        for definition in SHORTCUT_DEFINITIONS:
            editor = self._editors[definition.key]
            text = editor.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
            collected[definition.key] = normalize_shortcut(text)
        return collected

    def _refresh_conflict_hint(self) -> None:
        conflicts = find_shortcut_conflicts(self._collect_bindings())
        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)

        if not conflicts:
            self.conflict_hint_label.setText("无快捷键冲突。")
            self.conflict_hint_label.setStyleSheet("color: #87d4a0;")
            if ok_button is not None:
                ok_button.setEnabled(True)
            return

        lines: list[str] = []
        for sequence, keys in conflicts.items():
            display_sequence = format_shortcut_for_display(sequence) or sequence
            names = "、".join(SHORTCUT_DEFINITIONS_BY_KEY[key].title for key in keys)
            lines.append(f"{display_sequence}: {names}")

        self.conflict_hint_label.setText("检测到冲突，请调整后再保存：\n" + "\n".join(lines))
        self.conflict_hint_label.setStyleSheet("color: #ff8f8f;")
        if ok_button is not None:
            ok_button.setEnabled(False)

    def _accept_if_valid(self) -> None:
        conflicts = find_shortcut_conflicts(self._collect_bindings())
        if conflicts:
            return
        self._bindings = self._collect_bindings()
        self.accept()

    def shortcut_bindings(self) -> dict[str, str]:
        return self._collect_bindings()
