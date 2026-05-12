from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


class ClassChecklistDialog(QDialog):
    def __init__(
        self,
        class_items: list[tuple[int, str]],
        *,
        default_checked_ids: set[int] | None = None,
        parent=None,
        title: str = "选择抽取类别",
        prompt: str = "勾选要抽取的类别：",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        self.setMinimumHeight(420)

        icons_dir = Path(__file__).resolve().parent.parent / "assets" / "icons"
        checkmark_uri = (icons_dir / "checkmark.png").resolve().as_posix()
        stylesheet = """
QDialog {
    background: #161d26;
    color: #e7edf5;
}

QLabel {
    color: #e7edf5;
}

QListWidget {
    background: #121923;
    color: #e7edf5;
    border: 1px solid #314052;
    border-radius: 0px;
    padding: 2px;
}

QListWidget::item {
    background: transparent;
    color: #e7edf5;
    min-height: 26px;
    padding: 4px 8px;
}

QListWidget::item:hover {
    background: #1a2430;
}

QListWidget::item:selected {
    background: #2a4f74;
    color: #ffffff;
}

QListWidget::indicator {
    width: 14px;
    height: 14px;
    background: transparent;
    border: 1px solid #8ea0b6;
    border-radius: 0px;
}

QListWidget::indicator:hover {
    border-color: #4da3ff;
}

QListWidget::indicator:checked {
    background: transparent;
    border: 1px solid #4da3ff;
    image: url("__CHECKMARK__");
}

QListWidget::indicator:unchecked {
    image: none;
}

QPushButton {
    background: #1e2834;
    color: #e7edf5;
    border: 1px solid #314052;
    border-radius: 8px;
    padding: 7px 12px;
    min-height: 18px;
}

QPushButton:hover {
    background: #263241;
    border-color: #4da3ff;
}

QPushButton:pressed {
    background: #14202d;
}

QDialogButtonBox QPushButton {
    min-width: 84px;
}
""".replace("__CHECKMARK__", checkmark_uri)
        self.setStyleSheet(stylesheet)

        self._class_items = [(int(class_id), str(class_name).strip() or f"ID {int(class_id)}") for class_id, class_name in class_items]
        self._default_checked_ids = {int(class_id) for class_id in default_checked_ids} if default_checked_ids else None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        prompt_label = QLabel(prompt, self)
        prompt_label.setWordWrap(True)
        root_layout.addWidget(prompt_label)

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list_widget.setAlternatingRowColors(False)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setMinimumHeight(240)
        root_layout.addWidget(self.list_widget, 1)

        with QSignalBlocker(self.list_widget):
            for class_id, class_name in self._class_items:
                item = QListWidgetItem(f"ID {class_id} | {class_name}")
                item.setData(Qt.ItemDataRole.UserRole, class_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                if self._default_checked_ids is None or class_id in self._default_checked_ids:
                    item.setCheckState(Qt.CheckState.Checked)
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)
                self.list_widget.addItem(item)

        self.list_widget.itemChanged.connect(self._update_summary)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(8)
        select_all_button = QPushButton("全选", self)
        select_all_button.clicked.connect(self.select_all)
        clear_button = QPushButton("全不选", self)
        clear_button.clicked.connect(self.clear_all)
        controls_row.addWidget(select_all_button)
        controls_row.addWidget(clear_button)
        controls_row.addStretch(1)
        root_layout.addLayout(controls_row)

        self.summary_label = QLabel(self)
        self.summary_label.setObjectName("mutedLabel")
        self.summary_label.setWordWrap(True)
        root_layout.addWidget(self.summary_label)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        root_layout.addWidget(button_box)

        self._update_summary()

    def _update_summary(self) -> None:
        checked_count = len(self.selected_class_ids())
        total_count = self.list_widget.count()
        self.summary_label.setText(f"已选 {checked_count}/{total_count} 个类别")

    def select_all(self) -> None:
        with QSignalBlocker(self.list_widget):
            for row in range(self.list_widget.count()):
                item = self.list_widget.item(row)
                if item is not None:
                    item.setCheckState(Qt.CheckState.Checked)
        self._update_summary()

    def clear_all(self) -> None:
        with QSignalBlocker(self.list_widget):
            for row in range(self.list_widget.count()):
                item = self.list_widget.item(row)
                if item is not None:
                    item.setCheckState(Qt.CheckState.Unchecked)
        self._update_summary()

    def selected_class_ids(self) -> list[int]:
        selected_ids: list[int] = []
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is None:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                class_id = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(class_id, int):
                    selected_ids.append(int(class_id))
        return selected_ids

    @classmethod
    def get_selected_class_ids(
        cls,
        parent,
        class_items: list[tuple[int, str]],
        *,
        default_checked_ids: set[int] | None = None,
        title: str = "选择抽取类别",
        prompt: str = "勾选要抽取的类别：",
    ) -> tuple[list[int], bool]:
        dialog = cls(
            class_items,
            default_checked_ids=default_checked_ids,
            parent=parent,
            title=title,
            prompt=prompt,
        )
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        return (dialog.selected_class_ids() if accepted else [], accepted)