from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)


class ClassSelectorDialog(QDialog):
    def __init__(
        self,
        class_names: list[str],
        default_text: str = "",
        parent=None,
        title: str = "选择类别",
        prompt: str = "选择或输入类别名称：",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        prompt_label = QLabel(prompt, self)
        prompt_label.setWordWrap(True)
        root_layout.addWidget(prompt_label)

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(prompt_label.alignment())
        root_layout.addLayout(form_layout)

        self.class_combo = QComboBox(self)
        self.class_combo.setEditable(True)
        self.class_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        unique_names: list[str] = []
        for name in class_names:
            cleaned = str(name).strip()
            if cleaned and cleaned not in unique_names:
                unique_names.append(cleaned)
        self.class_combo.addItems(unique_names)
        if default_text:
            index = self.class_combo.findText(default_text)
            if index >= 0:
                self.class_combo.setCurrentIndex(index)
            else:
                self.class_combo.setEditText(default_text)
        form_layout.addRow("类别", self.class_combo)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        root_layout.addWidget(button_box)

    def selected_text(self) -> str:
        return self.class_combo.currentText().strip()

    @classmethod
    def get_class_name(
        cls,
        parent,
        class_names: list[str],
        default_text: str = "",
        title: str = "选择类别",
        prompt: str = "选择或输入类别名称：",
    ) -> tuple[str, bool]:
        dialog = cls(class_names, default_text=default_text, parent=parent, title=title, prompt=prompt)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        return (dialog.selected_text() if accepted else "", accepted)
