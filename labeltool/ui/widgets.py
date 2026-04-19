from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QFrame):
    toggled = pyqtSignal(bool)

    def __init__(self, title: str, expanded: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sectionCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(48)
        self._title = title

        self.toggle_button = QToolButton(self)
        self.toggle_button.setObjectName("sectionHeader")
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.toggle_button.clicked.connect(self._sync_state)
        self._update_toggle_label(expanded)

        self.content_widget = QWidget(self)
        self.content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.content_widget.setMinimumHeight(0)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        self.content_widget.setVisible(expanded)
        self.content_widget.setMaximumHeight(16777215 if expanded else 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_widget)

    def _sync_state(self, checked: bool) -> None:
        self.content_widget.setVisible(checked)
        self.content_widget.setMaximumHeight(16777215 if checked else 0)
        self._update_toggle_label(checked)
        self.updateGeometry()
        self.toggled.emit(checked)

    def _update_toggle_label(self, expanded: bool) -> None:
        marker = "▼" if expanded else "▶"
        self.toggle_button.setText(f"{marker} {self._title}")

    def set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setChecked(expanded)
        self._sync_state(expanded)

    def is_expanded(self) -> bool:
        return bool(self.toggle_button.isChecked())

    def collapsed_hint_height(self) -> int:
        margins = self.layout().contentsMargins()
        return margins.top() + self.toggle_button.sizeHint().height() + margins.bottom()

    def add_widget(self, widget: QWidget) -> None:
        self.content_layout.addWidget(widget)

    def content_layout_ref(self) -> QVBoxLayout:
        return self.content_layout
