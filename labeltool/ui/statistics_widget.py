from __future__ import annotations

import math
import re
from dataclasses import dataclass
from math import fsum

from PyQt6.QtCore import QPointF, QRectF, Qt, QSize, QTimer
from PyQt6.QtGui import QColor, QFontMetrics, QMouseEvent, QPainter, QPen, QWheelEvent
from PyQt6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..constants import PALETTE


@dataclass(slots=True)
class StatisticsItem:
    class_id: int | None
    class_name: str
    count: int
    color: QColor


class DatasetStatisticsWidget(QFrame):
    CALL_OUT_THRESHOLD = 8
    PIE_MAX_ITEMS = 12

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sectionCard")
        self.setMinimumHeight(420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._raw_counts: dict[str, int] = {}
        self._class_name_to_id: dict[str, int] = {}
        self._items: list[StatisticsItem] = []
        self._total_images = 0
        self._annotated_images = 0
        self._total_boxes = 0
        self._loading_text = ""
        self._sort_mode = "id"
        self._bar_panel_rect = QRectF()
        self._bar_visible_rows = 0
        self._bar_scroll_index = 0
        self._bar_panel_ratio = 0.42
        self._splitter_handle_height = 10.0
        self._charts_rect = QRectF()
        self._splitter_handle_rect = QRectF()
        self._splitter_hover = False
        self._splitter_dragging = False
        self._pie_panel_rect = QRectF()
        self._pie_chart_rect = QRectF()
        self._pie_slices: list[tuple[StatisticsItem, int, int, float]] = []
        self._pie_slice_progress: list[float] = []
        self._pie_hover_index = -1
        self._pie_hover_anchor = QPointF()
        self._pie_hover_timer = QTimer(self)
        self._pie_hover_timer.setInterval(16)
        self._pie_hover_timer.timeout.connect(self._advance_pie_hover_animation)
        self._bar_collapsed = False
        self._pie_collapsed = False
        self._bar_header_rect = QRectF()
        self._pie_header_rect = QRectF()
        self._panel_header_height = 34.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        controls = QWidget(self)
        controls.setFixedHeight(36)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        sort_label = QLabel("排序")
        sort_label.setObjectName("mutedLabel")
        self.sort_combo = QComboBox(controls)
        self.sort_combo.addItem("按 ID", "id")
        self.sort_combo.addItem("数量高 -> 低", "count_desc")
        self.sort_combo.addItem("数量低 -> 高", "count_asc")
        self.sort_combo.currentIndexChanged.connect(self._on_sort_mode_changed)
        controls_layout.addWidget(sort_label)
        controls_layout.addWidget(self.sort_combo)
        controls_layout.addStretch(1)
        layout.addWidget(controls)
        layout.addStretch(1)
        self._controls_widget = controls

    def sizeHint(self) -> QSize:  # pragma: no cover - Qt hint
        return QSize(560, 640)

    def set_class_mapping(self, name_to_id: dict[str, int]) -> None:
        self._class_name_to_id = {str(name): int(class_id) for name, class_id in name_to_id.items()}
        self._rebuild_items()

    def set_loading(self, message: str = "统计中...") -> None:
        self._loading_text = message
        self._raw_counts = {}
        self._items = []
        self._total_images = 0
        self._annotated_images = 0
        self._total_boxes = 0
        self._bar_scroll_index = 0
        self._reset_chart_regions()
        self.update()

    def set_statistics(
        self,
        class_counts: dict[str, int],
        total_images: int,
        annotated_images: int,
        total_boxes: int,
    ) -> None:
        self._raw_counts = {str(name): int(count) for name, count in class_counts.items() if int(count) > 0}
        self._total_images = int(total_images)
        self._annotated_images = int(annotated_images)
        self._total_boxes = int(total_boxes)
        self._loading_text = ""
        self._rebuild_items()

    def wheelEvent(self, event: QWheelEvent) -> None:  # pragma: no cover - UI interaction
        if (
            not self._bar_collapsed
            and
            self._bar_panel_rect.contains(event.position())
            and self._bar_visible_rows > 0
            and len(self._items) > self._bar_visible_rows
        ):
            steps = max(1, int(abs(event.angleDelta().y()) / 120))
            direction = -1 if event.angleDelta().y() > 0 else 1
            max_scroll = max(0, len(self._items) - self._bar_visible_rows)
            next_scroll = max(0, min(max_scroll, self._bar_scroll_index + direction * steps))
            if next_scroll != self._bar_scroll_index:
                self._bar_scroll_index = next_scroll
                self.update()
            event.accept()
            return
        super().wheelEvent(event)

    def _panel_toggle_at(self, point: QPointF) -> str | None:
        if self._bar_header_rect.contains(point):
            return "bar"
        if self._pie_header_rect.contains(point):
            return "pie"
        return None

    def _toggle_panel(self, panel: str) -> None:
        if panel == "bar":
            self._bar_collapsed = not self._bar_collapsed
        elif panel == "pie":
            self._pie_collapsed = not self._pie_collapsed
        self._splitter_dragging = False
        self._splitter_hover = False
        self._splitter_handle_rect = QRectF()
        if self._bar_collapsed:
            self._bar_panel_rect = QRectF()
            self._bar_visible_rows = 0
        if self._pie_collapsed:
            self._set_pie_hover_index(-1, immediate=True)
            self._pie_chart_rect = QRectF()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # pragma: no cover - UI interaction
        if event.button() == Qt.MouseButton.LeftButton:
            toggle_target = self._panel_toggle_at(event.position())
            if toggle_target is not None:
                self._toggle_panel(toggle_target)
                self.update()
                event.accept()
                return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._bar_collapsed
            and not self._pie_collapsed
            and self._splitter_handle_rect.contains(event.position())
        ):
            self._splitter_dragging = True
            self._splitter_hover = True
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # pragma: no cover - UI interaction
        self._pie_hover_anchor = event.position()

        if self._splitter_dragging:
            if self._update_bar_ratio(event.position().y()):
                self.update()
            event.accept()
            return

        toggle_target = self._panel_toggle_at(event.position())
        if toggle_target is not None:
            if self.cursor().shape() != Qt.CursorShape.PointingHandCursor:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            if self._splitter_hover:
                self._splitter_hover = False
                self.update()
            if self._pie_hover_index != -1:
                self._set_pie_hover_index(-1)
            super().mouseMoveEvent(event)
            return
        if self.cursor().shape() == Qt.CursorShape.PointingHandCursor:
            self.unsetCursor()

        hovering = (
            not self._bar_collapsed
            and not self._pie_collapsed
            and self._splitter_handle_rect.contains(event.position())
        )
        if hovering != self._splitter_hover:
            self._splitter_hover = hovering
            self.update()

        if hovering:
            if self.cursor().shape() != Qt.CursorShape.SizeVerCursor:
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            if self._pie_hover_index != -1:
                self._set_pie_hover_index(-1)
        else:
            if self.cursor().shape() == Qt.CursorShape.SizeVerCursor:
                self.unsetCursor()
            if not self._pie_collapsed:
                self._set_pie_hover_index(self._hit_test_pie_slice(event.position()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # pragma: no cover - UI interaction
        if self._splitter_dragging and event.button() == Qt.MouseButton.LeftButton:
            self._splitter_dragging = False
            self._splitter_hover = self._splitter_handle_rect.contains(event.position())
            if self._splitter_hover:
                self.setCursor(Qt.CursorShape.SizeVerCursor)
                self._set_pie_hover_index(-1)
            else:
                self.unsetCursor()
                self._set_pie_hover_index(self._hit_test_pie_slice(event.position()))
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # pragma: no cover - UI interaction
        if not self._splitter_dragging and self._splitter_hover:
            self._splitter_hover = False
            self.update()
        if self.cursor().shape() in (Qt.CursorShape.SizeVerCursor, Qt.CursorShape.PointingHandCursor):
            self.unsetCursor()
        if self._pie_hover_index != -1:
            self._set_pie_hover_index(-1)
        super().leaveEvent(event)

    def _on_sort_mode_changed(self, index: int) -> None:
        if index < 0:
            return
        mode = self.sort_combo.itemData(index)
        if isinstance(mode, str) and mode != self._sort_mode:
            self._sort_mode = mode
            self._rebuild_items()

    def _rebuild_items(self) -> None:
        raw_items: list[tuple[str, int, int | None]] = []
        for class_name, count in self._raw_counts.items():
            if count <= 0:
                continue
            raw_items.append((class_name, int(count), self._resolve_class_id(class_name)))

        raw_items.sort(key=self._sort_key)

        self._items = [
            StatisticsItem(
                class_id=class_id,
                class_name=class_name,
                count=count,
                color=QColor(*PALETTE[index % len(PALETTE)]),
            )
            for index, (class_name, count, class_id) in enumerate(raw_items)
        ]
        self._bar_scroll_index = 0
        self._pie_slices = []
        self._pie_slice_progress = []
        self._set_pie_hover_index(-1, immediate=True)
        self.update()

    def _resolve_class_id(self, class_name: str) -> int | None:
        name = str(class_name).strip()
        if not name:
            return None
        mapped = self._class_name_to_id.get(name)
        if mapped is not None:
            return int(mapped)

        match = re.match(r"^(?:ID\s+|unknown_id_)(\d+)$", name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        if name.isdigit():
            return int(name)
        return None

    def _sort_key(self, item: tuple[str, int, int | None]) -> tuple:
        class_name, count, class_id = item
        if self._sort_mode == "count_asc":
            return (count, class_name.lower(), class_id is None, class_id if class_id is not None else 10**9)
        if self._sort_mode == "count_desc":
            return (-count, class_name.lower(), class_id is None, class_id if class_id is not None else 10**9)
        return (class_id is None, class_id if class_id is not None else 10**9, class_name.lower())

    def _display_id_name(self, item: StatisticsItem) -> str:
        if item.class_name == "其他":
            return item.class_name
        if item.class_id is None:
            return item.class_name
        placeholder_name = f"ID {item.class_id}"
        fallback_name = f"unknown_id_{item.class_id}"
        if item.class_name in {placeholder_name, fallback_name}:
            return f"{item.class_id}:未命名"
        return f"{item.class_id}:{item.class_name}"

    def _display_pie_id(self, item: StatisticsItem) -> str:
        if item.class_name == "其他":
            return item.class_name
        if item.class_id is None:
            return item.class_name
        return f"ID {item.class_id}"

    def _pie_items(self) -> list[StatisticsItem]:
        if len(self._items) <= self.PIE_MAX_ITEMS:
            return list(self._items)

        visible = list(self._items[: self.PIE_MAX_ITEMS - 1])
        remaining_count = sum(item.count for item in self._items[self.PIE_MAX_ITEMS - 1 :])
        if remaining_count > 0:
            visible.append(
                StatisticsItem(
                    class_id=None,
                    class_name="其他",
                    count=remaining_count,
                    color=QColor("#6a7480"),
                )
            )
        return visible

    def _content_rect(self) -> QRectF:
        outer_margin = 12
        header_height = max(self._controls_widget.height(), self._controls_widget.sizeHint().height())
        top = outer_margin + header_height + 10
        return QRectF(self.rect().adjusted(outer_margin, int(round(top)), -outer_margin, -outer_margin))

    def _reset_chart_regions(self) -> None:
        self._bar_panel_rect = QRectF()
        self._charts_rect = QRectF()
        self._splitter_handle_rect = QRectF()
        self._pie_panel_rect = QRectF()
        self._pie_chart_rect = QRectF()
        self._bar_header_rect = QRectF()
        self._pie_header_rect = QRectF()
        self._pie_slices = []
        self._pie_slice_progress = []
        if self._pie_hover_timer.isActive():
            self._pie_hover_timer.stop()
        self._pie_hover_index = -1
        self._bar_visible_rows = 0
        if self._splitter_hover and not self._splitter_dragging:
            self._splitter_hover = False
            self.unsetCursor()

    def _set_pie_hover_index(self, index: int, *, immediate: bool = False) -> None:
        bounded_index = index if 0 <= index < len(self._pie_slices) else -1
        if bounded_index == self._pie_hover_index and not immediate:
            return

        self._pie_hover_index = bounded_index
        if not self._pie_slice_progress:
            return

        if immediate:
            for progress_index in range(len(self._pie_slice_progress)):
                self._pie_slice_progress[progress_index] = 1.0 if progress_index == self._pie_hover_index else 0.0
            if self._pie_hover_timer.isActive():
                self._pie_hover_timer.stop()
            return

        if not self._pie_hover_timer.isActive():
            self._pie_hover_timer.start()

    def _advance_pie_hover_animation(self) -> None:
        if not self._pie_slice_progress:
            self._pie_hover_timer.stop()
            return

        changed = False
        settled = True
        for index, current in enumerate(self._pie_slice_progress):
            target = 1.0 if index == self._pie_hover_index else 0.0
            delta = target - current
            if abs(delta) < 0.01:
                next_value = target
            else:
                next_value = current + delta * 0.28
                settled = False

            if abs(next_value - current) > 1e-4:
                changed = True
                self._pie_slice_progress[index] = next_value

            if abs(target - self._pie_slice_progress[index]) >= 0.01:
                settled = False

        if changed:
            self.update()
        if settled:
            self._pie_hover_timer.stop()

    def _hit_test_pie_slice(self, point: QPointF) -> int:
        if self._pie_chart_rect.width() <= 0 or not self._pie_slices:
            return -1

        center = self._pie_chart_rect.center()
        dx = point.x() - center.x()
        dy = center.y() - point.y()
        distance = math.hypot(dx, dy)
        outer_radius = self._pie_chart_rect.width() / 2 + 10.0
        hole_ratio = 0.55
        inner_radius = self._pie_chart_rect.width() * hole_ratio / 2
        if distance < inner_radius or distance > outer_radius:
            return -1

        angle = math.degrees(math.atan2(dy, dx)) % 360
        for index, (_, start_angle, span_angle, _) in enumerate(self._pie_slices):
            if self._angle_in_span(angle, start_angle / 16.0, span_angle / 16.0):
                return index
        return -1

    @staticmethod
    def _angle_in_span(angle: float, start: float, span: float) -> bool:
        normalized_angle = angle % 360
        normalized_start = start % 360
        if span < 0:
            return ((normalized_start - normalized_angle) % 360) <= (-span + 1e-6)
        if span > 0:
            return ((normalized_angle - normalized_start) % 360) <= (span + 1e-6)
        return False

    def _bar_height_bounds(self, available_height: float) -> tuple[float, float]:
        if available_height <= 0:
            return (0.0, 0.0)

        min_bar_height = 138.0
        min_pie_height = 182.0
        total_min = min_bar_height + min_pie_height
        if total_min > available_height:
            scale = available_height / total_min
            min_bar_height *= scale
            min_pie_height *= scale

        max_bar_height = max(min_bar_height, available_height - min_pie_height)
        return (min_bar_height, max_bar_height)

    def _update_bar_ratio(self, pointer_y: float) -> bool:
        if self._charts_rect.height() <= 0:
            return False

        available_height = self._charts_rect.height() - self._splitter_handle_height
        if available_height <= 0:
            return False

        min_bar_height, max_bar_height = self._bar_height_bounds(available_height)
        if max_bar_height <= 0:
            return False

        proposed_bar_height = pointer_y - self._charts_rect.top() - self._splitter_handle_height / 2.0
        clamped_bar_height = max(min_bar_height, min(max_bar_height, proposed_bar_height))
        next_ratio = clamped_bar_height / available_height
        if abs(next_ratio - self._bar_panel_ratio) < 0.001:
            return False

        self._bar_panel_ratio = next_ratio
        return True

    def paintEvent(self, event) -> None:  # pragma: no cover - visual rendering
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#161d26"))

        content_rect = self._content_rect()
        if self._loading_text:
            self._reset_chart_regions()
            self._draw_loading_state(painter, content_rect)
            return

        if content_rect.width() <= 0 or content_rect.height() <= 0:
            self._reset_chart_regions()
            return

        if not self._items:
            self._reset_chart_regions()
            self._draw_empty_state(painter, content_rect)
            return

        pie_items = self._pie_items()

        summary_height = 68
        summary_rect = QRectF(content_rect.left(), content_rect.top(), content_rect.width(), summary_height)
        chart_top_gap = 10.0
        chart_bottom_gap = 6.0
        chart_top = summary_rect.bottom() + chart_top_gap
        charts_height = max(0.0, content_rect.bottom() - chart_top - chart_bottom_gap)
        if charts_height <= 0:
            self._reset_chart_regions()
            self._draw_summary(painter, summary_rect)
            return

        collapsed_panel_height = self._panel_header_height + 12.0
        panel_gap = 8.0
        show_splitter = not self._bar_collapsed and not self._pie_collapsed

        if show_splitter:
            available_height = max(0.0, charts_height - self._splitter_handle_height)
            min_bar_height, max_bar_height = self._bar_height_bounds(available_height)

            if max_bar_height <= 0:
                self._reset_chart_regions()
                return

            bar_panel_height = available_height * self._bar_panel_ratio
            bar_panel_height = max(min_bar_height, min(max_bar_height, bar_panel_height))
            self._bar_panel_ratio = bar_panel_height / available_height if available_height > 0 else self._bar_panel_ratio
            pie_panel_height = max(0.0, available_height - bar_panel_height)

            self._charts_rect = QRectF(content_rect.left(), chart_top, content_rect.width(), charts_height)
            bar_panel_rect = QRectF(
                content_rect.left(),
                chart_top,
                content_rect.width(),
                bar_panel_height,
            )
            handle_width = max(96.0, content_rect.width() - 20.0)
            self._splitter_handle_rect = QRectF(
                content_rect.center().x() - handle_width / 2.0,
                bar_panel_rect.bottom(),
                handle_width,
                self._splitter_handle_height,
            )
            pie_panel_rect = QRectF(
                content_rect.left(),
                self._splitter_handle_rect.bottom(),
                content_rect.width(),
                pie_panel_height,
            )
        else:
            self._charts_rect = QRectF()
            self._splitter_handle_rect = QRectF()
            if self._bar_collapsed and self._pie_collapsed:
                bar_panel_height = min(collapsed_panel_height, max(34.0, charts_height * 0.22))
                remaining_height = max(34.0, charts_height - panel_gap - bar_panel_height)
                pie_panel_height = min(collapsed_panel_height, remaining_height)
            elif self._bar_collapsed:
                bar_panel_height = min(collapsed_panel_height, max(34.0, charts_height * 0.28))
                pie_panel_height = max(34.0, charts_height - panel_gap - bar_panel_height)
            else:
                pie_panel_height = min(collapsed_panel_height, max(34.0, charts_height * 0.28))
                bar_panel_height = max(34.0, charts_height - panel_gap - pie_panel_height)

            bar_panel_rect = QRectF(
                content_rect.left(),
                chart_top,
                content_rect.width(),
                bar_panel_height,
            )
            pie_panel_rect = QRectF(
                content_rect.left(),
                bar_panel_rect.bottom() + panel_gap,
                content_rect.width(),
                pie_panel_height,
            )
        self._pie_panel_rect = pie_panel_rect

        self._draw_summary(painter, summary_rect)
        self._bar_header_rect = self._draw_panel(painter, bar_panel_rect, "数量统计", collapsed=self._bar_collapsed)
        self._pie_header_rect = self._draw_panel(painter, pie_panel_rect, "比例分布", collapsed=self._pie_collapsed)

        if show_splitter:
            self._draw_splitter_handle(painter, self._splitter_handle_rect)

        if self._bar_collapsed:
            self._bar_panel_rect = QRectF()
            self._bar_visible_rows = 0
        else:
            self._draw_bar_chart(painter, bar_panel_rect)

        if self._pie_collapsed:
            self._pie_chart_rect = QRectF()
            self._pie_slices = []
            self._set_pie_hover_index(-1, immediate=True)
        else:
            self._draw_pie_chart(painter, pie_panel_rect, pie_items)

    def _draw_splitter_handle(self, painter: QPainter, rect: QRectF) -> None:
        if rect.width() <= 0 or rect.height() <= 0:
            return

        painter.save()
        background = QColor("#24405c") if (self._splitter_dragging or self._splitter_hover) else QColor("#1a2b3d")
        painter.setPen(QPen(QColor("#35526b"), 1))
        painter.setBrush(background)
        painter.drawRoundedRect(rect.adjusted(0.5, 0.0, -0.5, 0.0), 4, 4)

        center = rect.center()
        painter.setPen(QPen(QColor("#8ab3da"), 1.2))
        for offset in (-2.0, 0.0, 2.0):
            y = center.y() + offset
            painter.drawLine(
                QPointF(center.x() - 18.0, y),
                QPointF(center.x() + 18.0, y),
            )
        painter.restore()

    def _draw_loading_state(self, painter: QPainter, rect: QRectF) -> None:
        if rect.width() <= 0 or rect.height() <= 0:
            rect = QRectF(self.rect().adjusted(24, 24, -24, -24))
        painter.setPen(QPen(QColor("#2b3644"), 1))
        painter.setBrush(QColor("#111821"))
        painter.drawRoundedRect(rect, 12, 12)
        font = painter.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#9db0c4"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._loading_text)

    def _draw_summary(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setPen(QPen(QColor("#2b3644"), 1))
        painter.setBrush(QColor("#111821"))
        painter.drawRoundedRect(rect, 12, 12)

        metrics = [
            ("图片", self._total_images),
            ("已标注", self._annotated_images),
            ("标注框", self._total_boxes),
            ("类别", len(self._items)),
        ]
        padding = 14
        segment_width = rect.width() / max(1, len(metrics))
        title_font = painter.font()
        title_font.setBold(True)
        title_font.setPointSize(max(8, title_font.pointSize() - 1))
        painter.setFont(title_font)
        for index, (label, value) in enumerate(metrics):
            segment = QRectF(rect.left() + index * segment_width, rect.top(), segment_width, rect.height())
            painter.setPen(QColor("#9db0c4"))
            painter.drawText(segment.adjusted(padding, 10, -padding, -rect.height() / 2), Qt.AlignmentFlag.AlignLeft, label)
            value_font = painter.font()
            value_font.setPointSize(value_font.pointSize() + 6)
            value_font.setBold(True)
            painter.setFont(value_font)
            painter.setPen(QColor("#e7edf5"))
            painter.drawText(segment.adjusted(padding, 22, -padding, -6), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, str(value))
            painter.setFont(title_font)
            if index < len(metrics) - 1:
                painter.setPen(QColor("#2b3644"))
                line_x = segment.right()
                painter.drawLine(
                    QPointF(line_x, segment.top() + 12),
                    QPointF(line_x, segment.bottom() - 12),
                )
        painter.restore()

    def _draw_panel(self, painter: QPainter, rect: QRectF, title: str, *, collapsed: bool = False) -> QRectF:
        painter.save()
        painter.setPen(QPen(QColor("#2b3644"), 1))
        painter.setBrush(QColor("#161d26"))
        painter.drawRoundedRect(rect, 12, 12)

        header_rect = QRectF(
            rect.left() + 12,
            rect.top() + 6,
            max(0.0, rect.width() - 24),
            self._panel_header_height,
        )
        header_font = painter.font()
        header_font.setBold(True)
        header_font.setPointSize(max(9, header_font.pointSize()))
        painter.setFont(header_font)
        painter.setPen(QColor("#e7edf5"))
        marker = "▶" if collapsed else "▼"
        painter.drawText(
            header_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{marker} {title}",
        )

        if not collapsed:
            painter.setPen(QColor("#253446"))
            painter.drawLine(
                QPointF(rect.left() + 12, header_rect.bottom() + 2),
                QPointF(rect.right() - 12, header_rect.bottom() + 2),
            )
        painter.restore()
        return header_rect

    def _draw_empty_state(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setPen(QColor("#9db0c4"))
        font = painter.font()
        font.setPointSize(font.pointSize() + 1)
        painter.setFont(font)
        painter.drawText(rect.adjusted(20, 20, -20, -20), Qt.AlignmentFlag.AlignCenter, "暂无统计数据")
        painter.restore()

    def _draw_bar_chart(self, painter: QPainter, rect: QRectF) -> None:
        self._bar_panel_rect = rect
        self._bar_visible_rows = 0
        if not self._items:
            return

        metrics = QFontMetrics(painter.font())
        content_rect = rect.adjusted(14, 40, -14, -14)
        if content_rect.width() <= 0 or content_rect.height() <= 0:
            return

        header_height = 20.0
        rows_rect = QRectF(
            content_rect.left(),
            content_rect.top() + header_height,
            content_rect.width(),
            max(0.0, content_rect.height() - header_height),
        )
        if rows_rect.height() <= 0:
            return

        row_height = 28.0
        visible_rows = max(1, int(rows_rect.height() // row_height))
        self._bar_visible_rows = visible_rows
        max_scroll = max(0, len(self._items) - visible_rows)
        self._bar_scroll_index = max(0, min(max_scroll, self._bar_scroll_index))
        start_row = self._bar_scroll_index
        end_row = min(len(self._items), start_row + visible_rows)

        scrollbar_space = 10.0 if len(self._items) > visible_rows else 0.0
        usable_width = max(120.0, rows_rect.width() - scrollbar_space)

        label_width = max(120.0, min(230.0, usable_width * 0.42))
        count_width = 58.0
        percent_width = 76.0
        bar_width = usable_width - label_width - count_width - percent_width - 18.0
        if bar_width < 88.0:
            deficit = 88.0 - bar_width
            label_width = max(92.0, label_width - deficit)
            bar_width = 88.0

        label_x = rows_rect.left()
        count_x = label_x + label_width + 6
        percent_x = count_x + count_width + 6
        bar_x = percent_x + percent_width + 6

        painter.save()
        painter.setPen(QColor("#9db0c4"))
        painter.drawText(
            QRectF(label_x, content_rect.top(), label_width, header_height),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "类别(ID:名称)",
        )
        painter.drawText(
            QRectF(count_x, content_rect.top(), count_width, header_height),
            Qt.AlignmentFlag.AlignCenter,
            "数量",
        )
        painter.drawText(
            QRectF(percent_x, content_rect.top(), percent_width, header_height),
            Qt.AlignmentFlag.AlignCenter,
            "占比",
        )
        painter.drawText(
            QRectF(bar_x, content_rect.top(), bar_width, header_height),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "柱图",
        )

        max_count = max(item.count for item in self._items)
        for row, item_index in enumerate(range(start_row, end_row)):
            item = self._items[item_index]
            y = rows_rect.top() + row * row_height
            row_rect = QRectF(rows_rect.left(), y, usable_width, row_height)
            if row % 2 == 0:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#111821"))
                painter.drawRoundedRect(row_rect.adjusted(0, 1, 0, -1), 5, 5)

            percent = (item.count / self._total_boxes * 100.0) if self._total_boxes else 0.0
            label = metrics.elidedText(
                self._display_id_name(item),
                Qt.TextElideMode.ElideRight,
                int(label_width - 4),
            )

            painter.setPen(QColor("#e7edf5"))
            painter.drawText(
                QRectF(label_x + 4, y, label_width - 4, row_height),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            painter.setPen(QColor("#d6e3f3"))
            painter.drawText(
                QRectF(count_x, y, count_width, row_height),
                Qt.AlignmentFlag.AlignCenter,
                str(item.count),
            )
            painter.setPen(QColor("#9db0c4"))
            painter.drawText(
                QRectF(percent_x, y, percent_width, row_height),
                Qt.AlignmentFlag.AlignCenter,
                f"{percent:.1f}%",
            )

            ratio = (item.count / max_count) if max_count else 0.0
            track_rect = QRectF(bar_x, y + 7, bar_width, max(8.0, row_height - 14))
            bar_rect = QRectF(track_rect.left(), track_rect.top(), track_rect.width() * ratio, track_rect.height())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#263241"))
            painter.drawRoundedRect(track_rect, 5, 5)
            painter.setBrush(item.color)
            painter.drawRoundedRect(bar_rect, 5, 5)

        painter.restore()

        if len(self._items) > visible_rows:
            self._draw_bar_scrollbar(painter, rows_rect, start_row, visible_rows, len(self._items))

    def _draw_bar_scrollbar(
        self,
        painter: QPainter,
        rows_rect: QRectF,
        start_row: int,
        visible_rows: int,
        total_rows: int,
    ) -> None:
        if total_rows <= visible_rows:
            return

        painter.save()
        track_rect = QRectF(rows_rect.right() - 6, rows_rect.top(), 4, rows_rect.height())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#1f2a37"))
        painter.drawRoundedRect(track_rect, 2, 2)

        ratio = visible_rows / total_rows
        thumb_height = max(18.0, track_rect.height() * ratio)
        max_scroll = max(1, total_rows - visible_rows)
        thumb_offset = (track_rect.height() - thumb_height) * (start_row / max_scroll)
        thumb_rect = QRectF(track_rect.left(), track_rect.top() + thumb_offset, track_rect.width(), thumb_height)
        painter.setBrush(QColor("#4da3ff"))
        painter.drawRoundedRect(thumb_rect, 2, 2)
        painter.restore()

    def _draw_pie_chart(self, painter: QPainter, rect: QRectF, pie_items: list[StatisticsItem]) -> None:
        content_rect = rect.adjusted(18, 40, -18, -18)
        self._pie_chart_rect = QRectF()
        self._pie_slices = []
        if not pie_items:
            self._pie_slice_progress = []
            self._set_pie_hover_index(-1, immediate=True)
            return

        chart_width = min(content_rect.width() * 0.78, content_rect.height() * 0.88)
        chart_width = max(150.0, chart_width)
        chart_width = min(chart_width, content_rect.width(), content_rect.height())
        center_x = content_rect.center().x()
        pie_top = content_rect.top() + max(0.0, (content_rect.height() - chart_width) * 0.5)
        pie_rect = QRectF(
            center_x - chart_width / 2,
            pie_top,
            chart_width,
            chart_width,
        )
        self._pie_chart_rect = pie_rect
        total = fsum(item.count for item in pie_items)
        if total <= 0:
            self._pie_slice_progress = []
            self._set_pie_hover_index(-1, immediate=True)
            return

        slices: list[tuple[StatisticsItem, int, int, float]] = []
        start_angle = 90 * 16
        for item in pie_items:
            count = item.count
            percent = (count / total * 100.0) if total else 0.0
            span_angle = int(round(-(count / total) * 360 * 16))
            slices.append((item, start_angle, span_angle, percent))
            start_angle += span_angle

        self._pie_slices = slices
        if len(self._pie_slice_progress) != len(slices):
            self._pie_slice_progress = [0.0] * len(slices)
            self._set_pie_hover_index(self._pie_hover_index, immediate=True)

        for index, (item, start_angle, span_angle, _) in enumerate(slices):
            progress = self._pie_slice_progress[index]
            mid_angle = (start_angle + span_angle / 2.0) / 16.0
            radians = math.radians(mid_angle)
            explode = 8.0 * progress
            offset_x = math.cos(radians) * explode
            offset_y = -math.sin(radians) * explode
            slice_rect = pie_rect.translated(offset_x, offset_y)

            fill_color = QColor(item.color)
            if progress > 0.01:
                fill_color = fill_color.lighter(105 + int(progress * 10))

            painter.setBrush(fill_color)
            painter.setPen(QPen(QColor("#161d26"), 2))
            painter.drawPie(slice_rect, start_angle, span_angle)

        hole_ratio = 0.55
        hole = QRectF(
            pie_rect.center().x() - pie_rect.width() * hole_ratio / 2,
            pie_rect.center().y() - pie_rect.height() * hole_ratio / 2,
            pie_rect.width() * hole_ratio,
            pie_rect.height() * hole_ratio,
        )
        painter.setBrush(QColor("#161d26"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(hole)

        center_font = painter.font()
        center_font.setBold(True)
        center_font.setPointSize(center_font.pointSize() + 3)
        painter.setFont(center_font)
        painter.setPen(QColor("#e7edf5"))
        painter.drawText(hole.adjusted(-12, -18, 12, 10), Qt.AlignmentFlag.AlignCenter, str(self._total_boxes))

        if 0 <= self._pie_hover_index < len(slices):
            hover_item, _, _, hover_percent = slices[self._pie_hover_index]
            self._draw_pie_hover_card(painter, content_rect, hover_item, hover_percent)

    def _draw_pie_hover_card(
        self,
        painter: QPainter,
        content_rect: QRectF,
        item: StatisticsItem,
        percent: float,
    ) -> None:
        card_width = min(280.0, max(200.0, content_rect.width() * 0.56))
        card_height = 58.0
        anchor = self._pie_hover_anchor if self._pie_panel_rect.contains(self._pie_hover_anchor) else content_rect.center()
        card_x = min(max(content_rect.left() + 6.0, anchor.x() + 14.0), content_rect.right() - card_width - 4.0)
        card_y = min(max(content_rect.top() + 6.0, anchor.y() + 14.0), content_rect.bottom() - card_height - 4.0)
        card_rect = QRectF(card_x, card_y, card_width, card_height)

        painter.save()
        painter.setPen(QPen(QColor("#3f5f7f"), 1))
        painter.setBrush(QColor(15, 22, 31, 232))
        painter.drawRoundedRect(card_rect, 10, 10)

        title_font = painter.font()
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#e7edf5"))
        painter.drawText(
            card_rect.adjusted(12, 6, -12, -card_height / 2),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._display_pie_id(item),
        )

        detail_font = painter.font()
        detail_font.setBold(False)
        detail_font.setPointSize(max(8, detail_font.pointSize() - 1))
        painter.setFont(detail_font)
        painter.setPen(QColor("#9db0c4"))
        painter.drawText(
            card_rect.adjusted(12, 24, -12, -8),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"数量 {item.count}    占比 {percent:.1f}%",
        )
        painter.restore()
