from __future__ import annotations

import hashlib
from typing import Literal

from PyQt6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFontMetrics, QImage, QKeyEvent, QMouseEvent, QPaintEvent, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..constants import CANVAS_BACKGROUND, CROSSHAIR_COLOR, HANDLE_SIZE
from ..models import Box


HandleName = Literal["tl", "tr", "bl", "br"]


class AnnotationCanvas(QWidget):
    annotationChanged = pyqtSignal()
    selectionChanged = pyqtSignal(int)
    drawBoxRequested = pyqtSignal(int, int, int, int)
    cursorPositionChanged = pyqtSignal(int, int)
    editOperationStarted = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("canvasCard")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(720, 540)

        self._image: QImage | None = None
        self._display_image: QImage | None = None
        self._scale = 1.0
        self._boxes: list[Box] = []
        self._selected_index = -1
        self._edit_mode = False
        self._draw_mode = False
        self._cursor_image_pos: tuple[int, int] | None = None
        self._draft_start: tuple[int, int] | None = None
        self._draft_end: tuple[int, int] | None = None
        self._operation: dict[str, object] | None = None
        self._placeholder_size = QSize(960, 640)
        self._display_size = QSize(960, 640)
        self._class_name_to_id: dict[str, int] = {}
        self._class_color_rgb: dict[str, tuple[int, int, int]] = {}
        self._label_background_alpha = 140
        self._label_show_name = True
        self._label_show_id = False

    @property
    def selected_index(self) -> int:
        return self._selected_index

    @property
    def boxes(self) -> list[Box]:
        return self._boxes

    def set_document(self, image: QImage | None, boxes: list[Box] | None, scale: float | None = None) -> None:
        self._image = image
        self._boxes = boxes if boxes is not None else []
        if scale is not None:
            self._scale = self._clamp_scale(scale)
        self._selected_index = -1
        self._draft_start = None
        self._draft_end = None
        self._operation = None
        self._cursor_image_pos = None
        self._rebuild_display_image()
        self.update()

    def set_boxes(self, boxes: list[Box]) -> None:
        self._boxes = boxes
        if self._selected_index >= len(self._boxes):
            self._selected_index = -1
        self.update()

    def set_class_mapping(self, name_to_id: dict[str, int]) -> None:
        self._class_name_to_id = {str(name): int(class_id) for name, class_id in name_to_id.items()}
        known_names = set(self._class_name_to_id)
        self._class_color_rgb = {
            class_name: rgb
            for class_name, rgb in self._class_color_rgb.items()
            if class_name in known_names
        }
        self.update()

    def set_label_visual_options(
        self,
        *,
        background_alpha: int | None = None,
        show_name: bool | None = None,
        show_id: bool | None = None,
    ) -> None:
        changed = False

        if background_alpha is not None:
            next_alpha = max(0, min(255, int(background_alpha)))
            if next_alpha != self._label_background_alpha:
                self._label_background_alpha = next_alpha
                changed = True

        if show_name is not None:
            next_show_name = bool(show_name)
            if next_show_name != self._label_show_name:
                self._label_show_name = next_show_name
                changed = True

        if show_id is not None:
            next_show_id = bool(show_id)
            if next_show_id != self._label_show_id:
                self._label_show_id = next_show_id
                changed = True

        if changed:
            self.update()

    def set_scale(self, scale: float) -> None:
        scale = self._clamp_scale(scale)
        if abs(scale - self._scale) < 1e-6:
            return
        self._scale = scale
        self._rebuild_display_image()
        self.update()

    def set_modes(self, *, edit_mode: bool, draw_mode: bool) -> None:
        self._edit_mode = edit_mode
        self._draw_mode = draw_mode
        self._update_cursor()
        self.update()

    def set_selected_index(self, index: int) -> None:
        self._selected_index = index if 0 <= index < len(self._boxes) else -1
        self.update()

    def clear_interaction(self) -> None:
        self._draft_start = None
        self._draft_end = None
        self._operation = None
        self._cursor_image_pos = None
        self.update()

    def sizeHint(self) -> QSize:  # pragma: no cover - Qt hint
        return self._display_size if self._display_image is not None else self._placeholder_size

    def keyPressEvent(self, event: QKeyEvent) -> None:
        window = self.window()
        if hasattr(window, "handle_key_press") and callable(getattr(window, "handle_key_press")):
            if window.handle_key_press(event):
                return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._image is None or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        self.setFocus()
        x, y = self._widget_to_image(event.position())
        self._cursor_image_pos = (x, y)
        self.cursorPositionChanged.emit(x, y)

        if self._draw_mode:
            self._draft_start = (x, y)
            self._draft_end = (x, y)
            self._selected_index = -1
            self.selectionChanged.emit(-1)
            self.update()
            return

        if not self._edit_mode:
            self._selected_index = -1
            self.selectionChanged.emit(-1)
            self.update()
            return

        hit = self._hit_test(x, y)
        if hit is None:
            if self._selected_index != -1:
                self._selected_index = -1
                self.selectionChanged.emit(-1)
                self.update()
            return

        index, mode, handle = hit
        selection_changed = index != self._selected_index
        self._selected_index = index
        if selection_changed:
            self.selectionChanged.emit(index)

        if mode == "move":
            box = self._boxes[index].ordered()
            self._operation = {
                "type": "move",
                "index": index,
                "offset_x": x - box.x1,
                "offset_y": y - box.y1,
                "width": box.width(),
                "height": box.height(),
            }
            self.editOperationStarted.emit()
        elif mode == "resize" and handle is not None:
            self._operation = {
                "type": "resize",
                "index": index,
                "handle": handle,
            }
            self.editOperationStarted.emit()
        self._update_cursor()
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._image is None:
            super().mouseMoveEvent(event)
            return

        x, y = self._widget_to_image(event.position())
        self._cursor_image_pos = (x, y)
        self.cursorPositionChanged.emit(x, y)

        if self._draw_mode and self._draft_start is not None:
            self._draft_end = (x, y)
            self.update()
            return

        if self._edit_mode and self._operation is not None:
            self._apply_operation(x, y)
            self.update()
            return

        self._update_cursor(event.position())
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._image is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return

        if self._draw_mode and self._draft_start is not None and self._draft_end is not None:
            left, top = self._draft_start
            right, bottom = self._draft_end
            x1, x2 = sorted((left, right))
            y1, y2 = sorted((top, bottom))
            if x2 - x1 > 5 and y2 - y1 > 5:
                self.drawBoxRequested.emit(x1, y1, x2, y2)
            self._draft_start = None
            self._draft_end = None
            self._operation = None
            self.update()
            return

        if self._edit_mode and self._operation is not None and self._selected_index != -1:
            box = self._boxes[self._selected_index].ordered().clamp(self.image_width, self.image_height)
            self._boxes[self._selected_index].x1 = box.x1
            self._boxes[self._selected_index].y1 = box.y1
            self._boxes[self._selected_index].x2 = box.x2
            self._boxes[self._selected_index].y2 = box.y2
            self._operation = None
            self.annotationChanged.emit()
            self.update()
            return

        self._operation = None
        self.update()

    def leaveEvent(self, event) -> None:  # pragma: no cover - Qt interaction
        if self._operation is None:
            self._cursor_image_pos = None
            self.cursorPositionChanged.emit(-1, -1)
            self.update()
        super().leaveEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(CANVAS_BACKGROUND))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self._display_image is None:
            self._paint_placeholder(painter)
            return

        painter.drawImage(QPoint(0, 0), self._display_image)
        self._paint_boxes(painter)
        self._paint_crosshair(painter)
        self._paint_draft_box(painter)

    @property
    def image_width(self) -> int:
        return self._image.width() if self._image is not None else 0

    @property
    def image_height(self) -> int:
        return self._image.height() if self._image is not None else 0

    def _paint_placeholder(self, painter: QPainter) -> None:
        painter.setPen(QColor("#9db0c4"))
        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "打开数据集后开始标注\n支持拖拽、缩放、导出和自动保存",
        )

    def _paint_boxes(self, painter: QPainter) -> None:
        metrics = QFontMetrics(painter.font())
        for index, box in enumerate(self._boxes):
            ordered = box.ordered()
            color = self._color_for_class(box.class_name)
            selected = index == self._selected_index
            pen_width = 3 if selected else 2
            if self._operation is not None and self._operation.get("index") == index:
                pen_width = 4

            painter.setPen(QPen(color, pen_width))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rect = QRectF(
                ordered.x1 * self._scale,
                ordered.y1 * self._scale,
                max(1.0, ordered.width() * self._scale),
                max(1.0, ordered.height() * self._scale),
            )
            painter.drawRect(rect)

            label_text = self._compose_label_text(box)
            if label_text:
                text_width = metrics.horizontalAdvance(label_text)
                text_height = metrics.height()
                padding_x = 6
                padding_y = 4
                label_x = rect.left()
                label_y = max(0.0, rect.top() - text_height - padding_y * 2)
                label_rect = QRectF(label_x, label_y, text_width + padding_x * 2, text_height + padding_y * 2)
                label_fill = QColor(color.red(), color.green(), color.blue(), self._label_background_alpha)
                painter.fillRect(label_rect, label_fill)
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(
                    label_rect.adjusted(padding_x, padding_y, -padding_x, -padding_y),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    label_text,
                )

            if selected and self._edit_mode:
                self._draw_handles(painter, ordered)

    def _compose_label_text(self, box: Box) -> str:
        class_name = str(box.class_name).strip() or "未命名"
        class_id = self._class_name_to_id.get(class_name)

        if self._label_show_name and self._label_show_id:
            if class_id is not None:
                return f"{class_id} | {class_name}"
            return class_name

        if self._label_show_id:
            return f"ID {class_id}" if class_id is not None else "ID ?"

        if self._label_show_name:
            return class_name

        return ""

    def _color_for_class(self, class_name: str) -> QColor:
        normalized = str(class_name).strip() or "未命名"
        cached_rgb = self._class_color_rgb.get(normalized)
        if cached_rgb is not None:
            return QColor(*cached_rgb)

        mapped_id = self._class_name_to_id.get(normalized)
        seed_key = f"id:{mapped_id}" if mapped_id is not None else f"name:{normalized}"
        rgb = self._allocate_unique_class_color(seed_key)
        self._class_color_rgb[normalized] = rgb
        return QColor(*rgb)

    def _allocate_unique_class_color(self, seed_key: str) -> tuple[int, int, int]:
        used_colors = set(self._class_color_rgb.values())
        digest = hashlib.sha1(seed_key.encode("utf-8")).digest()
        hue_seed = int.from_bytes(digest[:2], "big") % 360
        sat_seed = 160 + int(digest[2] % 70)
        val_seed = 190 + int(digest[3] % 55)

        for attempt in range(720):
            hue = int((hue_seed + attempt * 37) % 360)
            saturation = int(min(245, sat_seed + (attempt % 3) * 5))
            value = int(max(150, min(250, val_seed - (attempt % 5) * 3)))
            candidate = QColor.fromHsv(hue, saturation, value)
            rgb = (candidate.red(), candidate.green(), candidate.blue())
            if rgb not in used_colors:
                return rgb

        fallback = QColor.fromHsv(int(hue_seed), int(sat_seed), int(val_seed))
        return (fallback.red(), fallback.green(), fallback.blue())

    def _paint_crosshair(self, painter: QPainter) -> None:
        if self._cursor_image_pos is None:
            return
        if not (self._draw_mode or self._edit_mode):
            return
        x, y = self._cursor_image_pos
        if not (0 <= x < self.image_width and 0 <= y < self.image_height):
            return

        screen_x = x * self._scale
        screen_y = y * self._scale
        painter.setPen(QPen(QColor(*CROSSHAIR_COLOR), 1))
        painter.drawLine(QPointF(screen_x, 0), QPointF(screen_x, self._display_size.height()))
        painter.drawLine(QPointF(0, screen_y), QPointF(self._display_size.width(), screen_y))

    def _paint_draft_box(self, painter: QPainter) -> None:
        if not self._draw_mode or self._draft_start is None or self._draft_end is None:
            return
        x1, y1 = self._draft_start
        x2, y2 = self._draft_end
        rect = QRectF(
            min(x1, x2) * self._scale,
            min(y1, y2) * self._scale,
            max(1.0, abs(x2 - x1) * self._scale),
            max(1.0, abs(y2 - y1) * self._scale),
        )
        painter.setPen(QPen(QColor(49, 196, 141), 2, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

    def _draw_handles(self, painter: QPainter, box: Box) -> None:
        points = [
            (box.x1, box.y1),
            (box.x2, box.y1),
            (box.x1, box.y2),
            (box.x2, box.y2),
        ]
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setBrush(QColor(255, 215, 0))
        size = HANDLE_SIZE
        for point_x, point_y in points:
            rect = QRectF(
                point_x * self._scale - size / 2,
                point_y * self._scale - size / 2,
                size,
                size,
            )
            painter.drawRect(rect)

    def _hit_test(self, x: int, y: int) -> tuple[int, str, HandleName | None] | None:
        tolerance = max(5, int(round(8 / max(self._scale, 0.1))))
        for index in range(len(self._boxes) - 1, -1, -1):
            box = self._boxes[index].ordered()
            handles = {
                "tl": (box.x1, box.y1),
                "tr": (box.x2, box.y1),
                "bl": (box.x1, box.y2),
                "br": (box.x2, box.y2),
            }
            for handle_name, (hx, hy) in handles.items():
                if abs(x - hx) <= tolerance and abs(y - hy) <= tolerance:
                    return index, "resize", handle_name
            if box.x1 < x < box.x2 and box.y1 < y < box.y2:
                return index, "move", None
        return None

    def _apply_operation(self, x: int, y: int) -> None:
        if self._operation is None:
            return
        op_type = self._operation.get("type")
        index = int(self._operation.get("index", -1))
        if index < 0 or index >= len(self._boxes):
            return

        if op_type == "move":
            offset_x = int(self._operation.get("offset_x", 0))
            offset_y = int(self._operation.get("offset_y", 0))
            width = int(self._operation.get("width", 1))
            height = int(self._operation.get("height", 1))
            max_x = max(0, self.image_width - width)
            max_y = max(0, self.image_height - height)
            new_x1 = self._clamp(x - offset_x, 0, max_x)
            new_y1 = self._clamp(y - offset_y, 0, max_y)
            self._boxes[index].x1 = new_x1
            self._boxes[index].y1 = new_y1
            self._boxes[index].x2 = new_x1 + width
            self._boxes[index].y2 = new_y1 + height
        elif op_type == "resize":
            handle = str(self._operation.get("handle", ""))
            box = self._boxes[index]
            if handle == "tl":
                box.x1 = self._clamp(x, 0, self.image_width - 1)
                box.y1 = self._clamp(y, 0, self.image_height - 1)
            elif handle == "tr":
                box.x2 = self._clamp(x, 0, self.image_width - 1)
                box.y1 = self._clamp(y, 0, self.image_height - 1)
            elif handle == "bl":
                box.x1 = self._clamp(x, 0, self.image_width - 1)
                box.y2 = self._clamp(y, 0, self.image_height - 1)
            elif handle == "br":
                box.x2 = self._clamp(x, 0, self.image_width - 1)
                box.y2 = self._clamp(y, 0, self.image_height - 1)

    def _rebuild_display_image(self) -> None:
        if self._image is None:
            self._display_image = None
            self._display_size = self._placeholder_size
            self.setFixedSize(self._placeholder_size)
            return

        scaled_width = max(1, int(round(self._image.width() * self._scale)))
        scaled_height = max(1, int(round(self._image.height() * self._scale)))
        self._display_size = QSize(scaled_width, scaled_height)
        self.setFixedSize(self._display_size)
        self._display_image = self._image.scaled(
            self._display_size,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )

    def _widget_to_image(self, position: QPointF) -> tuple[int, int]:
        if self._image is None:
            return -1, -1
        x = int(position.x() / max(self._scale, 1e-6))
        y = int(position.y() / max(self._scale, 1e-6))
        x = self._clamp(x, 0, self.image_width - 1)
        y = self._clamp(y, 0, self.image_height - 1)
        return x, y

    def _update_cursor(self, position: QPointF | None = None) -> None:
        if self._draw_mode or self._edit_mode:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _clamp_scale(self, scale: float) -> float:
        return max(0.1, min(8.0, float(scale)))

    def _clamp(self, value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(maximum, int(value)))
