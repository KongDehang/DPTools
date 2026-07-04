from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable


def natural_sort_key(value: object) -> list[object]:
    text = str(value)
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", text)]


@dataclass(slots=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    points: list[tuple[int, int]] = field(default_factory=list)
    shape_type: str = "rectangle"

    def copy(self) -> "Box":
        return Box(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.class_name,
            list(self.points),
            self.shape_type,
        )

    @property
    def is_polygon(self) -> bool:
        return self.shape_type == "polygon" and len(self.points) >= 3

    def polygon_points(self) -> list[tuple[int, int]]:
        if self.is_polygon:
            return [(int(x), int(y)) for x, y in self.points]
        ordered = self.ordered()
        return [
            (ordered.x1, ordered.y1),
            (ordered.x2, ordered.y1),
            (ordered.x2, ordered.y2),
            (ordered.x1, ordered.y2),
        ]

    def ordered(self) -> "Box":
        if self.is_polygon:
            xs = [int(x) for x, _ in self.points]
            ys = [int(y) for _, y in self.points]
            left = min(xs)
            top = min(ys)
            right = max(xs)
            bottom = max(ys)
        else:
            left = min(int(self.x1), int(self.x2))
            top = min(int(self.y1), int(self.y2))
            right = max(int(self.x1), int(self.x2))
            bottom = max(int(self.y1), int(self.y2))
        return Box(left, top, right, bottom, self.class_name, list(self.points), self.shape_type)

    def clamp(self, width: int, height: int) -> "Box":
        if width <= 0 or height <= 0:
            return self.ordered()

        if self.is_polygon:
            clamped_points = [
                (
                    max(0, min(width - 1, int(round(x)))),
                    max(0, min(height - 1, int(round(y)))),
                )
                for x, y in self.points
            ]
            xs = [x for x, _ in clamped_points]
            ys = [y for _, y in clamped_points]
            return Box(
                min(xs),
                min(ys),
                max(xs),
                max(ys),
                self.class_name,
                clamped_points,
                "polygon",
            )

        box = self.ordered()
        box.x1 = max(0, min(width - 1, box.x1))
        box.y1 = max(0, min(height - 1, box.y1))
        box.x2 = max(0, min(width - 1, box.x2))
        box.y2 = max(0, min(height - 1, box.y2))

        if box.x2 <= box.x1:
            if width > 1 and box.x1 >= width - 1:
                box.x1 = width - 2
                box.x2 = width - 1
            else:
                box.x2 = min(width - 1, box.x1 + 1)
        if box.y2 <= box.y1:
            if height > 1 and box.y1 >= height - 1:
                box.y1 = height - 2
                box.y2 = height - 1
            else:
                box.y2 = min(height - 1, box.y1 + 1)
        return box

    def width(self) -> int:
        return abs(int(self.x2) - int(self.x1))

    def height(self) -> int:
        return abs(int(self.y2) - int(self.y1))

    def area(self) -> int:
        if self.is_polygon:
            points = self.polygon_points()
            doubled_area = 0
            for index, (x1, y1) in enumerate(points):
                x2, y2 = points[(index + 1) % len(points)]
                doubled_area += x1 * y2 - x2 * y1
            return int(abs(doubled_area) / 2)
        return self.width() * self.height()

    def normalized(self, width: int, height: int) -> tuple[float, float, float, float]:
        ordered = self.ordered()
        if width <= 0 or height <= 0:
            return 0.0, 0.0, 0.0, 0.0
        cx = ((ordered.x1 + ordered.x2) / 2.0) / width
        cy = ((ordered.y1 + ordered.y2) / 2.0) / height
        bw = abs(ordered.x2 - ordered.x1) / width
        bh = abs(ordered.y2 - ordered.y1) / height
        return cx, cy, bw, bh

    def summary(self, index: int | None = None) -> str:
        prefix = f"#{index + 1} " if index is not None else ""
        shape_label = f"polygon {len(self.points)}点" if self.is_polygon else "rectangle"
        ordered = self.ordered()
        return (
            f"{prefix}{self.class_name} | {shape_label} | "
            f"({ordered.x1}, {ordered.y1}) -> ({ordered.x2}, {ordered.y2}) | "
            f"{self.width()} x {self.height()}"
        )

    def moved(self, dx: int, dy: int, width: int, height: int) -> "Box":
        if self.is_polygon:
            moved_points = [(int(x) + int(dx), int(y) + int(dy)) for x, y in self.points]
            return Box.from_polygon(moved_points, self.class_name).clamp(width, height)
        ordered = self.ordered()
        return Box(
            ordered.x1 + int(dx),
            ordered.y1 + int(dy),
            ordered.x2 + int(dx),
            ordered.y2 + int(dy),
            self.class_name,
        ).clamp(width, height)

    def transformed(
        self,
        scale_x: float,
        scale_y: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        width: int | None = None,
        height: int | None = None,
    ) -> "Box":
        if self.is_polygon:
            transformed_points = [
                (
                    int(round(x * scale_x + offset_x)),
                    int(round(y * scale_y + offset_y)),
                )
                for x, y in self.points
            ]
            result = Box.from_polygon(transformed_points, self.class_name)
        else:
            ordered = self.ordered()
            result = Box(
                int(round(ordered.x1 * scale_x + offset_x)),
                int(round(ordered.y1 * scale_y + offset_y)),
                int(round(ordered.x2 * scale_x + offset_x)),
                int(round(ordered.y2 * scale_y + offset_y)),
                self.class_name,
            )
        if width is not None and height is not None:
            return result.clamp(width, height)
        return result.ordered()

    @classmethod
    def from_points(cls, points: Iterable[Iterable[float]], class_name: str) -> "Box":
        point_list = list(points)
        if len(point_list) < 2:
            raise ValueError("A rectangle annotation needs at least two points")
        x1, y1 = point_list[0]
        x2, y2 = point_list[1]
        return cls(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), class_name)

    @classmethod
    def from_polygon(cls, points: Iterable[Iterable[float]], class_name: str) -> "Box":
        parsed_points: list[tuple[int, int]] = []
        for raw_point in points:
            try:
                x, y = raw_point
            except Exception:
                continue
            parsed_points.append((int(round(float(x))), int(round(float(y)))))
        if len(parsed_points) < 3:
            raise ValueError("A polygon annotation needs at least three points")

        xs = [x for x, _ in parsed_points]
        ys = [y for _, y in parsed_points]
        return cls(
            min(xs),
            min(ys),
            max(xs),
            max(ys),
            class_name,
            parsed_points,
            "polygon",
        )


@dataclass(slots=True)
class AnnotationDocument:
    image_path: Path
    label_path: Path | None
    image_size: tuple[int, int]
    boxes: list[Box] = field(default_factory=list)
    source_format: str = ""

    @property
    def image_name(self) -> str:
        return self.image_path.name

    @property
    def image_width(self) -> int:
        return int(self.image_size[0])

    @property
    def image_height(self) -> int:
        return int(self.image_size[1])
