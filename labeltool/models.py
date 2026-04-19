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

    def copy(self) -> "Box":
        return Box(self.x1, self.y1, self.x2, self.y2, self.class_name)

    def ordered(self) -> "Box":
        left = min(int(self.x1), int(self.x2))
        top = min(int(self.y1), int(self.y2))
        right = max(int(self.x1), int(self.x2))
        bottom = max(int(self.y1), int(self.y2))
        return Box(left, top, right, bottom, self.class_name)

    def clamp(self, width: int, height: int) -> "Box":
        if width <= 0 or height <= 0:
            return self.ordered()

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
        return (
            f"{prefix}{self.class_name} | "
            f"({self.x1}, {self.y1}) -> ({self.x2}, {self.y2}) | "
            f"{self.width()} x {self.height()}"
        )

    @classmethod
    def from_points(cls, points: Iterable[Iterable[float]], class_name: str) -> "Box":
        point_list = list(points)
        if len(point_list) < 2:
            raise ValueError("A rectangle annotation needs at least two points")
        x1, y1 = point_list[0]
        x2, y2 = point_list[1]
        return cls(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), class_name)


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
