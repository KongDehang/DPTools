from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..constants import FORMAT_SUFFIX
from ..models import Box
from .class_manager import ClassManager


class AnnotationIO:
    def _cv2(self):
        import cv2 as cv2_module

        return cv2_module

    def remove_class_id_from_txt_file(self, label_path: str | Path, target_class_id: int) -> int:
        path = Path(label_path)
        if not path.exists() or path.suffix.lower() != ".txt":
            return 0

        target = int(target_class_id)
        source_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        updated_lines: list[str] = []
        removed = 0

        for raw_line in source_lines:
            stripped = raw_line.strip()
            if not stripped:
                updated_lines.append(raw_line)
                continue

            parts = stripped.split()
            if len(parts) < 5:
                updated_lines.append(raw_line)
                continue

            try:
                current_id = int(float(parts[0]))
            except Exception:
                updated_lines.append(raw_line)
                continue

            if current_id == target:
                removed += 1
                continue
            updated_lines.append(raw_line)

        if removed > 0:
            path.write_text("\n".join(updated_lines), encoding="utf-8")
        return removed

    def remap_class_ids_in_file(self, label_path: str | Path, id_mapping: dict[int, int]) -> bool:
        if not id_mapping:
            return False

        path = Path(label_path)
        if not path.exists() or path.suffix.lower() != ".txt":
            return False

        return self._remap_txt_class_ids(path, id_mapping)

    def count_annotation(
        self,
        label_path: str | Path | None,
        class_id_to_name: dict[int, str] | None = None,
    ) -> tuple[dict[str, int], int]:
        if not label_path:
            return {}, 0

        path = Path(label_path)
        if not path.exists():
            return {}, 0

        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                return self._count_json(path), self._count_boxes_json(path)
            if suffix == ".txt":
                return self._count_txt(path, class_id_to_name or {}), self._count_boxes_txt(path)
            if suffix == ".xml":
                return self._count_xml(path), self._count_boxes_xml(path)
            if suffix == ".png":
                return self._count_mask_png(path, class_id_to_name or {}), self._count_shapes_mask_png(path)
        except Exception:
            return {}, 0
        return {}, 0

    def load_annotation(
        self,
        label_path: str | Path | None,
        image_size: tuple[int, int],
        class_manager: ClassManager,
    ) -> list[Box]:
        if not label_path:
            return []

        path = Path(label_path)
        if not path.exists():
            return []

        width, height = image_size
        suffix = path.suffix.lower()
        boxes: list[Box] = []

        try:
            if suffix == ".json":
                boxes = self._load_json(path, class_manager, width, height)
            elif suffix == ".txt":
                boxes = self._load_txt(path, class_manager, width, height)
            elif suffix == ".xml":
                boxes = self._load_xml(path, class_manager, width, height)
            elif suffix == ".png":
                boxes = self._load_mask_png(path, class_manager, width, height)
        except Exception:
            boxes = []

        if class_manager.is_dirty:
            class_manager.sync_to_yaml()
        return boxes

    def save_annotation(
        self,
        label_path: str | Path,
        boxes: list[Box],
        image_size: tuple[int, int],
        class_manager: ClassManager,
        image_name: str | None = None,
    ) -> Path:
        path = Path(label_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        width, height = image_size
        for box in boxes:
            class_manager.ensure_name(box.class_name)
        class_manager.sync_to_yaml()

        suffix = path.suffix.lower()
        if suffix == ".json":
            self._save_json(path, boxes, width, height, image_name)
        elif suffix == ".txt":
            self._save_txt(path, boxes, width, height, class_manager)
        elif suffix == ".xml":
            self._save_xml(path, boxes, width, height, image_name)
        elif suffix == ".png":
            self._save_mask_png(path, boxes, width, height, class_manager)
        else:
            raise ValueError(f"Unsupported annotation format: {path.suffix}")
        return path

    def convert_annotation(
        self,
        source_path: str | Path,
        target_path: str | Path,
        image_size: tuple[int, int],
        class_manager: ClassManager,
        image_name: str | None = None,
    ) -> Path:
        boxes = self.load_annotation(source_path, image_size, class_manager)
        return self.save_annotation(target_path, boxes, image_size, class_manager, image_name=image_name)

    def default_export_path(self, image_path: str | Path, target_format: str) -> Path:
        path = Path(image_path)
        if target_format == "mask_png":
            return path.with_name(f"{path.stem}_mask.png")
        suffix = FORMAT_SUFFIX[target_format]
        return path.with_suffix(suffix)

    def _load_json(self, path: Path, class_manager: ClassManager, width: int, height: int) -> list[Box]:
        data = json.loads(path.read_text(encoding="utf-8"))
        boxes: list[Box] = []
        for shape in data.get("shapes", []):
            points = shape.get("points", [])
            if len(points) < 2:
                continue
            class_name = class_manager.resolve_label_token(str(shape.get("label", "object")))
            shape_type = str(shape.get("shape_type", "") or "").strip().lower()
            if shape_type == "polygon" or len(points) >= 3:
                try:
                    box = Box.from_polygon(points, class_name).clamp(width, height)
                except Exception:
                    continue
            else:
                box = Box.from_points(points[:2], class_name).clamp(width, height)
            boxes.append(box)
        return boxes

    def _count_json(self, path: Path) -> dict[str, int]:
        data = json.loads(path.read_text(encoding="utf-8"))
        counts: dict[str, int] = {}
        for shape in data.get("shapes", []):
            label = str(shape.get("label", "object")).strip() or "object"
            counts[label] = counts.get(label, 0) + 1
        return counts

    def _count_boxes_json(self, path: Path) -> int:
        data = json.loads(path.read_text(encoding="utf-8"))
        return sum(1 for shape in data.get("shapes", []) if len(shape.get("points", [])) >= 2)

    def _count_txt(self, path: Path, class_id_to_name: dict[int, str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_token = parts[0]
            try:
                class_id = int(float(class_token))
                class_name = class_id_to_name.get(class_id, f"ID {class_id}")
            except Exception:
                class_name = class_token
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def _count_boxes_txt(self, path: Path) -> int:
        total = 0
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if len(line.strip().split()) >= 5:
                total += 1
        return total

    def _count_xml(self, path: Path) -> dict[str, int]:
        tree = ET.parse(path)
        root = tree.getroot()
        counts: dict[str, int] = {}
        for obj in root.findall("object"):
            name_node = obj.find("name")
            if name_node is None:
                continue
            class_name = str(name_node.text or "object").strip() or "object"
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def _count_boxes_xml(self, path: Path) -> int:
        tree = ET.parse(path)
        root = tree.getroot()
        return sum(1 for obj in root.findall("object") if obj.find("bndbox") is not None)

    def _load_txt(self, path: Path, class_manager: ClassManager, width: int, height: int) -> list[Box]:
        boxes: list[Box] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_token = parts[0]
                cx, cy, bw, bh = map(float, parts[1:5])
                try:
                    class_name = class_manager.ensure_id(int(float(class_token)))
                except Exception:
                    class_name = class_manager.resolve_label_token(class_token)

                if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                    coords = [float(value) for value in parts[1:]]
                    points = [
                        (
                            int(round(coords[index] * width)),
                            int(round(coords[index + 1] * height)),
                        )
                        for index in range(0, len(coords), 2)
                    ]
                    box = Box.from_polygon(points, class_name).clamp(width, height)
                else:
                    x1 = int(round((cx - bw / 2.0) * width))
                    y1 = int(round((cy - bh / 2.0) * height))
                    x2 = int(round((cx + bw / 2.0) * width))
                    y2 = int(round((cy + bh / 2.0) * height))
                    box = Box(x1, y1, x2, y2, class_name).clamp(width, height)
                boxes.append(box)
            except Exception:
                continue
        return boxes

    def _load_xml(self, path: Path, class_manager: ClassManager, width: int, height: int) -> list[Box]:
        tree = ET.parse(path)
        root = tree.getroot()
        boxes: list[Box] = []
        for obj in root.findall("object"):
            name_node = obj.find("name")
            bndbox = obj.find("bndbox")
            if name_node is None or bndbox is None:
                continue
            try:
                class_name = class_manager.resolve_label_token(name_node.text or "object")
                x1 = int(round(float(bndbox.findtext("xmin", "0"))))
                y1 = int(round(float(bndbox.findtext("ymin", "0"))))
                x2 = int(round(float(bndbox.findtext("xmax", "0"))))
                y2 = int(round(float(bndbox.findtext("ymax", "0"))))
                box = Box(x1, y1, x2, y2, class_name).clamp(width, height)
                boxes.append(box)
            except Exception:
                continue
        return boxes

    def _save_json(
        self,
        path: Path,
        boxes: list[Box],
        width: int,
        height: int,
        image_name: str | None,
    ) -> None:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}

        data.setdefault("version", "5.0.1")
        data.setdefault("flags", {})
        data["imagePath"] = image_name or path.stem
        data["imageHeight"] = height
        data["imageWidth"] = width
        shapes: list[dict[str, object]] = []
        for box in boxes:
            if box.is_polygon:
                points = [[int(x), int(y)] for x, y in box.polygon_points()]
                shape_type = "polygon"
            else:
                ordered = box.ordered()
                points = [[int(ordered.x1), int(ordered.y1)], [int(ordered.x2), int(ordered.y2)]]
                shape_type = "rectangle"
            shapes.append(
                {
                    "label": box.class_name,
                    "points": points,
                    "shape_type": shape_type,
                    "flags": {},
                }
            )
        data["shapes"] = shapes
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_txt(
        self,
        path: Path,
        boxes: list[Box],
        width: int,
        height: int,
        class_manager: ClassManager,
    ) -> None:
        lines: list[str] = []
        for box in boxes:
            class_id = class_manager.ensure_name(box.class_name)
            if box.is_polygon:
                coords: list[str] = []
                for x, y in box.polygon_points():
                    coords.append(f"{self._normalize_coordinate(x, width):.6f}")
                    coords.append(f"{self._normalize_coordinate(y, height):.6f}")
                if len(coords) >= 6:
                    lines.append(f"{class_id} {' '.join(coords)}")
                continue

            cx, cy, bw, bh = box.normalized(width, height)
            lines.append(
                f"{class_id} "
                f"{self._clamp01(cx):.6f} {self._clamp01(cy):.6f} "
                f"{self._clamp01(bw):.6f} {self._clamp01(bh):.6f}"
            )
        path.write_text("\n".join(lines), encoding="utf-8")

    def _save_xml(
        self,
        path: Path,
        boxes: list[Box],
        width: int,
        height: int,
        image_name: str | None,
    ) -> None:
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = path.parent.name
        ET.SubElement(root, "filename").text = image_name or path.stem
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"

        for box in boxes:
            ordered = box.ordered()
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = box.class_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bnd = ET.SubElement(obj, "bndbox")
            ET.SubElement(bnd, "xmin").text = str(int(ordered.x1))
            ET.SubElement(bnd, "ymin").text = str(int(ordered.y1))
            ET.SubElement(bnd, "xmax").text = str(int(ordered.x2))
            ET.SubElement(bnd, "ymax").text = str(int(ordered.y2))

        tree = ET.ElementTree(root)
        try:
            ET.indent(tree, space="  ")
        except Exception:
            pass
        tree.write(path, encoding="utf-8", xml_declaration=True)

    def _load_mask_png(self, path: Path, class_manager: ClassManager, width: int, height: int) -> list[Box]:
        cv2 = self._cv2()
        mask = self._read_mask_png(path)
        if mask.shape[1] != width or mask.shape[0] != height:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        boxes: list[Box] = []
        for value in self._mask_values(mask):
            class_name = class_manager.ensure_id(max(0, int(value) - 1))
            for contour in self._mask_contours(mask, value):
                polygon = self._contour_to_polygon(contour)
                if len(polygon) >= 3:
                    try:
                        boxes.append(Box.from_polygon(polygon, class_name).clamp(width, height))
                    except Exception:
                        continue
                    continue

                x, y, contour_width, contour_height = cv2.boundingRect(contour)
                if contour_width > 0 and contour_height > 0:
                    boxes.append(
                        Box(
                            x,
                            y,
                            x + contour_width,
                            y + contour_height,
                            class_name,
                        ).clamp(width, height)
                    )
        return boxes

    def _count_mask_png(self, path: Path, class_id_to_name: dict[int, str]) -> dict[str, int]:
        mask = self._read_mask_png(path)
        counts: dict[str, int] = {}
        for value in self._mask_values(mask):
            class_id = max(0, int(value) - 1)
            class_name = class_id_to_name.get(class_id, f"ID {class_id}")
            count = len(self._mask_contours(mask, value))
            if count > 0:
                counts[class_name] = counts.get(class_name, 0) + count
        return counts

    def _count_shapes_mask_png(self, path: Path) -> int:
        mask = self._read_mask_png(path)
        return sum(len(self._mask_contours(mask, value)) for value in self._mask_values(mask))

    def _save_mask_png(
        self,
        path: Path,
        boxes: list[Box],
        width: int,
        height: int,
        class_manager: ClassManager,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Mask export requires a positive image size")

        cv2 = self._cv2()
        class_ids = [class_manager.ensure_name(box.class_name) for box in boxes]
        max_value = max((int(class_id) + 1 for class_id in class_ids), default=0)
        dtype = np.uint16 if max_value > 255 else np.uint8
        mask = np.zeros((height, width), dtype=dtype)

        for box, class_id in zip(boxes, class_ids):
            points = np.array(box.clamp(width, height).polygon_points(), dtype=np.int32)
            if points.shape[0] < 3:
                continue
            cv2.fillPoly(mask, [points], int(class_id) + 1)

        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            raise ValueError("Failed to encode mask PNG")
        encoded.tofile(str(path))

    def _read_mask_png(self, path: Path) -> np.ndarray:
        cv2 = self._cv2()
        data = np.fromfile(str(path), dtype=np.uint8)
        mask = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Unable to read mask PNG: {path}")
        if mask.ndim == 3:
            if mask.shape[2] == 4:
                mask = mask[:, :, 0]
            else:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return mask

    def _mask_values(self, mask: np.ndarray) -> list[int]:
        values = np.unique(mask)
        return [int(value) for value in values.tolist() if int(value) != 0]

    def _mask_contours(self, mask: np.ndarray, value: int) -> list[np.ndarray]:
        cv2 = self._cv2()
        binary = (mask == value).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours: list[np.ndarray] = []
        for contour in contours:
            _, _, width, height = cv2.boundingRect(contour)
            if width > 0 and height > 0:
                valid_contours.append(contour)
        return valid_contours

    def _contour_to_polygon(self, contour: np.ndarray) -> list[tuple[int, int]]:
        cv2 = self._cv2()
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.0, perimeter * 0.002)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return [(int(point[0][0]), int(point[0][1])) for point in approx]

    def _normalize_coordinate(self, value: int | float, size: int) -> float:
        if size <= 0:
            return 0.0
        return self._clamp01(float(value) / float(size))

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _remap_txt_class_ids(self, path: Path, id_mapping: dict[int, int]) -> bool:
        source_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        changed = False
        updated_lines: list[str] = []

        for raw_line in source_lines:
            stripped = raw_line.strip()
            if not stripped:
                updated_lines.append(raw_line)
                continue

            parts = stripped.split()
            if len(parts) < 5:
                updated_lines.append(raw_line)
                continue

            try:
                current_id = int(float(parts[0]))
            except Exception:
                updated_lines.append(raw_line)
                continue

            next_id = id_mapping.get(current_id)
            if next_id is None:
                updated_lines.append(raw_line)
                continue

            parts[0] = str(int(next_id))
            updated_lines.append(" ".join(parts))
            changed = True

        if changed:
            path.write_text("\n".join(updated_lines), encoding="utf-8")
        return changed
