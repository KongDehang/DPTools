from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from ..constants import FORMAT_SUFFIX
from ..models import Box
from .class_manager import ClassManager


class AnnotationIO:
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
        data["shapes"] = [
            {
                "label": box.class_name,
                "points": [[int(box.ordered().x1), int(box.ordered().y1)], [int(box.ordered().x2), int(box.ordered().y2)]],
                "shape_type": "rectangle",
                "flags": {},
            }
            for box in boxes
        ]
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
            cx, cy, bw, bh = box.normalized(width, height)
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
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
