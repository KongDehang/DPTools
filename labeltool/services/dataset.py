from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..constants import IMAGE_EXTENSIONS, LABEL_EXTENSIONS, MASK_LABEL_EXTENSIONS
from ..models import natural_sort_key


class DatasetService:
    def __init__(self) -> None:
        self.root_dir: Path | None = None
        self.image_paths: list[Path] = []
        self.label_paths: list[Path] = []
        self._label_index: dict[str, list[Path]] = defaultdict(list)

    def scan(self, root_dir: str | Path) -> list[Path]:
        root = Path(root_dir)
        self.root_dir = root
        self.image_paths = []
        self.label_paths = []
        self._label_index = defaultdict(list)

        if not root.exists():
            return self.image_paths

        ignored_dirnames = {
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "env",
            "node_modules",
            "build",
            "dist",
        }

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [dirname for dirname in dirnames if dirname not in ignored_dirnames]
            current_dir = Path(dirpath)
            for filename in filenames:
                path = current_dir / filename
                if self._should_ignore(path):
                    continue

                suffix = path.suffix.lower()
                if suffix in LABEL_EXTENSIONS or self._is_mask_label_path(path):
                    self.label_paths.append(path)
                    for key in self._label_index_keys(path):
                        self._label_index[key].append(path)
                elif suffix in IMAGE_EXTENSIONS:
                    self.image_paths.append(path)

        self.image_paths.sort(key=lambda item: natural_sort_key(self._display_key(item)))
        self.label_paths.sort(key=lambda item: natural_sort_key(self._display_key(item)))
        for paths in self._label_index.values():
            paths.sort(key=lambda item: natural_sort_key(self._display_key(item)))
        return self.image_paths

    def apply_scan_result(
        self,
        root_dir: str | Path,
        image_paths: list[Path],
        label_paths: list[Path],
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_paths = list(image_paths)
        self.label_paths = list(label_paths)
        self._label_index = defaultdict(list)

        self.image_paths.sort(key=lambda item: natural_sort_key(self._display_key(item)))
        self.label_paths.sort(key=lambda item: natural_sort_key(self._display_key(item)))
        for path in self.label_paths:
            for key in self._label_index_keys(path):
                self._label_index[key].append(path)
        for paths in self._label_index.values():
            paths.sort(key=lambda item: natural_sort_key(self._display_key(item)))

    def image_count(self) -> int:
        return len(self.image_paths)

    def label_count(self) -> int:
        return len(self.label_paths)

    def find_label_for_image(self, image_path: str | Path) -> Path | None:
        image = Path(image_path)
        candidates: list[Path] = []

        for suffix in LABEL_EXTENSIONS:
            sibling = image.with_suffix(suffix)
            if sibling.exists():
                return sibling

        candidates.extend(self._label_index.get(image.stem.lower(), []))
        if not candidates:
            return None

        return max(candidates, key=lambda candidate: self._candidate_score(image, candidate))

    def display_name(self, path: str | Path) -> str:
        candidate = Path(path)
        if self.root_dir is None:
            return candidate.name
        try:
            return str(candidate.relative_to(self.root_dir))
        except Exception:
            return candidate.name

    def _candidate_score(self, image_path: Path, label_path: Path) -> tuple[int, int, int]:
        score = 0
        if label_path.parent == image_path.parent:
            score += 100
        if label_path.stem.lower() == image_path.stem.lower():
            score += 20
        if label_path.parent.name.lower() in {
            "label",
            "labels",
            "annot",
            "annotation",
            "annotations",
            "ann",
            "mask",
            "masks",
            "seg",
            "segs",
            "segmentation",
            "segmentations",
        }:
            score += 10
        if label_path.suffix.lower() in LABEL_EXTENSIONS or self._is_mask_label_path(label_path):
            score += 1
        if label_path.stem.lower() == f"{image_path.stem.lower()}_mask":
            score += 15
        return score, -len(label_path.parts), -len(label_path.name)

    def _display_key(self, path: Path) -> str:
        try:
            if self.root_dir is not None:
                return str(path.relative_to(self.root_dir))
        except Exception:
            pass
        return path.name

    def _should_ignore(self, path: Path) -> bool:
        ignored_parts = {
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "env",
            "node_modules",
            "build",
            "dist",
        }
        return any(part in ignored_parts for part in path.parts)

    def _label_index_keys(self, path: Path) -> tuple[str, ...]:
        stem = path.stem.lower()
        keys = [stem]
        for suffix in ("_mask", "-mask"):
            if stem.endswith(suffix) and len(stem) > len(suffix):
                keys.append(stem[: -len(suffix)])
        return tuple(dict.fromkeys(keys))

    def _is_mask_label_path(self, path: Path) -> bool:
        if path.suffix.lower() not in MASK_LABEL_EXTENSIONS:
            return False

        label_parts = {"label", "labels", "mask", "masks", "seg", "segs", "segmentation", "segmentations"}
        image_parts = {"image", "images", "img", "imgs"}
        lowered_parts = {part.lower() for part in path.parts}
        if lowered_parts & label_parts:
            return True
        if lowered_parts & image_parts:
            return False
        return path.stem.lower().endswith(("_mask", "-mask"))
