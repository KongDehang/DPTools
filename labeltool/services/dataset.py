from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..constants import IMAGE_EXTENSIONS, LABEL_EXTENSIONS
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

        for path in root.rglob("*"):
            if not path.is_file() or self._should_ignore(path):
                continue

            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                self.image_paths.append(path)
            elif suffix in LABEL_EXTENSIONS:
                self.label_paths.append(path)
                self._label_index[path.stem.lower()].append(path)

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
            self._label_index[path.stem.lower()].append(path)
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
        if label_path.parent.name.lower() in {"label", "labels", "annot", "annotation", "annotations", "ann"}:
            score += 10
        if label_path.suffix.lower() in LABEL_EXTENSIONS:
            score += 1
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
