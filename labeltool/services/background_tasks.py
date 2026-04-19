from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QImage

from ..constants import IMAGE_EXTENSIONS
from .annotation_io import AnnotationIO
from .dataset import DatasetService
from .class_manager import ClassManager


@dataclass(slots=True)
class DatasetLoadResult:
    root_dir: Path
    image_paths: list[Path]
    label_paths: list[Path]
    class_counts: dict[str, int]
    total_images: int
    annotated_images: int
    total_boxes: int


class DatasetScanWorker(QObject):
    progressChanged = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, root_dir: str | Path, class_id_to_name: dict[int, str]) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.class_id_to_name = dict(class_id_to_name)

    def run(self) -> None:
        try:
            dataset_service = DatasetService()
            dataset_service.scan(self.root_dir)
            annotation_io = AnnotationIO()
            class_manager = ClassManager()
            class_manager.load_from_root(self.root_dir)
            class_id_to_name = dict(class_manager.id_to_name)
            for class_id, class_name in self.class_id_to_name.items():
                if not class_name:
                    continue
                existing_name = class_id_to_name.get(class_id)
                if existing_name is None or existing_name.startswith(("unknown_id_", "ID ")):
                    class_id_to_name[class_id] = class_name
            class_counts: dict[str, int] = {}
            annotated_images = 0
            total_boxes = 0
            total_images = len(dataset_service.image_paths)

            if total_images == 0:
                self.finished.emit(
                    DatasetLoadResult(
                        root_dir=self.root_dir,
                        image_paths=[],
                        label_paths=[],
                        class_counts={},
                        total_images=0,
                        annotated_images=0,
                        total_boxes=0,
                    )
                )
                return

            for index, image_path in enumerate(dataset_service.image_paths):
                label_path = dataset_service.find_label_for_image(image_path)
                if label_path is not None:
                    counts, box_count = annotation_io.count_annotation(label_path, class_id_to_name)
                    if box_count > 0:
                        annotated_images += 1
                        total_boxes += box_count
                        for class_name, count in counts.items():
                            class_counts[class_name] = class_counts.get(class_name, 0) + count
                if index % 20 == 0 or index == total_images - 1:
                    self.progressChanged.emit(index + 1, total_images)

            self.finished.emit(
                DatasetLoadResult(
                    root_dir=self.root_dir,
                    image_paths=list(dataset_service.image_paths),
                    label_paths=list(dataset_service.label_paths),
                    class_counts=class_counts,
                    total_images=total_images,
                    annotated_images=annotated_images,
                    total_boxes=total_boxes,
                )
            )
        except Exception as exc:  # pragma: no cover - background error path
            self.failed.emit(str(exc))


class ThumbnailLoadWorker(QObject):
    thumbnailReady = pyqtSignal(int, object)
    progressChanged = pyqtSignal(int, int)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, image_paths: list[Path], thumbnail_size: tuple[int, int] = (128, 88)) -> None:
        super().__init__()
        self.image_paths = list(image_paths)
        self.thumbnail_size = thumbnail_size

    def run(self) -> None:
        try:
            total = len(self.image_paths)
            if total == 0:
                self.finished.emit()
                return

            thumb_width, thumb_height = self.thumbnail_size
            for index, image_path in enumerate(self.image_paths):
                if not image_path.exists():
                    continue
                try:
                    data = np.fromfile(str(image_path), dtype=np.uint8)
                    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    height, width = rgb.shape[:2]
                    qimage = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
                    thumbnail = qimage.scaled(
                        thumb_width,
                        thumb_height,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.FastTransformation,
                    )
                    self.thumbnailReady.emit(index, thumbnail)
                except Exception:
                    continue

                if index % 20 == 0 or index == total - 1:
                    self.progressChanged.emit(index + 1, total)

            self.finished.emit()
        except Exception as exc:  # pragma: no cover - background error path
            self.failed.emit(str(exc))
