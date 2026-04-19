from __future__ import annotations

from dataclasses import dataclass

from ..models import Box
from .class_manager import ClassManager, ClassManagerState


@dataclass(slots=True)
class DocumentSnapshot:
    boxes: list[Box]
    selected_index: int
    class_state: ClassManagerState

    @classmethod
    def capture(
        cls,
        boxes: list[Box],
        selected_index: int,
        class_manager: ClassManager,
    ) -> "DocumentSnapshot":
        return cls(
            boxes=[box.copy() for box in boxes],
            selected_index=int(selected_index),
            class_state=class_manager.snapshot(),
        )


@dataclass(slots=True)
class HistoryEntry:
    before: DocumentSnapshot
    after: DocumentSnapshot


class DocumentHistory:
    def __init__(self, limit: int = 64) -> None:
        self.limit = max(8, int(limit))
        self._undo_stack: list[HistoryEntry] = []
        self._redo_stack: list[HistoryEntry] = []
        self._pending_before: DocumentSnapshot | None = None

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._pending_before = None

    def clear_pending(self) -> None:
        self._pending_before = None

    def begin(self, boxes: list[Box], selected_index: int, class_manager: ClassManager) -> None:
        self._pending_before = DocumentSnapshot.capture(boxes, selected_index, class_manager)

    def commit(self, boxes: list[Box], selected_index: int, class_manager: ClassManager) -> bool:
        if self._pending_before is None:
            return False

        before = self._pending_before
        after = DocumentSnapshot.capture(boxes, selected_index, class_manager)
        self._pending_before = None

        if before == after:
            return False

        self._undo_stack.append(HistoryEntry(before=before, after=after))
        if len(self._undo_stack) > self.limit:
            self._undo_stack = self._undo_stack[-self.limit :]
        self._redo_stack.clear()
        return True

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> DocumentSnapshot | None:
        self._pending_before = None
        if not self._undo_stack:
            return None
        entry = self._undo_stack.pop()
        self._redo_stack.append(entry)
        return entry.before

    def redo(self) -> DocumentSnapshot | None:
        self._pending_before = None
        if not self._redo_stack:
            return None
        entry = self._redo_stack.pop()
        self._undo_stack.append(entry)
        return entry.after
