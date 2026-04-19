from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QObject, QTimer


class AutosaveController(QObject):
    def __init__(self, interval_ms: int = 2500, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._interval_ms = max(500, int(interval_ms))
        self._enabled = True
        self._dirty = False
        self._save_callback: Callable[[], bool] | None = None

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.flush)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def interval_ms(self) -> int:
        return self._interval_ms

    def set_save_callback(self, callback: Callable[[], bool]) -> None:
        self._save_callback = callback

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not self._enabled:
            self._timer.stop()
        elif self._dirty:
            self._timer.start(self._interval_ms)

    def set_interval_ms(self, interval_ms: int) -> None:
        self._interval_ms = max(500, int(interval_ms))
        if self._enabled and self._dirty:
            self._timer.start(self._interval_ms)

    def mark_dirty(self) -> None:
        self._dirty = True
        if self._enabled:
            self._timer.start(self._interval_ms)

    def clear(self) -> None:
        self._dirty = False
        self._timer.stop()

    def flush(self) -> bool:
        if not self._dirty:
            return True
        if self._save_callback is None:
            return False
        try:
            saved = bool(self._save_callback())
        except Exception:
            return False
        if saved:
            self.clear()
        return saved
