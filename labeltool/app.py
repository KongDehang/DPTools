from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    from labeltool.constants import APP_NAME, APP_ORG, STYLE_SHEET
    from labeltool.ui.main_window import AnnotationMainWindow
else:
    from .constants import APP_NAME, APP_ORG, STYLE_SHEET
    from .ui.main_window import AnnotationMainWindow


def _resolve_style_sheet() -> str:
    icons_dir = Path(__file__).resolve().parent / "assets" / "icons"

    def qss_path(path: Path) -> str:
        return str(path.resolve()).replace("\\", "/")

    replacements = {
        "__ICON_CHECK__": qss_path(icons_dir / "checkmark.png"),
        "__ICON_CHEVRON_DOWN__": qss_path(icons_dir / "chevron_down.png"),
    }
    style = STYLE_SHEET
    for token, uri in replacements.items():
        style = style.replace(token, uri)
    return style


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    app.setStyleSheet(_resolve_style_sheet())

    window = AnnotationMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
