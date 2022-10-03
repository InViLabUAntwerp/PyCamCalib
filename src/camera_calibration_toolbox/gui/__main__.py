"""Entry point for launching the calibration GUI."""

import sys
from PySide6.QtWidgets import QApplication
from camera_calibration_toolbox.gui.gui_application import CalibrationApp
from camera_calibration_toolbox.package_logger import configure_logger


def main() -> None:
    app = QApplication(sys.argv)
    main_window = CalibrationApp()
    main_window.show()
    main_window.raise_()
    sys.exit(app.exec())


if __name__ == "__main__":
    configure_logger()
    main()
