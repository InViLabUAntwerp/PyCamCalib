"""Entry point for launching the calibration GUI."""

import sys
from PyCamCalib.logger_configuration import configure_logger


def launch_gui() -> None:
    try:
        from PySide6.QtWidgets import QApplication
        from PyCamCalib.camera_calibration_gui.gui_application import CalibrationApp
    except ImportError as e:
        sys.exit(f"The GUI needs extra dependencies ({e}). Install with: pip install 'PyCamCalib[gui]'")

    app = QApplication(sys.argv)
    main_window = CalibrationApp()
    main_window.show()
    main_window.raise_()
    sys.exit(app.exec())


if __name__ == "__main__":
    configure_logger()
    launch_gui()
