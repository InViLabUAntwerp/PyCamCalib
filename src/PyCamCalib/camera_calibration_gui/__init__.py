"""This sub-package contains the code for the GUI."""
from pkg_resources import resource_filename
import os
import sys
from PySide6.QtWidgets import QApplication
from PyCamCalib.camera_calibration_gui.gui_application import CalibrationApp
import importlib

invilab_logo = resource_filename(__name__, os.path.join('resources', 'Logo.jpg'))
invilab_icon = resource_filename(__name__, os.path.join('resources', 'Logo_InViLab_Icon_color.jpg'))


def launch_gui() -> None:
    app = QApplication(sys.argv)
    main_window = CalibrationApp()
    main_window.show()
    main_window.raise_()
    sys.exit(app.exec())
