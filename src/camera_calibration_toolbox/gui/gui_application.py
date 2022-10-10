"""Module that contains the main code for the GUI."""

from PySide6.QtCore import Signal, QThread, QObject
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QListWidgetItem
from camera_calibration_toolbox.core.exceptions import ImageError, TagError
from camera_calibration_toolbox.gui.calibration_app_ui import Ui_CalibrationApp
from camera_calibration_toolbox.core.feature_detection import FeatureDetector
from camera_calibration_toolbox.core.camera_calibration import CameraCalibrator
from camera_calibration_toolbox.core.camera_calibration import CalibrationParameters
import os
import h5py
import cv2
import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import logging
try:
    from DataClass.GenericDataClass import GenericDataClass
except ModuleNotFoundError:
    data_class_available = False
else:
    data_class_available = True


class CalibrationApp(QMainWindow, Ui_CalibrationApp):
    """The main GUI app."""

    def __init__(self) -> None:
        """Class constructor."""
        super().__init__()
        self.setupUi(self)
        self.browseButton.clicked.connect(self.select_files)
        self.calibrateButton.clicked.connect(self.calibrate_camera)
        self.removeOutliersButton.clicked.connect(self.remove_outliers_pressed)
        self.resetButton.clicked.connect(self.reset_pressed)
        self.imageSpinBox.valueChanged.connect(self.spinbox_value_changed)
        self.imageHorizontalSlider.valueChanged.connect(self.slider_value_changed)
        self.exportButton.clicked.connect(self.export_calibration_parameters)

        # Hide some graphicsView functionalities
        self.graphicsView.ui.roiBtn.hide()
        self.graphicsView.ui.menuBtn.hide()
        self.graphicsView.ui.histogram.hide()
        self.image_title = pg.TextItem("", color='r')
        self.graphicsView.view.addItem(self.image_title)
        # self.graphicsView.setMenuEnabled(False)
        # vbox.setMenuEnabled(False)
        # self.graphicsView.scene.contextMenu.hide()

        self.calibrator: CameraCalibrator = CameraCalibrator()
        """Calibrator which will be used for calibration."""

        self.calibration_parameters: CalibrationParameters = CalibrationParameters()
        """Object where all calibration parameters are stored."""

        self.filenames: list = []
        """List of all selected file names."""

        self.filter: str = ''
        """Which filter was used when selecting files."""

        self.image_names: list = []
        """List of all loaded images. These are either the file names or indices."""

        self.logger = logging.getLogger(__name__)
        self.data_class_available = data_class_available
        if not self.data_class_available:
            self.logger.warning("Could not find DataClass module. You will be unable to load SEP and HDF5 files.")

    def select_files(self) -> None:
        """Open file explorer to select files."""

        if self.data_class_available:
            self.filenames, self.filter = QFileDialog.getOpenFileNames(self, "Select calibration file(s).",
                                                                       filter="Image files (*.png *.jpg *.jpeg *.bmp "
                                                                              "*.tiff *.tif);;"
                                                                              "DataClass files (*.h5 *.sep);;"
                                                                              "HDF5 files (*.h5)")
        else:
            self.filenames, self.filter = QFileDialog.getOpenFileNames(self, "Select calibration file(s).",
                                                                       filter="Image files (*.png *.jpg *.jpeg *.bmp "
                                                                              "*.tiff *.tif);;"
                                                                              "HDF5 files (*.h5)")
        self.namesListWidget.clear()
        if self.filenames:
            n_rows = len(self.filenames)
            for idx in range(n_rows):
                QListWidgetItem(os.path.basename(self.filenames[idx]), self.namesListWidget)
            if self.filter.startswith('HDF5'):
                self.headerField.setEnabled(True)
            else:
                self.headerField.setEnabled(False)
                self.headerField.clear()
        else:
            self.headerField.setEnabled(False)
            self.headerField.clear()

    def load_image_data(self, full_path) -> npt.NDArray:

        if self.filter.startswith("Image"):
            image_array = cv2.imread(full_path)
        elif self.filter.startswith("DataClass"):
            try:
                file = GenericDataClass().Open(full_path)
            except KeyError:
                raise ImageError("Selected file is not compatible with the InViLab ``DataClass`` and cannot be loaded.")
            else:
                n_frames = file.numberOfFrames
                image_array = file.GetFrame(0, n_frames - 1)
                file.Close()
        else:
            with h5py.File(full_path, "r") as file:
                key = self.headerField.text()
                try:
                    image_array = file[key][()]
                except KeyError:
                    raise KeyError("Key provided in ``data header field`` is not valid.")

        return image_array

    def calibrate_camera(self) -> None:
        """Perform camera calibration."""
        tag = self.tagField.text()
        if not tag or tag.isspace():
            self.display_error("No tag", "A feature tag is required for calibration.")
        elif not self.filenames:
            self.display_error("No images", "Provide images for calibration.")
        elif self.filter.startswith("HDF5") and not self.headerField.text():
            self.display_error("Missing header", "Provide the header for the image data.")
        else:
            self.statusText.setText("Calibrating")
            self.groupBox.setEnabled(False)
            self.exportButton.setEnabled(False)
            self.tabWidget.setEnabled(False)
            self.tabWidget.setCurrentIndex(0)
            self.reset_image_tab()
            self.reset_reproj_error_tab()
            self.reset_parameters_tab()
            self.calibration_thread = QThread()
            self.calibration_worker = CalibrateCameraWorker(self)
            self.calibration_worker.moveToThread(self.calibration_thread)
            self.calibration_thread.started.connect(self.calibration_worker.run)
            self.calibration_worker.finished.connect(self.calibration_thread.quit)
            self.calibration_worker.finished.connect(self.calibration_worker.deleteLater)
            self.calibration_thread.finished.connect(self.calibration_thread.deleteLater)
            self.calibration_worker.finished.connect(self.calibrate_camera_finished)
            self.calibration_worker.update.connect(self.update_image)
            self.calibration_thread.start()

    def remove_outliers_pressed(self):
        """Perform calibration without the marked calibration images."""
        indices = self.reprojErrPlot.get_good_indices()
        if indices:
            self.calibrate_indices(indices)
        else:
            self.display_error("Calibration error", "You can't remove all images.")

    def reset_pressed(self):
        """Reset the calibration so it includes all available calibration images."""
        self.calibrate_indices([])

    def calibrate_indices(self, indices: list) -> None:
        """Perform calibration with only the selected calibration images."""
        self.statusText.setText('Calibrating')
        self.groupBox.setEnabled(False)
        self.exportButton.setEnabled(False)
        self.tabWidget.setEnabled(False)
        self.calibration_thread = QThread()
        self.calibration_worker = CalibrateIndicesWorker(self, indices)
        self.calibration_worker.moveToThread(self.calibration_thread)
        self.calibration_thread.started.connect(self.calibration_worker.run)
        self.calibration_worker.finished.connect(self.calibration_thread.quit)
        self.calibration_worker.finished.connect(self.calibration_worker.deleteLater)
        self.calibration_thread.finished.connect(self.calibration_thread.deleteLater)
        self.calibration_worker.finished.connect(self.calibrate_camera_finished)
        self.calibration_thread.start()

    def calibrate_camera_finished(self, message):
        """Update the GUI after calibration has finished."""
        if message == 5:
            self.display_error("Unknown error", "An unknown error (probably related to loading the images) occurred!")
        elif message == 4:
            self.display_error("Tag error", "The provided tag is invalid, check the documentation for more info on "
                                            "how to construct feature tags.")
        elif message == 3:
            self.display_error("Image error", "Failed to load data or data has wrong dimensions.")
        elif message == 2:
            self.display_error("Key error", "Provided data header is not valid.")
        else:
            self.update_image_tab()
            if message == 1:
                self.display_error("Calibration error", "Failed to detect specified feature in every image.")
            else:
                self.update_parameters_tab()
                self.update_reproj_error_tab()
                self.exportButton.setEnabled(True)
                if message == 0:
                    self.display_warning("Calibration warning", "At least 11 good images are required for an accurate "
                                                                "calibration.")
        self.tabWidget.setEnabled(True)
        self.groupBox.setEnabled(True)
        self.statusText.setText("Ready")

    def update_image_tab(self):
        """Update the image tab after finishing calibration."""
        n_images = len(self.image_names)
        self.imageSpinBox.setMaximum(n_images)
        self.imageHorizontalSlider.setMaximum(n_images)
        self.maxImageLabel.setText(str(n_images))
        self.imageHorizontalSlider.setTickInterval(n_images)
        self.spinbox_value_changed()
        self.imageSpinBox.setEnabled(True)
        self.imageHorizontalSlider.setEnabled(True)

    def reset_image_tab(self):
        """Reset the image tab."""
        self.imageSpinBox.setEnabled(False)
        self.imageHorizontalSlider.setEnabled(False)
        self.graphicsView.clear()
        self.imageSpinBox.setValue(1)
        self.imageHorizontalSlider.setValue(1)

    def spinbox_value_changed(self):
        """Update image tab after the value of the spinbox is changed."""
        idx = self.imageSpinBox.value() - 1
        self.imageHorizontalSlider.blockSignals(True)
        self.imageHorizontalSlider.setValue(self.imageSpinBox.value())
        self.imageHorizontalSlider.blockSignals(False)
        self.image_tab_value_changed(idx)

    def slider_value_changed(self):
        """Update image tab after the value of the slider is changed."""
        idx = self.imageHorizontalSlider.value() - 1
        self.imageSpinBox.blockSignals(True)
        self.imageSpinBox.setValue(self.imageHorizontalSlider.value())
        self.imageSpinBox.blockSignals(False)
        self.image_tab_value_changed(idx)

    def image_tab_value_changed(self, idx: int) -> None:
        """Update the image tab when a new image is selected."""
        try:
            reproj_err_idx = self.calibrator.indices.index(idx)
            reproj_err = self.calibrator.per_view_err[reproj_err_idx]
            self.reprojErrLabel.setText("Re-projection error = " + str(round(reproj_err, 4)))
        except ValueError:
            self.reprojErrLabel.setText("Re-projection error = /")
        except IndexError:
            reproj_err = float(self.calibrator.per_view_err)
            self.reprojErrLabel.setText("Re-projection error = " + str(round(reproj_err, 4)))
        self.reprojErrLabel.adjustSize()
        image_name = self.image_names[idx]
        self.update_image(self.calibrator.feature_list[idx].feature_image, image_name)

    def update_image(self, image, title: str) -> None:
        """Update the image in the graphics view."""
        self.graphicsView.setImage(np.transpose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                                (1, 0, 2)), levels=(0, 255))
        self.image_title.setText(title)

    def update_reproj_error_tab(self):
        """Update the re-projection error tab after calibration."""
        rms_reproj_error = self.calibrator.rms_reproj_error
        self.RMSreprojErrLabel.setText("RMS re-projection error = " + str(round(rms_reproj_error, 4)))
        self.RMSreprojErrLabel.adjustSize()
        self.reprojErrPlot.plot_reproj_error(self.calibrator.per_view_err, [x+1 for x in self.calibrator.indices],
                                             rms_reproj_error)
        self.removeOutliersButton.setEnabled(True)
        self.resetButton.setEnabled(True)

    def reset_reproj_error_tab(self):
        """Reset the re-projection error tab."""
        self.removeOutliersButton.setEnabled(False)
        self.resetButton.setEnabled(False)
        self.RMSreprojErrLabel.setText("RMS re-projection error = /")
        self.RMSreprojErrLabel.adjustSize()
        self.reprojErrPlot.reset_plot()

    def update_parameters_tab(self):
        """Update the parameters tab after calibration."""
        intrinsics_matrix = self.calibration_parameters.get_intrinsics_matrix_opencv()
        distortion_coeffs = self.calibration_parameters.get_distortion_coeffs_opencv()
        sensor_dimensions = self.calibration_parameters.get_sensor_dimensions()
        self.distortionPlot.plot_distortion(sensor_dimensions, intrinsics_matrix, distortion_coeffs)

        self.intrinsicsTable.item(0, 0).setText(f'{round(self.calibration_parameters.f[0], 4):.4f}')
        self.intrinsicsTable.item(0, 1).setText(f'{round(self.calibration_parameters.f_std[0], 4):.4f}')
        self.intrinsicsTable.item(1, 0).setText(f'{round(self.calibration_parameters.f[1], 4):.4f}')
        self.intrinsicsTable.item(1, 1).setText(f'{round(self.calibration_parameters.f_std[1], 4):.4f}')
        self.intrinsicsTable.item(2, 0).setText(f'{round(self.calibration_parameters.c[0], 4):.4f}')
        self.intrinsicsTable.item(2, 1).setText(f'{round(self.calibration_parameters.c_std[0], 4):.4f}')
        self.intrinsicsTable.item(3, 0).setText(f'{round(self.calibration_parameters.c[1], 4):.4f}')
        self.intrinsicsTable.item(3, 1).setText(f'{round(self.calibration_parameters.c_std[1], 4):.4f}')
        self.intrinsicsTable.item(4, 0).setText(f'{round(self.calibration_parameters.s, 4):.4f}')
        self.intrinsicsTable.item(4, 1).setText(f'{round(self.calibration_parameters.s_std, 4):.4f}')

        self.distortionTable.item(0, 0).setText(f'{round(self.calibration_parameters.radial_dist_coeffs[0], 4):.4f}')
        self.distortionTable.item(0, 1).setText(f'{round(self.calibration_parameters.radial_dist_coeffs_std[0], 4):.4f}')
        self.distortionTable.item(1, 0).setText(f'{round(self.calibration_parameters.radial_dist_coeffs[1], 4):.4f}')
        self.distortionTable.item(1, 1).setText(f'{round(self.calibration_parameters.radial_dist_coeffs_std[1], 4):.4f}')
        self.distortionTable.item(2, 0).setText(f'{round(self.calibration_parameters.radial_dist_coeffs[2], 4):.4f}')
        self.distortionTable.item(2, 1).setText(f'{round(self.calibration_parameters.radial_dist_coeffs_std[2], 4):.4f}')
        self.distortionTable.item(3, 0).setText(f'{round(self.calibration_parameters.tangential_dist_coeffs[0], 4):.4f}')
        self.distortionTable.item(3, 1).setText(f'{round(self.calibration_parameters.tangential_dist_coeffs_std[0], 4):.4f}')
        self.distortionTable.item(4, 0).setText(f'{round(self.calibration_parameters.tangential_dist_coeffs[1], 4):.4f}')
        self.distortionTable.item(4, 1).setText(f'{round(self.calibration_parameters.tangential_dist_coeffs_std[1], 4):.4f}')

    def reset_parameters_tab(self):
        """Reset the parameters tab."""
        for row in range(5):
            for column in range(2):
                self.intrinsicsTable.item(row, column).setText('/')
                self.distortionTable.item(row, column).setText('/')
        self.distortionPlot.reset_plot()

    def export_calibration_parameters(self) -> None:
        """Export the calibration parameters to a file."""
        path = QFileDialog.getSaveFileName(self, "Save camera parameters file.", filter="HDF5 file (*.h5)",
                                           options=QFileDialog.DontConfirmOverwrite)
        if path[0]:
            try:
                self.calibration_parameters.save_parameters(path[0])
            except OSError:
                self.display_error("File error", "You are trying to write the parameters to an incompatible file.")

    def display_error(self, title: str, error_message: str) -> None:
        """"Display an error message in a separate window."""
        error_box = QMessageBox(self)
        error_box.setText(error_message)
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle(title)
        error_box.exec_()

    def display_warning(self, title: str, warning_message: str) -> None:
        """"Display an error message in a separate window."""

        error_box = QMessageBox(self)
        error_box.setText(warning_message)
        error_box.setIcon(QMessageBox.Warning)
        error_box.setWindowTitle(title)
        error_box.exec_()


class CalibrateCameraWorker(QObject):
    """Worker used for performing camera calibration in a separate thread."""

    finished = Signal(int)
    """Signal that fires when the worker is finished."""

    update = Signal(object, str)
    """Signal that fires when the displayed image needs to be updated."""

    def __init__(self, app: CalibrationApp) -> None:
        super().__init__()

        self.app = app
        """The CalibrationApp."""

    def run(self) -> None:
        """Perform standard calibration."""
        message = 5
        try:
            feature_detector = FeatureDetector(self.app.tagField.text())
        except TagError:
            logging.exception("message")
            message = 4
        else:
            self.app.calibrator.feature_list = []
            self.app.image_names = []
            normalize = self.app.normalizeCheckBox.isChecked()
            invert = self.app.invertCheckBox.isChecked()
            try:
                if len(self.app.filenames) == 1:
                    image_array = self.app.load_image_data(self.app.filenames[0])
                    n_dims = len(image_array.shape)
                    if not 3 <= n_dims <= 4:
                        raise ImageError("Image data from single file should be 3- or 4-dimensional.")
                    n_images = image_array.shape[-1]
                    for idx in range(n_images):
                        image = image_array[..., idx]
                        image_name = 'image ' + str(idx)
                        feature = feature_detector.detect_feature(image, normalize, invert)
                        if not feature.score:
                            self.app.logger.info("Failed feature detection on " + image_name + ".")
                        self.app.calibrator.feature_list.append(feature)
                        self.update.emit(feature.feature_image, image_name)
                        self.app.image_names.append(image_name)
                else:
                    n_images = len(self.app.filenames)
                    for idx in range(n_images):
                        image = self.app.load_image_data(self.app.filenames[idx])
                        n_dims = len(image.shape)
                        if not 2 <= n_dims <= 3:
                            raise ImageError("Image data from multiple files should be 2- or 3-dimensional.")
                        image_name = os.path.basename(self.app.filenames[idx])
                        feature = feature_detector.detect_feature(image, normalize, invert)
                        if not feature.score:
                            self.app.logger.info("Failed feature detection on " + image_name + ".")
                        self.app.calibrator.feature_list.append(feature)
                        self.update.emit(feature.feature_image, image_name)
                        self.app.image_names.append(image_name)
            except ImageError:
                logging.exception("message")
                message = 3
            except KeyError:
                logging.exception("message")
                message = 2
            else:
                image_points_list, object_points_list = self.app.calibrator.construct_points_lists([])
                if not image_points_list:
                    message = 1
                else:
                    if len(image_points_list) < 11:
                        message = 0
                    else:
                        message = -1
                    self.app.calibrator.sensor_dimensions = np.array([image.shape[1], image.shape[0]])
                    self.app.calibration_parameters = self.app.calibrator.opencv_calibration(image_points_list,
                                                                                             object_points_list)
        finally:
            self.finished.emit(message)


class CalibrateIndicesWorker(QObject):
    """Worker for repeating the calibration with the specified indices in a separate thread."""

    finished = Signal(int)
    """Signal that fires when the worker is finished."""

    def __init__(self, app: CalibrationApp, indices: list):
        super().__init__()

        self.app = app
        """The CalibrationApp."""

        self.indices = indices
        """Indices that should be used for calibration."""

    def run(self) -> None:
        """Perform calibration with indices."""
        image_points_list, object_points_list = self.app.calibrator.construct_points_lists(self.indices)
        if not image_points_list:
            message = 1
        else:
            if len(image_points_list) < 11:
                message = 0
            else:
                message = -1
            self.app.calibration_parameters = self.app.calibrator.opencv_calibration(image_points_list,
                                                                                     object_points_list)
        self.finished.emit(message)
