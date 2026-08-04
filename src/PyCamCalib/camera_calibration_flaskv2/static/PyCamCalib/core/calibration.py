"""Module that contains all code for camera calibration (Browser-Optimized Single-Threaded)."""

from __future__ import annotations
import os
import gc
import sys
import cv2
import h5py
import logging
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from .exceptions import CalibrationError
from .feature_detection import FeatureDetector
try:
    from CTPv.Transformation.TransformationMatrix import TransformationMatrix
except:
    from core_toolbox_python.Transformation.TransformationMatrix import TransformationMatrix
from .CameraParameters import CameraParameters


# --- SIMPLE PICKLABLE WRAPPER ---
class SimpleFeature:
    __slots__ = ["score", "image_points", "object_points"]

    def __init__(self, score, image_points, object_points):
        self.score = score
        self.image_points = image_points
        self.object_points = object_points


# ---------------------------------------------------------------------------
# SYNCHRONOUS FEATURE DETECTION HELPER
# ---------------------------------------------------------------------------

def _detect_single_image(args):
    idx, img, space_between_features, board_size, marker, kwargs_dict, show_processing, detector_type, detector_params = args
    
    if detector_type == "charuco":
        from .CharucoFeatureDetector import CharucoFeatureDetector
        preset = detector_params.get("preset")
        board = detector_params.get("board")
        if board is None and preset is not None:
            board = CharucoFeatureDetector.get_preset_board(preset)
        detector = CharucoFeatureDetector(
            space_between_features,
            preset=preset,
            board=board,
            **kwargs_dict
        )
    else:
        from .feature_detection import FeatureDetector
        detector = FeatureDetector(space_between_features, board_size, marker, **kwargs_dict)

    if hasattr(detector, 'detector') and hasattr(detector.detector, 'checkerboard_detector'):
        detector.detector.checkerboard_detector.detector.show_processing = show_processing

    feature = detector.detect_feature(img)

    try:
        img_pts = (
            np.array(feature.image_points, copy=True)
            if getattr(feature, "image_points", None) is not None
            else None
        )
        obj_pts = (
            np.array(feature.object_points, copy=True)
            if getattr(feature, "object_points", None) is not None
            else None
        )
        score = feature.score
    except Exception:
        img_pts = None
        obj_pts = None
        score = -1

    return idx, SimpleFeature(score, img_pts, obj_pts)


# ---------------------------------------------------------------------------
# CameraCalibrator CLASS
# ---------------------------------------------------------------------------

class CameraCalibrator:
    """Object used to calibrate a camera."""

    def __init__(self) -> None:
        """Class constructor."""
        self._logger = logging.getLogger(__name__)
        self.sensor_dimensions = np.zeros(2, dtype=np.int32)
        self.feature_list: list = []
        self.indices: list = []
        self.image_points_list: list = []
        self.object_points_list: list = []
        self.per_view_err: npt.NDArray[np.float64] = np.zeros(1)
        self.rms_reproj_error: np.float64 = np.float64(0)
        self.r_vecs: npt.NDArray[np.float64] = np.zeros((1, 3))
        self.t_vecs: npt.NDArray[np.float64] = np.zeros((1, 3))
        self.extrinsics_std: npt.NDArray[np.float64] = np.zeros(1)
        self.camera_parameters: CameraParameters = CameraParameters()
        self.FeatureDetector = None
        self.FeatureDetector_logging = False

    def calibrate(self, image_array: npt.NDArray, space_between_features: float,
                  board_size: Optional[Tuple[int, int]] = None,
                  marker: Optional[Tuple[int, int]] = None,
                  detector_type: str = "checkerboard",
                  detector_params: Optional[dict] = None,
                  absolute: bool = False, **kwargs) -> CameraParameters:

        if detector_params is None:
            detector_params = {}

        self._logger.info(f"Detecting features using {detector_type}")
        self.construct_feature_list(
            image_array, space_between_features, board_size, marker,
            detector_type=detector_type,
            detector_params=detector_params,
            **kwargs
        )
        
        if not isinstance(image_array, np.ndarray):
            raise TypeError("``image_array`` should be a numpy array.")
        if absolute:
            if board_size is None:
                raise TypeError("If `absolute` is set to True `board_size` needs to be known.")
        if marker is not None:
            raise NotImplementedError("Marked checkerboards have not been implemented.")

        sensor_dimensions = np.array([image_array.shape[1], image_array.shape[0]])

        self.construct_points_lists([], absolute)
        if not self.image_points_list:
            raise CalibrationError("Failed to detect features in all images, unable to perform calibration.")
        elif len(self.image_points_list) < 11:
            self._logger.warning("Only detected features for " + str(len(self.image_points_list))
                                 + " images. Features from at least 11 images are necessary for an accurate calibration.")
        self._logger.info("Performing calibration")
        camera_parameters = self.opencv_calibration(sensor_dimensions)

        return camera_parameters

    def construct_feature_list(self, image_array, space_between_features,
                               board_size=None, marker=None, detector_type="checkerboard",
                               detector_params=None, **kwargs):
        image_array = np.ascontiguousarray(image_array)
        n_images = image_array.shape[-1]

        # Synchronous sequential loop for WebAssembly compatibility
        results = []
        for i in range(n_images):
            task = (i, image_array[..., i], space_between_features, board_size, marker, dict(kwargs), self.FeatureDetector_logging, detector_type, detector_params or {})
            results.append(_detect_single_image(task))

        results.sort(key=lambda x: x[0])
        self.feature_list = [feature for _, feature in results]

    def construct_points_lists(self, indices: list, absolute: bool = False) -> None:
        """Construct lists of image points and object points for calibration."""
        if absolute:
            minimum_score = 2
        else:
            minimum_score = 1

        self.indices = []
        self.object_points_list = []
        self.image_points_list = []
        if not indices:
            indices = range(len(self.feature_list))
        for idx in indices:
            try:
                feature = self.feature_list[idx]
            except IndexError:
                pass
            else:
                if feature.score >= minimum_score:
                    self.image_points_list.append(feature.image_points)
                    self.object_points_list.append(feature.object_points)
                    self.indices.append(idx)

    def initilize_camera_parameters(self, fx, fy, cx, cy):
        self.camera_parameters = CameraParameters()
        self.camera_parameters.fx = fx
        self.camera_parameters.fy = fy
        self.camera_parameters.cx = cx
        self.camera_parameters.cy = cy

    def opencv_calibration(self, sensor_dimensions: npt.NDArray[np.int32]) -> CameraParameters:
        """Regular OpenCV camera calibration."""
        self.sensor_dimensions = sensor_dimensions

        if self.camera_parameters is None:
            cameraMatrix = None
        else:
            cameraMatrix = self.camera_parameters.get_intrinsics_matrix_opencv()
        rms_reproj_error, intrinsics_matrix, dist_coeffs, r_vecs, t_vecs, intrinsics_std, extrinsics_std, \
        per_view_err = cv2.calibrateCameraExtended(self.object_points_list, self.image_points_list, sensor_dimensions,
                                                   cameraMatrix, None)

        self.rms_reproj_error = np.float64(rms_reproj_error)
        self.per_view_err = np.squeeze(per_view_err)
        dist_coeffs = np.squeeze(dist_coeffs)
        intrinsics_std = np.squeeze(intrinsics_std)
        self.r_vecs = np.squeeze(np.array(r_vecs))
        self.t_vecs = np.squeeze(np.array(t_vecs))
        self.extrinsics_std = extrinsics_std

        if self.camera_parameters is None:
            self.camera_parameters = CameraParameters()
        self.camera_parameters.set_parameters_opencv(self.rms_reproj_error, intrinsics_matrix, dist_coeffs,
                                                     intrinsics_std, self.sensor_dimensions)

        return self.camera_parameters

    def calibrate_indices(self, indices: list, absolute: bool = False) -> CameraParameters:
        """Repeat calibration with selected samples."""
        if not isinstance(indices, list):
            raise TypeError("``indices`` should be a list.")

        self.construct_points_lists(indices, absolute)
        if not self.image_points_list:
            raise CalibrationError("No images with detected features remain, unable to perform calibration.")
        if len(self.image_points_list) < 11:
            self._logger.warning("Only " + str(len(self.image_points_list))
                                 + " samples left. At least 11 samples are necessary for an accurate calibration.")

        camera_parameters = self.opencv_calibration(self.sensor_dimensions)

        return camera_parameters

    def plot_and_filter_reproj_error(self) -> list:
        info = "Select high rms, and press exit"
        indices = self.indices
        per_view_err = self.per_view_err
        fig, ax = plt.subplots()
        bars = ax.bar(indices, per_view_err)
        ax.set_xticks(indices)
        ax.set_xticklabels(indices, rotation=90, fontsize=8)
        plt.xlabel("Image index")
        plt.ylabel("Reprojection error")
        plt.title("Reprojection error for each detected image")
        line = plt.axhline(self.rms_reproj_error, color='g', linestyle='--')
        plt.legend([bars, line], ['Image', 'RMS'], ncols=2)

        selected_indices = []

        def on_click(event):
            for i, bar in enumerate(bars):
                if bar.contains(event)[0]:
                    bar.set_color('r')
                    selected_indices.append(indices[i])
                    fig.canvas.draw()

        def on_key(event):
            if event.key == 'escape':
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=True)

        not_selected_indices = [index for index in indices if index not in selected_indices]
        return not_selected_indices

    def plot_reproj_error(self) -> None:
        """Plot mean re-projection error and re-projection error for each calibration image."""
        plt.cla()
        bars = plt.bar(list(map(str, self.indices)), self.per_view_err, color='b')
        line = plt.axhline(self.rms_reproj_error, color='g', linestyle='--')
        plt.xlabel("Image index")
        plt.ylabel("Reprojection error")
        plt.title("Reprojection error for each detected image")
        plt.legend([bars, line], ['Image', 'RMS'], ncols=2)
        plt.show()

    def calibrate_world_coordinate_system(self, image, space_between_features: float,
                                  board_size: Optional[Tuple[int, int]] = None,
                                  marker: Optional[Tuple[int, int]] = None,
                                  absolute: bool = False, **kwargs):
        """Calibrate world coordinate system from checkerboard detection."""
        image_array = image
        sensor_dimensions = np.array([image_array.shape[1], image_array.shape[0]])
        self.construct_feature_list(image_array, space_between_features, board_size, marker, **kwargs)
        self.construct_points_lists([])

        results = []
        for obj_points, img_points_1 in zip(self.object_points_list, self.image_points_list):
            _, rvec_1, tvec_1 = cv2.solvePnP(obj_points, img_points_1,
                                            self.camera_parameters.get_intrinsics_matrix_opencv(),
                                            self.camera_parameters.get_distortion_coeffs_opencv())

            reprojected_points, _ = cv2.projectPoints(obj_points, rvec_1, tvec_1,
                                                      self.camera_parameters.get_intrinsics_matrix_opencv(),
                                                      self.camera_parameters.get_distortion_coeffs_opencv())
            reprojected_points = reprojected_points.reshape(-1, 2)
            error = cv2.norm(img_points_1, reprojected_points, cv2.NORM_L2) / np.sqrt(len(reprojected_points))

            R, _ = cv2.Rodrigues(rvec_1)
            H = np.hstack((R, tvec_1))
            H = np.vstack((H, [0, 0, 0, 1]))
            results.append((H, error))

        return results

    def save_checkerboard_detection_to_images(self, image_array, path):
        """Save checkerboard detection results to images sequentially."""
        if not os.path.exists(path):
            os.makedirs(path)

        n_images = image_array.shape[-1]
        is_color = len(image_array.shape) == 4

        for image_idx in range(n_images):
            image = image_array[..., image_idx].copy()
            if is_color:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            if image_idx in self.indices:
                feature_index = self.indices.index(image_idx)
                used_pts = [tuple(p) for p in self.image_points_list[feature_index]]
                for pt in used_pts:
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), 1)
            elif self.feature_list[image_idx].score != 0:
                det_pts = [tuple(p) for p in self.feature_list[image_idx].image_points]
                for pt in det_pts:
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), 1)

            cv2.putText(
                image,
                'detected with pycbd, InViLab, doi:10.3390/math11224568',
                (10, image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
            filename = os.path.join(path, f'image_{image_idx + 1}.png')
            cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# StereoCalibrator CLASS (Stubbed / Simplified for single-threaded web)
# ---------------------------------------------------------------------------

class StereoCalibrator:
    """Object used to perform stereo calibration."""
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self.feature_list_1: list = []
        self.feature_list_2: list = []
        self.indices: list = []
        self.image_points_list_1: list = []
        self.image_points_list_2: list = []
        self.object_points_list: list = []
        self.stereo_parameters = StereoParameters()
        self.r_vecs: npt.NDArray[np.float64] = np.zeros((1, 3))
        self.t_vecs: npt.NDArray[np.float64] = np.zeros((1, 3))
        self.per_view_err: npt.NDArray[np.float64] = np.zeros(1)
        self.rms_reproj_error: np.float64 = np.float64(0)