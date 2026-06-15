"""Module that contains all code for camera calibration."""

from __future__ import annotations
import os
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

from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# --- SIMPLE PICKLABLE WRAPPER ---
class SimpleFeature:
    __slots__ = ["score", "image_points", "object_points"]

    def __init__(self, score, image_points, object_points):
        self.score = score
        self.image_points = image_points
        self.object_points = object_points


# ---------------------------------------------------------------------------
# OPTIMIZED CAMERA FEATURE DETECTION WORKERS
# ---------------------------------------------------------------------------

_SHM_CAM = None
_SHARED_CAM_ARRAY = None
_DETECTOR_CAM = None

def _init_camera_worker(shm_name, shape, dtype, space_between_features, board_size, marker, kwargs_dict, show_processing):
    """Initializer for CameraCalibrator workers. Attaches to shared memory and creates detector once."""
    global _SHM_CAM, _SHARED_CAM_ARRAY, _DETECTOR_CAM
    _SHM_CAM = shared_memory.SharedMemory(name=shm_name)
    _SHARED_CAM_ARRAY = np.ndarray(shape, dtype=dtype, buffer=_SHM_CAM.buf)

    from .feature_detection import FeatureDetector
    _DETECTOR_CAM = FeatureDetector(space_between_features, board_size, marker, **kwargs_dict)
    _DETECTOR_CAM.detector.checkerboard_detector.detector.show_processing = show_processing

def _camera_worker(idx):
    """Worker function for single camera feature detection. Only receives index."""
    global _SHARED_CAM_ARRAY, _DETECTOR_CAM

    feature = _DETECTOR_CAM.detect_feature(_SHARED_CAM_ARRAY[..., idx])

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
# OPTIMIZED STEREO FEATURE DETECTION WORKERS
# ---------------------------------------------------------------------------

_SHM_CAM1 = None
_SHM_CAM2 = None
_SHARED_CAM1_ARRAY = None
_SHARED_CAM2_ARRAY = None
_DETECTOR_STEREO = None

def _init_stereo_worker(shm_name_1, shape_1, dtype_1, shm_name_2, shape_2, dtype_2,
                        space_between_features, board_size, marker, kwargs_dict, show_processing):
    """Initializer for StereoCalibrator workers. Attaches to both shared memory blocks and creates detector once."""
    global _SHM_CAM1, _SHM_CAM2, _SHARED_CAM1_ARRAY, _SHARED_CAM2_ARRAY, _DETECTOR_STEREO

    _SHM_CAM1 = shared_memory.SharedMemory(name=shm_name_1)
    _SHARED_CAM1_ARRAY = np.ndarray(shape_1, dtype=dtype_1, buffer=_SHM_CAM1.buf)
    _SHM_CAM2 = shared_memory.SharedMemory(name=shm_name_2)
    _SHARED_CAM2_ARRAY = np.ndarray(shape_2, dtype=dtype_2, buffer=_SHM_CAM2.buf)

    from .feature_detection import FeatureDetector
    _DETECTOR_STEREO = FeatureDetector(space_between_features, board_size, marker, **kwargs_dict)
    _DETECTOR_STEREO.detector.checkerboard_detector.detector.show_processing = show_processing

def _stereo_worker(args):
    """Worker function for stereo camera feature detection. Receives only (idx, cam_id)."""
    idx, cam_id = args
    global _SHARED_CAM1_ARRAY, _SHARED_CAM2_ARRAY, _DETECTOR_STEREO

    image_array = _SHARED_CAM1_ARRAY if cam_id == 0 else _SHARED_CAM2_ARRAY
    feature = _DETECTOR_STEREO.detect_feature(image_array[..., idx])

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

    return idx, cam_id, SimpleFeature(score, img_pts, obj_pts)


# ---------------------------------------------------------------------------
# OPTIMIZED PnP ERROR WORKERS
# ---------------------------------------------------------------------------

_PNP_ALL_OBJ = None
_PNP_ALL_IMG1 = None
_PNP_ALL_IMG2 = None
_PNP_K1 = None
_PNP_d1 = None
_PNP_K2 = None
_PNP_d2 = None
_PNP_R = None
_PNP_T = None

def _init_pnp_worker(shm_obj_name, shm_obj_shape, shm_obj_dtype,
                     shm_img1_name, shm_img1_shape, shm_img1_dtype,
                     shm_img2_name, shm_img2_shape, shm_img2_dtype,
                     K1_flat, d1_flat, K2_flat, d2_flat, R_flat, T_flat):
    """Initializer for PnP workers. Attaches to all shared memory and reconstructs camera matrices once."""
    global _PNP_ALL_OBJ, _PNP_ALL_IMG1, _PNP_ALL_IMG2
    global _PNP_K1, _PNP_d1, _PNP_K2, _PNP_d2, _PNP_R, _PNP_T
    global _SHM_PNP_OBJ, _SHM_PNP_IMG1, _SHM_PNP_IMG2

    _SHM_PNP_OBJ = shared_memory.SharedMemory(name=shm_obj_name)
    _PNP_ALL_OBJ = np.ndarray(shm_obj_shape, dtype=shm_obj_dtype, buffer=_SHM_PNP_OBJ.buf)
    _SHM_PNP_IMG1 = shared_memory.SharedMemory(name=shm_img1_name)
    _PNP_ALL_IMG1 = np.ndarray(shm_img1_shape, dtype=shm_img1_dtype, buffer=_SHM_PNP_IMG1.buf)
    _SHM_PNP_IMG2 = shared_memory.SharedMemory(name=shm_img2_name)
    _PNP_ALL_IMG2 = np.ndarray(shm_img2_shape, dtype=shm_img2_dtype, buffer=_SHM_PNP_IMG2.buf)

    _PNP_K1 = np.array(K1_flat, dtype=np.float64).reshape(3, 3)
    _PNP_d1 = np.array(d1_flat, dtype=np.float64)
    _PNP_K2 = np.array(K2_flat, dtype=np.float64).reshape(3, 3)
    _PNP_d2 = np.array(d2_flat, dtype=np.float64)
    _PNP_R = np.array(R_flat, dtype=np.float64).reshape(3, 3)
    _PNP_T = np.array(T_flat, dtype=np.float64).reshape(3, 1)

def _pnp_worker(args):
    """Compute 3-D reprojection error for one checkerboard view. Receives only slice indices."""
    obj_start, obj_end, img1_start, img1_end, img2_start, img2_end = args
    global _PNP_ALL_OBJ, _PNP_ALL_IMG1, _PNP_ALL_IMG2
    global _PNP_K1, _PNP_d1, _PNP_K2, _PNP_d2, _PNP_R, _PNP_T

    obj_points   = _PNP_ALL_OBJ[obj_start:obj_end]
    img_points_1 = _PNP_ALL_IMG1[img1_start:img1_end]
    img_points_2 = _PNP_ALL_IMG2[img2_start:img2_end]

    _, rvec_1, tvec_1 = cv2.solvePnP(obj_points, img_points_1, _PNP_K1, _PNP_d1)
    _, rvec_2, tvec_2 = cv2.solvePnP(obj_points, img_points_2, _PNP_K2, _PNP_d2)

    R1, _ = cv2.Rodrigues(rvec_1)
    R2, _ = cv2.Rodrigues(rvec_2)

    ones = np.ones((obj_points.shape[0], 1), dtype=np.float64)
    obj_h = np.hstack((obj_points, ones))

    T1 = np.eye(4, dtype=np.float64)
    T1[:3, :3] = R1
    T1[:3, 3] = tvec_1.flatten()
    pts_cam1_h = (T1 @ obj_h.T).T
    pts_cam1 = pts_cam1_h[:, :3] / pts_cam1_h[:, 3:4]

    T2 = np.eye(4, dtype=np.float64)
    T2[:3, :3] = R2
    T2[:3, 3] = tvec_2.flatten()
    pts_cam2_h = (T2 @ obj_h.T).T

    R_inv = _PNP_R.T
    T_inv = -R_inv @ _PNP_T
    T3 = np.eye(4, dtype=np.float64)
    T3[:3, :3] = R_inv
    T3[:3, 3] = T_inv.flatten()
    pts_cam2_in1_h = (T3 @ pts_cam2_h.T).T
    pts_cam2_in1 = pts_cam2_in1_h[:, :3] / pts_cam2_in1_h[:, 3:4]

    error = float(np.mean(np.linalg.norm(pts_cam1 - pts_cam2_in1, axis=1)))
    return error, pts_cam1, pts_cam2_in1


def _pack_point_list(point_list):
    """Concatenate a list of (Ni, D) arrays into one contiguous float64 array stored in shared memory."""
    arrays  = [np.ascontiguousarray(a, dtype=np.float64) for a in point_list]
    lengths = [a.shape[0] for a in arrays]
    ncols   = arrays[0].shape[1] if arrays else 1
    total   = sum(lengths)

    shm    = shared_memory.SharedMemory(create=True, size=total * ncols * 8)
    packed = np.ndarray((total, ncols), dtype=np.float64, buffer=shm.buf)

    offsets = []
    row = 0
    for arr in arrays:
        n = arr.shape[0]
        packed[row:row + n] = arr
        offsets.append((row, row + n))
        row += n

    return shm, packed, offsets


# ---------------------------------------------------------------------------
# OPTIMIZED SAVE WORKERS
# ---------------------------------------------------------------------------

_SAVE_CAM_ARRAY = None
_SAVE_CAM_IS_COLOR = False
_SAVE_CAM_PATH = ""

def _init_save_cam_worker(shm_name, img_shape, img_dtype, is_color, path):
    """Initializer for save camera workers. Attaches to shared memory once."""
    global _SAVE_CAM_ARRAY, _SAVE_CAM_IS_COLOR, _SAVE_CAM_PATH, _SHM_SAVE_CAM
    _SHM_SAVE_CAM = shared_memory.SharedMemory(name=shm_name)
    _SAVE_CAM_ARRAY = np.ndarray(img_shape, dtype=img_dtype, buffer=_SHM_SAVE_CAM.buf)
    _SAVE_CAM_IS_COLOR = is_color
    _SAVE_CAM_PATH = path

def _save_cam_worker(args):
    """Draw detections on one image and save it to disk. Receives only (image_idx, used_points, detected_points)."""
    image_idx, used_points, detected_points = args
    global _SAVE_CAM_ARRAY, _SAVE_CAM_IS_COLOR, _SAVE_CAM_PATH

    image = _SAVE_CAM_ARRAY[..., image_idx].copy()

    if _SAVE_CAM_IS_COLOR:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if used_points is not None:
        for pt in used_points:
            cv2.circle(image, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), 1)
    elif detected_points is not None:
        for pt in detected_points:
            cv2.circle(image, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), 1)

    cv2.putText(
        image,
        'detected with pycbd, InViLab, doi:10.3390/math11224568',
        (10, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )

    filename = os.path.join(_SAVE_CAM_PATH, f'image_{image_idx + 1}.png')
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


_SAVE_STEREO_ARR1 = None
_SAVE_STEREO_ARR2 = None
_SAVE_STEREO_IS_COLOR = False
_SAVE_STEREO_PATH = ""

def _init_save_stereo_worker(shm_name_1, img_shape_1, img_dtype_1,
                             shm_name_2, img_shape_2, img_dtype_2,
                             is_color, path):
    """Initializer for save stereo workers. Attaches to both shared memory blocks once."""
    global _SAVE_STEREO_ARR1, _SAVE_STEREO_ARR2, _SAVE_STEREO_IS_COLOR, _SAVE_STEREO_PATH
    global _SHM_SAVE_STEREO1, _SHM_SAVE_STEREO2

    _SHM_SAVE_STEREO1 = shared_memory.SharedMemory(name=shm_name_1)
    _SAVE_STEREO_ARR1 = np.ndarray(img_shape_1, dtype=img_dtype_1, buffer=_SHM_SAVE_STEREO1.buf)
    _SHM_SAVE_STEREO2 = shared_memory.SharedMemory(name=shm_name_2)
    _SAVE_STEREO_ARR2 = np.ndarray(img_shape_2, dtype=img_dtype_2, buffer=_SHM_SAVE_STEREO2.buf)
    _SAVE_STEREO_IS_COLOR = is_color
    _SAVE_STEREO_PATH = path

def _save_stereo_worker(args):
    """Draw detections on one stereo image pair and save both to disk. Receives only indices and points."""
    image_idx, used_pts_1, used_pts_2, det_pts_1, det_pts_2 = args
    global _SAVE_STEREO_ARR1, _SAVE_STEREO_ARR2, _SAVE_STEREO_IS_COLOR, _SAVE_STEREO_PATH

    img1 = _SAVE_STEREO_ARR1[..., image_idx].copy()
    img2 = _SAVE_STEREO_ARR2[..., image_idx].copy()

    cvt = cv2.COLOR_BGR2RGB if _SAVE_STEREO_IS_COLOR else cv2.COLOR_GRAY2RGB
    img1 = cv2.cvtColor(img1, cvt)
    img2 = cv2.cvtColor(img2, cvt)

    tag = 'detected with pycbd, InViLab, doi:10.3390/math11224568'

    if used_pts_1 is not None:
        for pt in used_pts_1:
            cv2.circle(img1, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), 1)
        for pt in used_pts_2:
            cv2.circle(img2, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), 1)
    elif det_pts_1 is not None:
        for pt in det_pts_1:
            cv2.circle(img1, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), 1)
        for pt in det_pts_2:
            cv2.circle(img2, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), 1)

    for img in (img1, img2):
        cv2.putText(img, tag, (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(
        os.path.join(_SAVE_STEREO_PATH, f'image_1_{image_idx + 1}.png'),
        cv2.cvtColor(img1, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(_SAVE_STEREO_PATH, f'image_2_{image_idx + 1}.png'),
        cv2.cvtColor(img2, cv2.COLOR_RGB2BGR),
    )


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
                  absolute: bool = False, **kwargs) -> CameraParameters:
        """Calibrate camera."""
        if not isinstance(image_array, np.ndarray):
            raise TypeError("``image_array`` should be a numpy array.")
        if absolute:
            if board_size is None:
                raise TypeError("If `absolute` is set to True `board_size` needs to be known.")
        if marker is not None:
            raise NotImplementedError("Marked checkerboards have not been implemented.")

        self._logger.info("Detecting features")
        sensor_dimensions = np.array([image_array.shape[1], image_array.shape[0]])
        self.construct_feature_list(image_array, space_between_features, board_size, marker, **kwargs)
        self.construct_points_lists([], absolute)
        if not self.image_points_list:
            raise CalibrationError("Failed to detect features in all images, unable to perform calibration.")
        elif len(self.image_points_list) < 11:
            self._logger.warning("Only detected features for " + str(len(self.image_points_list))
                                 + " images. Features from at least 11 images are necessary for an accurate calibration.")
        self._logger.info("Performing calibration")
        camera_parameters = self.opencv_calibration(sensor_dimensions)

        return camera_parameters

    def construct_feature_list(
            self,
            image_array,
            space_between_features,
            board_size=None,
            marker=None,
            **kwargs,
    ):
        # FIX: Replaced ProcessPoolExecutor + shared_memory with a simple loop.
        # This prevents the 0xC0000005 (ACCESS_VIOLATION) crash on Windows caused by
        # shared_memory race conditions during unlinking.
        from .feature_detection import FeatureDetector
        detector = FeatureDetector(space_between_features, board_size, marker, **kwargs)
        detector.detector.checkerboard_detector.detector.show_processing = self.FeatureDetector_logging

        n_images = image_array.shape[-1]
        results = []
        for idx in range(n_images):
            feature = detector.detect_feature(image_array[..., idx])
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
            results.append((idx, SimpleFeature(score, img_pts, obj_pts)))

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
        """Save checkerboard detection results to images using optimized workers."""
        if not os.path.exists(path):
            os.makedirs(path)

        n_images   = image_array.shape[-1]
        is_color   = len(image_array.shape) == 4
        n_workers  = min(os.cpu_count() or 4, n_images)

        shm = shared_memory.SharedMemory(create=True, size=image_array.nbytes)
        try:
            shared_arr = np.ndarray(image_array.shape, dtype=image_array.dtype, buffer=shm.buf)
            shared_arr[:] = image_array[:]

            tasks = []
            for image_idx in range(n_images):
                if image_idx in self.indices:
                    feature_index = self.indices.index(image_idx)
                    used_pts = [tuple(p) for p in self.image_points_list[feature_index]]
                    det_pts  = None
                elif self.feature_list[image_idx].score != 0:
                    used_pts = None
                    det_pts  = [tuple(p) for p in self.feature_list[image_idx].image_points]
                else:
                    used_pts = None
                    det_pts  = None

                # Simplified task: only variable data
                tasks.append((image_idx, used_pts, det_pts))

            with ThreadPoolExecutor(max_workers=n_workers,
                                    initializer=_init_save_cam_worker,
                                    initargs=(shm.name, image_array.shape, image_array.dtype, is_color, path)) as executor:
                list(executor.map(_save_cam_worker, tasks))
        finally:
            shm.close()
            shm.unlink()


# ---------------------------------------------------------------------------
# StereoCalibrator CLASS
# ---------------------------------------------------------------------------

class StereoCalibrator:
    """Object used to perform stereo calibration."""

    def __init__(self) -> None:
        """Class constructor."""
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
        self.FeatureDetector = None
        self.FeatureDetector_logging = False

    def calibrate(self,
                  image_array_1: npt.NDArray,
                  image_array_2: npt.NDArray,
                  parameters_1: CameraParameters,
                  parameters_2: CameraParameters,
                  space_between_features: float,
                  board_size: Tuple[int, int],
                  marker: Optional[Tuple[int, int]] = None,
                  **kwargs) -> StereoParameters:
        """Perform stereo calibration."""
        if not isinstance(image_array_1, np.ndarray) or not isinstance(image_array_2, np.ndarray):
            raise TypeError("``image_array`` should be a numpy array.")
        if marker is not None:
            raise NotImplementedError("Marked checkerboards have not been implemented.")

        self._logger.info("Detecting features")
        self.construct_feature_lists(image_array_1, image_array_2, space_between_features, board_size, marker, **kwargs)

        self.construct_points_lists([])
        if not self.object_points_list:
            raise CalibrationError("Failed to detect common features in all images, unable to perform calibration.")

        self._logger.info("Performing calibration")
        self.stereo_parameters = self.opencv_calibration(parameters_1, parameters_2)

        return self.stereo_parameters

    def calibrate_from_features(self,
                  image_points_list_1: list,
                  image_points_list_2: list,
                  parameters_1: CameraParameters,
                  parameters_2: CameraParameters,
                  space_between_features: float,
                  board_size: Tuple[int, int],
                  objectlist,
                  **kwargs) -> StereoParameters:
        """Perform stereo calibration from pre-computed feature lists."""
        self.image_points_list_1 = image_points_list_1
        self.image_points_list_2 = image_points_list_2
        self.board_size = board_size
        self.space_between_features = space_between_features
        self.object_points_list = objectlist
        self._logger.info("Performing calibration")

        self.stereo_parameters = self.opencv_calibration(parameters_1, parameters_2)

        return self.stereo_parameters

    def calibrate_indices(self, indices: list) -> StereoParameters:
        """Repeat calibration with selected samples."""
        if not isinstance(indices, list):
            raise TypeError("``indices`` should be a list.")

        self.construct_points_lists(indices)
        if not self.object_points_list:
            raise CalibrationError("No images with detected features remain, unable to perform calibration.")

        self.stereo_parameters = self.opencv_calibration(self.stereo_parameters.camera_parameters_1,
                                                         self.stereo_parameters.camera_parameters_2)

        return self.stereo_parameters

    def construct_feature_lists(
        self,
        image_array_1,
        image_array_2,
        space_between_features,
        board_size,
        marker,
        **kwargs,
    ):
        n_images_1 = image_array_1.shape[-1]
        n_images_2 = image_array_2.shape[-1]
        n_workers = min(os.cpu_count() or 4, max(n_images_1, n_images_2))

        shm1 = shared_memory.SharedMemory(create=True, size=image_array_1.nbytes)
        shm2 = shared_memory.SharedMemory(create=True, size=image_array_2.nbytes)

        try:
            shared_arr1 = np.ndarray(
                image_array_1.shape, dtype=image_array_1.dtype, buffer=shm1.buf
            )
            shared_arr1[:] = image_array_1[:]
            shared_arr2 = np.ndarray(
                image_array_2.shape, dtype=image_array_2.dtype, buffer=shm2.buf
            )
            shared_arr2[:] = image_array_2[:]

            # FIX 1: Tasks should ONLY contain the variable parts (idx, cam_id).
            # The detector parameters are now handled by the worker initializer.
            tasks = []
            for idx in range(n_images_1):
                tasks.append((idx, 0))
            for idx in range(n_images_2):
                tasks.append((idx, 1))

            # Executor finishes completely here before we reach finally
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_stereo_worker,
                initargs=(
                    shm1.name,
                    image_array_1.shape,
                    image_array_1.dtype,
                    shm2.name,
                    image_array_2.shape,
                    image_array_2.dtype,
                    # FIX 2: Pass the missing detector initialization arguments here!
                    space_between_features,
                    board_size,
                    marker,
                    kwargs,
                    self.FeatureDetector_logging,
                ),
            ) as executor:
                results = list(executor.map(_stereo_worker, tasks))
                results.sort(key=lambda x: (x[1], x[0]))
                self.feature_list_1 = [
                    feature for _, cam_id, feature in results if cam_id == 0
                ]
                self.feature_list_2 = [
                    feature for _, cam_id, feature in results if cam_id == 1
                ]

        finally:
            # Now it's safe to release on the main process side
            for shm in (shm1, shm2):
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass

    def construct_points_lists(self, indices: list) -> None:
        """Construct lists of image points and object points for calibration."""
        if self.feature_list_1 == []:
            tobject_points_list = []
            timage_points_list_1 = []
            timage_points_list_2 = []
            self.indices = []
            for idx in indices:
                timage_points_list_1.append(self.image_points_list_1[idx])
                timage_points_list_2.append(self.image_points_list_2[idx])
                tobject_points_list.append(self.object_points_list[idx])
            self.image_points_list_1 = timage_points_list_1
            self.image_points_list_2 = timage_points_list_2
            self.object_points_list = tobject_points_list
            self.indices = list(range(len(self.image_points_list_1)))
            return

        self.indices = []
        self.object_points_list = []
        self.image_points_list_1 = []
        self.image_points_list_2 = []
        if not indices:
            indices = range(len(self.feature_list_1))
        for idx in indices:
            try:
                feature_1 = self.feature_list_1[idx]
                feature_2 = self.feature_list_2[idx]
            except IndexError:
                pass
            else:
                if feature_1.score >= 1 and feature_2.score >= 1:
                    common_indices = np.where((feature_1.object_points == feature_2.object_points[:, None]).all(-1))
                    if common_indices[0].size != 0:
                        self.image_points_list_1.append(feature_1.image_points[common_indices[1]])
                        self.image_points_list_2.append(feature_2.image_points[common_indices[0]])
                        self.object_points_list.append(feature_1.object_points[common_indices[1]])
                        self.indices.append(idx)

    def opencv_calibration(self, parameters_1: CameraParameters, parameters_2: CameraParameters,
                          show_3d_plot: bool = True) -> StereoParameters:
        """Stereo calibration followed by parallel per-board 3-D error computation."""
        print(len(self.image_points_list_1))

        self.rms_reproj_error, _, _, _, _, R, T, E, F, self.r_vecs, self.t_vecs, self.per_view_err \
            = cv2.stereoCalibrateExtended(
                self.object_points_list,
                self.image_points_list_1,
                self.image_points_list_2,
                parameters_1.get_intrinsics_matrix_opencv(),
                parameters_1.get_distortion_coeffs_opencv(),
                parameters_2.get_intrinsics_matrix_opencv(),
                parameters_2.get_distortion_coeffs_opencv(),
                parameters_1.sensor_dimensions,
                None, None,
                flags=cv2.CALIB_FIX_INTRINSIC,
            )

        # FIX: Replaced ThreadPoolExecutor + shared_memory with a simple loop for PnP error calculation.
        # This further prevents any shared_memory related crashes on Windows.
        n_boards = len(self.object_points_list)
        board_results = []

        K1 = parameters_1.get_intrinsics_matrix_opencv()
        d1 = parameters_1.get_distortion_coeffs_opencv()
        K2 = parameters_2.get_intrinsics_matrix_opencv()
        d2 = parameters_2.get_distortion_coeffs_opencv()

        for i in range(n_boards):
            obj_points = self.object_points_list[i]
            img_points_1 = self.image_points_list_1[i]
            img_points_2 = self.image_points_list_2[i]

            _, rvec_1, tvec_1 = cv2.solvePnP(obj_points, img_points_1, K1, d1)
            _, rvec_2, tvec_2 = cv2.solvePnP(obj_points, img_points_2, K2, d2)

            R1, _ = cv2.Rodrigues(rvec_1)
            R2, _ = cv2.Rodrigues(rvec_2)

            ones = np.ones((obj_points.shape[0], 1), dtype=np.float64)
            obj_h = np.hstack((obj_points, ones))

            T1 = np.eye(4, dtype=np.float64)
            T1[:3, :3] = R1
            T1[:3, 3] = tvec_1.flatten()
            pts_cam1_h = (T1 @ obj_h.T).T
            pts_cam1 = pts_cam1_h[:, :3] / pts_cam1_h[:, 3:4]

            T2 = np.eye(4, dtype=np.float64)
            T2[:3, :3] = R2
            T2[:3, 3] = tvec_2.flatten()
            pts_cam2_h = (T2 @ obj_h.T).T

            R_inv = R.T
            T_inv = -R_inv @ T
            T3 = np.eye(4, dtype=np.float64)
            T3[:3, :3] = R_inv
            T3[:3, 3] = T_inv.flatten()
            pts_cam2_in1_h = (T3 @ pts_cam2_h.T).T
            pts_cam2_in1 = pts_cam2_in1_h[:, :3] / pts_cam2_in1_h[:, 3:4]

            error = float(np.mean(np.linalg.norm(pts_cam1 - pts_cam2_in1, axis=1)))
            board_results.append((error, pts_cam1, pts_cam2_in1))

        errors_in_mm     = [r[0] for r in board_results]
        all_points_cam1  = [r[1] for r in board_results]
        all_points_cam2  = [r[2] for r in board_results]

        mean_error_in_mm = float(np.mean(errors_in_mm))
        self.errors_in_mm = errors_in_mm
        print(f"Mean Error in millimeters: {mean_error_in_mm}")

        # 3-D scatter plot (Default True to maintain original behavior, but can be disabled for speed)
        if show_3d_plot:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            for pts1, pts2 in zip(all_points_cam1, all_points_cam2):
                ax.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2],
                           color='red',  marker='o', s=50,
                           label='Camera 1' if not ax.get_legend_handles_labels()[0] else "")
                ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2],
                           color='blue', marker='x', s=50,
                           label='Camera 2' if not ax.get_legend_handles_labels()[0] else "")
            ax.set_title('Checkerboards Detected by Both Cameras in 3D')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Z Coordinate')
            ax.legend()
            plt.show()

        calibration_parameters = StereoParameters()
        calibration_parameters.set_parameters_opencv(self.rms_reproj_error, R, T, E, F)
        calibration_parameters.camera_parameters_1 = parameters_1
        calibration_parameters.camera_parameters_2 = parameters_2

        return calibration_parameters

    def plot_reproj_error(self) -> None:
        """Plot mean re-projection error and re-projection error for each calibration image."""
        indices = list(map(str, self.indices))
        cam_errs = {'camera 1': self.per_view_err[:, 0], 'camera 2': self.per_view_err[:, 1]}

        fig, ax = plt.subplots()
        ax.axhline(self.rms_reproj_error, color='g', linestyle='--', label='RMS')
        x = np.arange(len(indices))
        width = 0.25
        multiplier = 0
        for camera, measurement in cam_errs.items():
            offset = width * multiplier
            ax.bar(x + offset, measurement, width, label=camera)
            multiplier += 1

        ax.set_xlabel("Image index")
        ax.set_ylabel("Reprojection error")
        ax.set_title("Reprojection error for each detected image")
        ax.set_xticks(x + width, indices)
        ax.legend(ncols=3)
        plt.show()

    def plot_and_filter_reproj_error(self) -> list:
        """Plot mean re-projection error and re-projection error for each calibration image."""
        indices = list(map(str, self.indices))
        cam_errs = {'camera 1': self.per_view_err[:, 0], 'camera 2': self.per_view_err[:, 1]}

        fig, ax = plt.subplots()
        ax.axhline(self.rms_reproj_error, color='g', linestyle='--', label='RMS')
        x = np.arange(len(indices))
        width = 0.25
        multiplier = 0
        bars = []
        for camera, measurement in cam_errs.items():
            offset = width * multiplier
            bars.append(ax.bar(x + offset, measurement, width, label=camera))
            multiplier += 1

        ax.set_xlabel("Image index")
        ax.set_ylabel("Reprojection error")
        ax.set_title("Reprojection error for each detected image")
        ax.set_xticks(x + width, indices)
        ax.legend(ncols=3)

        selected_indices = []

        def on_click(event):
            for bar_group in bars:
                for i, bar in enumerate(bar_group):
                    if bar.contains(event)[0]:
                        bar.set_color('r')
                        selected_indices.append(self.indices[i])
                        fig.canvas.draw()

        def on_key(event):
            if event.key == 'escape':
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        not_selected_indices = [index for index in self.indices if index not in selected_indices]
        return not_selected_indices

    def save_checkerboard_detection_to_images(self, image_array_1, image_array_2, path):
        """Save detection results for both cameras to disk in parallel using optimized workers."""
        if not os.path.exists(path):
            os.makedirs(path)
        n_images = image_array_1.shape[-1]
        is_color = len(image_array_1.shape) == 4
        n_workers = min(os.cpu_count() or 4, n_images)

        shm1 = shared_memory.SharedMemory(create=True, size=image_array_1.nbytes)
        shm2 = shared_memory.SharedMemory(create=True, size=image_array_2.nbytes)
        try:
            arr1 = np.ndarray(
                image_array_1.shape, dtype=image_array_1.dtype, buffer=shm1.buf
            )
            arr2 = np.ndarray(
                image_array_2.shape, dtype=image_array_2.dtype, buffer=shm2.buf
            )
            arr1[:] = image_array_1[:]
            arr2[:] = image_array_2[:]

            tasks = []
            for image_idx in range(n_images):
                if image_idx in self.indices:
                    fi = self.indices.index(image_idx)
                    used_pts_1 = [tuple(p) for p in self.image_points_list_1[fi]]
                    used_pts_2 = [tuple(p) for p in self.image_points_list_2[fi]]
                    det_pts_1 = det_pts_2 = None
                elif (
                    image_idx < len(self.feature_list_1)
                    and image_idx < len(self.feature_list_2)
                    and self.feature_list_1[image_idx].score != 0
                    and self.feature_list_2[image_idx].score != 0
                ):
                    used_pts_1 = used_pts_2 = None
                    det_pts_1 = [
                        tuple(p) for p in self.feature_list_1[image_idx].image_points
                    ]
                    det_pts_2 = [
                        tuple(p) for p in self.feature_list_2[image_idx].image_points
                    ]
                else:
                    used_pts_1 = used_pts_2 = det_pts_1 = det_pts_2 = None
                tasks.append((image_idx, used_pts_1, used_pts_2, det_pts_1, det_pts_2))

            def worker(task):
                image_idx, used_pts_1, used_pts_2, det_pts_1, det_pts_2 = task

                img1 = arr1[..., image_idx].copy()
                img2 = arr2[..., image_idx].copy()

                cvt = cv2.COLOR_BGR2RGB if is_color else cv2.COLOR_GRAY2RGB
                img1 = cv2.cvtColor(img1, cvt)
                img2 = cv2.cvtColor(img2, cvt)

                if used_pts_1 is not None:
                    for pt in used_pts_1:
                        cv2.circle(img1, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), 1)
                    for pt in used_pts_2:
                        cv2.circle(img2, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), 1)
                elif det_pts_1 is not None:
                    for pt in det_pts_1:
                        cv2.circle(img1, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), 1)
                    for pt in det_pts_2:
                        cv2.circle(img2, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), 1)

                tag = "detected with pycbd, InViLab, doi:10.3390/math11224568"
                for img in (img1, img2):
                    cv2.putText(
                        img,
                        tag,
                        (10, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.imwrite(
                    os.path.join(path, f"image_1_{image_idx + 1}.png"),
                    cv2.cvtColor(img1, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(path, f"image_2_{image_idx + 1}.png"),
                    cv2.cvtColor(img2, cv2.COLOR_RGB2BGR),
                )

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(worker, tasks))
        finally:
            shm1.close()
            shm1.unlink()
            shm2.close()
            shm2.unlink()

# ---------------------------------------------------------------------------
# StereoParameters CLASS
# ---------------------------------------------------------------------------

class StereoParameters:
    """Object that contains all camera calibration parameters."""

    def __init__(self) -> None:
        """Class constructor."""
        self.camera_parameters_1: CameraParameters = CameraParameters()
        self.camera_parameters_2: CameraParameters = CameraParameters()
        self.rms_reproj_error: np.float64 = np.float64(0)
        self.R: npt.NDArray[np.float64] = np.zeros((3, 3))
        self.T: npt.NDArray[np.float64] = np.zeros((3, 1))
        self.E: npt.NDArray[np.float64] = np.zeros((3, 3))
        self.F: npt.NDArray[np.float64] = np.zeros((3, 3))
        self.R_1 = None
        self.R_2 = None
        self.P_1 = None
        self.P_2 = None
        self.Q = None
        self.roi_1 = None
        self.roi_2 = None
        self.map_1_x = None
        self.map_1_y = None
        self.map_2_x = None
        self.map_2_y = None
        self.units = 'mm'
        self.info = ["Camera1", "Camera2"]

    def set_parameters_opencv(self, rms_reproj_error: float, R: npt.NDArray, T: npt.NDArray, E: npt.NDArray,
                              F: npt.NDArray) -> None:
        """Set stereo parameters obtained from OpenCV calibration."""
        self.rms_reproj_error = rms_reproj_error
        self.R = R
        self.T = T
        self.E = E
        self.F = F

    def save_parameters(self, full_save_path: str) -> None:
        """Save calibration parameters to .h5 file"""
        directory_path = os.path.dirname(full_save_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        if not os.path.exists(directory_path+"/camera_calibration"):
            os.makedirs(directory_path+"/camera_calibration")
        self.camera_parameters_1.save_parameters(directory_path+"/camera_calibration/camera_1_parameters.h5")
        self.camera_parameters_1.save_parameters_to_json(directory_path+"/camera_calibration/camera_1_parameters.json")
        self.camera_parameters_2.save_parameters(directory_path+"/camera_calibration/camera_2_parameters.h5")
        self.camera_parameters_2.save_parameters_to_json(directory_path+"/camera_calibration/camera_2_parameters.json")
        self.camera_parameters_1.save_parameters(full_save_path, "camera_calibration/camera_1_parameters")
        self.camera_parameters_2.save_parameters(full_save_path, "camera_calibration/camera_2_parameters")

        self.H = TransformationMatrix()
        self.H.T = self.T
        self.H.R = self.R
        self.H.units = self.units
        self.H.info = self.info
        self.H.save_to_json(os.path.splitext(full_save_path)[0] + '_TransformationMatrix.json')
        with h5py.File(full_save_path, "a") as file:
            try:
                file.create_dataset("camera_calibration/stereo_parameters/rms_reproj_error", data=self.rms_reproj_error)
                file.create_dataset("camera_calibration/stereo_parameters/R", data=self.R)
                file.create_dataset("camera_calibration/stereo_parameters/T", data=self.T)
                file.create_dataset("camera_calibration/stereo_parameters/E", data=self.E)
                file.create_dataset("camera_calibration/stereo_parameters/F", data=self.F)
                group = file.create_group("camera_calibration/stereo_parameters/TransformationMatrix")
                group.create_dataset("H", data=self.H.H.tolist())
                group.attrs["info"] = self.H.info
                group.attrs["units"] = self.H.units
            except ValueError:
                file["camera_calibration/stereo_parameters/rms_reproj_error"][()] = self.rms_reproj_error
                file["camera_calibration/stereo_parameters/R"][()] = self.R
                file["camera_calibration/stereo_parameters/T"][()] = self.T
                file["camera_calibration/stereo_parameters/E"][()] = self.E
                file["camera_calibration/stereo_parameters/F"][()] = self.F
                group = file["camera_calibration/stereo_parameters/TransformationMatrix"]
                if "H" in group:
                    del group["H"]
                group.create_dataset("H", data=self.H.H.tolist())
                group.attrs["info"]  = self.H.info
                group.attrs["units"] = self.H.units

    def load_parameters(self, full_save_path: str) -> None:
        """Load calibration parameters from .h5 file into object."""
        self.camera_parameters_1.load_parameters(full_save_path, "camera_calibration/camera_1_parameters")
        self.camera_parameters_2.load_parameters(full_save_path, "camera_calibration/camera_2_parameters")
        with h5py.File(full_save_path, "r") as file:
            try:
                self.rms_reproj_error = file["camera_calibration/stereo_parameters/rms_reproj_error"][()]
                self.R = file["camera_calibration/stereo_parameters/R"][()]
                self.T = file["camera_calibration/stereo_parameters/T"][()]
                self.E = file["camera_calibration/stereo_parameters/E"][()]
                self.F = file["camera_calibration/stereo_parameters/F"][()]
                self.H = self.load_homogeneous(TransformationMatrix,
                    file["camera_calibration/stereo_parameters/TransformationMatrix"])
            except KeyError as e:
                raise KeyError("File does not contain stereo calibration parameters.") from e

    @staticmethod
    def load_homogeneous(cls, group):
        obj = cls()
        obj.H = group["H"][()]
        obj.info = group.attrs.get("info", ["unknown", "unknown"])
        obj.units = group.attrs.get("units", "mm")
        return obj

    def calculate_undistort_rectify_maps(self, alpha: float = 0, fixed_point_maps: bool = False) -> None:
        """Calculate rectification transforms and maps necessary for remapping."""
        sensor_dim_1 = self.camera_parameters_1.sensor_dimensions
        sensor_dim_2 = self.camera_parameters_2.sensor_dimensions
        if not np.array_equal(sensor_dim_1, sensor_dim_2):
            raise NotImplementedError("Rectification for different sensor sizes has not been implemented.")
        intrinsics_1 = self.camera_parameters_1.get_intrinsics_matrix_opencv()
        distortion_1 = self.camera_parameters_1.get_distortion_coeffs_opencv()
        intrinsics_2 = self.camera_parameters_2.get_intrinsics_matrix_opencv()
        distortion_2 = self.camera_parameters_2.get_distortion_coeffs_opencv()

        self.R_1, self.R_2, self.P_1, self.P_2, self.Q, self.roi_1, self.roi_2 \
            = cv2.stereoRectify(intrinsics_1, distortion_1, intrinsics_2, distortion_2,
                                sensor_dim_1, self.R, self.T,
                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)

        map_type = cv2.CV_16SC2 if fixed_point_maps else cv2.CV_32FC1
        self.map_1_x, self.map_1_y = cv2.initUndistortRectifyMap(
            intrinsics_1, distortion_1, self.R_1, self.P_1, sensor_dim_1, map_type)
        self.map_2_x, self.map_2_y = cv2.initUndistortRectifyMap(
            intrinsics_2, distortion_2, self.R_2, self.P_2, sensor_dim_2, map_type)

    def remap_images(self, frame_1: npt.NDArray, frame_2: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Remap the images so they are undistorted and rectified."""
        try:
            rectified_1 = cv2.remap(frame_1, self.map_1_x, self.map_1_y, cv2.INTER_LINEAR)
            rectified_2 = cv2.remap(frame_2, self.map_2_x, self.map_2_y, cv2.INTER_LINEAR)
        except cv2.error as e:
            raise Exception("You probably did not calculate the undistort and rectification map before remapping.") from e

        return rectified_1, rectified_2