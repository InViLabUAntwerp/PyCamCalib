"""Module that contains all code for camera calibration."""

from __future__ import annotations
import os
import cv2
import h5py
import logging
import numpy as np
import numpy.typing as npt
from rx import empty
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from .exceptions import CalibrationError
from .feature_detection import FeatureDetector
from CTPv.Transformation.TransformationMatrix import TransformationMatrix
from .CameraParameters import CameraParameters

class CameraCalibrator:
    """Object used to calibrate a camera.

    :var sensor_dimensions: Sensor dimensions in pixels (w, h).
    :var feature_list: List with :py:class:`~PyCamCalib.core.feature_detection.CalibrationFeature` objects for all images.
    :var indices: Indices of all images in :py:attr:`feature_list` that were used for the current calibration.
    :var image_points_list: Contains all image space points used for the current calibration.
    :var object_points_list: Contains all object space points used for the current calibration.
    :var per_view_err: Per view re-projection errors for all images that are listed in :py:attr:`indices`.
    :var rms_reproj_error: The rms re-projection error for the current calibration.
    :var r_vecs: Rotation vectors for each image.
    :var t_vecs: Translation vectors for each image.
    :var extrinsics_std: Standard deviations for extrinsic parameters.
    :var camera_parameters: :py:class:`~PyCamCalib.core.calibration.CameraParameters` for current calibration.
    """

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
        """Calibrate camera.

        Calibrate a camera with an array of images of a calibration target. At least 11 good images are required for
        a good calibration.

        :param image_array: Array containing all images that will be used for calibration. It can either be a 3D array
           (h,w,n) for grayscale images or a 4D array (h,w,c,n) for BGR images.
        :param space_between_features: Checker size in mm. You can set this to 1 if you don't care about scaling.
        :param board_size: Size of the board in (rows, columns)
        :param marker: Position of the marker if there is a marker present. Not implemented yet.
        :param absolute: If set to true only images where the absolute object space coordinates of the checkerboards are
           known are used for the calibration. This ensures that the extrinsic parameters for each image are correct.
           Either `board_size` or `marker` needs to be known in order to use this option.
        :param kwargs: kwargs passed along to the :py:class:`PyCamCalib.core.feature_detection.FeatureDetector`.
           Available keywords are:v`expand`: set this to True, if you want to try to expand the checkerboard past
           obstructionsv`predict`: set this to True if you want to predict missing checkerboard corners `out_of_image`:
           set this to True if you want the predictions to include points that lie beyond the imagevborders.
        :returns: An object that contains all calibration parameter data.
        :raises TypeError: When image_array is not a numpy array or when `absolute` is set to True but `board_size` and
           `marker` are not given.
        :raises CalibrationError: When no features were detected in any of the images.
        """
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

    def construct_feature_list(self, image_array: npt.NDArray, space_between_features: float,
                               board_size: Optional[Tuple[int, int]] = None, marker: Optional[Tuple[int, int]] = None,
                               **kwargs) -> None:
        """Construct a list with the calibration features for all images in the image_array.

        Unless you want to perform the calibration steps separately, you should not use this method.
        """
        self.feature_list = []
        if self.FeatureDetector is None:
            self.FeatureDetector = FeatureDetector(space_between_features, board_size, marker, **kwargs)
            self.FeatureDetector.detector.checkerboard_detector.detector.show_processing= self.FeatureDetector_logging

        #check if board_size, marker is none
        if board_size is not None and marker is not None:
            self.FeatureDetector= FeatureDetector(space_between_features, board_size, marker, **kwargs)
            self.FeatureDetector.detector.checkerboard_detector.detector.show_processing= self.FeatureDetector_logging
        feature_detector = self.FeatureDetector

        n_images = image_array.shape[-1]
        for idx in range(n_images):
            feature = feature_detector.detect_feature(image_array[..., idx])
            self.feature_list.append(feature)
            if not feature.score > 0:
                self._logger.info("Failed feature detection on image nr " + str(idx + 1) + ".")

    def construct_points_lists(self, indices: list, absolute: bool = False) -> None:
        """Construct lists of image points and object points for calibration.

        If indices is empty all images where a feature was detected will be used, otherwise only images that
        correspond to the elements in indices will be used. Unless you want to perform the calibration steps separately,
        you should not use this method.
        """
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
            try:  # Safeguard for when indices that don't exist are passed into the function.
                feature = self.feature_list[idx]
            except IndexError:
                pass
            else:
                if feature.score >= minimum_score:
                    self.image_points_list.append(feature.image_points)
                    self.object_points_list.append(feature.object_points)
                    self.indices.append(idx)

    def initilize_camera_parameters(self,fx,fy,cx,cy):
        self.camera_parameters = CameraParameters()
        self.camera_parameters.fx = fx
        self.camera_parameters.fy = fy
        self.camera_parameters.cx = cx
        self.camera_parameters.cy = cy

    def opencv_calibration(self, sensor_dimensions: npt.NDArray[np.int32]) -> CameraParameters:
        """Regular OpenCV camera calibration.

         Unless you want to perform the calibration steps separately, you should not use this method.
        """
        self.sensor_dimensions = sensor_dimensions
        # flip the sensor dimensions
        #sensor_dimensions = np.flip(sensor_dimensions)

        if self.camera_parameters is None:
            cameraMatrix = None
        else:
            cameraMatrix = self.camera_parameters.get_intrinsics_matrix_opencv()
        rms_reproj_error, intrinsics_matrix, dist_coeffs, r_vecs, t_vecs, intrinsics_std, extrinsics_std, \
        per_view_err = cv2.calibrateCameraExtended(self.object_points_list, self.image_points_list, sensor_dimensions,
                                                   cameraMatrix, None)

        # Convert some parameter datatypes and reshape some matrices
        self.rms_reproj_error = np.float64(rms_reproj_error)
        self.per_view_err = np.squeeze(per_view_err)
        dist_coeffs = np.squeeze(dist_coeffs)
        intrinsics_std = np.squeeze(intrinsics_std)
        self.r_vecs = np.squeeze(np.array(r_vecs))
        self.t_vecs = np.squeeze(np.array(t_vecs))
        self.extrinsics_std = extrinsics_std

        # Save parameters in CalibrationParameters object
        if self.camera_parameters is None:
            self.camera_parameters = CameraParameters()
        self.camera_parameters.set_parameters_opencv(self.rms_reproj_error, intrinsics_matrix, dist_coeffs,
                                                     intrinsics_std, self.sensor_dimensions)

        return self.camera_parameters

    def calibrate_indices(self, indices: list, absolute: bool = False) -> CameraParameters:
        """Repeat calibration with selected samples.

        Repeat the camera calibration with the samples specified in indices. This can be used to improve the
        camera calibration by removing outliers.

        :param indices: A list that contains the indices that correspond to the elements in :py:data:`feature_list`
           which should be used for a new calibration. Passing an empty list will perform a calibration with all good
           images.
        :param absolute: If set to true only images where the absolute object space coordinates of the checkerboards are
           known are used for the calibration. This ensures that the extrinsic parameters for each image are correct.
        :returns: An object that contains all calibration parameter data.
        :raises TypeError: When indices is not a list.
        :raises CalibrationError: When no images with detected features were selected.
        """
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

    def calibrate_world_coordinate_system(self,image,space_between_features: float,                  board_size: Optional[Tuple[int, int]] = None,
                  marker: Optional[Tuple[int, int]] = None,
                  absolute: bool = False, **kwargs) -> TransformationMatrix:

        image_array = image
        sensor_dimensions = np.array([image_array.shape[1], image_array.shape[0]])
        self.construct_feature_list(image_array, space_between_features, board_size, marker, **kwargs)
        self.construct_points_lists([])

        for obj_points, img_points_1 in zip(self.object_points_list, self.image_points_list,):
            # Estimate the position of the object points in the first camera coordinate system using PnP
            _, rvec_1, tvec_1 = cv2.solvePnP(obj_points, img_points_1, self.camera_parameters.get_intrinsics_matrix_opencv(),
                                             self.camera_parameters.get_distortion_coeffs_opencv())

            # --- 3. CALCULATE RMS ERROR ---

            # Re-project the 3D points to the 2D image plane
            reprojected_points, _ = cv2.projectPoints(obj_points, rvec_1, tvec_1, self.camera_parameters.get_intrinsics_matrix_opencv(), self.camera_parameters.get_distortion_coeffs_opencv())

            # Reshape reprojected_points to match the shape of img_points_1 for calculation
            reprojected_points = reprojected_points.reshape(-1, 2)

            # Calculate the absolute error between the original and reprojected points
            # cv2.norm calculates the L2 norm (Euclidean distance) between the two sets of points
            # We then divide by the square root of the number of points to get the RMS
            error = cv2.norm(img_points_1, reprojected_points, cv2.NORM_L2) / np.sqrt(len(reprojected_points))

            # convert rvec_1 and tvec_ to homogenous matrix
            R, _ = cv2.Rodrigues(rvec_1)
            H = np.hstack((R, tvec_1))
            H = np.vstack((H, [0, 0, 0, 1]))
            #convert error to string



            return H,error




    def save_checkerboard_detection_to_images(self,image_array,path):
        if not os.path.exists(path):
            os.makedirs(path)
        for image_idx in range(image_array.shape[-1]):
            image = image_array[..., image_idx]
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            if image_idx in self.indices:
                feature_index = self.indices.index(image_idx)
                for point in self.image_points_list[feature_index]:
                    try:
                        cv2.circle(image, (int(point[0]), int(point[1])), 10, (0, 255, 0), 1)  # Green for used features
                    except:
                        pass
            elif self.feature_list[image_idx].score != 0:
                for point in self.feature_list[image_idx].image_points:
                    cv2.circle(image, (int(point[0]), int(point[1])), 10, (255, 0, 0), 1)  # Red for detected features
            # Add the text to the image
            cv2.putText(image, 'detected with pycbd, InViLab, doi:10.3390/math11224568', (10, image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            save_filename = os.path.join(path, f'image_{image_idx + 1}.png')
            cv2.imwrite(save_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))




class StereoCalibrator:
    """Object used to perform stereo calibration.

    :var feature_list_1: List with :py:class:`~PyCamCalib.core.feature_detection.CalibrationFeature` objects for all
       images from camera 1.
    :var feature_list_2: List with :py:class:`~PyCamCalib.core.feature_detection.CalibrationFeature` objects for all
       images from camera 2.
    :var indices: Indices of all images in :py:attr:`feature_list_1` and :py:attr:`feature_list_2` that were
       used for the current calibration.
    :var image_points_list_1: Contains all image space points used for the current calibration for camera 1.
    :var image_points_list_2: Contains all image space points used for the current calibration for camera 2.
    :var object_points_list: Contains all object space points used for the current calibration.
    :var per_view_err: Per view re-projection errors for all images that are listed in :py:attr:`indices`.
    :var rms_reproj_error: The rms re-projection error for the current calibration.
    :var r_vecs: Rotation vectors for each image.
    :var t_vecs: Translation vectors for each image.
    :var stereo_parameters: :py:class:`~PyCamCalib.core.calibration.StereoParameters` for current calibration.
    """

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
        """Perform stereo calibration.

        Stereo calibrate 2 cameras with 2 matching arrays of images of a calibration target, one for each image.

        :param image_array_1: Array containing all images for camera 1 that will be used for calibration. It can either
           be a 3D array (h,w,n) for grayscale images or a 4D array (h,w,c,n) for BGR images.
        :param image_array_2: Array containing all images for camera 2 that will be used for calibration. It can either
           be a 3D array (h,w,n) for grayscale images or a 4D array (h,w,c,n) for BGR images.
        :param parameters_1: The :py:class:`~PyCamCalib.core.calibration.CameraParameters` for camera 1.
        :param parameters_2: The :py:class:`~PyCamCalib.core.calibration.CameraParameters` for camera 2.
        :param space_between_features: Checker size in mm. You can set this to 1 if you don't care about scaling.
        :param board_size: Size of the board in (rows, columns)
        :param marker: Position of the marker if there is a marker present. Not implemented yet.
        :param kwargs: kwargs passed along to the :py:class:`~PyCamCalib.core.feature_detection.FeatureDetector`.
           Available keywords are:
           `expand`: set this to True, if you want to try to expand the checkerboard past obstructions
           `predict`: set this to True if you want to predict missing checkerboard corners
           `out_of_image`: set this to True if you want the predictions to include points that lie beyond the image
           borders.
        :returns: An object that contains all calibration parameter data.
        :raises TypeError: When image_array_1 or image_array_2 is not a numpy array.
        :raises CalibrationError: When no features were detected in any of the images.
        """
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
        """Perform stereo calibration.

        Stereo calibrate 2 cameras with 2 matching arrays of images of a calibration target, one for each image.

        :param image_array_1: Array containing all images for camera 1 that will be used for calibration. It can either
           be a 3D array (h,w,n) for grayscale images or a 4D array (h,w,c,n) for BGR images.
        :param image_array_2: Array containing all images for camera 2 that will be used for calibration. It can either
           be a 3D array (h,w,n) for grayscale images or a 4D array (h,w,c,n) for BGR images.
        :param parameters_1: The :py:class:`~PyCamCalib.core.calibration.CameraParameters` for camera 1.
        :param parameters_2: The :py:class:`~PyCamCalib.core.calibration.CameraParameters` for camera 2.
        :param space_between_features: Checker size in mm. You can set this to 1 if you don't care about scaling.
        :param board_size: Size of the board in (rows, columns)
        :param marker: Position of the marker if there is a marker present. Not implemented yet.
        :param kwargs: kwargs passed along to the :py:class:`~PyCamCalib.core.feature_detection.FeatureDetector`.
           Available keywords are:
           `expand`: set this to True, if you want to try to expand the checkerboard past obstructions
           `predict`: set this to True if you want to predict missing checkerboard corners
           `out_of_image`: set this to True if you want the predictions to include points that lie beyond the image
           borders.
        :returns: An object that contains all calibration parameter data.
        :raises TypeError: When image_array_1 or image_array_2 is not a numpy array.
        :raises CalibrationError: When no features were detected in any of the images.
        """

        self.image_points_list_1 = image_points_list_1
        self.image_points_list_2 = image_points_list_2
        self.board_size = board_size
        self.space_between_features = space_between_features
        self.object_points_list = objectlist
        self._logger.info("Performing calibration")

        self.stereo_parameters = self.opencv_calibration(parameters_1, parameters_2)

        return self.stereo_parameters
    def calibrate_indices(self, indices: list) -> StereoParameters:
        """Repeat calibration with selected samples.

        Repeat the stereo calibration with the samples specified in indices. This can be used to improve the
        stereo calibration by removing outliers.

        :param indices: A list that contains the indices that correspond to the elements in :py:data:`feature_list`
           which should be used for a new calibration. Passing an empty list will perform a calibration with all good
           images.
        :returns: An object that contains all calibration parameter data.
        :raises TypeError: When indices is not a list.
        :raises CalibrationError: When no images with detected features were selected.
        """
        if not isinstance(indices, list):
            raise TypeError("``indices`` should be a list.")

        self.construct_points_lists(indices)
        if not self.object_points_list:
            raise CalibrationError("No images with detected features remain, unable to perform calibration.")

        self.stereo_parameters = self.opencv_calibration(self.stereo_parameters.camera_parameters_1,
                                                         self.stereo_parameters.camera_parameters_2)

        return self.stereo_parameters

    def construct_feature_lists(self, image_array_1: npt.NDArray, image_array_2: npt.NDArray,
                                space_between_features: float, board_size: Tuple[int, int],
                                marker: Optional[Tuple[int, int]], **kwargs) -> None:
        """Construct a list with the calibration features for all images in image_array.

        Unless you want to perform the calibration steps separately, you should not use this method.
        """

        if self.FeatureDetector is None:
            self.FeatureDetector = FeatureDetector(space_between_features, board_size, marker, **kwargs)
            self.FeatureDetector.detector.checkerboard_detector.detector.show_processing= self.FeatureDetector_logging

        #check if board_size, marker is none
        if board_size is not None and marker is not None:
            self.FeatureDetector= FeatureDetector(space_between_features, board_size, marker, **kwargs)
            self.FeatureDetector.detector.checkerboard_detector.detector.show_processing= self.FeatureDetector_logging
        feature_detector = self.FeatureDetector
        self.feature_list_1 = []
        n_images = image_array_1.shape[-1]
        for idx in range(n_images):
            feature = feature_detector.detect_feature(image_array_1[..., idx])
            self.feature_list_1.append(feature)
            if not feature.score > 0:
                self._logger.info("Failed feature detection on cam 1 image nr " + str(idx + 1) + ".")

        self.feature_list_2 = []
        n_images = image_array_2.shape[-1]
        for idx in range(n_images):
            feature = feature_detector.detect_feature(image_array_2[..., idx])
            self.feature_list_2.append(feature)
            if not feature.score > 0:
                self._logger.info("Failed feature detection on cam 2 image nr " + str(idx + 1) + ".")

    def construct_points_lists(self, indices: list) -> None:
        """Construct lists of image points and object points for calibration.

        If indices is empty all images where a feature was detected will be used, otherwise only images that
        correspond to the elements in indices will be used. Unless you want to perform the calibration steps separately,
        you should not use this method.
        """

        # check if self.feature_list_1 is empty(), if so, work on the raw image points
        if self.feature_list_1 ==[]:
            tobject_points_list=[]
            timage_points_list_1=[]
            timage_points_list_2=[]
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
            try:  # Safeguard for when indices that don't exist are passed into the function.
                feature_1 = self.feature_list_1[idx]
                feature_2 = self.feature_list_2[idx]
            except IndexError:
                pass
            else:
                if feature_1.score >= 2 and feature_2.score >= 2:
                    common_indices = np.where((feature_1.object_points == feature_2.object_points[:, None]).all(-1))
                    if common_indices[0].size != 0:
                        self.image_points_list_1.append(feature_1.image_points[common_indices[1]])
                        self.image_points_list_2.append(feature_2.image_points[common_indices[0]])
                        self.object_points_list.append(feature_1.object_points[common_indices[1]])
                        self.indices.append(idx)

    def opencv_calibration(self, parameters_1: CameraParameters, parameters_2: CameraParameters) -> StereoParameters:
        """Regular OpenCV stereo calibration.

        Unless you want to perform the calibration steps separately, you should not use this method.
        """
        # print the size of self.image_points_list_1
        print(len(self.image_points_list_1))
        self.rms_reproj_error, _, _, _, _, R, T, E, F, self.r_vecs, self.t_vecs, self.per_view_err \
            = cv2.stereoCalibrateExtended(self.object_points_list,
                                          self.image_points_list_1,
                                          self.image_points_list_2,
                                          parameters_1.get_intrinsics_matrix_opencv(),
                                          parameters_1.get_distortion_coeffs_opencv(),
                                          parameters_2.get_intrinsics_matrix_opencv(),
                                          parameters_2.get_distortion_coeffs_opencv(),
                                          parameters_1.sensor_dimensions,  # Doesn't matter
                                          None,
                                          None,
                                          flags=cv2.CALIB_FIX_INTRINSIC)

        # Initialize a list to store the errors
        errors_in_mm = []

        # Initialize a list to store the errors
        errors_in_mm = []

        # Lists to store all points for plotting
        all_points_cam1 = []
        all_points_cam2 = []
        # Iterate over the object points and their corresponding image points
        for obj_points, img_points_1, img_points_2 in zip(self.object_points_list, self.image_points_list_1,
                                                          self.image_points_list_2):
            # Estimate the position of the object points in the first camera coordinate system using PnP
            _, rvec_1, tvec_1 = cv2.solvePnP(obj_points, img_points_1, parameters_1.get_intrinsics_matrix_opencv(),
                                             parameters_1.get_distortion_coeffs_opencv())

            # Estimate the position of the object points in the second camera coordinate system using PnP
            _, rvec_2, tvec_2 = cv2.solvePnP(obj_points, img_points_2, parameters_2.get_intrinsics_matrix_opencv(),
                                             parameters_2.get_distortion_coeffs_opencv())

            # Convert rotation vectors to rotation matrices
            R_1, _ = cv2.Rodrigues(rvec_1)
            R_2, _ = cv2.Rodrigues(rvec_2)

            # Convert object points to homogeneous coordinates
            obj_points_homogeneous = np.hstack((obj_points, np.ones((obj_points.shape[0], 1))))

            # Create the transformation matrix for the first camera
            T_1 = np.eye(4)
            T_1[:3, :3] = R_1
            T_1[:3, 3] = tvec_1.flatten()

            # Transform the object points from the object coordinate system to the first camera coordinate system
            points_cam1_homogeneous = (T_1 @ obj_points_homogeneous.T).T
            points_cam1 = points_cam1_homogeneous[:, :3] / points_cam1_homogeneous[:, 3][:, np.newaxis]

            # Create the transformation matrix for the second camera relative to the first camera
            T_2 = np.eye(4)
            T_2[:3, :3] = R_2
            T_2[:3, 3] = tvec_2.flatten()

            # Transform the object points from the object coordinate system to the second camera coordinate system
            points_cam2_homogeneous = (T_2 @ obj_points_homogeneous.T).T
            points_cam2 = points_cam2_homogeneous[:, :3] / points_cam2_homogeneous[:, 3][:, np.newaxis]

            # Assuming R and T are already defined
            R_inv = R.T  # Transpose of the rotation matrix
            T_inv = -R_inv @ T  # Inverse translation

            # Create the transformation matrix for the second camera relative to the first camera
            T_3 = np.eye(4)
            T_3[:3, :3] = R_inv
            T_3[:3, 3] = T_inv.flatten()


            # Transform the points from the second camera coordinate system to the first camera coordinate system
            points_cam2_homogeneous = (T_3 @ points_cam2_homogeneous.T ).T
            points_cam2_in_cam1 = points_cam2_homogeneous[:, :3] / points_cam2_homogeneous[:, 3][:, np.newaxis]

            error_board=[]
            # Calculate the Euclidean distance between the transformed points and the actual object points
            for actual_point, transformed_point in zip(points_cam1, points_cam2_in_cam1):
                error = np.linalg.norm(actual_point - transformed_point)
                error_board.append(error)
            error = np.mean(error_board)
            errors_in_mm.append(error)

            # Store points for plotting
            all_points_cam1.append(points_cam1)
            all_points_cam2.append(points_cam2_in_cam1)

        # Calculate the mean error in millimeters
        mean_error_in_mm = np.mean(errors_in_mm)
        self.errors_in_mm = errors_in_mm

        print(f"Mean Error in millimeters: {mean_error_in_mm}")

        # Plot all checkerboards in 3D
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for points_cam1, points_cam2 in zip(all_points_cam1, all_points_cam2):
            ax.scatter(points_cam1[:, 0], points_cam1[:, 1], points_cam1[:, 2], color='red', marker='o', s=50,
                       label='Camera 1' if not ax.get_legend_handles_labels()[0] else "")
            ax.scatter(points_cam2[:, 0], points_cam2[:, 1], points_cam2[:, 2], color='blue', marker='x', s=50,
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
        cam_errs = {'camera 1': self.per_view_err[:, 0], 'camera 2': self.per_view_err[:, 0]}

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
        cam_errs = {'camera 1': self.per_view_err[:, 0], 'camera 2': self.per_view_err[:, 0]}

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
        if not os.path.exists(path):
            os.makedirs(path)
        for image_idx in range(image_array_1.shape[-1]):
            image_1 = image_array_1[..., image_idx]
            image_2 = image_array_2[..., image_idx]
            if len(image_1.shape) == 3:
                image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
                image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
            else:
                image_1 = cv2.cvtColor(image_1, cv2.COLOR_GRAY2RGB)
                image_2 = cv2.cvtColor(image_2, cv2.COLOR_GRAY2RGB)

            if image_idx in self.indices:
                feature_index = self.indices.index(image_idx)
                for point in self.image_points_list_1[feature_index]:
                    cv2.circle(image_1, (int(point[0]), int(point[1])), 10, (0, 255, 0), 1)  # Green for used features
                for point in self.image_points_list_2[feature_index]:
                    cv2.circle(image_2, (int(point[0]), int(point[1])), 10, (0, 255, 0), 1)  # Green for used features
            elif self.feature_list_1[image_idx].score != 0 and self.feature_list_2[image_idx].score != 0:
                for point in self.feature_list_1[image_idx].image_points:
                    cv2.circle(image_1, (int(point[0]), int(point[1])), 10, (255, 0, 0), 1)  # Red for detected features
                for point in self.feature_list_2[image_idx].image_points:
                    cv2.circle(image_2, (int(point[0]), int(point[1])), 10, (255, 0, 0), 1)  # Red for detected features

            # Add the text to the images
            cv2.putText(image_1, 'detected with pycbd, InViLab, doi:10.3390/math11224568', (10, image_1.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image_2, 'detected with pycbd, InViLab, doi:10.3390/math11224568', (10, image_2.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            save_filename_1 = os.path.join(path, f'image_1_{image_idx + 1}.png')
            save_filename_2 = os.path.join(path, f'image_2_{image_idx + 1}.png')
            cv2.imwrite(save_filename_1, cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(save_filename_2, cv2.cvtColor(image_2, cv2.COLOR_RGB2BGR))


class StereoParameters:
    """
    Object that contains all camera calibration parameters, this includes the calibration parameters for both cameras.

    :var camera_parameters_1: :py:class:`~PyCamCalib.core.calibration.CameraParameters` for camera 1.
    :var camera_parameters_2: :py:class:`~PyCamCalib.core.calibration.CameraParameters` for camera 2.
    :var rms_reproj_error: Overall rms re-projection error.
    :var R: The rotation matrix.
    :var T: The translation matrix.
    :var E: The essential matrix.
    :var F: The fundamental matrix.
    :var R_1: Rectification transform of camera 1.
    :var R_2: Rectification transform of camera 2.
    :var P_1: Projection matrix of camera 1.
    :var P_2: Projection matrix of camera 2.
    :var Q: Disparity-to-depth mapping matrix.
    :var roi_1: ROI where all pixels for camera 1 are valid.
    :var roi_2: ROI where all pixels for camera 2 are valid.
    :var map_1_x: x map for distortion correction and rectification for camera 1.
    :var map_1_y: y map for distortion correction and rectification for camera 1.
    :var map_2_x: x map for distortion correction and rectification for camera 2.
    :var map_2_y: y map for distortion correction and rectification for camera 2.
    """
    def __init__(self) -> None:
        """Class constructor."""
        self.camera_parameters_1: CameraParameters = CameraParameters()
        self.camera_parameters_2: CameraParameters = CameraParameters()
        self.rms_reproj_error: np.float64 = np.float64(0)
        self.R: npt.NDArray[np.float4] = np.zeros((3, 3))
        self.T: npt.NDArray[np.float4] = np.zeros((3, 1))
        self.E: npt.NDArray[np.float4] = np.zeros((3, 3))
        self.F: npt.NDArray[np.float4] = np.zeros((3, 3))
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
        """Save calibration parameters to .h5 file

        If the file specified in `full_save_path` does not exist, a new file will be created. If the file already
        exists, the parameters will be added to the specified file. If the file already exists and it already has
        calibration parameters, these parameters will be overwritten.

        :param full_save_path: Full filepath with directory and filename.
        :raises OSError: If the path contains forbidden characters or the selected file is not compatible.
        :raises FileNotFoundError: If the specified directory does not exist.
        """
        directory_path = os.path.dirname(full_save_path)
        # if dir does not exist
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
        self.H.save_to_json(os.path.splitext(full_save_path)[0]+ '_TransformationMatrix.json')
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
        """Load calibration parameters from .h5 file into object.

        :param full_save_path: Full filepath with directory and filename.
        :raises KeyError: If there are no calibration parameters in the h5 file.
        :raises OSError: If the path contains forbidden characters or an incompatible file is used.
        :raises FileNotFoundError: If specified file does not exist.
        """
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
        obj.H = group["H"][()]  # this is now a (4,4) array
        obj.info = group.attrs.get("info", ["unknown", "unknown"])
        obj.units = group.attrs.get("units", "mm")
        return obj

    def calculate_undistort_rectify_maps(self, alpha: float = 0, fixed_point_maps: bool = False) -> None:
        """Calculate rectification transforms and maps necessary for remapping.

        :param alpha: Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1
           (when all the source image pixels are retained in the undistorted image). If you set this at -1 OpenCV
           automatically pick a value.
        :param fixed_point_maps: Whether to transform the floating points map to a fixed-point representation. This
           speeds up pixel remapping, which might be useful for live video feeds.
        """
        sensor_dim_1 = self.camera_parameters_1.sensor_dimensions
        sensor_dim_2 = self.camera_parameters_2.sensor_dimensions
        if not np.array_equal(sensor_dim_1, sensor_dim_2):
            raise NotImplementedError("Rectification for different sensor sizes has not been implemented.")
        intrinsics_1 = self.camera_parameters_1.get_intrinsics_matrix_opencv()
        distortion_1 = self.camera_parameters_1.get_distortion_coeffs_opencv()
        intrinsics_2 = self.camera_parameters_2.get_intrinsics_matrix_opencv()
        distortion_2 = self.camera_parameters_2.get_distortion_coeffs_opencv()

        self.R_1, self.R_2, self.P_1, self.P_2, self.Q, self.roi_1, self.roi_2 \
            = cv2.stereoRectify(intrinsics_1,
                                distortion_1,
                                intrinsics_2,
                                distortion_2,
                                sensor_dim_1,
                                self.R,
                                self.T,
                                flags=cv2.CALIB_ZERO_DISPARITY,
                                alpha=alpha)

        if fixed_point_maps:
            map_type = cv2.CV_16SC2
        else:
            map_type = cv2.CV_32FC1
        self.map_1_x, self.map_1_y = cv2.initUndistortRectifyMap(intrinsics_1,
                                                                 distortion_1,
                                                                 self.R_1,
                                                                 self.P_1,
                                                                 sensor_dim_1,
                                                                 map_type)
        self.map_2_x, self.map_2_y = cv2.initUndistortRectifyMap(intrinsics_2,
                                                                 distortion_2,
                                                                 self.R_2,
                                                                 self.P_2,
                                                                 sensor_dim_2,
                                                                 map_type)

    def remap_images(self, frame_1: npt.NDArray, frame_2: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Remap the images so they are undistorted and rectified.

        :param frame_1: The image for camera 1 that needs to be remapped, this is either a 2D or 3D array.
        :param frame_2: The image for camera 2 that needs to be remapped, this is either a 2D or 3D array.
        :returns: The undistorted and rectified images.
        """
        try:
            rectified_1 = cv2.remap(frame_1, self.map_1_x, self.map_1_y, cv2.INTER_LINEAR)
            rectified_2 = cv2.remap(frame_2, self.map_2_x, self.map_2_y, cv2.INTER_LINEAR)
        except cv2.error as e:
            raise Exception("You probably did not calculate the undistort and rectification map before remapping.") from e

        return rectified_1, rectified_2
