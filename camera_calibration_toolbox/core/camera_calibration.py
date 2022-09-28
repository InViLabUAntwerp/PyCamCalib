"""Module that contains all code for camera calibration."""

from __future__ import annotations
import numpy.typing as npt
from typing import Tuple
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from camera_calibration_toolbox.core.exceptions import ImageError, CalibrationError
from camera_calibration_toolbox.core.feature_detection import FeatureDetector
import logging


class CameraCalibrator:
    """Object used to calibrate a camera."""

    def __init__(self) -> None:
        """Class constructor."""
        self._logger = logging.getLogger(__name__)

        self.feature_list = []
        """List with :py:class:`.CalibrationFeature` objects for all images."""

        self.indices = []
        """Indices of all images in :py:data:`feature_list` that were used for the current calibration."""

        self.per_view_err = []
        """Per view re-projection errors for all images that are listed in :py:data:`indices`."""

        self.rms_reproj_error = 0
        """The rms re-projection error for the current calibration."""

        self.height = 0
        """Sensor height in pixels."""

        self.width = 0
        """Sensor width in pixels."""

    def calibrate_camera(self, image_array: npt.NDArray, tag: str, normalize: bool = False,
                         invert: bool = False) -> CalibrationParameters:
        """Calibrate camera.

        Calibrate a camera with an array of images of a calibration target. At least 11 good images are required for
        a good calibration.

        :param image_array: Array containing all images that will be used for calibration. It can either be a 3D array
            (h,w,n) for grayscale images or a 4D array (h,w,c,n) for multi channel images.
        :param tag: Feature tag of the calibration target that needs to be detected.
        :param normalize: Specifies whether the images should be normalized before feature detection is carried out,
            defaults to False.
        :param invert: Specifies whether the images should be inverted before feature detection is carried out,
            defaults to False.
        :returns: An object that contains all calibration parameter data.
        :raises TypeError: When image_array is not a numpy array or calibrator is not a string.
        :raises CalibrationError: When no features were detected in any of the images.
        """
        if not isinstance(image_array, np.ndarray):
            raise TypeError("``image_array`` should be a numpy array.")

        self.construct_feature_list(image_array, tag, normalize, invert)
        image_points_list, object_points_list = self.construct_points_lists([])
        if not image_points_list:
            raise CalibrationError("Failed to detect features in all images, unable to perform calibration.")
        elif len(image_points_list) < 11:
            self._logger.warning("Only detected features for " + str(len(image_points_list))
                                 + " images. Features from at least 11 images are necessary for an accurate calibration.")

        calibration_parameters = self.opencv_calibration(image_points_list, object_points_list)

        return calibration_parameters

    def calibrate_indices(self, indices: list) -> CalibrationParameters:
        """Repeat calibration with selected samples.

        Repeat the camera calibration with the samples specified in indices. This can be used to improve the
        camera calibration by removing outliers.

        :param indices: A list that contains the indices that correspond to the elements in :py:data:`feature_list`
           which should be used for a new calibration. Passing an empty list will perform a calibration with all good
           images.
        :returns: An object that contains all calibration parameter data.
        :raises TypeError: When indices is not a list.
        :raises CalibrationError: When no images with detected features were selected.
        """
        if not isinstance(indices, list):
            raise TypeError("``indices`` should be a list.")

        image_points_list, object_points_list = self.construct_points_lists(indices)
        if not image_points_list:
            raise CalibrationError("No images with detected features remain, unable to perform calibration.")
        if len(image_points_list) < 11:
            self._logger.warning("Only " + str(len(image_points_list))
                                 + " samples left. At least 11 samples are necessary for an accurate calibration.")

        calibration_parameters = self.opencv_calibration(image_points_list, object_points_list)

        return calibration_parameters

    def construct_feature_list(self, image_array: npt.NDArray, tag: str, normalize: bool, invert: bool) -> None:
        """Construct a list with the calibration features for all images in the image_array."""
        self.feature_list = []
        feature_detector = FeatureDetector(tag)
        if len(image_array.shape) == 3:
            self.height, self.width, n_images = image_array.shape
            for idx in range(n_images):
                feature = feature_detector.detect_feature(image_array[:, :, idx], normalize, invert)
                self.feature_list.append(feature)
                if not feature.score:
                    self._logger.info("Failed feature detection on image nr " + str(idx+1) + ".")
        elif len(image_array.shape) == 4:
            self.height, self.width, n_channels, n_images = image_array.shape
            for idx in range(n_images):
                feature = feature_detector.detect_feature(image_array[:, :, :, idx], normalize, invert)
                self.feature_list.append(feature)
                if not feature.score:
                    self._logger.info("Failed feature detection on image nr " + str(idx + 1) + ".")
        else:
            raise ImageError("``image_array`` should have 2 or 3 dimensions, the given array has  "
                             + str(len(image_array.shape)) + " dimensions.")

    def construct_points_lists(self, indices: list) -> Tuple[list, list]:
        """Construct lists of image points and object points for calibration.

        If indices is empty all images where a feature was detected will be used, otherwise only images that
        correspond to the elements in indices will be used.
        """
        self.indices = []
        object_points_list = []
        image_points_list = []
        if indices:
            for idx in indices:
                try:
                    feature = self.feature_list[idx]
                except IndexError:
                    pass
                else:
                    if feature.score:
                        image_points_list.append(feature.image_points)
                        object_points_list.append(feature.object_points)
                        self.indices.append(idx)
        else:
            for idx in range(len(self.feature_list)):
                feature = self.feature_list[idx]
                if feature.score:
                    image_points_list.append(feature.image_points)
                    object_points_list.append(feature.object_points)
                    self.indices.append(idx)

        return image_points_list, object_points_list

    def opencv_calibration(self, image_points_list: list, object_points_list: list) -> CalibrationParameters:
        """Regular OpenCV camera calibration."""

        rms_reproj_error, intrinsics_matrix, dist_coeffs, r_vecs, t_vecs, intrinsics_std, extrinsics_std, per_view_err \
            = cv2.calibrateCameraExtended(object_points_list, image_points_list, (self.width, self.height), None, None)

        self.rms_reproj_error = rms_reproj_error
        self.per_view_err = np.squeeze(per_view_err)
        dist_coeffs = np.squeeze(dist_coeffs)
        intrinsics_std = np.squeeze(intrinsics_std)
        extrinsics_std = np.squeeze(extrinsics_std)

        calibration_parameters = CalibrationParameters()
        calibration_parameters.set_parameters_opencv(self.rms_reproj_error, intrinsics_matrix, dist_coeffs, r_vecs,
                                                     t_vecs, intrinsics_std, extrinsics_std, self.per_view_err,
                                                     self.width, self.height)

        return calibration_parameters


class CalibrationParameters:
    """Object that contains all calibration parameters."""

    def __init__(self) -> None:
        """Class constructor."""
        self.fx = float(0)
        """x focal length in pixels."""

        self.fy = float(0)
        """y focal length in pixels."""

        self.cx = float(0)
        """x-coordinate of principal point in pixels."""

        self.cy = float(0)
        """y-coordinate of principal point in pixels."""

        self.s = float(0)
        """Skew."""

        self.fx_std = float(0)
        """Standard deviation of x focal length."""

        self.fy_std = float(0)
        """Standard deviation of y focal length."""

        self.cx_std = float(0)
        """Standard deviation of x-coordinate of principal point."""

        self.cy_std = float(0)
        """Standard deviation of y-coordinate of principal point."""

        self.s_std = float(0)
        """Standard deviation of skew."""

        self.radial_dist_coeffs = np.zeros(3)
        """Radial distortion coefficients."""

        self.tangential_dist_coeffs = np.zeros(2)
        """Tangential distortion coefficients."""

        self.radial_dist_coeffs_std = np.zeros(3)
        """Standard deviations of radial distortion coefficients."""

        self.tangential_dist_coeffs_std = np.zeros(2)
        """Standard deviations of tangential distortion coefficients."""

        self.r_vecs = np.array([])
        """Rotation vectors for each image."""

        self.t_vecs = np.array([])
        """Translation vectors for each image."""

        self.extrinsics_std = np.array([])
        """Standard deviations for extrinsic parameters."""

        self.rms_reproj_error = float(0)
        """Overall rms re-projection error."""

        self.per_view_err = np.array([])
        """Array of the re-projection error per image."""

        self.width = 0
        """Sensor width in pixels."""

        self.height = 0
        """Sensor height in pixels."""

    def get_sensor_dimensions(self) -> Tuple[int, int]:
        """Get the camera sensor dimensions in pixels.

        :returns: A tuple with the sensor width and height in pixels.
        """
        return self.width, self.height

    def get_afov(self) -> Tuple[float, float]:
        """Get the angular field of view in degrees.

        :returns: A tuple with the horizontal and vertical angular field of view in degrees.
        """
        h_afov = np.rad2deg(2 * np.arctan2(self.width, 2 * self.fx))
        v_afov = np.rad2deg(2 * np.arctan2(self.height, 2 * self.fy))

        return h_afov, v_afov

    def get_intrinsics_matrix_opencv(self) -> npt.NDArray[np.float64]:
        """Get the intrinsics matrix in OpenCV format.

        :returns: A 3x3 array with the intrinsic camera parameters in OpenCV format.
        """
        intrinsics_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        return intrinsics_matrix

    def get_distortion_coeffs_opencv(self) -> npt.NDArray[np.float64]:
        """Get a vector of the distortion coefficients in OpenCV format.

        :returns: A 5 element vector with the distortion coefficients in opencv format.
        """
        return np.array([self.radial_dist_coeffs[0], self.radial_dist_coeffs[1], self.tangential_dist_coeffs[0],
                         self.tangential_dist_coeffs[1], self.radial_dist_coeffs[2]])

    def get_remap_parameters(self, alpha: float = 0,
                             fixed_point_maps: bool = False) -> Tuple[npt.NDArray, npt.NDArray, Tuple]:
        """Get the parameters necessary for pixel remapping (removing distortion).

        :param alpha: Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1
            (when all the source image pixels are retained in the undistorted image).
        :param fixed_point_maps: Whether to transform the floating points map to a fixed-point representation. This
            speeds up pixel remapping, which might be useful for live video feeds.
        :returns: the x- and y-maps, and the coordinates that describe the new ROI.
        """
        intrinsics_matrix = self.get_intrinsics_matrix_opencv()
        distortion_coeffs = self.get_distortion_coeffs_opencv()
        dimensions = (self.width, self.height)
        new_intrinsics_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsics_matrix, distortion_coeffs, dimensions,
                                                                   alpha)
        map_x, map_y = cv2.initUndistortRectifyMap(intrinsics_matrix, distortion_coeffs, None, new_intrinsics_matrix,
                                                   dimensions, 5)
        if fixed_point_maps:
            map_x, map_y = cv2.convertMaps(map_x, map_y, dstmap1type=cv2.CV_16SC2)

        return map_x, map_y, roi

    def set_parameters_opencv(self, rms_reproj_error: float, intrinsics_matrix: npt.NDArray, dist_coeffs: npt.NDArray,
                              r_vecs: npt.NDArray, t_vecs: npt.NDArray, intrinsics_std: npt.NDArray,
                              extrinsics_std: npt.NDArray, per_view_err: npt.NDArray, width: int, height: int) -> None:
        """Save parameters from opencv calibration to object."""
        self.fx = intrinsics_matrix[0, 0]
        self.fy = intrinsics_matrix[1, 1]
        self.cx = intrinsics_matrix[0, 2]
        self.cy = intrinsics_matrix[1, 2]
        self.s = float(0)
        self.fx_std = intrinsics_std[0]
        self.fy_std = intrinsics_std[1]
        self.cx_std = intrinsics_std[2]
        self.cy_std = intrinsics_std[3]
        self.s_std = float(0)
        self.rms_reproj_error = rms_reproj_error
        self.radial_dist_coeffs[0] = dist_coeffs[0]
        self.radial_dist_coeffs[1] = dist_coeffs[1]
        self.radial_dist_coeffs[2] = dist_coeffs[4]
        self.tangential_dist_coeffs[0] = dist_coeffs[2]
        self.tangential_dist_coeffs[0] = dist_coeffs[3]
        self.radial_dist_coeffs_std[0] = intrinsics_std[4]
        self.radial_dist_coeffs_std[0] = intrinsics_std[5]
        self.radial_dist_coeffs_std[0] = intrinsics_std[8]
        self.tangential_dist_coeffs_std[0] = intrinsics_std[6]
        self.tangential_dist_coeffs_std[0] = intrinsics_std[7]
        self.r_vecs = r_vecs
        self.t_vecs = t_vecs
        self.extrinsics_std = extrinsics_std
        self.per_view_err = per_view_err
        self.width = width
        self.height = height

    def load_parameters_csv(self, full_path: str) -> None:
        """Load calibration parameters from .csv file into object.

        :param full_path: The full path to the .csv file that contains the calibration parameters.
        :raises FileNotFoundError: If specified file does not exist.
        """
        with open(full_path, newline='') as file:
            reader = csv.reader(file, delimiter=',', quotechar='|')
            next(reader)  # skip header row
            row = next(reader)
            self.fx = float(row[0])
            self.fy = float(row[1])
            self.cx = float(row[2])
            self.cy = float(row[3])
            self.s = float(row[4])
            self.fx_std = float(row[5])
            self.fy_std = float(row[6])
            self.cx_std = float(row[7])
            self.cy_std = float(row[8])
            self.s_std = float(row[9])
            self.rms_reproj_error = float(row[10])
            self.radial_dist_coeffs[0] = float(row[11])
            self.radial_dist_coeffs[1] = float(row[12])
            self.radial_dist_coeffs[2] = float(row[13])
            self.tangential_dist_coeffs[0] = float(row[14])
            self.tangential_dist_coeffs[1] = float(row[15])
            self.radial_dist_coeffs_std[0] = float(row[16])
            self.radial_dist_coeffs_std[1] = float(row[17])
            self.radial_dist_coeffs_std[2] = float(row[18])
            self.tangential_dist_coeffs_std[0] = float(row[19])
            self.tangential_dist_coeffs_std[1] = float(row[20])
            self.width = int(row[21])
            self.height = int(row[22])

    def save_parameters_csv(self, full_path: str) -> None:
        """Save calibration parameters to .csv file

        :raises OSError: If the path contains forbidden characters.
        :raises FileNotFoundError: If the specified directory does not exist.
        """
        header = ['fx', 'fy', 'cx', 'cy', 's', 'fx_std', 'fy_std', 'cx_std', 'cy_std', 's_std', 'rms_reproj_error',
                  'k1', 'k2', 'k3', 'p1', 'p2', 'k1_std', 'k2_std', 'k3_std', 'p1_std', 'p2_std', 'width', 'height']

        row = (self.fx, self.fy, self.cx, self.cy, self.s, self.fx_std, self.fy_std, self.cx_std, self.cy_std,
               self.s_std, self.rms_reproj_error, self.radial_dist_coeffs[0], self.radial_dist_coeffs[1],
               self.radial_dist_coeffs[2], self.tangential_dist_coeffs[0], self.tangential_dist_coeffs[1],
               self.radial_dist_coeffs_std[0], self.radial_dist_coeffs_std[1], self.radial_dist_coeffs_std[2],
               self.tangential_dist_coeffs_std[0], self.tangential_dist_coeffs_std[1], self.width, self.height)

        with open(full_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerow(row)


def remap_image(image: npt.NDArray, map_x: npt.NDArray, map_y: npt.NDArray,
                roi: Tuple[int, int, int, int]) -> npt.NDArray:
    """Remaps the image to remove distortion

    :param image: The image that needs to be remapped, this is either a 2D or 3D array.
    :param map_x: x-map for remapping, obtained from :py:class:`CalibrationParameters`. Fixed-point maps can remap
       faster than floating-point maps but are less accurate.
    :param map_y: y-map for remapping, obtained from :py:class:`CalibrationParameters`. Fixed-point maps can remap
       faster than floating-point maps but are less accurate.
    :param roi: Boundaries of the new ROI, obtained from :py:class:`CalibrationParameters`.
    :returns: The undistorted image.
    """

    undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    undistorted = undistorted[y:y + h + 1, x:x + w + 1]

    return undistorted


def calculate_fov(afov: Tuple[float, float], working_distance: float) -> Tuple[float, float]:
    """Calculate the size of the field of view.

    :param afov: The angular field of view as (horizontal, vertical).
    :param working_distance: The distance between the camera and object.
    :returns: The field of view as (horizontal, vertical).
    """

    h_fov = 2 * working_distance * np.tan(afov[0] / 2)
    v_fov = 2 * working_distance * np.tan(afov[1] / 2)

    return h_fov, v_fov


def plot_reproj_error(per_view_err: npt.NDArray[np.float64], image_indices: list, rms_reproj_error: float) -> None:
    """Plot mean re-projection error and re-projection error for each calibration image.

    :param per_view_err: re-projection error for each image.
    :param image_indices: Indices of the images that were used for calibration.
    :param rms_reproj_error: The RMS re-projection error.
    """
    plt.cla()
    bars = plt.bar(list(map(str, image_indices)), per_view_err, color='b')
    line = plt.axhline(rms_reproj_error, color='g', linestyle='--')
    plt.xlabel("Image index")
    plt.ylabel("Reprojection error")
    plt.title("Reprojection error for each detected image")
    plt.legend([bars, line], ['Image', 'RMS'])
    plt.show()


def plot_distortion(sensor_size: Tuple[int, int], m: npt.NDArray[np.float64], d: npt.NDArray[np.float64]) -> None:
    """Plot camera distortion.

    :param sensor_size: The size of the sensor in pixels (width, height)
    :param m: The intrinsics camera matrix in the OpenCV format.
    :param d: The distortion coefficients in OpenCV format.
    """
    width = sensor_size[0]
    height = sensor_size[1]
    n_steps = 20
    [u, v] = np.meshgrid(np.linspace(0, width - 1, n_steps), np.linspace(0, height - 1, n_steps))
    xyz = np.linalg.solve(m, np.vstack((np.ravel(u, order='F'), np.ravel(v, order='F'), np.ones(u.size))))
    xp = xyz[0, :] / xyz[2, :]
    yp = xyz[1, :] / xyz[2, :]
    r2 = xp ** 2 + yp ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3
    coef = 1 + np.dot(d[0], r2) + np.dot(d[1], r4) + np.dot(d[4], r6)
    xpp = xp * coef + 2 * np.dot(d[2], (xp * yp)) + np.dot(d[3], (r2 + 2 * xp ** 2))
    ypp = yp * coef + np.dot(d[2], (r2 + 2 * yp ** 2)) + 2 * np.dot(d[3], (xp * yp))
    u2 = m[0, 0] * xpp + m[0, 2]
    v2 = m[1, 1] * ypp + m[1, 2]
    du = u2 - np.ravel(u, order='F')
    dv = v2 - np.ravel(v, order='F')
    dr = np.reshape(np.hypot(du, dv), u.shape, order='F')

    # plot
    plt.cla()
    plt.quiver(np.ravel(u, order='F') + 1, np.ravel(v, order='F') + 1, du, dv, color='b')
    plt.plot(width / 2, height / 2, 'x', label='Sensor center')
    plt.plot(m[0, 2], m[1, 2], 'o', label='Principal point')
    contour_set = plt.contour(u[0, :] + 1, v[:, 0] + 1, dr, colors='k')
    plt.clabel(contour_set, inline=1, fontsize=10)
    plt.xlim(1, width)
    plt.ylim(1, height)
    plt.title('Radial distortion model')
    plt.xlabel('Horizontal')
    plt.ylabel('Vertical')
    plt.show()
