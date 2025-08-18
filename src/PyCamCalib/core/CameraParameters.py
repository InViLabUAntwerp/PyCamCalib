from __future__ import annotations
import numpy.typing as npt
from typing import Tuple, Optional
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
import json
import os
import copy
try:
    from CTPv.Plucker.Line import Line
except:
    from core_toolbox_python.Plucker.Line import Line
try:
    from CTPv.Transformation.TransformationMatrix import TransformationMatrix
except:
    from core_toolbox_python.Transformation.TransformationMatrix import TransformationMatrix
import matplotlib

matplotlib.use('TkAgg')


class CameraParameters:
    """Object that contains all camera calibration parameters.

    :var f: Focal length in pixels (x, y).
    :var f_std: Standard deviation of focal length (x, y).
    :var c: Principal point in pixels (x, y).
    :var c_std: Standard deviation of principal point (x, y).
    :var s: Skew.
    :var s_std: Standard deviation of skew.
    :var radial_dist_coeffs: Radial distortion coefficients.
    :var radial_dist_coeffs_std: Standard deviations of radial distortion coefficients.
    :var tangential_dist_coeffs: Tangential distortion coefficients.
    :var tangential_dist_coeffs_std: Standard deviations of tangential distortion coefficients.
    :var rms_reproj_error: Overall rms re-projection error.
    :var sensor_dimensions: Sensor dimensions in pixels (w, h).
    :var map_x: x map for distortion correction.
    :var map_y: y map for distortion correction.
    :var roi: ROI used to crop image after distortion correction.
    :var pixel_size: Pixel size in mm.
    :var info: Additional information (camera id, lens id, ...).
    """

    def __init__(self) -> None:
        """Class constructor."""
        self.f: npt.NDArray[np.float64] = np.zeros(2)
        self.f_std: npt.NDArray[np.float64] = np.zeros(2)
        self.c: npt.NDArray[np.float64] = np.zeros(2)
        self.c_std: npt.NDArray[np.float64] = np.zeros(2)
        self.s: np.float64 = np.float64(0)
        self.s_std: np.float64 = np.float64(0)
        self.radial_dist_coeffs: npt.NDArray[np.float64] = np.zeros(3)
        self.radial_dist_coeffs_std: npt.NDArray[np.float64] = np.zeros(3)
        self.tangential_dist_coeffs: npt.NDArray[np.float64] = np.zeros(2)
        self.tangential_dist_coeffs_std: npt.NDArray[np.float64] = np.zeros(2)
        self.rms_reproj_error: np.float64 = np.float64(0)
        self.sensor_dimensions: npt.NDArray[np.int32] = np.zeros(2, dtype=np.int32)
        self.map_x: Optional[npt.NDArray] = None
        self.map_y: Optional[npt.NDArray] = None
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.pixel_size: Optional[float] = None
        self.info: Optional[str] = None

    def get_afov(self) -> Tuple[np.float64, np.float64]:
        """Get the angular field of view in degrees.

        :returns: A tuple with the horizontal and vertical angular field of view in degrees.
        """
        h_afov = np.rad2deg(2 * np.arctan2(self.sensor_dimensions[0], 2 * self.f[0]))
        v_afov = np.rad2deg(2 * np.arctan2(self.sensor_dimensions[1], 2 * self.f[1]))
        return h_afov, v_afov

    def get_fov(self, working_distance: float) -> Tuple[float, float]:
        """Calculate the size of the field of view.

        :param working_distance: The distance between the camera and object.
        :returns: The field of view as (horizontal, vertical).
        """
        afov = self.get_afov()
        h_fov = 2 * working_distance * np.tan(np.deg2rad(afov[0] / 2))
        v_fov = 2 * working_distance * np.tan(np.deg2rad(afov[1] / 2))
        return h_fov, v_fov

    def get_intrinsics_matrix_opencv(self) -> npt.NDArray[np.float64]:
        """Get the intrinsics matrix in OpenCV format.

        :returns: A 3x3 array with the intrinsic camera parameters in OpenCV format.
        """
        return np.array([[self.f[0], 0, self.c[0]], [0, self.f[1], self.c[1]], [0, 0, 1]])

    def get_distortion_coeffs_opencv(self) -> npt.NDArray[np.float64]:
        """Get a vector of the distortion coefficients in OpenCV format.

        :returns: A 5 element vector with the distortion coefficients in OpenCV format.
        """
        return np.array([self.radial_dist_coeffs[0], self.radial_dist_coeffs[1], self.tangential_dist_coeffs[0],
                         self.tangential_dist_coeffs[1], self.radial_dist_coeffs[2]])

    def calculate_undistort_map(self, alpha: float = 0, fixed_point_maps: bool = False) -> None:
        """Calculate the maps necessary for pixel remapping (removing distortion).

        :param alpha: Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1
            (when all the source image pixels are retained in the undistorted image). If you set this at -1 OpenCV
            automatically pick a value.
        :param fixed_point_maps: Whether to transform the floating points map to a fixed-point representation. This
            speeds up pixel remapping, which might be useful for live video feeds.
        """
        intrinsics_matrix = self.get_intrinsics_matrix_opencv()
        distortion_coeffs = self.get_distortion_coeffs_opencv()
        dimensions = self.sensor_dimensions
        if fixed_point_maps:
            map_type = cv2.CV_16SC2
        else:
            map_type = cv2.CV_32FC1
        new_intrinsics_matrix, self.roi = cv2.getOptimalNewCameraMatrix(intrinsics_matrix, distortion_coeffs,
                                                                        dimensions, alpha)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(intrinsics_matrix, distortion_coeffs, None,
                                                             new_intrinsics_matrix, dimensions, map_type)

    def remap_image(self, image: npt.NDArray) -> npt.NDArray:
        """Remaps the image to remove distortion

        Undistort an image with the maps that were calculated using :py:meth:`calculate_undistort_map`.

        :param image: The image that needs to be remapped, this is either a 2D or 3D array.
        :returns: The undistorted image.
        """
        try:
            undistorted = cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)
        except cv2.error as e:
            raise Exception("You probably did not calculate the undistort map before remapping.") from e
        return undistorted

    def set_parameters_opencv(self, rms_reproj_error: float, intrinsics_matrix: npt.NDArray,
                              dist_coeffs: npt.NDArray, intrinsics_std: npt.NDArray,
                              sensor_dimensions: npt.NDArray) -> None:
        """Save parameters from OpenCV calibration to object."""
        self.f = np.array([intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]])
        self.f_std = np.array([intrinsics_std[0], intrinsics_std[1]])
        self.c = np.array([intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]])
        self.c_std = np.array([intrinsics_std[2], intrinsics_std[3]])
        self.s = np.float64(0)
        self.s_std = np.float64(0)
        self.radial_dist_coeffs = np.array([dist_coeffs[0], dist_coeffs[1], dist_coeffs[4]])
        self.radial_dist_coeffs_std = np.array([intrinsics_std[4], intrinsics_std[5], intrinsics_std[8]])
        self.tangential_dist_coeffs = np.array([dist_coeffs[2], dist_coeffs[3]])
        self.tangential_dist_coeffs_std = np.array([intrinsics_std[6], intrinsics_std[7]])
        self.rms_reproj_error = rms_reproj_error
        self.sensor_dimensions = sensor_dimensions
        self.map_x = None
        self.map_y = None
        self.roi = None

    def save_parameters(self, full_save_path: str, internal_path: str = "camera_calibration/camera_parameters") -> None:
        """Save calibration parameters to .h5 file

        If the file specified in `full_save_path` does not exist, a new file will be created. If the file already
        exists, the parameters will be added to the specified file. If the file already exists and it already has
        calibration parameters, these parameters will be overwritten.

        :param full_save_path: Full filepath with directory and filename.
        :param internal_path: Internal h5 file directory where the parameters should be saved, use '/'as separator.
        :raises OSError: If the path contains forbidden characters or the selected file is not compatible.
        :raises FileNotFoundError: If the specified directory does not exist.
        """
        with h5py.File(full_save_path, "a") as file:
            try:
                file.create_dataset(internal_path + "/f", data=self.f)
                file.create_dataset(internal_path + "/f_std", data=self.f_std)
                file.create_dataset(internal_path + "/c", data=self.c)
                file.create_dataset(internal_path + "/c_std", data=self.c_std)
                file.create_dataset(internal_path + "/s", data=self.s)
                file.create_dataset(internal_path + "/s_std", data=self.s_std)
                file.create_dataset(internal_path + "/radial_dist_coeffs", data=self.radial_dist_coeffs)
                file.create_dataset(internal_path + "/radial_dist_coeffs_std", data=self.radial_dist_coeffs_std)
                file.create_dataset(internal_path + "/tangential_dist_coeffs", data=self.tangential_dist_coeffs)
                file.create_dataset(internal_path + "/tangential_dist_coeffs_std", data=self.tangential_dist_coeffs_std)
                file.create_dataset(internal_path + "/rms_reproj_err", data=self.rms_reproj_error)
                file.create_dataset(internal_path + "/sensor_dimensions", data=self.sensor_dimensions)
            except ValueError:
                file[internal_path + "/f"][()] = self.f
                file[internal_path + "/f_std"][()] = self.f_std
                file[internal_path + "/c"][()] = self.c
                file[internal_path + "/c_std"][()] = self.c_std
                file[internal_path + "/s"][()] = self.s
                file[internal_path + "/s_std"][()] = self.s_std
                file[internal_path + "/radial_dist_coeffs"][()] = self.radial_dist_coeffs
                file[internal_path + "/radial_dist_coeffs_std"][()] = self.radial_dist_coeffs_std
                file[internal_path + "/tangential_dist_coeffs"][()] = self.tangential_dist_coeffs
                file[internal_path + "/tangential_dist_coeffs_std"][()] = self.tangential_dist_coeffs_std
                file[internal_path + "/rms_reproj_err"][()] = self.rms_reproj_error
                file[internal_path + "/sensor_dimensions"][()] = self.sensor_dimensions

    def load_parameters(self, full_save_path: str, internal_path: str = "camera_calibration/camera_parameters") -> None:
        """Load calibration parameters from .h5 file into object.

        :param full_save_path: Full filepath with directory and filename.
        :param internal_path: Internal h5 file directory where the parameters are saved, use '/'as separator.
        :raises KeyError: If there are no calibration parameters in the h5 file.
        :raises OSError: If the path contains forbidden characters or an incompatible file is used.
        :raises FileNotFoundError: If specified file does not exist.
        """
        with h5py.File(full_save_path, "r") as file:
            try:
                self.f = file[internal_path + "/f"][()]
                self.f_std = file[internal_path + "/f_std"][()]
                self.c = file[internal_path + "/c"][()]
                self.c_std = file[internal_path + "/c_std"][()]
                self.s = file[internal_path + "/s"][()]
                self.s_std = file[internal_path + "/s_std"][()]
                self.radial_dist_coeffs = file[internal_path + "/radial_dist_coeffs"][()]
                self.radial_dist_coeffs_std = file[internal_path + "/radial_dist_coeffs_std"][()]
                self.tangential_dist_coeffs = file[internal_path + "/tangential_dist_coeffs"][()]
                self.tangential_dist_coeffs_std = file[internal_path + "/tangential_dist_coeffs_std"][()]
                self.rms_reproj_error = file[internal_path + "/rms_reproj_err"][()]
                self.sensor_dimensions = file[internal_path + "/sensor_dimensions"][()]
            except KeyError as e:
                raise KeyError("File does not contain calibration parameters.") from e

    def plot_distortion(self) -> None:
        """Plot camera distortion."""
        """Plot camera distortion."""
        width = self.sensor_dimensions[0]
        height = self.sensor_dimensions[1]
        m = self.get_intrinsics_matrix_opencv()
        d = self.get_distortion_coeffs_opencv()
        n_steps = 20
        [u, v] = np.meshgrid(np.linspace(0, width - 1, n_steps), np.linspace(0, height - 1, n_steps))
        xyz = np.linalg.solve(m, np.vstack((np.ravel(u, order='F'), np.ravel(v, order='F'), np.ones(u.size))))
        xp = xyz[0, :] / xyz[2, :]
        yp = xyz[1, :] / xyz[2, :]
        r2 = xp ** 2 + yp ** 2
        r4 = r2 ** 2
        r6 = r2 ** 3

        # Check if there is no distortion
        if d[0] ==0:
            xpp = xp
            ypp = yp
        else:
            # Radial distortion
            coef = 1 + d[0] * r2 + d[1] * r4 + d[4] * r6



            # Tangential distortion
            if d[2] != 0 or d[3] != 0:
                xpp = xp * coef + 2 * d[2] * (xp * yp) + d[3] * (r2 + 2 * xp ** 2)
                ypp = yp * coef + d[2] * (r2 + 2 * yp ** 2) + 2 * d[3] * (xp * yp)
            else:
                xpp = xp * coef
                ypp = yp * coef

        u2 = m[0, 0] * xpp + m[0, 2]
        v2 = m[1, 1] * ypp + m[1, 2]
        du = u2 - np.ravel(u, order='F')
        dv = v2 - np.ravel(v, order='F')
        dr = np.reshape(np.hypot(du, dv), u.shape, order='F')

        # plot
        plt.cla()
        if d[0] != 0:
            plt.quiver(np.ravel(u, order='F') + 1, np.ravel(v, order='F') + 1, du, dv, color='b')
        plt.plot(width / 2, height / 2, 'x', label='Sensor center')
        plt.plot(m[0, 2], m[1, 2], 'o', label='Principal point')
        if d[0] != 0:
            contour_set = plt.contour(u[0, :] + 1, v[:, 0] + 1, dr, colors='k')
            plt.clabel(contour_set, inline=1, fontsize=10)
        plt.xlim(1, width)
        plt.ylim(1, height)
        plt.title('Radial distortion model')
        plt.xlabel('Horizontal')
        plt.ylabel('Vertical')
        plt.show()

    def save_distortion_image(self, save_path: str = "distortion_plot.png") -> None:
        """Plot and save camera distortion image instead of displaying it."""
        width = self.sensor_dimensions[0]
        height = self.sensor_dimensions[1]
        m = self.get_intrinsics_matrix_opencv()
        d = self.get_distortion_coeffs_opencv()
        n_steps = 20
        [u, v] = np.meshgrid(np.linspace(0, width - 1, n_steps), np.linspace(0, height - 1, n_steps))
        xyz = np.linalg.solve(m, np.vstack((np.ravel(u, order='F'), np.ravel(v, order='F'), np.ones(u.size))))
        xp = xyz[0, :] / xyz[2, :]
        yp = xyz[1, :] / xyz[2, :]
        r2 = xp ** 2 + yp ** 2
        r4 = r2 ** 2
        r6 = r2 ** 3

        # Check if there is no distortion
        if d[0] == 0:
            xpp = xp
            ypp = yp
        else:
            # Radial distortion
            coef = 1 + d[0] * r2 + d[1] * r4 + d[4] * r6

            # Tangential distortion
            if d[2] != 0 or d[3] != 0:
                xpp = xp * coef + 2 * d[2] * (xp * yp) + d[3] * (r2 + 2 * xp ** 2)
                ypp = yp * coef + d[2] * (r2 + 2 * yp ** 2) + 2 * d[3] * (xp * yp)
            else:
                xpp = xp * coef
                ypp = yp * coef

        u2 = m[0, 0] * xpp + m[0, 2]
        v2 = m[1, 1] * ypp + m[1, 2]
        du = u2 - np.ravel(u, order='F')
        dv = v2 - np.ravel(v, order='F')
        dr = np.reshape(np.hypot(du, dv), u.shape, order='F')

        # Plot
        plt.cla()
        if d[0] != 0:
            plt.quiver(np.ravel(u, order='F') + 1, np.ravel(v, order='F') + 1, du, dv, color='b')
        plt.plot(width / 2, height / 2, 'x', label='Sensor center')
        plt.plot(m[0, 2], m[1, 2], 'o', label='Principal point')
        if d[0] != 0:
            contour_set = plt.contour(u[0, :] + 1, v[:, 0] + 1, dr, colors='k')
            plt.clabel(contour_set, inline=1, fontsize=10)
        plt.xlim(1, width)
        plt.ylim(1, height)
        plt.title('Radial distortion model')
        plt.xlabel('Horizontal')
        plt.ylabel('Vertical')

        # Save the plot instead of showing it
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

    @property
    def focal_length_mm(self) -> Tuple[float, float]:
        """Get the focal length in millimeters."""
        if self.pixel_size is None:
            raise ValueError('Pixel size not set!')
        return self.f[0] * self.pixel_size, self.f[1] * self.pixel_size

    @property
    def PerspectiveAngle(self) -> Tuple[np.float64, np.float64]:
        """Get the perspective angle in degrees."""
        if self.sensor_dimensions[0] == 0 or self.sensor_dimensions[1] == 0:
            raise ValueError('Sensor dimensions not set!')
        h_afov = 2 * np.arctan(self.sensor_dimensions[0] / (2 * self.f[0])) * 180 / np.pi
        v_afov = 2 * np.arctan(self.sensor_dimensions[1] / (2 * self.f[1])) * 180 / np.pi
        return h_afov, v_afov

    @PerspectiveAngle.setter
    def PerspectiveAngle(self, p: float) -> None:
        """Set the perspective angle in degrees."""
        if self.sensor_dimensions[0] == 0 or self.sensor_dimensions[1] == 0:
            raise ValueError('Sensor dimensions not set!')
        aspect_ratio = self.sensor_dimensions[0] / self.sensor_dimensions[1]
        if aspect_ratio > 1:
            self.f[0] = (self.sensor_dimensions[0] / 2) / np.tan(np.deg2rad(p) / 2)
            self.f[1] = self.f[0]
            self.c[0] = self.sensor_dimensions[0] / 2
            self.c[1] = self.sensor_dimensions[1] / 2
        else:
            self.f[1] = (self.sensor_dimensions[1] / 2) / np.tan(np.deg2rad(p) / 2)
            self.f[0] = self.f[1]
            self.c[0] = self.sensor_dimensions[0] / 2
            self.c[1] = self.sensor_dimensions[1] / 2

    def generate_rays(self, schaal: float = 2.0, subsampling: float = 1.0) -> Line:
        """
        Generate rays for every pixel in the image based on the intrinsic matrix,
        with optional sub-pixel resolution control.

        Parameters:
        - schaal: float (default=2.0), scaling factor for visualization.
        - subsampling: float (default=1.0), factor to control ray density:
            * < 1.0 increases ray count (super-sampling),
            * > 1.0 decreases ray count (downsampling).
        """
        if subsampling <= 0:
            raise ValueError("Subsampling must be greater than 0")

        # Calculate new resolution based on subsampling factor
        height, width = self.sensor_dimensions[1], self.sensor_dimensions[0]
        new_width = int(width / subsampling)
        new_height = int(height / subsampling)

        # Create a grid of pixel coordinates based on the new resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(0, width - 1, new_width),
            np.linspace(0, height - 1, new_height)
        )

        # Apply intrinsic matrix correction with or without distortion
        if self.radial_dist_coeffs[0] == 0:
            x_vals = (x_vals - self.c[0]) / self.f[0] * schaal
            y_vals = (y_vals - self.c[1]) / self.f[1] * schaal
            z_vals = np.ones_like(x_vals) * schaal
        else:
            x_norm = (x_vals - self.c[0]) / self.f[0]
            y_norm = (y_vals - self.c[1]) / self.f[1]
            r2 = x_norm ** 2 + y_norm ** 2
            radial_factor = 1 + self.radial_dist_coeffs[0] * r2 + \
                            self.radial_dist_coeffs[1] * r2 ** 2 + \
                            self.radial_dist_coeffs[2] * r2 ** 3
            x_vals = x_norm * radial_factor * schaal
            y_vals = y_norm * radial_factor * schaal
            z_vals = np.ones_like(x_vals) * schaal

        Ps = np.stack([x_vals, y_vals, z_vals], axis=-1)
        Pf = np.zeros_like(Ps)
        Pf[..., 2] = 0  # Rays originate in the image plane (Z = 0)

        directions = Ps - Pf
        rays = Line()
        rays.Ps = Pf.reshape(-1, 3)
        rays.V = directions.reshape(-1, 3)

        return rays


    def save_parameters_to_json(self, full_save_path: str) -> None:
        """Save calibration parameters to a JSON file.

        If the file specified in `full_save_path` does not exist, a new file will be created. If the file already
        exists, it will be overwritten.

        :param full_save_path: Full filepath with directory and filename.
        :raises OSError: If the path contains forbidden characters or the selected file is not compatible.
        :raises FileNotFoundError: If the specified directory does not exist.
        """
        directory = os.path.dirname(full_save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        data = {
            'f': self.f.tolist(),
            'f_std': self.f_std.tolist(),
            'c': self.c.tolist(),
            'c_std': self.c_std.tolist(),
            's': self.s,
            's_std': self.s_std,
            'radial_dist_coeffs': self.radial_dist_coeffs.tolist(),
            'radial_dist_coeffs_std': self.radial_dist_coeffs_std.tolist(),
            'tangential_dist_coeffs': self.tangential_dist_coeffs.tolist(),
            'tangential_dist_coeffs_std': self.tangential_dist_coeffs_std.tolist(),
            'rms_reproj_error': self.rms_reproj_error,
            'sensor_dimensions': self.sensor_dimensions.tolist(),
            'pixel_size': self.pixel_size,
            'info': self.info
        }

        with open(full_save_path, 'w') as file:
            json.dump(data, file, indent=4)

    def load_parameters_from_json(self, full_save_path: str) -> None:
        """Load calibration parameters from a JSON file into the object.

        :param full_save_path: Full filepath with directory and filename.
        :raises FileNotFoundError: If the specified file does not exist.
        :raises json.JSONDecodeError: If the file is not a valid JSON file.
        :raises KeyError: If the JSON file does not contain the expected keys.
        """
        with open(full_save_path, 'r') as file:
            data = json.load(file)

        self.f = np.array(data['f'])
        self.f_std = np.array(data['f_std'])
        self.c = np.array(data['c'])
        self.c_std = np.array(data['c_std'])
        self.s = data['s']
        self.s_std = data['s_std']
        self.radial_dist_coeffs = np.array(data['radial_dist_coeffs'])
        self.radial_dist_coeffs_std = np.array(data['radial_dist_coeffs_std'])
        self.tangential_dist_coeffs = np.array(data['tangential_dist_coeffs'])
        self.tangential_dist_coeffs_std = np.array(data['tangential_dist_coeffs_std'])
        self.rms_reproj_error = data['rms_reproj_error']
        self.sensor_dimensions = np.array(data['sensor_dimensions'])
        self.pixel_size = data['pixel_size']
        self.info = data['info']


if __name__ == "__main__":

    I_n = CameraParameters()
    I_n.load_parameters(full_save_path = r"C:\Users\Seppe\PycharmProjects\Calibration_Toolbox\calibration_toolbox_python\examples\example_camera_parameters.h5")
    I_n.plot_distortion()
    I = CameraParameters()
    I.info = "testCamera"
    I.f = np.array([1770, 1770])
    I.sensor_dimensions = np.array([1440, 1080])
    I.c = np.array([685, 492])
    I.radial_dist_coeffs = np.array([0, 0, 0])
    I.tangential_dist_coeffs = np.array([0, 0])
    I.plot_distortion()


    #I.save_intrinsics_to_json('test.json')
    I.save_parameters_to_json('test.json')
    rays = I.generate_rays()
    rays.PlotLine()
    I2 = CameraParameters()
    I2.load_parameters_from_json('test.json')
    # rays.PlotLine()

    H = TransformationMatrix()
    H.T = [0, 0, 0]
    H.angles_degree = [45, 0, 0]
    rayst = copy.deepcopy(rays)
    rayst.TransformLines(H)
    # rayst.PlotLine()
    import time
    start = time.time()

    from core_toolbox_python.Plucker.Line import *

    p, d = intersection_between_2_lines(rays, rayst)
    print("elapsed time: ", time.time() - start)

    # standard deviation and mean of d
    print("mean: ", np.mean(d))
    print("std: ", np.std(d))

    print(I.PerspectiveAngle)
    I.PerspectiveAngle = np.deg2rad(60)
    print(I.get_intrinsics_matrix_opencv())
    print(I.PerspectiveAngle)