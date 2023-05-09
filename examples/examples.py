"""Collection of example scripts for the camera calibration toolbox.

Some scripts need to be executed in order.
"""

#%% Imports
from os import listdir
from os.path import join
import cv2
import numpy as np
from camera_calibration_toolbox.core.camera_calibration import (CameraCalibrator, CalibrationParameters, remap_image,
                                                                plot_reproj_error, plot_distortion, calculate_fov)
import matplotlib.pyplot as plt
from camera_calibration_toolbox.package_logger import configure_logger
#%% Configure logger (optional)
configure_logger()

#%% Load images into array
directory = 'examples/PT'
files = [join(directory, f) for f in listdir(directory)]
n_images = len(files)
for image_idx in range(n_images):
    image = cv2.imread(files[image_idx])
    if image_idx == 0:
        image_array = np.zeros((image.shape + (n_images,)), dtype=image.dtype)
    image_array[..., image_idx] = image

#%% Perform calibration
calibrator = CameraCalibrator()
calibration_parameters = calibrator.calibrate_camera(image_array, 30)

#%% Plot all images with features if features were detected
for image_idx in range(n_images):
    plt.cla()
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image_array[..., image_idx], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image_array[..., image_idx])
    plt.plot(calibrator.feature_list[image_idx].image_points[:, 0],
             calibrator.feature_list[image_idx].image_points[:, 1],
             'o', color='lime')
    for i in range(0, calibrator.feature_list[image_idx].image_points.shape[0]):
        plt.text(calibrator.feature_list[image_idx].image_points[i, 0] + 5,
                 calibrator.feature_list[image_idx].image_points[i, 1] - 5,
                 i + 1, color="lime")
    plt.title('Image nr ' + str(image_idx+1))
    plt.show()

#%% Plot reprojection errors
per_view_err = calibrator.per_view_err
image_indices = calibrator.indices
rms_reproj_error = calibrator.rms_reproj_error
plot_reproj_error(per_view_err, image_indices, rms_reproj_error)

#%% Visualize distortion
sensor_size = calibration_parameters.sensor_dimensions
intrinsics_matrix = calibration_parameters.get_intrinsics_matrix_opencv()
distortion_coeffs = calibration_parameters.get_distortion_coeffs_opencv()
plot_distortion(sensor_size, intrinsics_matrix, distortion_coeffs)

#%% Remove distortion from image
image = image_array[:, :, :, 0]
map_x, map_y, roi = calibration_parameters.get_remap_parameters()
undistorted = remap_image(image, map_x, map_y, roi)

#%% Calculate FOV
working_distance = 2000.00
afov = calibration_parameters.get_afov()
h_fov, v_fov = calculate_fov(afov, working_distance)

#%% Save parameters
calibration_parameters.save_parameters('examples/new_parameters.h5')

#%% Load parameters
calibration_parameters = CalibrationParameters()
calibration_parameters.load_parameters('examples/example_parameters.h5')
