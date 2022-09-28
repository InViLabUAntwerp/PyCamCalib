"""Collection of example scripts for the camera calibration toolbox.

Some of the scripts need to be executed in order.
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
configure_logger('INFO')

#%% Load images into array
directory = 'examples/test_images'
files = [join(directory, f) for f in listdir(directory)]
n_images = len(files)
for image_idx in range(n_images):
    image = cv2.imread(files[image_idx])
    if image_idx == 0:
        h, w, c = image.shape
        image_array = np.zeros((h, w, c, n_images), dtype='uint8')
    image_array[:, :, :, image_idx] = image

#%% Perform calibration
calibrator = CameraCalibrator()
calibration_parameters = calibrator.calibrate_camera(image_array, 'NCH0000050630')

#%% Plot all images with features if features were detected
for feature_image_idx in range(n_images):
    plt.cla()
    plt.imshow(calibrator.feature_list[feature_image_idx].feature_image)
    plt.title('Image nr ' + str(feature_image_idx+1))
    plt.show()

#%% Plot reprojection errors
per_view_err = calibrator.per_view_err
image_indices = calibrator.indices
rms_reproj_error = calibrator.rms_reproj_error
plot_reproj_error(per_view_err, image_indices, rms_reproj_error)

#%% Visualize distortion
sensor_size = calibration_parameters.get_sensor_dimensions()
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
calibration_parameters.save_parameters_csv('examples/new_test_parameters.csv')

#%% Load parameters
calibration_parameters = CalibrationParameters()
calibration_parameters.load_parameters_csv('examples/test_parameters.csv')
