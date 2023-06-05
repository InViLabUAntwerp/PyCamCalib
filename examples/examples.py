"""Collection of example scripts for the camera calibration toolbox.

Some scripts need to be executed in order.
"""

#%% Imports
from os import listdir
from os.path import join
import cv2
import numpy as np
from camera_calibration_toolbox.core.camera_calibration import CameraCalibrator, CameraParameters
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
calibration_parameters = calibrator.calibrate(image_array, 30)

#%% Plot all images with features if features were detected
for image_idx in range(n_images):
    plt.cla()
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image_array[..., image_idx], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image_array[..., image_idx])
    if image_idx in calibrator.indices:
        feature_index = calibrator.indices.index(image_idx)
        plt.plot(calibrator.image_points_list[feature_index][:, 0],
                 calibrator.image_points_list[feature_index][:, 1],
                 'o', color='lime')
        for i in range(0, calibrator.object_points_list[feature_index].shape[0]):
            plt.text(calibrator.image_points_list[feature_index][i, 0] + 5,
                     calibrator.image_points_list[feature_index][i, 1] - 5,
                     i + 1, color="lime")
        color = 'g'
    else:
        color = 'r'
    plt.title('Image nr ' + str(image_idx+1), color=color)
    plt.show()

#%% Plot reprojection errors
calibrator.plot_reproj_error()

#%% Visualize distortion
calibration_parameters.plot_distortion()

#%% Remove distortion from image
calibration_parameters.calculate_remap_parameters()
undistorted = calibration_parameters.remap_image(image)

plt.imshow(image)
plt.title("Original")
plt.show()

plt.imshow(undistorted)
plt.title("Undistorted")
plt.show()

#%% Save parameters
calibration_parameters.save_parameters('examples/example_parameters.h5')

#%% Load parameters
calibration_parameters = CameraParameters()
calibration_parameters.load_parameters('examples/example_parameters.h5')
