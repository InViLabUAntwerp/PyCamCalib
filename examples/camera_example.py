"""Collection of example scripts for performing camera calibration.

Some scripts need to be executed in order.
"""

#%% Imports
from src.PyCamCalib.core.calibration import CameraCalibrator
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

#%% Load images into array
files = glob.glob('camera_images/*.tiff')
n_images = len(files)
for image_idx in range(n_images):
    image = cv2.imread(files[image_idx])
    if image_idx == 0:
        image_array = np.zeros((image.shape + (n_images,)), dtype=image.dtype)
    image_array[..., image_idx] = image

#%% Perform calibration
calibrator = CameraCalibrator()
camera_parameters = calibrator.calibrate(image_array, 30)

#%% Plot all images with features if features were detected
for image_idx in range(n_images):
    image = image_array[..., image_idx]
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image)
    if image_idx in calibrator.indices:
        feature_index = calibrator.indices.index(image_idx)
        plt.plot(calibrator.image_points_list[feature_index][:, 0],
                 calibrator.image_points_list[feature_index][:, 1],
                 '-o', color='lime')
        color = 'g'
    elif calibrator.feature_list[image_idx].score != 0:
        plt.plot(calibrator.feature_list[image_idx].image_points[:, 0],
                 calibrator.feature_list[image_idx].image_points[:, 1],
                 '-o', color='r')
        color = 'r'
    else:
        color = 'r'
    plt.title('Image nr ' + str(image_idx+1), color=color)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.show(block = True)

#%% Plot reprojection errors
calibrator.plot_reproj_error()

#%% Retry calibration after removing potential outliers
# There are none but this is for the sake of the example.

#calibrator.calibrate_indices([0, 1, 2, 4, 5, 7, 8, 9, 10])
#calibrator.plot_reproj_error()

new_indices = calibrator.plot_and_filter_reproj_error()
calibrator.calibrate_indices(new_indices)
calibrator.plot_reproj_error()
#%% Visualize distortion
camera_parameters.plot_distortion()

#%% Remove distortion from image
# Calculate maps, only needs to be performed once.
camera_parameters.calculate_undistort_map()

# Undistort image.
image_index = 0
image = image_array[..., image_index]
undistorted = camera_parameters.remap_image(image)

# Plot results
fig, axarr = plt.subplots(1, 2, constrained_layout=True)
axarr[0].imshow(image)
axarr[0].set_title("Original")
axarr[1].imshow(undistorted)
rect = Rectangle((camera_parameters.roi[0], camera_parameters.roi[1]),
                   camera_parameters.roi[2], camera_parameters.roi[3],
                   linewidth=2, edgecolor='lime', facecolor='none')
axarr[1].add_patch(rect)
axarr[1].set_title("Undistorted")
axarr[0].get_xaxis().set_visible(False)
axarr[0].get_yaxis().set_visible(False)
axarr[1].get_xaxis().set_visible(False)
axarr[1].get_yaxis().set_visible(False)
plt.show()

#%% Save parameters
camera_parameters.save_parameters('example_camera_parameters.h5')
