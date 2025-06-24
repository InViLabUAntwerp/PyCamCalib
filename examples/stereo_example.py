"""Collection of example scripts for performing stereo calibration.

Some scripts need to be executed in order.
"""

#%% Imports
from src.PyCamCalib.core.calibration import CameraCalibrator, StereoCalibrator
a=1
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle


#%% Load image sets for both cameras
files = sorted(glob.glob('stereo_images/left/*.tiff'))
n_images = len(files)
for image_idx in range(n_images):
    image = cv2.imread(files[image_idx])
    if image_idx == 0:
        image_array_1 = np.zeros((image.shape + (n_images,)), dtype=image.dtype)
    image_array_1[..., image_idx] = image

files = sorted(glob.glob('stereo_images/right/*.tiff'))
n_images = len(files)
for image_idx in range(n_images):
    image = cv2.imread(files[image_idx])
    if image_idx == 0:
        image_array_2 = np.zeros((image.shape + (n_images,)), dtype=image.dtype)
    image_array_2[..., image_idx] = image

#%% Perform camera and stereo calibration
# Calibrate each camera.
camera_calibrator = CameraCalibrator()
param_l = camera_calibrator.calibrate(image_array_1, 1)
param_r = camera_calibrator.calibrate(image_array_2, 1)

# Perform stereo calibration
stereo_calibrator = StereoCalibrator()
stereo_parameters = stereo_calibrator.calibrate(image_array_1, image_array_2, param_l, param_r, 1, (6, 9))
# Remove bad detections
idx_new = stereo_calibrator.plot_and_filter_reproj_error()
print(idx_new)
stereo_parameters = stereo_calibrator.calibrate_indices(idx_new)
# Save checkerboard detection to images (for debugging)
stereo_calibrator.save_checkerboard_detection_to_images(image_array_1,image_array_2, '.\stereo_imagesdetected')

#%% Show detections in matching images
for image_idx in range(n_images):
    image_1 = image_array_1[..., image_idx]
    image_2 = image_array_2[..., image_idx]
    fig, axarr = plt.subplots(1, 2, constrained_layout=True)
    if len(image.shape) == 3:
        axarr[0].imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
        axarr[1].imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    else:
        axarr[0].imshow(image_1)
        axarr[1].imshow(image_2)
    if image_idx in stereo_calibrator.indices:
        feature_index = stereo_calibrator.indices.index(image_idx)
        axarr[0].plot(stereo_calibrator.image_points_list_1[feature_index][:, 0],
                      stereo_calibrator.image_points_list_1[feature_index][:, 1],
                      '-o', color='lime')
        axarr[1].plot(stereo_calibrator.image_points_list_2[feature_index][:, 0],
                      stereo_calibrator.image_points_list_2[feature_index][:, 1],
                      '-o', color='lime')
        color = 'g'
    else:
        axarr[0].plot(stereo_calibrator.feature_list_1[image_idx].image_points[:, 0],
                      stereo_calibrator.feature_list_1[image_idx].image_points[:, 1],
                      '-o', color='red')
        axarr[1].plot(stereo_calibrator.feature_list_2[image_idx].image_points[:, 0],
                      stereo_calibrator.feature_list_2[image_idx].image_points[:, 1],
                      '-o', color='red')
        color = 'r'
    fig.suptitle('Image nr ' + str(image_idx+1), color=color)
    axarr[0].get_xaxis().set_visible(False)
    axarr[0].get_yaxis().set_visible(False)
    axarr[1].get_xaxis().set_visible(False)
    axarr[1].get_yaxis().set_visible(False)
    plt.show()

#%% Plot reprojection errors
#stereo_calibrator.plot_reproj_error()

#%% Retry calibration after removing potential outliers
#stereo_calibrator.calibrate_indices([1, 2, 3, 4, 5, 6, 7, 8])
#stereo_calibrator.plot_reproj_error()


#%% Undistort and rectify images
# Calculate rectification transform and maps, only needs to be executed once.
stereo_parameters.calculate_undistort_rectify_maps()

# Undistort and rectify image pair.
image_index = 8
image_1 = image_array_1[..., image_index]
image_2 = image_array_2[..., image_index]
corr_1, corr_2 = stereo_parameters.remap_images(image_1, image_2)

# Plot results
n_divisions = 25
fig, axarr = plt.subplots(2, 2, constrained_layout=True)
if len(image.shape) == 3:
    axarr[0, 0].imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    axarr[0, 1].imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    axarr[1, 0].imshow(cv2.cvtColor(corr_1, cv2.COLOR_BGR2RGB))
    axarr[1, 1].imshow(cv2.cvtColor(corr_2, cv2.COLOR_BGR2RGB))
else:
    axarr[0, 0].imshow(image_1)
    axarr[0, 1].imshow(image_2)
    axarr[1, 0].imshow(corr_1)
    axarr[1, 1].imshow(corr_2)

width_o = image_1.shape[1]
height_o = image_1.shape[0]
steps_o = height_o / n_divisions + 1
width_c = corr_1.shape[1]
height_c = corr_1.shape[0]
steps_c = height_c / n_divisions + 1
for patch_idx in range(1, n_divisions):
    x1_o = 0
    y1_o = patch_idx * steps_o
    x2_o = width_o
    y2_o = patch_idx * steps_o
    x1_c = 0
    y1_c = patch_idx * steps_c
    x2_c = width_c
    y2_c = patch_idx * steps_c
    con_o = ConnectionPatch(xyA=(x1_o, y1_o), coordsA=axarr[0, 0].transData, xyB=(x2_o, y2_o),
                            coordsB=axarr[0, 1].transData, color='red')
    con_c = ConnectionPatch(xyA=(x1_c, y1_c), coordsA=axarr[1, 0].transData, xyB=(x2_c, y2_c),
                            coordsB=axarr[1, 1].transData, color='red')
    fig.add_artist(con_o)
    fig.add_artist(con_c)

rect_1 = Rectangle((stereo_parameters.roi_1[0], stereo_parameters.roi_1[1]),
                   stereo_parameters.roi_1[2], stereo_parameters.roi_1[3],
                   linewidth=2, edgecolor='lime', facecolor='none')
axarr[1, 0].add_patch(rect_1)
rect_2 = Rectangle((stereo_parameters.roi_2[0], stereo_parameters.roi_2[1]),
                   stereo_parameters.roi_2[2], stereo_parameters.roi_2[3],
                   linewidth=2, edgecolor='lime', facecolor='none')
axarr[1, 1].add_patch(rect_2)

axarr[0, 0].get_xaxis().set_visible(False)
axarr[0, 0].get_yaxis().set_visible(False)
axarr[0, 1].get_xaxis().set_visible(False)
axarr[0, 1].get_yaxis().set_visible(False)
axarr[1, 0].get_xaxis().set_visible(False)
axarr[1, 0].get_yaxis().set_visible(False)
axarr[1, 1].get_xaxis().set_visible(False)
axarr[1, 1].get_yaxis().set_visible(False)
plt.show()

#%% Save parameters
stereo_parameters.save_parameters("./calib/example_stereo_parameters.h5")
