# %% Imports
from src.PyCamCalib.core.calibration import CameraCalibrator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
# Everything below must be guarded on Windows
if __name__ == '__main__':
    # %% Configuration
    EXTRINSIC_ROOT = "intrinsic"
    N_SCANS = 50
    CAMERA = "L_scan"
    # %% Load images
    image_list = []
    for scan_idx in range(N_SCANS):
        filename = os.path.join(
            EXTRINSIC_ROOT,
            CAMERA,
            f"{scan_idx}.png",
        )
        image = cv2.imread(filename)
        if image is None:
            print(f"Warning: could not read {filename}")
            continue
        image_list.append(image)
    if len(image_list) == 0:
        raise RuntimeError("No images found.")
    image_array = np.stack(image_list, axis=-1)
    print(f"Loaded {image_array.shape[-1]} images")
    print(f"Array shape: {image_array.shape}")
    # %% Perform calibration
    calibrator = CameraCalibrator()
    camera_parameters = calibrator.calibrate(image_array, 8.0,(8,11))
    # %% Plot detections
    for image_idx in range(image_array.shape[-1]):
        image = image_array[..., image_idx]
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if image_idx in calibrator.indices:
            feature_index = calibrator.indices.index(image_idx)
            pts = calibrator.image_points_list[feature_index]
            plt.plot(pts[:, 0], pts[:, 1], "-o", color="lime")
            color = "green"
        elif calibrator.feature_list[image_idx].score != 0:
            pts = calibrator.feature_list[image_idx].image_points
            plt.plot(pts[:, 0], pts[:, 1], "-o", color="red")
            color = "red"
        else:
            color = "red"
        plt.title(f"Image {image_idx + 1}", color=color)
        plt.axis("off")
        plt.show()
    # %% Plot reprojection error
    calibrator.plot_reproj_error()
    # %% Remove outliers and recalibrate
    new_indices = calibrator.plot_and_filter_reproj_error()
    camera_parameters = calibrator.calibrate_indices(new_indices)
    calibrator.plot_reproj_error()
    # %% Visualize distortion
    camera_parameters.plot_distortion()
    # %% Undistort an example image
    camera_parameters.calculate_undistort_map()
    image = image_array[..., 0]
    undistorted = camera_parameters.remap_image(image)
    fig, axarr = plt.subplots(1, 2, constrained_layout=True)
    axarr[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axarr[0].set_title("Original")
    axarr[1].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    axarr[1].set_title("Undistorted")
    rect = Rectangle(
        (camera_parameters.roi[0], camera_parameters.roi[1]),
        camera_parameters.roi[2],
        camera_parameters.roi[3],
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    axarr[1].add_patch(rect)
    for ax in axarr:
        ax.axis("off")
    plt.show()
    # %% Save parameters
    camera_parameters.save_parameters(f"{CAMERA}_intrinsic_parameters.h5")
    print("Intrinsic calibration complete.")