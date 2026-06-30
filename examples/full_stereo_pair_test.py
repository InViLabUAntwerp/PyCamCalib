"""Example script for stereo calibration using the extrinsic folder structure.

Loads L_pattern, R_pattern, and RGB images (black + white variants) across
scan_1..scan_50 and runs stereo calibration for all three camera pairs:
  - Left   <-> Right
  - Left   <-> RGB
  - Right  <-> RGB
"""

#%% Imports
from src.PyCamCalib.core.calibration import CameraCalibrator, StereoCalibrator
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import time
import os
from glob import glob

# ── Configuration ─────────────────────────────────────────────────────────────
EXTRINSIC_ROOT = "extrinsic"
N_SCANS        = 50
VARIANTS       = ["black"]            # both illumination variants
CHECKERBOARD   = (8, 11)                       # inner corners (cols, rows)
SQUARE_SIZE    = 8.0                           # passed to calibrators

# Map camera label → subfolder prefix
CAM_FOLDERS = {
    "L":   "L_pattern",
    "R":   "R_pattern",
    "RGB": "RGB",
}

# The three stereo pairs to calibrate
STEREO_PAIRS = [
    ("L",   "R"),
    ("L",   "RGB"),
    ("R",   "RGB"),
]


def load_camera_images(cam_key):
    image_list = []

    prefix = CAM_FOLDERS[cam_key]

    for scan_idx in range(1, N_SCANS + 1):

        scan_folder = os.path.join(EXTRINSIC_ROOT, f"scan_{scan_idx}")

        for variant in VARIANTS:

            pattern = os.path.join(scan_folder, f"{prefix}_{variant}.*")
            matches = sorted(glob(pattern))
            if len(matches) == 0:
                print(f"Missing {pattern}")
                continue

            img = cv2.imread(matches[0])

            if img is None:
                print(f"Could not read {matches[0]}")
                continue

            image_list.append(img)

    return np.stack(image_list, axis=-1)


# ── Helper: run one stereo calibration pair ────────────────────────────────────
def run_stereo_pair(
    label_a: str,
    label_b: str,
    images_a: np.ndarray,
    images_b: np.ndarray,
    output_root: str = "./calib",
) -> None:
    pair_name = f"{label_a}_vs_{label_b}"
    print(f"\n{'='*60}")
    print(f"  Stereo calibration: {label_a}  <->  {label_b}")
    print(f"{'='*60}")

    t0 = time.time()

    # ── Individual camera calibration ──────────────────────────────────────────
    camera_calibrator = CameraCalibrator()
    print(f"  Calibrating camera {label_a} …")
    param_a = camera_calibrator.calibrate(images_a, SQUARE_SIZE)
    print(f"  Calibrating camera {label_b} …")
    param_b = camera_calibrator.calibrate(images_b, SQUARE_SIZE)

    # ── Stereo calibration ─────────────────────────────────────────────────────
    stereo_calibrator = StereoCalibrator()
    print(f"  Running stereo calibration …")
    stereo_params = stereo_calibrator.calibrate(
        images_a, images_b, param_a, param_b, SQUARE_SIZE, CHECKERBOARD
    )

    # Filter outliers by reprojection error
    idx_new = stereo_calibrator.plot_and_filter_reproj_error()
    print(f"  Retained indices after filtering: {idx_new}")
    stereo_params = stereo_calibrator.calibrate_indices(idx_new)

    # Save detection images for debugging
    det_path = os.path.join(output_root, pair_name, "detected")
    os.makedirs(det_path, exist_ok=True)
    stereo_calibrator.save_checkerboard_detection_to_images(images_a, images_b, det_path)

    print(f"  Elapsed: {time.time() - t0:.1f}s")

    # ── Visualise detections ───────────────────────────────────────────────────
    n_images = images_a.shape[-1]
    for image_idx in range(n_images):
        img_a = images_a[..., image_idx]
        img_b = images_b[..., image_idx]
        fig, axarr = plt.subplots(1, 2, constrained_layout=True)

        def _show(ax, img):
            if img.ndim == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap="gray")

        _show(axarr[0], img_a)
        _show(axarr[1], img_b)

        if image_idx in stereo_calibrator.indices:
            fi = stereo_calibrator.indices.index(image_idx)
            pts_a = stereo_calibrator.image_points_list_1[fi]
            pts_b = stereo_calibrator.image_points_list_2[fi]
            axarr[0].plot(pts_a[:, 0], pts_a[:, 1], "-o", color="lime")
            axarr[1].plot(pts_b[:, 0], pts_b[:, 1], "-o", color="lime")
            color = "g"
        else:
            pts_a = stereo_calibrator.feature_list_1[image_idx].image_points
            pts_b = stereo_calibrator.feature_list_2[image_idx].image_points
            axarr[0].plot(pts_a[:, 0], pts_a[:, 1], "-o", color="red")
            axarr[1].plot(pts_b[:, 0], pts_b[:, 1], "-o", color="red")
            color = "r"

        fig.suptitle(f"{pair_name}  –  image {image_idx + 1}", color=color)
        for ax in axarr:
            ax.axis("off")
        plt.show()

    # ── Reprojection error plot ────────────────────────────────────────────────
    stereo_calibrator.plot_reproj_error()

    # ── Rectification example (first retained image) ───────────────────────────
    stereo_params.calculate_undistort_rectify_maps()
    example_idx = idx_new[0] if idx_new else 0
    img_a = images_a[..., example_idx]
    img_b = images_b[..., example_idx]
    rect_a, rect_b = stereo_params.remap_images(img_a, img_b)

    n_div = 25
    fig, axarr = plt.subplots(2, 2, constrained_layout=True)
    pairs_plot = [(img_a, img_b), (rect_a, rect_b)]
    for row, (pa, pb) in enumerate(pairs_plot):
        for col, img in enumerate((pa, pb)):
            ax = axarr[row, col]
            if img.ndim == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap="gray")
            ax.axis("off")

    # Draw horizontal epipolar lines on the rectified row
    for patch_idx in range(1, n_div):
        h_c, w_c = rect_a.shape[:2]
        y = patch_idx * h_c / n_div
        con = ConnectionPatch(
            xyA=(0, y), coordsA=axarr[1, 0].transData,
            xyB=(w_c, y), coordsB=axarr[1, 1].transData,
            color="red",
        )
        fig.add_artist(con)

    for roi, ax in ((stereo_params.roi_1, axarr[1, 0]),
                    (stereo_params.roi_2, axarr[1, 1])):
        rect = Rectangle((roi[0], roi[1]), roi[2], roi[3],
                          linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

    axarr[0, 0].set_title(f"Original  {label_a}")
    axarr[0, 1].set_title(f"Original  {label_b}")
    axarr[1, 0].set_title(f"Rectified  {label_a}")
    axarr[1, 1].set_title(f"Rectified  {label_b}")
    fig.suptitle(f"Rectification: {pair_name}")
    plt.show()

    # ── Save parameters ────────────────────────────────────────────────────────
    os.makedirs(output_root, exist_ok=True)
    save_path = os.path.join(output_root, f"{pair_name}_stereo_parameters.h5")
    stereo_params.save_parameters(save_path)
    print(f"  Saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    overall_start = time.time()

    # Load each camera's images once, reuse across pairs
    print("Loading images …")
    camera_images = {}
    for cam_key in CAM_FOLDERS:
        camera_images[cam_key] = load_camera_images(cam_key)

    # Run all three stereo pairs
    for label_a, label_b in STEREO_PAIRS:
        run_stereo_pair(
            label_a,
            label_b,
            camera_images[label_a],
            camera_images[label_b],
        )

    print(f"\nAll pairs done.  Total elapsed: {time.time() - overall_start:.1f}s")