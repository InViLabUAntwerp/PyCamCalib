import os
import os
# gunicorn --chdir PyCamCalib/camera_calibration_flask/ -w 4 -b 0.0.0.0:8000 app:app

os.environ["MPLBACKEND"] = "Agg"
import cv2
import base64
import numpy as np
import io
import matplotlib
matplotlib.use('Agg') # Force matplotlib to run in background without windows
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Import your exact InViLab calibration logic
from PyCamCalib.core.calibration import CameraCalibrator, CameraParameters
from PyCamCalib.core.exceptions import ImageError, CalibrationError

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Global state to mimic the PySide6 MainWindow state.
class AppState:
    def __init__(self):
        self.calibrator = CameraCalibrator()
        self.calibration_parameters = CameraParameters()
        self.image_names = []
        self.image_array = None
        self.absolute = False
        self.pixel_size_um = None
        self.info = None
        self.board_size = None


state = AppState()


def calculate_distortion_map(m, d, sensor_size):
    """Replicates the math from DistortionPlotWidget.plot_distortion to generate an image with arrows."""
    try:
        width = int(sensor_size[0])
        height = int(sensor_size[1])
        n_steps = 20
        u, v = np.meshgrid(np.linspace(0, width - 1, n_steps), np.linspace(0, height - 1, n_steps))

        # Using order='F' to match your original PySide6 logic
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

        # Create Matplotlib Figure (Exactly like your original PySide6 app)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.quiver(np.ravel(u, order='F') + 1, np.ravel(v, order='F') + 1, du, dv, color='b')
        ax.plot(width / 2, height / 2, 'x', label='Sensor center')
        ax.plot(m[0, 2], m[1, 2], 'o', label='Principal point')
        contour_set = ax.contour(u[0, :] + 1, v[:, 0] + 1, dr, colors='k')
        ax.clabel(contour_set, inline=1, fontsize=10)
        ax.set_xlim(1, width)
        ax.set_ylim(1, height)
        ax.set_aspect('equal')
        ax.set_title('Radial distortion model')
        ax.set_xlabel('Horizontal')
        ax.set_ylabel('Vertical')
        ax.legend(loc='upper right')

        # Save to buffer and encode to base64 image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error calculating distortion map: {e}")
        return None


def format_calibration_response(params: CameraParameters, calibrator: CameraCalibrator):
    """Helper method to extract data for JSON serialization."""

    # Grab the matrices for the distortion plot calculation
    m = params.get_intrinsics_matrix_opencv()
    d = params.get_distortion_coeffs_opencv()

    focal_length_mm = None
    if params.pixel_size is not None:
        focal_length_mm = [float(params.focal_length_mm[0]), float(params.focal_length_mm[1])]
        
    fov_deg = [float(params.PerspectiveAngle[0]), float(params.PerspectiveAngle[1])]

    extrinsics_data = []
    if hasattr(calibrator, 'r_vecs') and hasattr(calibrator, 't_vecs'):
        r_vecs = np.atleast_2d(calibrator.r_vecs)
        t_vecs = np.atleast_2d(calibrator.t_vecs)
        if len(r_vecs) > 0 and len(r_vecs) == len(calibrator.indices):
            for idx, r, t, obj_pts in zip(calibrator.indices, r_vecs, t_vecs, calibrator.object_points_list):
                try:
                    R_mat, _ = cv2.Rodrigues(r)
                    transformed_pts = (R_mat @ obj_pts.T).T + t.reshape(3)
                    extrinsics_data.append({
                        "index": int(idx),
                        "points": transformed_pts.tolist()
                    })
                except Exception as e:
                    print(f"Error transforming points for index {idx}: {e}")

    return {
        "status": "success",
        "intrinsics": {
            "fx": [float(params.f[0]), float(params.f_std[0])],
            "fy": [float(params.f[1]), float(params.f_std[1])],
            "cx": [float(params.c[0]), float(params.c_std[0])],
            "cy": [float(params.c[1]), float(params.c_std[1])],
            "s": [float(params.s), float(params.s_std)],
            "focal_length_mm": focal_length_mm,
            "fov_deg": fov_deg
        },
        "distortion": {
            "k1": [float(params.radial_dist_coeffs[0]), float(params.radial_dist_coeffs_std[0])],
            "k2": [float(params.radial_dist_coeffs[1]), float(params.radial_dist_coeffs_std[1])],
            "k3": [float(params.radial_dist_coeffs[2]), float(params.radial_dist_coeffs_std[2])],
            "p1": [float(params.tangential_dist_coeffs[0]), float(params.tangential_dist_coeffs_std[0])],
            "p2": [float(params.tangential_dist_coeffs[1]), float(params.tangential_dist_coeffs_std[1])]
        },
        "plot_data": {
            "indices": [int(x) for x in calibrator.indices],
            "errors": [float(e) for e in calibrator.per_view_err],
            "rms": float(calibrator.rms_reproj_error),
            "distortion_map": calculate_distortion_map(m, d, params.sensor_dimensions),
            "extrinsics": extrinsics_data
        },
        "total_images": state.image_array.shape[-1] if state.image_array is not None else 0,
        "sensor_dimensions": [int(params.sensor_dimensions[0]), int(params.sensor_dimensions[1])],
        "pixel_size": params.pixel_size,
        "info": params.info,
        "board_size": state.board_size if hasattr(state, 'board_size') else None,
        "warning": "At least 11 good images are required for an accurate calibration." if len(
            calibrator.indices) < 11 else None
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/calibrate/initial', methods=['POST'])
def calibrate_initial():
    if 'files' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    # Complete reset of state for a clean new calibration run
    state.calibrator = CameraCalibrator()
    state.calibration_parameters = CameraParameters()
    state.image_names = []
    state.image_array = None

    files = request.files.getlist('files')
    checker_size = float(request.form.get('checker_size', 1.0))
    state.absolute = request.form.get('absolute') == 'true'
    pixel_size_str = request.form.get('pixel_size')
    if pixel_size_str:
        try:
            state.pixel_size_um = float(pixel_size_str)
        except ValueError:
            state.pixel_size_um = None

    info_str = request.form.get('info')
    state.info = info_str if info_str else None

    board_width = request.form.get('board_width')
    board_height = request.form.get('board_height')
    board_size = None
    if board_width and board_height:
        try:
            board_size = (int(board_width), int(board_height))
        except ValueError:
            pass

    marker_location = None
    expand = False
    predict = False

    valid_files = [f for f in files if f.filename != '']
    if not valid_files:
        return jsonify({"error": "No valid images provided."}), 400

    n_images = len(valid_files)
    for idx, file in enumerate(valid_files):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        if image is None:
            return jsonify({"error": f"Failed to load image: {filename}"}), 400

        if state.image_array is None:
            state.image_array = np.zeros((image.shape + (n_images,)), dtype=image.dtype)
        elif image.shape != state.image_array.shape[:-1]:
            return jsonify({"error": f"Image dimensions mismatch for {filename}. All images must be exactly the same size."}), 400
            
        state.image_array[..., idx] = image
        state.image_names.append(filename)

    try:
        sensor_dimensions = np.array([state.image_array.shape[1], state.image_array.shape[0]])
        
        # Auto-detect board size if not explicitly provided
        detected_board_size = board_size
        if detected_board_size is None:
            for i in range(state.image_array.shape[-1]):
                temp_calibrator = CameraCalibrator()
                temp_calibrator.construct_feature_list(
                    state.image_array[..., i:i+1], checker_size, None, None, expand=expand, predict=predict
                )
                if temp_calibrator.feature_list and temp_calibrator.feature_list[0].score > 0:
                    obj_pts = temp_calibrator.feature_list[0].object_points
                    if obj_pts is not None and len(obj_pts) > 0:
                        xs = np.unique(np.round(obj_pts[:, 0], decimals=3))
                        ys = np.unique(np.round(obj_pts[:, 1], decimals=3))
                        detected_board_size = (len(xs), len(ys))
                        break
        
        state.board_size = detected_board_size

        state.calibrator.construct_feature_list(
            state.image_array, checker_size, detected_board_size, marker_location, expand=expand, predict=predict
        )

        # If the user explicitly requested absolute positions, the feature detector might still
        # return a score of 1 because marked checkerboards (which guarantee absolute orientation)
        # are not implemented. We manually upgrade the score if the full expected board was found.
        if state.absolute and detected_board_size is not None:
            expected_points = detected_board_size[0] * detected_board_size[1]
            for feature in state.calibrator.feature_list:
                if feature.score > 0 and feature.object_points is not None:
                    if len(feature.object_points) == expected_points:
                        feature.score = 2

        state.calibrator.construct_points_lists([], state.absolute)
        if not state.calibrator.image_points_list:
            return jsonify({"error": "Failed to detect specified feature in every image."}), 400

        state.calibration_parameters = state.calibrator.opencv_calibration(sensor_dimensions)
        if state.pixel_size_um is not None:
            state.calibration_parameters.pixel_size = state.pixel_size_um / 1000.0
        if state.info is not None:
            state.calibration_parameters.info = state.info
        return jsonify(format_calibration_response(state.calibration_parameters, state.calibrator))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/calibrate/recalculate', methods=['POST'])
def calibrate_recalculate():
    data = request.json
    excluded_indices = set(data.get('excluded_indices', []))

    info_str = data.get('info')
    state.info = info_str if info_str else None

    total_original_images = state.image_array.shape[-1] if state.image_array is not None else 0
    good_indices = [i for i in range(total_original_images) if i not in excluded_indices]

    if not good_indices:
        return jsonify({"error": "You can't remove all images."}), 400

    try:
        state.calibration_parameters = state.calibrator.calibrate_indices(good_indices, state.absolute)
        if state.pixel_size_um is not None:
            state.calibration_parameters.pixel_size = state.pixel_size_um / 1000.0
        if state.info is not None:
            state.calibration_parameters.info = state.info
        return jsonify(format_calibration_response(state.calibration_parameters, state.calibrator))
    except Exception as e:
        return jsonify({"error": f"Calibration error: {str(e)}"}), 500


@app.route('/api/detections/<int:index>', methods=['GET'])
def get_detection_image(index):
    """Returns the image at the given index with detection points drawn on it."""
    if state.image_array is None or index < 0 or index >= state.image_array.shape[-1]:
        return jsonify({"error": "Invalid image index"}), 400

    # Get a copy of the image to draw on
    img = state.image_array[..., index].copy()

    # Check if feature list exists for this image
    if index < len(state.calibrator.feature_list):
        feature = state.calibrator.feature_list[index]
        if feature.score > 0:
            # Replicating your PySide6 Color Logic
            min_score = 2 if state.absolute else 1
            if feature.score >= min_score:
                color = (0, 255, 0) if index in state.calibrator.indices else (0, 0,
                                                                               255)  # Green if used, Red if excluded
            else:
                color = (255, 0, 0)  # Blue for poor score

            # Draw the points
            for pt in feature.image_points:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(img, (x, y), 5, color, -1)

        shape_str = "N/A"
        if feature.score > 0 and feature.object_points is not None and len(feature.object_points) > 0:
            # Calculate size dynamically based on 3D Object Space points assigned
            xs = np.unique(np.round(feature.object_points[:, 0], decimals=3))
            ys = np.unique(np.round(feature.object_points[:, 1], decimals=3))
            shape_str = f"{len(xs)}x{len(ys)}"

    # Encode image to Base64 to send to HTML
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Find the error if it exists for this image
    try:
        err_idx = state.calibrator.indices.index(index)
        reproj_err = round(float(state.calibrator.per_view_err[err_idx]), 4)
    except ValueError:
        reproj_err = "N/A (Excluded)"

    return jsonify({
        "image_data": f"data:image/jpeg;base64,{img_base64}",
        "name": state.image_names[index],
        "error": reproj_err,
        "shape": shape_str
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=7000)