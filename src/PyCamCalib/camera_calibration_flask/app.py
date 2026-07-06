import os
import time
import uuid
#gunicorn --chdir PyCamCalib/camera_calibration_flask/ -w 1 --threads 4 -b 0.0.0.0:8000 app:app

os.environ["MPLBACKEND"] = "Agg"
import cv2
import base64
import numpy as np
import io
import matplotlib
matplotlib.use('Agg') # Force matplotlib to run in background without windows
import glob
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename

# Import your exact InViLab calibration logic
from PyCamCalib.core.calibration import CameraCalibrator, CameraParameters
from PyCamCalib.core.exceptions import ImageError, CalibrationError

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# State class to mimic the PySide6 MainWindow state.
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


# --- User & State Tracking via Tab ID ---
active_tabs = {}
user_states = {}  # Map Tab IDs to their own isolated AppState
MAX_USERS = 8
USER_TIMEOUT = 300  # 5 minutes of inactivity before a slot opens up


def get_tab_id():
    """Retrieves the Tab ID sent by the frontend, fallback to session if missing."""
    tab_id = request.headers.get('X-Tab-ID')
    if not tab_id:
        if 'session_id' not in session:
            session['session_id'] = uuid.uuid4().hex
        return session['session_id']
    return tab_id


def get_user_state():
    """Retrieves or creates a unique state for the current Tab ID."""
    tid = get_tab_id()
    if tid not in user_states:
        user_states[tid] = AppState()
    return user_states[tid]


@app.before_request
def track_users():
    if request.endpoint == 'static':
        return

    tid = get_tab_id()
    current_time = time.time()

    # 1. Clean up old inactive tabs and their heavy memory states
    expired_tids = [k for k, v in active_tabs.items() if current_time - v > USER_TIMEOUT]
    for k in expired_tids:
        del active_tabs[k]
        if k in user_states:
            del user_states[k]  # Free up memory!

    # 2. Register or update the current tab
    if tid not in active_tabs and len(active_tabs) >= MAX_USERS:
        # Enforce maximum user limit
        if request.endpoint not in ('index', 'get_status'):
            return jsonify({"error": "Server is at maximum capacity. Please wait for a slot."}), 429
    else:
        active_tabs[tid] = current_time


@app.route('/api/status', methods=['GET'])
def get_status():
    tid = get_tab_id()
    count = len(active_tabs)
    # If the server is full and the tab isn't already inside, they are waitlisted
    is_waitlisted = (count >= MAX_USERS and tid not in active_tabs)
    return jsonify({"count": count, "max": MAX_USERS, "waitlisted": is_waitlisted})
@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    """Instantly frees up a user slot when they close the tab."""
    tid = request.headers.get('X-Tab-ID') or get_tab_id()
    if tid in active_tabs:
        del active_tabs[tid]
    if tid in user_states:
        del user_states[tid]
    return jsonify({"status": "disconnected"}), 200
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

        # Create Matplotlib Figure
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


def format_calibration_response(params: CameraParameters, calibrator: CameraCalibrator, state: AppState):
    """Helper method to extract data for JSON serialization."""

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

@app.route('/api/info/<path:filename>')
def get_info_html(filename):
    """Serves the static HTML info files from the templates/info directory."""
    # Securely serve files from a specific directory
    return send_from_directory('templates/info', filename)



@app.route('/api/calibrate/initial', methods=['POST'])
def calibrate_initial():

    if 'files' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    # Get the unique state for this specific tab
    state = get_user_state()

    # Complete reset of state for a clean new calibration run for this tab
    state.calibrator = CameraCalibrator()
    state.calibration_parameters = CameraParameters()
    state.image_names = []
    state.image_array = None

    # --- Get calibration settings from form ---
    files = request.files.getlist('files')
    detector_type = request.form.get('detector_type', 'checkerboard')
    charuco_preset = request.form.get('charuco_preset') # Will be 'wenglor' or 'None'
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
        # --- Prepare detector parameters ---
        detector_params = {}
        if detector_type == "charuco":
            # The string 'None' from the form becomes Python's None
            detector_params["preset"] = None if charuco_preset == 'None' else charuco_preset

        # --- Perform Calibration ---
        # The `calibrate` method handles feature detection and calibration internally
        state.calibration_parameters = state.calibrator.calibrate(
            image_array=state.image_array,
            space_between_features=checker_size,
            board_size=board_size,
            detector_type=detector_type,
            detector_params=detector_params,
            absolute=state.absolute
        )

        # --- Post-calibration processing ---
        # Try to determine board size for display if it was auto-detected
        if board_size is None and state.calibrator.feature_list:
            for feature in state.calibrator.feature_list:
                if feature.score > 0 and feature.object_points is not None and len(feature.object_points) > 0:
                    # This logic works for both checkerboard and charuco object points
                    xs = np.unique(np.round(feature.object_points[:, 0] / checker_size))
                    ys = np.unique(np.round(feature.object_points[:, 1] / checker_size))
                    state.board_size = (len(xs), len(ys))
                    break
        else:
            state.board_size = board_size

        # Set optional metadata
        if state.pixel_size_um is not None:
            state.calibration_parameters.pixel_size = state.pixel_size_um / 1000.0
        if state.info is not None:
            state.calibration_parameters.info = state.info

        return jsonify(format_calibration_response(state.calibration_parameters, state.calibrator, state))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/load-example', methods=['POST'])
def load_example_data():
    """Loads the example checkerboard images and runs an initial calibration."""
    state = get_user_state()

    # Reset state for a clean run
    state.calibrator = CameraCalibrator()
    state.calibration_parameters = CameraParameters()
    state.image_names = []
    state.image_array = None

    try:
        # --- Get settings from the request body ---
        settings = request.json
        checker_size = float(settings.get('checker_size', 14.0)) # Default to 14mm for the example
        state.absolute = settings.get('absolute', False)
        pixel_size_str = settings.get('pixel_size')
        state.pixel_size_um = float(pixel_size_str) if pixel_size_str else None
        state.info = settings.get('info') or "Example Dataset"

        board_width = settings.get('board_width')
        board_height = settings.get('board_height')
        board_size = (int(board_width), int(board_height)) if board_width and board_height else None

        # --- Load example images from static folder ---
        example_dir = os.path.join(app.static_folder, 'examples', 'checkerboard')
        if not os.path.isdir(example_dir):
            return jsonify({"error": "Example image directory not found on server."}), 404

        # Using glob to find all common image types
        image_paths = sorted(glob.glob(os.path.join(example_dir, '*.png')))
        if not image_paths:
            return jsonify({"error": "No example images found in the directory."}), 404

        n_images = len(image_paths)
        for idx, path in enumerate(image_paths):
            image = cv2.imread(path)
            if image is None:
                continue # Skip if an image fails to load

            if state.image_array is None:
                state.image_array = np.zeros((image.shape + (n_images,)), dtype=image.dtype)

            state.image_array[..., idx] = image
            state.image_names.append(os.path.basename(path))

        # --- Perform Calibration ---
        state.calibration_parameters = state.calibrator.calibrate(
            image_array=state.image_array,
            space_between_features=checker_size,
            board_size=board_size,
            detector_type="checkerboard", # Example data is checkerboard
            detector_params={},
            absolute=state.absolute
        )

        # --- Post-calibration processing ---
        if board_size is None and state.calibrator.feature_list:
            for feature in state.calibrator.feature_list:
                if feature.score > 0 and feature.object_points is not None and len(feature.object_points) > 0:
                    xs = np.unique(np.round(feature.object_points[:, 0] / checker_size))
                    ys = np.unique(np.round(feature.object_points[:, 1] / checker_size))
                    state.board_size = (len(xs), len(ys))
                    break
        else:
            state.board_size = board_size

        if state.pixel_size_um is not None:
            state.calibration_parameters.pixel_size = state.pixel_size_um / 1000.0
        if state.info is not None:
            state.calibration_parameters.info = state.info

        return jsonify(format_calibration_response(state.calibration_parameters, state.calibrator, state))

    except Exception as e:
        return jsonify({"error": f"An error occurred while processing example data: {str(e)}"}), 500

@app.route('/api/calibrate/recalculate', methods=['POST'])
def calibrate_recalculate():
    # Retrieve the unique state for this specific tab
    state = get_user_state()

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
        return jsonify(format_calibration_response(state.calibration_parameters, state.calibrator, state))
    except Exception as e:
        return jsonify({"error": f"Calibration error: {str(e)}"}), 500


@app.route('/api/detections/<int:index>', methods=['GET'])
def get_detection_image(index):
    """Returns the image at the given index with detection points drawn on it."""
    # Retrieve the unique state for this specific tab
    state = get_user_state()

    if state.image_array is None or index < 0 or index >= state.image_array.shape[-1]:
        return jsonify({"error": "Invalid image index"}), 400

    img = state.image_array[..., index].copy()

    if index < len(state.calibrator.feature_list):
        feature = state.calibrator.feature_list[index]
        if feature.score > 0:
            min_score = 2 if state.absolute else 1
            if feature.score >= min_score:
                color = (0, 255, 0) if index in state.calibrator.indices else (0, 0, 255)
            else:
                color = (255, 0, 0)

            for pt in feature.image_points:
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(img, (x, y), 5, color, -1)

        shape_str = "N/A"
        if feature.score > 0 and feature.object_points is not None and len(feature.object_points) > 0:
            xs = np.unique(np.round(feature.object_points[:, 0], decimals=3))
            ys = np.unique(np.round(feature.object_points[:, 1], decimals=3))
            shape_str = f"{len(xs)}x{len(ys)}"

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

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