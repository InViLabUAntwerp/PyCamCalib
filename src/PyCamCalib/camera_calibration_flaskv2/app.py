import os
import time
import uuid
import base64
import glob
import io
import json

os.environ["MPLBACKEND"] = "Agg"
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Import InViLab calibration logic
from PyCamCalib.core.calibration import CameraCalibrator, CameraParameters

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# --- CONFIGURATION ---
MAX_USERS = 8
USER_TIMEOUT = 300  # 5 minutes
GLOBAL_MEMORY_LIMIT_MB = 1024  # 1GB
MAX_UPLOAD_SIZE_MB = 50
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE_MB * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
LOG_DIR = os.environ.get('INVILAB_DATA_DIR', 'data')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
COUNT_FILE_PATH = os.path.join(LOG_DIR, 'calibration_count.log')

@app.errorhandler(RequestEntityTooLarge)
def handle_file_size_error(e):
    return jsonify({"error": f"Upload rejected: Total file size exceeds the {MAX_UPLOAD_SIZE_MB}MB server limit."}), 413

# --- STATE TRACKING (SERVER MODE ONLY) ---
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

active_tabs = {}
user_states = {} 

def get_tab_id():
    return request.headers.get('X-Tab-ID') or request.args.get('tid')

def get_user_state():
    tid = get_tab_id()
    if not tid: return AppState()
    if tid not in user_states: user_states[tid] = AppState()
    return user_states[tid]

@app.before_request
def track_users():
    if request.endpoint in ('static', 'index', 'app_view', 'get_info_html', 'get_example_files') or request.path == '/favicon.ico':
        return
    tid = get_tab_id()
    if not tid: return
    current_time = time.time()
    
    # Cleanup old users
    expired_tids = [k for k, v in active_tabs.items() if current_time - v > USER_TIMEOUT]
    for k in expired_tids:
        del active_tabs[k]
        if k in user_states: del user_states[k]
            
    if tid not in active_tabs and len(active_tabs) >= MAX_USERS:
        if request.endpoint != 'get_status':
            return jsonify({"error": "Server is at maximum capacity. Please wait for a slot."}), 429
    else:
        active_tabs[tid] = current_time

# --- PERSISTENT COUNTER ---
def get_calibration_count():
    try:
        with open(COUNT_FILE_PATH, 'r') as f: return int(f.read().strip())
    except (IOError, ValueError): return 0

def increment_calibration_count():
    count = get_calibration_count() + 1
    with open(COUNT_FILE_PATH, 'w') as f: f.write(str(count))
    return count

@app.route('/api/increment-counter', methods=['POST'])
def api_increment_counter():
    count = increment_calibration_count()
    return jsonify({"success": True, "count": count})

# --- VIEWS ---
@app.route('/')
def index():
    return render_template('index.html', calibration_count=get_calibration_count())

@app.route('/app/<mode>')
def app_view(mode):
    if mode not in ['local', 'server']:
        return "Invalid mode", 400
    return render_template('calibration.html', 
                           compute_mode=mode, 
                           calibration_count=get_calibration_count(),
                           max_upload_mb=MAX_UPLOAD_SIZE_MB)

@app.route('/api/info/<path:filename>')
def get_info_html(filename):
    return send_from_directory('templates/info', filename)

@app.route('/api/status', methods=['GET'])
def get_status():
    tid = get_tab_id()
    count = len(active_tabs)
    is_waitlisted = (count >= MAX_USERS and tid not in active_tabs) if tid else (count >= MAX_USERS)
    return jsonify({"count": count, "max": MAX_USERS, "waitlisted": is_waitlisted})

@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    tid = get_tab_id()
    if tid in active_tabs: del active_tabs[tid]
    if tid in user_states: del user_states[tid]
    return jsonify({"status": "disconnected"}), 200

# --- HELPER FUNCTIONS ---
def get_total_used_memory_mb(exclude_tid=None):
    total_bytes = 0
    for tid, state in user_states.items():
        if exclude_tid and tid == exclude_tid: continue
        if state.image_array is not None: total_bytes += state.image_array.nbytes
    return total_bytes / (1024 * 1024)

def calculate_distortion_map(m, d, sensor_size):
    try:
        width, height = int(sensor_size[0]), int(sensor_size[1])
        u, v = np.meshgrid(np.linspace(0, width - 1, 20), np.linspace(0, height - 1, 20))
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

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.quiver(np.ravel(u, order='F') + 1, np.ravel(v, order='F') + 1, du, dv, color='b')
        ax.plot(width / 2, height / 2, 'x', label='Sensor center')
        ax.plot(m[0, 2], m[1, 2], 'o', label='Principal point')
        contour_set = ax.contour(u[0, :] + 1, v[:, 0] + 1, dr, colors='k')
        ax.clabel(contour_set, inline=1, fontsize=10)
        ax.set_xlim(1, width)
        ax.set_ylim(1, height)
        ax.set_aspect('equal')
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
    m = params.get_intrinsics_matrix_opencv()
    d = params.get_distortion_coeffs_opencv()
    focal_length_mm = [float(params.focal_length_mm[0]), float(params.focal_length_mm[1])] if params.pixel_size else None
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
                    extrinsics_data.append({"index": int(idx), "points": transformed_pts.tolist()})
                except Exception: pass

    return {
        "status": "success",
        "intrinsics": {
            "fx": [float(params.f[0]), float(params.f_std[0])], "fy": [float(params.f[1]), float(params.f_std[1])],
            "cx": [float(params.c[0]), float(params.c_std[0])], "cy": [float(params.c[1]), float(params.c_std[1])],
            "s": [float(params.s), float(params.s_std)],
            "focal_length_mm": focal_length_mm, "fov_deg": fov_deg
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
        "pixel_size": params.pixel_size, "info": params.info,
        "board_size": state.board_size if hasattr(state, 'board_size') else None,
        "warning": "At least 11 good images are required." if len(calibrator.indices) < 11 else None
    }

# --- SERVER COMPUTE ENDPOINTS ---
@app.route('/api/calibrate/initial', methods=['POST'])
def calibrate_initial():
    increment_calibration_count()
    if 'files' not in request.files: return jsonify({"error": "No images provided"}), 400
    
    state = get_user_state()
    state.__init__() # Reset

    files = request.files.getlist('files')
    detector_type = request.form.get('detector_type', 'checkerboard')
    checker_size = float(request.form.get('checker_size', 1.0))
    state.absolute = request.form.get('absolute') == 'true'
    
    px_str, info_str = request.form.get('pixel_size'), request.form.get('info')
    state.pixel_size_um = float(px_str) if px_str else None
    state.info = info_str if info_str else None

    bw, bh = request.form.get('board_width'), request.form.get('board_height')
    board_size = (int(bw), int(bh)) if bw and bh else None

    valid_files = [f for f in files if f.filename != '']
    if not valid_files: return jsonify({"error": "No valid images provided."}), 400

    n_images = len(valid_files)
    for idx, file in enumerate(valid_files):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image = cv2.imread(filepath)
        if os.path.exists(filepath):
            os.remove(filepath)
        if image is None: return jsonify({"error": f"Failed to load image: {filename}"}), 400

        if state.image_array is None:
            required_mb = (image.nbytes * n_images) / (1024 * 1024)
            if required_mb > GLOBAL_MEMORY_LIMIT_MB:
                return jsonify({"error": "Dataset Too Large", "details": f"Requires {required_mb:.0f}MB. Max is {GLOBAL_MEMORY_LIMIT_MB}MB."}), 413
            current_used_mb = get_total_used_memory_mb(exclude_tid=get_tab_id())
            if (current_used_mb + required_mb) > GLOBAL_MEMORY_LIMIT_MB:
                return jsonify({"error": "Server Memory Full", "details": "Server is processing other users. Please wait."}), 503
            state.image_array = np.zeros((image.shape + (n_images,)), dtype=image.dtype)
        
        state.image_array[..., idx] = image
        state.image_names.append(filename)

    try:
        detector_params = {}
        if detector_type == "charuco":
            preset = request.form.get('charuco_preset')
            detector_params["preset"] = None if preset == 'None' else preset

        state.calibration_parameters = state.calibrator.calibrate(
            image_array=state.image_array, space_between_features=checker_size, board_size=board_size,
            detector_type=detector_type, detector_params=detector_params, absolute=state.absolute
        )

        if board_size is None and state.calibrator.feature_list:
            for feature in state.calibrator.feature_list:
                if feature.score > 0 and feature.object_points is not None and len(feature.object_points) > 0:
                    xs = np.unique(np.round(feature.object_points[:, 0] / checker_size))
                    ys = np.unique(np.round(feature.object_points[:, 1] / checker_size))
                    state.board_size = (len(xs), len(ys))
                    break
        else:
            state.board_size = board_size

        if state.pixel_size_um is not None: state.calibration_parameters.pixel_size = state.pixel_size_um / 1000.0
        if state.info is not None: state.calibration_parameters.info = state.info

        return jsonify(format_calibration_response(state.calibration_parameters, state.calibrator, state))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/calibrate/recalculate', methods=['POST'])
def calibrate_recalculate():
    state = get_user_state()
    data = request.json
    excluded_indices = set(data.get('excluded_indices', []))
    if data.get('info'): state.info = data.get('info')

    total = state.image_array.shape[-1] if state.image_array is not None else 0
    good_indices = [i for i in range(total) if i not in excluded_indices]
    if not good_indices: return jsonify({"error": "You can't remove all images."}), 400

    try:
        state.calibration_parameters = state.calibrator.calibrate_indices(good_indices, state.absolute)
        if state.pixel_size_um is not None: state.calibration_parameters.pixel_size = state.pixel_size_um / 1000.0
        if state.info is not None: state.calibration_parameters.info = state.info
        return jsonify(format_calibration_response(state.calibration_parameters, state.calibrator, state))
    except Exception as e:
        return jsonify({"error": f"Calibration error: {str(e)}"}), 500

@app.route('/api/detections/<int:index>', methods=['GET'])
def get_detection_image(index):
    state = get_user_state()
    if state.image_array is None or index < 0 or index >= state.image_array.shape[-1]:
        return jsonify({"error": "Invalid index"}), 400

    img = state.image_array[..., index].copy()
    shape_str, reproj_err = "N/A", "N/A (Excluded)"
    
    if index < len(state.calibrator.feature_list):
        feature = state.calibrator.feature_list[index]
        if feature.score > 0:
            min_score = 2 if state.absolute else 1
            color = (0, 255, 0) if (feature.score >= min_score and index in state.calibrator.indices) else ((0, 0, 255) if feature.score >= min_score else (255,0,0))
            for pt in feature.image_points:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 5, color, -1)
        if feature.score > 0 and feature.object_points is not None and len(feature.object_points) > 0:
            xs, ys = np.unique(np.round(feature.object_points[:, 0], decimals=3)), np.unique(np.round(feature.object_points[:, 1], decimals=3))
            shape_str = f"{len(xs)}x{len(ys)}"

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    try:
        err_idx = state.calibrator.indices.index(index)
        reproj_err = round(float(state.calibrator.per_view_err[err_idx]), 4)
    except ValueError: pass

    return jsonify({"image_data": f"data:image/jpeg;base64,{img_base64}", "name": state.image_names[index], "reproj_error": reproj_err, "shape": shape_str})

# --- LOCAL/SHARED DATA ENDPOINTS ---
@app.route('/api/example-files', methods=['GET', 'POST'])
def get_example_files():
    # Supports GET (for local Pyodide fetch) and POST (for Server-side load triggering)
    increment_calibration_count()
    example_dir = os.path.join(app.static_folder, 'examples', 'checkerboard')
    if not os.path.exists(example_dir):
        return jsonify({"error": "Folder not found"}), 404

    # --- SMART LOADER: Prevent duplicates if both .jpg and .png exist ---
    all_files = sorted(os.listdir(example_dir))
    unique_files = {}
    for f in all_files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(f)[0]
            if base_name not in unique_files:
                unique_files[base_name] = f

    target_files = sorted(unique_files.values())
    if not target_files:
        return jsonify({"error": "No example images found."}), 404
    # --------------------------------------------------------------------

    if request.method == 'POST':
        # Server-side emulation of loading examples
        state = get_user_state()
        state.__init__()  # Reset memory

        settings = request.json
        state.absolute = settings.get('absolute', False)
        px, info = settings.get('pixel_size'), settings.get('info')
        state.pixel_size_um = float(px) if px else 3.45
        state.info = info or "Example Dataset"

        n_images = len(target_files)
        for idx, filename in enumerate(target_files):
            path = os.path.join(example_dir, filename)
            image = cv2.imread(path)
            if image is None: continue

            if state.image_array is None:
                state.image_array = np.zeros((image.shape + (n_images,)), dtype=image.dtype)

            state.image_array[..., idx] = image
            state.image_names.append(filename)

        state.calibration_parameters = state.calibrator.calibrate(
            image_array=state.image_array,
            space_between_features=float(settings.get('checker_size', 14.0)),
            board_size=None,
            detector_type="checkerboard",
            detector_params={},
            absolute=state.absolute
        )

        if state.pixel_size_um: state.calibration_parameters.pixel_size = state.pixel_size_um / 1000.0
        if state.info: state.calibration_parameters.info = state.info

        return jsonify(format_calibration_response(state.calibration_parameters, state.calibrator, state))

    else:
        # Local-side raw data fetch (GET)
        file_data = []
        for filename in target_files:
            path = os.path.join(example_dir, filename)
            with open(path, 'rb') as f:
                file_data.append({"name": filename, "data": base64.b64encode(f.read()).decode('utf-8')})

        return jsonify({
            "files": file_data,
            "checker_size": 14.0,
            "pixel_size": 3.45,
            "info": "acA1440-220uc Example"
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=7000)