// --- 1. GLOBAL STATE & UTILITIES ---
const uaBlue = '#002e65';
const uaRed = '#ea2c38';
const ftiPurple = '#b10097';
let excludedIndices = [];
let pendingRemoval = [];
let currentColors = [];
let lastCalibrationData = null;
let totalLoadedImages = 0;
let lastClickTime = 0;

let TAB_ID = sessionStorage.getItem('invilab_tab_id');
if (!TAB_ID) {
    TAB_ID = (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : Math.random().toString(36).slice(2);
    sessionStorage.setItem('invilab_tab_id', TAB_ID);
}

// Disconnect when leaving the page (useful for Server mode)
window.addEventListener('pagehide', () => navigator.sendBeacon(`/api/disconnect?tid=${TAB_ID}`));

function setStatus(text, colorClass) {
    const badge = document.getElementById('statusBadge');
    if (badge) {
        badge.className = `badge ${colorClass} p-2 fs-6 shadow-sm`;
        badge.innerText = `Status: ${text}`;
    }
}

function showAlert(id, msg) {
    const el = document.getElementById(id);
    if (msg) {
        el.innerText = msg;
        el.classList.remove('d-none');
    } else {
        el.classList.add('d-none');
    }
}

// --- 2. BACKEND ADAPTER INTERFACES ---
const ServerBackend = {
    init: async () => {
        const hdrs = document.getElementById('headerBadges');
        hdrs.innerHTML = `
            <span id="userCountBadge" class="badge bg-info text-dark p-2 fs-6 me-2 shadow-sm">Users: 1/8</span>
            <span id="statusBadge" class="badge bg-secondary p-2 fs-6 shadow-sm">Status: Ready</span>
        `;

        const checkStatus = async () => {
            try {
                const res = await fetch('/api/status', { headers: { 'X-Tab-ID': TAB_ID } });
                const data = await res.json();
                const badge = document.getElementById('userCountBadge');
                badge.innerText = `Users: ${data.count}/${data.max}`;
                const btn = document.getElementById('btnCalibrate');

                if (data.waitlisted) {
                    btn.disabled = true; badge.className = 'badge bg-danger p-2 fs-6 me-2 shadow-sm';
                    showAlert('warningAlert', 'The server is currently at maximum capacity. Please wait for a slot.');
                } else {
                    if (document.getElementById('statusBadge').innerText === "Status: Ready") btn.disabled = false;
                    badge.className = 'badge bg-info text-dark p-2 fs-6 me-2 shadow-sm';
                    const warn = document.getElementById('warningAlert');
                    if (!warn.classList.contains('d-none') && warn.innerText.includes('maximum capacity')) showAlert('warningAlert', null);
                }
            } catch(e) {}
        };
        checkStatus(); setInterval(checkStatus, 10000);
        return true;
    },
    calibrateInitial: async (files, settings) => {
        const fd = new FormData();
        for (let f of files) fd.append('files', f);
        for (const [k, v] of Object.entries(settings)) { if (v !== null && v !== '') fd.append(k, v); }
        const res = await fetch(`/api/calibrate/initial`, { method: 'POST', headers: {'X-Tab-ID': TAB_ID}, body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.details || data.error || "Server Error");
        return data;
    },
    recalculate: async (excluded, info) => {
        const res = await fetch(`/api/calibrate/recalculate`, {
            method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Tab-ID': TAB_ID },
            body: JSON.stringify({ excluded_indices: excluded, info: info })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Server Error");
        return data;
    },
    getDetection: async (idx) => {
        const res = await fetch(`/api/detections/${idx}`, { headers: { 'X-Tab-ID': TAB_ID } });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        return data;
    },
    loadExample: async (settings) => {
    // CHANGE THIS URL from `/api/load-example` to `/api/example-files`
    const res = await fetch(`/api/example-files`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Tab-ID': TAB_ID },
        body: JSON.stringify(settings)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    return data;
}
};

const LocalBackend = {
    init: async () => {
        const hdrs = document.getElementById('headerBadges');
        hdrs.innerHTML = `
            <span id="engineBadge" class="badge bg-warning text-dark p-2 fs-6 me-2 shadow-sm">Loading Pyodide Engine...</span>
            <span id="statusBadge" class="badge bg-secondary p-2 fs-6 shadow-sm">Status: Initializing</span>
        `;
        const eb = document.getElementById('engineBadge');
        try {
            window.pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/" });
            eb.innerText = "Installing Packages...";
            await window.pyodide.loadPackage(['numpy', 'opencv-python', 'matplotlib', 'h5py', 'scikit-learn']);

            eb.innerText = "Loading Custom Wheel...";
            await window.pyodide.loadPackage(APP_CONFIG.urls.libcbdetect);

            const extractZip = async (filename, folderName) => {
                const res = await fetch(APP_CONFIG.urls.staticRoot + filename);
                if (res.ok) {
                    const buf = await res.arrayBuffer();
                    window.pyodide.FS.writeFile(filename, new Uint8Array(buf));
                    await window.pyodide.runPythonAsync(`
import zipfile, os, sys
with zipfile.ZipFile('${filename}', 'r') as zip_ref:
    has_folder = any(name.startswith('${folderName}/') for name in zip_ref.namelist())
    if has_folder: 
        zip_ref.extractall('.')
    else: 
        os.makedirs('${folderName}', exist_ok=True)
        zip_ref.extractall('${folderName}')
if '.' not in sys.path: sys.path.append('.')
                    `);
                }
            };
            eb.innerText = "Loading PyCBD..."; await extractZip('PyCBD.zip', 'PyCBD');
            eb.innerText = "Loading CTPv..."; await extractZip('CTPv.zip', 'CTPv');
            eb.innerText = "Loading PyCamCalib..."; await extractZip('PyCamCalib.zip', 'PyCamCalib');

            eb.innerText = "Booting Logic Engine...";

            // Restored clean python string without dangerous minification
            await window.pyodide.runPythonAsync(`
import os
import sys
import io
import base64
import json
import importlib
import importlib.abc
import importlib.machinery
from types import ModuleType

class UniversalMock(metaclass=type):
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return UniversalMock()
    def __call__(self, *args, **kwargs):
        return UniversalMock()
    def __iter__(self):
        return iter([])

class UniversalMockModule(ModuleType):
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return UniversalMock()

class DummyPool:
    def __init__(self, *args, **kwargs):
        pass
    def map(self, func, iterable, chunksize=None):
        return [func(x) for x in iterable]
    def starmap(self, func, iterable, chunksize=None):
        return [func(*x) for x in iterable]
    def apply(self, func, args=(), kwds={}):
        return func(*args, **kwds)
    def apply_async(self, func, args=(), kwds={}, callback=None, error_callback=None):
        res = type('AsyncRes', (), {})()
        try:
            val = func(*args, **kwds)
            res.get = lambda timeout=None: val
            if callback:
                callback(val)
        except Exception as e:
            if error_callback:
                error_callback(e)
            else:
                raise e
        return res
    def close(self):
        pass
    def join(self):
        pass
    def terminate(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

class DummyProcess:
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs
    def start(self):
        if self.target:
            self.target(*self.args, **self.kwargs)
    def join(self, timeout=None):
        pass
    def terminate(self):
        pass
    def is_alive(self):
        return False

class DummySharedMemory:
    def __init__(self, name=None, create=False, size=0):
        self.name = name or "dummy_shm"
        self.size = size
        self.buf = bytearray(size)
    def close(self):
        pass
    def unlink(self):
        pass

class MultiprocessingMockFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'multiprocessing' or fullname.startswith('multiprocessing.') or fullname in ('_multiprocessing', '_posixshmem'):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None
    def create_module(self, spec):
        name = spec.name
        mod = UniversalMockModule(name)
        if name == 'multiprocessing':
            mod.Pool = DummyPool
            mod.Process = DummyProcess
            mod.cpu_count = lambda: 1
            shm_mod = UniversalMockModule('multiprocessing.shared_memory')
            shm_mod.SharedMemory = DummySharedMemory
            mod.shared_memory = shm_mod
        elif name == 'multiprocessing.shared_memory':
            mod.SharedMemory = DummySharedMemory
        return mod
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, MultiprocessingMockFinder())

import concurrent.futures
cf_process_mock = ModuleType('concurrent.futures.process')
cf_process_mock.ProcessPoolExecutor = concurrent.futures.ThreadPoolExecutor
sys.modules['concurrent.futures.process'] = cf_process_mock

sys.modules['PyCBD.checkerboard_enhancement.checkerboard_enhancer'] = UniversalMockModule('PyCBD.checkerboard_enhancement.checkerboard_enhancer')
sys.modules['PyCBD.checkerboard_enhancement'] = UniversalMockModule('PyCBD.checkerboard_enhancement')

class DummyFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, target_names):
        self.target_names = target_names
    def find_spec(self, fullname, path, target=None):
        if any(fullname == t or fullname.startswith(t + '.') for t in self.target_names):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None
    def create_module(self, spec):
        return UniversalMockModule(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, DummyFinder(['torch', 'gpytorch', 'torchvision', 'open3d', 'vispy', 'plyfile']))

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

site_packages = '/lib/python3.11/site-packages/'
cb_folder = next((f for f in os.listdir(site_packages) if f.lower() == 'libcbdetect'), None)
if cb_folder and os.path.join(site_packages, cb_folder) not in sys.path:
    sys.path.append(os.path.join(site_packages, cb_folder))

from PyCamCalib.core.calibration import CameraCalibrator, CameraParameters

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
    except: 
        return None

def format_calibration_response():
    m = state.calibration_parameters.get_intrinsics_matrix_opencv()
    d = state.calibration_parameters.get_distortion_coeffs_opencv()
    fl_mm = None
    if state.calibration_parameters.pixel_size:
        fl_mm = [float(state.calibration_parameters.focal_length_mm[0]), float(state.calibration_parameters.focal_length_mm[1])]
    
    fov_deg = [float(state.calibration_parameters.PerspectiveAngle[0]), float(state.calibration_parameters.PerspectiveAngle[1])]
    
    extrinsics_data = []
    if hasattr(state.calibrator, 'r_vecs') and hasattr(state.calibrator, 't_vecs'):
        r_vecs = np.atleast_2d(state.calibrator.r_vecs)
        t_vecs = np.atleast_2d(state.calibrator.t_vecs)
        if len(r_vecs) > 0 and len(r_vecs) == len(state.calibrator.indices):
            for idx, r, t, obj_pts in zip(state.calibrator.indices, r_vecs, t_vecs, state.calibrator.object_points_list):
                try:
                    R_mat, _ = cv2.Rodrigues(r)
                    transformed_pts = (R_mat @ obj_pts.T).T + t.reshape(3)
                    extrinsics_data.append({"index": int(idx), "points": transformed_pts.tolist()})
                except:
                    pass

    return {
        "status": "success",
        "intrinsics": {
            "fx": [float(state.calibration_parameters.f[0]), float(state.calibration_parameters.f_std[0])],
            "fy": [float(state.calibration_parameters.f[1]), float(state.calibration_parameters.f_std[1])],
            "cx": [float(state.calibration_parameters.c[0]), float(state.calibration_parameters.c_std[0])],
            "cy": [float(state.calibration_parameters.c[1]), float(state.calibration_parameters.c_std[1])],
            "s": [float(state.calibration_parameters.s), float(state.calibration_parameters.s_std)],
            "focal_length_mm": fl_mm,
            "fov_deg": fov_deg
        },
        "distortion": {
            "k1": [float(state.calibration_parameters.radial_dist_coeffs[0]), float(state.calibration_parameters.radial_dist_coeffs_std[0])],
            "k2": [float(state.calibration_parameters.radial_dist_coeffs[1]), float(state.calibration_parameters.radial_dist_coeffs_std[1])],
            "k3": [float(state.calibration_parameters.radial_dist_coeffs[2]), float(state.calibration_parameters.radial_dist_coeffs_std[2])],
            "p1": [float(state.calibration_parameters.tangential_dist_coeffs[0]), float(state.calibration_parameters.tangential_dist_coeffs_std[0])],
            "p2": [float(state.calibration_parameters.tangential_dist_coeffs[1]), float(state.calibration_parameters.tangential_dist_coeffs_std[1])]
        },
        "plot_data": {
            "indices": [int(x) for x in state.calibrator.indices],
            "errors": [float(e) for e in state.calibrator.per_view_err],
            "rms": float(state.calibrator.rms_reproj_error),
            "distortion_map": calculate_distortion_map(m, d, state.calibration_parameters.sensor_dimensions),
            "extrinsics": extrinsics_data
        },
        "total_images": state.image_array.shape[-1] if state.image_array is not None else 0,
        "sensor_dimensions": [int(state.calibration_parameters.sensor_dimensions[0]), int(state.calibration_parameters.sensor_dimensions[1])],
        "pixel_size": state.calibration_parameters.pixel_size,
        "info": state.calibration_parameters.info,
        "board_size": state.board_size,
        "warning": "At least 11 good images are required." if len(state.calibrator.indices) < 11 else None
    }

def py_get_detection_image(index):
    try:
        index = int(index)
        if state.image_array is None or index < 0 or index >= state.image_array.shape[-1]: 
            return json.dumps({"error": "Invalid index"})
        
        img = state.image_array[..., index].copy()
        shape_str = "N/A"
        reproj_err = "N/A (Excluded)"
        
        if index < len(state.calibrator.feature_list):
            feature = state.calibrator.feature_list[index]
            if feature.score > 0:
                min_score = 2 if state.absolute else 1
                if feature.score >= min_score and index in state.calibrator.indices:
                    color = (0, 255, 0)
                elif feature.score >= min_score:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                    
                if feature.image_points is not None:
                    for pt in feature.image_points:
                        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, color, -1)
            
            if feature.score > 0 and feature.object_points is not None and len(feature.object_points) > 0:
                xs = np.unique(np.round(feature.object_points[:, 0], decimals=3))
                ys = np.unique(np.round(feature.object_points[:, 1], decimals=3))
                shape_str = f"{len(xs)}x{len(ys)}"
                
        try:
            err_idx = state.calibrator.indices.index(index)
            reproj_err = round(float(state.calibrator.per_view_err[err_idx]), 4)
        except ValueError:
            pass
            
        success, buffer = cv2.imencode('.jpg', img)
        if not success:
            return json.dumps({"error": "Failed encode"})
            
        img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        return json.dumps({
            "image_data": f"data:image/jpeg;base64,{img_base64}", 
            "name": state.image_names[index], 
            "reproj_error": reproj_err, 
            "shape": shape_str
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

def py_calibrate_initial(file_bytes_list, file_names, settings):
    state.__init__()
    images = []
    for b in file_bytes_list:
        decoded = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            images.append(decoded)
            
    if not images:
        return json.dumps({"error": "No valid images"})
        
    state.image_names = list(file_names)
    h, w, c = images[0].shape
    state.image_array = np.zeros((h, w, c, len(images)), dtype=images[0].dtype)
    for i, img in enumerate(images):
        state.image_array[..., i] = img
    
    dt = settings.get('detector_type', 'checkerboard')
    cs = float(settings.get('checker_size', 1.0))
    state.absolute = bool(settings.get('absolute', False))
    
    px = settings.get('pixel_size')
    info = settings.get('info')
    state.pixel_size_um = float(px) if px else None
    state.info = info
    
    bw = settings.get('board_width')
    bh = settings.get('board_height')
    board_size = (int(bw), int(bh)) if bw and bh else None
    
    dp = {}
    if dt == "charuco":
        preset = settings.get('charuco_preset')
        dp["preset"] = None if preset == 'None' else preset

    try:
        state.calibration_parameters = state.calibrator.calibrate(
            image_array=state.image_array, 
            space_between_features=cs, 
            board_size=board_size,
            detector_type=dt, 
            detector_params=dp, 
            absolute=state.absolute
        )
        
        if board_size is None and state.calibrator.feature_list:
            for feature in state.calibrator.feature_list:
                if feature.score > 0 and feature.object_points is not None and len(feature.object_points) > 0:
                    xs = np.unique(np.round(feature.object_points[:, 0] / cs))
                    ys = np.unique(np.round(feature.object_points[:, 1] / cs))
                    state.board_size = (len(xs), len(ys))
                    break
        else:
            state.board_size = board_size
            
        if state.pixel_size_um:
            state.calibration_parameters.pixel_size = state.pixel_size_um / 1000.0
        if state.info:
            state.calibration_parameters.info = state.info
            
        return json.dumps(format_calibration_response())
    except Exception as e:
        return json.dumps({"error": str(e)})

def py_calibrate_recalculate(excluded_indices, info_str):
    if state.image_array is None:
        return json.dumps({"error": "No image data"})
        
    good_indices = [i for i in range(state.image_array.shape[-1]) if i not in excluded_indices]
    if not good_indices:
        return json.dumps({"error": "You can't remove all images."})
        
    try:
        state.calibration_parameters = state.calibrator.calibrate_indices(good_indices, state.absolute)
        if state.pixel_size_um:
            state.calibration_parameters.pixel_size = state.pixel_size_um / 1000.0
        if info_str:
            state.calibration_parameters.info = info_str
        return json.dumps(format_calibration_response())
    except Exception as e:
        return json.dumps({"error": str(e)})
            `);

            window.pyCalibrateInitial = window.pyodide.globals.get('py_calibrate_initial');
            window.pyCalibrateRecalculate = window.pyodide.globals.get('py_calibrate_recalculate');
            window.pyGetDetectionImage = window.pyodide.globals.get('py_get_detection_image');

            eb.className = 'badge bg-success p-2 fs-6 me-2 shadow-sm'; eb.innerText = "Python Engine: Local (WASM)";
            setStatus("Ready", "bg-success");
            document.getElementById('btnCalibrate').disabled = false;
            return true;
        } catch(e) {
            eb.className = 'badge bg-danger p-2 fs-6 me-2 shadow-sm'; eb.innerText = "Python Engine: Failed";
            showAlert('errorAlert', e.message); return false;
        }
    },
    calibrateInitial: async (files, settings) => {
        const fileNames = []; const filePromises = [];
        for (let f of files) { fileNames.push(f.name); filePromises.push(f.arrayBuffer()); }
        const buffs = await Promise.all(filePromises);
        const bytesList = buffs.map(b => new Uint8Array(b));
        const jsonStr = window.pyCalibrateInitial(window.pyodide.toPy(bytesList), window.pyodide.toPy(fileNames), window.pyodide.toPy(settings));
        const data = JSON.parse(jsonStr);
        if (data.error) throw new Error(data.error);
        return data;
    },
    recalculate: async (excluded, info) => {
        const jsonStr = window.pyCalibrateRecalculate(window.pyodide.toPy(excluded), info);
        const data = JSON.parse(jsonStr);
        if (data.error) throw new Error(data.error);
        return data;
    },
    getDetection: async (idx) => {
        const jsonStr = window.pyGetDetectionImage(parseInt(idx));
        const data = JSON.parse(jsonStr);
        if (data.error) throw new Error(data.error);
        return data;
    },
    loadExample: async (settings) => {
        const res = await fetch('/api/example-files');
        const fileData = await res.json();
        if (!res.ok) throw new Error(fileData.error);
        const fileNames = []; const bytesList = [];
        for (const item of fileData.files) {
            fileNames.push(item.name);
            const binaryStr = atob(item.data); const bytes = new Uint8Array(binaryStr.length);
            for (let i=0; i<binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
            bytesList.push(bytes);
        }
        const localSettings = { ...settings, checker_size: 14.0, pixel_size: 3.45, info: "acA1440-220uc Example", detector_type: "checkerboard" };
        const jsonStr = window.pyCalibrateInitial(window.pyodide.toPy(bytesList), window.pyodide.toPy(fileNames), window.pyodide.toPy(localSettings));
        const data = JSON.parse(jsonStr);
        if (data.error) throw new Error(data.error);
        return data;
    }
};

// Instantiate backend based on mode
const Backend = APP_CONFIG.computeMode === 'local' ? LocalBackend : ServerBackend;
Backend.init();

// --- 3. UI CONTROLLERS ---

function toggleDetectorOptions() {
    const dt = document.getElementById('detectorType').value;
    const charucoOpts = document.getElementById('charucoOptions');
    const bsGroup = document.getElementById('boardSizeGroup');
    const absGroup = document.getElementById('absoluteCheckGroup');
    if (dt === 'charuco') {
        charucoOpts.classList.remove('d-none'); bsGroup.classList.add('d-none'); absGroup.classList.add('d-none');
    } else {
        charucoOpts.classList.add('d-none'); bsGroup.classList.remove('d-none'); absGroup.classList.remove('d-none');
    }
}

function getSettingsObject() {
    return {
        checker_size: document.getElementById('checkerSize').value,
        detector_type: document.getElementById('detectorType').value,
        charuco_preset: document.getElementById('charucoPreset').value,
        absolute: (document.getElementById('detectorType').value === 'checkerboard' && document.getElementById('absoluteCheck').checked),
        pixel_size: document.getElementById('pixelSize').value || null,
        info: document.getElementById('infoInput').value || null,
        board_width: document.getElementById('boardWidth').value || null,
        board_height: document.getElementById('boardHeight').value || null
    };
}

async function appLoadExampleData() {
    setStatus("Loading Example...", "bg-warning text-dark");
    showAlert('errorAlert', null); showAlert('warningAlert', null);
    excludedIndices = []; pendingRemoval = [];

    document.getElementById('checkerSize').value = 14.0;
    document.getElementById('pixelSize').value = 3.45;
    document.getElementById('infoInput').value = "acA1440-220uc Example";
    document.getElementById('detectorType').value = 'checkerboard';
    toggleDetectorOptions();

    try {
        const data = await Backend.loadExample(getSettingsObject());
        handleCalibrationSuccess(data);
    } catch(e) { showAlert('errorAlert', e.message); setStatus("Error", "bg-danger"); }
}

async function appInitialCalibration() {
    const fi = document.getElementById('imageFiles');
    if (fi.files.length === 0) return showAlert('errorAlert', "Please select images first.");

    if (APP_CONFIG.computeMode === 'server') {
        let total = 0; for(let f of fi.files) total += f.size;
        if (total > (APP_CONFIG.maxUploadMB * 1024 * 1024)) return showAlert('errorAlert', `Upload exceeds ${APP_CONFIG.maxUploadMB}MB limit.`);
    }

    setStatus("Processing...", "bg-warning text-dark");
    showAlert('errorAlert', null); showAlert('warningAlert', null);
    excludedIndices = []; pendingRemoval = [];

    try {
        const data = await Backend.calibrateInitial(fi.files, getSettingsObject());
        handleCalibrationSuccess(data);
    } catch(e) { showAlert('errorAlert', e.message); setStatus("Error", "bg-danger"); }
}

async function appRecalculate() {
    if (pendingRemoval.length === 0) return alert("Select an outlier first.");
    setStatus("Recalculating...", "bg-warning text-dark");
    excludedIndices = excludedIndices.concat(pendingRemoval); pendingRemoval = [];
    try {
        const data = await Backend.recalculate(excludedIndices, document.getElementById('infoInput').value);
        handleCalibrationSuccess(data);
    } catch(e) { showAlert('errorAlert', e.message); setStatus("Error", "bg-danger"); }
}

async function appResetOutliers() {
    setStatus("Resetting...", "bg-warning text-dark");
    excludedIndices = []; pendingRemoval = [];
    try {
        const data = await Backend.recalculate(excludedIndices, document.getElementById('infoInput').value);
        handleCalibrationSuccess(data);
    } catch(e) { showAlert('errorAlert', e.message); setStatus("Error", "bg-danger"); }
}

async function appFetchDetectionImage(idx) {
    try {
        const data = await Backend.getDetection(idx);
        const img = document.getElementById('detectionImg');
        img.src = data.image_data; img.classList.remove('d-none');
        document.getElementById('viewerName').innerText = `Index: ${idx} | Image: ${data.name}`;
        document.getElementById('viewerError').innerText = `Error: ${data.reproj_error}`;
        document.getElementById('viewerShape').innerText = `Shape: ${data.shape}`;
        detZoom.reset();
    } catch(e) { console.error("Fetch failed", e); }
}

// --- 4. RENDERERS ---
function handleCalibrationSuccess(data) {
    setStatus("Ready", "bg-success");
    if (data.warning) showAlert('warningAlert', data.warning);
    lastCalibrationData = data; totalLoadedImages = data.total_images;

    updateTable(data.intrinsics, data.distortion);
    renderRMSPlot(data.plot_data);
    renderExtrinsicsPlot(data.plot_data);

    if (data.plot_data.distortion_map) {
        const d = document.getElementById('distImg');
        d.src = data.plot_data.distortion_map; d.classList.remove('d-none'); distZoom.reset();
    }

    const s = document.getElementById('imageSlider');
    s.max = totalLoadedImages - 1; s.disabled = false; s.value = 0;
    document.getElementById('lblMaxIdx').innerText = totalLoadedImages - 1;
    appFetchDetectionImage(0);

    document.getElementById('btnRecalculate').disabled = false; document.getElementById('btnReset').disabled = false;
    document.getElementById('btnExport').disabled = false; document.getElementById('btnExportLensfun').disabled = false;
    document.getElementById('btnExportDistortion').disabled = false;

    fetch('/api/increment-counter', {method: 'POST'})
        .then(r=>r.json()).then(d=>{ if(d.success) document.querySelector('.calibration-counter').innerText = `Total calibrations performed: ${d.count}`; })
        .catch(()=>{});
}

function updateTable(intrinsics, distortion) {
    const tbody = document.querySelector('#resultsTable tbody'); tbody.innerHTML = '';
    const add = (k, v) => {
        const vS = v && v[0]!=null ? v[0].toFixed(4) : '/'; const sS = v && v[1]!=null ? v[1].toFixed(4) : '/';
        tbody.innerHTML += `<tr><td class="fw-bold">${k}</td><td>${vS}</td><td>${sS}</td></tr>`;
    };
    for (const [k, v] of Object.entries(intrinsics)) { if(k!=='focal_length_mm' && k!=='fov_deg') add(k, v); }
    for (const [k, v] of Object.entries(distortion)) add(k, v);
    if (intrinsics.focal_length_mm) tbody.innerHTML += `<tr><td class="fw-bold text-success">Focal Length (mm)</td><td colspan="2" class="text-success">fx: ${intrinsics.focal_length_mm[0].toFixed(2)}, fy: ${intrinsics.focal_length_mm[1].toFixed(2)}</td></tr>`;
    if (intrinsics.fov_deg) tbody.innerHTML += `<tr><td class="fw-bold text-primary">FOV (degrees)</td><td colspan="2" class="text-primary">X: ${intrinsics.fov_deg[0].toFixed(2)}&deg;, Y: ${intrinsics.fov_deg[1].toFixed(2)}&deg;</td></tr>`;
}

function updatePlotsColors() {
    if(!lastCalibrationData) return;
    currentColors = lastCalibrationData.plot_data.indices.map(idx => pendingRemoval.includes(idx) ? uaRed : uaBlue);
    Plotly.restyle('reprojPlot', 'marker.color', [currentColors]);
    if(lastCalibrationData.plot_data.extrinsics) {
        const exColors = lastCalibrationData.plot_data.extrinsics.map(ex => pendingRemoval.includes(ex.index) ? uaRed : uaBlue);
        const trIdx = Array.from({length: exColors.length}, (_,i)=>i+1);
        if(trIdx.length > 0) Plotly.restyle('extrinsicsPlot', 'marker.color', exColors, trIdx);
    }
}

function renderRMSPlot(pd) {
    pendingRemoval = []; currentColors = pd.indices.map(() => uaBlue);
    const trace = { x: pd.indices.map(String), y: pd.errors, type: 'bar', marker: {color: currentColors} };
    const layout = { margin: {t:30,b:40,l:50,r:10}, xaxis: {title: 'Image Index', type: 'category', fixedrange: true}, yaxis: {title: 'RMS (pixels)', fixedrange: true}, dragmode: false, shapes: [{type:'line', x0:-0.5, x1:pd.indices.length-0.5, y0:pd.rms, y1:pd.rms, line:{color:ftiPurple, width:2, dash:'dash'}}], plot_bgcolor:"#fff", paper_bgcolor:"#fff" };
    Plotly.newPlot('reprojPlot', [trace], layout);
    document.getElementById('reprojPlot').on('plotly_click', (d) => {
        const now = Date.now(); if(now - lastClickTime < 400) return; lastClickTime = now;
        const idx = parseInt(d.points[0].x); const pIdx = pendingRemoval.indexOf(idx);
        if(pIdx === -1) pendingRemoval.push(idx); else pendingRemoval.splice(pIdx, 1);
        updatePlotsColors();
    });
}

function renderExtrinsicsPlot(pd) {
    if (!pd.extrinsics || pd.extrinsics.length === 0) {
        document.getElementById('extrinsicsPlot').innerHTML = '<p class="text-muted text-center mt-5">No 3D data</p>'; return;
    }
    const data = [{ x:[0], y:[0], z:[0], mode:'markers+text', text:['Camera'], textposition:'bottom center', marker:{size:8, color:'#000'}, name:'Camera', type:'scatter3d', hoverinfo:'name' }];
    pd.extrinsics.forEach(ext => {
        data.push({ x: ext.points.map(p=>p[0]), y: ext.points.map(p=>p[1]), z: ext.points.map(p=>p[2]), mode:'markers', marker:{size:3, color:pendingRemoval.includes(ext.index)?uaRed:uaBlue}, name:`Image ${ext.index}`, type:'scatter3d', hoverinfo:'name' });
    });
    Plotly.newPlot('extrinsicsPlot', data, { margin:{t:0,b:0,l:0,r:0}, scene:{xaxis:{title:'X'}, yaxis:{title:'Y'}, zaxis:{title:'Z'}, aspectmode:'data'}, showlegend:false, plot_bgcolor:"#fff", paper_bgcolor:"#fff" });
    document.getElementById('extrinsicsPlot').on('plotly_click', (d) => {
        const now = Date.now(); if(now - lastClickTime < 400) return; lastClickTime = now;
        if(d.points[0].curveNumber === 0) return;
        const match = d.points[0].data.name.match(/Image (\d+)/); if(!match) return;
        const idx = parseInt(match[1]); const pIdx = pendingRemoval.indexOf(idx);
        if(pIdx === -1) pendingRemoval.push(idx); else pendingRemoval.splice(pIdx, 1);
        updatePlotsColors();
    });
}

// --- ZOOM LOGIC ---
class ZoomPanController {
    constructor(cid, iid) {
        this.c = document.getElementById(cid); this.i = document.getElementById(iid);
        this.s = 1; this.tx = 0; this.ty = 0; this.drag = false; this.sx = 0; this.sy = 0;
        this.c.addEventListener('wheel', e => { e.preventDefault(); this.s = Math.max(0.5, this.s + (e.deltaY > 0 ? -0.1 : 0.1)); this.apply(); });
        this.c.addEventListener('mousedown', e => { this.drag = true; this.sx = e.clientX - this.tx; this.sy = e.clientY - this.ty; this.c.style.cursor = 'grabbing'; });
        this.c.addEventListener('mouseup', () => { this.drag = false; this.c.style.cursor = 'grab'; });
        this.c.addEventListener('mouseleave', () => { this.drag = false; this.c.style.cursor = 'grab'; });
        this.c.addEventListener('mousemove', e => { if(!this.drag) return; e.preventDefault(); this.tx = e.clientX - this.sx; this.ty = e.clientY - this.sy; this.apply(); });
    }
    apply() { this.i.style.transform = `translate(${this.tx}px, ${this.ty}px) scale(${this.s})`; }
    zoom(d) { this.s = Math.max(0.5, this.s + d); this.apply(); }
    reset() { this.s = 1; this.tx = 0; this.ty = 0; this.apply(); }
}
const detZoom = new ZoomPanController('imageContainer', 'detectionImg');
const distZoom = new ZoomPanController('distContainer', 'distImg');

// --- EXPORTS & MODALS ---
function escapeXML(str) { return str.replace(/[<>&'"]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','\'':'&apos;','"':'&quot;'}[c])); }
function exportJSON() {
    if(!lastCalibrationData) return;
    const e = {
        f: [lastCalibrationData.intrinsics.fx[0], lastCalibrationData.intrinsics.fy[0]], f_std: [lastCalibrationData.intrinsics.fx[1], lastCalibrationData.intrinsics.fy[1]],
        c: [lastCalibrationData.intrinsics.cx[0], lastCalibrationData.intrinsics.cy[0]], c_std: [lastCalibrationData.intrinsics.cx[1], lastCalibrationData.intrinsics.cy[1]],
        s: lastCalibrationData.intrinsics.s[0], s_std: lastCalibrationData.intrinsics.s[1],
        radial_dist_coeffs: [lastCalibrationData.distortion.k1[0], lastCalibrationData.distortion.k2[0], lastCalibrationData.distortion.k3[0]],
        radial_dist_coeffs_std: [lastCalibrationData.distortion.k1[1], lastCalibrationData.distortion.k2[1], lastCalibrationData.distortion.k3[1]],
        tangential_dist_coeffs: [lastCalibrationData.distortion.p1[0], lastCalibrationData.distortion.p2[0]],
        tangential_dist_coeffs_std: [lastCalibrationData.distortion.p1[1], lastCalibrationData.distortion.p2[1]],
        rms_reproj_error: lastCalibrationData.plot_data.rms, sensor_dimensions: lastCalibrationData.sensor_dimensions,
        pixel_size: lastCalibrationData.pixel_size || null, info: lastCalibrationData.info || null
    };
    const d = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(e, null, 4));
    const a = document.createElement('a'); a.href = d;
    a.download = lastCalibrationData.info ? `calibration_parameters_${lastCalibrationData.info.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.json` : "calibration_parameters.json";
    a.click();
}
function exportLensfunXML() {
    if(!lastCalibrationData) return;
    const info = lastCalibrationData.info || "Custom Camera";
    let f = 50.0; if(lastCalibrationData.intrinsics.focal_length_mm) f = (lastCalibrationData.intrinsics.focal_length_mm[0]+lastCalibrationData.intrinsics.focal_length_mm[1])/2;
    else if (lastCalibrationData.pixel_size) f = lastCalibrationData.intrinsics.fx[0] * lastCalibrationData.pixel_size;
    const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<lensdatabase version="1">\n  <camera>\n    <maker>InViLab</maker>\n    <model>${escapeXML(info)}</model>\n    <sensor width="${lastCalibrationData.sensor_dimensions[0]}" height="${lastCalibrationData.sensor_dimensions[1]}" cropfactor="1.0"/>\n  </camera>\n  <lens>\n    <maker>InViLab</maker>\n    <model>${escapeXML(info)} Lens Profile</model>\n    <calibration model="poly7" focal="${f.toFixed(2)}" crop="1.0">\n      <distortion k1="${lastCalibrationData.distortion.k1[0]}" k2="${lastCalibrationData.distortion.k2[0]}" k3="${lastCalibrationData.distortion.k3[0]}"/>\n    </calibration>\n  </lens>\n</lensdatabase>`;
    const url = URL.createObjectURL(new Blob([xml], {type: 'application/xml'}));
    const a = document.createElement('a'); a.href = url;
    a.download = lastCalibrationData.info ? `lensfun_profile_${lastCalibrationData.info.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.xml` : "lensfun_profile.xml";
    a.click(); URL.revokeObjectURL(url);
}
function exportDistortion() {
    const src = document.getElementById('distImg').src; if(!src) return;
    const a = document.createElement('a'); a.href = src; a.download = "distortion_plot.png"; a.click();
}

let codeModal;
function displayModal(title, code) {
    document.getElementById('codeModalTitle').innerText = title;
    document.getElementById('codeSnippetContent').innerText = code;
    if (!codeModal) codeModal = new bootstrap.Modal(document.getElementById('codeModal'));
    codeModal.show();
}
function copyCodeSnippet() { navigator.clipboard.writeText(document.getElementById('codeSnippetContent').innerText).then(()=>alert("Copied!")); }
function showCameraParametersScript() { displayModal("Python: Use CameraParameters Class", `from PyCamCalib.core.CameraParameters import CameraParameters\nimport cv2\nimport matplotlib.pyplot as plt\n\ndef main():\n    cam_params = CameraParameters()\n    cam_params.load_parameters_from_json('calibration_parameters.json')\n    print(f"Focal Length (pixels): fx={cam_params.f[0]:.2f}, fy={cam_params.f[1]:.2f}")\n    cam_params.plot_distortion()\n\n    img = cv2.imread('myimage.png')\n    if img is not None:\n        plt.imshow(cam_params.undistort_image(img, alpha=1, crop=False))\n        plt.show()\n\nif __name__ == "__main__":\n    main()`); }
function showBatchCalibrationScript() { displayModal("Python: Headless Batch Calibration", `import cv2, numpy as np\nfrom PyCamCalib.core.calibration import CameraCalibrator\n\ndef main():\n    images = [cv2.imread(p) for p in ['image1.jpg', 'image2.jpg'] if cv2.imread(p) is not None]\n    if not images: return\n    arr = np.zeros((images[0].shape[0], images[0].shape[1], images[0].shape[2], len(images)), dtype=images[0].dtype)\n    for i, img in enumerate(images): arr[..., i] = img\n    \n    cal = CameraCalibrator()\n    cal.construct_feature_list(arr, 1.0)\n    cal.construct_points_lists([], False)\n    params = cal.opencv_calibration(np.array([arr.shape[1], arr.shape[0]]))\n    print(f"fx, fy: {params.f}\\nRMS: {cal.rms_reproj_error}")\n\nif __name__ == "__main__":\n    main()`); }

async function showInfoModal(file) {
    const c = document.getElementById('infoModalContent'); c.innerHTML = '<p class="text-center">Loading...</p>';
    new bootstrap.Modal(document.getElementById('infoModal')).show();
    try {
        const res = await fetch(`/api/info/${file}`);
        if(!res.ok) throw new Error(); c.innerHTML = await res.text();
    } catch(e) { c.innerHTML = '<p class="text-center text-danger">Failed to load info.</p>'; }
}