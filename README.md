# Camera calibration toolbox

## About

The camera calibration toolbox contains tools for performing camera and stereo calibration, evaluating the calibration 
results, and some basic uses for the parameters such as distortion correction and rectification. 

## Installation

Requires Python >= 3.10.

| Command | What you get |
|---|---|
| `pip install PyCamCalib` | The headless library (`PyCamCalib.core`): calibration, parameters, undistortion. No Qt, safe for servers and CI. |
| `pip install "PyCamCalib[gui]"` | The library plus the desktop GUI (PySide6, pyqtgraph). |
| `pip install "PyCamCalib[web]"` | The library plus the Flask dependencies (Flask, gunicorn) for the browser app in the repository. |
| `pip install "PyCamCalib[gui,web]"` | Everything. |

The extras can be combined. Only one OpenCV wheel can be installed at a time; the library depends on
`opencv-contrib-python`. If you need `opencv-contrib-python-headless` (for example in a container), install it instead
after uninstalling the former.

To develop from a clone, use [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/InViLabUAntwerp/PyCamCalib
cd PyCamCalib
uv sync --extra gui --extra web
```

## Usage

### Camera calibration

To calibrate a camera you need a stack of images of a calibration target (checkerboard). The image stack should either 
be a 3D grayscale (x, y, n) or 4D BGR (x, y, c, n) numpy array with a datatype that can be converted to numpy float64. 
In order to get a reliable calibration, at least 11 images with different checkerboard poses and locations are required, 
it is also important to have images of the checkerboard in each corner. The camera can then be calibrated with the 
`calibrate` method of `CameraCalibrator`, which returns an instance of `CameraParameters`. The size of the checkerboard
(inner corners (n_rows, n_cols) is not required. If you don't care about te scale you can set the checker size as 1. 
For additional arguments, check the documentation.

```
from PyCamCalib.core.calibration import CameraCalibrator


calibrator = CameraCalibrator()
parameters = calibrator.calibrate(image_array, checker_size)
```

The `CameraCalibrator` retains some information about the calibration process: the detected points for each image,
which images were used, the overall RMS of the re-projection error and the re-projection error for each image that was 
used. The re-projection errors can be plotted with:

```
calibrator.plot_reproj_error()
```

Sometimes the re-projection error of an image or a few images might be higher than the rest, which may indicate it is an
outlier. In this case it might be beneficial to exclude these from the calibration process. The `calibrate_indices` 
method allows you to calibrate the camera with only the selected images by giving a list of the indices. Since all 
necessary data was stored inside the `CameraCalibrator`, this can be repeated as many times as you want without having
to start the calibration process all over again.

```
indices = [0, 1, 5, 6, 7, 8, 9, 10]
parameters = calibrator.calibrate_indices(indices)
```

All calibration parameters are stored in a `CameraParameters` object. You can extract parameters and other information 
with a variety of methods, such as:

```
parameters.plot_distortion()  
intrinsics_matrix = parameters.get_intrinsics_matrix_opencv()
distortion_coeffs = parameters.get_distortion_coeffs_opencv()
```

You can use `CameraParameters` to remove distortion from images. First, you need to calculate the maps necessary for 
remapping:

```
parameters.calculate_undistort_map()
```

The maps are stored internally, so they can be reused. You can then undistort an image with:

```
corrected_image = parameters.remap_image(image)
```

You can store the parameters in an `.h5` file for later use:

```
your_file = some_folder/some_name.h5
parameters.save_parameters(your_file)
```

and load them into a new `CameraParameters` object:

```
from PyCamCalib.core.calibration import CameraParameters


parameters = CameraParameters()
parameters.load_parameters(your_file)
```

### Stereo calibration

Stereo calibration can be performed for camera's with different sensor/image sizes, but you won't be able to rectify the
images. Should you wish to do so, crop (if the FOV is much bigger than that of the other camera) and rescale the image
of the camera with the higher resolution, so it matches the image size of the other camera exactly **BEFORE performing 
ALL calibration steps**.

In order to perform a stereo calibration, you need to calibrate both cameras separately first, as described above. The 
`CameraParameters` are used as inputs for the stereo calibration. In addition, you need two image stacks (one for each
camera), which contain a few matching images of the calibration pattern. The stereo calibration is performed with the 
`StereoCalibrator` and returns `StereoParameters`, which also contains the `CameraParameters` for each camera, so you 
are not required to keep/store these separately. Here the size of the checkerboard is required.

```
from PyCamCalib.core.calibration import StereoCalibrator


calibrator = StereoCalibrator()
stereo_params = calibrator.calibrate(image_array_1, image_array_2, param_1, param_2, checker_size, board_dims)
```

Similarly to the `CameraCalibrator`, you can also check the re-projection errors and retry the calibration with only
certain images. 

In order to rectify the images, you first need to calculate the maps for removing distortion and rectification:

```
stereo_params.calculate_undistort_rectify_maps()
```

after which `StereoParameters` can be used to rectify and undistort both images at once:

```
corr_1, corr_2 = stereo_parameters.remap_images(image_1, image_2)
```

The parameters can be stored in an `.h5` file for later use:

```
your_file = some_folder/some_name.h5
parameters.save_parameters(your_file)
```

and loaded again into a new `StereoParameters` object:

```
from PyCamCalib.core.calibration import StereoParameters


parameters = CameraParameters()
parameters.load_parameters(your_file)
```

### GUI

For your convenience, there is also a GUI for the camera calibration process. It requires the `gui` extra
(`pip install "PyCamCalib[gui]"`) and is launched by executing `camera_calibration_gui` in the terminal. Without the
extra, the command prints an installation hint instead of starting.

Examples for all these use cases can be found in the
[online repository](https://github.com/InViLabUAntwerp/PyCamCalib). Please refer to the documentation for additional
in-depth information.

