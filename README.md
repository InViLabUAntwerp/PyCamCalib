# Camera calibration toolbox

## Introduction

The camera calibration toolbox contains tools for performing a geometric camera calibration, evaluating the calibration 
results, removing distortion and calculating the FOV. 

## Requirements

* Python >= 3.7
* (Optional) camera_toolbox_python (specifically the DataClass) if you want to be able to load SEP and HDF5 image data 
  files with the GUI

## Installation

1. Clone/download the distribution branch of the repository
2. Make sure you have the latest version of pip
3. Use `pip` to install the `.whl` file located in `Calibration_toolbox_python/dist`. 
```
pip install path/camera_calibration_toolbox-x.x.x-py3-none-any.whl
```
3. Properly configure `sys.path` or `PYTHONPATH` if you want to use the `DataClass`

## Use

The full documentation for the toolbox can be accessed from `Calibration_toolbox_python/docs/build/index.html`.
Examples for how to use the toolbox can be found in`/examples/examples.py`. There is also a GUI which
can be used for calibration and evaluation of the results, but it supports only a limited amount of image file types.

### Feature tags

Both the GUI and `CameraCalibrator` class use a *feature tag* in order to determine which calibration feature they 
should look for in the image. The tag should be constructed as follows:

#### Normal checkerboard

e.g.: NCH0000070630  
* 0 - 2: checkerboard type, normal Checkerboard = NCH 
* 3 - 6: unused 
* 7 - 10: board size (the amount of inner corners) in rows (7-8) columns (9-10), e.g. size (7, 6) = 0706
* 12 - end: checker size in mm, for decimal numbers use a point separator, e.g. 30 mm = 30

### GUI

The GUI can be launched from the terminal:
```
python -m camera_calibration_toolbox.gui
```
or by running `camera_calibration_toolbox/gui/__main__.py` in a python console.

Load the image files using the *browse* button. Normal HDF5 files, i.e. files not created with the `DataClass` require a 
header for the image data. Provide a *feature tag* in the *tag* field and click the *calibrate* button. The calibration 
parameters can be exported to a HDF5 file with the *export* button. The calibration can be refined by removing outliers 
in the *Reproj errors* tab. Click on the bars in the bar graph to mark images as outliers, then click the 
*Remove outliers* button to perform a calibration without the selected images. The *Reset* button will repeat the 
calibration with all available images.