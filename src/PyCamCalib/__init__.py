"""This is the camera calibration toolbox package."""

from os.path import join
from importlib.resources import files


camera_images = files(__name__).joinpath(join('examples', 'camera_images'))
stereo_left = files(__name__).joinpath(join('examples', 'stereo_images', 'left'))
stereo_right = files(__name__).joinpath(join('examples', 'stereo_images', 'right'))
