"""This sub-package contains the code for the GUI."""
from os.path import join
from importlib.resources import files

invilab_logo = files(__name__).joinpath(join('resources', 'Logo.jpg'))
invilab_icon = files(__name__).joinpath(join('resources', 'Logo_InViLab_Icon_color.jpg'))
