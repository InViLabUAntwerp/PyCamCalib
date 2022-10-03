# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Camera calibration toolbox'
copyright = '2022, Michael Hillen <michael.hillen@uantwerpen.be>, Seppe Sels <seppe.sels@uantwerpen.be>'
author = 'Michael Hillen <michael.hillen@uantwerpen.be>, Seppe Sels <seppe.sels@uantwerpen.be>'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx',
              'sphinx_autodoc_typehints']

autosummary_generate = True

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/1.21/', None)}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member_callback)


def autodoc_skip_member_callback(app, what, name, obj, skip, options):
    exclusions = ('../../calibration_toolbox/gui/',)
    exclude = name in exclusions
    return skip or exclude
