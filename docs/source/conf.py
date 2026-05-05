# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
import sys

# -- Path setup ---------------------------------------------------------------
# Make the larnd-sim source tree importable for autodoc.
# Adjust this path if conf.py is moved relative to the repo root.
sys.path.insert(0, os.path.abspath('../..'))

# Mock heavy C-extension / GPU dependencies so autodoc can import the
# package on a plain CI or Read-the-Docs runner without CUDA.
from unittest.mock import MagicMock

MOCK_MODULES = [
    # Heavy GPU / C-extension runtime deps
    'numba',
    'numba.cuda',
    'numba.cuda.random',
    'cupy',
    'pynvml',
    # Scientific stack (may not be installed in a bare docs environment)
    'numpy',
    'h5py',
    'yaml',
    'scipy',
    'scipy.interpolate',
    'skimage',
    'skimage.measure',
    'ROOT',
    # Others,
    'fire',
    'tqdm',
    # DUNE / LArPix-specific packages
    'adc64format',
    'larpix',
    'larpix.format',
    'larpix.format.hdf5format',
]

# for mod_name in MOCK_MODULES:
    # sys.modules[mod_name] = MagicMock()

# -- Project information ------------------------------------------------------
project = 'larnd-sim'
copyright = '2026, DUNE collaboration'
author = 'DUNE collaboration'

# The version is managed by setuptools-scm; fall back gracefully when the
# package is not installed (e.g. in a bare docs build).
try:
    from larndsim._version import __version__ as release
except ImportError:
    release = 'latest'
version = release

# -- General configuration ----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',        # Pull docstrings from source
    'sphinx.ext.autosummary',    # Generate summary tables
    'sphinx.ext.napoleon',       # NumPy / Google-style docstrings
    'sphinx.ext.viewcode',       # [source] links next to each object
    'sphinx.ext.intersphinx',    # Cross-link to NumPy, Python docs, etc.
    'sphinx.ext.mathjax',        # Render LaTeX math
    'sphinx.ext.githubpages',    # Drop .nojekyll into the build output
    'sphinx.ext.linkcode',       # Link to source on GitHub
    'myst_parser',               # Parse Markdown files (README, CONTRIBUTING)
]

def get_object_line_number(info):
    """Return object line number from module."""
    try:
        module = sys.modules.get(info["module"])
        if module is None:
            return None

        # walk through the nested module structure
        obj = module
        for part in info["fullname"].split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None

        # return inspect.getsourcelines(obj)[1]
        return inspect.getsourcelines(getattr(obj, "py_func", obj))[1]
    except (TypeError, OSError, ValueError):
        return None

def linkcode_resolve(domain, info):
    """Return GitHub link to source code."""
    if domain != "py":
        return None
    if not info["module"]:
        return None

    filename = info["module"].replace(".", "/")
    line = get_object_line_number(info)
    github_repo = "https://github.com/DUNE/larnd-sim"

    return f"{github_repo}/blob/develop/{filename}.py#L{line}"

# MyST options — allow standard Markdown heading anchors
myst_heading_anchors = 3

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- autodoc options ----------------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}
# Preserve the order in which members appear in the source file.
autodoc_member_order = 'bysource'
# Do not expand type aliases — keeps signatures readable.
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# autosummary: auto-generate stub .rst files
autosummary_generate = True

autodoc_mock_imports = MOCK_MODULES

# -- Napoleon (NumPy / Google docstrings) ------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx mapping ------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
    'cupy':   ('https://docs.cupy.dev/en/stable/', None),
    'h5py':   ('https://docs.h5py.org/en/stable', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

# -- Options for HTML output --------------------------------------------------
# html_theme = 'pydata_sphinx_theme'
html_theme = 'shibuya'
html_logo = '_static/logo.png'

html_theme_options = {
    'github_url': 'https://github.com/DUNE/larnd-sim',
    'accent_color': 'orange',
    'navigation_with_keys': True,
    'show_ai_links': False,
}

html_context = {
    'source_type': 'github',
    'source_user': 'DUNE',
    'source_repo': 'larnd-sim',
}

html_static_path = ['_static']
# html_css_files = ['custom.css']

html_title = 'larnd-sim'
html_short_title = 'larnd-sim'
html_show_sphinx = True
html_show_sourcelink = True

# -- Options for LaTeX / PDF output -------------------------------------------
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
}
