#!/usr/bin/env python

VER = "0.3.1"

reqs = ["numpy", "pytest", "numba>=0.52", "larpix-control", "larpix-geometry", "tqdm", "fire", "nvidia-ml-py"]

try:
    import cupy
    msg = '''
    ############ INFORMATION ############
    Detected and using the installed cupy
    Version: %s
    Source : %s
    #####################################\n
    '''
    print(msg % (str(cupy.__version__),str(cupy.__file__)))
except ImportError:
    reqs.append('cupy')

import os
if 'SKIP_CUPY_INSTALL' in os.environ:
    try:
        _ = reqs.pop(reqs.index('cupy'))
    except ValueError:
        pass

import setuptools

setuptools.setup(
    name="larndsim",
    version=VER,
    author="DUNE collaboration",
    author_email="roberto@lbl.gov",
    description="Simulation framework for the DUNE LArND",
    url="https://github.com/DUNE/larnd-sim",
    packages=setuptools.find_packages(),
    scripts=["cli/simulate_pixels.py", "cli/dumpTree.py"],
    install_requires=reqs,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: by End-User Class :: Developers",
        "Operating System :: Grouping and Descriptive Categories :: OS Independent (Written in an interpreted language)",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.7',
)
