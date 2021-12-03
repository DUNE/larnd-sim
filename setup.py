#!/usr/bin/env python

import setuptools

VER = "0.2.1"

setuptools.setup(
    name="larndsim",
    version=VER,
    author="DUNE collaboration",
    author_email="roberto@lbl.gov",
    description="Simulation framework for the DUNE LArND",
    url="https://github.com/DUNE/larnd-sim",
    packages=setuptools.find_packages(),
    scripts=["cli/simulate_pixels.py", "cli/dumpTree.py"],
    install_requires=["numpy", "pytest", "numba==0.52", "larpix-control", "larpix-geometry", "tqdm", "fire", "cupy"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: by End-User Class :: Developers",
        "Operating System :: Grouping and Descriptive Categories :: OS Independent (Written in an interpreted language)",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.7',
)
