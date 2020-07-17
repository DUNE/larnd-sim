#!/usr/bin/env python

import setuptools

VER = "0.0.1"

setuptools.setup(
    name="larndsim",
    version=VER,
    author="DUNE collaboration",
    author_email="roberto@lbl.gov",
    description="Simulation framework for the DUNE LArND",
    url="https://github.com/soleti/larnd-sim",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "scikit-image", "tqdm", "pytest", "numba"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: by End-User Class :: Developers",
        "Operating System :: Grouping and Descriptive Categories :: OS Independent (Written in an interpreted language)",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.6',
)
