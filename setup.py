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
try:
    cuda_dir = os.path.basename(os.environ['CUDA_HOME'])
    cuda_ver = float(cuda_dir)
    cuda_major_ver = int(cuda_ver)
except:
    cuda_ver = cuda_major_ver = -1

if 'cupy' in reqs:
    if 'SKIP_CUPY_INSTALL' in os.environ:
        reqs.remove('cupy')
    else:
        if 'ALWAYS_COMPILE_CUPY' not in os.environ:
            if cuda_major_ver in [11, 12]:
                reqs.remove('cupy')
                reqs.append(f'cupy-cuda{cuda_major_ver}x')

import setuptools

setuptools.setup(
    name="larndsim",
    version=VER,
    author="DUNE collaboration",
    author_email="roberto@lbl.gov",
    description="Simulation framework for the DUNE LArND",
    url="https://github.com/DUNE/larnd-sim",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'larndsim': ['config/*.yaml',
    'simulation_properties/*.yaml',
    'pixel_layouts/*.yaml',
    'detector_properties/*.yaml',
    'detector_properties/*.json',
    'bin/*.npy',
    'bin/*.npz',
    ]},
    scripts=["cli/simulate_pixels.py", "cli/dumpTree.py", "cli/list_config_keys.py"],
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

if os.getenv('LMOD_SYSTEM_NAME') == 'perlmutter':
    # Revisit this after Perlmutter driver update (currently 525.105.17)
    if cuda_ver >= 12.4 and 'SKIP_PYNVJITLINK_INSTALL' not in os.environ:
        os.system('pip install --extra-index-url https://pypi.nvidia.com pynvjitlink-cu12')
