from setuptools import setup
import os

dependencies = [
    "numpy",
    "pytest",
    "numba>=0.52",
    "larpix-control",
    "larpix-geometry",
    "tqdm",
    "fire",
    "nvidia-ml-py",
]

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
    dependencies.append('cupy')

try:
    cuda_dir = os.path.basename(os.environ['CUDA_HOME'])
    cuda_ver = float(cuda_dir)
    cuda_major_ver = int(cuda_ver)
    print(f"larnd-sim -- Found CUDA version: {cuda_ver}")
except:
    cuda_ver = cuda_major_ver = -1

if 'cupy' in dependencies:
    if 'SKIP_CUPY_INSTALL' in os.environ:
        dependencies.remove('cupy')
    else:
        if 'ALWAYS_COMPILE_CUPY' not in os.environ:
            if cuda_major_ver in [11, 12]:
                dependencies.remove('cupy')
                dependencies.append(f'cupy-cuda{cuda_major_ver}x')

setup(
    install_requires=dependencies,
)
