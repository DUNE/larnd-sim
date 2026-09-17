from setuptools import setup
import os
import re
import subprocess

dependencies = [
    "cupy",
    "numpy",
    "numba>=0.52",
    "pytest",
    "larpix-control @ git+https://github.com/larpix/larpix-control.git@release",
    "larpix-geometry",
    "tqdm",
    "fire",
    "nvidia-ml-py",
]

# Extra development tools or CI/CD dependencies
optional_dep = {
    "linting" : ["MonkeyType", "mypy", "ruff"],
    "runner" : ["pandas", "rich"]
}

def get_cuda_version():
    # Try nvcc --version (the CUDA compiler)
    try:
        res = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
        match = re.search(r"release (\d+)", res)
        if match:
            return int(match.group(1))
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try $CUDA_HOME and extract version from directory name
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        # Match digits in the basename, e.g. "12.2" -> 12
        match = re.search(r"(\d+)", os.path.basename(cuda_home))
        if match:
            return int(match.group(1))

    return None

cuda_major_ver = get_cuda_version()

# We only support CUDA 11 and higher
if cuda_major_ver is None or cuda_major_ver < 11:
    if "SKIP_CUPY_INSTALL" not in os.environ:
        raise RuntimeError(
            f"CUDA version {cuda_major_ver} not detected or unsupported (<11). "
            "Please set CUDA_HOME or ensure nvcc is in PATH, or use SKIP_CUPY_INSTALL=1 to skip CuPy."
        )
else:
    print(f"larnd-sim -- Found CUDA version: {cuda_major_ver}")

# Confirmed to have supported CUDA, check how to (or if) install CuPy
if "cupy" in dependencies:
    if "SKIP_CUPY_INSTALL" in os.environ:
        print("larnd-sim -- Skipping CuPy installation.")
        dependencies.remove("cupy")
    elif "ALWAYS_COMPILE_CUPY" not in os.environ:
        print("larnd-sim -- Using pre-built wheel for CuPy and numba-cuda.")
        dependencies.remove("cupy")
        dependencies.append(f"cupy-cuda{cuda_major_ver}x")
        dependencies.append(f"numba-cuda[cu{cuda_major_ver}]") # TODO: Update numba-cuda to numba-cuda-mlir
    else:
        print("larnd-sim -- Attempting to build CuPy from source.")

setup(
    install_requires=dependencies,
    extras_require=optional_dep,
)
