"""
run this command to build the cython code
python setup.py build_ext --inplace
"""

from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np

extensions = [
    Extension(
        "src.stong_core_cython.gaussian_integrals",
        ["./src/stong_core_cython/gaussian_integrals.pyx"],
    ),
]

setup(
    name="gaussian_integrals",
    ext_modules=cythonize(extensions, annotate=True),  # annotate=True をここに移動
    include_dirs=[np.get_include()],
)
