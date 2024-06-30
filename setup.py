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
        "sto_ng",
        ["./cython/sto_ng.pyx"],
        include_dirs=["./cython"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        "one_electron",
        ["./cython/one_electron.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        "two_electron",
        ["./cython/two_electron.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="gaussian_integrals",
    ext_modules=cythonize(extensions, annotate=True),  # annotate=True をここに移動
    include_dirs=[np.get_include()],
)
