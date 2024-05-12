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
        "src.stong_core_cython.md.sto_ng",
        ["./src/stong_core_cython/md/sto_ng.pyx"],
        include_dirs=["./src/stong_core_cython/md"],  # この行を追加
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        "src.stong_core_cython.md.one_electron",
        [
            "./src/stong_core_cython/md/one_electron.pyx",
        ],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        "src.stong_core_cython.md.two_electron",
        [
            "./src/stong_core_cython/md/two_electron.pyx",
        ],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="gaussian_integrals",
    ext_modules=cythonize(extensions, annotate=True),  # annotate=True をここに移動
    include_dirs=[np.get_include()],
)
