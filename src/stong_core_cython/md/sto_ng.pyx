# cython: language_level=3

from libc.math cimport exp, sqrt, M_PI
import numpy as np
cimport numpy as np
from scipy.special.cython_special cimport hyp1f1
from cython cimport view
include "utils.pxi"

cdef class BasisFunction:
    cdef public double[:] origin
    cdef public long[:] lmn
    cdef public double[:] exps
    cdef public double[:] coefs
    cdef public double[:] norm

    def __init__(self, double[:] center, long[:] lmn, double[:] exps, double[:] coefs):
        self.origin = center
        self.lmn = lmn
        self.exps = exps
        self.coefs = coefs
        self.norm = np.zeros_like(coefs)
        self.normalize()

    cdef void normalize(self):
        cdef long l, m, n, num_exps, ia, ja
        cdef double prefactor, N, L
        cdef double[:] norm_view = self.norm
        cdef double[:] exps_view = self.exps
        cdef double[:] coefs_view = self.coefs

        l, m, n = self.lmn
        L = l + m + n
        prefactor = (
            custom_fact2(2 * l - 1)
            * custom_fact2(2 * m - 1)
            * custom_fact2(2 * n - 1)
            * M_PI ** 1.5
            / 2 ** (2 * L + 1.5)
        )

        num_exps = exps_view.shape[0]
        for ia in range(num_exps):
            norm_view[ia] = exps_view[ia] ** (L / 2 + 0.75) / sqrt(prefactor)

        N = 0
        for ia in range(num_exps):
            for ja in range(num_exps):
                N += (
                    coefs_view[ia]
                    * coefs_view[ja]
                    * norm_view[ia]
                    * norm_view[ja]
                    / ((exps_view[ia] + exps_view[ja]) / 2) ** (L + 1.5)
                )

        N *= prefactor
        N = 1 / sqrt(N)

        for ia in range(num_exps):
            coefs_view[ia] = coefs_view[ia] * N