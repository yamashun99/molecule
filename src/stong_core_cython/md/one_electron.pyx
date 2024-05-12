# cython: language_level=3

from libc.math cimport exp, sqrt, M_PI, pow
import numpy as np
cimport numpy as np
from scipy.special.cython_special cimport hyp1f1
from cython cimport view
include "utils.pxi"

cdef double overlap(double a, long[3] lmn1, double[:] A, double b, long[3] lmn2, double[:] B) nogil:
    """
    カーテシアンガウス関数の重なり積分を計算する関数
    """
    cdef long l1 = lmn1[0], m1 = lmn1[1], n1 = lmn1[2]
    cdef long l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2]
    cdef double S1 = E(l1, l2, 0, a, b, A[0] - B[0])
    cdef double S2 = E(m1, m2, 0, a, b, A[1] - B[1])
    cdef double S3 = E(n1, n2, 0, a, b, A[2] - B[2])
    return S1 * S2 * S3 * pow(M_PI / (a+b), 1.5)


cpdef double S(object a, object b):
    """
    縮約されたカーテシアンガウス関数の重なり積分を計算する関数
    """
    cdef long num_exps = len(a.exps), i, j
    cdef double s = 0.0
    cdef double[:] norms_a = a.norm, norms_b = b.norm
    cdef double[:] coefs_a = a.coefs, coefs_b = b.coefs
    cdef double[:] exps_a = a.exps, exps_b = b.exps
    cdef long[3] lmn_a = a.lmn, lmn_b = b.lmn
    cdef double[:] origin_a = a.origin, origin_b = b.origin

    with nogil:
        for i in range(num_exps):
            for j in range(num_exps):
                s += (
                    norms_a[i]
                    * norms_b[j]
                    * coefs_a[i]
                    * coefs_b[j]
                    * overlap(exps_a[i], lmn_a, origin_a, exps_b[j], lmn_b, origin_b)
                )
    return s


cdef double kinetic(double a, long[3] lmn1, double[:] A, double b, long[3] lmn2, double[:] B) nogil:
    """
    カーテシアンガウス関数の重なり積分を計算する関数
    """
    cdef long l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2]
    cdef long[3] lmn2_l = [l2 +2, m2, n2]
    cdef long[3] lmn2_m = [l2, m2 + 2, n2]
    cdef long[3] lmn2_n = [l2, m2, n2 + 2]
    cdef long[3] lmnm2_l = [l2 - 2, m2, n2]
    cdef long[3] lmnm2_m = [l2, m2 - 2, n2]
    cdef long[3] lmnm2_n = [l2, m2, n2 - 2]

    cdef double term0 = b * (2 * (l2 + m2 + n2) + 3) * overlap(a, lmn1, A, b, lmn2, B)
    cdef double term1 = (
        -2
        * b**2
        * (
            overlap(a, lmn1, A, b, lmn2_l, B)
            + overlap(a, lmn1, A, b, lmn2_m, B)
            + overlap(a, lmn1, A, b, lmn2_n, B)
        )
    )
    cdef double term2 = -0.5 * (
        l2 * (l2 - 1) * overlap(a, lmn1, A, b, lmnm2_l, B)
        + m2 * (m2 - 1) * overlap(a, lmn1, A, b, lmnm2_m, B)
        + n2 * (n2 - 1) * overlap(a, lmn1, A, b, lmnm2_n, B)
    )
    return term0 + term1 + term2


cpdef double T(object a, object b):
    """
    縮約されたカーテシアンガウス関数の運動エネルギー積分を計算する関数
    """
    cdef double t = 0.0
    cdef int i, j, num_exps = len(a.exps)
    cdef long[3] lmn_a = a.lmn, lmn_b = b.lmn
    cdef double[:] origin_a = a.origin, origin_b = b.origin
    cdef double[:] norms_a = a.norm, norms_b = b.norm
    cdef double[:] coefs_a = a.coefs, coefs_b = b.coefs
    cdef double[:] exps_a = a.exps, exps_b = b.exps

    with nogil:
        for i in range(num_exps):
            for j in range(num_exps):
                t += (
                    norms_a[i]
                    * norms_b[j]
                    * coefs_a[i]
                    * coefs_b[j]
                    * kinetic(exps_a[i], lmn_a, origin_a, exps_b[j], lmn_b, origin_b)
                )
        return t

cdef double nuclear_attraction(double a, long[3] lmn1, double[:] A,
                                 double b, long[3] lmn2, double[:] B, double[:] C) nogil:
    """
    カーテシアンガウス関数の原子核からのクーロン相互作用を計算する関数
    """
    cdef double p = a + b
    cdef long l1 = lmn1[0], m1 = lmn1[1], n1 = lmn1[2]
    cdef long l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2]
    cdef double[3] RPC
    cdef long t, u, v, i
    for i in range(3):
        RPC[i] = (a * A[i] + b * B[i]) / p - C[i]
    cdef double val = 0.0
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                val += (
                    E(l1, l2, t, a, b, A[0] - B[0])
                    * E(m1, m2, u, a, b, A[1] - B[1])
                    * E(n1, n2, v, a, b, A[2] - B[2])
                    * R(0, t, u, v, p, RPC)
                )
    return 2 * M_PI / p * val

cpdef double V(object a, object b, double[:] RC):
    """
    縮約されたカーテシアンガウス関数の原子核からのクーロン相互作用積分を計算する関数
    """
    cdef double v = 0.0
    cdef long num_exps = len(a.exps), i, j
    cdef long[3] lmn_a = a.lmn, lmn_b = b.lmn
    cdef double[:] origin_a = a.origin, origin_b = b.origin
    cdef double[:] norms_a = a.norm, norms_b = b.norm
    cdef double[:] coefs_a = a.coefs, coefs_b = b.coefs
    cdef double[:] exps_a = a.exps, exps_b = b.exps
    with nogil:
        for i in range(num_exps):
            for j in range(num_exps):
                v += (
                    norms_a[i]
                    * norms_b[j]
                    * coefs_a[i]
                    * coefs_b[j]
                    * nuclear_attraction(
                        exps_a[i], lmn_a, origin_a, exps_b[j], lmn_b, origin_b, RC
                    )
                )
        return v