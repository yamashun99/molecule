# cython: language_level=3

from libc.math cimport exp, sqrt, M_PI
import numpy as np
cimport numpy as np
from scipy.special.cython_special cimport hyp1f1
from cython cimport view
include "utils.pxi"

cdef double overlap(double a, long[:] lmn1, double[:] A, double b, long[:] lmn2, double[:] B):
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
    for i in range(num_exps):
        for j in range(num_exps):
            s += (
                a.norm[i]
                * b.norm[j]
                * a.coefs[i]
                * b.coefs[j]
                * overlap(a.exps[i], a.lmn, a.origin, b.exps[j], b.lmn, b.origin)
            )
    return s


cdef double kinetic(double a, long[:] lmn1, double[:] A, double b, long[:] lmn2, double[:] B):
    """
    カーテシアンガウス関数の重なり積分を計算する関数
    """
    cdef long l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2]
    cdef double term0 = b * (2 * (l2 + m2 + n2) + 3) * overlap(a, lmn1, A, b, lmn2, B)
    cdef double term1 = (
        -2
        * b**2
        * (
            overlap(a, lmn1, A, b, np.array([l2 + 2, m2, n2]), B)
            + overlap(a, lmn1, A, b, np.array([l2, m2 + 2, n2]), B)
            + overlap(a, lmn1, A, b, np.array([l2, m2, n2 + 2]), B)
        )
    )
    cdef double term2 = -0.5 * (
        l2 * (l2 - 1) * overlap(a, lmn1, A, b, np.array([l2 - 2, m2, n2]), B)
        + m2 * (m2 - 1) * overlap(a, lmn1, A, b, np.array([l2, m2 - 2, n2]), B)
        + n2 * (n2 - 1) * overlap(a, lmn1, A, b, np.array([l2, m2, n2 - 2]), B)
    )
    return term0 + term1 + term2


cpdef double T(object a, object b):
    """
    縮約されたカーテシアンガウス関数の運動エネルギー積分を計算する関数
    """
    cdef double t = 0.0
    cdef int i, j, num_exps = len(a.exps)
    for i in range(num_exps):
        for j in range(num_exps):
            t += (
                a.norm[i]
                * b.norm[j]
                * a.coefs[i]
                * b.coefs[j]
                * kinetic(a.exps[i], a.lmn, a.origin, b.exps[j], b.lmn, b.origin)
            )
    return t

cdef double nuclear_attraction(double a, long[:] lmn1, double[:] A,
                                 double b, long[:] lmn2, double[:] B, double[:] C):
    """
    カーテシアンガウス関数の原子核からのクーロン相互作用を計算する関数
    """
    cdef double p = a + b
    cdef long l1 = lmn1[0], m1 = lmn1[1], n1 = lmn1[2]
    cdef long l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2]
    cdef double[:] RPC = view.array(shape=(3,), itemsize=sizeof(double), format="d")
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

    for i in range(num_exps):
        for j in range(num_exps):
            v += (
                a.norm[i]
                * b.norm[j]
                * a.coefs[i]
                * b.coefs[j]
                * nuclear_attraction(
                    a.exps[i], a.lmn, a.origin, b.exps[j], b.lmn, b.origin, RC
                )
            )
    return v