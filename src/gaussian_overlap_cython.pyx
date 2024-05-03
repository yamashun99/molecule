# cython: language_level=3

from libc.math cimport sin
from libc.math cimport exp
import numpy as np
cimport numpy as np
from cpython.tuple cimport tuple
from libc.math cimport sqrt, M_PI

cdef long custom_fact2(long n):
    cdef long result = 1
    cdef long i
    for i in range(n, 0, -2):
        result *= i
    return result

cdef class BasisFunction:
    cdef public double[:] origin
    cdef public tuple[long, long, long] lmn
    cdef public double[:] exps
    cdef public double[:] coefs
    cdef public double[:] norm

    def __init__(self, double[:] center, tuple[long, long, long] lmn, double[:] exps, double[:] coefs):
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


cdef double E_memo(long i, long j, long t, double a, double b, double Qx, double[:, :, :] memo):
    if i < 0 or j < 0 or t < 0 or i + j < t:
        return 0
    
    if memo[i, j, t] != 0:
        return memo[i, j, t]

    cdef double result
    if i == j == t == 0:
        result = exp(-a * b * Qx * Qx / (a + b))
    elif j == 0:  # decrement i
        result = (
            1 / (2 * (a + b)) * E_memo(i - 1, j, t - 1, a, b, Qx, memo)
            - a * b * Qx / (a * (a + b)) * E_memo(i - 1, j, t, a, b, Qx, memo)
            + (t + 1) * E_memo(i - 1, j, t + 1, a, b, Qx, memo)
        )
    else:  # decrement j
        result = (
            1 / (2 * (a + b)) * E_memo(i, j - 1, t - 1, a, b, Qx, memo)
            + a * b * Qx / (b * (a + b)) * E_memo(i, j - 1, t, a, b, Qx, memo)
            + (t + 1) * E_memo(i, j - 1, t + 1, a, b, Qx, memo)
        )

    memo[i, j, t] = result
    return result

cdef double E(long i, long j, long t, double a, double b, double Qx):
    """
    カーテシアンガウス関数の積をエルミートガウス関数で展開したときの係数を計算する関数
    
    Parameters
    ----------
    i : int
        カーテシアンガウス関数の次数
    j : int
        カーテシアンガウス関数の次数
    t : int
        エルミートガウス関数の次数
    a : float
        カーテシアンガウス関数の幅の逆数
    b : float
        カーテシアンガウス関数の幅の逆数
    Qx : float
        カーテシアンガウス関数の位置の差
    
    Returns
    -------
    float
        エルミートガウス関数の係数
    """
    cdef long max_dim = max(i, j, t) + 1
    cdef double[:, :, :] memo = np.zeros((max_dim, max_dim, max_dim), dtype=np.float64)
    return E_memo(i, j, t, a, b, Qx, memo)


cdef double overlap(double a, tuple[long, long, long] lmn1, double[:] A, double b, tuple[long, long, long] lmn2, double[:] B):
    """
    カーテシアンガウス関数の重なり積分を計算する関数
    """
    cdef long l1 = lmn1[0], m1 = lmn1[1], n1 = lmn1[2]
    cdef long l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2]
    cdef double S1 = E(l1, l2, 0, a, b, A[0] - B[0])
    cdef double S2 = E(m1, m2, 0, a, b, A[1] - B[1])
    cdef double S3 = E(n1, n2, 0, a, b, A[2] - B[2])
    return S1 * S2 * S3 * pow(np.pi / (a+b), 1.5)


cpdef double S(BasisFunction a, BasisFunction b):
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


cdef double kinetic(double a, tuple[long, long, long] lmn1, double[:] A, double b, tuple[long, long, long] lmn2, double[:] B):
    """
    カーテシアンガウス関数の重なり積分を計算する関数
    """
    cdef long l2 = lmn2[0], m2 = lmn2[1], n2 = lmn2[2]
    cdef double term0 = b * (2 * (l2 + m2 + n2) + 3) * overlap(a, lmn1, A, b, lmn2, B)
    cdef double term1 = (
        -2
        * b**2
        * (
            overlap(a, lmn1, A, b, (l2 + 2, m2, n2), B)
            + overlap(a, lmn1, A, b, (l2, m2 + 2, n2), B)
            + overlap(a, lmn1, A, b, (l2, m2, n2 + 2), B)
        )
    )
    cdef double term2 = -0.5 * (
        l2 * (l2 - 1) * overlap(a, lmn1, A, b, (l2 - 2, m2, n2), B)
        + m2 * (m2 - 1) * overlap(a, lmn1, A, b, (l2, m2 - 2, n2), B)
        + n2 * (n2 - 1) * overlap(a, lmn1, A, b, (l2, m2, n2 - 2), B)
    )
    return term0 + term1 + term2


cpdef double T(BasisFunction a, BasisFunction b):
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