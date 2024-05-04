# cython: language_level=3

from libc.math cimport sin
from libc.math cimport exp
import numpy as np
cimport numpy as np
from cpython.tuple cimport tuple
from libc.math cimport sqrt, M_PI
from scipy.special.cython_special cimport hyp1f1
from cython cimport view

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

cpdef double E(long i, long j, long t, double a, double b, double Qx):
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
    cdef double[:, :, :] memo = np.zeros((i+1, j+1, i+j+t+1), dtype=np.float64)
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
    return S1 * S2 * S3 * pow(M_PI / (a+b), 1.5)


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

cpdef double boys(int n, double T):
    return hyp1f1(n + 0.5, n + 1.5, -T) / (2 * n + 1)

cdef double R_memo(int n, int t, int u, int v, double p, double[:] RPC, dict memo):
    cdef double norm_RPC_sq = RPC[0]**2 + RPC[1]**2 + RPC[2]** 2
    cdef tuple key = (n, t, u, v)
    
    if key in memo:
        return memo[key]
    
    cdef double result
    if t < 0 or u < 0 or v < 0:
        result = 0
    elif t == 0 and u == 0 and v == 0:
        result = (-2 * p) ** n * boys(n, p * norm_RPC_sq)
    elif t == 0 and u == 0:
        result = (v - 1) * R_memo(n + 1, t, u, v - 2, p, RPC, memo) + RPC[2] * R_memo(n + 1, t, u, v - 1, p, RPC, memo)
    elif t == 0:
        result = (u - 1) * R_memo(n + 1, t, u - 2, v, p, RPC, memo) + RPC[1] * R_memo(n + 1, t, u - 1, v, p, RPC, memo)
    else:
        result = (t - 1) * R_memo(n + 1, t - 2, u, v, p, RPC, memo) + RPC[0] * R_memo(n + 1, t - 1, u, v, p, RPC, memo)
    
    memo[key] = result
    return result

cdef double R(int n, int t, int u, int v, double p, double[:] RPC):
    cdef dict memo = {}
    return R_memo(n, t, u, v, p, RPC, memo)

#cdef double R_memo(int n, int t, int u, int v, double p, double[:] RPC, double[:, :, :, :] memo):
#    cdef double norm_RPC_sq = pow(RPC[0], 2) + pow(RPC[1], 2) + pow(RPC[2], 2)
#    
#    if memo[n,t,u,v] != 0:
#        return memo[n,t,u,v]
#    
#    cdef double result
#    if t < 0 or u < 0 or v < 0:
#        result = 0
#    elif t == 0 and u == 0 and v == 0:
#        result = (-2 * p) ** n * boys(n, p * norm_RPC_sq)
#    elif t == 0 and u == 0:
#        result = (v - 1) * R_memo(n + 1, t, u, v - 2, p, RPC, memo) + RPC[2] * R_memo(n + 1, t, u, v - 1, p, RPC, memo)
#    elif t == 0:
#        result = (u - 1) * R_memo(n + 1, t, u - 2, v, p, RPC, memo) + RPC[1] * R_memo(n + 1, t, u - 1, v, p, RPC, memo)
#    else:
#        result = (t - 1) * R_memo(n + 1, t - 2, u, v, p, RPC, memo) + RPC[0] * R_memo(n + 1, t - 1, u, v, p, RPC, memo)
#    
#    memo[n,t,u,v] = result
#    return result
#
#cdef double R(int n, int t, int u, int v, double p, double[:] RPC):
#    cdef double[:,:,:,:] memo = np.zeros((t+u+v+n+1, t+1, u+1, v+1), dtype=np.float64)
#    return R_memo(n, t, u, v, p, RPC, memo)

cpdef double nuclear_attraction(double a, tuple[long, long, long] lmn1, double[:] A,
                                 double b, tuple[long, long, long] lmn2, double[:] B, double[:] C):
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

cpdef double V(BasisFunction a, BasisFunction b, double[:] RC):
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


cpdef double electron_repulsion(double a, tuple[long, long, long] lmn1, double[:] A,
                         double b, tuple[long, long, long] lmn2, double[:] B,
                         double c, tuple[long, long, long] lmn3, double[:] C,
                         double d, tuple[long, long, long] lmn4, double[:] D):
    """
    カーテシアンガウス関数の電子間反発積分を計算する関数
    """
    cdef double p = a + b
    cdef double q = c + d
    cdef double alpha = p * q / (p + q)
    cdef double[:] RPQ = view.array(shape=(3,), itemsize=sizeof(double), format="d")

    for i in range(3):
        RPQ[i] = (a * A[i] + b * B[i]) / p - (c * C[i] + d * D[i]) / q
    cdef long l1 = lmn1[0], m1=lmn1[1], n1 = lmn1[2]
    cdef long l2 = lmn2[0], m2=lmn2[1], n2 = lmn2[2]
    cdef long l3 = lmn3[0], m3=lmn3[1], n3 = lmn3[2]
    cdef long l4 = lmn4[0], m4=lmn4[1], n4 = lmn4[2]
    cdef double val = 0.0
    cdef long t, u, v, tau, nu, phi
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                for tau in range(l3 + l4 + 1):
                    for nu in range(m3 + m4 + 1):
                        for phi in range(n3 + n4 + 1):
                            val += (
                                (-1) ** (tau + nu + phi)
                                * E(l1, l2, t, a, b, A[0] - B[0])
                                * E(m1, m2, u, a, b, A[1] - B[1])
                                * E(n1, n2, v, a, b, A[2] - B[2])
                                * E(l3, l4, tau, c, d, C[0] - D[0])
                                * E(m3, m4, nu, c, d, C[1] - D[1])
                                * E(n3, n4, phi, c, d, C[2] - D[2])
                                * R(0, t + tau, u + nu, v + phi, alpha, RPQ)
                            )
    return 2 * M_PI**2.5 / (p * q * sqrt(p + q)) * val


cpdef double ERI(BasisFunction a, BasisFunction b, BasisFunction c, BasisFunction d):
    """
    縮約されたカーテシアンガウス関数の電子間反発積分を計算する関数
    """
    cdef double e = 0.0
    cdef long i, j, k, l
    cdef long num_exps = len(a.exps)
    for i in range(num_exps):
        for j in range(num_exps):
            for k in range(num_exps):
                for l in range(num_exps):
                    e += (
                        a.norm[i]
                        * b.norm[j]
                        * c.norm[k]
                        * d.norm[l]
                        * a.coefs[i]
                        * b.coefs[j]
                        * c.coefs[k]
                        * d.coefs[l]
                        * electron_repulsion(
                            a.exps[i],
                            a.lmn,
                            a.origin,
                            b.exps[j],
                            b.lmn,
                            b.origin,
                            c.exps[k],
                            c.lmn,
                            c.origin,
                            d.exps[l],
                            d.lmn,
                            d.origin,
                        )
                    )
    return e
