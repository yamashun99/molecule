# cython: language_level=3

from libc.math cimport exp, sqrt, M_PI
import numpy as np
cimport numpy as np
import cython
from scipy.special.cython_special cimport hyp1f1
from cython cimport view
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
include "utils.pxi"

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, :, :, :] get_ERImat(list basis):
    cdef int num_basis = len(basis)
    cdef double[:, :, :, :] mat = np.zeros((num_basis, num_basis, num_basis, num_basis))
    cdef int i, j, k, l
    for i in range(num_basis):
        for j in range(num_basis):
            for k in range(num_basis):
                for l in range(num_basis):
                    if j >= i and l >= k:
                        mat[i, j, k, l] = ERI(basis[i], basis[j], basis[k], basis[l])
                    elif j < i and l >= k:
                        mat[i, j, k, l] = mat[j, i, k, l]
                    elif j >= i and l < k:
                        mat[i, j, k, l] = mat[i, j, l, k]
                    else:
                        mat[i, j, k, l] = mat[j, i, l, k]
    return mat



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double electron_repulsion(double a, long[:] lmn1, double[:] A,
                         double b, long[:] lmn2, double[:] B,
                         double c, long[:] lmn3, double[:] C,
                         double d, long[:] lmn4, double[:] D) nogil:
    """
    カーテシアンガウス関数の電子間反発積分を計算する関数
    """
    cdef double p = a + b
    cdef double q = c + d
    cdef double alpha = p * q / (p + q)
    cdef double[3] RPQ
    cdef long i
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

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double ERI(object a, object b, object c, object d, int num_threads=4):
    """
    縮約されたカーテシアンガウス関数の電子間反発積分を計算する関数
    """
    cdef double e = 0.0, temp_e
    cdef long i, j, k, l, t, index
    cdef int thread_id
    cdef long num_exps = len(a.exps)
    cdef double[:] norms_a = a.norm, norms_b = b.norm, norms_c = c.norm, norms_d = d.norm
    cdef double[:] coefs_a = a.coefs, coefs_b = b.coefs, coefs_c = c.coefs, coefs_d = d.coefs
    cdef double[:] exps_a = a.exps, exps_b = b.exps, exps_c = c.exps, exps_d = d.exps
    cdef long[:] lmn_a = a.lmn, lmn_b = b.lmn, lmn_c = c.lmn, lmn_d = d.lmn
    cdef double[:] origin_a = a.origin, origin_b = b.origin, origin_c = c.origin, origin_d = d.origin
    cdef long num_iterations = num_exps**4

    cdef double* partial_sums = <double*>malloc(num_threads * sizeof(double))
    for t in range(num_threads):
        partial_sums[t] = 0.0

    # 並列化対象のループ
    for index in prange(num_iterations, nogil=True, num_threads=num_threads, schedule='static'):
        i = index // (num_exps ** 3)
        j = (index % (num_exps ** 3)) // (num_exps ** 2)
        k = (index % (num_exps ** 2)) // num_exps
        l = index % num_exps
        thread_id = cython.parallel.threadid()
        partial_sums[thread_id] += (
            norms_a[i]
            * norms_b[j]
            * norms_c[k]
            * norms_d[l]
            * coefs_a[i]
            * coefs_b[j]
            * coefs_c[k]
            * coefs_d[l]
            * electron_repulsion(
                exps_a[i],
                lmn_a,
                origin_a,
                exps_b[j],
                lmn_b,
                origin_b,
                exps_c[k],
                lmn_c,
                origin_c,
                exps_d[l],
                lmn_d,
                origin_d,
            )
        )

    for t in range(num_threads):
        e += partial_sums[t]
    
    free(partial_sums)

    return e
