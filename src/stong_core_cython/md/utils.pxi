# cython: language_level=3

from libc.math cimport exp, sqrt, M_PI
import numpy as np
cimport numpy as np
import cython
from scipy.special.cython_special cimport hyp1f1
from cython cimport view

cdef long custom_fact2(long n):
    cdef long result = 1
    cdef long i
    for i in range(n, 0, -2):
        result *= i
    return result

cdef double E(long i, long j, long t, double a, double b, double Qx):
    if i < 0 or j < 0 or t < 0 or i + j < t:
        return 0

    if i == j == t == 0:
        return exp(-a * b * Qx * Qx / (a + b))
    elif j == 0:  # decrement i
        return (
            1 / (2 * (a + b)) * E(i - 1, j, t - 1, a, b, Qx)
            - a * b * Qx / (a * (a + b)) * E(i - 1, j, t, a, b, Qx)
            + (t + 1) * E(i - 1, j, t + 1, a, b, Qx)
        )
    else:  # decrement j
        return  (
            1 / (2 * (a + b)) * E(i, j - 1, t - 1, a, b, Qx)
            + a * b * Qx / (b * (a + b)) * E(i, j - 1, t, a, b, Qx)
            + (t + 1) * E(i, j - 1, t + 1, a, b, Qx)
        )


cdef double boys(int n, double T):
    return hyp1f1(n + 0.5, n + 1.5, -T) / (2 * n + 1)


cdef double R(int n, int t, int u, int v, double p, double[:] RPC):
    cdef double norm_RPC_sq = RPC[0]**2 + RPC[1]**2 + RPC[2]** 2
    
    cdef double result
    if t < 0 or u < 0 or v < 0:
        return 0
    elif t == 0 and u == 0 and v == 0:
        return (-2 * p) ** n * boys(n, p * norm_RPC_sq)
    elif t == 0 and u == 0:
        return (v - 1) * R(n + 1, t, u, v - 2, p, RPC) + RPC[2] * R(n + 1, t, u, v - 1, p, RPC)
    elif t == 0:
        return (u - 1) * R(n + 1, t, u - 2, v, p, RPC) + RPC[1] * R(n + 1, t, u - 1, v, p, RPC)
    else:
        return (t - 1) * R(n + 1, t - 2, u, v, p, RPC) + RPC[0] * R(n + 1, t - 1, u, v, p, RPC)