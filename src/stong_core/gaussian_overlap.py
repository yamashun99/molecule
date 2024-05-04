import numpy as np
from scipy.special import factorial2 as fact2
from scipy.special import hyp1f1


def custom_fact2(n):
    if n == -1:
        return 1  # -1 の入力に対しては 1 を返す
    else:
        return fact2(n)  # それ以外の場合は通常の二重階乗を計算


class BasisFunction:
    def __init__(self, center, lmn, exps, coefs):
        self.origin = center
        self.lmn = lmn
        self.exps = exps
        self.coefs = coefs
        self.norm = None
        self.normalize()

    def normalize(self):
        l, m, n = self.lmn
        L = l + m + n
        prefactor = (
            custom_fact2(2 * l - 1)
            * custom_fact2(2 * m - 1)
            * custom_fact2(2 * n - 1)
            * np.power(np.pi, 1.5)
            / np.power(2, 2 * L + 3 / 2)
        )
        self.norm = np.power(self.exps, L / 2 + 3 / 4) / np.sqrt(prefactor)
        N = 0
        num_exps = len(self.exps)
        for ia in range(num_exps):
            for ja in range(num_exps):
                N += (
                    self.coefs[ia]
                    * self.coefs[ja]
                    * self.norm[ia]
                    * self.norm[ja]
                    / np.power((self.exps[ia] + self.exps[ja]) / 2, L + 3 / 2)
                )
        N *= prefactor
        N = 1 / np.sqrt(N)
        self.coefs = self.coefs * N


def E(i, j, t, a, b, Qx):
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
        カーテシアンuガウス関数の幅
    b : float
        カーテシアンガウス関数の幅
    Qx : float
        カーテシアンガウス関数の位置の差

    Returns
    -------
    float
        エルミートガウス関数の係数
    """

    p = a + b
    q = a * b / p
    if t < 0 or i + j < t:
        return 0
    elif i == j == t == 0:
        return np.exp(-q * Qx**2)
    elif j == 0:  # decrement i
        return (
            1 / (2 * p) * E(i - 1, j, t - 1, a, b, Qx)
            - q * Qx / a * E(i - 1, j, t, a, b, Qx)
            + (t + 1) * E(i - 1, j, t + 1, a, b, Qx)
        )
    else:  # decrement j
        return (
            1 / (2 * p) * E(i, j - 1, t - 1, a, b, Qx)
            + q * Qx / b * E(i, j - 1, t, a, b, Qx)
            + (t + 1) * E(i, j - 1, t + 1, a, b, Qx)
        )


def overlap(a, lmn1, A, b, lmn2, B):
    """
    カーテシアンガウス関数の重なり積分を計算する関数
    """
    p = a + b
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    S1 = E(l1, l2, 0, a, b, A[0] - B[0])
    S2 = E(m1, m2, 0, a, b, A[1] - B[1])
    S3 = E(n1, n2, 0, a, b, A[2] - B[2])
    return S1 * S2 * S3 * np.power(np.pi / p, 1.5)


def S(a, b):
    """
    縮約されたカーテシアンガウス関数の重なり積分を計算する関数
    """
    s = 0.0
    for i in range(len(a.exps)):
        for j in range(len(b.exps)):
            s += (
                a.norm[i]
                * b.norm[j]
                * a.coefs[i]
                * b.coefs[j]
                * overlap(a.exps[i], a.lmn, a.origin, b.exps[j], b.lmn, b.origin)
            )
    return s


def kinetic(a, lmn1, A, b, lmn2, B):
    """
    カーテシアンガウス関数の重なり積分を計算する関数
    """
    l2, m2, n2 = lmn2
    term0 = b * (2 * (l2 + m2 + n2) + 3) * overlap(a, lmn1, A, b, lmn2, B)
    term1 = (
        -2
        * b**2
        * (
            overlap(a, lmn1, A, b, [l2 + 2, m2, n2], B)
            + overlap(a, lmn1, A, b, [l2, m2 + 2, n2], B)
            + overlap(a, lmn1, A, b, [l2, m2, n2 + 2], B)
        )
    )
    term2 = -0.5 * (
        l2 * (l2 - 1) * overlap(a, lmn1, A, b, [l2 - 2, m2, n2], B)
        + m2 * (m2 - 1) * overlap(a, lmn1, A, b, [l2, m2 - 2, n2], B)
        + n2 * (n2 - 1) * overlap(a, lmn1, A, b, [l2, m2, n2 - 2], B)
    )
    return term0 + term1 + term2


def T(a, b):
    """
    縮約されたカーテシアンガウス関数の運動エネルギー積分を計算する関数
    """
    t = 0.0
    for i in range(len(a.exps)):
        for j in range(len(b.exps)):
            t += (
                a.norm[i]
                * b.norm[j]
                * a.coefs[i]
                * b.coefs[j]
                * kinetic(a.exps[i], a.lmn, a.origin, b.exps[j], b.lmn, b.origin)
            )
    return t


def boys(n, T):
    return hyp1f1(n + 0.5, n + 1.5, -T) / (2 * n + 1)


def R(n, t, u, v, p, RPC):
    if t < 0 or u < 0 or v < 0:
        return 0
    if t == 0 and u == 0 and v == 0:
        return (-2 * p) ** n * boys(n, p * np.linalg.norm(RPC) ** 2)
    elif t == 0 and u == 0:
        return (v - 1) * R(n + 1, t, u, v - 2, p, RPC) + RPC[2] * R(
            n + 1, t, u, v - 1, p, RPC
        )
    elif t == 0:
        return (u - 1) * R(n + 1, t, u - 2, v, p, RPC) + RPC[1] * R(
            n + 1, t, u - 1, v, p, RPC
        )
    else:
        return (t - 1) * R(n + 1, t - 2, u, v, p, RPC) + RPC[0] * R(
            n + 1, t - 1, u, v, p, RPC
        )


def nuclear_attraction(a, lmn1, A, b, lmn2, B, C):
    """
    カーテシアンガウス関数の原子核からのクーロン相互作用を計算する関数
    """
    p = a + b
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    P = (a * A + b * B) / p
    RPC = P - C
    val = 0.0
    for t in range(l1 + l2 + 1):
        for u in range(m1 + m2 + 1):
            for v in range(n1 + n2 + 1):
                val += (
                    E(l1, l2, t, a, b, A[0] - B[0])
                    * E(m1, m2, u, a, b, A[1] - B[1])
                    * E(n1, n2, v, a, b, A[2] - B[2])
                    * R(0, t, u, v, p, RPC)
                )
    return 2 * np.pi / p * val


def V(a, b, RC):
    """
    縮約されたカーテシアンガウス関数の原子核からのクーロン相互作用積分を計算する関数
    """
    v = 0.0
    for i in range(len(a.exps)):
        for j in range(len(b.exps)):
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


def electron_repulsion(a, lmn1, A, b, lmn2, B, c, lmn3, C, d, lmn4, D):
    """
    カーテシアンガウス関数の電子間反発積分を計算する関数
    """
    p = a + b
    q = c + d
    alpha = p * q / (p + q)
    P = (a * A + b * B) / p
    Q = (c * C + d * D) / q
    RPQ = P - Q
    l1, m1, n1 = lmn1
    l2, m2, n2 = lmn2
    l3, m3, n3 = lmn3
    l4, m4, n4 = lmn4
    val = 0.0
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
    return 2 * np.pi**2.5 / (p * q * np.sqrt(p + q)) * val


def ERI(a, b, c, d):
    """
    縮約されたカーテシアンガウス関数の電子間反発積分を計算する関数
    """
    e = 0.0
    for i in range(len(a.exps)):
        for j in range(len(b.exps)):
            for k in range(len(c.exps)):
                for l in range(len(d.exps)):
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
