import numpy as np
from scipy.special import erf


def F(t):
    return 0.5 * np.sqrt(np.pi / (t + 1e-10)) * erf(np.sqrt(t + 1e-10))


def T(alpha, beta, RA, RB):
    return (
        alpha
        * beta
        / (alpha + beta)
        * (6 - 4 * alpha * beta / (alpha + beta) * np.linalg.norm(RA - RB) ** 2)
        * (np.pi / (alpha + beta)) ** (3 / 2)
        * np.exp(-alpha * beta * np.linalg.norm(RA - RB) ** 2 / (alpha + beta))
        / 2
    )


def V(alpha, beta, RA, RB, RC):
    RP = (alpha * RA + beta * RB) / (alpha + beta)
    return (
        -2
        * np.pi
        / (alpha + beta)
        * np.exp(-alpha * beta * np.linalg.norm(RA - RB) ** 2 / (alpha + beta))
        * F((alpha + beta) * np.linalg.norm(RP - RC) ** 2)
    )


def U(alpha, beta, gamma, delta, RA, RB, RC, RD):
    RP = (alpha * RA + gamma * RC) / (alpha + gamma)
    RQ = (beta * RB + delta * RD) / (beta + delta)
    return (
        2
        * np.pi ** (5 / 2)
        / ((alpha + gamma) * (beta + delta) * np.sqrt(alpha + gamma + beta + delta))
        # * np.exp(
        #    -alpha * gamma * np.linalg.norm(RA - RC) ** 2 / (alpha + gamma)
        #    - beta * delta * np.linalg.norm(RB - RD) ** 2 / (beta + delta)
        # )
        # * F(
        #    (alpha + gamma)
        #    * (beta + delta)
        #    * np.linalg.norm(RP - RQ) ** 2
        #    / (alpha + gamma + beta + delta)
        # )
    )


def S(alpha, beta, RA, RB):
    return (np.pi / (alpha + beta)) ** (3 / 2) * np.exp(
        -alpha * beta / (alpha + beta) * np.linalg.norm(RA - RB) ** 2
    )


# def T(alpha1, alpha2):
#    return (
#        6
#        * np.sqrt(2)
#        * alpha1 ** (7 / 4)
#        * alpha2 ** (7 / 4)
#        / (alpha1 + alpha2) ** (5 / 2)
#    )


# def V(alpha1, alpha2):
#    return (
#        -4
#        * np.sqrt(2)
#        * alpha1 ** (3 / 4)
#        * alpha2 ** (3 / 4)
#        / (np.sqrt(np.pi) * (alpha1 + alpha2))
#    )
#
