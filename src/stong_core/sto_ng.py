import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from math import factorial


class BasisFunction:
    def __init__(self, principal_quantum_number, exps, coefs):
        self.exps = exps
        self.coefs = coefs
        self.principal_quantum_number = principal_quantum_number

    def params(self):
        return np.concatenate([[self.exps], self.coefs]).flatten()


def params2expscoefs(principal_quantum_number, params):
    num_exps = len(params) // (1 + principal_quantum_number)
    exps = params[:num_exps]
    coefs = [
        params[num_exps * i : num_exps * (i + 1)]
        for i in range(1, principal_quantum_number + 1)
    ]
    return (exps, coefs)


def chi_ns(n, r):
    zeta = 1
    return (
        np.sqrt(zeta ** (2 * n + 1) * 2 ** (2 * n - 1) / (np.pi * factorial(2 * n)))
        * r ** (n - 1)
        * np.exp(-zeta * r)
    )


def chi_np(n, r):
    return np.sqrt(3) * chi_ns(n, r)


def chi_nd(n, r):
    return np.sqrt(15) * chi_ns(n, r)


def chi_nf(n, r):
    return np.sqrt(105) * chi_ns(n, r)


def phi_s(alpha, r):
    return (2 * alpha / np.pi) ** (3 / 4) * np.exp(-alpha * r**2)


def phi_p(alpha, r):
    return (128 * alpha**5 / np.pi**3) ** 0.25 * r * np.exp(-alpha * r**2)


def sto_ng_s(a, r):
    sum_phi = 0
    for i in range(len(a.exps)):
        exp = a.exps[i]
        coefs = a.coefs[0][i]
        sum_phi += coefs * phi_s(exp, r)
    return sum_phi


def sto_ng_p(a, r):
    sum_phi = 0
    for i in range(len(a.exps)):
        exp = a.exps[i]
        coefs = a.coefs[1][i]
        sum_phi += coefs * phi_p(exp, r)
    return sum_phi


class StoNg1s:
    def __init__(self):
        self.constraints = [{"type": "eq", "fun": constraint_s, "args": [1]}]

    def integrand(self, r, a):
        return np.abs(sto_ng_s(a, r) - chi_ns(1, r)) ** 2 * 4 * np.pi * r**2

    def objective_function(self, params):
        exps, coefs = params2expscoefs(1, params)
        a = BasisFunction(1, exps, coefs)
        integral, error = quad(self.integrand, 0, np.inf, args=(a,))
        return integral


class StoNg2s2p:
    def __init__(self):
        self.constraints = [
            {"type": "eq", "fun": constraint_s, "args": [2]},
            {"type": "eq", "fun": constraint_p, "args": [2]},
        ]

    def integrand(self, r, a):
        return (
            np.abs(sto_ng_s(a, r) - chi_ns(2, r)) ** 2 * 4 * np.pi * r**2
            + np.abs(sto_ng_p(a, r) - chi_np(2, r)) ** 2 * 4 * np.pi * r**2 / 3
        )

    def objective_function(self, params):
        exps, coefs = params2expscoefs(2, params)
        a = BasisFunction(2, exps, coefs)
        integral, error = quad(self.integrand, 0, np.inf, args=(a,))
        return integral


def constraint_s(params, principle_quantum_number):
    exps, coefs = params2expscoefs(principle_quantum_number, params)
    a = BasisFunction(principle_quantum_number, exps, coefs)
    integral, error = quad(lambda r: sto_ng_s(a, r) ** 2 * 4 * np.pi * r**2, 0, np.inf)
    return integral - 1


def constraint_p(params, principle_quantum_number):
    exps, coefs = params2expscoefs(principle_quantum_number, params)
    a = BasisFunction(principle_quantum_number, exps, coefs)
    integral, error = quad(
        lambda r: sto_ng_p(a, r) ** 2 * 4 * np.pi * r**2 / 3, 0, np.inf
    )
    return integral - 1


def optimize_sto_ng(a):
    params = a.params()
    bounds = [(0, 10)] * len(a.exps) + [(-1, 1)] * len(
        a.exps
    ) * a.principal_quantum_number  # パラメータの範囲
    if a.principal_quantum_number == 1:
        sto_ng = StoNg1s()
    elif a.principal_quantum_number == 2:
        sto_ng = StoNg2s2p()

    result = minimize(
        sto_ng.objective_function,
        params,
        bounds=bounds,
        constraints=sto_ng.constraints,
        method="SLSQP",
    )
    return result
