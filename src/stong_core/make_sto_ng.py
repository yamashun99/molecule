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


def chi_n(n, r, orbital_type="s"):
    zeta = 1
    if orbital_type == "p":
        return np.sqrt(3) * chi_n(n, r, "s")
    elif orbital_type == "d":
        return np.sqrt(15) * chi_n(n, r, "s")
    elif orbital_type == "f":
        return np.sqrt(105) * chi_n(n, r, "s")
    else:
        return (
            np.sqrt(zeta ** (2 * n + 1) * 2 ** (2 * n - 1) / (np.pi * factorial(2 * n)))
            * r ** (n - 1)
            * np.exp(-zeta * r)
        )


def phi_s(alpha, r):
    return (2 * alpha / np.pi) ** (3 / 4) * np.exp(-alpha * r**2)


def phi_p(alpha, r):
    return (128 * alpha**5 / np.pi**3) ** 0.25 * r * np.exp(-alpha * r**2)


def phi_d(alpha, r):
    return (2048 * alpha**7 / np.pi**3) ** 0.25 * r**2 * np.exp(-alpha * r**2)


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


def sto_ng_d(a, r):
    sum_phi = 0
    for i in range(len(a.exps)):
        exp = a.exps[i]
        coefs = a.coefs[2][i]
        sum_phi += coefs * phi_d(exp, r)
    return sum_phi


class StoNg:
    def __init__(self, principal_quantum_number):
        self.principal_quantum_number = principal_quantum_number
        self.constraints = [
            {"type": "eq", "fun": self.constraint, "args": ["s"]},
        ]
        if principal_quantum_number > 1:
            self.constraints.append(
                {"type": "eq", "fun": self.constraint, "args": ["p"]}
            )

    def integrand(self, r, a):
        if self.principal_quantum_number == 1:
            return np.abs(sto_ng_s(a, r) - chi_n(1, r, "s")) ** 2 * 4 * np.pi * r**2
        elif self.principal_quantum_number == 2:
            return (
                np.abs(sto_ng_s(a, r) - chi_n(2, r, "s")) ** 2 * 4 * np.pi * r**2
                + np.abs(sto_ng_p(a, r) - chi_n(2, r, "p")) ** 2 * 4 * np.pi * r**2 / 3
            )
        elif self.principal_quantum_number == 3:
            return (
                np.abs(sto_ng_s(a, r) - chi_n(3, r, "s")) ** 2 * 4 * np.pi * r**2
                + np.abs(sto_ng_p(a, r) - chi_n(3, r, "p")) ** 2 * 4 * np.pi * r**2 / 3
                + np.abs(sto_ng_d(a, r) - chi_n(3, r, "d")) ** 2
                * 4
                * np.pi
                * r**2
                * 4
                / 15
            )

    def objective_function(self, params):
        exps, coefs = params2expscoefs(self.principal_quantum_number, params)
        a = BasisFunction(self.principal_quantum_number, exps, coefs)
        integral, error = quad(self.integrand, 0, np.inf, args=(a,))
        return integral

    def constraint(self, params, orbital_type="s"):
        exps, coefs = params2expscoefs(self.principal_quantum_number, params)
        a = BasisFunction(self.principal_quantum_number, exps, coefs)
        if orbital_type == "s":
            integral, error = quad(
                lambda r: sto_ng_s(a, r) ** 2 * 4 * np.pi * r**2, 0, np.inf
            )
        elif orbital_type == "p":
            integral, error = quad(
                lambda r: sto_ng_p(a, r) ** 2 * 4 * np.pi * r**2 / 3, 0, np.inf
            )
        elif orbital_type == "d":
            integral, error = quad(
                lambda r: sto_ng_d(a, r) ** 2 * 4 * np.pi * r**2 * 4 / 15, 0, np.inf
            )
        return integral - 1


def optimize_sto_ng(a):
    params = a.params()
    bounds = [(0, 10)] * len(a.exps) + [(-1, 1)] * len(
        a.exps
    ) * a.principal_quantum_number  # パラメータの範囲
    sto_ng = StoNg(a.principal_quantum_number)

    result = minimize(
        sto_ng.objective_function,
        params,
        bounds=bounds,
        constraints=sto_ng.constraints,
        method="SLSQP",
    )
    return result
