import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from math import factorial


class BasisFunction:
    def __init__(self, n, max_l, exps, coefs):
        self.exps = exps
        self.coefs = coefs
        self.principal_quantum_number = n
        self.max_l = max_l

    def params(self):
        return np.concatenate([[self.exps], self.coefs]).flatten()


def params2expscoefs(n, max_l, params):
    num_exps = len(params) // (1 + max_l)
    exps = params[:num_exps]
    coefs = [params[num_exps * i : num_exps * (i + 1)] for i in range(1, n + 1)]
    return (exps, coefs)


def chi_n(n, r, orbital_type="s"):
    """
    規格化されたSTO
    C. C. Pye and C. J. Mercer,
    "On the Least-Squares Fitting of Slater-Type Orbitals with Gaussians:
    Reproduction of the STO-NG Fits Using Microsoft Excel and Maple,"
    J. Chem. Educ. 89, 1405 (2012).
    """
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


def phi(n, alpha, r):
    coef = (2 ** (4 * n - 1) * alpha ** (2 * n + 1) / np.pi**3) ** 0.25
    return coef * r ** (n - 1) * np.exp(-alpha * r**2)


def sto_ng(a, r, n):
    sum_phi = 0
    for i in range(len(a.exps)):
        exp = a.exps[i]
        coefs = a.coefs[n - 1][i]
        sum_phi += coefs * phi(n, exp, r)
    return sum_phi


class StoNg:
    def __init__(self, principal_quantum_number, max_l):
        self.principal_quantum_number = principal_quantum_number
        self.max_l = max_l
        self.constraints = [
            {"type": "eq", "fun": self.constraint, "args": ["s"]},
        ]
        if principal_quantum_number > 1:
            self.constraints.append(
                {"type": "eq", "fun": self.constraint, "args": ["p"]}
            )
        if principal_quantum_number > 2:
            self.constraints.append(
                {"type": "eq", "fun": self.constraint, "args": ["d"]}
            )

    def integrand(self, r, a):
        n = self.principal_quantum_number
        terms = [np.abs(sto_ng(a, r, 1) - chi_n(n, r, "s")) ** 2 * 4 * np.pi * r**2]
        if n > 1:
            terms.append(
                np.abs(sto_ng(a, r, 2) - chi_n(n, r, "p")) ** 2 * 4 * np.pi * r**2 / 3
            )
        if n > 2:
            terms.append(
                np.abs(sto_ng(a, r, 3) - chi_n(n, r, "d")) ** 2 * np.pi * r**2 * 4 / 15
            )
        return sum(terms)

    def objective_function(self, params):
        exps, coefs = params2expscoefs(
            self.principal_quantum_number, self.max_l, params
        )
        a = BasisFunction(self.principal_quantum_number, self.max_l, exps, coefs)
        integral, error = quad(self.integrand, 0, np.inf, args=(a,))
        return integral

    def constraint(self, params, orbital_type):
        exps, coefs = params2expscoefs(
            self.principal_quantum_number, self.max_l, params
        )
        a = BasisFunction(self.principal_quantum_number, self.max_l, exps, coefs)
        if orbital_type == "s":
            integral, error = quad(
                lambda r: sto_ng(a, r, 1) ** 2 * 4 * np.pi * r**2, 0, np.inf
            )
        elif orbital_type == "p":
            integral, error = quad(
                lambda r: sto_ng(a, r, 2) ** 2 * 4 * np.pi * r**2 / 3, 0, np.inf
            )
        elif orbital_type == "d":
            integral, error = quad(
                lambda r: sto_ng(a, r, 3) ** 2 * 4 * np.pi * r**2 / 15, 0, np.inf
            )
        return integral - 1


def optimize_sto_ng(a):
    params = a.params()
    bounds = [(0, 10)] * len(a.exps) + [(-2, 2)] * len(a.exps) * a.max_l
    sto_ng = StoNg(a.principal_quantum_number, a.max_l)

    result = minimize(
        sto_ng.objective_function,
        params,
        bounds=bounds,
        constraints=sto_ng.constraints,
        method="SLSQP",
    )
    return result
