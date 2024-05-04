import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt


class GaussianApproximator1s:
    def __init__(self, initial_params, zeta):
        self.n_gaussians = len(initial_params) // 2
        self.zeta = zeta
        self.params = initial_params
        self.bounds = [(0, 10), (-1, 1)] * self.n_gaussians  # パラメータの範囲

    @staticmethod
    def phi_1s(zeta_1, r):
        return (zeta_1**3 / np.pi) ** 0.5 * np.exp(-zeta_1 * r)

    @staticmethod
    def phi_g1s(alpha, r):
        return (2 * alpha / np.pi) ** (3 / 4) * np.exp(-alpha * r**2)

    def phi_approx_g1s(self, params, r):
        sum_phi = 0
        for i in range(self.n_gaussians):
            alpha = params[2 * i]
            c1s = params[2 * i + 1]
            sum_phi += c1s * self.phi_g1s(alpha, r)
        return sum_phi

    def integrand(self, r, params):
        return (
            np.abs(self.phi_approx_g1s(params, r) - self.phi_1s(self.zeta, r)) ** 2
            * 4
            * np.pi
            * r**2
        )

    def objective_function(self, params):
        integral, error = quad(self.integrand, 0, np.inf, args=(params,))
        return integral

    def normalization_constraint_1s(self, params):
        integral, error = quad(
            lambda r: self.phi_approx_g1s(params, r) ** 2 * 4 * np.pi * r**2, 0, np.inf
        )
        return integral - 1

    def optimize(self):
        constraints = [
            {"type": "eq", "fun": self.normalization_constraint_1s},
        ]
        result = minimize(
            self.objective_function,
            self.params,
            bounds=self.bounds,
            constraints=constraints,
        )
        return result


class GaussianApproximator2s2p:
    def __init__(self, initial_params, zeta):
        self.n_gaussians = len(initial_params) // 3
        self.zeta = zeta
        self.params = initial_params
        self.bounds = [(0, 10), (-1, 1), (-1, 1)] * self.n_gaussians  # パラメータの範囲

    @staticmethod
    def phi_2s(zeta_2, r):
        return (zeta_2**5 / (3 * np.pi)) ** 0.5 * r * np.exp(-zeta_2 * r)

    @staticmethod
    def phi_2p(zeta_2, r, theta=0):  # thetaのデフォルト値を0に設定
        return (zeta_2**5 / np.pi) ** 0.5 * r * np.exp(-zeta_2 * r) * np.cos(theta)

    @staticmethod
    def phi_g1s(alpha, r):
        return (2 * alpha / np.pi) ** (3 / 4) * np.exp(-alpha * r**2)

    @staticmethod
    def phi_g2p(alpha, r):
        return (128 * alpha**5 / np.pi**3) ** (1 / 4) * r * np.exp(-alpha * r**2)

    def phi_approx_g1s(self, params, r):
        sum_phi = 0
        for i in range(self.n_gaussians):
            alpha = params[3 * i]
            c1s = params[3 * i + 1]
            sum_phi += c1s * self.phi_g1s(alpha, r)
        return sum_phi

    def phi_approx_g2p(self, params, r):
        sum_phi = 0
        for i in range(self.n_gaussians):
            alpha = params[3 * i]
            c2p = params[3 * i + 2]
            sum_phi += c2p * self.phi_g2p(alpha, r)
        return sum_phi

    def integrand(self, r, params):
        return (
            np.abs(self.phi_approx_g1s(params, r) - self.phi_2s(self.zeta, r)) ** 2
            * 4
            * np.pi
            * r**2
            + np.abs(self.phi_approx_g2p(params, r) - self.phi_2p(self.zeta, r)) ** 2
            * 4
            * np.pi
            * r**2
            / 3
        )

    def objective_function(self, params):
        integral, error = quad(self.integrand, 0, np.inf, args=(params,))
        return integral

    def normalization_constraint_2s(self, params):
        integral, error = quad(
            lambda r: self.phi_approx_g1s(params, r) ** 2 * 4 * np.pi * r**2, 0, np.inf
        )
        return integral - 1

    def normalization_constraint_2p(self, params):
        integral, error = quad(
            lambda r: self.phi_approx_g2p(params, r) ** 2 * 4 * np.pi * r**2 / 3,
            0,
            np.inf,
        )
        return integral - 1

    def optimize(self):
        constraints = [
            {"type": "eq", "fun": self.normalization_constraint_2s},
            {"type": "eq", "fun": self.normalization_constraint_2p},
        ]
        result = minimize(
            self.objective_function,
            self.params,
            bounds=self.bounds,
            constraints=constraints,
        )
        return result
