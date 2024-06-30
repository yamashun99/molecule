import sys

sys.path.append("../../cython")
sys.path.append("../../../cython")
from scipy.linalg import eigh
import numpy as np
from matrix import *
from two_electron import *


class RestrictedHartreeFock:
    def __init__(self, molecule, max_iter=1000, tol=1e-6):
        self.molecule = molecule
        self.S = None
        self.T = None
        self.V = None
        self.ERI = None
        self.F = None
        self.energy = None
        self.n_electrons = molecule.total_electrons()
        self.max_iter = max_iter
        self.tol = tol
        self.val = None
        self.vec = None

        # 各種マトリックスの計算
        matrices = calculate_matrices(self.molecule)
        self.S = matrices["Smat"]
        self.T = matrices["Tmat"]
        self.V = matrices["Vmat"]
        self.ERI = matrices["ERImat"]

        self.P = np.zeros((self.S.shape[0], self.S.shape[0]))

    def set_density_matrix(self, P):
        self.P = P

    def scf(self):
        # 自己無撞着フィールド(SCF)プロセス
        print(f"Start SCF for {self.molecule}")
        old_energy = 0
        new_energy = 0
        for i in range(self.max_iter):
            self.build_fock_matrix()
            self.val, self.vec = eigh(self.F, self.S)
            new_P = self.compute_density_matrix(self.vec)
            self.P = 0.5 * self.P + 0.5 * new_P
            old_energy = new_energy
            new_energy = sum(self.val[: self.n_electrons // 2])
            if abs(new_energy - old_energy) < self.tol:
                break
            if i == self.max_iter - 1:
                print("Did not converge")
        self.calculate_total_energy()

    def build_fock_matrix(self):
        # Fock行列を構築
        hmat = self.T + self.V
        Jmat = np.einsum("ijkl,kl->ij", self.ERI, self.P)
        Kmat = np.einsum("ijkl,jk->il", self.ERI, self.P)
        self.F = hmat + 2 * Jmat - Kmat

    def compute_density_matrix(self, vec):
        P = np.dot(vec[:, : self.n_electrons // 2], vec[:, : self.n_electrons // 2].T)
        return P

    def calculate_total_energy(self):
        h_mat = self.T + self.V
        one_electron_energy = np.einsum("ij, ij", h_mat, self.P)
        self.energy = sum(self.val[: self.n_electrons // 2]) + one_electron_energy

    def get_results(self):
        # 計算結果の取得
        return {
            "energy": self.energy,
            "P": self.P,
            "val": self.val,
            "vec": self.vec,
            # 他にも必要に応じてデータを追加
        }
