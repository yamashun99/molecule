from scipy.linalg import eigh
import numpy as np
from matrix import *


class HartreeFock:
    def __init__(self, molecule, basis_set, max_iter=1000, tol=1e-6):
        self.molecule = molecule
        self.basis_set = basis_set
        self.S = None
        self.T = None
        self.V = None
        self.ERI = None
        self.F_up = None
        self.F_dn = None
        self.P_up = None
        self.P_dn = None
        self.energy = None
        self.n_up, self.n_dn = self.molecule.spin_electron_counts()
        self.max_iter = max_iter
        self.tol = tol
        self.val_up = None
        self.val_dn = None
        self.vec_up = None
        self.vec_dn = None

    def build_matrices(self):
        # 各種マトリックスの計算
        matrices = calculate_matrices(self.molecule, self.basis_set)
        self.S = matrices["Smat"]
        self.T = matrices["Tmat"]
        self.V = matrices["Vmat"]
        self.ERI = matrices["ERImat"]

    def scf(self):
        # 自己無撞着フィールド(SCF)プロセス
        self.build_matrices()
        self.P_up = np.ones((self.S.shape[0], self.S.shape[0]))
        self.P_dn = np.zeros((self.S.shape[0], self.S.shape[0]))
        old_energy = 0
        new_energy = 0
        for i in range(self.max_iter):
            self.build_fock_matrix()
            self.val_up, self.vec_up = eigh(self.F_up, self.S)
            self.val_dn, self.vec_dn = eigh(self.F_dn, self.S)
            new_P_up, new_P_dn = self.compute_density_matrix(self.vec_up, self.vec_dn)
            self.P_up = 0.9 * self.P_up + 0.1 * new_P_up
            self.P_dn = 0.9 * self.P_dn + 0.1 * new_P_dn
            e_up, e_dn = self.error()
            old_energy = new_energy
            new_energy = sum(self.val_up[: self.n_up]) + sum(self.val_dn[: self.n_dn])
            if abs(new_energy - old_energy) < self.tol:
                break
            if i == self.max_iter - 1:
                print("Did not converge")

    def error(self):
        e_up = self.F_up @ self.P_up @ self.S - self.S @ self.P_up @ self.F_up
        e_dn = self.F_dn @ self.P_dn @ self.S - self.S @ self.P_dn @ self.F_dn
        return e_up, e_dn

    def build_fock_matrix(self):
        # Fock行列を構築
        hmat = self.T + self.V
        Jmat_up = np.einsum("ijkl,kl->ij", self.ERI, self.P_up) + np.einsum(
            "ijkl,kl->ij", self.ERI, self.P_dn
        )
        Kmat_up = np.einsum("ijkl,jk->il", self.ERI, self.P_up)
        self.F_up = hmat + Jmat_up - Kmat_up
        Jmat_dn = np.einsum("ijkl,kl->ij", self.ERI, self.P_up) + np.einsum(
            "ijkl,kl->ij", self.ERI, self.P_dn
        )
        Kmat_dn = np.einsum("ijkl,jk->il", self.ERI, self.P_dn)
        self.F_dn = hmat + Jmat_dn - Kmat_dn

    def diagonalize_fock_matrix(self):
        # Fock行列を対角化して軌道を決定
        pass

    def compute_density_matrix(self, vec_up, vec_dn):
        self.P_up = vec_up[:, : self.n_up] @ vec_up[:, : self.n_up].T
        self.P_dn = vec_dn[:, : self.n_dn] @ vec_dn[:, : self.n_dn].T
        return self.P_up, self.P_dn

    def check_convergence(self):
        # 収束の確認
        pass

    def calculate_total_energy(self):
        h_mat = self.T + self.V
        one_electron_energy = np.einsum("ij, ij", h_mat, self.P_up) + np.einsum(
            "ij, ij", h_mat, self.P_dn
        )
        self.energy = (
            sum(self.val_up[: self.n_up])
            + sum(self.val_dn[: self.n_dn])
            + one_electron_energy
        ) / 2.0

    def get_results(self):
        # 計算結果の取得
        return {
            "energy": self.energy,
            "P_up": self.P_up,
            "P_dn": self.P_dn,
            "val_up": self.val_up,
            "val_dn": self.val_dn,
            # 他にも必要に応じてデータを追加
        }
