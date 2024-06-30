import sys

sys.path.append("../../cython")
sys.path.append("../../../cython")
from scipy.linalg import eigh
import numpy as np
from matrix import *
from scipy.ndimage import gaussian_filter1d
from two_electron import *
import pandas as pd


class UnrestrictedHartreeFock:
    def __init__(self, molecule, max_iter=1000, tol=1e-6):
        self.molecule = molecule
        self.S = None
        self.T = None
        self.V = None
        self.ERI = None
        self.F_up = None
        self.F_dn = None
        self.energy = None
        self.n_up, self.n_dn = self.molecule.spin_electron_counts()
        self.max_iter = max_iter
        self.tol = tol
        self.val_up = None
        self.val_dn = None
        self.vec_up = None
        self.vec_dn = None
        self.basis_labels = self.get_basis_labels()

        # 各種マトリックスの計算
        matrices = calculate_matrices(self.molecule)
        self.S = matrices["Smat"]
        self.T = matrices["Tmat"]
        self.V = matrices["Vmat"]
        self.ERI = matrices["ERImat"]

        self.P_up = np.ones((self.S.shape[0], self.S.shape[0]))
        self.P_dn = np.zeros((self.S.shape[0], self.S.shape[0]))

    def get_basis_labels(self):
        # 基底関数のラベルを取得
        labels = []
        for atom_basis in self.molecule.basis.values():
            for orbital in atom_basis.keys():
                labels.append(orbital)
        return labels

    def set_density_matrix(self, P_up, P_dn):
        self.P_up = P_up
        self.P_dn = P_dn

    def scf(self):
        # 自己無撞着フィールド(SCF)プロセス
        print(f"Start SCF for {self.molecule}")
        old_energy = 0
        new_energy = 0
        for i in range(self.max_iter):
            self.build_fock_matrix()
            self.val_up, self.vec_up = eigh(self.F_up, self.S)
            self.val_dn, self.vec_dn = eigh(self.F_dn, self.S)
            new_P_up, new_P_dn = self.compute_density_matrix(self.vec_up, self.vec_dn)
            self.P_up = 0.5 * self.P_up + 0.5 * new_P_up
            self.P_dn = 0.5 * self.P_dn + 0.5 * new_P_dn
            e_up, e_dn = self.error()
            old_energy = new_energy
            new_energy = sum(self.val_up[: self.n_up]) + sum(self.val_dn[: self.n_dn])
            if abs(new_energy - old_energy) < self.tol and i > 1:
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

    def calculate_pdos_weight(self):
        # PDOS (Partial Density of States) の計算
        pdos_up = {"energy": self.val_up}
        pdos_dn = {"energy": self.val_dn}

        for label in self.basis_labels:
            pdos_up[label] = np.zeros(self.val_up.shape)
            pdos_dn[label] = np.zeros(self.val_dn.shape)

        for i in range(self.vec_up.shape[1]):
            for j, label in enumerate(self.basis_labels):
                pdos_up[label][i] += np.abs(self.vec_up[j, i]) ** 2
        for i in range(self.vec_dn.shape[1]):
            for j, label in enumerate(self.basis_labels):
                pdos_dn[label][i] += np.abs(self.vec_dn[j, i]) ** 2
        pdos_up = pd.DataFrame(pdos_up)
        pdos_dn = pd.DataFrame(pdos_dn)

        return pdos_up, pdos_dn

    def calculate_smoothed_pdos(self, sigma=0.1, bins=100):
        # 平滑化したPDOSの計算
        pdos_up, pdos_dn = self.calculate_pdos_weight()
        energy_min = min(np.min(self.val_up), np.min(self.val_dn)) - 1.0
        energy_max = max(np.max(self.val_up), np.max(self.val_dn)) + 1.0
        energy_grid = np.linspace(energy_min, energy_max, bins)

        smoothed_pdos_up = {"energy": energy_grid}
        smoothed_pdos_dn = {"energy": energy_grid}

        for label in self.basis_labels:
            counts_up, bin_edges = np.histogram(
                pdos_up["energy"],
                bins=energy_grid,
                weights=pdos_up[label],
                density=True,
            )
            counts_dn, _ = np.histogram(
                pdos_dn["energy"],
                bins=energy_grid,
                weights=pdos_dn[label],
                density=True,
            )
            smoothed_counts_up = gaussian_filter1d(counts_up, sigma)
            smoothed_counts_dn = gaussian_filter1d(counts_dn, sigma)
            smoothed_pdos_up[label] = smoothed_counts_up
            smoothed_pdos_dn[label] = smoothed_counts_dn

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed_pdos_up["energy"] = bin_centers
        smoothed_pdos_dn["energy"] = bin_centers
        smoothed_pdos_up = pd.DataFrame(smoothed_pdos_up)
        smoothed_pdos_dn = pd.DataFrame(smoothed_pdos_dn)

        return smoothed_pdos_up, smoothed_pdos_dn

    def calculate_dos(self, sigma=0.1, bins=100):
        # DOSの計算
        levels = np.concatenate((self.val_up, self.val_dn))
        counts, bin_edges = np.histogram(levels, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed_counts = gaussian_filter1d(counts, sigma)

        dos = pd.DataFrame({"energy": bin_centers, "dos": smoothed_counts})

        return dos

    def get_results(self):
        # 計算結果の取得
        return {
            "energy": self.energy,
            "P_up": self.P_up,
            "P_dn": self.P_dn,
            "val_up": self.val_up,
            "val_dn": self.val_dn,
            "vec_up": self.vec_up,
            "vec_dn": self.vec_dn,
            # 他にも必要に応じてデータを追加
        }
