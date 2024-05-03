import numpy as np

from gaussian_overlap_s import *


class StoNG:
    def __init__(self, cs, alphas):
        self.cs = cs
        self.alphas = alphas

    def norm(self, zetaA, zetaB, RA, RB):
        return (
            self.cs
            @ GaussianOverlapFunction.S(
                zetaA**2 * self.alphas[:, None], zetaB**2 * self.alphas[None, :], RA, RB
            )
            @ self.cs
        )

    def kinetic(self, zetaA, zetaB, RA, RB):
        return (
            self.cs
            @ GaussianOverlapFunction.T(
                zetaA**2 * self.alphas, zetaB**2 * self.alphas, RA, RB
            )
            @ self.cs
        )

    def coulomb_ion(self, zetaA, zetaB, RA, RB, RC):
        return (
            self.cs
            @ GaussianOverlapFunction.V(
                zetaA**2 * self.alphas, zetaB**2 * self.alphas, RA, RB, RC
            )
            @ self.cs
        )

    def coulomb_ee(self, zetaA, zetaB, zetaC, zetaD, RA, RB, RC, RD):
        U_mat = GaussianOverlapFunction.U(
            zetaA**2 * self.alphas,
            zetaB**2 * self.alphas,
            zetaC**2 * self.alphas,
            zetaD**2 * self.alphas,
            RA,
            RB,
            RC,
            RD,
        )

        return self.cs @ ((self.cs @ U_mat) @ self.cs) @ self.cs


class StoNGMatrix:
    def __init__(self, stong, zetas, Rs):
        self.stong = stong
        self.zetas = zetas
        self.Rs = Rs

    def get_smat(self):
        M = len(self.zetas)
        s_mat = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                s_mat[i, j] = self.stong.norm(
                    self.zetas[i], self.zetas[j], self.Rs[i], self.Rs[j]
                )
        return s_mat

    def get_kinetic_mat(self):
        M = len(self.zetas)
        kinetic_mat = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                kinetic_mat[i, j] = self.stong.kinetic(
                    self.zetas[i], self.zetas[j], self.Rs[i], self.Rs[j]
                )
        return kinetic_mat

    def get_electron_ion0_mat(self):
        M = len(self.zetas)
        potential_mat = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                potential_mat[i, j] = self.stong.coulomb_ion(
                    self.zetas[i], self.zetas[j], self.Rs[i], self.Rs[j], self.Rs[0]
                )
        return potential_mat

    def get_electron_ion1_mat(self):
        M = len(self.zetas)
        potential_mat = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                potential_mat[i, j] = self.stong.coulomb_ion(
                    self.zetas[i], self.zetas[j], self.Rs[i], self.Rs[j], self.Rs[1]
                )
        return potential_mat

    def get_electron_ion_mat(self, Zs=[1, 1]):
        return (
            Zs[0] * self.get_electron_ion0_mat() + Zs[1] * self.get_electron_ion1_mat()
        )

    def get_electron_electron_mat(self):
        M = len(self.zetas)
        ee_coulomb_mat = np.zeros((M, M, M, M))
        for i in range(M):
            for j in range(M):
                for k in range(M):
                    for l in range(M):
                        ee_coulomb_mat[i, j, k, l] = self.stong.coulomb_ee(
                            self.zetas[i],
                            self.zetas[j],
                            self.zetas[k],
                            self.zetas[l],
                            self.Rs[i],
                            self.Rs[j],
                            self.Rs[k],
                            self.Rs[l],
                        )
        return ee_coulomb_mat

    def get_h_mat(self, Zs=[1, 1]):
        kinetic_mat = self.get_kinetic_mat()
        electron_ion_mat = self.get_electron_ion_mat(Zs=Zs)
        return kinetic_mat + electron_ion_mat

    def get_J_mat(self, c_sto3g):
        ee_coulomb_mat = self.get_electron_electron_mat()
        ee_coulomb_mat_1 = np.tensordot(c_sto3g, ee_coulomb_mat, axes=(0, 1))
        ee_coulomb_mat_13 = np.tensordot(c_sto3g, ee_coulomb_mat_1, axes=(0, 2))
        return ee_coulomb_mat_13
