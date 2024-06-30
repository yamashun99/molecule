import numpy as np
from scipy.linalg import eigh


class FullCI:
    def __init__(self, hartreefock):
        self.vec = hartreefock.vec
        self.ERI = hartreefock.ERI
        self.T = hartreefock.T
        self.V = hartreefock.V
        self.num_orb = self.vec.shape[1]
        self.ci_matrix = np.zeros((self.num_orb, self.num_orb))
        self.build_ci_matrix()

    def build_ci_matrix(self):
        # 対角要素の計算
        for i in range(self.num_orb):
            self.ci_matrix[i, i] += 2 * np.einsum(
                "ij, i,j", self.T, self.vec[:, i], self.vec[:, i]
            )
            self.ci_matrix[i, i] += 2 * np.einsum(
                "ij, i,j", self.V, self.vec[:, i], self.vec[:, i]
            )
            self.ci_matrix[i, i] += np.einsum(
                "ijkl,i, j, k,l",
                self.ERI,
                self.vec[:, i],
                self.vec[:, i],
                self.vec[:, i],
                self.vec[:, i],
            )

        # 非対角要素の計算
        for i in range(self.num_orb):
            for j in range(i + 1, self.num_orb):
                self.ci_matrix[i, j] += np.einsum(
                    "ijkl,i, j, k,l",
                    self.ERI,
                    self.vec[:, i],
                    self.vec[:, j],
                    self.vec[:, i],
                    self.vec[:, j],
                )
                self.ci_matrix[j, i] = self.ci_matrix[i, j]  # 対称行列なので

    def kernel(self):
        self.val, self.vec = eigh(self.ci_matrix)
        return self.val
