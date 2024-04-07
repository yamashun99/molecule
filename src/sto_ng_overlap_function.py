import numpy as np

from gaussian_overlap_function import *


class Sto3g:
    @staticmethod
    def kinetic(cs, alphas, betas, RA, RB):
        return cs @ GaussianOverlapFunction.T(alphas, betas, RA, RB) @ cs

    @staticmethod
    def norm(cs, alphas, betas, RA, RB):
        return (
            cs @ GaussianOverlapFunction.S(alphas[:, None], betas[None, :], RA, RB) @ cs
        )

    @staticmethod
    def potential(cs, alphas, betas, RA, RB, RC):
        return (
            cs
            @ GaussianOverlapFunction.V(
                alphas,
                betas,
                RA,
                RB,
                RC,
            )
            @ cs
        )

    @staticmethod
    def ee_coulomb(cs, alphas, betas, gammas, deltas, RA, RB, RC, RD):
        U_mat = GaussianOverlapFunction.U(
            alphas,
            betas,
            gammas,
            deltas,
            RA,
            RB,
            RC,
            RD,
        )

        return cs @ ((cs @ U_mat) @ cs) @ cs
        # Uc = np.tensordot(U_mat, cs, axes=(1, 0))
        # cUc = np.tensordot(cs, Uc, axes=(0, 2))
        # return cs @ cUc @ cs
