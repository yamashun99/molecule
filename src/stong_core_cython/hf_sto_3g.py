import numpy as np
import gaussian_overlap
import importlib
import h5py
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import copy

from scipy.optimize import minimize

importlib.reload(gaussian_overlap)
from gaussian_overlap import *
from sto_ng_matrix import *


def make_matrix(zeta_1s, zeta_2sp, df, Z):
    param1s = {
        "center": np.array([0.0, 0.0, 0.0]),
        "lmn": (0, 0, 0),
        "exps": np.array(df["exps_1"]) * zeta_1s**2,
        "coefs": np.array(df["coefs_1s"]),
    }
    param2s = {
        "center": np.array([0.0, 0.0, 0.0]),
        "lmn": (0, 0, 0),
        "exps": np.array(df["exps_2"]) * zeta_2sp**2,
        "coefs": np.array(df["coefs_2s"]),
    }
    param2px = {
        "center": np.array([0.0, 0.0, 0.0]),
        "lmn": (1, 0, 0),
        "exps": np.array(df["exps_2"]) * zeta_2sp**2,
        "coefs": np.array(df["coefs_2p"]),
    }
    param2py = {
        "center": np.array([0.0, 0.0, 0.0]),
        "lmn": (0, 1, 0),
        "exps": np.array(df["exps_2"]) * zeta_2sp**2,
        "coefs": np.array(df["coefs_2p"]),
    }
    param2pz = {
        "center": np.array([0.0, 0.0, 0.0]),
        "lmn": (0, 0, 1),
        "exps": np.array(df["exps_2"]) * zeta_2sp**2,
        "coefs": np.array(df["coefs_2p"]),
    }
    base1s = BasisFunction(**param1s)
    base2s = BasisFunction(**param2s)
    base2px = BasisFunction(**param2px)
    base2py = BasisFunction(**param2py)
    base2pz = BasisFunction(**param2pz)
    basis = [base1s, base2s, base2px, base2py, base2pz]
    Tmat = get_Tmat(basis)
    Vmat = -Z * get_Vmat(basis, np.array([0.0, 0.0, 0.0]))
    ERImat = get_ERImat(basis)
    Smat = get_Smat(basis)
    return Tmat, Vmat, ERImat, Smat


def scf(zeta_1s, zeta_2sp, df, n_up, n_dn, Z, max_iter=100000, tol=1e-6):
    c_up = np.zeros((n_up, 5))
    c_dn = np.zeros((n_dn, 5))
    Tmat, Vmat, ERImat, Smat = make_matrix(zeta_1s, zeta_2sp, df, Z)
    hmat = Tmat + Vmat
    new_energy = 0
    for i in range(max_iter):
        Jmat_up = sum(
            np.einsum("ijkl,k, l->ij", ERImat, c_up[i], c_up[i]) for i in range(n_up)
        ) + sum(
            np.einsum("ijkl,k, l->ij", ERImat, c_dn[i], c_dn[i]) for i in range(n_dn)
        )
        Kmat_up = sum(
            np.einsum("ijkl,j, k->il", ERImat, c_up[i], c_up[i]) for i in range(n_up)
        )
        Fmat_up = hmat + Jmat_up - Kmat_up
        Jmat_dn = sum(
            np.einsum("ijkl,k, l->ij", ERImat, c_up[i], c_up[i]) for i in range(n_up)
        ) + sum(
            np.einsum("ijkl,k, l->ij", ERImat, c_dn[i], c_dn[i]) for i in range(n_dn)
        )
        Kmat_dn = sum(
            np.einsum("ijkl,j, k->il", ERImat, c_dn[i], c_dn[i]) for i in range(n_dn)
        )
        Fmat_dn = hmat + Jmat_dn - Kmat_dn
        val_up, vec_up = eigh(Fmat_up, Smat)
        val_dn, vec_dn = eigh(Fmat_dn, Smat)
        c_up = 0.9 * c_up + 0.1 * vec_up[:, :n_up].T
        c_dn = 0.9 * c_dn + 0.1 * vec_dn[:, :n_dn].T
        print(c_up, c_dn)
        old_energy = new_energy
        new_energy = (
            sum(val_up[:n_up])
            + sum(val_dn[:n_dn])
            - 0.5
            * sum(
                np.einsum("ij, i, j", (Jmat_up - Kmat_up), c_up[i], c_up[i])
                for i in range(n_up)
            )
            - 0.5
            * sum(
                np.einsum("ij, i, j", (Jmat_dn - Kmat_dn), c_dn[i], c_dn[i])
                for i in range(n_dn)
            )
        )
        if abs(new_energy - old_energy) < tol:
            break
        if i == max_iter - 1:
            print("Did not converge")
    return new_energy, c_up, c_dn
