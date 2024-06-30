import sys

sys.path.append("..")
import numpy as np
from one_electron import *
from two_electron import *
from sto_ng import *


def get_Tmat(basis):
    mat = np.zeros((len(basis), len(basis)))
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            if j >= i:
                mat[i, j] = T(a, b)
            else:
                mat[i, j] = mat[j, i]
    return mat


def get_Vmat(basis, R):
    mat = np.zeros((len(basis), len(basis)))
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            mat[i, j] = V(a, b, np.array(R))
    return mat


def get_Smat(basis):
    mat = np.zeros((len(basis), len(basis)))
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            if j >= i:
                mat[i, j] = S(a, b)
            else:
                mat[i, j] = mat[j, i]
    return mat


def create_basis_function(molecule, basis_data):
    basis_functions = []

    # 各原子に対して基底関数を設定
    for atom in molecule.atoms:
        # 原子番号または元素記号から基底データを取得
        element_basis = basis_data[atom.symbol]
        for orbital_type, params in element_basis.items():
            exps = np.array(params["exps"]) * params["zeta"] ** 2
            coefs = np.array(params["coefs"])
            lmn = params["lmn"]
            basis_func = BasisFunction(
                center=atom.position, lmn=lmn, exps=exps, coefs=coefs
            )
            basis_functions.append(basis_func)
    return basis_functions


def calculate_matrices(molecule, basis_data):
    basis_functions = create_basis_function(molecule, basis_data)

    # マトリックスの計算
    Tmat = get_Tmat(basis_functions)
    Vmat = np.zeros_like(Tmat)
    for atom in molecule.atoms:
        Vmat -= get_Vmat(basis_functions, atom.position) * atom.atomic_number
    ERImat = get_ERImat(basis_functions)
    Smat = get_Smat(basis_functions)
    return {"Tmat": Tmat, "Vmat": Vmat, "ERImat": ERImat, "Smat": Smat}
