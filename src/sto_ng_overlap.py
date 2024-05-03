from gaussian_overlap import *


def get_Tmat(basis):
    mat = np.zeros((len(basis), len(basis)))
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            mat[i, j] = T(a, b)
    return mat


def get_Vmat(basis, R):
    mat = np.zeros((len(basis), len(basis)))
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            mat[i, j] = V(a, b, np.array(R))
    return mat


def get_ERImat(basis):
    mat = np.zeros((len(basis), len(basis), len(basis), len(basis)))
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            for k, c in enumerate(basis):
                for l, d in enumerate(basis):
                    mat[i, j, k, l] = ERI(a, b, c, d)
    return mat


def get_Smat(basis):
    mat = np.zeros((len(basis), len(basis)))
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            mat[i, j] = S(a, b)
    return mat
