from gaussian_overlap import *


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


def get_ERImat(basis):
    mat = np.zeros((len(basis), len(basis), len(basis), len(basis)))
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            for k, c in enumerate(basis):
                for l, d in enumerate(basis):
                    if j >= i and l >= k:
                        mat[i, j, k, l] = ERI(a, b, c, d)
                    elif j < i and l >= k:
                        mat[i, j, k, l] = mat[j, i, k, l]
                    elif j >= i and l < k:
                        mat[i, j, k, l] = mat[i, j, l, k]
                    else:
                        mat[i, j, k, l] = mat[j, i, l, k]
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
