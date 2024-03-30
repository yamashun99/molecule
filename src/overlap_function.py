import numpy as np


def T(alpha1, alpha2):
    return (
        6
        * np.sqrt(2)
        * alpha1 ** (7 / 4)
        * alpha2 ** (7 / 4)
        / (alpha1 + alpha2) ** (5 / 2)
    )


def V(alpha1, alpha2):
    return (
        -4
        * np.sqrt(2)
        * alpha1 ** (3 / 4)
        * alpha2 ** (3 / 4)
        / (np.sqrt(np.pi) * (alpha1 + alpha2))
    )
