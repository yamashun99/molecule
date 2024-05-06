import numpy as np


def create_basis_data(df, zetas, max_n, max_l):
    basis_data = {}
    for n in range(1, max_n + 1):
        # s orbital
        exps_s = f"exps_{n}"
        coefs_s = f"coefs_{n}s"
        basis_data[f"{n}s"] = {
            "exps": np.array(df[exps_s]),
            "coefs": np.array(df[coefs_s]),
            "lmn": (0, 0, 0),
            "zeta": zetas[n - 1],
        }
        if max_l < 1:
            continue

        if n >= 2:
            # p orbitals
            exps_p = f"exps_{n}"
            coefs_p = f"coefs_{n}p"
            for axis in ["x", "y", "z"]:
                lmn = (
                    1 if axis == "x" else 0,
                    1 if axis == "y" else 0,
                    1 if axis == "z" else 0,
                )
                basis_data[f"{n}p{axis}"] = {
                    "exps": np.array(df[exps_p]),
                    "coefs": np.array(df[coefs_p]),
                    "lmn": lmn,
                    "zeta": zetas[n - 1],
                }
        if max_l < 2:
            continue

        if n >= 3:
            # d orbitals
            exps_d = f"exps_{n}"
            coefs_d = f"coefs_{n}d"
            d_orbitals = ["xy", "xz", "yz", "x2-y2", "z2"]
            lmn_d = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 2, 0), (0, 0, 2)]
            for d, lmn in zip(d_orbitals, lmn_d):
                basis_data[f"{n}d{d}"] = {
                    "exps": np.array(df[exps_d]),
                    "coefs": np.array(df[coefs_d]),
                    "lmn": lmn,
                    "zeta": zetas[n - 1],
                }

    return basis_data
