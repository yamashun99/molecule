import numpy as np


class Atom:
    # 元素記号から原子番号へのマッピング
    symbol_to_atomic_number = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,  # その他の元素も追加可能
    }

    def __init__(self, symbol, position):
        self.symbol = symbol
        self.position = np.array(position)
        # 元素記号から原子番号を自動的に設定
        self.atomic_number = self.symbol_to_atomic_number.get(symbol, None)
        if self.atomic_number is None:
            raise ValueError(f"Unknown element symbol: {symbol}")

    def __repr__(self):
        return f"Atom(symbol={self.symbol}, atomic_number={self.atomic_number}, position={self.position})"


class Molecule:
    def __init__(self, atoms):
        self.atoms = atoms

    def __repr__(self):
        atom_reprs = ", ".join(repr(atom) for atom in self.atoms)
        return f"Molecule(atoms=[{atom_reprs}])"

    def total_electrons(self):
        return sum(atom.atomic_number for atom in self.atoms)

    def spin_electron_counts(self):
        Z = self.total_electrons()
        n_up = Z // 2 + Z % 2  # スピンアップ電子数
        n_dn = Z // 2  # スピンダウン電子数
        return n_up, n_dn
