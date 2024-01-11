import re

from rdkit import Chem


def read_nmredata_peaks(compound: Chem.Mol) -> list:
    """
    Extracts NMR Signals in ppm from NMREDATA_ASSIGNMENT property of the Mol file into a matrix.
    The signal position in the list corresponds to atom position in the molecule (atom number id in the molecule).
    If signal for atom is not in NMREDATA_ASSIGNMENT sets value to None
    :param compound: RDkit Mol object
    :return: a list: [[atomic_number: int, nmr_shift: float|None]...] where the position of each sub-list
     corresponds to the atom id in the molecule
    """

    try:
        signals = re.findall(r"(.*)(?=\\)", compound.GetProp("NMREDATA_ASSIGNMENT"))
        nmr = [0] * len(list(compound.GetAtoms()))
        out = [0] * len(compound.GetAtoms())

        for signal in signals:
            if signal != "":
                line = signal.split(", ")
                indexes = line[2:]
                for index in indexes:
                    nmr[int(index) - 1] = float(line[1])

        for atom in compound.GetAtoms():
            element = atom.GetAtomicNum()
            index = atom.GetIdx()
            out[index] = [element, nmr[index]]
        return out
    except KeyError:
        raise ValueError(
            f"Compound {compound.GetProp('_Name')} does not have associated NMRE_ASSIGNMENT"
        )


def read_shielding(compound: Chem.Mol) -> list:
    """
    Extracts Shielding constants in ppm from Shielding property of the Mol file into a matrix.
    :param compound: RDkit Mol object
    :return: a list: [[atomic_number: int, shielding_constant: float]...] where the position of each sub-list
     corresponds to the  atom id in the molecule
    """
    try:
        shielding = [float(x) for x in compound.GetProp("Shielding").split("; ")]
        out = [0] * len(compound.GetAtoms())
        for atom in compound.GetAtoms():
            element = atom.GetAtomicNum()
            index = atom.GetIdx()
            out[index] = [element, shielding[index]]
        return out

    except KeyError:
        raise ValueError(
            f"Compound {compound.GetProp('_Name')} does not have associated Shielding"
        )
