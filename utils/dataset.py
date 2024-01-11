import logging

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .data_utils import read_sdf_compounds
from .molgraph import MolGraph

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


class MolecularDataset(Dataset):
    def __init__(
        self,
        path_to_sdf_file: "str",
        shielding_sort: bool = True,
        absolute_norm: bool = True,
    ):
        """
        Returns atoms_matrix, adjacency matrix and an id for a structure from .sdf file.
        :param path_to_sdf_file: Path to .sdf file with structures
        :param shielding_sort: If True - returns the element-shielding sorted representation of the structure
        :param absolute_norm: If True - values of isotropic shielding constants are normalised to (-1000, 1000) range
        """
        compounds = read_sdf_compounds(path_to_sdf_file)
        logging.info("Creating Dataset...")
        structures = [
            MolGraph.from_mol(mol=x, shielding=True, remove_hs=True) for x in compounds
        ]

        ids = [x.GetProp("_Name") for x in compounds]

        if shielding_sort:
            sorted_structures = []
            logging.info("Sorting structures...")
            for i in tqdm(range(len(structures))):
                structure = structures[i]
                sorted_structures.append(structure.sort(shielding=True)[0])

            self.structures = sorted_structures
        else:
            self.structures = structures

        self.absolute_norm = absolute_norm
        self.ids = ids

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        structure = self.structures[idx]
        s_id = self.ids[idx]

        sample = {
            "atoms_matrix": structure.nn_atoms_matrix(absolute_norm=self.absolute_norm),
            "adjacency_matrix": structure.adjacency_matrix(),
            "id": s_id,
        }

        return sample
