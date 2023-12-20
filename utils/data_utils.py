import datetime
import logging
import os
import random
import time
import typing
from datetime import date

import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem

from .config import MAX_N_ATOMS, MAX_PUBCHEM_CID, PERMITTED_ELEMENTS

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def generate_mol_file(compound: Chem.Mol, path: str) -> None:
    """
    Writes RDKit .mol files to a file
    :param compound: RDKit Mol file
    :param path: path to the .mol file
    :return: None
    """
    f = open(path, "w+")
    f.write(Chem.MolToMolBlock(compound))
    f.close()


def create_id_list(path_to_sdf: str) -> list:
    """
    Generates a list with ids of each compound in the .sdf file
    :param path_to_sdf: path to an .sdf file
    :return: List of compounds ids
    """
    compounds = read_sdf_compounds(path=path_to_sdf)
    ids = [
        int(compound.GetProp("_Name").lower().replace("pubchem_cid_", ""))
        for compound in compounds
    ]
    return ids


def read_sdf_compounds(path: str) -> list:
    """
    Reads compounds from an .sdf file
    :param path: path to an .sdf file
    :return: List of RDKit objects for compounds
    """
    compounds = Chem.SDMolSupplier(path, removeHs=False)
    return list(compounds)


def generate_pubchem_sample(
    size: int,
    path: str = "../data/pubchem_raw_data",
    compounds_to_exclude: typing.Optional[typing.List[int]] = None,
) -> str:
    """
    Function to create a Sample Dataset from PubChem records.
    Creates a .mol files with 3d structures of all compounds
    :param path: path to the parent folder
    :param size: Number of molecules in the DataSet
    :param compounds_to_exclude: provide a list of CID of Compounds to exclude in int format
    :return: Creates a directory and fills it with .mol files containing structures
    """

    sample_path = f"{path}/sample_{date.today()}_size_{size}"
    os.makedirs(sample_path, exist_ok=True)

    if compounds_to_exclude is None:
        forbidden_compounds = []
    else:
        forbidden_compounds = compounds_to_exclude

    pubchem_ids = []
    request_start_time = 0

    start = time.time()
    while len(pubchem_ids) < size:
        # Generate random PubChem CID
        pubchem_id = random.randint(1, MAX_PUBCHEM_CID)
        while pubchem_id in pubchem_ids:
            pubchem_id = random.randint(1, MAX_PUBCHEM_CID)

        time_passed_from_old_request = time.time() - request_start_time
        if time_passed_from_old_request > 0.21:
            try:
                # Get SMILES for corresponding Random CID
                request_start_time = time.time()
                smiles = pcp.Compound.from_cid(pubchem_id).isomeric_smiles

                # Create RDKit compound
                compound = Chem.MolFromSmiles(smiles)

                # Add Hydrogens
                compound = Chem.AddHs(compound)
                # Check elements
                elements = set([atom.GetAtomicNum() for atom in compound.GetAtoms()])
                # Check if compound is suitable
                if (
                    # Check that charge of the molecule is 0
                    (Chem.rdmolops.GetFormalCharge(compound) == 0)
                    # Check that the compound contains only permitted elements
                    and (elements.issubset(PERMITTED_ELEMENTS))
                    # Check that the compound contains not more than permitted amount of atoms
                    and (compound.GetNumAtoms() <= MAX_N_ATOMS)
                    # Check if the compound is in the NMRShiftDB
                    and not (pubchem_id in forbidden_compounds)
                ):
                    # Set Name of the Compound to Pubchem CID
                    compound.SetProp("_Name", f"pubchem_cid_{pubchem_id}")
                    # Compute 3D coordinates of atoms
                    AllChem.EmbedMolecule(compound, randomSeed=0xF00D)
                    # Optimise Geometry using MMFF94 MD Forcefield
                    AllChem.MMFFOptimizeMolecule(compound)
                    # Write processed compound to a .mol file
                    generate_mol_file(
                        compound=compound,
                        path=f"{sample_path}/PubChem_CID_{pubchem_id}.mol",
                    )
                    # Append Ids List
                    pubchem_ids.append(pubchem_id)

            except:
                pass

            # Logging progress
            n = len(pubchem_ids)
            est_time_s = round((size - n) * time_passed_from_old_request)
            if n > 1:
                logging.info(
                    f" {n} out of {size} requested compounds - {round(n/size*100,2)}%"
                    f" - Estimated time: {datetime.timedelta(seconds=est_time_s)}"
                )
        else:
            pass
    end = time.time()

    print(f"Sample Dataset generated in {datetime.timedelta(seconds=round(end-start))}")
    return sample_path


def batch(iterable: list, n: int):
    """
    Creates batches from a given iterable
    :param iterable: the iterable to be divided into batcehs
    :param n: batch size
    :return: i batches of size n
    """
    l = len(iterable)
    i = 0
    for ndx in range(0, l, n):
        i += 1
        yield i, iterable[ndx : min(ndx + n, l)]
