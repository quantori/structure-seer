import logging
import os

from rdkit import Chem
from rdkit.Chem import rdmolfiles

from utils import orca_output_file_check

"""
Merges optimised structures from .out files to an .sdf file
"""

INPUT_FOLDER = "../data"
CALC_TYPE = "OPT"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

input_dir = os.listdir(INPUT_FOLDER)
compound_folders = []

# Check if folder contains corresponding .out file and ORCA terminated normally
for folder in input_dir:
    try:
        if orca_output_file_check(
            path=INPUT_FOLDER, compound_id=folder, calc_type=CALC_TYPE
        ):
            compound_folders.append(folder)
            logging.info(f"{input_dir.index(folder)} out of {len(input_dir)} processed")
        else:
            logging.info(f"{folder} - Calculations failed")
    except Exception as e:
        logging.info(f"{folder}: {e}")

# Collect optimised geometries as .xyz files and move them to a single .sd file
compounds = []
n = len(compound_folders)
logging.info("Parsing Geometries ...")
for folder in compound_folders:
    logging.info(
        f"Parsing compound {compound_folders.index(folder)} of {len(compound_folders)}"
    )
    compound = rdmolfiles.MolFromXYZFile(
        f"{INPUT_FOLDER}/{folder}/{folder}_{CALC_TYPE}.xyz"
    )
    compound.SetProp("_Name", f"{folder}")
    compounds.append(compound)

filename = INPUT_FOLDER.split("/")[-1]
batch_n = filename.split("_")[-2]

logging.info("Writing to .SDF file ...")
with Chem.SDWriter(
    f"{INPUT_FOLDER}/{n}_optimised_structures_from_batch_{batch_n}.sdf"
) as w:
    for compound in compounds:
        logging.info(
            f"Writing compound {compounds.index(compound)} of {len(compounds)}"
        )
        w.write(compound)


logging.info(f"Total folders processed {len(input_dir)}")
logging.info(f"Successfuly computed compounds {len(compound_folders)}")
