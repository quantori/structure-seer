import logging
import os

from rdkit import Chem

from utils import (
    is_successful_orca_run,
    nmr_shielding_from_out_file,
    orca_output_file_check,
    read_sdf_compounds,
)

"""
This script can be used to extract calculated
shielding constants from .out files and append them to corresponding structures in the .sd file
"""

PATH_TO_SD = "../data/structures.sdf"
INPUT_FOLDER = "../data"
CALC_TYPE = "NMR"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

compounds = read_sdf_compounds(PATH_TO_SD)

input_dir = os.listdir(INPUT_FOLDER)
compound_nmrs = dict()

# Check if folder contains corresponding .out file and ORCA terminated normally
for folder in input_dir:
    if orca_output_file_check(
        path=INPUT_FOLDER, compound_id=folder, calc_type=CALC_TYPE
    ):
        logging.info(f"{input_dir.index(folder)} out of {len(input_dir)} processed")
        # Parse NMR Shielding constants from .out file
        nmr = nmr_shielding_from_out_file(
            path_to_out_file=f"{INPUT_FOLDER}/{folder}/{folder}_{CALC_TYPE}.out"
        )
        compound_nmrs[folder] = "; ".join([str(x) for x in nmr])

n = len(compound_nmrs.keys())
with Chem.SDWriter(f"{INPUT_FOLDER}/{n}_qm9_structures_HF-3c_shielding.sdf") as w:
    for compound in compounds:
        name = compound.GetProp("_Name")
        if name in compound_nmrs.keys():
            compound.SetProp("Shielding", str(compound_nmrs[name]))
            logging.info(f"Writing compound {compounds.index(compound)} of {n}")
            w.write(compound)
