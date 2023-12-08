from utils import is_successful_orca_run, nmr_shielding_from_out_file, read_sdf_compounds
from rdkit import Chem
import os
import logging

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
    if (
        os.path.isdir(f"{INPUT_FOLDER}/{folder}")
        and os.path.isfile(f"{INPUT_FOLDER}/{folder}/{folder}_{CALC_TYPE}.out")
        and is_successful_orca_run(f"{INPUT_FOLDER}/{folder}/{folder}_{CALC_TYPE}.out")
    ):
        logging.info(f"{input_dir.index(folder)} out of {len(input_dir)} processed")
        # Parse NMR Shielding constants from .out file
        print(folder)
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
