import logging

from tqdm import tqdm

from utils import batch, generate_input_file, read_sdf_compounds

"""
Generates a directory with folders containing corresponding ORCA input file
for each compound from .sd file specified
"""

# Indicate path to the .sd file, containing compounds
IN_BATCHES = True
BATCH_SIZE = 45000
PATH_TO_STRUCTURES_SD = "./example_datasets/demo_compounds_pubchem.sdf"
INPUT_FOLDER_PATH = "./Input_folder_path"
CALCULATION_TYPE = "NMR"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

total_compounds = read_sdf_compounds(path=PATH_TO_STRUCTURES_SD)

if IN_BATCHES:
    batches = []
    for single_batch in batch(total_compounds, BATCH_SIZE):
        batches.append(single_batch[1])
    for i in range(len(batches)):
        logging.info(f"Preparing batch {i+1}")
        for j in tqdm(range(len(batches[i]))):
            compound = batches[i][j]
            generate_input_file(
                compound=compound,
                calc_type=CALCULATION_TYPE,
                save_path=f"{INPUT_FOLDER_PATH}/{len(batches[i])}_{CALCULATION_TYPE}_batch_{i+1}",
            )

else:
    for compound in total_compounds:
        generate_input_file(
            compound=compound, calc_type=CALCULATION_TYPE, save_path=INPUT_FOLDER_PATH
        )
