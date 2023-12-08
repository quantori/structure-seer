from utils import read_sdf_compounds, generate_input_file, batch
from tqdm import tqdm

"""
Generates a directory with folders containing corresponding ORCA input file
for each compound from .sd file specified
"""

# Indicate path to the .sd file, containing compounds
IN_BATCHES = True
BATCH_SIZE = 45000
PATH_TO_STRUCTURES_SD = "../data"
INPUT_FOLDER_PATH = "./Input_folder_path"
CALCULATION_TYPE = "NMR"

total_compounds = read_sdf_compounds(path=PATH_TO_STRUCTURES_SD)

if IN_BATCHES:
    batches = []
    for batch in batch(total_compounds, BATCH_SIZE):
        batches.append(batch[1])
    for i in range(len(batches)):
        print(f"Preparing batch {i+1}")
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
