from utils import generate_input_file, read_sdf_compounds

compound = read_sdf_compounds("example_datasets/demo_compounds_qm9.sdf")[3]

generate_input_file(compound=compound, save_path="test_inputs", calc_type="OPT")
