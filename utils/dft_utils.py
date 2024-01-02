import datetime
import logging
import os
import re
import time
import typing

from rdkit import Chem

from .config import DFT_NMR_HEADER, DFT_OPT_HEADER

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def nmr_shielding_from_out_file(path_to_out_file: str) -> typing.List[float]:
    """
    Returns a list containing  corresponding nmr shielding constant from .out file -
    position of the constant corresponds to the atom number in the molecule
    :param path_to_out_file: path to the .out file
    :return: a List with shielding constants
    """
    with open(path_to_out_file, "r") as f:
        lines = f.readlines()
        try:
            shielding_table_start = (
                lines.index("CHEMICAL SHIELDING SUMMARY (ppm)\n") + 6
            )
        except ValueError:
            logging.info("The OUT file does not have any NMR information")
        table_lines = lines[shielding_table_start:]
        nmr = []
        for line in table_lines:
            if line == "\n":
                break
            s = re.findall(r"\d+", line)

            j = line.index(s[1]) - 1
            if line[j] == "-":
                s[1] = "-" + s[1]
            nmr.append(float((s[1] + "." + s[2])))
    return nmr


def is_successful_orca_run(path_to_out_file: str) -> bool:
    """
    Checks if the optimisation was successful according to .out file in the compounds directory.
    :param path_to_out_file: path to the  .out file
    :return: True if Run was Successful, False otherwise
    """
    with open(path_to_out_file, "r") as f:
        last_line = f.readlines()[-2].replace(" ", "")

        return last_line == "****ORCATERMINATEDNORMALLY****\n"


def calculate_with_orca(
    path: str, orca_path: str, calc_type: typing.Optional[str] = "DFT"
) -> None:
    """
    Run ORCA calculation from every folder within the specified directory, if it contains appropriate .inp file
    :param path: path to directory with folders, containing input files
    :param calc_type: suffix of the .out file, optional DFT by default
    :return: Creates a bunch of calculation files in each folder, where appropriate .inp files was located
    """
    # List all directories with input files for calculations
    logging.info("Preparing Computation")
    input_dir = os.listdir(path)
    compound_folders = []
    for folder in input_dir:
        # Check if folder contains corresponding input file
        if os.path.isdir(f"{path}/{folder}") and os.path.isfile(
            f"{path}/{folder}/{folder}.inp"
        ):
            compound_folders.append(folder)

    logging.info(f"Starting computation for {len(compound_folders)} compounds")
    n = len(compound_folders)
    # Start ORCA calculation
    start = time.time()
    for compound in compound_folders:
        try:
            logging.info(
                f"Starting computation for {compound_folders.index(compound)} out of {n} - {compound}"
            )
            job_start = time.time()
            os.system(
                f"{orca_path}/orca {path}/{compound}/{compound}.inp > {path}/{compound}/{compound}_{calc_type}.out"
            )
            job_end = time.time()
            logging.info(
                f"{compound} computed in {round(job_end-job_start, 2)} seconds"
            )
        except Exception as e:
            logging.info(f"Calculations for {compound} failed due to:\n {e}")

    end = time.time()
    logging.info(
        f" {n} compounds computed in {datetime.timedelta(seconds=round(end-start))}"
    )


def generate_input_file(compound: Chem.Mol, save_path: str, calc_type: str) -> None:
    """
    Creates a folder with an ORCA input file in a specified directory. Folder and input file will have the
    same name as RDKit Mol object ("_Name" Prop)
    :param compound: RDKit Mol object
    :param save_path: path to a directory, where the folder with an input file will be created
    :param calc_type: NMR or OPT type of calculation
    :return: None
    """
    # Calculate spin
    spin = calculate_spin_multiplicity(compound)
    # Generate .xyz data
    xyz = "\n".join(Chem.MolToXYZBlock(compound).splitlines()[2:])

    os.makedirs(save_path, exist_ok=True)

    # Add Headers according to selected calculation type
    if calc_type == "OPT":
        inp_file = f"{DFT_OPT_HEADER}\n* xyz 0 {spin}\n{xyz}\n*"
    elif calc_type == "NMR":
        inp_file = f"{DFT_NMR_HEADER}\n* xyz 0 {spin}\n{xyz}\n*"
    else:
        raise ValueError("OPT or NMR types are only supported ")

    compound_folder = f"{save_path}/{compound.GetProp('_Name')}"
    os.mkdir(compound_folder)

    with open(
        f"{compound_folder}/{compound.GetProp('_Name')}_{calc_type}.inp", "w+"
    ) as f:
        try:
            f.write(inp_file)
            f.close()
        except Exception as e:
            logging.info(
                f"Unable to create compound folder with an input file due to {e}"
            )


def calculate_spin_multiplicity(compound: Chem.Mol) -> int:
    """Calculate spin multiplicity of a molecule. The spin multiplicity is calculated
    from the number of free radical electrons using Hund's rule of maximum
      multiplicity defined as 2S + 1 where S is the total electron spin. The
      total spin is 1/2 the number of free radical electrons in a molecule.

      Arguments:
          compound (object): RDKit molecule object.

      Returns:
          int : Spin multiplicity.

    """

    # Calculate spin multiplicity using Hund's rule of maximum multiplicity...
    num_radical_electrons = 0
    for atom in compound.GetAtoms():
        num_radical_electrons += atom.GetNumRadicalElectrons()

    total_electronic_spin = num_radical_electrons / 2
    spin_multiplicity = 2 * total_electronic_spin + 1

    return int(spin_multiplicity)


def orca_output_file_check(path: str, compound_id: str, calc_type: str):
    """
    Checks if the output file exists and calculations terminated normally
    :param path: path to a directory with compound folders
    :param compound_id: the name of the compound (compound folder)
    :param calc_type: type of the calculation
    :return: True if folder exists and calculation terminated normally, else - False
    """
    check = (
        os.path.isdir(f"{path}/{compound_id}")
        and os.path.isfile(f"{path}/{compound_id}/{compound_id}_{calc_type}.out")
        and is_successful_orca_run(
            f"{path}/{compound_id}/{compound_id}_{calc_type}.out"
        )
    )

    return check
