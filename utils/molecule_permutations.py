import torch
from rdkit.Chem import rdMolDescriptors

from rdkit import Chem
import itertools


def generate_molecular_formula(compound) -> str:
    """
    Generates molecular formula for a compound as a string
    :param compound: RDKti mol object
    :return: molecular formula
    """
    formula = Chem.rdMolDescriptors.CalcMolFormula(compound)
    return formula


def generate_shielding_permutations(nodes) -> list:
    """
    Generates a series of possible shielding constants permutations
    :param nodes: nodes vector of a graph
    :return: possible shielding permutations
    """
    possible_shieldings = []
    atom_count = {
        "C": [],
        "N": [],
        "O": [],
        "F": [],
    }

    possible_permutations = {
        "C": [],
        "N": [],
        "O": [],
        "F": [],
    }

    atom_count["C"] = [e[1] for e in nodes if e[0] == 6]
    atom_count["N"] = [e[1] for e in nodes if e[0] == 7]
    atom_count["O"] = [e[1] for e in nodes if e[0] == 8]
    atom_count["F"] = [e[1] for e in nodes if e[0] == 9]

    for key in atom_count.keys():
        shielding = atom_count[key]
        perms = list(itertools.permutations(shielding))
        possible_permutations[key] = perms

    for c_permutation in possible_permutations["C"]:
        for n_permutation in possible_permutations["N"]:
            for o_permutation in possible_permutations["O"]:
                for f_permutation in possible_permutations["F"]:
                    synthetic_shielding = (
                        list(c_permutation)
                        + list(n_permutation)
                        + list(o_permutation)
                        + list(f_permutation)
                    )
                    possible_shieldings.append(synthetic_shielding)

    return possible_shieldings


def generate_adjacency_matrix_permutations(
    elements, permutations, initial_matrix
) -> torch.Tensor:
    """
    Generates a series of possible adjacency matrix permutations
    :param elements: elements vector
    :param permutations: possible shielding vetor permutations
    :param initial_matrix: reference adjacency matrix
    :return: torch.Tensor with all possible permutations of a reference adjacency matrix
    """
    permuted_matrices = []

    for permutation in permutations:
        el_count = len(elements)

        # Create index_holder list
        index_holder = [i for i in range(el_count)]
        permutation_nodes = list(zip(elements, permutation))

        # zip old_atom_order and index holder into container to retain old indexes
        container = list(zip(permutation_nodes, index_holder))

        def s_sort(e):
            return e[0][0], e[0][1]

        container.sort(key=s_sort)
        _, old_ids = zip(*container)

        # Modify old adjacency matrix
        new_matrix = initial_matrix.clone()

        for i in range(el_count):
            for o in range(el_count):
                new_matrix[i][o] = initial_matrix[old_ids[i], old_ids[o]]

        permuted_matrices.append(new_matrix)

    return torch.stack(permuted_matrices)
