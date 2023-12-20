import torch

from models.structure_seer_model import StructureSeer
from utils.data_utils import read_sdf_compounds
from utils.metrics import excess_bonds, heatmap_similarity, wrong_bonds
from utils.molgraph import MolGraph, show_adjacency_matrix, show_bond_probabilities

"""
Illustrates model capabilities on a set of pre-defined examples
"""

# Create Model Instance
structure_seer = StructureSeer()

# Load weights, trained on QM9 Dataset
structure_seer.encoder.load_state_dict(
    torch.load(
        "../weights/structure_seer/qm9/qm9_structure_seer_encoder.weights",
        map_location="cpu",
    )
)
structure_seer.decoder.load_state_dict(
    torch.load(
        "../weights/structure_seer/qm9/qm9_structure_seer_decoder.weights",
        map_location="cpu",
    )
)

qm9_compounds = read_sdf_compounds(
    "../example_datasets/illustrative_predictions/9_qm9_structures_illustrative.sdf"
)

for compound in qm9_compounds:
    g = MolGraph.from_mol(compound).sort(shielding=True)[0]
    n = g.x.size(dim=0)
    atoms_matrix = g.nn_atoms_matrix(absolute_norm=True)
    target_adjacency_matrix = g.adjacency_matrix()

    predicted_matrix = structure_seer.predict(atoms_matrix)

    bond_metrics = wrong_bonds(
        prediction=predicted_matrix[None, :, :, :],
        target=target_adjacency_matrix[None, :, :, :],
    )
    fr_acc = excess_bonds(
        prediction=predicted_matrix[None, :, :, :],
        target=target_adjacency_matrix[None, :, :, :],
    )
    psnr = heatmap_similarity(
        prediction=predicted_matrix[None, :, :, :],
        target=target_adjacency_matrix[None, :, :, :],
    )

    print(f"Id: {compound.GetProp('_Name')}")
    print(f" Wrong bonds, positions: {bond_metrics['pos_wrong']}")
    print(f" Wrong bonds: {bond_metrics['wrong']}")
    print(f"Heatmap similarity: {psnr}")
    print(f" Fragment accuracy: {1 - fr_acc}")

    show_adjacency_matrix(matrix=predicted_matrix, title="Prediction", size=n)
    show_bond_probabilities(matrix=predicted_matrix, title="Heatmap", size=n)
    show_adjacency_matrix(matrix=target_adjacency_matrix, title="Target", size=n)

    pred_g = MolGraph.from_adjacency_matrix(
        nodes=g.x, adjacency_matrix=predicted_matrix
    )
    g.show()
    pred_g.show()
