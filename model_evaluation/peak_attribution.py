from utils import read_sdf_compounds, MolGraph
import torch
from tqdm import tqdm
from statistics import mean
from models.structure_seer_model import StructureSeer
from utils.molecule_permutations import (
    generate_adjacency_matrix_permutations,
    generate_shielding_permutations,
)
"""
Attributes peaks of the given structure based on the model's predictions
by evaluating and scoring all possible atom permutations
"""
PERMUTATION_LIMIT = 5000
TOP = 1
device = torch.device("cpu")

# Create Model Instance
structure_seer = StructureSeer()

# Load weights, trained on QM9 Dataset
structure_seer.encoder.load_state_dict(
    torch.load(
        f"../weights/structure_seer/qm9/qm9_structure_seer_encoder.weights",
        map_location="cpu",
    )
)
structure_seer.decoder.load_state_dict(
    torch.load(
        f"../weights/structure_seer/qm9/qm9_structure_seer_decoder.weights",
        map_location="cpu",
    )
)

qm9_compounds = read_sdf_compounds("../example_datasets/demo_compounds_qm9.sdf")

structure_count = len(qm9_compounds)
accuracy = 0

processed_samples = 0
average_max_shielding_error = []
average_max_relative_shielding_error = []
average_wrong_atoms_attributed = []
average_mean_attribution_error = []
average_mean_relative_attribution_error = []

for u in tqdm(range(len(qm9_compounds))):
    compound = qm9_compounds[u]
    input_g = MolGraph.from_mol(compound).sort(shielding=True)[0]
    atoms_matrix = input_g.nn_atoms_matrix(absolute_norm=True)
    target = input_g.adjacency_matrix()

    prediction = structure_seer.predict(atoms_matrix)
    permutations = generate_shielding_permutations(input_g.x)
    elements = [e[0] for e in input_g.x]
    perm_count = len(permutations)

    # Shuffle input graph

    if perm_count <= PERMUTATION_LIMIT:
        # print(f"Processing compound {u + 1} out of {structure_count}")
        # Generate all permutations of the input graph adjacency matrix
        target = torch.argmax(target, dim=-1)
        input_permutations = generate_adjacency_matrix_permutations(
            elements, permutations, target
        )

        processed_samples += 1
        prediction = torch.argmax(prediction, dim=-1)
        prediction_tensor = torch.stack([prediction] * perm_count, dim=0)

        # scores = score(prediction_tensor, input_permutations)
        diff = torch.abs(prediction_tensor - input_permutations)
        s_diff = torch.reshape(
            diff, (diff.size(dim=0), diff.size(dim=1) * diff.size(dim=2))
        )
        summs = torch.sum(s_diff, dim=-1)

        _, idx = summs.sort()

        top_idx = idx[:TOP]

        s_diff_tensor = torch.ones(len(top_idx))
        max_absolute_error_tensor = torch.ones(len(top_idx))
        max_relative_error_tensor = torch.ones(len(top_idx))
        average_absolute_error_tensor = torch.ones(len(top_idx))
        average_relative_error_tensor = torch.ones(len(top_idx))
        for h, index in enumerate(top_idx):
            elements = [s[0] for s in input_g.x]
            carbon_count = elements.count(6)
            candidate_shielding = torch.tensor(permutations[index])[:carbon_count]
            target_shielding = torch.tensor([s[1] for s in input_g.x])[:carbon_count]

            s_diff = torch.count_nonzero(target_shielding - candidate_shielding).item()

            s_diff_tensor[h] = s_diff
            error = torch.nn.functional.l1_loss(
                candidate_shielding, target_shielding, reduction="none"
            )
            average_absolute_error_tensor[h] = torch.mean(error)
            relative_error = torch.abs(error / target_shielding)
            average_relative_error_tensor[h] = torch.mean(relative_error)
            max_error = torch.max(error)
            max_absolute_error_tensor[h] = max_error
            relative_max_error = torch.max(relative_error)
            max_relative_error_tensor[h] = relative_max_error

        best_id = torch.argmin(s_diff_tensor)

        average_max_shielding_error.append(max_absolute_error_tensor[best_id].item())
        average_max_relative_shielding_error.append(
            max_relative_error_tensor[best_id].item()
        )
        average_mean_attribution_error.append(
            average_absolute_error_tensor[best_id].item()
        )
        average_mean_relative_attribution_error.append(
            average_relative_error_tensor[best_id].item()
        )
        average_wrong_atoms_attributed.append(s_diff_tensor[best_id].item())

        if s_diff_tensor[best_id] == 0:
            accuracy += 1
    else:
        pass

print(
    f"OVERALL RUN STATISTICS for TOP {TOP} candidates (Carbon atoms only)- total compounds processed {processed_samples}\n"
    f"Permutation Limit {PERMUTATION_LIMIT}"
    f"Average accuracy {accuracy / processed_samples}\n"
    f"Average absolute error in shielding value {mean(average_mean_attribution_error)}\n"
    f"Average relative error in shielding value {mean(average_mean_relative_attribution_error)}\n"
    f"Average Maximal absolute error in value for top candidate {mean(average_max_shielding_error)}\n"
    f"Average Maximal relative error in value for top candidate {mean(average_max_relative_shielding_error)}\n"
    f"Average atoms with wrongly attributed shielding {mean(average_wrong_atoms_attributed)}"
)
