from utils import MolecularDataset
import torch

"""
Current script was used to perform characterisation of the datasets
"""

dataset = MolecularDataset(
    "../data/Example.dataset",
    absolute_norm=True,
    shielding_sort=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
)

atom_nums_count = {}
bonds_count = {}
for item in test_loader:
    adj_matrix = item["adjacency_matrix"]

    atoms_matrix = [
        item["atoms_matrix"][0],
        item["atoms_matrix"][1],
    ]

    atoms_num = torch.count_nonzero(atoms_matrix[0], dim=-1).item()
    num_bonds = torch.sum(
        torch.count_nonzero(torch.argmax(adj_matrix, dim=3), dim=1), dim=1
    ).item()

    if atoms_num in atom_nums_count.keys():
        atom_nums_count[atoms_num] += 1
    else:
        atom_nums_count[atoms_num] = 1

    if num_bonds in bonds_count:
        bonds_count[num_bonds] += 1
    else:
        bonds_count[num_bonds] = 1

print(bonds_count)
print(atom_nums_count)
