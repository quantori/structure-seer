import typing

import torch
import numpy
import random

from rdkit import Chem
from rdkit.Chem import rdmolops

import networkx as nx
import matplotlib.pyplot as plt

from .nmredata_utils import read_shielding

from networkx import is_isomorphic, get_edge_attributes, graph_edit_distance
import networkx.algorithms.isomorphism as iso


from .config import DIMENSION, NUM_BOND_TYPES, SHIELDING_MAX, SHIELDING_MIN

# allowable node and edge features
allowable_features = {
    "possible_atomic_num_list": list(range(1, 35)),
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
}


class MolGraph:
    """
    A class to handle molecular graphs using torch.Tensors:
    - Generation of adjacency matrix
    - Creation of the Graph from nodes and adjacency matrix
    - Sorting and shuffling of a graph representation
    - Creation from RdKit Mol object
    - Capability to see if two molecular graphs are equal
    - Visualisation
    """

    def __init__(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __eq__(self, other: "MolGraph"):
        """
        Compares two molecules represented as MolGraph objects
        :param other: other MolGraph object
        :return: True, if both molecules have Same bonds between same atoms, irrespective of their absolute numbering,
                 Otherwise - False
        """

        # Check if Graphs are isomorphic
        g1 = self.to_networkx()
        g2 = other.to_networkx()

        # Check that bonds of the same type are between the same atoms
        # edge_match = iso.numerical_edge_match("bond_type_idx", 1)
        node_match = iso.numerical_node_match("atom_num_idx", 6)
        isomorphic_check = is_isomorphic(
            g1,
            g2,
            # edge_match=edge_match,
            node_match=node_match,
        )

        return isomorphic_check

    @classmethod
    def from_adjacency_matrix(
        cls,
        nodes: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> "MolGraph":
        """
        Create a Molecular Graph from Nodes tensor and given 0-1 normalised adjacency matrix of a proper dimension.
        Nodes tensor as [[atomic_num, shielding]..] or separate tensors - atoms_vector as [atomic_num] and
        shieldings_vector as [shielding] can be used
        :param nodes: [[atomic_num, shielding]..] dtype = torch.float
        :param adjacency_matrix: torch tensor of size (DIMENSION, DIMENSION)
        :return: MolGraph object
        """
        if nodes is None:
            raise ValueError(f"Either Nodes tensor or Atom Matrix should be specified.")

        n = len(nodes)

        if adjacency_matrix is None:
            raise ValueError(f"Adjacency matrix should be Specified")

        if adjacency_matrix.size() != torch.Size(
            [DIMENSION, DIMENSION, NUM_BOND_TYPES]
        ):
            raise ValueError(
                f"Adjacency matrix should be of size {DIMENSION} with bond encoding with size of {NUM_BOND_TYPES}"
            )

        edge_index = [[], []]
        edge_attr = []

        repr_m = torch.argmax(adjacency_matrix, dim=2)

        for i in range(n):
            for j in range(n):
                # Find out the bond type by indexing 1 in the matrix bond
                bond_type = repr_m[i, j]

                if bond_type != 0:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_attr.append(bond_type)

        return cls(
            x=nodes,
            edge_index=torch.tensor(edge_index),
            edge_attr=torch.tensor(edge_attr),
        )

    @classmethod
    def from_mol(
        cls, mol: Chem.Mol, shielding: bool = True, remove_hs: bool = True
    ) -> "MolGraph":
        """
        Converts rdkit mol object to MolGraph object
        geometric package. Strips hydrogens from the Mol object. Ignores Atoms and Bonds Chirality,
        Bonds are represented in edge_attrs as integers:
        1 - Single
        2 - Double
        3 - Triple
        4 - Aromatic
        :param mol: rdkit mol object
        :param shielding: include shielding constants to atoms or not
        :param remove_hs: if H atoms are to be removed
        :return: graph data object with the attributes: x, edge_index, edge_attr
        """
        # Remove hydrogens from the molecule - to simplify graph structure. Ids of atoms remain unchanged.
        if remove_hs:
            mol = rdmolops.RemoveHs(mol)
        # Read atomic numbers and Shielding Constants
        if shielding:
            shielding = read_shielding(mol)

            x = torch.tensor(numpy.array(shielding), dtype=torch.float)

        else:
            out = [0] * len(mol.GetAtoms())
            for atom in mol.GetAtoms():
                element = atom.GetAtomicNum()
                index = atom.GetIdx()
                out[index] = [element, 0]

            x = torch.tensor(out, dtype=torch.float)

        # bonds
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = (
                    allowable_features["possible_bonds"].index(bond.GetBondType()) + 1
                )
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(numpy.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(
                numpy.array(edge_features_list, dtype=float), dtype=torch.float
            )
        else:  # mol has no bonds
            raise ValueError(
                f"Bonds must be specified for the molecule - {mol.GetProp('_Name')}."
            )

        return cls(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @classmethod
    def from_networkx(cls, graph: nx.Graph) -> "MolGraph":
        """
        Converts nx graph to MolGraph object. Assume node indices
        are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
        features, and represent as indices. NB: possible issues with
        recapitulating relative stereochemistry since the edges in the nx
        object are unordered.
        :param graph: nx graph object
        :return: MolGraph object
        """
        # atoms
        atom_features_list = []
        for _, node in graph.nodes(data=True):
            atom_feature = [node["atom_num_idx"], node["shielding_tag_idx"]]
            atom_features_list.append(atom_feature)
        x = torch.tensor(numpy.array(atom_features_list), dtype=torch.float)

        # bonds
        if len(graph.edges()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for i, j, edge in graph.edges(data=True):
                edge_feature = edge["bond_type_idx"]
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(numpy.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(numpy.array(edge_features_list), dtype=torch.long)
        else:  # mol has no bonds
            raise ValueError("Bonds must be specified for the molecule.")

        return cls(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def adjacency_matrix(self, padded: bool = True) -> torch.tensor:
        """
        Creates a 0-1 normalised adjacency matrix with a specified size from a MolGraph object
        representing a molecule. Bond types are represented as follows:
        0 - No Bond
        1 - Single
        2 - Double
        3 - Triple
        4 - Aromatic
        :return: adjacency matrix of a restricted shape as torch tensor
        """
        graph_size = len(self.x)
        bonds_size = len(self.edge_attr)

        if padded:
            adjacency_matrix = torch.zeros(
                DIMENSION, DIMENSION, NUM_BOND_TYPES, dtype=torch.float
            )
        else:
            adjacency_matrix = torch.zeros(
                graph_size, graph_size, NUM_BOND_TYPES, dtype=torch.float
            )

        adjacency_matrix[:, :, 0] = 1

        if graph_size > DIMENSION:
            raise ValueError(f"The graph should have not more than {DIMENSION} nodes")
        if self.edge_attr is None:
            raise ValueError(f"Bond types should be specified in edge_attr of Data")

        for i in range(bonds_size):
            x = self.edge_index[0][i]
            y = self.edge_index[1][i]

            adjacency_matrix[x][y][0] = 0
            adjacency_matrix[y][x][0] = 0

            adjacency_matrix[x][y][self.edge_attr[i].long().item()] = 1
            adjacency_matrix[y][x][self.edge_attr[i].long().item()] = 1

        return adjacency_matrix

    def show(self, size: int = 7, title: str = "molecular graph"):
        """
        Visualises MolGraph as network_x graph object
        :return: None
        """
        g = self.to_networkx()
        node_color = [x[0] for x in self.x]
        label_dict = {idx: round(x[1].item(), 3) for idx, x in enumerate(self.x)}
        width = list(get_edge_attributes(g, "bond_type_idx").values())
        plt.figure(figsize=(size, size))
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx(
            g,
            pos=nx.spring_layout(g, seed=42),
            with_labels=True,
            labels=label_dict,
            node_color=node_color,
            width=width,
            cmap="Set2",
        )
        plt.show()

    def nn_atoms_matrix(
        self, absolute_norm: bool = False, raw: bool = False
    ) -> typing.List[torch.Tensor]:
        """
        Returns a fixed-sized vector with atom features
        :return: [[atomic_num, shielding_const]...0...] size(DIMENSION,2)
        """
        elements_vector = torch.zeros(DIMENSION, dtype=torch.long)
        shielding_constants_vector = torch.zeros(DIMENSION, dtype=torch.float)
        if not raw:
            if absolute_norm:
                s_min = SHIELDING_MIN
                s_max = SHIELDING_MAX
            else:
                s_min = torch.min(self.x, dim=0).values[1]
                s_max = torch.max(self.x, dim=0).values[1]
            for i in range(len(self.x)):
                elements_vector[i] = self.x[i][0]
                # Apply min-max normalisation on shielding constants to convert their values in [0, 1]
                shielding_constants_vector[i] = (self.x[i][1] - s_min) / (s_max - s_min)
        else:
            for i in range(len(self.x)):
                elements_vector[i] = self.x[i][0]
                # Return raw shielding constants
                shielding_constants_vector[i] = self.x[i][1]

        return [elements_vector, shielding_constants_vector]

    def shuffle(self) -> "MolGraph":
        """
        The Functions shuffles order of the nodes and edges in the tensor representation of the graph,
        leaving its structure intact.
        :return: Molgraph object with random order of the nodes
        """
        # Store Old order
        old_atom_order = self.x.tolist()

        # Create index_holder list
        index_holder = [i for i in range(len(old_atom_order))]

        # zip old_atom_order and index holder into container to retain old indexes
        container = list(zip(old_atom_order, index_holder))

        # Shuffle atoms inside the container
        random.shuffle(container)
        new_atom_order, old_ids = zip(*container)
        new_atom_order = torch.tensor(new_atom_order, dtype=torch.float)

        # Modify old adjacency matrix
        old_matrix = self.adjacency_matrix()
        new_matrix = torch.zeros(DIMENSION, DIMENSION, 5, dtype=torch.long)
        n = len(old_atom_order)
        for i in range(n):
            for j in range(n):
                new_matrix[i][j] = old_matrix[old_ids[i]][old_ids[j]]

        return MolGraph.from_adjacency_matrix(
            nodes=new_atom_order, adjacency_matrix=new_matrix
        )

    def sort(self, shielding: bool = False) -> typing.Tuple["MolGraph", torch.Tensor]:
        """
        The Functions sorts the nodes and modifies edges accordingly in the tensor representation of the graph,
        leaving its structure intact. Nodes are sorted according to their atomic num
        :return: Molgraph object with sorted order of the nodes
        """
        # Store Old order
        old_atom_order = self.x.tolist()

        # Create index_holder list
        index_holder = [i for i in range(len(old_atom_order))]

        # zip old_atom_order and index holder into container to retain old indexes
        container = list(zip(old_atom_order, index_holder))

        def zip_sort(e):
            return e[0][0]

        def s_sort(e):
            return e[0][0], e[0][1]

        if shielding:
            f = s_sort
        else:
            f = zip_sort

        # Sort the container
        container.sort(key=f)
        new_atom_order, old_ids = zip(*container)
        new_atom_order = torch.tensor(new_atom_order, dtype=torch.float)

        # Modify old adjacency matrix
        old_matrix = self.adjacency_matrix()
        new_matrix = torch.zeros(
            DIMENSION, DIMENSION, NUM_BOND_TYPES, dtype=torch.float
        )

        n = len(old_atom_order)
        for i in range(n):
            for j in range(n):
                new_matrix[i][j] = old_matrix[old_ids[i]][old_ids[j]]

        # 1 is added to the index to account for "no-atom case" which is 0
        rep_order = [i + 1 for i in old_ids]
        rep_order = rep_order + [0] * (DIMENSION - len(rep_order))

        return MolGraph.from_adjacency_matrix(
            nodes=new_atom_order, adjacency_matrix=new_matrix
        ), torch.tensor(rep_order, dtype=torch.long)

    def to_networkx(self):
        """
        Converts MolGraph object required by the pytorch geometric package to
        network x data object.
        :return: network x object
        """

        return molgraph_to_networkx(self)


def show_adjacency_matrix(
    matrix: torch.Tensor,
    title: str = "Matrix",
    size: int = DIMENSION,
) -> None:
    """
    Show 2-dimensional matrix
    :param matrix: torch.Tensor of 3 dim (DIMENSION, DIMENSION, NUM_BOND_TYPES)
    :param title: Title of the figure
    :param size:
    :return: plot
    """
    matrix_repr = torch.argmax(matrix, dim=2)
    matrix_repr = matrix_repr[:size, :size]

    plt.matshow(matrix_repr.cpu().detach().numpy())
    plt.title(title)
    plt.colorbar()
    plt.show()


def show_bond_probabilities(
    matrix: torch.Tensor,
    title: str = "Matrix",
    size: int = DIMENSION,
) -> None:
    """
    Show 2-dimensional matrix
    :param matrix: torch.Tensor of 3 dim (DIMENSION, DIMENSION, NUM_BOND_TYPES)
    :param title: Title of the figure
    :param size: The size of the matrix to be displayed
    :return: plot
    """

    matrix_repr = torch.zeros(DIMENSION, DIMENSION, dtype=torch.float)
    softmax = torch.nn.functional.softmax(matrix, dim=2)
    matrix_repr[:, :] = 1 - softmax[:, :, 0]
    matrix_repr = matrix_repr[:size, :size]

    plt.matshow(matrix_repr.cpu().detach().numpy())
    plt.title(title)
    plt.colorbar()
    plt.show()


def molgraph_to_networkx(data: MolGraph):
    """
    Converts MolGraph object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices.
    :param data: MolGraph object
    :return: network x object
    """
    G = nx.Graph()

    # Atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]

    for i in range(num_atoms):
        atomic_num_idx, shielding_tag_idx = atom_features[i]
        G.add_node(i, atom_num_idx=atomic_num_idx, shielding_tag_idx=shielding_tag_idx)
        pass

    # Bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(num_bonds):
        begin_idx = int(edge_index[0][j])
        end_idx = int(edge_index[1][j])
        bond_type_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, bond_type_idx=bond_type_idx)

    return G
