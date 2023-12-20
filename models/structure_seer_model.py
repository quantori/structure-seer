import typing

import torch
import torch.nn as nn

from utils import DIMENSION, NUM_BOND_TYPES

# Dimension of Embedding Must be a number divisible by 8
EMBEDDING_DIM = 64
# Number of tokens to embed maximal atomic number + 1
NUM_EMBEDDINGS = 36


class StructureSeer(nn.Module):
    """
    An implementation of the Structure Seer model

    Forward pass:
    Input size [(batch_size, DIMENSION), (batch_size, DIMENSION)]
    Output size (batch_size, DIMENSION, DIMENSION, NUM_BOND_TYPES)

    Prediction:
    Input size [(DIMENSION), (DIMENSION)]
    Output size (DIMENSION, DIMENSION, NUM_BOND_TYPES)
    """

    def __init__(
        self,
        n_hidden: int = 256,
        generic_dim: int = 256,
        dimension: int = DIMENSION,
        embedding_dim: int = EMBEDDING_DIM,
        num_embeddings: int = NUM_EMBEDDINGS,
        num_bond_types: int = NUM_BOND_TYPES,
        decoder_nhead: int = 8,
        decoder_nlayers: int = 6,
        device: torch.device = "cpu",
    ):
        super().__init__()

        self.encoder = GCNEncoder(
            dimension=dimension,
            n_hidden=n_hidden,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            generic_dim=generic_dim,
            device=device,
        )

        self.decoder = Decoder(
            dimension=dimension,
            num_bond_types=num_bond_types,
            embedding_dim=embedding_dim,
            nhead=decoder_nhead,
            nlayers=decoder_nlayers,
        )

    def forward(self, atoms_matrix: typing.List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        :param atoms_matrix: a list of elements_vector and shielding_constants_vector batches
                            [(batch_size, DIMENSION), (batch_size, DIMENSION)]
        :return: predicted adjacency matrix batch of size (batch_size, DIMENSION, DIMENSION, NUM_BOND_TYPES)
        """
        src, embedding = self.encoder(atoms_matrix)
        output = self.decoder(src, embedding)

        return output

    def predict(self, atoms_matrix: typing.List[torch.Tensor]) -> torch.Tensor:
        """
        Create a prediction with a trained model
        :param atoms_matrix: a list of elements_vector and shielding_constants_vector [(DIMENSION), (DIMENSION)]
        :return: predicted adjacency matrix of size (DIMENSION, DIMENSION, NUM_BOND_TYPES)
        """
        self.eval()
        atoms_matrix = [
            atoms_matrix[0][None, :],
            atoms_matrix[1][None, :],
        ]
        with torch.no_grad():
            prediction = self.forward(atoms_matrix)

        return prediction[0]


class GCNEncoder(nn.Module):
    """
    Encodes elements_vector and shielding_constants vector to Element-Shielding embedding with the aid of
    generic-matrix GCN approach.
    Input size [(batch_size, DIMENSION), (batch_size, DIMENSION)]
    Output size (batch_size, DIMENSION, EMBEDDING_DIM)
    """

    def __init__(
        self,
        dimension: int = DIMENSION,
        n_hidden: int = 256,
        embedding_dim: int = EMBEDDING_DIM,
        num_embeddings: int = NUM_EMBEDDINGS,
        generic_dim: int = 256,
        device: torch.device = "cpu",
    ):
        super().__init__()
        self.dimension = dimension
        self.embedding_dim = embedding_dim

        self.gcn1 = GraphConv(embedding_dim, n_hidden, device=device)
        self.gcn2 = GraphConv(n_hidden, n_hidden, device=device)
        self.gcn3 = GraphConv(n_hidden, n_hidden, device=device)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()

        self.fc1 = nn.Linear(n_hidden, embedding_dim)

        self.nodes_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.nodes_shielding_fc = nn.Linear(dimension, dimension * embedding_dim)
        self.nodes_scale = nn.Linear(embedding_dim, embedding_dim * dimension)

        self.matrix_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.matrix_shielding_fc = nn.Linear(dimension, dimension * embedding_dim)
        self.matrix_scale = nn.Linear(embedding_dim, embedding_dim * dimension)

        self.generic1 = nn.Linear(embedding_dim, generic_dim)
        self.generic_act1 = nn.Sigmoid()
        self.generic2 = nn.Linear(generic_dim, generic_dim)
        self.generic_act2 = nn.Sigmoid()
        self.generic3 = nn.Linear(generic_dim, dimension)
        self.generic_act3 = nn.Sigmoid()

    def forward(
        self,
        atoms_matrix: typing.List[torch.Tensor],
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        :param atoms_matrix: a list of elements_vector and shielding_constants_vector batches
                            [(batch_size, DIMENSION), (batch_size, DIMENSION)]
        :return: Element-Shielding embedding batch of size (batch_size, DIMENSION, EMBEDDING_DIM)
        """

        # Create nodes embedding
        nodes_embedded = self.nodes_embedding(
            atoms_matrix[0]
        )  # size (batch_size, DIMENSION, EMBEDDING_DIM)

        # Scale shielding
        nodes_weighted_shielding = torch.reshape(
            self.nodes_shielding_fc(atoms_matrix[1]),
            (nodes_embedded.size(dim=0), self.dimension, self.embedding_dim),
        )  # size (batch_size, DIMENSION, EMBEDDING_DIM)

        # Include Shielding Constants by elementwise addition of weighted shielding
        nodes_merged = (
            nodes_embedded + nodes_weighted_shielding
        )  # size (batch_size, DIMENSION, EMBEDDING_DIM)

        # Create embedding for the generic matrix
        matrix_embedded = self.matrix_embedding(
            atoms_matrix[0]
        )  # size (batch_size, DIMENSION, EMBEDDING_DIM)

        # Scale shielding
        matrix_weighted_shielding = torch.reshape(
            self.matrix_shielding_fc(atoms_matrix[1]),
            (matrix_embedded.size(dim=0), self.dimension, self.embedding_dim),
        )  # size (batch_size, DIMENSION, EMBEDDING_DIM)

        # Include Shielding Constants by elementwise addition of weighted shielding
        matrix_merged = matrix_embedded + matrix_weighted_shielding
        # size (batch_size, DIMENSION, EMBEDDING_DIM)

        # Create a Generic Adjacency Matrix from embedded input
        generic_matrix = self.generic_act1(self.generic1(matrix_merged))
        generic_matrix = self.generic_act2(self.generic2(generic_matrix))
        generic_matrix = self.generic_act3(self.generic3(generic_matrix))
        generic_matrix = torch.reshape(
            generic_matrix, (matrix_merged.size(dim=0), self.dimension, self.dimension)
        )  # size (batch_size, DIMENSION, DIMENSION)

        # Symmetrize the generic matrix
        generic_matrix = torch.add(
            torch.transpose(generic_matrix, 1, 2), generic_matrix
        )

        conv1 = self.act1(self.gcn1(x=nodes_merged, adjacency_matrix=generic_matrix))
        conv2 = self.act2(self.gcn2(x=conv1, adjacency_matrix=generic_matrix))
        conv3 = self.act3(
            self.gcn3(x=conv2, adjacency_matrix=generic_matrix)
        )  # size (batch_size, DIMENSION, n_hidden)

        gcn_embedding = self.fc1(conv3)  # size (batch_size, DIMENSION, EMBEDDING_DIM)

        return nodes_merged, gcn_embedding


class Decoder(torch.nn.Module):
    """
    Decodes Element-Shielding embedding to adjacency matrix.
    Input size (batch_size, DIMENSION, EMBEDDING_DIM)
    Output size (batch_size, DIMENSION, DIMENSION, NUM_BOND_TYPES)
    """

    def __init__(
        self,
        dimension: int = DIMENSION,
        num_bond_types: int = NUM_BOND_TYPES,
        embedding_dim: int = EMBEDDING_DIM,
        nhead: int = 8,
        nlayers: int = 6,
    ):
        super().__init__()
        self.dimension = dimension
        self.num_bond_types = num_bond_types

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, nhead=nhead
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=nlayers
        )
        self.resize = nn.Linear(embedding_dim, dimension * num_bond_types)

    def forward(self, src: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        res = self.transformer_decoder(embedding, src)
        scaled_res = self.resize(res)
        adjacency_matrix = torch.reshape(
            scaled_res,
            (scaled_res.shape[0], self.dimension, self.dimension, self.num_bond_types),
        )
        # Symmetrize the output
        adjacency_matrix = torch.add(
            torch.transpose(adjacency_matrix, 1, 2), adjacency_matrix
        )  # size of (batch_size, DIMENSION, DIMENSION, NUM_BOND_TYPES)

        return adjacency_matrix


class GraphConv(nn.Module):
    """
    The graph convolutional operator inspired by `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper;
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dimension: int = DIMENSION,
        device: torch.device = "cpu",
    ):
        super().__init__()

        self.dimension = dimension
        self.linear = nn.Linear(in_features, out_features)
        self.device = device

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        :param x: element-shielding embedding of size (batch_size, DIMENSION, EMBEDDING_DIM)
        :param adjacency_matrix: adjacency matrix with size (batch_size, DIMENSION, DIMENSION, NUM_BOND_TYPES)
        """

        d_norm = torch.diag_embed(torch.pow(adjacency_matrix.sum(dim=-1), -0.5))
        l_norm = torch.bmm(torch.bmm(d_norm, adjacency_matrix), d_norm).to(self.device)
        # Embed nodes and apply linear transformation to x
        x = self.linear(x)

        output = torch.bmm(l_norm, x)

        return output
