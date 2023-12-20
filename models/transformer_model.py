import math
import typing

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils import DIMENSION

# Dimension of Embedding Must be a number divisible by 8
EMBEDDING_DIM = 64
# Number of tokens to embed maximal atomic number + 1
NUM_EMBEDDINGS = 36


class TEncoder(torch.nn.Module):
    """
    Returns Embedded vector of Atoms with Shielding constants and its transformer-encoded version
    Output size (batch_size, DIMENSION, EMBEDDING_DIM)
    """

    def __init__(
        self,
        dimension: int = DIMENSION,
        num_embeddings: int = NUM_EMBEDDINGS,
        embedding_dim: int = EMBEDDING_DIM,
        n_hidden: int = 256,
        nhead: int = 8,
        sum_layer: bool = True,
    ):
        super(TEncoder, self).__init__()
        self.dimension = dimension
        self.sum_layer = sum_layer
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.shielding_fc = nn.Linear(dimension, dimension * embedding_dim)
        # self.norm = nn.BatchNorm1d(dimension)
        self.transformer = TransformerModel(nhead=nhead, d_hid=n_hidden, nlayers=6)

    def forward(
        self, atoms_matrix: typing.List[torch.Tensor]
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # atoms_vector is a tuple of tensors of size (batch_size, DIMENSION) DIMENSION = 54 by default
        # Embed Atomic Numbers for each atom in the molecule
        embedded = self.embedding(
            atoms_matrix[0]
        )  # size (batch_size, DIMENSION, EMBEDDING_DIM)
        # Scale shielding
        weighted_shielding = torch.reshape(
            self.shielding_fc(atoms_matrix[1]),
            (embedded.size(dim=0), self.dimension, self.embedding_dim),
        )
        # Include Shielding Constants by elementwise addition of weighted shielding
        merged = embedded + weighted_shielding

        output = self.transformer(merged)

        return merged, output


class TransformerModel(nn.Module):
    def __init__(
        self,
        nhead: int,
        d_hid: int,
        nlayers: int,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_type = "Transformer"
        encoder_layers = TransformerEncoderLayer(embedding_dim, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = embedding_dim
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, embedded_src: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Arguments:
            embedded_src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = embedded_src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
