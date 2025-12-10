import torch
import torch.nn as nn

from collections import OrderedDict


class SimpleNCF(nn.Module):

    def __init__(self, n_users: int, n_items: int, **kwargs):
        """
        emb_dim = 32

        item and user embeddings get concatenated, resulting in an embedding with 64d.
        """
        super().__init__()

        emb_dim = kwargs.get("emb_dim")

        # learnable parameters - user and item embedding matrices
        # user embedding matrix size = n_users x emb_dim
        # item embedding matrix size = n_items x emb_dim
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # single linear layer: 64 -> 1
        self.fc = nn.Linear(2 * emb_dim, 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Zero hidden layers.

        All it does: for the 64d concatenated embedding, it outputs a single value that
        passes through a linear layer.
        """
        u_emb = self.user_embedding(user_ids)  # size: [batch, 32]
        i_emb = self.item_embedding(item_ids)  # size: [batch, 32]

        x = torch.cat([u_emb, i_emb], dim=1)
        output = self.fc(x)

        return output


class DeepNCF(nn.Module):
    """
    nn.Module - when we call the class, it automatically executes the forward function
    """

    def __init__(self, n_users: int, n_items: int, **kwargs):
        """
        (NCF original paper - MLP version)
        layers (#3): 64d - 32d - 16d
        (2 hidden layers + output layer / last hidden layer)
        """
        super().__init__()

        emb_dim = kwargs.get("emb_dim")
        dropout = kwargs.get("dropout")

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        concat_dim = emb_dim * 2

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    # Layer 1: 64 -> 32
                    ("lin_1", nn.Linear(concat_dim, concat_dim // 2)),
                    ("relu_1", nn.ReLU()),
                    ("drop_1", nn.Dropout(dropout)),
                    # Layer 2: 32 -> 16
                    ("lin_2", nn.Linear(concat_dim // 2, concat_dim // 4)),
                    ("relu_2", nn.ReLU()),
                    ("drop_2", nn.Dropout(dropout)),
                    # Output Layer: 16 -> 1
                    ("lin_3", nn.Linear(concat_dim // 4, 1)),
                ]
            )
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Look up
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        # Concat (Default Size 64)
        x = torch.cat([u_emb, i_emb], dim=1)

        # Pass through the tower
        output = self.fc(x)

        return output
