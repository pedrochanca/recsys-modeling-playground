from typing import List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader

from collections import OrderedDict


class NCFModel(object):

    def __init__(
        self,
        n_users: int,
        n_items: int,
        epochs,
        step_size,
        gamma,
        learning_rate,
        log_every,
        threshold,
        layers: List[int] = None,
        dropout: float = None,
        seed: int = 42,
        model_type: str = "SimpleNCF",
        device: str = "cpu",
    ):

        self.epochs = epochs

        self.step_size = step_size
        self.gamma = gamma

        self.learning_rate = learning_rate

        self.log_every = log_every
        self.threshold = threshold

        self.layers = layers
        self.dropout = dropout

        self.device = device

        if model_type == "SimpleNCF":
            self.model = SimpleNCF(
                n_users=n_users, n_items=n_items, layers=self.layers
            ).to(self.device)
        elif model_type == "DeepNCF":
            self.model = DeepNCF(
                n_users=n_users,
                n_items=n_items,
                layers=self.layers,
                dropout=self.dropout,
            ).to(self.device)
        else:
            raise ValueError(f"Model type '{model_type}' not found in code.")

        self.loss_func = nn.MSELoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Every `step_size` (epoch) calls to scheduler.step(), multiply the learning
        # rate by `gamma`
        # By default, Adam has a learning rate of 0.001
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_size, gamma=self.gamma
        )

    def train(self, loader: DataLoader) -> Tuple[nn.Module, List[float]]:
        """
        Train loop for pointwise implicit data with on-the-fly negatives.

        Expects each batch from `loader` to be a dict:
        - "users":  LongTensor of shape (B,) or (B, K+1)
        - "items":  LongTensor of shape (B,) or (B, K+1)
        - "targets": FloatTensor of shape (B,) or (B, K+1) with values in {0,1}

        The function flattens these to 1D before feeding them to the model:
        users_flat  : (N,)
        items_flat  : (N,)
        targets_flat : (N,)

        epochs: # nb. of times we go through the train set

        model.train():
            - Puts the model into "training mode"

            - It changes how some layers behave:
                1. Dropout layers (nn.Dropout)
                    - In train() mode: randomly zero out some activations (adds noise,
                    regularizes).
                    - In eval() mode: no dropout, they pass everything through (but
                    scaled appropriately during training).

                2. BatchNorm layers (nn.BatchNorm1d, nn.BatchNorm2d, etc.)
                    (it fixes the "internal covariate shift" problem)
                    - In train() mode: use the current batch's mean/variance and update
                    running stats.
                    - In eval() mode: use the stored running mean/variance (fixed
                    statistics).
        """
        self.model.to(self.device)
        self.model.train()

        total_loss: float = 0.0
        total_samples: int = 0
        all_losses_list: List[float] = []

        for epoch_i in range(self.epochs):
            for step_i, batch in enumerate(loader):
                # ----------------------------------------------------------------------
                # ----- Move to device & flatten
                # ----------------------------------------------------------------------
                users = batch["users"].to(self.device)
                items = batch["items"].to(self.device)
                true_targets = batch["targets"].to(self.device)

                # users/items/targets may be (B,) or (B, K+1). Flatten to 1D.
                users = users.view(-1)
                items = items.view(-1)
                true_targets = true_targets.view(-1).to(torch.float32)

                # ----------------------------------------------------------------------
                # ----- Forward pass
                # ----------------------------------------------------------------------
                # Model should return shape (N,) or (N,1); we flatten to (N,)
                pred_targets = self.model(users, items).view(-1)

                # loss_func is expected to have reduction="none"; shape (N,)
                loss = self.loss_func(pred_targets, true_targets)

                # ----------------------------------------------------------------------
                # ----- Backward pass
                # ----------------------------------------------------------------------
                # clears old gradients from previous iteration
                self.optimizer.zero_grad()

                # Manually calculate Mean for Backpropagation
                # The optimizer needs a single scalar to minimize.
                loss_scalar = loss.mean()
                loss_scalar.backward()

                # param update: uses the gradients in param.grads to update the parameters
                self.optimizer.step()

                # ----------------------------------------------------------------------
                # ----- Logging
                # ----------------------------------------------------------------------
                # Sum the vector directly for logging
                # loss.sum() adds up the squared errors of all X users in the batch
                total_loss += loss.sum().item()
                total_samples += true_targets.numel()

                if (step_i + 1) % self.log_every == 0:
                    avg_loss = total_loss / max(total_samples, 1)
                    print(
                        "Epoch: {} | Step: {} | Loss: {}".format(
                            epoch_i, step_i + 1, avg_loss
                        )
                    )
                    all_losses_list.append(avg_loss)

                    # reset accumulators
                    total_loss = 0
                    total_samples = 0

            if self.scheduler is not None:
                self.scheduler.step()

        return all_losses_list

    def evaluate(
        self,
        loader: DataLoader,
    ) -> float:
        """
        Evaluation loop for pointwise implicit data.

        Expects the same batch structure as `train_model`:
            - "users", "items", "targets" tensors, possibly (B, K+1) or (B,).

        Returns the average loss per example over the entire loader.
        """

        self.model.to(self.device)
        self.model.eval()  # Important: turns off dropout!

        total_loss: float = 0.0
        total_samples: int = 0

        with torch.no_grad():  # Important: saves memory, no gradients
            for batch in loader:
                # ----------------------------------------------------------------------
                # ----- Move to device & flatten
                # ----------------------------------------------------------------------
                users = batch["users"].to(self.device)
                items = batch["items"].to(self.device)
                true_targets = batch["targets"].to(self.device)

                users = users.view(-1)
                items = items.view(-1)
                true_targets = true_targets.view(-1).to(torch.float32)

                # ----------------------------------------------------------------------
                # ----- Pred & Loss
                # ----------------------------------------------------------------------
                pred_targets = self.model(users, items)

                loss = self.loss_func(pred_targets, true_targets)  # (N,)
                total_loss += loss.sum().item()
                total_samples += true_targets.numel()

        return total_loss / max(total_samples, 1)


class SimpleNCF(nn.Module):

    def __init__(self, n_users: int, n_items: int, layers: List[int]):
        """
        emb_dim = X//2

        item and user embeddings get concatenated, resulting in an embedding with Xd.
        """
        super().__init__()

        layers = layers

        # learnable parameters - user and item embedding matrices
        # user embedding matrix size = n_users x layers[0] // 2
        # item embedding matrix size = n_items x layers[0] // 2
        self.user_embedding = nn.Embedding(n_users, layers[0] // 2)
        self.item_embedding = nn.Embedding(n_items, layers[0] // 2)

        # single linear layer
        self.fc = nn.Linear(layers[0], 1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Zero hidden layers.

        All it does: for the Xd concatenated embedding, it outputs a single value that
        passes through a linear layer.
        """
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        x = torch.cat([u_emb, i_emb], dim=1)
        output = self.fc(x)

        return output


class DeepNCF(nn.Module):
    """
    nn.Module - when we call the class, it automatically executes the forward function
    """

    def __init__(self, n_users: int, n_items: int, dropout: float, layers: List[int]):
        """
        (NCF original paper - MLP version)
        layers (#3): 64d - 32d - 16d
        (2 hidden layers + output layer / last hidden layer)
        """
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, layers[0] // 2)
        self.item_embedding = nn.Embedding(n_items, layers[0] // 2)

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    # Layer 1
                    ("lin_1", nn.Linear(layers[0], layers[1])),
                    ("relu_1", nn.ReLU()),
                    ("drop_1", nn.Dropout(dropout)),
                    # Layer 2
                    ("lin_2", nn.Linear(layers[1], layers[2])),
                    ("relu_2", nn.ReLU()),
                    ("drop_2", nn.Dropout(dropout)),
                    # Output Layer
                    ("lin_3", nn.Linear(layers[2], 1)),
                ]
            )
        )

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Look up
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        # concat
        x = torch.cat([u_emb, i_emb], dim=1)

        # pass through the tower
        output = self.fc(x)

        return output
