from typing import List, Tuple, Optional, Dict, Any
import torch
from torch import nn
from torch.utils.data import DataLoader

from collections import OrderedDict


class NCF(object):

    def __init__(
        self,
        epochs,
        batch_size,
        n_workers,
        step_size,
        gamma,
        n_negatives,
        learning_rate,
        log_every,
        threshold,
        model_type="SimpleNCF",
        seed=None,
    ):

        self.epochs = epochs
        self.batch_size = batch_size
        self.n_workers = n_workers

        self.step_size = step_size
        self.gamma = gamma

        self.n_negatives = n_negatives
        self.learning_rate = learning_rate

        self.log_every = log_every
        self.threshold = threshold

    def train(
        self,
        loader: DataLoader,
        model: nn.Module,
        loss_func: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epochs: int,
        device: torch.device,
        log_every: int = 1000,
    ) -> Tuple[nn.Module, List[float]]:
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
        model.to(device)
        model.train()

        total_loss: float = 0.0
        total_samples: int = 0
        all_losses_list: List[float] = []

        for epoch_i in range(epochs):
            for step_i, batch in enumerate(loader):
                # ----------------------------------------------------------------------
                # ----- Move to device & flatten
                # ----------------------------------------------------------------------
                users = batch["users"].to(device)
                items = batch["items"].to(device)
                true_targets = batch["targets"].to(device)

                # users/items/targets may be (B,) or (B, K+1). Flatten to 1D.
                users = users.view(-1)
                items = items.view(-1)
                true_targets = true_targets.view(-1).to(torch.float32)

                # ----------------------------------------------------------------------
                # ----- Forward pass
                # ----------------------------------------------------------------------
                # Model should return shape (N,) or (N,1); we flatten to (N,)
                pred_targets = model(users, items).view(-1)

                # loss_func is expected to have reduction="none"; shape (N,)
                loss = loss_func(pred_targets, true_targets)

                # ----------------------------------------------------------------------
                # ----- Backward pass
                # ----------------------------------------------------------------------
                # clears old gradients from previous iteration
                optimizer.zero_grad()

                # Manually calculate Mean for Backpropagation
                # The optimizer needs a single scalar to minimize.
                loss_scalar = loss.mean()
                loss_scalar.backward()

                # param update: uses the gradients in param.grads to update the parameters
                optimizer.step()

                # ----------------------------------------------------------------------
                # ----- Logging
                # ----------------------------------------------------------------------
                # Sum the vector directly for logging
                # loss.sum() adds up the squared errors of all X users in the batch
                total_loss += loss.sum().item()
                total_samples += true_targets.numel()

                if (step_i + 1) % log_every == 0:
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

            if scheduler is not None:
                scheduler.step()

        return model, all_losses_list

    def evaluate(
        self,
        loader: DataLoader,
        model: nn.Module,
        loss_func: nn.Module,
        device: torch.device,
    ) -> float:
        """
        Evaluation loop for pointwise implicit data.

        Expects the same batch structure as `train_model`:
            - "users", "items", "targets" tensors, possibly (B, K+1) or (B,).

        Returns the average loss per example over the entire loader.
        """

        model.to(device)
        model.eval()  # Important: turns off dropout!

        total_loss: float = 0.0
        total_samples: int = 0

        with torch.no_grad():  # Important: saves memory, no gradients
            for batch in loader:
                # ----------------------------------------------------------------------
                # ----- Move to device & flatten
                # ----------------------------------------------------------------------
                users = batch["users"].to(device)
                items = batch["items"].to(device)
                true_targets = batch["targets"].to(device)

                users = users.view(-1)
                items = items.view(-1)
                true_targets = true_targets.view(-1).to(torch.float32)

                # ----------------------------------------------------------------------
                # ----- Pred & Loss
                # ----------------------------------------------------------------------
                pred_targets = model(users, items)

                loss = loss_func(pred_targets, true_targets)  # (N,)
                total_loss += loss.sum().item()
                total_samples += true_targets.numel()

        return total_loss / max(total_samples, 1)


class SimpleNCF(nn.Module):

    def __init__(self, n_users: int, n_items: int, **kwargs):
        """
        emb_dim = X//2

        item and user embeddings get concatenated, resulting in an embedding with Xd.
        """
        super().__init__()

        layers = kwargs.get("layers")

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

    def __init__(self, n_users: int, n_items: int, **kwargs):
        """
        (NCF original paper - MLP version)
        layers (#3): 64d - 32d - 16d
        (2 hidden layers + output layer / last hidden layer)
        """
        super().__init__()

        dropout = kwargs.get("dropout")
        layers = kwargs.get("layers")

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
