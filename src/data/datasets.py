from typing import Any
from torch.utils.data import Dataset
import torch

from src.data.samplers import NegativeSampler


class OfflineImplicitDataset(Dataset):
    """
    Simple offline implicit dataset:
    Each row is a single (user, item, target) example.

    Intended for evaluation where negatives have been precomputed offline.
    """

    def __init__(self, users, items, targets):
        self.users = users
        self.items = items
        self.targets = targets

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return {
            "users": torch.tensor(int(self.users[idx]), dtype=torch.long),
            "items": torch.tensor(int(self.items[idx]), dtype=torch.long),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


class PointwiseImplicitDataset(Dataset):
    """
    Each index:
      - returns 1 positive (u, i, 1)
      - plus K negatives (u, j, 0) sampled on-the-fly.
    """

    def __init__(
        self,
        users,
        items,
        timestamps,
        negative_sampler: NegativeSampler,
        n_negatives: int,
    ):
        self.users = users
        self.items = items
        self.timestamps = timestamps
        self.negative_sampler = negative_sampler
        self.n_negatives = n_negatives

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Any:
        user = int(self.users[idx])
        item = int(self.items[idx])
        timestamp = int(self.timestamps[idx])

        # 1 positive
        user_ids = [user]
        item_ids = [item]
        targets = [1.0]

        # K negatives
        neg_items = self.negative_sampler.sample(
            user_id=user,
            n_negatives=self.n_negatives,
            current_timestamp=timestamp,  # ignored for now, used later if time-aware
        )

        for j in neg_items:
            user_ids.append(user)
            item_ids.append(int(j))
            targets.append(0.0)

        return {
            "users": torch.tensor(user_ids, dtype=torch.long),
            "items": torch.tensor(item_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.float32),
        }
