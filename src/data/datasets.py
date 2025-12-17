from typing import Any
from torch.utils.data import Dataset
import torch

from src.data.samplers import NegativeSampler


class OfflineImplicitDataset(Dataset):
    """
    Simple offline implicit dataset:
    Each row is a single (user, item, label) example.

    Intended for evaluation where negatives have been precomputed offline.
    """

    def __init__(self, users, items, labels):
        self.users = users
        self.items = items
        self.labels = labels

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        return {
            "users": torch.tensor(self.users[idx], dtype=torch.long),
            "items": torch.tensor(self.items[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
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
        u = int(self.users[idx])
        i_pos = int(self.items[idx])
        t = int(self.timestamps[idx])

        # 1 positive
        user_ids = [u]
        item_ids = [i_pos]
        labels = [1.0]

        # K negatives
        neg_items = self.negative_sampler.sample(
            user_id=u,
            n_negatives=self.n_negatives,
            current_timestamp=t,  # ignored for now, used later if time-aware
        )

        for j in neg_items:
            user_ids.append(u)
            item_ids.append(int(j))
            labels.append(0.0)

        return {
            "users": torch.tensor(user_ids, dtype=torch.long),
            "items": torch.tensor(item_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }
