from typing import Any
from torch.utils.data import Dataset
import torch

from src.data.samplers import NegativeSampler


class ImplicitDataset(Dataset):
    def __init__(self, users, items, targets):
        self.users = users
        self.items = items
        self.targets = targets

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        users = self.users[idx]
        items = self.items[idx]
        labels = self.label[idx]

        return (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


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
        num_negatives: int,
    ):
        self.users = users
        self.items = items
        self.timestamps = timestamps
        self.negative_sampler = negative_sampler
        self.num_negatives = num_negatives

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
            num_negatives=self.num_negatives,
            current_timestamp=t,  # ignored for now, used later if time-aware
        )

        for j in neg_items:
            user_ids.append(u)
            item_ids.append(int(j))
            labels.append(0.0)

        return (
            torch.tensor(user_ids, dtype=torch.long),
            torch.tensor(item_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )
