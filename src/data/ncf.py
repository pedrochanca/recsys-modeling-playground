import pandas as pd
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.samplers import NegativeSampler, GlobalUniformNegativeSampler
from src.utils.constants import (
    DEFAULT_USER_COL as USER_COL,
    DEFAULT_ITEM_COL as ITEM_COL,
    DEFAULT_TARGET_COL as TARGET_COL,
    DEFAULT_TIMESTAMP_COL as TIMESTAMP_COL,
)


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


class NCFDataset:

    def __init__(
        self,
        train_file_path: str,
        test_file_path: str,
        full_file_path: str,
        n_negatives: int = 4,
        user_col: str = USER_COL,
        item_col: str = ITEM_COL,
        timestamp_col: str = TIMESTAMP_COL,
        target_col: str = TARGET_COL,
    ):

        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.n_negatives = n_negatives

        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col
        self.target_col = target_col

        self.df_train = pd.read_parquet(train_file_path)
        self.df_test = pd.read_parquet(test_file_path)
        self.df_full = pd.read_parquet(full_file_path)

        self.n_users = self.df_full[self.user_col].max() + 1
        self.n_items = self.df_full[self.item_col].max() + 1

        self.train_set = self.train_dataset()
        self.test_set = self.test_dataset()

    def train_dataset(self):

        self.user_positive_items = (
            self.df_full.groupby(self.user_col)[self.item_col].apply(set).to_dict()
        )

        self.negative_sampler = GlobalUniformNegativeSampler(
            self.n_items, self.user_positive_items
        )

        return PointwiseImplicitDataset(
            users=self.df_train[self.user_col].values,
            items=self.df_train[self.item_col].values,
            timestamps=self.df_train[self.timestamp_col].values,
            negative_sampler=self.negative_sampler,
            n_negatives=self.n_negatives,
        )

    def test_dataset(self):

        return OfflineImplicitDataset(
            users=self.df_test[self.user_col].values,
            items=self.df_test[self.item_col].values,
            targets=self.df_test[self.target_col].values,
        )

    def train_loader(self, batch_size: int, n_workers: int, shuffle: bool = True):
        return DataLoader(
            self.train_set,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=shuffle,
        )

    def test_loader(self, batch_size: int, n_workers: int, shuffle: bool = False):
        return DataLoader(
            self.test_set,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=shuffle,
        )
