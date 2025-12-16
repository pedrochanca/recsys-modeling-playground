from typing import Type
from torch.utils.data import Dataset, DataLoader


def prep_batch(
    train_dataset: Type[Dataset],
    test_dataset: Type[Dataset],
    batch_size: int,
    n_workers: int,
    shuffle: bool = True,
    verbose: bool = False,
):
    """
    total number of batches = nb. training points / batch_size
    """

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
    )

    if verbose:
        dataiter = iter(train_loader)
        print(next(dataiter))

    return train_loader, test_loader
