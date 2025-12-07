import pandas as pd
import argparse

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from sklearn import model_selection, preprocessing

import matplotlib.pyplot as plt


from models.ncf import SimpleNCF, DeepNCF


class ExplicitDataset(Dataset):
    def __init__(self, users, items, targets):
        self.users = users
        self.items = items
        self.targets = targets

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        users = self.users[item]
        items = self.items[item]
        targets = self.targets[item]

        return {
            "users": torch.tensor(users, dtype=torch.long),
            "items": torch.tensor(items, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.float),
        }


def prep_datasets(df: pd.DataFrame, verbose: bool = False):

    # encode the user and item id to start from 0 (this is what nn.Embedding expects)
    # this prevents us from run into index out of bound "error" with Embedding lookup
    lbl_user = preprocessing.LabelEncoder()
    lbl_item = preprocessing.LabelEncoder()

    df.user_id = lbl_user.fit_transform(df.user_id.values)
    df.item_id = lbl_item.fit_transform(df.item_id.values)

    df_train, df_test = model_selection.train_test_split(
        df, test_size=0.1, random_state=42, stratify=df.rating.values
    )

    train_dataset = ExplicitDataset(
        users=df_train.user_id.values,
        items=df_train.item_id.values,
        targets=df_train.rating.values,
    )

    test_dataset = ExplicitDataset(
        users=df_test.user_id.values,
        items=df_test.item_id.values,
        targets=df_test.rating.values,
    )

    n_users = len(lbl_user.classes_)
    n_items = len(lbl_item.classes_)

    if verbose:
        print(
            "Lengths: train set = {}; test set = {}".format(
                len(train_dataset), len(test_dataset)
            ),
            "Types: train set = {}; test set = {}".format(
                type(train_dataset), type(test_dataset)
            ),
        )

    return train_dataset, test_dataset, n_users, n_items


def prep_batch(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
    verbose: bool = False,
):
    """
    total number of batches = nb. training points / batch_size
    """

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    if verbose:
        dataiter = iter(train_loader)
        print(next(dataiter))

    return train_loader, test_loader


def train_model(train_loader, device, model, loss_func, optimizer, epochs: int = 1):
    """
    epochs: # nb. of times we go through the train set

    model.train():
        - Puts the model into "training mode"

        - It changes how some layers behave:
            1. Dropout layers (nn.Dropout)
                - In train() mode: randomly zero out some activations (adds noise,
                regularizes).
                - In eval() mode: no dropout, they pass everything through (but scaled
                appropriately during training).

            2. BatchNorm layers (nn.BatchNorm1d, nn.BatchNorm2d, etc.)
                (it fixes the "internal covariate shift" problem)
                - In train() mode: use the current batch's mean/variance and update
                running stats.
                - In eval() mode: use the stored running mean/variance (fixed
                statistics).
    """

    model.train()

    total_loss = 0
    total_samples = 0
    all_losses_list = []

    # log loss every X batches
    log_every = 1000

    for epoch_i in range(epochs):
        for i, train_data in enumerate(train_loader):
            # move data to device
            users = train_data["users"].to(device)
            items = train_data["items"].to(device)
            targets = train_data["targets"].to(device)

            # foward pass
            pred_target = model(users, items)
            true_target = targets.view(targets.size(0), -1).to(torch.float32)

            # reduction = "none" --> Vector [Batch_Size, 1]
            loss = loss_func(pred_target, true_target)

            # clears old gradients from previous iteration
            optimizer.zero_grad()

            # backpropagation: performs backward propragation
            # (fills param.grad for every parameter in model.parameters())

            # Manually calculate Mean for Backpropagation
            # The optimizer needs a single scalar to minimize.
            loss_scalar = loss.mean()
            loss_scalar.backward()
            # param update: uses the gradients in param.grads to update the parameters
            optimizer.step()

            # ---- Plot releated
            # Sum the vector directly for logging
            # loss.sum() adds up the squared errors of all X users in the batch
            total_loss += loss.sum().item()
            total_samples += users.size(0)
            if (i + 1) % log_every == 0:
                avg_loss = total_loss / total_samples
                print(
                    "Epoch: {} | Step: {} | Loss: {}".format(epoch_i, i + 1, avg_loss)
                )
                all_losses_list.append(avg_loss)

                # Reset
                total_loss = 0
                total_samples = 0

    return model, all_losses_list


def main():

    # setup Argument Parser
    parser = argparse.ArgumentParser(description="Train NCF models")

    # define the --model argument
    parser.add_argument(
        "--model",
        type=str,
        default="DeepNCF",
        choices=["SimpleNCF", "DeepNCF"],
        help="Model architecture to use: SimpleNCF or DeepNCF",
    )
    args = parser.parse_args()
    MODEL_ARCHITECTURE = args.model

    VERBOSE = True

    # Train parameters
    STEP_SIZE = 3
    GAMMA = 0.7
    EPOCHS = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("datasets/ml-latest-small/ratings.csv")
    df.rename(
        columns={"userId": "user_id", "movieId": "item_id", "timestamp": "ts"},
        inplace=True,
    )

    train_set, test_set, n_users, n_items = prep_datasets(df, verbose=VERBOSE)
    train_loader, test_loader = prep_batch(train_set, test_set, verbose=VERBOSE)

    # --- DYNAMIC MODEL INSTANTIATION ---
    print(f"Initializing {MODEL_ARCHITECTURE}...")
    try:
        # Get the class by name from global scope
        model_class = globals()[MODEL_ARCHITECTURE]
        model = model_class(n_users=n_users, n_items=n_items).to(device)
    except KeyError:
        raise ValueError(
            f"Model architecture '{MODEL_ARCHITECTURE}' not found in code."
        )

    optimizer = torch.optim.Adam(model.parameters())

    # Every `step_size` calls to scheduler.step(), multiply the learning rate by `gamma`
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
    )

    loss_func = nn.MSELoss(reduction="none")

    model, all_losses_list = train_model(
        train_loader, device, model, loss_func, optimizer, epochs=EPOCHS
    )

    # Plot Loss
    plt.figure()
    plt.plot(all_losses_list)
    plt.show()


if __name__ == "__main__":
    main()
