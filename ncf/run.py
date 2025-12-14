import pandas as pd
import numpy as np
import argparse
import yaml
import itertools

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from sklearn import model_selection, preprocessing

import matplotlib.pyplot as plt


from ncf.model import SimpleNCF, DeepNCF
from metrics.engine import collect_user_predictions, compute_metrics


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def preprocess_dataframe(df: pd.DataFrame):

    # encode the user and item id to start from 0 (this is what nn.Embedding expects)
    # this prevents us from run into index out of bound "error" with Embedding lookup
    lbl_user = preprocessing.LabelEncoder()
    lbl_item = preprocessing.LabelEncoder()

    df.user_id = lbl_user.fit_transform(df.user_id.values)
    df.item_id = lbl_item.fit_transform(df.item_id.values)

    n_users = len(lbl_user.classes_)
    n_items = len(lbl_item.classes_)

    return df, n_users, n_items


def prep_datasets(
    df: pd.DataFrame,
    val_split: float,
    test_split: float,
    use_validation: bool,
    random_seed: int = 42,
    verbose: bool = False,
):

    df_train_val, df_test = model_selection.train_test_split(
        df, test_size=test_split, random_state=random_seed, stratify=df["rating"].values
    )

    if use_validation:
        # Adjust val ratio relative to the remaining data
        relative_val_size = val_split / (1 - test_split)

        df_train, df_test = model_selection.train_test_split(
            df_train_val,
            test_size=relative_val_size,
            random_state=random_seed,
            stratify=df_train_val["rating"].values,
        )

        test_dataset = ExplicitDataset(
            users=df_test["user_id"].values,
            items=df_test["item_id"].values,
            targets=df_test["rating"].values,
        )
    else:
        # Keep all remaining data as training
        df_train = df_train_val

    train_dataset = ExplicitDataset(
        users=df_train["user_id"].values,
        items=df_train["item_id"].values,
        targets=df_train["rating"].values,
    )

    test_dataset = ExplicitDataset(
        users=df_test["user_id"].values,
        items=df_test["item_id"].values,
        targets=df_test["rating"].values,
    )

    if verbose:
        print(
            "Lengths: train set = {}; test set = {}".format(
                len(train_dataset), len(test_dataset)
            ),
            "Types: train set = {}; test set = {}".format(
                type(train_dataset), type(test_dataset)
            ),
        )

    return train_dataset, test_dataset


def prep_batch(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    n_workers: int = 2,
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


def train_model(
    loader,
    model,
    loss_func,
    optimizer,
    scheduler,
    device: str,
    epochs: int = 1,
    log_every: int = 1000,
):
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

    for epoch_i in range(epochs):
        for i, batch in enumerate(loader):
            users = batch["users"].to(device)
            items = batch["items"].to(device)
            targets = batch["targets"].to(device)

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

        if scheduler:
            scheduler.step()

    return model, all_losses_list


def evaluate_model(loader, model, loss_func, device: str):
    model.eval()  # Important: turns off dropout!

    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # Important: saves memory, no gradients
        for batch in loader:
            users = batch["users"].to(device)
            items = batch["items"].to(device)
            targets = batch["targets"].to(device).view(-1, 1).float()

            pred_target = model(users, items)
            true_target = targets.view(targets.size(0), -1).to(torch.float32)

            loss = loss_func(pred_target, true_target)

            total_loss += loss.sum().item()
            total_samples += users.size(0)

    return total_loss / total_samples


def param_comb(config, is_tune: bool):

    if is_tune:

        keys, values = zip(*config.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    else:
        combinations = [config]

    return combinations


def main(MODEL_ARCHITECTURE, PLOT, VERBOSE, TUNE, CONFIG):

    # ----------------------------------------------------------------------------------
    # ------ Parameters
    # ----------------------------------------------------------------------------------

    RANDOM_SEED = CONFIG["system"]["random_seed"]
    DEVICE = CONFIG["system"]["device"]

    PATH = CONFIG["data"]["path"]
    VAL_SPLIT = CONFIG["data"]["val_split"]
    TEST_SPLIT = CONFIG["data"]["test_split"]

    # ----------------------------------------------------------------------------------
    # ------ Data / batch setup
    # ----------------------------------------------------------------------------------

    df = pd.read_parquet(PATH)
    df.rename(
        columns={"userId": "user_id", "movieId": "item_id", "timestamp": "ts"},
        inplace=True,
    )

    df, n_users, n_items = preprocess_dataframe(df)

    train_set, test_set = prep_datasets(
        df,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        use_validation=TUNE,
        random_seed=RANDOM_SEED,
        verbose=VERBOSE,
    )

    # ----------------------------------------------------------------------------------
    # ------ Tune OR evaluate
    # ----------------------------------------------------------------------------------

    if TUNE:
        model_config = CONFIG[MODEL_ARCHITECTURE]["tuning"]
    else:
        model_config = CONFIG[MODEL_ARCHITECTURE]["optim_params"]

    param_combinations = param_comb(config=model_config, is_tune=TUNE)

    for params in param_combinations:
        # MERGE: Combine fixed settings with current trial settings
        # This ensures 'step_size' and 'gamma' are available

        print(f"Testing: {params}")

        # ------------------------------------------------------------------------------
        # ------ Model Related Parameters
        # ------------------------------------------------------------------------------

        EPOCHS = params["epochs"]
        BATCH_SIZE = params["batch_size"]
        N_WORKERS = params["n_workers"]
        STEP_SIZE = params["step_size"]
        GAMMA = params["gamma"]
        LOG_EVERY = params["log_every"]
        THRESHOLD = params["threshold"]

        train_loader, test_loader = prep_batch(
            train_set,
            test_set,
            batch_size=BATCH_SIZE,
            n_workers=N_WORKERS,
            verbose=VERBOSE,
        )

        # ------------------------------------------------------------------------------
        # ------ Batch Preparation
        # ------------------------------------------------------------------------------

        train_loader, test_loader = prep_batch(
            train_set, test_set, batch_size=BATCH_SIZE, verbose=VERBOSE
        )

        # ------------------------------------------------------------------------------
        # ------ Model Dynamic Instantiation
        # ------------------------------------------------------------------------------

        print(f"Initializing {MODEL_ARCHITECTURE}...")
        try:
            # Get the class by name from global scope
            model_class = globals()[MODEL_ARCHITECTURE]
            model = model_class(n_users=n_users, n_items=n_items, **params).to(DEVICE)

        except KeyError:
            raise ValueError(
                f"Model architecture '{MODEL_ARCHITECTURE}' not found in code."
            )

        # ------------------------------------------------------------------------------
        # ------ Train
        # ------------------------------------------------------------------------------

        loss_func = nn.MSELoss(reduction="none")

        optimizer = torch.optim.Adam(model.parameters())

        # Every `step_size` (epoch) calls to scheduler.step(), multiply the learning
        # rate by `gamma`
        # By default, Adam has a learning rate of 0.001
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=STEP_SIZE, gamma=GAMMA
        )

        model, all_losses_list = train_model(
            train_loader,
            model,
            loss_func,
            optimizer,
            scheduler,
            device=DEVICE,
            epochs=EPOCHS,
            log_every=LOG_EVERY,
        )

        # Plot Loss
        if PLOT:
            plt.figure()
            plt.plot(all_losses_list)
            plt.show()

        print("Train Loss: {}\n".format(np.round(all_losses_list[-1], 4)))

        # ------------------------------------------------------------------------------
        # ------ Evaluation (Test set)
        # ------------------------------------------------------------------------------
        test_loss = evaluate_model(test_loader, model, loss_func, DEVICE)
        print("Test Loss: {}\n".format(np.round(test_loss, 4)))

        if not TUNE:

            K = [1, 3, 5, 10, 20, 50, 100]
            metrics_to_compute = ["precision", "recall", "hit_rate", "ndcg", "rmse"]

            user_pred_true = collect_user_predictions(test_loader, model, DEVICE)

            for k in K:

                if "rmse" in metrics_to_compute and k != K[0]:
                    metrics_to_compute.remove("rmse")

                print(metrics_to_compute)

                metrics = compute_metrics(
                    user_pred_true=user_pred_true,
                    metrics=metrics_to_compute,
                    k=k,
                    threshold=THRESHOLD,
                )

                for metric in metrics_to_compute:
                    if metric != "rmse":
                        print(
                            "{} @ {}: {}\n".format(
                                metric.upper(), k, np.round(metrics[metric], 4)
                            )
                        )
                    else:
                        print(
                            "{}: {}\n".format(
                                metric.upper(), np.round(metrics[metric], 4)
                            )
                        )


if __name__ == "__main__":
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
    parser.add_argument(
        "--config", type=str, default="ncf/config.yml", help="Path to config file"
    )
    parser.add_argument(
        "--plot",
        action="store_true",  # Sets value to True if argument is present
        help="Enable plotting",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",  # Sets value to True if argument is present
        help="Enable verbose",
    )
    parser.add_argument(
        "--tune",
        action="store_true",  # Sets value to True if argument is present
        help="Run hyperparameter tuning",
    )
    args = parser.parse_args()

    main(
        MODEL_ARCHITECTURE=args.model,
        PLOT=args.plot,
        VERBOSE=args.verbose,
        TUNE=args.tune,
        CONFIG=load_config(args.config),
    )
