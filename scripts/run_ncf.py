import pandas as pd
import numpy as np
import argparse
import yaml
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.ncf import SimpleNCF, DeepNCF
from src.training.eval import collect_user_predictions, compute_metrics
from src.training.train_mlp import train_model, evaluate_model
from src.data.datasets import PointwiseImplicitDataset, OfflineImplicitDataset
from src.utils.hparam_search import param_comb
from src.data.samplers import GlobalUniformNegativeSampler


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(MODEL_ARCHITECTURE, PLOT, TUNE, CONFIG, VERBOSE):

    # ----------------------------------------------------------------------------------
    # ------ Global Parameters
    # ----------------------------------------------------------------------------------

    DEVICE = CONFIG["system"]["device"]
    LOCATION = CONFIG["data"]["location"]
    if TUNE:
        MODEL_CONFIG = CONFIG[MODEL_ARCHITECTURE]["tuning"]
        TRAIN_FILE = "train"
        TEST_FILE = "val"
    else:
        MODEL_CONFIG = CONFIG[MODEL_ARCHITECTURE]["optim_params"]
        TRAIN_FILE = "train_val"
        TEST_FILE = "test"

    # ----------------------------------------------------------------------------------
    # ------ Data
    # ----------------------------------------------------------------------------------

    df_train = pd.read_parquet(f"{LOCATION}/{TRAIN_FILE}.parquet")
    df_test = pd.read_parquet(f"{LOCATION}/{TEST_FILE}.parquet")

    df_interactions = pd.read_parquet(f"{LOCATION}/interactions.parquet")
    user_positive_items = (
        df_interactions.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    n_users = df_interactions["user_id"].max() + 1
    n_items = df_interactions["item_id"].max() + 1

    negative_sampler = GlobalUniformNegativeSampler(n_items, user_positive_items)

    # ----------------------------------------------------------------------------------
    # ------ MAIN: Tune OR evaluate
    # ----------------------------------------------------------------------------------

    param_combinations = param_comb(config=MODEL_CONFIG, is_tune=TUNE)

    for params in param_combinations:
        # MERGE: Combine fixed settings with current trial settings
        # This ensures 'step_size' and 'gamma' are available

        print(f"Testing: {params}")

        # ------------------------------------------------------------------------------
        # ------ Model Related Parameters
        # ------------------------------------------------------------------------------

        EPOCHS = params["epochs"]
        N_NEGATIVES = params["n_negatives"]
        BATCH_SIZE = params["batch_size"]
        N_WORKERS = params["n_workers"]
        STEP_SIZE = params["step_size"]
        GAMMA = params["gamma"]
        LOG_EVERY = params["log_every"]
        THRESHOLD = params["threshold"]

        # ------------------------------------------------------------------------------
        # ------ Prepare Dataset / Loader
        # ------------------------------------------------------------------------------

        train_dataset = PointwiseImplicitDataset(
            users=df_train["user_id"].values,
            items=df_train["item_id"].values,
            timestamps=df_train["ts"].values,
            negative_sampler=negative_sampler,
            n_negatives=N_NEGATIVES,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=True
        )

        test_dataset = OfflineImplicitDataset(
            users=df_test["user_id"].values,
            items=df_test["item_id"].values,
            labels=df_test["label"].values,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False
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
            metrics_to_compute = ["precision", "recall", "hit_rate", "ndcg"]

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
        "--config", type=str, default="src/config/ncf.yml", help="Path to config file"
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
        TUNE=args.tune,
        CONFIG=load_config(args.config),
        VERBOSE=args.verbose,
    )
