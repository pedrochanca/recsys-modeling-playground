import pandas as pd
import numpy as np
import argparse
import yaml

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from src.models.ncf import SimpleNCF, DeepNCF
from src.training.eval import collect_user_predictions, compute_metrics
from src.training.train_mlp import train_model, evaluate_model
from src.data.preprocessing import preprocess_dataframe, prep_datasets
from src.data.datasets import ExplicitDataset
from src.data.dataloaders import prep_batch
from src.utils.hparam_search import param_comb


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(MODEL_ARCHITECTURE, PLOT, VERBOSE, TUNE, CONFIG):

    # ----------------------------------------------------------------------------------
    # ------ Parameters
    # ----------------------------------------------------------------------------------

    # system-related
    RANDOM_SEED = CONFIG["system"]["random_seed"]
    DEVICE = CONFIG["system"]["device"]
    # data-related
    PATH = CONFIG["data"]["path"]
    IMPLICIT_FEEDBACK = CONFIG["data"]["implicit_feedback"]
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
    if IMPLICIT_FEEDBACK:
        df["rating"] = 1.0

    df, n_users, n_items = preprocess_dataframe(df)

    train_set, test_set = prep_datasets(
        df,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        use_validation=TUNE,
        dataset_cls=ExplicitDataset,
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
        VERBOSE=args.verbose,
        TUNE=args.tune,
        CONFIG=load_config(args.config),
    )
