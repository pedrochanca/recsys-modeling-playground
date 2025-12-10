from typing import DefaultDict, Dict, List, Tuple
from collections import defaultdict

import torch

from sklearn.metrics import root_mean_squared_error


# --------------------------------------------------------------------------------------
# ----- Helpers
# --------------------------------------------------------------------------------------


def collect_user_predictions(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    verbose: bool = False,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Run the model on the test set and collect predicted and true targets per user.

    Each batch from `test_loader` is expected to be a dict with keys:
    - "users": user IDs
    - "items": item IDs
    - "targets": true target values (e.g., ratings, clicks, scores, etc.)

    Parameters
    ----------
    model
        Trained PyTorch model that takes (users, items) and outputs predictions.
    loader
        DataLoader yielding test batches as dicts with "users", "items", "targets".
    device
        Device on which to run inference (e.g., torch.device("cuda") or torch.device("cpu")).
    verbose
        If True, prints predictions and targets per batch.

    Returns
    -------
    user_pred_true
        A dictionary mapping each user_id (int) to a list of (predicted_value, true_value)
        tuples.
    """
    model.eval()
    user_pred_true: DefaultDict[int, List[Tuple[float, float]]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            users = batch["users"].to(device)
            items = batch["items"].to(device)
            targets = batch["targets"].to(device)

            pred_target = model(users, items)
            true_target = targets.view(targets.size(0), -1).to(torch.float32)

            for idx in range(len(users)):
                user_id = users[idx].item()
                item_id = items[idx].item()
                pred = pred_target[idx][0].item()
                true = true_target[idx][0].item()

                if verbose:
                    print("{}, {}, {}, {}".format(user_id, item_id, pred, true))
                user_pred_true[user_id].append((pred, true))

    return user_pred_true


# --------------------------------------------------------------------------------------
# ----- Metrics
# --------------------------------------------------------------------------------------


def rmse(
    user_pred_true: Dict[int, List[Tuple[float, float]]],
    verbose: bool = False,
) -> float:
    """
    Compute RMSE on a test set using the output of collect_user_predictions.

    This calculates the global RMSE (equivalent to 'per_sample' mode).

    Parameters
    ----------
    user_pred_true
        A dictionary mapping each user_id (int) to a list of (predicted_value,
        true_value) tuples.
    verbose
        If True, prints predictions and targets during collection.

    Returns
    -------
    score
        The computed root mean squared error.
    """
    pred_list = []
    true_list = []

    # Flatten the dictionary to get global lists of predictions and targets
    for user_id, interactions in user_pred_true.items():
        for pred, true in interactions:
            pred_list.append(pred)
            true_list.append(true)

            if verbose:
                print("user_id: {}; pred: {}; true: {}".format(user_id, pred, true))

    score = root_mean_squared_error(true_list, pred_list)

    return score


def precision_recall_at_k(
    user_pred_true: Dict[int, List[Tuple[float, float]]],
    k: int = 10,
    threshold: float = 3.5,
    verbose: bool = False,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute per-user Precision@K and Recall@K from predicted and true targets.

    The `user_pred_true` mapping is expected to store, for each user, a list of
    (predicted_value, true_value) tuples. Predicted values are used to rank items,
    and both predicted and true values are compared to a relevance threshold.

    Parameters
    ----------
    user_pred_true
        A dictionary mapping each user_id (int) to a list of (predicted_value, true_value)
        tuples.
    k
        The cutoff rank K for Precision@K and Recall@K.
    threshold
        Relevance threshold. Predicted/true values greater than or equal to this are
        considered relevant.
    verbose
        If True, prints per-user counts used in the metric computations.

    Returns
    -------
    precisions
        A dictionary mapping each user_id (int) to its Precision@K.
    recalls
        A dictionary mapping each user_id (int) to its Recall@K.
    """
    precisions: Dict[int, float] = {}
    recalls: Dict[int, float] = {}

    for user_id, user_targets in user_pred_true.items():
        # Sort user targets by predicted value (descending)
        sorted_targets = sorted(user_targets, key=lambda x: x[0], reverse=True)

        # Number of actually relevant items
        n_rel = sum(true_r >= threshold for (_, true_r) in sorted_targets)

        # Number of recommended items that are predicted relevant within top-K
        top_k = sorted_targets[:k]
        n_rec_k = sum(pred_r >= threshold for (pred_r, _) in top_k)

        # Number of recommended items that are predicted relevant AND actually relevant
        # within top-K
        n_rec_rel_k = sum(
            (pred_r >= threshold) and (true_r >= threshold)
            for (pred_r, true_r) in top_k
        )

        if verbose:
            print(
                "user_id: {}; n_rel: {}; n_rec_k: {}; n_rec_rel_k: {}".format(
                    user_id, n_rel, n_rec_k, n_rec_rel_k
                )
            )

        # Precision@K: Proportion of recommended items that are relevant.
        precisions[user_id] = n_rec_rel_k / n_rec_k if n_rec_k != 0 else 0.0

        # Recall@K: Proportion of relevant items that are recommended.
        recalls[user_id] = n_rec_rel_k / n_rel if n_rel != 0 else 0.0

    return precisions, recalls
