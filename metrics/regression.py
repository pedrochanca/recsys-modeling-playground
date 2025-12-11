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


from typing import Dict, List, Tuple
import math

# --------------------------------------------------------------------------------------
# ----- Metric Logic (Micro-Kernels)
# --------------------------------------------------------------------------------------


def _calc_squared_error(user_targets: List[Tuple[float, float]]) -> Tuple[float, int]:
    """
    Calculates sum of squared errors and count for a user.
    Used for Global RMSE computation.
    """
    if not user_targets:
        return 0.0, 0

    sq_error = sum((pred - true) ** 2 for pred, true in user_targets)
    return sq_error, len(user_targets)


def _calc_precision(top_k: List[Tuple[float, float]], threshold: float) -> float:
    """Calculates Precision@K given the top K items."""
    if not top_k:
        return 0.0

    n_rel_k = sum(true_r >= threshold for (_, true_r) in top_k)
    return n_rel_k / len(top_k)


def _calc_recall(
    top_k: List[Tuple[float, float]], n_rel_total: int, threshold: float
) -> float:
    """Calculates Recall@K given top K items and total relevant count."""
    if n_rel_total == 0:
        return 0.0

    n_rel_k = sum(true_r >= threshold for (_, true_r) in top_k)
    return n_rel_k / n_rel_total


def _calc_hit_rate(top_k: List[Tuple[float, float]], threshold: float) -> float:
    """Calculates HitRate@K (1.0 if any relevant item is in top K, else 0.0)."""
    is_hit = any(true_r >= threshold for (_, true_r) in top_k)
    return 1.0 if is_hit else 0.0


def _calc_ndcg(
    top_k: List[Tuple[float, float]], n_rel_total: int, k: int, threshold: float
) -> float:
    """Calculates NDCG@K using binary relevance."""
    dcg = 0.0

    # 1. Calculate DCG (based on predicted rank in top_k)
    for i, (_, true_r) in enumerate(top_k):
        if true_r >= threshold:
            dcg += 1.0 / math.log2(i + 2)

    # 2. Calculate IDCG (Ideal DCG)
    idcg = 0.0
    num_ideal_relevant = min(n_rel_total, k)

    for i in range(num_ideal_relevant):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------------------
# ----- Main Controller
# --------------------------------------------------------------------------------------


def compute_metrics(
    user_pred_true: Dict[int, List[Tuple[float, float]]],
    metrics: List[str] = None,
    k: int = 10,
    threshold: float = 3.5,
) -> Dict[str, float]:
    """
    Generalizable function to compute selected metrics in a single pass.

    - Ranking metrics (Precision, Recall, NDCG, HitRate) use Top K.
    - RMSE uses ALL test predictions (Global RMSE).

    Parameters
    ----------
    user_pred_true
        Dictionary {user_id: [(pred_rating, true_rating), ...]}.
    metrics
        List of metrics to compute. Options: ["precision", "recall", "hit_rate", "ndcg", "rmse"].
    k
        The rank cutoff.
    threshold
        The rating threshold for binary relevance.

    Returns
    -------
    results
        Dictionary containing average score for each requested metric.
    """
    # Defaults
    if metrics is None:
        metrics = ["precision", "recall", "hit_rate", "ndcg", "rmse"]

    # Validation
    valid_metrics = {"precision", "recall", "hit_rate", "ndcg", "rmse"}
    for m in metrics:
        if m not in valid_metrics:
            raise ValueError(f"Metric '{m}' not supported. Choose from {valid_metrics}")

    # Accumulators for ranking metrics (Sum of per-user scores)
    ranking_sums = {m: 0.0 for m in metrics if m != "rmse"}

    # Accumulators for RMSE (Global calculation)
    rmse_total_sq_err = 0.0
    rmse_total_count = 0

    n_users = len(user_pred_true)
    if n_users == 0:
        return {m: 0.0 for m in metrics}

    # Optimization: Iterate users once
    for user_id, user_targets in user_pred_true.items():

        # --- RMSE Calculation (Uses ALL items, no sorting needed) ---
        if "rmse" in metrics:
            sq_err, count = _calc_squared_error(user_targets)
            rmse_total_sq_err += sq_err
            rmse_total_count += count

        # --- Ranking Metrics (Need Sorting & Top K) ---
        # Only sort if we actually need ranking metrics
        if len(ranking_sums) > 0:

            # 1. HEAVY LIFTING: Sort once per user
            sorted_targets = sorted(user_targets, key=lambda x: x[0], reverse=True)

            # 2. Prepare common data structures
            top_k = sorted_targets[:k]

            # We only need total relevant count if recall or ndcg is requested
            n_rel_total = 0
            if "recall" in metrics or "ndcg" in metrics:
                n_rel_total = sum(true_r >= threshold for (_, true_r) in sorted_targets)

            # 3. Dispatch to separate functions
            if "precision" in metrics:
                ranking_sums["precision"] += _calc_precision(top_k, threshold)

            if "recall" in metrics:
                ranking_sums["recall"] += _calc_recall(top_k, n_rel_total, threshold)

            if "hit_rate" in metrics:
                ranking_sums["hit_rate"] += _calc_hit_rate(top_k, threshold)

            if "ndcg" in metrics:
                ranking_sums["ndcg"] += _calc_ndcg(top_k, n_rel_total, k, threshold)

    # Finalize Results
    final_results = {}

    # Average Ranking metrics over Users
    for m, value in ranking_sums.items():
        final_results[m] = value / n_users

    # Calculate Global RMSE
    if "rmse" in metrics:
        if rmse_total_count > 0:
            final_results["rmse"] = math.sqrt(rmse_total_sq_err / rmse_total_count)
        else:
            final_results["rmse"] = 0.0

    return final_results
