from typing import DefaultDict, Dict, List, Tuple
from collections import defaultdict
import math
import torch

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
    """
    Calculates Precision@K given the top K items.
    """

    if not top_k:
        return 0.0

    n_rel_k = sum(true_r >= threshold for (_, true_r) in top_k)
    return n_rel_k / len(top_k)


def _calc_recall(
    top_k: List[Tuple[float, float]], n_rel_total: int, threshold: float
) -> float:
    """
    Calculates Recall@K given top K items and total relevant count.
    """

    if n_rel_total == 0:
        return 0.0

    n_rel_k = sum(true_r >= threshold for (_, true_r) in top_k)
    return n_rel_k / n_rel_total


def _calc_hit_rate(top_k: List[Tuple[float, float]], threshold: float) -> float:
    """
    Calculates HitRate@K (1.0 if any relevant item is in top K, else 0.0).
    """

    is_hit = any(true_r >= threshold for (_, true_r) in top_k)
    return 1.0 if is_hit else 0.0


def _calc_ndcg(
    top_k: List[Tuple[float, float]], n_rel_total: int, k: int, threshold: float
) -> float:
    """
    Calculates NDCG@K using binary relevance.
    """

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
        List of metrics to compute.
        Options: ["precision", "recall", "hit_rate", "ndcg", "rmse"].
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
    for _, user_targets in user_pred_true.items():

        # --- RMSE Calculation (Uses ALL items, no sorting needed) ---
        if "rmse" in metrics:
            sq_err, count = _calc_squared_error(user_targets)
            rmse_total_sq_err += sq_err
            rmse_total_count += count

        # --- Ranking Metrics (Need Sorting & Top K) ---
        # Only sort if we actually need ranking metrics
        if len(ranking_sums) > 0:

            # Sort once per user
            sorted_targets = sorted(user_targets, key=lambda x: x[0], reverse=True)

            # Prepare common data structures
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
