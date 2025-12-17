import argparse
import numpy as np
import pandas as pd
from typing import Dict, Set
from sklearn import model_selection, preprocessing

# --------------------------------------------------------------------------------------
# ----- Helper
# --------------------------------------------------------------------------------------


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


def add_negative_interactions(
    df: pd.DataFrame,
    user_positive_items: Dict[int, Set[int]],
    n_items: int,
    n_negatives: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Build an offline evaluation dataframe with fixed negatives.

    Parameters
    ----------
    df
        DataFrame with at least columns ["user_id", "item_id"] representing held-out
        positives for each user.
    user_positive_items
        Dict[user] -> set(items) with ALL items the user has interacted with
        (train + val + test), used to avoid sampling true positives as negatives.
    n_items
        Total number of items (embedding size).
    n_negatives
        Number of negatives to sample per positive.
    random_seed
        Random seed for reproducible negative sampling.

    Returns
    -------
    df_with_negatives
        DataFrame with columns ["user_id", "item_id", "label"].
    """
    rng = np.random.default_rng(random_seed)
    all_items = np.arange(n_items, dtype=np.int64)

    # Normalise column names
    u_col, i_col = "user_id", "item_id"

    rows = []

    for row in df[[u_col, i_col]].itertuples(index=False):
        u = int(getattr(row, u_col))
        i_pos = int(getattr(row, i_col))

        # positive
        rows.append((u, i_pos, 1.0))

        pos_items = user_positive_items.get(u, set())
        if len(pos_items) >= n_items:
            # Degenerate case: user interacted with all items
            continue

        # candidate negatives = all items user has never interacted with
        # (convert set -> array once per row; you can optimise per user later)
        pos_arr = (
            np.fromiter(pos_items, dtype=np.int64)
            if pos_items
            else np.array([], dtype=np.int64)
        )
        candidates = np.setdiff1d(all_items, pos_arr, assume_unique=True)

        k = min(n_negatives, len(candidates))
        if k == 0:
            continue

        neg_items = rng.choice(candidates, size=k, replace=False)
        for j in neg_items:
            rows.append((u, int(j), 0.0))

    df_with_negatives = pd.DataFrame(rows, columns=[u_col, i_col, "label"])

    return df_with_negatives


# --------------------------------------------------------------------------------------
# ----- Main
# --------------------------------------------------------------------------------------


def ml_latest_small_user_item_interactions(
    val_split: float, test_split: float, n_negatives: int, random_seed: int
):

    input_file = f"data/silver/ml-latest-small/ratings.parquet"

    df = pd.read_parquet(input_file)

    df.rename(
        columns={
            "userId": "user_id",
            "movieId": "item_id",
            "timestamp": "ts",
            "rating": "label",
        },
        inplace=True,
    )
    df["label"] = 1.0

    df, _, _ = preprocess_dataframe(df)

    df_train_val, df_test = model_selection.train_test_split(
        df, test_size=test_split, random_state=random_seed, stratify=df["label"].values
    )

    # Adjust val ratio relative to the remaining data
    relative_val_size = val_split / (1 - test_split)

    df_train, df_val = model_selection.train_test_split(
        df_train_val,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=df_train_val["label"].values,
    )

    if n_negatives != 0:

        user_positive_items = df.groupby("user_id")["item_id"].apply(set).to_dict()

        n_items = df["item_id"].max() + 1

        df_val = add_negative_interactions(
            df=df_val,
            user_positive_items=user_positive_items,
            n_items=n_items,
            n_negatives=n_negatives,
            random_seed=random_seed,
        )

        df_test = add_negative_interactions(
            df=df_test,
            user_positive_items=user_positive_items,
            n_items=n_items,
            n_negatives=n_negatives,
            random_seed=random_seed,
        )

    print(len(df_train), len(df_val), len(df_train_val), len(df_test))

    location = "data/gold/ml-latest-small"
    df.to_parquet(f"{location}/interactions.parquet", index=False)
    df_train.to_parquet(f"{location}/train.parquet", index=False)
    df_val.to_parquet(f"{location}/val.parquet", index=False)
    df_train_val.to_parquet(f"{location}/train_val.parquet", index=False)
    df_test.to_parquet(f"{location}/test.parquet", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute dataset gold version")

    # define the --dataset argument
    parser.add_argument(
        "--dataset", type=str, choices=["ml_latest_small_user_item_interactions"]
    )

    # define the --val_split argument
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
    )

    # define the --test_split argument
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--n_negatives",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    globals()[args.dataset](
        val_split=args.val_split,
        test_split=args.test_split,
        n_negatives=args.n_negatives,
        random_seed=args.random_seed,
    )
