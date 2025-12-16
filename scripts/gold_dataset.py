import pandas as pd
import argparse
from sklearn import model_selection

from src.data.preprocessing import preprocess_dataframe


def ml_latest_small_user_item_interactions(
    val_split: float, test_split: float, random_seed=42
):

    input_file = f"data/silver/ml-latest-small/ratings.parquet"

    df = pd.read_parquet(input_file)

    df.rename(
        columns={
            "userId": "user_id",
            "movieId": "item_id",
            "timestamp": "ts",
            "rating": "target",
        },
        inplace=True,
    )
    df["target"] = 1.0

    df, _, _ = preprocess_dataframe(df)

    df_train_val, df_test = model_selection.train_test_split(
        df, test_size=test_split, random_state=random_seed, stratify=df["target"].values
    )

    # Adjust val ratio relative to the remaining data
    relative_val_size = val_split / (1 - test_split)

    df_train, df_val = model_selection.train_test_split(
        df_train_val,
        test_size=relative_val_size,
        random_state=random_seed,
        stratify=df_train_val["target"].values,
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

    args = parser.parse_args()

    globals()[args.dataset](val_split=args.val_split, test_split=args.test_split)
