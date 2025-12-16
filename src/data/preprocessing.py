from typing import Type
import pandas as pd
from sklearn import model_selection, preprocessing
from torch.utils.data import Dataset


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
    dataset_cls: Type[Dataset],
    random_seed: int = 42,
    verbose: bool = False,
):

    df_train_val, df_test = model_selection.train_test_split(
        df, test_size=test_split, random_state=random_seed, stratify=df["target"].values
    )

    if use_validation:
        # Adjust val ratio relative to the remaining data
        relative_val_size = val_split / (1 - test_split)

        df_train, df_test = model_selection.train_test_split(
            df_train_val,
            test_size=relative_val_size,
            random_state=random_seed,
            stratify=df_train_val["target"].values,
        )

        test_dataset = dataset_cls(
            users=df_test["user_id"].values,
            items=df_test["item_id"].values,
            targets=df_test["target"].values,
        )
    else:
        # Keep all remaining data as training
        df_train = df_train_val

    train_dataset = dataset_cls(
        users=df_train["user_id"].values,
        items=df_train["item_id"].values,
        targets=df_train["target"].values,
    )

    test_dataset = dataset_cls(
        users=df_test["user_id"].values,
        items=df_test["item_id"].values,
        targets=df_test["target"].values,
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
