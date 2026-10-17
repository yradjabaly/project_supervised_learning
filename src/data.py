"""Loading, cleaning and train/test split."""
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE


def load_data():
    adult = fetch_openml(name="adult", version=2, as_frame=True)

    X = adult.data.copy()
    X.columns = X.columns.str.replace("-", "_")
    X = X.astype({col: "object" for col in X.select_dtypes("category").columns})
    y = (adult.target == ">50K").astype(int).rename("income")

    df = pd.concat([X, y], axis=1)
    return df


def basic_summary(df):
    return {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "n_missing": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "target_rate": df["income"].mean(),
    }


def make_train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def prepare_data():
    """Raw features and target, split into training and test sets.

    Duplicates are removed on the raw data, before fnlwgt is dropped:
    without fnlwgt, distinct people with identical profiles would wrongly
    look like duplicates. Feature engineering happens inside the pipeline.
    """
    df = load_data().drop_duplicates()
    X = df.drop(columns="income")
    y = df["income"]
    return make_train_test_split(X, y)