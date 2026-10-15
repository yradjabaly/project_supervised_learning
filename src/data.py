"""Loading, cleaning, feature engineering and train/test split."""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
FIG_DIR = "report/figures"
RESULTS_DIR = "results"


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


def add_features(df):
    """Create three new features and remove the columns they replace."""
    df = df.copy()

    # 1. Net capital income, log-transformed (keeps the sign)
    net = df["capital_gain"] - df["capital_loss"]
    df["capital_net"] = np.sign(net) * np.log1p(np.abs(net))

    # 2. Married and living with spouse
    df["is_married"] = df["marital_status"].isin(
        ["Married-civ-spouse", "Married-AF-spouse"]).astype(int)

    # 3. Born in the United States (stays missing if the country is missing)
    df["native_us"] = (df["native_country"] == "United-States").astype(float)
    df.loc[df["native_country"].isna(), "native_us"] = np.nan

    return df.drop(columns=["fnlwgt", "education_num", "capital_gain",
                            "capital_loss", "marital_status", "native_country"])


def make_train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def prepare_data():
    """Full data preparation used by the modelling scripts.

    Duplicates are removed before dropping fnlwgt: without fnlwgt, distinct
    people with identical profiles would wrongly look like duplicates.
    """
    df = load_data().drop_duplicates()
    df = add_features(df)
    X = df.drop(columns="income")
    y = df["income"]
    return make_train_test_split(X, y)