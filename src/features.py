"""Feature engineering, written as a scikit-learn step so it can sit in the pipeline."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DROPPED_COLS = ["fnlwgt", "education_num", "capital_gain", "capital_loss",
                "marital_status", "native_country"]


class CensusFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create three new features and remove the columns they replace.

    Each value is computed from the person's own row only, so this step
    learns nothing from the data (fit does nothing).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Net capital income, log-transformed (keeps the sign)
        net = X["capital_gain"] - X["capital_loss"]
        X["capital_net"] = np.sign(net) * np.log1p(np.abs(net))

        # 2. Married and living with spouse
        X["is_married"] = X["marital_status"].isin(
            ["Married-civ-spouse", "Married-AF-spouse"]).astype(int)

        # 3. Born in the United States (stays missing if the country is missing)
        X["native_us"] = (X["native_country"] == "United-States").astype(float)
        X.loc[X["native_country"].isna(), "native_us"] = np.nan

        return X.drop(columns=DROPPED_COLS)