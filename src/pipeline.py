"""The single pipeline used by every model: feature engineering, preprocessing, model."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import N_FOLDS, RANDOM_STATE
from src.features import CensusFeatureEngineer

NUM_COLS = ["age", "hours_per_week", "capital_net", "is_married", "native_us"]
CAT_COLS = ["workclass", "education", "occupation", "relationship", "race", "sex"]

# Same folds for the model comparison (step 4) and the tuning (step 6)
CV = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),  # only native_us has missing values
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUM_COLS),
        ("cat", categorical, CAT_COLS),
    ])


def build_pipeline(model):
    """Raw data in, prediction out. Every step is fitted on the training data only."""
    return Pipeline([
        ("features", CensusFeatureEngineer()),
        ("prep", build_preprocessor()),
        ("model", model),
    ])