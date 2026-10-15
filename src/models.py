"""Model comparison (step 4) with default hyperparameters.

All scores come from 5-fold cross-validation on the training set.
The test set is not used here.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import FIG_DIR, RANDOM_STATE, RESULTS_DIR, prepare_data

NUM_COLS = ["age", "hours_per_week", "capital_net", "is_married", "native_us"]
CAT_COLS = ["workclass", "education", "occupation", "relationship", "race", "sex"]
SCORING = ["precision", "recall", "f1", "roc_auc", "average_precision"]


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


def get_models():
    """Default hyperparameters, without class weighting.
    The class weight is a hyperparameter: it is tuned in step 6 as
    class_weight={0: 1, 1: w}, with w between 1 (no weighting) and 5."""
    return {
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    }


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def cross_validate_model(model, X_train, y_train):
    """Mean 5-fold cross-validated scores, plus the training PR-AUC to detect overfitting."""
    pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
    scores = cross_validate(pipe, X_train, y_train, cv=CV,
                            scoring=SCORING, return_train_score=True)
    row = {metric: scores[f"test_{metric}"].mean() for metric in SCORING}
    row["average_precision_std"] = scores["test_average_precision"].std()
    row["train_average_precision"] = scores["train_average_precision"].mean()
    return row


def plot_pr_curves(models, X_train, y_train):
    """Precision-recall curves from out-of-fold predictions: each person is
    predicted by a model that did not see them during training."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        proba = cross_val_predict(pipe, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
        PrecisionRecallDisplay.from_predictions(y_train, proba, name=name, ax=ax)
    ax.axhline(y_train.mean(), color="black", linestyle="--",
               label=f"Random model (prevalence = {y_train.mean():.3f})")
    ax.set_title("Precision-recall curves (5-fold cross-validation)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/pr_curves.pdf")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data()  # X_test is NOT used here

    results = {}
    for name, model in get_models().items():
        print(f"Cross-validating {name}...")
        results[name] = cross_validate_model(model, X_train, y_train)
    results = pd.DataFrame(results).T.round(3)
    print("\nModel comparison\n", results.to_string())
    results.to_csv(f"{RESULTS_DIR}/model_comparison.csv")

    plot_pr_curves(get_models(), X_train, y_train)


if __name__ == "__main__":
    main()