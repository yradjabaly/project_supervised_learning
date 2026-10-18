"""Model comparison (step 4) with default hyperparameters.

All scores come from 5-fold cross-validation on the training set.
The test set is not used here.
"""
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import cross_val_predict, cross_validate

from src.config import FIG_DIR, MODEL_COLORS, RANDOM_STATE, RESULTS_DIR
from src.data import prepare_data
from src.pipeline import CV, build_pipeline
from src.plots import plot_metric_bars

SCORING = ["precision", "recall", "f1", "roc_auc", "average_precision"]


def get_models():
    """Default hyperparameters, without class weighting.
    The class weight is a hyperparameter: it is tuned in step 6 (src/tune.py) as
    class_weight={0: 1, 1: w}, with w between 1 (no weighting) and n0/n1
    (full rebalancing), computed on the training set."""
    return {
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    }

def get_final_model():
    """The final model: LightGBM with the best hyperparameters found in step 6
    (saved by src/tune.py in results/best_params.json)."""
    with open(f"{RESULTS_DIR}/best_params.json") as f:
        params = json.load(f)["LightGBM"]
    return LGBMClassifier(
        n_estimators=params["n_estimators"], num_leaves=params["num_leaves"],
        learning_rate=params["learning_rate"], class_weight={0: 1, 1: params["w"]},
        random_state=RANDOM_STATE, verbose=-1)

def cross_validate_model(model, X_train, y_train):
    """Mean 5-fold cross-validated scores, plus the training PR-AUC to detect overfitting."""
    pipe = build_pipeline(model)
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
        pipe = build_pipeline(model)
        proba = cross_val_predict(pipe, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
        PrecisionRecallDisplay.from_predictions(y_train, proba, name=name, ax=ax,
                                                color=MODEL_COLORS[name])
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

    # Number of columns the models receive (used for the max_features range in step 6)
    n_features = build_pipeline("passthrough").fit_transform(X_train).shape[1]
    print(f"Number of features after encoding: {n_features}")

    results = {}
    for name, model in get_models().items():
        print(f"Cross-validating {name}...")
        results[name] = cross_validate_model(model, X_train, y_train)
    results = pd.DataFrame(results).T.round(3)
    print("\nModel comparison\n", results.to_string())
    results.to_csv(f"{RESULTS_DIR}/model_comparison.csv")

    metrics = {"precision": "Precision", "recall": "Recall", "f1": "F1",
               "roc_auc": "ROC-AUC", "average_precision": "PR-AUC",
               "train_average_precision": "Train PR-AUC"}
    best = {m: "max" for m in ["precision", "recall", "f1", "roc_auc", "average_precision"]}
    plot_metric_bars({"Default": results}, metrics, MODEL_COLORS,
                     "model_comparison_bars.pdf", best=best, ylim=(0, 1.1))

    plot_pr_curves(get_models(), X_train, y_train)


if __name__ == "__main__":
    main()