"""Calibration of the final model (step 9), on the training set only.

Three candidates are compared with 5-fold cross-validation: the uncalibrated model,
Platt scaling (sigmoid) and isotonic regression. Rule decided in advance: we try both methods and, if their results are very close,
we keep the simpler one (Platt scaling). Brier score and ECE are reported to check it. 
The test set is not used here.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_predict

from src.config import AQUA, BLUE, FIG_DIR, GRAY, RESULTS_DIR
from src.data import prepare_data
from src.models import get_final_model
from src.pipeline import CV, build_pipeline
from src.plots import plot_metric_bars

N_BINS = 10  # equal-width bins for the ECE and the reliability diagram
METHODS = {"Uncalibrated": None, "Platt (sigmoid)": "sigmoid", "Isotonic": "isotonic"}
METHOD_COLORS = {"Uncalibrated": GRAY, "Platt (sigmoid)": BLUE, "Isotonic": AQUA}

def build_calibrated_model(method):
    """Final pipeline, with or without calibration.
    With calibration, the calibrator is learned on out-of-fold predictions (cv=CV),
    i.e. on predictions for people the model did not train on. The model itself is
    then refitted on all the data it receives (ensemble=False)."""
    pipe = build_pipeline(get_final_model())
    if method is None:
        return pipe
    return CalibratedClassifierCV(pipe, method=method, cv=CV, ensemble=False)


def expected_calibration_error(y, proba, n_bins=N_BINS):
    """ECE = sum over bins of (share of people in the bin) x |observed rate - mean prediction|."""
    y = np.asarray(y)
    bins = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(y[in_bin].mean() - proba[in_bin].mean())
    return ece


def probability_scores(y, proba):
    """Probability quality (Brier, ECE), ranking (PR-AUC) and decisions at threshold 0.5."""
    return {
        "brier": brier_score_loss(y, proba),
        "ece": expected_calibration_error(y, proba),
        "average_precision": average_precision_score(y, proba),
        "mean_predicted_proba": proba.mean(),
        "observed_rate": np.mean(y),
        "precision": precision_score(y, proba >= 0.5),
        "recall": recall_score(y, proba >= 0.5),
    }


def plot_reliability(y, probas, filename, title):
    """Reliability diagram (top) and distribution of the predicted probabilities (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, proba in probas.items():
        observed, predicted = calibration_curve(y, proba, n_bins=N_BINS, strategy="uniform")
        ax1.plot(predicted, observed, "o-", color=METHOD_COLORS[name], label=name)
        ax2.hist(proba, bins=20, range=(0, 1), histtype="step",
                 color=METHOD_COLORS[name], label=name)
    ax1.set_ylabel("Observed share of >50K")
    ax1.set_title(title)
    ax1.legend()
    ax2.set_yscale("log")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Number of people")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{filename}")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data()  # X_test is NOT used here

    probas, rows = {}, {}
    for name, method in METHODS.items():
        print(f"Cross-validating {name}...")
        model = build_calibrated_model(method)
        probas[name] = cross_val_predict(model, X_train, y_train, cv=CV,
                                         method="predict_proba")[:, 1]
        rows[name] = probability_scores(y_train, probas[name])

    results = pd.DataFrame(rows).T
    print("\nCalibration (5-fold cross-validation on the training set)\n",
          results.round(4).to_string())
    results.round(4).to_csv(f"{RESULTS_DIR}/calibration_comparison.csv")

    # Rule decided in advance: keep the simpler method (Platt) when results are very close
    best = "Platt (sigmoid)"
    print(f"\nChosen: {best}")
    with open(f"{RESULTS_DIR}/calibration_method.json", "w") as f:
        json.dump({"name": best, "method": METHODS[best]}, f, indent=2)

    plot_reliability(y_train, probas, "reliability_cv.pdf",
                     "Reliability diagram (5-fold cross-validation)")

    # Appendix: one panel per metric
    metrics = {"brier": "Brier score", "ece": "ECE", "average_precision": "PR-AUC",
               "precision": "Precision", "recall": "Recall"}
    best = {"brier": "min", "ece": "min", "average_precision": "max"}
    plot_metric_bars({"CV": results}, metrics, METHOD_COLORS,
                     "calibration_bars.pdf", best=best)    


if __name__ == "__main__":
    main()