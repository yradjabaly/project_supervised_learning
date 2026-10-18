"""Final evaluation on the test set.

The final LightGBM is fitted on the whole training set, without and with the
calibration chosen in step 9, and both versions are scored once on the test set.
Nothing is chosen from these results.
"""
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, roc_auc_score

from src.calibration import METHOD_COLORS, build_calibrated_model, probability_scores
from src.config import FIG_DIR, RESULTS_DIR
from src.data import prepare_data
from src.plots import plot_metric_bars


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data()
    with open(f"{RESULTS_DIR}/calibration_method.json") as f:
        chosen = json.load(f)

    versions = {"Uncalibrated": None, chosen["name"]: chosen["method"]}
    rows, predictions = {}, pd.DataFrame({"income": y_test, "sex": X_test["sex"]})
    for name, method in versions.items():
        model = build_calibrated_model(method).fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        rows[name] = probability_scores(y_test, proba)
        rows[name]["roc_auc"] = roc_auc_score(y_test, proba)
        rows[name]["f1"] = f1_score(y_test, proba >= 0.5)
        predictions[f"proba_{name}"] = proba

    results = pd.DataFrame(rows).T
    print("Final evaluation on the test set\n", results.round(4).to_string())
    results.round(4).to_csv(f"{RESULTS_DIR}/test_results.csv")
    # Kept for the fairness audit (step 10)
    predictions.to_csv(f"{RESULTS_DIR}/test_predictions.csv", index=False)

    # Confusion matrix of the final (calibrated) model, threshold 0.5
    final = predictions[f"proba_{chosen['name']}"] >= 0.5
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ConfusionMatrixDisplay.from_predictions(y_test, final, display_labels=["<=50K", ">50K"],
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/confusion_matrix.pdf")
    plt.close(fig)

    # Appendix: one panel per metric
    metrics = {"average_precision": "PR-AUC", "precision": "Precision", "recall": "Recall",
               "f1": "F1", "brier": "Brier score", "ece": "ECE"}
    best = {"average_precision": "max", "precision": "max", "recall": "max", "f1": "max",
            "brier": "min", "ece": "min"}
    plot_metric_bars({"Test": results}, metrics, METHOD_COLORS, "test_bars.pdf", best=best)    


if __name__ == "__main__":
    main()