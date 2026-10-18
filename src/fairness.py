"""Fairness audit by sex, with one mitigation (step 10).

Fairness definition (fixed before measuring): equal opportunity, i.e. the same true
positive rate (TPR) for men and women. Mitigation: Fairlearn's ThresholdOptimizer,
which chooses a decision threshold per group. The thresholds are learned on the
training set; the test set is only used to measure the gaps before and after.
"""
import json
import os

import pandas as pd
from fairlearn.metrics import (MetricFrame, false_positive_rate, selection_rate,
                               true_positive_rate)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import f1_score, precision_score, recall_score

from src.calibration import build_calibrated_model
from src.config import BLUE, ORANGE, RANDOM_STATE, RESULTS_DIR
from src.data import prepare_data
from src.plots import plot_metric_bars

SEX_COLORS = {"Female": BLUE, "Male": ORANGE}

METRICS = {
    "selection_rate": selection_rate,     # share predicted >50K
    "tpr": true_positive_rate,            # share of real high earners who are detected
    "fpr": false_positive_rate,           # share of the others wrongly predicted >50K
    "precision": precision_score,         # share of correct >50K predictions
}


def audit(y_true, y_pred, sex):
    """Metrics by sex, and the gap between the two groups."""
    frame = MetricFrame(metrics=METRICS, y_true=y_true, y_pred=y_pred, sensitive_features=sex)
    return frame.by_group, frame.difference()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data()
    with open(f"{RESULTS_DIR}/calibration_method.json") as f:
        chosen = json.load(f)

    # Final model (step 9), fitted on the training set
    model = build_calibrated_model(chosen["method"]).fit(X_train, y_train)

    # Mitigation: one threshold per sex, learned on the TRAINING set
    # Among the thresholds that equalise the TPR, keep those with the best accuracy:
    # with calibrated probabilities, this stays close to our reference threshold of 0.5
    optimizer = ThresholdOptimizer(
        estimator=model, constraints="true_positive_rate_parity",
        objective="accuracy_score", prefit=True, predict_method="predict_proba")
    optimizer.fit(X_train, y_train, sensitive_features=X_train["sex"])
    print("Thresholds learned on the training set:")
    for group, rule in optimizer.interpolated_thresholder_.interpolation_dict.items():
        print(f"  {group}: {rule}")

    # Measurement on the TEST set: same model, two decision rules
    sex_test = X_test["sex"]
    # Before: threshold 0.5 for everyone. After: thresholds of ThresholdOptimizer
    predictions = {
        "Before": (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int),
        "After": optimizer.predict(X_test, sensitive_features=sex_test,
                                   random_state=RANDOM_STATE),
    }

    groups = pd.DataFrame({"n": sex_test.value_counts(),
                           "base_rate": y_test.groupby(sex_test).mean()})
    print("\nTest set by sex\n", groups.round(3).to_string())

    tables, rows = {}, []
    for version, y_pred in predictions.items():
        by_group, gap = audit(y_test, y_pred, sex_test)
        tables[version] = by_group
        print(f"\n{version}\n", by_group.round(3).to_string())
        print(" Gap between groups:", gap.round(3).to_dict())
        print(f" Overall: precision = {precision_score(y_test, y_pred):.3f}, "
              f"recall = {recall_score(y_test, y_pred):.3f}, F1 = {f1_score(y_test, y_pred):.3f}")
        for group, values in by_group.iterrows():
            rows.append({"version": version, "sex": group, **values.to_dict()})
        rows.append({"version": version, "sex": "Gap", **gap.to_dict()})

    pd.DataFrame(rows).round(4).to_csv(f"{RESULTS_DIR}/fairness_results.csv", index=False)
    # Main text: TPR only (our fairness definition). Appendix: the four metrics.
    plot_metric_bars(tables, {"tpr": "True positive rate"}, SEX_COLORS,
                     "fairness_tpr.pdf", ylim=(0, 1))
    plot_metric_bars(tables, {"selection_rate": "Selection rate", "tpr": "TPR",
                              "fpr": "FPR", "precision": "Precision"},
                     SEX_COLORS, "fairness_bars.pdf", ylim=(0, 1))


if __name__ == "__main__":
    main()