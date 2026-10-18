"""SHAP explanations of the final LightGBM model (step 8).

The model is fitted on the whole training set and explained on the whole training
set: SHAP describes the behaviour of the model, it does not evaluate it, so the
test set is not needed here.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import FIG_DIR, RESULTS_DIR
from src.data import prepare_data
from src.models import get_final_model
from src.pipeline import NUM_COLS, build_pipeline

N_DISPLAY = 12  # number of features shown in the beeswarm plot


def compute_shap_values(pipe, X):
    """SHAP values of the fitted pipeline's model, for the rows of X.
    For LightGBM, SHAP values are on the log-odds scale, not the probability scale."""
    X_model = pipe[:-1].transform(X)          # feature engineering + preprocessing
    if hasattr(X_model, "toarray"):           # the one-hot output can be a sparse matrix
        X_model = X_model.toarray()
    shap_values = shap.TreeExplainer(pipe.named_steps["model"])(X_model)
    # "cat__sex_Female" -> "sex_Female"
    names = pipe.named_steps["prep"].get_feature_names_out()
    shap_values.feature_names = [name.split("__")[1] for name in names]

    # Show the original values in the plots, not the standardised ones
    # (the numerical features are the first columns of the preprocessed data)
    X_features = pipe.named_steps["features"].transform(X)
    for j, col in enumerate(NUM_COLS):
        shap_values.data[:, j] = X_features[col].values
    return shap_values


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data()  # X_test is NOT used here

    pipe = build_pipeline(get_final_model()).fit(X_train, y_train)
    shap_values = compute_shap_values(pipe, X_train)

    # Global importance: mean absolute SHAP value of each feature
    importance = pd.Series(np.abs(shap_values.values).mean(axis=0),
                           index=shap_values.feature_names).sort_values(ascending=False)
    print("Mean |SHAP| (log-odds), top features\n", importance.head(N_DISPLAY).round(3).to_string())
    importance.round(4).to_csv(f"{RESULTS_DIR}/shap_importance.csv", header=["mean_abs_shap"])

    # Global view: one point per person and per feature
    shap.plots.beeswarm(shap_values, max_display=N_DISPLAY, show=False)
    plt.gcf().set_size_inches(4.5, 4)
    plt.savefig(f"{FIG_DIR}/shap_beeswarm.pdf", bbox_inches="tight")
    plt.close()

    # Local view: the first high earner of the training set that the model predicts correctly
    proba = pipe.predict_proba(X_train)[:, 1]
    i = np.where((y_train.values == 1) & (proba >= 0.5))[0][0]
    print(f"\nLocal explanation: row {i}, predicted probability = {proba[i]:.3f}, "
          f"baseline (log-odds) = {shap_values.base_values[i]:.3f}")
    print(X_train.iloc[i].to_string())
    shap.plots.waterfall(shap_values[i], max_display=10, show=False)
    plt.gcf().set_size_inches(4.5, 4)
    plt.savefig(f"{FIG_DIR}/shap_waterfall.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()