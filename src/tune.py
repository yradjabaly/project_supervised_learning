"""Hyperparameter tuning with Optuna (step 6) and experiment tracking with MLflow (step 7).

Each model gets the same budget of N_TRIALS trials. Every trial is scored with the
same 5-fold cross-validation as step 4, on the training set only: the test set is
not used. Every trial is logged in MLflow with its parameters and metrics.
Run src.models first: the default scores (results/model_comparison.csv) are
needed for the default-versus-tuned figure.
"""
import json
import os

import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import (FIG_DIR, MLFLOW_EXPERIMENT, MLFLOW_URI, N_TRIALS,
                        RANDOM_STATE, RESULTS_DIR)
from src.data import prepare_data
from src.models import cross_validate_model

MODEL_NAMES = ["Logistic regression", "Random forest", "LightGBM"]


def suggest_model(trial, name, max_weight):
    """Search space of each model. w is the weight of the positive class (Section 3.2)."""
    w = trial.suggest_float("w", 1.0, max_weight)
    class_weight = {0: 1, 1: w}

    if name == "Logistic regression":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 10.0, log=True),
            class_weight=class_weight, max_iter=2000, random_state=RANDOM_STATE)

    if name == "Random forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=100),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_int("max_features", 4, 28),
            class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1)

    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 1000, step=50),
        num_leaves=trial.suggest_int("num_leaves", 7, 63),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        class_weight=class_weight, random_state=RANDOM_STATE, verbose=-1)


def tune_model(name, X_train, y_train, max_weight):
    """Run one Optuna study. Each trial is an MLflow run nested in the model's run."""

    def objective(trial):
        model = suggest_model(trial, name, max_weight)
        scores = cross_validate_model(model, X_train, y_train)
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metrics(scores)
        trial.set_user_attr("scores", scores)
        return scores["average_precision"]

    # Fixed seed: the same trials are proposed at every run
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS)
    return study


def plot_histories(studies):
    """Cross-validated PR-AUC of each trial, and the best value found so far."""
    fig, axes = plt.subplots(1, len(studies), figsize=(12, 3.5))
    for ax, (name, study) in zip(axes, studies.items()):
        values = pd.Series([trial.value for trial in study.trials])
        ax.plot(values.index, values, "o", alpha=0.5, label="Trial")
        ax.plot(values.index, values.cummax(), color="black", label="Best so far")
        ax.set_title(name)
        ax.set_xlabel("Trial")
    axes[0].set_ylabel("Cross-validated PR-AUC")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/optuna_history.pdf")
    plt.close(fig)


def plot_default_vs_tuned(tuned):
    """Default (step 4) versus tuned scores, all from 5-fold cross-validation.
    Left: cross-validated and training PR-AUC (the gap shows overfitting).
    Right: precision and recall at the default threshold of 0.5."""
    default = pd.read_csv(f"{RESULTS_DIR}/model_comparison.csv", index_col=0)
    panels = [
        ("PR-AUC\n(solid: cross-validation, dashed: training)",
         "average_precision", "train_average_precision"),
        ("Precision and recall at threshold 0.5\n(solid: precision, dashed: recall)",
         "precision", "recall"),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, (title, solid, dashed) in zip(axes, panels):
        for name, color in zip(MODEL_NAMES, colors):
            ax.plot([0, 1], [default.loc[name, solid], tuned.loc[name, solid]],
                    "o-", color=color, label=name)
            ax.plot([0, 1], [default.loc[name, dashed], tuned.loc[name, dashed]],
                    "o--", color=color)
        ax.set_xticks([0, 1], ["Default", "Tuned"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(title)
    # One legend for both panels, below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(f"{FIG_DIR}/default_vs_tuned.pdf")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train, X_test, y_train, y_test = prepare_data()  # X_test is NOT used here
    max_weight = (y_train == 0).sum() / (y_train == 1).sum()  # n0 / n1
    print(f"Search range for w: [1, {max_weight:.2f}]")

    studies, rows, best_params = {}, {}, {}
    for name in MODEL_NAMES:
        print(f"Tuning {name} ({N_TRIALS} trials)...")
        with mlflow.start_run(run_name=name):
            study = tune_model(name, X_train, y_train, max_weight)
            best = study.best_trial
            mlflow.log_params({f"best_{key}": value for key, value in best.params.items()})
            mlflow.log_metrics({f"best_{key}": value
                                for key, value in best.user_attrs["scores"].items()})
            mlflow.log_dict(best.params, "best_params.json")

        studies[name] = study
        rows[name] = best.user_attrs["scores"]
        best_params[name] = best.params
        print(f"  best CV PR-AUC = {study.best_value:.3f}, parameters = {best.params}")

    results = pd.DataFrame(rows).T.round(3)
    print("\nBest trial of each model (5-fold cross-validation)\n", results.to_string())
    results.to_csv(f"{RESULTS_DIR}/tuning_results.csv")
    with open(f"{RESULTS_DIR}/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    plot_histories(studies)
    plot_default_vs_tuned(results)

    # Summary run: the comparison table and the tuning figures, as artifacts
    with mlflow.start_run(run_name="Tuning summary"):
        mlflow.log_artifact(f"{RESULTS_DIR}/tuning_results.csv")
        mlflow.log_artifact(f"{RESULTS_DIR}/best_params.json")
        mlflow.log_artifact(f"{FIG_DIR}/optuna_history.pdf")
        mlflow.log_artifact(f"{FIG_DIR}/default_vs_tuned.pdf")


if __name__ == "__main__":
    main()