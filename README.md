# Adult Census Income — Supervised Learning PoC

**Dataset:** Adult / Census Income (OpenML, version 2).
**Task:** predict whether annual income exceeds $50K (binary classification).

## Project structure
```
src/
├── config.py      settings shared by all scripts (seed, folds, trials, paths, colours)
├── data.py        loading, cleaning and stratified train/test split
├── features.py    feature engineering (CensusFeatureEngineer)
├── pipeline.py    the single pipeline: features + preprocessing + model
├── plots.py       bar chart shared by several steps (one panel per metric)
├── eda.py         step 1: exploratory analysis and figures
├── models.py      step 4: model comparison with default hyperparameters
├── tune.py        steps 6-7: Optuna tuning and MLflow tracking
├── explain.py     step 8: SHAP explanations of the final model
├── calibration.py step 9: choice of the calibration method (cross-validation)
├── evaluate.py    final evaluation on the test set (measurement only)
└── fairness.py    step 10: fairness audit by sex and mitigation (measurement only)
report/            LaTeX report and figures
results/           tables produced by the scripts
mlflow.db, mlruns/ MLflow tracking of the tuning (94 runs)
```

## Setup
```bash
pip install -r requirements.txt
```

## Run (from the repository root)
```bash
python -m src.eda           # data checks and EDA figures
python -m src.models        # model comparison (default hyperparameters)
python -m src.tune          # Optuna tuning of the three models, logged in MLflow (5-10 min)
python -m src.explain       # SHAP figures of the final LightGBM
python -m src.calibration   # compares no calibration, Platt and isotonic
python -m src.evaluate      # final evaluation on the test set
python -m src.fairness      # fairness audit (uses the test set, decides nothing)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the logged runs
```

`mlflow.db` is included, so the 94 logged runs can be browsed without re-running the
tuning. To re-run `src.tune`, first delete `mlflow.db` and `mlruns/`, otherwise the new
runs are added to the existing ones.

Figures are saved to `report/figures/`, tables to `results/`.
The test set (20%) is only used at the very end, by `src/evaluate.py` and `src/fairness.py`,
to measure the final model: every choice (model, hyperparameters, calibration, fairness
thresholds) is made before, on the training set.