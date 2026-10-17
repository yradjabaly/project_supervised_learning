# Adult Census Income — Supervised Learning PoC

**Dataset:** Adult / Census Income (OpenML, version 2).
**Task:** predict whether annual income exceeds $50K (binary classification).

## Project structure
```
src/
├── config.py      settings shared by all scripts (seed, folds, trials, paths)
├── data.py        loading, cleaning and stratified train/test split
├── features.py    feature engineering (CensusFeatureEngineer)
├── pipeline.py    the single pipeline: features + preprocessing + model
├── eda.py         step 1: exploratory analysis and figures
├── models.py      step 4: model comparison with default hyperparameters
└── tune.py        steps 6-7: Optuna tuning and MLflow tracking
report/            LaTeX report and figures
results/           tables produced by the scripts
```

## Setup
```bash
pip install -r requirements.txt
```
## Run (from the repository root)
```bash
python -m src.eda      # data checks and EDA figures
python -m src.models   # model comparison (default hyperparameters)
python -m src.tune     # Optuna tuning of the three models, logged in MLflow (about 40 min)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the logged runs
```
Figures are saved to `report/figures/`, tables to `results/`.
The test set (20%) is not used by these scripts: it is kept for the final evaluation.