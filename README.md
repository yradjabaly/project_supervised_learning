# Adult Census Income — Supervised Learning PoC

**Dataset:** Adult / Census Income (OpenML, version 2).
**Task:** predict whether annual income exceeds $50K (binary classification).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
## Run (from the repository root)
```bash
python -m src.eda               # data checks and EDA figures
python -m src.models            # class weighting (step 3) and model comparison (step 4)
```
Figures are saved to `report/figures/`, tables to `results/`.
The test set (20%) is never used by these scripts.