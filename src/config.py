"""Settings shared by all scripts."""

RANDOM_STATE = 42
TEST_SIZE = 0.20      # share of the data kept for the final evaluation
N_FOLDS = 5           # folds of the stratified cross-validation
N_TRIALS = 30         # Optuna trials per model (same budget for every model)

FIG_DIR = "report/figures"
RESULTS_DIR = "results"
MLFLOW_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "adult_income"

# Colours shared by all figures (first three slots of a colour-blind-safe palette)
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a9a"
MODEL_COLORS = {"Logistic regression": BLUE, "Random forest": ORANGE, "LightGBM": AQUA}