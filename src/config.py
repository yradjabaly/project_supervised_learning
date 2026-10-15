"""Central configuration: paths, seed, split sizes and column groups."""
from pathlib import Path

SEED = 42
POSITIVE_LABEL = ">50K"
TEST_SIZE = 0.20   # fraction of the full dataset
VAL_SIZE = 0.20    # fraction of the full dataset

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"
RESULTS_DIR = ROOT / "results"