from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "fraudTrain.csv"

MODEL_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'tree_method': 'hist',           # Fast histogram-based training
    'enable_categorical': True,      # THE MAGIC FLAG
    'use_label_encoder': False,
    'eval_metric': 'aucpr',          # Best for fraud (Precision-Recall)
    'random_state': 42
}