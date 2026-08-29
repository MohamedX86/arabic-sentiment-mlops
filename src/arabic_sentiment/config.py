from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

RANDOM_SEED = 42
NUM_LABELS = 3

LABEL_NAMES = [
    "negative",
    "neutral",
    "positive",
]