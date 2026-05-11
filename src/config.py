from pathlib import Path

# Resolve paths relative to the repo root so the project works regardless
# of the working directory the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # Kaggle downloads land here
PROCESSED_DIR = DATA_DIR / "processed"  # Output of preprocess.py
BRONZE_DIR = DATA_DIR / "bronze"    # New data dropped here triggers CT workflow
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"  # Metrics, metadata, feature config
MODEL_DIR = ARTIFACTS_DIR / "model"  # Serialised model file (model.joblib)


def ensure_dirs() -> None:
    # Called at the start of each pipeline step to guarantee directories exist
    # before any file writes happen.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
