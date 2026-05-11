import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs
from src.features.engineering import add_time_features, get_feature_columns


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_data_profile(frame: pd.DataFrame) -> dict:
    profile = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": {},
    }

    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        column_info = {
            "dtype": str(series.dtype),
            "missing": missing,
            "unique": unique,
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if not clean.empty:
                column_info.update(
                    {
                        "min": float(clean.min()),
                        "max": float(clean.max()),
                        "mean": float(clean.mean()),
                        "std": float(clean.std(ddof=0)),
                    }
                )
            else:
                column_info.update({"min": None, "max": None, "mean": None, "std": None})

        profile["columns"][column] = column_info

    return profile


def _winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip outliers to the given quantile bounds (Winsorization).

    Winsorization caps extreme values at a chosen percentile rather than
    removing the rows. This preserves dataset size — critical when training
    data is limited — while preventing outliers from dominating the squared-
    error loss function. MSE and RMSLE penalise large residuals quadratically,
    so a single extreme value can disproportionately skew gradient updates.

    The 1st/99th percentile bounds are a standard choice: conservative enough
    to leave the bulk of the distribution untouched while clipping the tails
    that are most likely measurement artefacts or edge-case conditions.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def preprocess(train_path: Path, output_path: Path) -> None:
    raw_frame = pd.read_csv(train_path)
    raw_rows = int(len(raw_frame))
    raw_columns = [str(col) for col in raw_frame.columns]

    # casual and registered are sub-components of the target (count = casual + registered).
    # Including them would cause target leakage — the model would trivially learn
    # count ≈ casual + registered and generalise poorly to inference time when
    # these columns are unavailable.
    # atemp is dropped for multicollinearity reasons (see src/features/engineering.py).
    frame = raw_frame.drop(columns=["casual", "registered", "atemp"], errors="ignore")

    # Apply Winsorization to the three columns identified by IQR analysis as
    # having meaningful outlier rates:
    #   weather:   categorical but can have erroneous extreme codes
    #   humidity:  tails represent unusual atmospheric conditions unlikely to
    #              repeat at the same frequency in production data
    #   windspeed: highest outlier rate in the dataset (1.91% of rows outside
    #              the IQR fence), likely from sensor spikes or storm events
    for col in ["weather", "humidity", "windspeed"]:
        if col in frame.columns:
            frame[col] = _winsorize(frame[col])

    frame = add_time_features(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    feature_config = {
        "target": "count",
        "feature_columns": get_feature_columns(),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "feature_config.json").write_text(
        json.dumps(feature_config, indent=2)
    )

    dataset_meta = {
        "source": "kaggle/bike-sharing-demand",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "raw_path": str(train_path),
        "raw_sha256": file_sha256(train_path),
        "raw_rows": raw_rows,
        "raw_columns": raw_columns,
        "processed_path": str(output_path),
        "processed_sha256": file_sha256(output_path),
        "processed_rows": int(len(frame)),
        "processed_columns": [str(col) for col in frame.columns],
    }
    (ARTIFACTS_DIR / "dataset_meta.json").write_text(
        json.dumps(dataset_meta, indent=2)
    )

    data_profile = build_data_profile(frame)
    (ARTIFACTS_DIR / "data_profile.json").write_text(
        json.dumps(data_profile, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess bike sharing data")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR / "train.csv",
        help="Path to raw train.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "train_processed.csv",
        help="Path to processed dataset",
    )
    args = parser.parse_args()

    ensure_dirs()
    preprocess(args.input, args.output)


if __name__ == "__main__":
    main()
