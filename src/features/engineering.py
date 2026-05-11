import pandas as pd

# --- Feature selection rationale ---
#
# BASE_FEATURES are raw columns from the Kaggle dataset that have meaningful
# predictive signal for bike demand.
#
# atemp DROPPED: Pearson correlation with temp = 0.99. Including both features
# causes multicollinearity — the model cannot distinguish their individual
# contributions. In tree-based models this also splits feature importance
# across both columns, inflating apparent importance of neither and making
# SHAP/feature importance plots misleading. Dropping atemp retains the same
# thermal information without the collinearity cost.
#
# workingday DROPPED: Binary flag (0/1). However, dayofweek (0=Monday … 6=Sunday)
# encodes strictly more information — weekday vs weekend is recoverable from
# dayofweek (dayofweek >= 5 → weekend), but the reverse is not true. Keeping
# both would introduce a near-redundant feature that only adds noise to the
# model's split decisions.
BASE_FEATURES = [
    "season",
    "holiday",
    "weather",
    "temp",
    "humidity",
    "windspeed",
]

# TIME_FEATURES are derived from the datetime column. They encode cyclical
# temporal patterns that drive rental behaviour.
#
# hour: Strongest single predictor in SHAP analysis. Demand peaks sharply at
# commute hours (08:00, 17:00–18:00) and drops overnight — a non-linear
# pattern that XGBoost captures well with tree splits.
#
# dayofweek: Encodes weekly rhythm (commuter vs leisure usage). More granular
# than the binary workingday flag.
#
# month / year: Capture seasonality and year-on-year growth trend (2011→2012
# showed meaningful growth in the Capital Bikeshare system).
#
# day DROPPED: Day of month has no cyclical relationship with demand. It was
# not among the top predictors in feature importance analysis and adds noise.
#
# is_weekend DROPPED: Entirely derivable from dayofweek (dayofweek >= 5).
# Including it creates a linearly dependent feature — a deterministic function
# of an existing column — which does not add new information but increases
# dimensionality.
TIME_FEATURES = [
    "year",
    "month",
    "hour",
    "dayofweek",
]


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    if "datetime" not in frame.columns:
        raise ValueError("datetime column is required")

    result = frame.copy()
    dt = pd.to_datetime(result["datetime"], errors="coerce")
    if dt.isna().any():
        raise ValueError("datetime contains invalid values")

    result["year"] = dt.dt.year
    result["month"] = dt.dt.month
    result["hour"] = dt.dt.hour
    result["dayofweek"] = dt.dt.dayofweek
    return result


def get_feature_columns() -> list[str]:
    return BASE_FEATURES + TIME_FEATURES


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = add_time_features(frame)
    feature_cols = get_feature_columns()
    missing = [col for col in feature_cols if col not in enriched.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return enriched[feature_cols]
