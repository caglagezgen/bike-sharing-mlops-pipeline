import pandas as pd

from src.features import build_feature_frame, get_feature_columns


def test_build_feature_frame_adds_columns():
    frame = pd.DataFrame(
        [
            {
                "datetime": "2011-01-01 00:00:00",
                "season": 1,
                "holiday": 0,
                "weather": 1,
                "temp": 9.84,
                "humidity": 81,
                "windspeed": 0.0,
            }
        ]
    )

    features = build_feature_frame(frame)
    assert list(features.columns) == get_feature_columns()
    assert len(features) == 1
