import pandas as pd

from src.features import preprocess_features


def test_preprocess_features_builds_model_columns():
    raw = pd.DataFrame(
        [
            {
                "trans_date_trans_time": "2026-05-02 23:15:00",
                "amt": 125.50,
                "category": "shopping_net",
                "lat": 6.5244,
                "long": 3.3792,
                "merch_lat": 6.5244,
                "merch_long": 3.3792,
                "city_pop": 8000000,
                "dob": "1996-05-02",
                "gender": "M",
                "state": "LG",
            }
        ]
    )

    features = preprocess_features(raw)

    assert list(features.columns) == [
        "amt",
        "city_pop",
        "dist_to_merchant",
        "age",
        "hour",
        "day_of_week",
        "is_weekend",
        "category",
        "gender",
        "state",
    ]
    row = features.iloc[0]
    assert row["hour"] == 23
    assert row["day_of_week"] == 5
    assert row["is_weekend"] == 1
    assert row["age"] == 30
    assert row["dist_to_merchant"] == 0
    assert str(features["category"].dtype) == "category"


def test_preprocess_features_for_agent_returns_human_readable_columns():
    raw = pd.DataFrame(
        [
            {
                "trans_date_trans_time": "2026-05-02 03:15:00",
                "amt": 950.0,
                "category": "grocery_pos",
                "lat": 40.7128,
                "long": -74.0060,
                "merch_lat": 34.0522,
                "merch_long": -118.2437,
                "city_pop": 8336817,
                "dob": "1990-01-01",
                "gender": "F",
                "job": "Engineer",
                "city": "New York",
                "state": "NY",
            }
        ]
    )

    features = preprocess_features(raw, for_agent=True)

    assert list(features.columns) == [
        "category",
        "amt",
        "job",
        "age",
        "dist_to_merchant",
        "hour",
        "gender",
        "city",
        "state",
    ]
    assert features.iloc[0]["hour"] == 3
    assert features.iloc[0]["dist_to_merchant"] > 0
