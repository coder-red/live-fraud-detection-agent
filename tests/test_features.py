import pytest
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

    assert list(features.columns) == [  # The output columns must be exactly what the model expects.
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
    row = features.iloc[0]  # This selects the first processed row.
    assert row["hour"] == 23  # The hour should be taken from the transaction time.
    assert row["day_of_week"] == 5  # May 2, 2026 is a Saturday, so pandas returns 5.
    assert row["is_weekend"] == 1  # Saturday should be marked as weekend.
    assert row["age"] == 30  # The customer should be 30 on this transaction date.
    assert row["dist_to_merchant"] == 0  # Same customer and merchant location should give zero distance.
    assert str(features["category"].dtype) == "category"  # XGBoost expects this column as a categorical type.

def test_preprocess_features_for_agent_returns_human_readable_columns():  # This checks the agent-facing output.
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

    features = preprocess_features(raw, for_agent=True)  # This runs the agent version of the feature code.

    assert list(features.columns) == [  # The agent output should use readable columns.
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
    assert features.iloc[0]["hour"] == 3  # The hour should be taken from the transaction time.
    assert features.iloc[0]["dist_to_merchant"] > 0  # New York and Los Angeles should not be zero distance apart.


def test_preprocess_features_marks_sunday_as_weekend():  # EDGE CASE: this checks the other weekend day, not just Saturday.
    raw = pd.DataFrame(
        [
            {
                "trans_date_trans_time": "2026-05-03 12:00:00",
                "amt": 50.0,
                "category": "misc_net",
                "lat": 6.5244,
                "long": 3.3792,
                "merch_lat": 6.5244,
                "merch_long": 3.3792,
                "city_pop": 8000000,
                "dob": "2000-05-03",
                "gender": "F",
                "state": "LG",
            }
        ]
    )

    features = preprocess_features(raw)  # This runs the normal model feature code.

    assert features.iloc[0]["day_of_week"] == 6  # May 3, 2026 is a Sunday, so pandas returns 6.
    assert features.iloc[0]["is_weekend"] == 1  # Sunday should be marked as weekend.


def test_preprocess_features_rejects_bad_datetime():  # INVALID CASE: datetime text must be parseable by pandas.
    raw = pd.DataFrame(
        [
            {
                "trans_date_trans_time": "not-a-date",
                "amt": 50.0,
                "category": "misc_net",
                "lat": 6.5244,
                "long": 3.3792,
                "merch_lat": 6.5244,
                "merch_long": 3.3792,
                "city_pop": 8000000,
                "dob": "2000-05-03",
                "gender": "F",
                "state": "LG",
            }
        ]
    )

    with pytest.raises(ValueError):  # Pandas should raise because "not-a-date" is not a real datetime.
        preprocess_features(raw)  # This sends the invalid datetime into feature engineering.
