import pandas as pd
from src.data import time_split, get_xy
from src.threshold import total_cost

def make_df():
    return pd.DataFrame({
        "Time":   [5, 1, 3, 2, 4, 6, 8, 7, 10, 9],
        "V1":     list(range(10)),
        "Amount": [10.0] * 10,
        "Class":  [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    })

def test_split_is_in_time_order():
    train, valid, test = time_split(make_df())
    assert train["Time"].max() < valid["Time"].min()
    assert valid["Time"].max() < test["Time"].min()

def test_split_covers_all_rows():
    train, valid, test = time_split(make_df())
    assert len(train) + len(valid) + len(test) == 10

def test_get_xy_drops_time_and_label():
    X, y = get_xy(make_df())
    assert "Time" not in X.columns and "Class" not in X.columns

def test_total_cost():
    # one missed fraud (amount 100) + one false alarm (cost 5)
    cost = total_cost([1, 0], [0.1, 0.9], [100.0, 20.0], threshold=0.5, fp_cost=5.0)
    assert cost == 105.0