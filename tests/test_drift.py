import pandas as pd

from src.monitoring import simulate


def test_simulate_drift():
    df = pd.DataFrame(
        {
            "limit_bal": [10000, 20000],
            "age": [25, 40],
        }
    )
    original = df.copy(deep=True)

    result = simulate.simulate_drift(df, factor=0.5)

    assert result["limit_bal"].tolist() == [5000, 10000]
    pd.testing.assert_series_equal(result["age"], df["age"])
    pd.testing.assert_frame_equal(df, original)
