import pandas as pd
import pytest

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


def test_monitoring_uses_training_reference(tmp_path, monkeypatch, capsys):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"limit_bal": [10000, 20000, 30000]}).to_csv(train_path, index=False)
    pd.DataFrame({"limit_bal": [40000, 50000]}).to_csv(test_path, index=False)
    monkeypatch.setattr(simulate, "TRAIN_REF_DATA", train_path)
    monkeypatch.setattr(simulate, "TEST_REF_DATA", test_path)
    monkeypatch.setattr("sys.argv", ["simulate", "--port", "8123"])

    def predictions(df, port):
        assert port == 8123
        return pd.DataFrame({"default_probability": df["limit_bal"] / 100000})

    monkeypatch.setattr(simulate, "test_my_api", predictions)
    comparisons = []

    def capture_psi(reference, current):
        comparisons.append((reference.tolist(), current.tolist()))
        return 0.1

    monkeypatch.setattr(simulate, "psi_calculation", capture_psi)
    simulate.main()

    assert len(comparisons) == 4
    assert comparisons[0] == ([10000, 20000, 30000], [40000, 50000])
    assert comparisons[2] == ([10000, 20000, 30000], [28000, 35000])
    assert comparisons[1][0] == pytest.approx([0.1, 0.2, 0.3])
    assert comparisons[1][1] == pytest.approx([0.4, 0.5])
    assert comparisons[3][0] == pytest.approx([0.1, 0.2, 0.3])
    assert comparisons[3][1] == pytest.approx([0.28, 0.35])
    assert "train vs test" in capsys.readouterr().out
