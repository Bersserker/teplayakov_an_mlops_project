import json

import pandas as pd
import pytest

from src.monitoring import simulate


def test_simulate_drift():
    df = pd.DataFrame(
        {
            "pay_0": [-2, 2],
            "age": [25, 40],
        }
    )
    original = df.copy(deep=True)

    result = simulate.simulate_drift(df)

    assert result["pay_0"].tolist() == [-1, 3]
    assert pd.api.types.is_integer_dtype(result["pay_0"])
    pd.testing.assert_series_equal(result["age"], df["age"])
    pd.testing.assert_frame_equal(df, original)


def test_monitoring_uses_training_reference(tmp_path, monkeypatch, capsys):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"pay_0": [-2, 0, 2]}).to_csv(train_path, index=False)
    pd.DataFrame({"pay_0": [1, 3]}).to_csv(test_path, index=False)
    monkeypatch.setattr(simulate, "TRAIN_REF_DATA", train_path)
    monkeypatch.setattr(simulate, "TEST_REF_DATA", test_path)
    monkeypatch.setattr(simulate, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr("sys.argv", ["simulate", "--port", "8123"])

    def predictions(df, port):
        assert port == 8123
        return pd.DataFrame({"default_probability": (df["pay_0"] + 2) / 10})

    monkeypatch.setattr(simulate, "test_my_api", predictions)
    comparisons = []

    def capture_psi(reference, current):
        comparisons.append((reference.tolist(), current.tolist()))
        return 0.1

    monkeypatch.setattr(simulate, "psi_calculation", capture_psi)
    simulate.main()

    assert len(comparisons) == 4
    assert comparisons[0] == ([-2, 0, 2], [1, 3])
    assert comparisons[2] == ([-2, 0, 2], [2, 4])
    assert comparisons[1][0] == pytest.approx([0.0, 0.2, 0.4])
    assert comparisons[1][1] == pytest.approx([0.3, 0.5])
    assert comparisons[3][0] == pytest.approx([0.0, 0.2, 0.4])
    assert comparisons[3][1] == pytest.approx([0.4, 0.6])
    assert "train vs test" in capsys.readouterr().out
    for comparison in ("test", "drifted_test"):
        report_path = tmp_path / "reports" / f"psi_{comparison}"
        assert json.loads(report_path.with_suffix(".json").read_text()) == {
            "pay_0_psi": 0.1,
            "default_probability_psi": 0.1,
        }
        assert "pay_0" in report_path.with_suffix(".md").read_text()


def test_save_test_report_keeps_both_comparisons(tmp_path):
    reports_dir = tmp_path / "reports"
    metrics = simulate.save_test_report(0.02, 0.03, reports_dir)
    drifted_metrics = simulate.save_test_report(0.4, 0.5, reports_dir, "drifted_test")

    assert metrics == {"pay_0_psi": 0.02, "default_probability_psi": 0.03}
    assert json.loads((reports_dir / "psi_test.json").read_text()) == metrics
    assert (
        json.loads((reports_dir / "psi_drifted_test.json").read_text())
        == drifted_metrics
    )
    report = (reports_dir / "psi_drifted_test.md").read_text(encoding="utf-8")
    assert "тестовая выборка со сдвигом" in report
    assert "| pay_0 | 0.400000 |" in report
    assert "| default_probability | 0.500000 |" in report
