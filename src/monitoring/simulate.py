"""Тестирование PSI на обученныз данных и данных со смещением"""

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from src.api.models import CreditRequest, CreditResponse
from src.config import REPORTS_DIR, TEST_REF_DATA, TRAIN_REF_DATA
from src.monitoring.drift import psi_calculation


def test_my_api(df, port=8000):
    """Тестирование PSI на обученныз данных и данных со смещением"""
    if df.empty:
        raise ValueError("The reference dataset is empty.")
    reference = df["default"].reset_index(drop=True)
    # The reference CSV also contains identifiers and engineered features.
    samples = df[list(CreditRequest.model_fields)].to_dict(orient="records")
    predictions = []
    for sample in samples:
        payload = CreditRequest.model_validate(sample).model_dump()
        resp = requests.post(
            f"http://127.0.0.1:{port}/predict", json=payload, timeout=30
        )
        resp.raise_for_status()
        prediction = CreditResponse.model_validate(resp.json()).model_dump()
        predictions.append(prediction)

    results = pd.DataFrame(predictions)
    results.insert(0, "default", reference)
    return results


def simulate_drift(df, feature="pay_0", shift=1):
    """Создаем дрифт в данных"""
    drifted_df = df.copy()

    drifted_df[feature] = drifted_df[feature] + shift

    return drifted_df


def save_test_report(
    feature_psi: float,
    probability_psi: float,
    reports_dir: Path,
    comparison: str = "test",
) -> dict[str, float]:
    """Формируемм отчет"""
    if comparison not in {"test", "drifted_test"}:
        raise ValueError("comparison must be 'test' or 'drifted_test'")
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "pay_0_psi": float(feature_psi),
        "default_probability_psi": float(probability_psi),
    }
    report_path = reports_dir / f"psi_{comparison}"
    report_path.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    label = (
        "тестовая выборка" if comparison == "test" else "тестовая выборка со сдвигом"
    )
    rows = "\n".join(
        f"| {feature} | {value:.6f} |"
        for feature, value in (
            ("pay_0", metrics["pay_0_psi"]),
            ("default_probability", metrics["default_probability_psi"]),
        )
    )
    report_path.with_suffix(".md").write_text(
        "# Результаты PSI-теста\n\n"
        f"Сравнение: обучающая выборка и {label}.\n\n"
        "Синтетический сдвиг: увеличение `pay_0` на 1 "
        "в изменённой тестовой выборке.\n\n"
        "| Признак | PSI |\n| --- | ---: |\n"
        f"{rows}\n",
        encoding="utf-8",
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Сравните тестовые данные с обучающим референсом с помощью PSI (Population Stability Index)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="API port (default: 8000)"
    )
    args = parser.parse_args()
    train_df = pd.read_csv(TRAIN_REF_DATA)
    test_df = pd.read_csv(TEST_REF_DATA)
    drifted_df = simulate_drift(test_df)
    reference_predictions = test_my_api(train_df, port=args.port)
    for comparison, current_df in (("test", test_df), ("drifted_test", drifted_df)):
        label = comparison.replace("_", " ")
        current_predictions = test_my_api(current_df, port=args.port)
        feature_psi = psi_calculation(train_df["pay_0"], current_df["pay_0"])
        probability_psi = psi_calculation(
            reference_predictions["default_probability"],
            current_predictions["default_probability"],
        )
        save_test_report(feature_psi, probability_psi, REPORTS_DIR, comparison)
        print(f"PSI pay_0 (train vs {label}): {feature_psi:.6f}")
        print(f"PSI default_probability (train vs {label}): {probability_psi:.6f}")


if __name__ == "__main__":
    main()
