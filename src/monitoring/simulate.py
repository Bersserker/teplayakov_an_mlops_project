"""Compare held-out and shifted data with the training reference using PSI."""

import argparse

import pandas as pd
import requests

from src.api.models import CreditRequest, CreditResponse
from src.config import TEST_REF_DATA, TRAIN_REF_DATA
from src.monitoring.drift import psi_calculation


def test_my_api(df, port=8000):
    """Return API predictions aligned with the reference target labels."""
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


def simulate_drift(df, feature="limit_bal", factor=0.7):
    drifted_df = df.copy()

    drifted_df[feature] = drifted_df[feature] * factor

    return drifted_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare test data with the training reference using PSI."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="API port (default: 8000)"
    )
    args = parser.parse_args()
    train_df = pd.read_csv(TRAIN_REF_DATA)
    test_df = pd.read_csv(TEST_REF_DATA)
    drifted_df = simulate_drift(test_df)
    reference_predictions = test_my_api(train_df, port=args.port)
    for label, current_df in (("test", test_df), ("drifted test", drifted_df)):
        current_predictions = test_my_api(current_df, port=args.port)
        feature_psi = psi_calculation(train_df["limit_bal"], current_df["limit_bal"])
        probability_psi = psi_calculation(
            reference_predictions["default_probability"],
            current_predictions["default_probability"],
        )
        print(f"PSI limit_bal (train vs {label}): {feature_psi:.6f}")
        print(f"PSI default_probability (train vs {label}): {probability_psi:.6f}")


if __name__ == "__main__":
    main()
