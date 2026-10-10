"""Тестовый скрипт для проверки работы API."""

import argparse

import pandas as pd
import requests

from src.config import RAW_DATA_PATH


def test_my_api(line, port=8000):
    """Запрос к API и вывод результата."""
    df = pd.read_csv(RAW_DATA_PATH).drop(columns=["ID", "default.payment.next.month"])
    df.columns = df.columns.str.lower()
    test_sample = df.iloc[[line]].to_dict(orient="records")[0]
    resp = requests.post(
        f"http://127.0.0.1:{port}/predict", json=test_sample, timeout=30
    )
    print(resp.status_code)
    print(resp.json())
    resp.raise_for_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the credit scoring API.")
    parser.add_argument(
        "--port", type=int, default=8000, help="API port (default: 8000)"
    )
    args = parser.parse_args()
    test_my_api(0, port=args.port)
    test_my_api(2, port=args.port)
