# tests/test_raw_schema.py
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from src.data.validation import RAW_SCHEMA


# Фикстуры
@pytest.fixture(scope="module")
def valid_df() -> pd.DataFrame:
    """Синтетический валидный датасет."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "ID": np.arange(n, dtype="int64"),
            "LIMIT_BAL": rng.uniform(0, 1_000_000, n),
            "SEX": rng.choice([1, 2], n).astype("int64"),
            "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], n).astype("int64"),
            "MARRIAGE": rng.choice([0, 1, 2, 3], n).astype("int64"),
            "AGE": rng.integers(18, 80, n).astype("int64"),
            "PAY_0": rng.integers(-3, 9, n).astype("int64"),
            "PAY_2": rng.integers(-3, 9, n).astype("int64"),
            "PAY_3": rng.integers(-3, 9, n).astype("int64"),
            "PAY_4": rng.integers(-3, 9, n).astype("int64"),
            "PAY_5": rng.integers(-3, 9, n).astype("int64"),
            "PAY_6": rng.integers(-3, 9, n).astype("int64"),
            "BILL_AMT1": rng.uniform(0, 500_000, n),
            "BILL_AMT2": rng.uniform(0, 500_000, n),
            "BILL_AMT3": rng.uniform(0, 500_000, n),
            "BILL_AMT4": rng.uniform(0, 500_000, n),
            "BILL_AMT5": rng.uniform(0, 500_000, n),
            "BILL_AMT6": rng.uniform(0, 500_000, n),
            "PAY_AMT1": rng.uniform(0, 100_000, n),
            "PAY_AMT2": rng.uniform(0, 100_000, n),
            "PAY_AMT3": rng.uniform(0, 100_000, n),
            "PAY_AMT4": rng.uniform(0, 100_000, n),
            "PAY_AMT5": rng.uniform(0, 100_000, n),
            "PAY_AMT6": rng.uniform(0, 100_000, n),
            "default.payment.next.month": rng.choice([0, 1], n).astype("int64"),
        },
        index=pd.Index(np.arange(n, dtype="int64")),
    )


@pytest.fixture(scope="module")
def unvalid_df() -> pd.DataFrame:
    """Синтетический валидный датасет."""
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame(
        {
            "ID": np.arange(n, dtype="int64"),
            "LIMIT_BAL": rng.uniform(0, 1_000_000, n),
            "SEX": rng.choice([1, 2], n).astype("int64"),
            "EDUCATION": rng.choice([0, 1, 2, 3], n).astype("int64"),
            "MARRIAGE": rng.choice([0, 1, 2, 3], n).astype("int64"),
            "AGE": rng.integers(18, 80, n).astype("int64"),
            "PAY_0": rng.integers(-10, 9, n).astype("int64"),
            "PAY_2": rng.integers(-3, 9, n).astype("int64"),
            "PAY_3": rng.integers(-4, 9, n).astype("int64"),
            "PAY_4": rng.integers(-3, 9, n).astype("int64"),
            "PAY_5": rng.integers(-3, 9, n).astype("int64"),
            "PAY_6": rng.integers(-3, 9, n).astype("int64"),
            "BILL_AMT1": rng.uniform(0, 500_000, n),
            "BILL_AMT2": rng.uniform(0, 500_000, n),
            "BILL_AMT3": rng.uniform(0, 500_000, n),
            "BILL_AMT4": rng.uniform(0, 500_000, n),
            "BILL_AMT5": rng.uniform(0, 500_000, n),
            "BILL_AMT6": rng.uniform(0, 500_000, n),
            "PAY_AMT1": rng.uniform(0, 100_000, n),
            "PAY_AMT2": rng.uniform(0, 100_000, n),
            "PAY_AMT3": rng.uniform(0, 100_000, n),
            "PAY_AMT4": rng.uniform(0, 100_000, n),
            "PAY_AMT5": rng.uniform(0, 100_000, n),
            "PAY_AMT6": rng.uniform(0, 100_000, n),
            "default.payment.next.month": rng.choice([0, 1], n).astype("int64"),
        },
        index=pd.Index(np.arange(n, dtype="int64")),
    )


# Позитивный тест


def test_valid_dataset_passes(valid_df):
    RAW_SCHEMA.validate(valid_df, lazy=True)


def test_empty_dataset_fails(valid_df):
    with pytest.raises(pa.errors.SchemaErrors):
        RAW_SCHEMA.validate(valid_df.iloc[0:0].copy(), lazy=True)


#  тест с ошибкой
def test_anomaly_fails_validation(unvalid_df):
    """Портим одну колонку — схема должна упасть."""
    df = unvalid_df.copy()

    with pytest.raises(pa.errors.SchemaErrors):
        RAW_SCHEMA.validate(df, lazy=True)
