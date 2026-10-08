"""Отчистка данных без предобработки"""

import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Отчистка данных без предобработки"""
    df = df.copy().drop_duplicates()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "_", regex=False)
    )

    df = df.rename(columns={"default_payment_next_month": "default"})
    df = df.dropna(subset=["default"]).reset_index(drop=True)

    return df