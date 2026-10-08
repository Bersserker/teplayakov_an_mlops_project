"""Схемы валидации данных
Схемы Pandera для исходных и очищенных наборов данных для кредитного скоринга.
Обе схемы требуют наличия всех ожидаемых столбцов и отклоняют записи с пропущенными значениями.
Дополнительные столбцы допускаются; значения приводятся к заявленным типам.
"""

import pandera.pandas as pa

# Создание схемы для исходных данных
RAW_SCHEMA = pa.DataFrameSchema(
    {
        "ID": pa.Column("int64", pa.Check.ge(0), nullable=False),
        "LIMIT_BAL": pa.Column("float64", pa.Check.ge(0), nullable=False),
        "SEX": pa.Column("int64", pa.Check.isin([1, 2]), nullable=False),
        "EDUCATION": pa.Column(
            "int64", pa.Check.isin([0, 1, 2, 3, 4, 5, 6]), nullable=False
        ),
        "MARRIAGE": pa.Column("int64", pa.Check.isin([0, 1, 2, 3]), nullable=False),
        "AGE": pa.Column("int64", pa.Check.ge(0), nullable=False),
        "PAY_0": pa.Column("int64", pa.Check.ge(-3), nullable=False),
        "PAY_2": pa.Column("int64", pa.Check.ge(-3), nullable=False),
        "PAY_3": pa.Column("int64", pa.Check.ge(-3), nullable=False),
        "PAY_4": pa.Column("int64", pa.Check.ge(-3), nullable=False),
        "PAY_5": pa.Column("int64", pa.Check.ge(-3), nullable=False),
        "PAY_6": pa.Column("int64", pa.Check.ge(-3), nullable=False),
        "BILL_AMT1": pa.Column("float64", nullable=False),
        "BILL_AMT2": pa.Column("float64", nullable=False),
        "BILL_AMT3": pa.Column("float64", nullable=False),
        "BILL_AMT4": pa.Column("float64", nullable=False),
        "BILL_AMT5": pa.Column("float64", nullable=False),
        "BILL_AMT6": pa.Column("float64", nullable=False),
        "PAY_AMT1": pa.Column("float64", pa.Check.ge(0), nullable=False),
        "PAY_AMT2": pa.Column("float64", pa.Check.ge(0), nullable=False),
        "PAY_AMT3": pa.Column("float64", pa.Check.ge(0), nullable=False),
        "PAY_AMT4": pa.Column("float64", pa.Check.ge(0), nullable=False),
        "PAY_AMT5": pa.Column("float64", pa.Check.ge(0), nullable=False),
        "PAY_AMT6": pa.Column("float64", pa.Check.ge(0), nullable=False),
        "default.payment.next.month": pa.Column(
            "int64", pa.Check.isin([0, 1]), nullable=False
        ),
    },
    index=pa.Index("int64", nullable=False),
    coerce=True,
    checks=pa.Check(lambda df: len(df) > 0, error="Данные не могут быть пустыми"),
    strict=False,
    name="raw_credit_dataset",
)
# Создание схемы для отчищенных данных
PROCESSED_SCHEMA = RAW_SCHEMA.rename_columns(
    {
        column: "default" if column == "default.payment.next.month" else column.lower()
        for column in RAW_SCHEMA.columns
    }
)
PROCESSED_SCHEMA.name = "processed_credit_dataset"
PROCESSED_SCHEMA.checks = [
    *PROCESSED_SCHEMA.checks,
    pa.Check(
        lambda df: not df.duplicated().any(), error="В данных есть дублирующиеся записи"
    ),
]
