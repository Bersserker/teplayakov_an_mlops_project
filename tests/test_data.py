import pandas as pd

from src.data.clean_dataset import clean_data


def test_clean_data():
    # Arrange — подготовка данных
    df = pd.DataFrame(
        {"AGE": [20, 20, 20, 40], "default_payment_next_month": [1, 1, 0, 1]}
    )

    # Act — вызов функции
    result = clean_data(df)

    # Assert — проверка результата
    assert result.columns.to_list() == ["age", "default"]
    assert result["default"].isna().sum() == 0
    assert result.duplicated().sum() == 0
