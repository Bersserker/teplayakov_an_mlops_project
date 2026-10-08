"""Загрузка исходного датасета и подготовка данных."""

from pathlib import Path

import kagglehub

from src.config import RAW_DATA_DIR, RAW_DATA_PATH
from src.data.prepare_dataset import prepare_dataset

DATASET = "uciml/default-of-credit-card-clients-dataset"


def load_data(path: Path = RAW_DATA_DIR) -> Path:
    """Загрузка исходного в папку data/raw/"""
    path = Path(path)
    dataset_path = path / RAW_DATA_PATH.name
    path.mkdir(parents=True, exist_ok=True)
    if not dataset_path.is_file():
        kagglehub.dataset_download(DATASET, output_dir=str(path), force_download=True)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Датасет {dataset_path.name} не найден в {path}")

    print(f"Датасет {dataset_path.name} загружен в {path}")
    return dataset_path


if __name__ == "__main__":
    prepare_dataset(load_data())
