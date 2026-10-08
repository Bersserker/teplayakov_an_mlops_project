"""Загрузка исходного датасета, отчистка данных и сохранение валидированных данных"""
from pathlib import Path
import argparse
from src.data.clean_dataset import clean_data
from src.data.validation import PROCESSED_SCHEMA, RAW_SCHEMA
from src.features.build_features import build_features
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "UCI_Credit_Card.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "UCI_Credit_Card.csv"

def prepare_dataset(
    raw_path: Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
) -> Path:
    """Загрузка исходного датасета, отчистка данных и сохранение валидированных данных"""
    raw = RAW_SCHEMA.validate(pd.read_csv(raw_path), lazy=True)
    processed = PROCESSED_SCHEMA.validate(build_features(clean_data(raw)), lazy=True)
    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False)
    print(f"Данные проверены и очищены: {processed_path} ({len(processed)} строк)")
    return processed_path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--processed-path", type=Path, default=PROCESSED_DATA_PATH)
    args = parser.parse_args()
    prepare_dataset(args.raw_path, args.processed_path)

if __name__ == "__main__":
    main()
