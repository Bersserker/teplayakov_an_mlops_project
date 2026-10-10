"""Расположение всех тестовых данных, обучающих данных , моделей и MLflow."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = RAW_DATA_DIR / "UCI_Credit_Card.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / RAW_DATA_PATH.name

MODELS_DIR = PROJECT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
TRAIN_REF_DATA = MODELS_DIR / "train_reference.csv"
TEST_REF_DATA = MODELS_DIR / "test_reference.csv"
REPORTS_DIR = PROJECT_DIR / "reports"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MLFLOW_DIR = Path(os.environ.get("MLFLOW_DIR", ARTIFACTS_DIR / "mlflow")).resolve()
