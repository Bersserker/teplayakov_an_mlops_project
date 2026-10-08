"""Training, model selection, and saved artifact regression checks."""

import importlib

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest

from src.data.validation import PROCESSED_SCHEMA
from src.model_training.models_for_training import MODEL_NAMES
from src.model_training.pipeline import create_pipeline

training = importlib.import_module("src.model_training.train")


@pytest.fixture
def credit_data():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {name: rng.integers(0, 1000, 80) for name in PROCESSED_SCHEMA.columns}
    )
    data["id"] = np.arange(len(data))
    data["sex"] = rng.integers(1, 3, len(data))
    data["education"] = rng.integers(0, 7, len(data))
    data["marriage"] = rng.integers(0, 4, len(data))
    data["age"] = rng.integers(20, 70, len(data))
    for name in training.CATEGORICAL_FEATURES[3:]:
        data[name] = rng.integers(-2, 4, len(data))
    data["default"] = np.tile([0, 1], len(data) // 2)
    return data


SMALL_GRIDS = {
    "catboost": {"iterations": [5], "depth": [2]},
    "logistic_regression": {"C": [1.0]},
    "random_forest": {"n_estimators": [5], "max_depth": [2]},
}


def test_search_compares_all_models(credit_data):
    search = create_pipeline(
        training.NUMERIC_FEATURES,
        training.CATEGORICAL_FEATURES,
        SMALL_GRIDS,
        n_jobs=1,
    )
    search.fit(credit_data, credit_data["default"])
    assert len(search.cv_results_["params"]) == len(MODEL_NAMES)
    assert len({type(p["classifier"]) for p in search.cv_results_["params"]}) == 3
    # The preprocessor excludes the target and record identifier.
    changed = credit_data.copy()
    changed["default"] = 1 - changed["default"]
    changed["id"] += 10000
    np.testing.assert_array_equal(search.predict(credit_data), search.predict(changed))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_train_saves_and_registers_reloadable_model(
    name, credit_data, tmp_path, monkeypatch
):
    monkeypatch.setattr(training, "MLFLOW_DIR", tmp_path / "mlflow")
    model_path = tmp_path / "models" / "best_model.joblib"
    monkeypatch.setattr(training, "BEST_MODEL_PATH", model_path)
    pipeline, metrics = training.train(
        credit_data, models=[name], param_grid={name: SMALL_GRIDS[name]}, n_jobs=1
    )
    features = credit_data[
        training.NUMERIC_FEATURES + training.CATEGORICAL_FEATURES
    ].astype(float)
    expected = pipeline.predict(features)
    np.testing.assert_array_equal(joblib.load(model_path).predict(features), expected)
    registered = mlflow.sklearn.load_model("models:/CreditDefaultModel/1")
    np.testing.assert_array_equal(registered.predict(features), expected)
    assert all(0 <= value <= 1 for value in metrics.values())
    train_reference = pd.read_csv(model_path.parent / "train_reference.csv")
    test_reference = pd.read_csv(model_path.parent / "test_reference.csv")
    assert len(train_reference) + len(test_reference) == len(credit_data)
    assert set(train_reference.id).isdisjoint(test_reference.id)


@pytest.mark.parametrize("models", [[], ["catboost", "catboost"], ["unknown"]])
def test_invalid_model_selection(models):
    with pytest.raises(ValueError):
        create_pipeline(["age"], ["sex"], models=models)
