"""Training, model selection, and saved artifact regression checks."""

from functools import partial
import importlib
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

from src.data.validation import PROCESSED_SCHEMA
from src.features.build_features import build_features
from src.model_training.models_configuration import MODELS
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
    "log_reg": {"C": [0.1, 1.0]},
    "random_forest": {"n_estimators": [5], "max_depth": [2]},
}


@pytest.fixture
def small_models(monkeypatch):
    """Use small grids through the same configuration as production training."""
    for name, config in MODELS.items():
        monkeypatch.setitem(MODELS, name, {**config, "grid": SMALL_GRIDS[name]})
    return MODELS


def test_search_compares_all_models(credit_data, small_models):
    search = create_pipeline(
        training.NUMERIC_FEATURES,
        training.CATEGORICAL_FEATURES,
        n_jobs=1,
    )
    features = build_features(credit_data).replace([np.inf, -np.inf], np.nan)
    search.fit(features, credit_data["default"])
    assert len(search.cv_results_["params"]) == sum(
        len(ParameterGrid(config["grid"])) for config in small_models.values()
    )
    assert {type(p["classifier"]) for p in search.cv_results_["params"]} == {
        config["class"] for config in small_models.values()
    }
    for config, grid in zip(small_models.values(), search.param_grid, strict=True):
        assert {
            key.removeprefix("classifier__"): value
            for key, value in grid.items()
            if key != "classifier"
        } == config["grid"]
    # The preprocessor excludes the target and record identifier.
    changed = features.copy()
    changed["default"] = 1 - changed["default"]
    changed["id"] += 10000
    np.testing.assert_array_equal(search.predict(features), search.predict(changed))


# Known deprecations inside MLflow; keep other dependency warnings visible.
@pytest.mark.filterwarnings(
    r"ignore:The ``noload`` loader strategy is deprecated:"
    r"sqlalchemy.exc.SADeprecationWarning:mlflow\.store\.tracking\.sqlalchemy_store"
)
@pytest.mark.filterwarnings(
    r"ignore:For backward compatibility, 'str' dtypes are included by select_dtypes:"
    r"pandas.errors.Pandas4Warning:mlflow\.tracking\.client"
)
@pytest.mark.parametrize("name", MODELS.keys())
def test_train_saves_and_registers_reloadable_model(
    name, credit_data, small_models, tmp_path, monkeypatch
):
    for other_name in list(small_models):
        if other_name != name:
            monkeypatch.delitem(small_models, other_name)
    monkeypatch.setattr(training, "create_pipeline", partial(create_pipeline, n_jobs=1))
    # MLflow's default artifact location is relative to the working directory.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "MLFLOW_DIR", tmp_path / "mlflow")
    model_path = tmp_path / "models" / "best_model.joblib"
    monkeypatch.setattr(training, "BEST_MODEL_PATH", model_path)
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(training, "REPORTS_DIR", reports_dir)
    pipeline, metrics = training.train(credit_data)
    assert isinstance(pipeline.named_steps["classifier"], small_models[name]["class"])
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("credit-default")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FINISHED"
    assert runs[0].data.params["candidate_models"] == name
    assert runs[0].data.params["model_type"] == small_models[name]["class"].__name__
    features = (
        build_features(credit_data)
        .replace([np.inf, -np.inf], np.nan)[
            training.NUMERIC_FEATURES + training.CATEGORICAL_FEATURES
        ]
        .astype(float)
    )
    expected = pipeline.predict(features)
    np.testing.assert_array_equal(joblib.load(model_path).predict(features), expected)
    registered = mlflow.sklearn.load_model("models:/CreditDefaultModel/1")
    np.testing.assert_array_equal(registered.predict(features), expected)
    assert all(0 <= value <= 1 for value in metrics.values())
    train_reference = pd.read_csv(model_path.parent / "train_reference.csv")
    test_reference = pd.read_csv(model_path.parent / "test_reference.csv")
    assert len(train_reference) + len(test_reference) == len(credit_data)
    assert set(train_reference.id).isdisjoint(test_reference.id)
    test_features = (
        build_features(test_reference)
        .replace([np.inf, -np.inf], np.nan)[
            training.NUMERIC_FEATURES + training.CATEGORICAL_FEATURES
        ]
        .astype(float)
    )
    y_test = test_reference["default"]
    predictions = pipeline.predict(test_features)
    expected_metrics = {
        "test_auc": roc_auc_score(y_test, pipeline.predict_proba(test_features)[:, 1]),
        "test_precision": precision_score(y_test, predictions, zero_division=0),
        "test_recall": recall_score(y_test, predictions, zero_division=0),
        "test_f1": f1_score(y_test, predictions, zero_division=0),
    }
    assert metrics == pytest.approx(expected_metrics)
    assert json.loads((reports_dir / "test_metrics.json").read_text()) == metrics
    assert "ROC-AUC" in (reports_dir / "test_metrics.md").read_text()
    assert (
        (reports_dir / "figures" / "roc_curve.png")
        .read_bytes()
        .startswith(b"\x89PNG\r\n\x1a\n")
    )
