"""Prepare data, search the model grids, and log the best model with MLflow."""

from pathlib import Path

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd

from src.config import BEST_MODEL_PATH, MLFLOW_DIR, PROCESSED_DATA_PATH, REPORTS_DIR
from src.model_training.evaluation import save_test_report
from src.model_training.models_configuration import MODELS
from src.model_training.pipeline import create_pipeline
from src.model_training.training_data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_training_data,
    save_training_outputs,
)


def train(df):
    search = create_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    # print(search)
    reference, (X_train, X_test, y_train, y_test) = prepare_training_data(df)
    artifact_dir = MLFLOW_DIR / "mlartifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
    mlflow.set_experiment("credit-default")
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name="Model grid search"):
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        # print(pipeline.named_steps["classifier"])
        classifier = pipeline.named_steps["classifier"]
        print(classifier)
        mlflow.log_params(
            {
                "candidate_models": ",".join(list(MODELS.keys())),
                "model_type": type(classifier).__name__,
                "cv_folds": search.cv.get_n_splits(),
                "scoring": search.scoring,
                **{
                    key: value
                    for key, value in search.best_params_.items()
                    if key != "classifier"
                },
            }
        )
        mlflow.log_metric("best_cv_accuracy", float(search.best_score_))
        mlflow.log_table(
            pd.DataFrame(
                {
                    "params": [str(params) for params in search.cv_results_["params"]],
                    "mean_test_score": search.cv_results_["mean_test_score"],
                    "std_test_score": search.cv_results_["std_test_score"],
                    "rank_test_score": search.cv_results_["rank_test_score"],
                }
            ),
            artifact_file="cv_results.json",
        )
        metrics = save_test_report(pipeline, X_test, y_test, REPORTS_DIR)
        mlflow.log_metrics(metrics)
        for filename in (
            "test_metrics.json",
            "test_metrics.md",
            "figures/roc_curve.png",
        ):
            path = REPORTS_DIR / filename
            mlflow.log_artifact(
                str(path), artifact_path=str(Path("reports") / Path(filename).parent)
            )
        example = X_train.head(5)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name="CreditDefaultModel",
            input_example=example,
            signature=infer_signature(example, pipeline.predict(example)),
            # CatBoost's native extension cannot be saved with skops.
            serialization_format=(
                "cloudpickle"
                if type(classifier).__name__ == "CatBoostClassifier"
                else "skops"
            ),
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
                "sklearn.compose._column_transformer._RemainderColsList",
            ],
        )

    save_training_outputs(pipeline, BEST_MODEL_PATH, reference, X_train, X_test)
    print(
        f"Best model: {type(classifier).__name__}; CV accuracy: {search.best_score_:.4f}"
    )
    print(metrics)
    return pipeline, metrics


if __name__ == "__main__":
    train(pd.read_csv(PROCESSED_DATA_PATH))
