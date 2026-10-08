"""Model factories and default hyperparameter grids."""

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_NAMES = ("catboost", "logistic_regression", "random_forest")

PARAM_GRIDS = {
    "catboost": {"iterations": [200], "depth": [4, 6], "learning_rate": [0.1]},
    "logistic_regression": {"C": [0.1, 1.0, 10.0]},
    "random_forest": {"n_estimators": [200], "max_depth": [10, None]},
}


def build_model(name: str, params: dict | None = None):
    """Build a fresh estimator by name, overriding its defaults with params."""
    params = {} if params is None else params
    if name == "catboost":
        return CatBoostClassifier(
            **{
                "random_seed": 42,
                "verbose": False,
                "allow_writing_files": False,
                "thread_count": 1,
                **params,
            }
        )
    if name == "logistic_regression":
        return LogisticRegression(**{"max_iter": 2000, "random_state": 42, **params})
    if name == "random_forest":
        return RandomForestClassifier(**{"random_state": 42, "n_jobs": 1, **params})
    raise ValueError(f"Unknown model {name!r}. Choose from {MODEL_NAMES}.")
