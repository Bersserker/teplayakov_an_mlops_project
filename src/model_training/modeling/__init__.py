"""Model factories and their default hyperparameter grids."""

from .catboost import build_model as catboost
from .log_reg import build_model as log_reg
from .random_forest import build_model as random_forest

MODEL_FACTORIES = {
    "log_reg": log_reg,
    "random_forest": random_forest,
    "catboost": catboost,
}

MODEL_NAMES = tuple(MODEL_FACTORIES)
PARAM_GRIDS = {
    "log_reg": {"C": [0.1, 1.0, 10.0]},
    "random_forest": {"n_estimators": [50, 100], "max_depth": [5, None]},
    "catboost": {"iterations": [500, 1000], "depth": [4, 6]},
}


def build_model(name):
    if name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model {name!r}. Choose from: {', '.join(MODEL_NAMES)}"
        )
    return MODEL_FACTORIES[name]({})
