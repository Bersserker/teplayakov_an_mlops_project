from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


MODELS = {
    "log_reg": {
        "class": LogisticRegression,
        "params": {"max_iter": 2000, "random_state": 42},
        "grid": {"C": [0.1, 1.0, 10.0]},
    },
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {"random_state": 42, "n_jobs": 1},
        "grid": {"n_estimators": [50, 100], "max_depth": [5, None]},
    },
    "catboost": {
        "class": CatBoostClassifier,
        "params": {
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": 1,
        },
        "grid": {"iterations": [500, 1000], "depth": [4, 6]},
    },
}

