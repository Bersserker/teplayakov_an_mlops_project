"""Конфигурационный файл для моделей"""

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODELS = {
    "log_reg": {
        "class": LogisticRegression,
        "params": {
            "max_iter": 2000,
            "random_state": 42,
            "solver": "lbfgs",  # стабильный и быстрый для большинства задач
            "l1_ratio": 0,
            "class_weight": "balanced",  # полезно при дисбалансе классов
        },
        "grid": {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
    },
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {
            "random_state": 42,
            "n_jobs": 1,  # можно поставить -1 для всех ядер, если нет ограничений
            "oob_score": False,  # out-of-bag оценка без отдельной валидации
            "bootstrap": True,  # стандартная выборка с возвращением
            "class_weight": "balanced",  # борьба с дисбалансом
            "criterion": "gini",  # или "entropy", можно добавить в grid
        },
        "grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    },
    "xgboost": {
        "class": XGBClassifier,
        "params": {
            "random_state": 42,
            "n_jobs": 1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
        },
        "grid": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.1, 0.3],
        },
    },
    "catboost": {
        "class": CatBoostClassifier,
        "params": {
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": 1,
            "loss_function": "Logloss",  # явное указание целевой функции
            "eval_metric": "AUC",  # метрика для отслеживания в процессе обучения
            "early_stopping_rounds": 50,  # защита от переобучения
            "cat_features": None,  # заполняется позже, если есть категориальные признаки
        },
        "grid": {
            "iterations": [300, 500, 1000],
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.1, 0.3],
            "l2_leaf_reg": [1, 3, 10],  # регуляризация для CatBoost
        },
    },
}
