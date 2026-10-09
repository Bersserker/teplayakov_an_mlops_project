# pipeline.py
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.model_training.models_configuration import MODELS


def create_pipeline(numeric_features, categorical_features, n_jobs=-1):
    """Создадим пайплайн обучения"""
    models = list(MODELS.keys())
    grids = []
    for name in models:
        config = MODELS[name]
        estimator = config["class"](**config["params"])
        grid = config["grid"]
        grids.append(
            {
                "classifier": [estimator],
                **{f"classifier__{key}": values for key, values in grid.items()},
            }
        )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", grids[0]["classifier"][0]),
        ]
    )
    return GridSearchCV(
        pipeline,
        grids,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=n_jobs,
        error_score="raise",
    )
