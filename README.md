# Teplayakov_AN_MLOps_project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Классификация кредитного дефолта

Пути к исходным и обработанным данным, обученной модели, отчётам и артефактам
MLflow задаются в [src/config.py](src/config.py). Пути вычисляются относительно
корня проекта; расположение MLflow также можно задать переменной `MLFLOW_DIR`.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         teplayakov_an_mlops_project and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── teplayakov_an_mlops_project   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes teplayakov_an_mlops_project a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

### Метрики качества модели

Обучение запускается командой `make train` без аргументов командной строки.
Модели и сетки параметров заданы в `src/model_training/modeling/__init__.py`.

При запуске `make train` лучшая модель оценивается на отложенной тестовой
выборке (20% данных, стратифицированное разбиение с `random_state=42`).
ROC-AUC рассчитывается по вероятностям дефолта, а Precision, Recall и F1-Score —
по предсказаниям положительного класса `default = 1`.

Результаты сохраняются в [reports/test_metrics.md](reports/test_metrics.md),
числовые значения — в [reports/test_metrics.json](reports/test_metrics.json),
ROC-кривая — в [reports/figures/roc_curve.png](reports/figures/roc_curve.png).
Эти файлы также сохраняются как артефакты запуска MLflow.
