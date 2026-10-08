"""Calculate scalar training and held-out metrics for credit models."""

import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def test_metrics(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = pipeline.predict(X)
    return {
        "test_auc": float(roc_auc_score(y, probabilities)),
        "test_precision": float(precision_score(y, predictions, zero_division=0)),
        "test_recall": float(recall_score(y, predictions, zero_division=0)),
        "test_f1": float(f1_score(y, predictions, zero_division=0)),
    }


def save_test_report(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, reports_dir: Path
) -> dict[str, float]:
    """Save binary metrics and the ROC curve for the held-out test split."""
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = test_metrics(pipeline, X, y)
    (reports_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    labels = {
        "test_auc": "ROC-AUC",
        "test_precision": "Precision",
        "test_recall": "Recall",
        "test_f1": "F1-Score",
    }
    rows = "\n".join(
        f"| {label} | {metrics[key]:.6f} |" for key, label in labels.items()
    )
    (reports_dir / "test_metrics.md").write_text(
        "# Качество бинарной классификации\n\n"
        f"Отложенная тестовая выборка: {len(y)} записей. "
        "Положительный класс: default = 1 (кредитный дефолт).\n\n"
        "ROC-AUC рассчитан по вероятности положительного класса. "
        "Precision, Recall и F1-Score рассчитаны по предсказаниям модели "
        "(`predict`) для положительного класса. "
        "При отсутствии предсказаний положительного класса Precision равен 0.\n\n"
        "| Метрика | Значение |\n| --- | ---: |\n"
        f"{rows}\n\n![ROC-кривая](figures/roc_curve.png)\n",
        encoding="utf-8",
    )
    fpr, tpr, _ = roc_curve(y, pipeline.predict_proba(X)[:, 1], pos_label=1)
    figure = Figure(figsize=(7, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(fpr, tpr, label=f"Model (ROC-AUC = {metrics['test_auc']:.4f})")
    axes.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
    axes.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC curve — held-out test set",
    )
    axes.grid(alpha=0.3)
    axes.legend(loc="lower right")
    figure.savefig(figures_dir / "roc_curve.png", dpi=180)
    return metrics
