from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    average: str = 'macro'
) -> Dict[str, float]:
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    if average == 'macro':
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
        except (ValueError, IndexError):
            pass

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None
) -> str:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )
    return report


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def print_metrics_summary(
    metrics: Dict[str, float],
    title: str = "Classification Metrics"
) -> None:
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")

    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name.replace('_', ' ').title():<30}: {metric_value:.4f}")
        else:
            print(f"{metric_name.replace('_', ' ').title():<30}: {metric_value}")

    print(f"{'='*50}\n")


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, float]:
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    if verbose:
        print_metrics_summary(metrics)

        print("Detailed Classification Report:")
        print("-" * 50)
        report = print_classification_report(y_true, y_pred, target_names=target_names)
        print(report)

        print("\nConfusion Matrix:")
        print("-" * 50)
        cm = compute_confusion_matrix(y_true, y_pred)
        print(cm)
        print()

    return metrics


def calculate_confidence_intervals(
    scores: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    from scipy import stats

    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    std = np.std(scores_array)
    n = len(scores_array)

    margin = std * stats.t.ppf((1 + confidence) / 2, n - 1) / np.sqrt(n)

    return mean, mean - margin, mean + margin

