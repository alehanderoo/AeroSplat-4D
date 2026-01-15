"""
Classification metrics.
"""

from typing import Dict, List, Optional
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        precision_recall_curve,
        average_precision_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ClassificationMetrics:
    """Compute classification metrics."""

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Classification threshold for probabilities
        """
        self.threshold = threshold

    def compute(
        self,
        predictions: List[int],
        labels: List[int],
        probabilities: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            predictions: List of predicted labels (0/1)
            labels: List of ground truth labels (0/1)
            probabilities: Optional list of predicted probabilities

        Returns:
            Dictionary with accuracy, precision, recall, f1, and optionally AUC
        """
        preds = np.array(predictions).flatten()
        labels_arr = np.array(labels).flatten()

        # Handle edge cases
        if len(preds) == 0 or len(labels_arr) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
            }

        if SKLEARN_AVAILABLE:
            metrics = {
                'accuracy': accuracy_score(labels_arr, preds),
                'precision': precision_score(labels_arr, preds, zero_division=0),
                'recall': recall_score(labels_arr, preds, zero_division=0),
                'f1': f1_score(labels_arr, preds, zero_division=0),
            }

            if probabilities is not None:
                probs = np.array(probabilities).flatten()
                try:
                    metrics['auc'] = roc_auc_score(labels_arr, probs)
                    metrics['ap'] = average_precision_score(labels_arr, probs)
                except ValueError:
                    # Only one class present
                    metrics['auc'] = 0.0
                    metrics['ap'] = 0.0
        else:
            # Simple implementation without sklearn
            metrics = self._compute_basic(preds, labels_arr)

        return metrics

    def _compute_basic(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Basic metrics computation without sklearn."""
        # True/False positives/negatives
        tp = ((predictions == 1) & (labels == 1)).sum()
        tn = ((predictions == 0) & (labels == 0)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

    def compute_confusion_matrix(
        self,
        predictions: List[int],
        labels: List[int]
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Returns:
            2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        preds = np.array(predictions).flatten()
        labels_arr = np.array(labels).flatten()

        if SKLEARN_AVAILABLE:
            return confusion_matrix(labels_arr, preds)
        else:
            tp = ((preds == 1) & (labels_arr == 1)).sum()
            tn = ((preds == 0) & (labels_arr == 0)).sum()
            fp = ((preds == 1) & (labels_arr == 0)).sum()
            fn = ((preds == 0) & (labels_arr == 1)).sum()
            return np.array([[tn, fp], [fn, tp]])

    def compute_threshold_metrics(
        self,
        probabilities: List[float],
        labels: List[int],
        thresholds: List[float] = None
    ) -> Dict[str, List[float]]:
        """
        Compute metrics at various thresholds.

        Args:
            probabilities: Predicted probabilities
            labels: Ground truth labels
            thresholds: Thresholds to evaluate (default: 0.1 to 0.9)

        Returns:
            Dictionary with lists of precision, recall, f1 at each threshold
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        probs = np.array(probabilities).flatten()
        labels_arr = np.array(labels).flatten()

        results = {
            'thresholds': thresholds,
            'precision': [],
            'recall': [],
            'f1': [],
        }

        for t in thresholds:
            preds = (probs >= t).astype(int)
            metrics = self.compute(preds.tolist(), labels_arr.tolist())
            results['precision'].append(metrics['precision'])
            results['recall'].append(metrics['recall'])
            results['f1'].append(metrics['f1'])

        return results

    def find_optimal_threshold(
        self,
        probabilities: List[float],
        labels: List[int],
        metric: str = 'f1'
    ) -> tuple:
        """
        Find optimal classification threshold.

        Args:
            probabilities: Predicted probabilities
            labels: Ground truth labels
            metric: Metric to optimize ('f1', 'precision', 'recall')

        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        thresholds = np.arange(0.01, 1.0, 0.01)
        probs = np.array(probabilities).flatten()
        labels_arr = np.array(labels).flatten()

        best_threshold = 0.5
        best_value = 0.0

        for t in thresholds:
            preds = (probs >= t).astype(int)
            metrics = self.compute(preds.tolist(), labels_arr.tolist())
            value = metrics.get(metric, 0.0)

            if value > best_value:
                best_value = value
                best_threshold = t

        return float(best_threshold), float(best_value)


class MetricTracker:
    """Track metrics over training."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked values."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
        self.losses = []

    def update(
        self,
        predictions: List[int],
        labels: List[int],
        probabilities: List[float] = None,
        loss: float = None
    ):
        """Add batch results."""
        self.predictions.extend(predictions)
        self.labels.extend(labels)
        if probabilities is not None:
            self.probabilities.extend(probabilities)
        if loss is not None:
            self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """Compute metrics for all tracked data."""
        metrics_calc = ClassificationMetrics()

        probs = self.probabilities if self.probabilities else None
        metrics = metrics_calc.compute(self.predictions, self.labels, probs)

        if self.losses:
            metrics['loss'] = np.mean(self.losses)

        return metrics
