"""
Unit Tests for Centralized Evaluation Metrics
Verifies F1, AUC, ADL, FPR@A6, and confusion matrix calculation logic.
"""

import pytest
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_detection_metrics, confusion_matrix_dict


class TestConfusionMatrixDict:
    def test_basic_matrix(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        cm = confusion_matrix_dict(y_true, y_pred)
        assert cm["true_negatives"] == 2
        assert cm["false_positives"] == 1
        assert cm["false_negatives"] == 1
        assert cm["true_positives"] == 2
        # Ensure all types are python native integers
        assert all(isinstance(v, int) for v in cm.values())


class TestComputeDetectionMetrics:
    def test_basic_metrics(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        metrics = compute_detection_metrics(
            y_true, y_pred, method="test_method", split_name="val"
        )
        assert metrics["method"] == "test_method"
        assert metrics["split"] == "val"
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert metrics["n_predicted"] == 3
        assert metrics["n_true"] == 3

    def test_zero_division(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        metrics = compute_detection_metrics(y_true, y_pred)
        assert metrics["f1"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_with_threshold_and_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_scores = np.array([0.1, 0.6, 0.8, 0.3])
        metrics = compute_detection_metrics(
            y_true, y_pred, y_scores=y_scores, threshold=0.5
        )
        assert metrics["threshold"] == 0.5
        assert metrics["roc_auc"] == pytest.approx(0.75)

    def test_auc_single_class_nan(self):
        """AUC-ROC should return NaN or be omitted if y_true contains only one class."""
        y_true = np.array([0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3])
        metrics = compute_detection_metrics(y_true, y_true, y_scores=y_scores)
        assert "roc_auc" not in metrics

    def test_with_dataframe_group_metrics(self):
        n = 10
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 0])
        df = pd.DataFrame({
            "anomaly_type": ["normal"] * 4 + ["low_irradiance"] * 3 + ["sudden_drop"] * 3,
            "is_anomaly": [False] * 7 + [True] * 3,
        })
        metrics = compute_detection_metrics(y_true, y_pred, df=df)
        assert "fpr_a6" in metrics
        assert "fpr_global" in metrics
        # low_irradiance is index 4,5,6 (is_anomaly=False).
        # y_pred[4,5,6] = [0, 1, 1] -> 2/3 flagged.
        assert metrics["fpr_a6"] == pytest.approx(2/3)
        # normal indices are 0,1,2,3,4,5,6 (is_anomaly=False).
        # y_pred[0..6] = [0, 0, 1, 0, 0, 1, 1] -> 3/7 flagged.
        assert metrics["fpr_global"] == pytest.approx(3/7)
        # Check F1 for sudden_drop (index 7,8,9)
        # y_true_slice = [1, 1, 1]
        # y_pred_slice = [1, 0, 0] -> F1 = 2 * (1 * 1/3) / (1 + 1/3) = 0.5
        assert metrics["f1_sudden_drop"] == pytest.approx(0.5)
