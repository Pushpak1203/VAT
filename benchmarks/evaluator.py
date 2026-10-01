"""Benchmark metrics."""
from __future__ import annotations

import statistics
from sklearn.metrics import accuracy_score, f1_score


def evaluate_predictions(y_true, y_pred, latencies, confidences):
    result = {
        "n": len(y_true),
        "latency_ms_mean": statistics.mean(latencies) if latencies else 0.0,
        "confidence_mean": statistics.mean(confidences) if confidences else 0.0,
    }
    if y_true:
        result["accuracy"] = accuracy_score(y_true, y_pred)
        result["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    else:
        result["accuracy"] = None
        result["f1_macro"] = None
    return result
