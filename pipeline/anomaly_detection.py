"""Step 3b: Isolation Forest anomaly detection."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from utils.schema import AnomalyResult


class AnomalyDetector:
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42,
        )
        self.fitted = False

    @staticmethod
    def _vector(tx) -> np.ndarray:
        vat_rate = tx.vat_amount / tx.net_amount if tx.net_amount else 0
        return np.array([[
            np.log1p(abs(tx.net_amount)),
            np.log1p(abs(tx.vat_amount)),
            vat_rate,
            len(tx.description or ""),
            int(bool(tx.buyer_vat_id)),
            int(tx.seller_country != (tx.buyer_country or tx.destination_country)),
        ]], dtype=float)

    def fit(self, transactions):
        X = np.vstack([self._vector(tx) for tx in transactions])
        self.model.fit(X)
        self.fitted = True
        return self

    def detect(self, tx) -> AnomalyResult:
        if not self.fitted:
            # A single observation cannot train Isolation Forest meaningfully.
            # Use a conservative deterministic bootstrap until a transaction batch is fitted.
            score = 0.0
            if tx.net_amount > 100000:
                score = 0.35
            if tx.vat_amount == 0 and tx.net_amount > 25000 and not tx.buyer_vat_id:
                score = max(score, 0.75)
            return AnomalyResult(
                anomaly_score=score,
                is_anomaly=score >= 0.65,
                model_version="IsolationForest-bootstrap",
            )
        raw = float(self.model.decision_function(self._vector(tx))[0])
        anomaly_score = float(np.clip(0.5 - raw, 0, 1))
        return AnomalyResult(
            anomaly_score=anomaly_score,
            is_anomaly=raw < 0,
            model_version="IsolationForest",
        )
