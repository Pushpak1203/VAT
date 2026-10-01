"""LightGBM VAT risk model with artifact loading and train/save helpers."""
from __future__ import annotations

from pathlib import Path
import numpy as np


class LightGBMRiskModel:
    FEATURES = [
        "net_amount", "vat_amount", "vat_rate_observed", "cross_border",
        "b2b", "missing_vat_id", "amount_roundness", "description_length",
    ]

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        self._load()

    def _load(self):
        if self.model_path.exists():
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(self.model_path))

    @classmethod
    def _features(cls, tx, classification, anomaly=None) -> np.ndarray:
        vat_rate = (tx.vat_amount / tx.net_amount) if tx.net_amount else 0.0
        amount = abs(tx.net_amount)
        roundness = 1.0 if amount and amount % 1000 == 0 else 0.0
        return np.array([[
            min(amount / 100000, 10.0),
            min(abs(tx.vat_amount) / 100000, 10.0),
            min(vat_rate, 1.0),
            float(tx.seller_country != (tx.buyer_country or tx.destination_country)),
            float(classification.customer_type == "B2B"),
            float(not bool(tx.buyer_vat_id)),
            roundness,
            min(len(tx.description or "") / 500, 1.0),
        ]], dtype=float)

    def predict(self, tx, classification, anomaly=None) -> tuple[float, str]:
        x = self._features(tx, classification, anomaly)
        if self.model is not None:
            score = float(self.model.predict(x)[0])
            return float(np.clip(score, 0, 1)), "LightGBM-trained-artifact"
        # Conservative deterministic bootstrap score until a labelled model artifact exists.
        score = 0.08
        if classification.region == "CROSS_BORDER":
            score += 0.12
        if not tx.buyer_vat_id and classification.customer_type == "B2B":
            score += 0.25
        if tx.net_amount > 50000:
            score += 0.12
        if tx.vat_amount == 0 and classification.vat_category == "domestic":
            score += 0.22
        if anomaly is not None and anomaly.is_anomaly:
            score += 0.25
        return float(np.clip(score, 0, 1)), "LightGBM-deterministic-bootstrap"

    @classmethod
    def train(cls, X, y, output_path: Path, num_boost_round: int = 100):
        import lightgbm as lgb
        dataset = lgb.Dataset(X, label=y, feature_name=cls.FEATURES)
        booster = lgb.train(
            {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 42},
            dataset,
            num_boost_round=num_boost_round,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(output_path))
        return booster
