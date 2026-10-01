"""Step 3a: LightGBM risk scoring."""
from models.lightgbm_model import LightGBMRiskModel
from utils.schema import RiskResult


class RiskScorer:
    def __init__(self, settings):
        self.model = LightGBMRiskModel(settings.lightgbm_model_path)

    def score(self, transaction, classification, anomaly=None) -> RiskResult:
        score, version = self.model.predict(transaction, classification, anomaly)
        return RiskResult(
            risk_score=score,
            high_risk=score >= 0.65,
            model_version=version,
        )
