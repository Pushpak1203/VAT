"""Step 7: isotonic confidence calibration."""
from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
from sklearn.isotonic import IsotonicRegression


class ConfidenceCalibrator:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.model = None
        if self.model_path.exists():
            with self.model_path.open("rb") as fh:
                self.model = pickle.load(fh)

    def calibrate(self, raw_confidence: float, correctness_proxy: float) -> float:
        x = float(np.clip(raw_confidence, 0, 1))
        if self.model is not None:
            return float(np.clip(self.model.predict([x])[0], 0, 1))
        # No calibration artifact means this is a bounded proxy, not a claimed calibrated probability.
        proxy = 0.6 * x + 0.4 * float(np.clip(correctness_proxy, 0, 1))
        return float(np.clip(proxy, 0, 1))

    @staticmethod
    def fit(raw_confidences, y_true, output_path: Path):
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(raw_confidences, y_true)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as fh:
            pickle.dump(model, fh)
        return model
