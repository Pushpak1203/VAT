"""Run all five research configurations."""
from __future__ import annotations

import json
import time
from pathlib import Path

from config import PipelineMode, Settings
from pipeline.orchestrator import VATChainAuditor
from benchmarks.evaluator import evaluate_predictions


MODES = [
    PipelineMode.LLM_ONLY,
    PipelineMode.RULES_ONLY,
    PipelineMode.HYBRID,
    PipelineMode.HYBRID_RAG,
    PipelineMode.DEEPSEEK_RAG,
]


def run_all_benchmarks(settings: Settings, dataset_path: Path | None = None):
    dataset_path = dataset_path or Path("data/sample_transactions.json")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]

    output = {"dataset": str(dataset_path), "modes": {}}
    for mode in MODES:
        local = Settings(**{k: v for k, v in settings.__dict__.items()})
        local.pipeline_mode = mode
        auditor = VATChainAuditor(local)
        y_true, y_pred, latencies, confidences, records = [], [], [], [], []
        for row in payload:
            start = time.perf_counter()
            result = auditor.run_from_transaction(row) if hasattr(auditor, "run_from_transaction") else None
            if result is None:
                # Reuse orchestrator's normal input contract through a temporary JSON file.
                temp = Path("benchmarks/results/_benchmark_input.json")
                temp.parent.mkdir(parents=True, exist_ok=True)
                temp.write_text(json.dumps(row), encoding="utf-8")
                result = auditor.run(temp)
                temp.unlink(missing_ok=True)
            latencies.append((time.perf_counter() - start) * 1000)
            confidences.append(result.calibrated_confidence)
            records.append(result.model_dump(mode="json"))
            expected = row.get("expected_treatment")
            if expected:
                y_true.append(expected)
                y_pred.append(result.treatment)
        output["modes"][mode.value] = {
            **evaluate_predictions(y_true, y_pred, latencies, confidences),
            "records": records,
        }

    out_path = Path("benchmarks/results/benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    return output
