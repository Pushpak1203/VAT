"""
VAT Chain Auditor - CLI entry point.

Python: 3.10.10
Run:
    python main.py --input data/sample_transactions.json
    python main.py --input data/sample_invoices/sample_invoice.png --mode HYBRID_RAG
    python main.py --benchmark
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import PipelineMode, Settings, get_settings
from pipeline.orchestrator import VATChainAuditor
from utils.logger import configure_logging


def _parse_mode(value: str) -> PipelineMode:
    try:
        return PipelineMode(value.upper())
    except ValueError as exc:
        valid = ", ".join(m.value for m in PipelineMode)
        raise argparse.ArgumentTypeError(f"Invalid mode. Choose one of: {valid}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="VAT Chain Auditor research/CLI runner")
    parser.add_argument("--input", type=Path, help="Transaction JSON, invoice image, or invoice PDF")
    parser.add_argument("--mode", type=_parse_mode, default=None, help="Pipeline mode")
    parser.add_argument("--benchmark", action="store_true", help="Run all five benchmark configurations")
    parser.add_argument("--no-model-download", action="store_true", help="Disable HuggingFace model downloads")
    args = parser.parse_args()

    settings = get_settings()
    if args.mode:
        settings.pipeline_mode = args.mode
    if args.no_model_download:
        settings.allow_model_download = False

    configure_logging(settings.log_level)

    if args.benchmark:
        from benchmarks.run_benchmarks import run_all_benchmarks
        result = run_all_benchmarks(settings)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if not args.input:
        parser.error("--input is required unless --benchmark is used")

    auditor = VATChainAuditor(settings)
    result = auditor.run(args.input)
    print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
