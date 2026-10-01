import json
from pathlib import Path

from config import Settings, PipelineMode
from pipeline.orchestrator import VATChainAuditor


def test_rules_only_pipeline(tmp_path):
    payload = {
        "transaction_id": "TEST-001",
        "seller_country": "DE",
        "buyer_country": "FR",
        "buyer_vat_id": "FR12345678901",
        "customer_type": "B2B",
        "supply_type": "goods",
        "net_amount": 1000,
        "vat_amount": 0,
        "description": "Intra-EU supply of goods",
    }
    path = tmp_path / "tx.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    settings = Settings(pipeline_mode=PipelineMode.RULES_ONLY, allow_model_download=False)
    result = VATChainAuditor(settings).run(path)
    assert result.transaction_id == "TEST-001"
    assert result.treatment.startswith("intra-EU B2B")
    assert 0 <= result.calibrated_confidence <= 1
