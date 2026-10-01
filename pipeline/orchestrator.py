"""Strict seven-step VAT Chain Auditor orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

from config import PipelineMode, Settings
from pipeline.ocr_extraction import OCRExtractor
from pipeline.transaction_classifier import TransactionClassifier
from pipeline.risk_scoring import RiskScorer
from pipeline.anomaly_detection import AnomalyDetector
from pipeline.vat_rules_engine import VATRulesEngine
from pipeline.legal_rag import LegalRAG
from pipeline.llm_reasoning import LLMReasoner
from pipeline.confidence_calibration import ConfidenceCalibrator
from utils.schema import TransactionInput, VATAuditResult


class VATChainAuditor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ocr = OCRExtractor(settings)
        self.classifier = TransactionClassifier(settings)
        self.risk = RiskScorer(settings)
        self.anomaly = AnomalyDetector()
        self.rules = VATRulesEngine()
        self.rag = LegalRAG(settings)
        self.llm = LLMReasoner(settings)
        self.calibrator = ConfidenceCalibrator(settings.calibration_model_path)

    def _load_input(self, path: Path) -> tuple[TransactionInput, dict]:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                payload = payload[0]
            tx = TransactionInput.model_validate(payload)
            return tx, {"source": str(path), "structured": True}
        ocr = self.ocr.extract(path)
        fields = ocr.fields
        text = ocr.text
        vat_ids = fields.get("vat_ids", [])
        amounts = fields.get("amounts", [])
        net = 0.0
        if amounts:
            import re
            cleaned = re.sub(r"[^0-9.,]", "", amounts[0]).replace(",", "")
            try:
                net = float(cleaned)
            except ValueError:
                net = 0.0
        tx = TransactionInput(
            transaction_id=path.stem,
            buyer_vat_id=vat_ids[0] if vat_ids else None,
            description=text,
            net_amount=net,
            gross_amount=net,
        )
        return tx, {"source": str(path), "structured": False, "ocr": ocr.model_dump()}

    def run(self, input_path: Path) -> VATAuditResult:
        tx, extraction_meta = self._load_input(Path(input_path))
        mode = self.settings.pipeline_mode

        # Step 1 is performed by _load_input.
        # Step 2.
        classification = self.classifier.classify(tx)

        # Step 3a/3b.
        anomaly = self.anomaly.detect(tx)
        risk = self.risk.score(tx, classification, anomaly)

        # Step 4.
        rules = self.rules.evaluate(tx, classification)

        # Baseline modes intentionally bypass downstream components.
        legal_evidence = []
        if mode in {PipelineMode.HYBRID_RAG, PipelineMode.DEEPSEEK_RAG}:
            query = f"{tx.description} {classification.vat_category} {tx.seller_country} {tx.buyer_country}"
            legal_evidence = [e.model_dump() for e in self.rag.retrieve(query)]

        if mode == PipelineMode.RULES_ONLY:
            explanation = f"Rules-only decision: {rules.treatment}. {rules.compliance_status}."
            rationale = "Deterministic VAT rules engine only."
            model_name = "RULES_ONLY"
            raw_confidence = 0.85
        else:
            llm_mode = PipelineMode.DEEPSEEK_RAG if mode == PipelineMode.DEEPSEEK_RAG else PipelineMode.HYBRID_RAG
            if mode == PipelineMode.LLM_ONLY:
                legal_evidence = []
                llm_mode = PipelineMode.LLM_ONLY
            reasoning = self.llm.reason(tx, classification, risk, anomaly, rules, legal_evidence, llm_mode)
            explanation = reasoning.explanation
            rationale = reasoning.rationale
            model_name = reasoning.model_name
            raw_confidence = min(
                0.99,
                max(0.05, 0.45 * classification.confidence + 0.35 * (1 - risk.risk_score) + 0.20 * (0.5 if anomaly.is_anomaly else 1.0)),
            )

        calibrated = self.calibrator.calibrate(raw_confidence, 1.0 - risk.risk_score)

        return VATAuditResult(
            transaction_id=tx.transaction_id,
            treatment=rules.treatment,
            compliance_status=rules.compliance_status,
            risk_score=risk.risk_score,
            anomaly_flag=anomaly.is_anomaly,
            explanation=explanation,
            rationale=rationale,
            legal_references=[e.get("citation", "") for e in legal_evidence if e.get("citation")],
            calibrated_confidence=calibrated,
            model_mode=mode.value,
            evidence=legal_evidence,
            intermediate={
                "classification": classification.model_dump(),
                "risk": risk.model_dump(),
                "anomaly": anomaly.model_dump(),
                "rules": rules.model_dump(),
                "extraction": extraction_meta,
                "reasoning_model": model_name,
            },
        )


    def run_from_transaction(self, payload: dict) -> VATAuditResult:
        tx = TransactionInput.model_validate(payload)
        mode = self.settings.pipeline_mode
        classification = self.classifier.classify(tx)
        anomaly = self.anomaly.detect(tx)
        risk = self.risk.score(tx, classification, anomaly)
        rules = self.rules.evaluate(tx, classification)
        legal_evidence = []
        if mode in {PipelineMode.HYBRID_RAG, PipelineMode.DEEPSEEK_RAG}:
            query = f"{tx.description} {classification.vat_category} {tx.seller_country} {tx.buyer_country}"
            legal_evidence = [e.model_dump() for e in self.rag.retrieve(query)]
        if mode == PipelineMode.RULES_ONLY:
            explanation = f"Rules-only decision: {rules.treatment}. {rules.compliance_status}."
            rationale = "Deterministic VAT rules engine only."
            model_name = "RULES_ONLY"
            raw_confidence = 0.85
        else:
            llm_mode = PipelineMode.DEEPSEEK_RAG if mode == PipelineMode.DEEPSEEK_RAG else PipelineMode.HYBRID_RAG
            if mode == PipelineMode.LLM_ONLY:
                legal_evidence = []
                llm_mode = PipelineMode.LLM_ONLY
            reasoning = self.llm.reason(tx, classification, risk, anomaly, rules, legal_evidence, llm_mode)
            explanation, rationale, model_name = reasoning.explanation, reasoning.rationale, reasoning.model_name
            raw_confidence = min(0.99, max(0.05, 0.45 * classification.confidence + 0.35 * (1 - risk.risk_score) + 0.20 * (0.5 if anomaly.is_anomaly else 1.0)))
        calibrated = self.calibrator.calibrate(raw_confidence, 1.0 - risk.risk_score)
        return VATAuditResult(
            transaction_id=tx.transaction_id,
            treatment=rules.treatment,
            compliance_status=rules.compliance_status,
            risk_score=risk.risk_score,
            anomaly_flag=anomaly.is_anomaly,
            explanation=explanation,
            rationale=rationale,
            legal_references=[e.get("citation", "") for e in legal_evidence if e.get("citation")],
            calibrated_confidence=calibrated,
            model_mode=mode.value,
            evidence=legal_evidence,
            intermediate={
                "classification": classification.model_dump(),
                "risk": risk.model_dump(),
                "anomaly": anomaly.model_dump(),
                "rules": rules.model_dump(),
                "reasoning_model": model_name,
            },
        )
