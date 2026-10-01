"""Step 6: primary/benchmark LLM reasoning."""
from __future__ import annotations

from config import PipelineMode
from models.model_loader import load_causal_lm
from utils.schema import LLMReasoningResult


class LLMReasoner:
    def __init__(self, settings):
        self.settings = settings
        self._loaded = {}

    def _model_id(self, mode: PipelineMode) -> str:
        return self.settings.deepseek_model_id if mode == PipelineMode.DEEPSEEK_RAG else self.settings.qwen_model_id

    def _ensure(self, mode):
        key = mode.value
        if key not in self._loaded:
            self._loaded[key] = load_causal_lm(
                self._model_id(mode),
                self.settings.torch_device,
                self.settings.hf_cache_dir,
                quantize=True,
            )
        return self._loaded[key]

    def reason(self, tx, classification, risk, anomaly, rules, legal_evidence, mode):
        context = "\n".join(
            f"- {e.get('title')}: {e.get('citation')} | {e.get('text')}"
            for e in legal_evidence
        )
        prompt = f"""You are a VAT compliance reasoning model. Do not invent legal authorities.
Assess this transaction using the deterministic rule result and retrieved legal evidence.

Transaction:
{tx.model_dump_json()}

Classification:
{classification.model_dump_json()}

Risk:
{risk.model_dump_json()}

Anomaly:
{anomaly.model_dump_json()}

Deterministic VAT decision:
{rules.model_dump_json()}

Retrieved legal evidence:
{context or "No legal evidence retrieved."}

Return a concise audit explanation, rationale, and compliance assessment. Cite only authorities present in the evidence.
"""
        try:
            tokenizer, model = self._ensure(mode)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            import torch
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.settings.max_new_tokens,
                    do_sample=False,
                    temperature=self.settings.temperature,
                )
            raw = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            return LLMReasoningResult(
                explanation=raw,
                rationale=f"LLM assessment grounded in deterministic rules and {len(legal_evidence)} retrieved legal items.",
                model_name=self._model_id(mode),
                raw_output=raw,
            )
        except Exception as exc:
            # Deterministic fallback keeps the CLI operational when model weights are unavailable.
            legal = "; ".join(e.get("citation", "") for e in legal_evidence if e.get("citation"))
            explanation = (
                f"Deterministic VAT treatment: {rules.treatment}. "
                f"Compliance status: {rules.compliance_status}. "
                f"Risk score: {risk.risk_score:.2f}; anomaly={anomaly.is_anomaly}. "
                f"Legal evidence: {legal or 'none retrieved'}. "
                f"LLM inference was unavailable: {type(exc).__name__}."
            )
            return LLMReasoningResult(
                explanation=explanation,
                rationale="Fallback uses deterministic rules and retrieved evidence; no unsupported legal conclusion was generated.",
                model_name=self._model_id(mode),
                raw_output=None,
            )
