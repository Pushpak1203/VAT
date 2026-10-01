"""BGE reranker v2 m3."""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEReranker:
    def __init__(self, settings):
        self.settings = settings
        self.tokenizer = None
        self.model = None

    def _ensure(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.reranker_model_id,
                cache_dir=self.settings.hf_cache_dir,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.settings.reranker_model_id,
                cache_dir=self.settings.hf_cache_dir,
            )
            self.model.to(self.settings.torch_device)
            self.model.eval()

    def rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        if not documents:
            return []
        try:
            self._ensure()
            pairs = [[query, d["text"]] for d in documents]
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            inputs = {k: v.to(self.settings.torch_device) for k, v in inputs.items()}
            with torch.no_grad():
                scores = self.model(**inputs).logits.view(-1).float().cpu().tolist()
            ranked = []
            for doc, score in zip(documents, scores):
                item = dict(doc)
                item["rerank_score"] = float(score)
                ranked.append(item)
            return sorted(ranked, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        except Exception as exc:
            # Deterministic lexical fallback preserves retrieval semantics without introducing another model.
            q = set(query.lower().split())
            ranked = []
            for d in documents:
                overlap = len(q & set(d["text"].lower().split()))
                item = dict(d)
                item["rerank_score"] = float(overlap)
                item["rerank_error"] = str(exc)
                ranked.append(item)
            return sorted(ranked, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
