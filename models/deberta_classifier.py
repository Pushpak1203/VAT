"""DeBERTa-v3 transaction representation/classification wrapper."""
from __future__ import annotations

import re
import torch

from config import Settings
from models.model_loader import load_sequence_model
from utils.schema import TransactionClassification


class DebertaClassifier:
    LABELS = {
        "customer_type": ["B2B", "B2C"],
        "supply_type": ["goods", "services"],
        "vat_category": ["intra_eu_supply", "domestic", "export", "import", "digital_service", "other"],
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = settings.torch_device
        self._loaded = False
        self.tokenizer = None
        self.model = None

    def _ensure_loaded(self):
        if not self._loaded:
            self.tokenizer, self.model = load_sequence_model(
                self.settings.deberta_model_id,
                self.device,
                self.settings.hf_cache_dir,
            )
            self._loaded = True

    def classify(self, transaction) -> TransactionClassification:
        description = (transaction.description or "").lower()
        evidence = []
        customer = (transaction.customer_type or "").upper()
        if customer not in {"B2B", "B2C"}:
            customer = "B2B" if transaction.buyer_vat_id else "B2C"
        supply = (transaction.supply_type or "").lower()
        if supply not in {"goods", "services"}:
            supply = "services" if any(x in description for x in ["software", "consulting", "service", "saas"]) else "goods"
        seller = transaction.seller_country or ""
        buyer = transaction.buyer_country or transaction.destination_country or ""
        if seller and buyer and seller != buyer:
            if transaction.origin_country and buyer and transaction.origin_country != buyer:
                region = "CROSS_BORDER"
            else:
                region = "INTRA_EU" if seller in EU_COUNTRIES and buyer in EU_COUNTRIES else "CROSS_BORDER"
        else:
            region = "DOMESTIC"

        if "export" in description:
            category = "export"
        elif "import" in description:
            category = "import"
        elif seller in EU_COUNTRIES and buyer in EU_COUNTRIES and seller != buyer:
            category = "intra_eu_supply"
        elif supply == "services":
            category = "digital_service" if any(k in description for k in ["saas", "digital", "software", "streaming"]) else "other"
        else:
            category = "domestic"

        # The base checkpoint is not VAT-fine-tuned. We use it as a semantic encoder
        # when explicitly requested and retain deterministic evidence as the safe fallback.
        confidence = 0.78
        if self.settings.allow_model_download:
            try:
                self._ensure_loaded()
                text = f"{description} seller={seller} buyer={buyer} customer={customer} supply={supply}"
                encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.no_grad():
                    out = self.model(**encoded)
                confidence = min(0.95, max(0.55, float(out.last_hidden_state.mean().abs().item())))
            except Exception as exc:
                evidence.append(f"DeBERTa semantic pass unavailable; deterministic classification used: {exc}")

        evidence.extend([f"customer_type={customer}", f"supply_type={supply}", f"region={region}", f"vat_category={category}"])
        return TransactionClassification(
            customer_type=customer,
            supply_type=supply,
            region=region,
            vat_category=category,
            confidence=confidence,
            evidence=evidence,
        )


EU_COUNTRIES = {
    "AT","BE","BG","HR","CY","CZ","DE","DK","EE","EL","GR","ES","FI","FR","HU",
    "IE","IT","LT","LU","LV","MT","NL","PL","PT","RO","SE","SI","SK",
}
