"""BGE-M3 embedding generation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from models.model_loader import load_encoder


class BGEEmbedder:
    def __init__(self, settings):
        self.settings = settings
        self.tokenizer = None
        self.model = None

    def _ensure(self):
        if self.model is None:
            self.tokenizer, self.model = load_encoder(
                self.settings.embedding_model_id,
                self.settings.torch_device,
                self.settings.hf_cache_dir,
            )

    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure()
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        encoded = {k: v.to(self.settings.torch_device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()
