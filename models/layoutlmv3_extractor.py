"""LayoutLMv3 document representation wrapper."""
from __future__ import annotations

from pathlib import Path
from PIL import Image

from models.model_loader import load_encoder


class LayoutLMv3Extractor:
    def __init__(self, model_id: str, device: str, cache_dir: str | None):
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self.processor = None
        self.model = None

    def _ensure_loaded(self):
        from transformers import AutoProcessor, AutoModel
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            self.model = AutoModel.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            self.model.to(self.device)
            self.model.eval()

    def extract_representation(self, image_path: Path, ocr_result: dict | None = None) -> dict:
        self._ensure_loaded()
        image = Image.open(image_path).convert("RGB")
        words = []
        boxes = []
        if ocr_result:
            words = ocr_result.get("words", [])
            boxes = ocr_result.get("boxes", [])
        if not words:
            words = ["invoice"]
            boxes = [[0, 0, 1000, 1000]]
        encoding = self.processor(
            image,
            text=words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items() if hasattr(v, "to")}
        with __import__("torch").no_grad():
            output = self.model(**encoding)
        pooled = output.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().tolist()
        return {"embedding": pooled, "token_count": len(words), "model": self.model_id}
