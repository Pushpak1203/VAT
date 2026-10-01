"""Step 1: PaddleOCR + LayoutLMv3 invoice extraction."""
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

from utils.schema import OCRResult


class OCRExtractor:
    def __init__(self, settings):
        self.settings = settings
        self.ocr = None
        self.layout = None

    def _ensure_ocr(self):
        if self.ocr is None:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

    @staticmethod
    def _parse_fields(text: str) -> dict:
        vat_ids = re.findall(r"\b[A-Z]{2}[A-Z0-9]{8,14}\b", text.upper())
        dates = re.findall(r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b", text)
        amounts = re.findall(r"(?:EUR|€|\$|USD|GBP|£)\s?[\d,]+(?:\.\d{1,2})?", text, flags=re.I)
        return {
            "vat_ids": list(dict.fromkeys(vat_ids)),
            "dates": list(dict.fromkeys(dates)),
            "amounts": list(dict.fromkeys(amounts)),
        }

    def extract(self, path: Path) -> OCRResult:
        path = Path(path)
        if path.suffix.lower() in {".json"}:
            return OCRResult(text=path.read_text(encoding="utf-8"), confidence=1.0, source_path=str(path))

        self._ensure_ocr()
        result = self.ocr.predict(str(path))
        texts = []
        words = []
        boxes = []
        confidence_values = []
        for page in result:
            data = getattr(page, "json", None)
            if callable(data):
                data = data()
            if isinstance(data, str):
                import json
                data = json.loads(data)
            if not isinstance(data, dict):
                continue
            res = data.get("res", data)
            rec_texts = res.get("rec_texts", []) or []
            rec_scores = res.get("rec_scores", []) or []
            rec_boxes = res.get("rec_boxes", []) or []
            texts.extend(str(x) for x in rec_texts)
            words.extend(str(x) for x in rec_texts)
            boxes.extend([list(map(int, b)) for b in rec_boxes])
            confidence_values.extend(float(x) for x in rec_scores)
        text = "\n".join(texts)
        fields = self._parse_fields(text)
        fields["words"] = words
        fields["boxes"] = boxes

        # LayoutLMv3 is used as a document representation layer after OCR.
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            try:
                from models.layoutlmv3_extractor import LayoutLMv3Extractor
                self.layout = self.layout or LayoutLMv3Extractor(
                    self.settings.layoutlm_model_id,
                    self.settings.torch_device,
                    self.settings.hf_cache_dir,
                )
                fields["layoutlmv3"] = self.layout.extract_representation(path, fields)
            except Exception as exc:
                fields["layoutlmv3_error"] = str(exc)

        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        return OCRResult(
            text=text,
            fields=fields,
            confidence=confidence,
            source_path=str(path),
        )
