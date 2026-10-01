"""Central Hugging Face model loader with optional 4-bit quantization."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from config import Settings
from utils.device_utils import ensure_vram_budget

logger = logging.getLogger(__name__)


def _quantization_config():
    from transformers import BitsAndBytesConfig
    import torch
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def _model_kwargs(settings: Settings, quantize: bool = True) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if settings.hf_cache_dir:
        kwargs["cache_dir"] = settings.hf_cache_dir
    if quantize and settings.torch_device == "cuda":
        try:
            import bitsandbytes  # noqa: F401
            ensure_vram_budget(1.0)
            kwargs["quantization_config"] = _quantization_config()
            kwargs["device_map"] = "auto"
        except ImportError:
            logger.warning("bitsandbytes is unavailable; loading without 4-bit quantization.")
    return kwargs


@lru_cache(maxsize=8)
def load_causal_lm(model_id: str, device: str, cache_dir: str | None, quantize: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    settings = Settings(device=device, hf_cache_dir=cache_dir)
    if not settings.allow_model_download:
        raise RuntimeError("Model download disabled by configuration.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, **_model_kwargs(settings, quantize=quantize)
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=8)
def load_encoder(model_id: str, device: str, cache_dir: str | None):
    from transformers import AutoModel, AutoTokenizer
    if not Settings(device=device, hf_cache_dir=cache_dir).allow_model_download:
        raise RuntimeError("Model download disabled by configuration.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=8)
def load_sequence_model(model_id: str, device: str, cache_dir: str | None):
    return load_encoder(model_id, device, cache_dir)
