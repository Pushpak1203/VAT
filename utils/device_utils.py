"""Hardware and memory helpers."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def gpu_summary() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False, "device": "cpu"}
        props = torch.cuda.get_device_properties(0)
        return {
            "cuda": True,
            "device": torch.cuda.get_device_name(0),
            "total_vram_gb": round(props.total_memory / 1024**3, 2),
            "free_vram_gb": round((props.total_memory - torch.cuda.memory_allocated(0)) / 1024**3, 2),
        }
    except Exception as exc:
        return {"cuda": False, "device": "cpu", "error": str(exc)}


def ensure_vram_budget(required_gb: float = 5.5) -> None:
    summary = gpu_summary()
    if not summary.get("cuda"):
        return
    free = summary.get("free_vram_gb", 0)
    if free < required_gb:
        raise RuntimeError(
            f"Insufficient free GPU memory: {free:.2f} GB available, "
            f"{required_gb:.2f} GB requested. Close GPU-heavy processes or use CPU."
        )
