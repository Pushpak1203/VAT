"""Central configuration for VAT Chain Auditor."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class PipelineMode(str, Enum):
    LLM_ONLY = "LLM_ONLY"
    RULES_ONLY = "RULES_ONLY"
    HYBRID = "HYBRID"
    HYBRID_RAG = "HYBRID_RAG"
    DEEPSEEK_RAG = "DEEPSEEK_RAG"


@dataclass
class Settings:
    python_version: str = "3.10.10"
    pipeline_mode: PipelineMode = PipelineMode.HYBRID_RAG
    device: str = os.getenv("VAT_DEVICE", "auto")
    hf_cache_dir: str | None = os.getenv("HF_HOME")
    allow_model_download: bool = os.getenv("ALLOW_MODEL_DOWNLOAD", "true").lower() == "true"
    load_optional_gemma: bool = os.getenv("LOAD_OPTIONAL_GEMMA", "false").lower() == "true"

    qwen_model_id: str = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-8B")
    deepseek_model_id: str = os.getenv(
        "DEEPSEEK_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    gemma_model_id: str = os.getenv("GEMMA_MODEL_ID", "google/gemma-3-4b-it")
    embedding_model_id: str = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")
    reranker_model_id: str = os.getenv(
        "RERANKER_MODEL_ID", "BAAI/bge-reranker-v2-m3"
    )
    deberta_model_id: str = os.getenv(
        "DEBERTA_MODEL_ID", "microsoft/deberta-v3-base"
    )
    layoutlm_model_id: str = os.getenv(
        "LAYOUTLM_MODEL_ID", "microsoft/layoutlmv3-base"
    )

    postgres_dsn: str = os.getenv(
        "POSTGRES_DSN",
        "postgresql://postgres:postgres@localhost:5432/vat_chain_auditor",
    )
    vector_table: str = os.getenv("VECTOR_TABLE", "vat_legal_documents")
    embedding_dimension: int = 1024
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "12"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))

    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "700"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    lightgbm_model_path: Path = field(
        default_factory=lambda: Path(os.getenv("LIGHTGBM_MODEL_PATH", "artifacts/lightgbm.txt"))
    )
    calibration_model_path: Path = field(
        default_factory=lambda: Path(os.getenv("CALIBRATION_MODEL_PATH", "artifacts/isotonic.pkl"))
    )

    @property
    def torch_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


def get_settings() -> Settings:
    return Settings()
