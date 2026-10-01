"""Pydantic schemas shared by the pipeline."""
from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    transaction_id: str
    seller_country: str | None = None
    buyer_country: str | None = None
    buyer_vat_id: str | None = None
    seller_vat_id: str | None = None
    customer_type: str | None = None
    supply_type: str | None = None
    invoice_date: date | None = None
    currency: str = "EUR"
    net_amount: float = 0.0
    vat_amount: float = 0.0
    gross_amount: float | None = None
    description: str = ""
    destination_country: str | None = None
    origin_country: str | None = None
    product_category: str | None = None
    is_distance_sale: bool = False
    scheme: str | None = None

    @field_validator("seller_country", "buyer_country", "destination_country", "origin_country")
    @classmethod
    def upper_country(cls, value: str | None) -> str | None:
        return value.upper() if value else value


class OCRResult(BaseModel):
    text: str = ""
    fields: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    source_path: str | None = None


class TransactionClassification(BaseModel):
    customer_type: str
    supply_type: str
    region: str
    vat_category: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)


class RiskResult(BaseModel):
    risk_score: float
    high_risk: bool
    model_version: str


class AnomalyResult(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    model_version: str


class VATDecision(BaseModel):
    treatment: str
    compliance_status: str
    reason_codes: list[str] = Field(default_factory=list)
    legal_basis: list[str] = Field(default_factory=list)
    applicable_rate: float | None = None


class LegalEvidence(BaseModel):
    title: str
    jurisdiction: str
    citation: str
    text: str
    score: float


class LLMReasoningResult(BaseModel):
    explanation: str
    rationale: str
    model_name: str
    raw_output: str | None = None


class VATAuditResult(BaseModel):
    transaction_id: str
    treatment: str
    compliance_status: str
    risk_score: float
    anomaly_flag: bool
    explanation: str
    rationale: str
    legal_references: list[str]
    calibrated_confidence: float
    model_mode: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    intermediate: dict[str, Any] = Field(default_factory=dict)
