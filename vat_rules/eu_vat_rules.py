"""Deterministic EU VAT rules."""
from utils.schema import VATDecision
from models.deberta_classifier import EU_COUNTRIES


def evaluate_eu_transaction(tx, classification) -> VATDecision:
    seller = tx.seller_country
    buyer = tx.buyer_country or tx.destination_country
    if seller in EU_COUNTRIES and buyer in EU_COUNTRIES and seller != buyer:
        if classification.customer_type == "B2B" and tx.buyer_vat_id:
            return VATDecision(
                treatment="intra-EU B2B supply; customer-accounted/reverse-charge treatment subject to conditions",
                compliance_status="COMPLIANT_IF_CONDITIONS_MET",
                reason_codes=["INTRA_EU_B2B", "VALID_VAT_ID_REQUIRED"],
                legal_basis=["Council Directive 2006/112/EC, Article 138"],
                applicable_rate=0.0,
            )
        return VATDecision(
            treatment="cross-border EU B2C supply; destination/place-of-supply rules require review",
            compliance_status="REVIEW_REQUIRED",
            reason_codes=["INTRA_EU_B2C"],
            legal_basis=["Council Directive 2006/112/EC, place-of-supply provisions"],
        )
    if seller == buyer and seller in EU_COUNTRIES:
        if tx.vat_amount > 0 and tx.net_amount > 0:
            return VATDecision(
                treatment="domestic taxable supply",
                compliance_status="REVIEW_REQUIRED",
                reason_codes=["DOMESTIC_TAXABLE"],
                legal_basis=["Council Directive 2006/112/EC"],
            )
        return VATDecision(
            treatment="domestic supply with no VAT charged; taxable status requires review",
            compliance_status="NON_COMPLIANT_POTENTIAL",
            reason_codes=["DOMESTIC_ZERO_VAT"],
            legal_basis=["Council Directive 2006/112/EC"],
        )
    if seller in EU_COUNTRIES and buyer and buyer not in EU_COUNTRIES:
        return VATDecision(
            treatment="export supply outside EU; zero-rating/exemption subject to evidence",
            compliance_status="COMPLIANT_IF_CONDITIONS_MET",
            reason_codes=["EXPORT"],
            legal_basis=["Council Directive 2006/112/EC"],
            applicable_rate=0.0,
        )
    return VATDecision(
        treatment="cross-border transaction outside configured EU scope",
        compliance_status="REVIEW_REQUIRED",
        reason_codes=["OUT_OF_SCOPE"],
        legal_basis=[],
    )
