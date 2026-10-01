"""UK post-Brexit VAT rules."""
from utils.schema import VATDecision


def evaluate_uk_transaction(tx, classification) -> VATDecision:
    seller = tx.seller_country
    buyer = tx.buyer_country or tx.destination_country
    if seller == "GB" and buyer and buyer != "GB":
        return VATDecision(
            treatment="UK export; zero-rating may apply subject to export evidence",
            compliance_status="COMPLIANT_IF_CONDITIONS_MET",
            reason_codes=["UK_EXPORT"],
            legal_basis=["HMRC VAT guidance - exports"],
            applicable_rate=0.0,
        )
    if buyer == "GB" and seller and seller != "GB":
        return VATDecision(
            treatment="UK import; import VAT/customs treatment required",
            compliance_status="REVIEW_REQUIRED",
            reason_codes=["UK_IMPORT"],
            legal_basis=["HMRC VAT guidance - imports"],
        )
    return VATDecision(
        treatment="UK domestic taxable supply",
        compliance_status="REVIEW_REQUIRED",
        reason_codes=["UK_DOMESTIC"],
        legal_basis=["HMRC VAT guidance"],
    )
