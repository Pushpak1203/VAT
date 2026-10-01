"""OSS/IOSS scheme handling."""
from utils.schema import VATDecision


def evaluate_scheme(tx):
    scheme = (tx.scheme or "").upper()
    if scheme == "OSS":
        return VATDecision(
            treatment="EU One Stop Shop eligible transaction; destination-country VAT reporting via OSS subject to eligibility",
            compliance_status="REVIEW_REQUIRED",
            reason_codes=["OSS"],
            legal_basis=["Council Directive 2006/112/EC, Articles 369a et seq."],
        )
    if scheme == "IOSS":
        return VATDecision(
            treatment="Import One Stop Shop transaction; import VAT simplification subject to IOSS conditions",
            compliance_status="REVIEW_REQUIRED",
            reason_codes=["IOSS"],
            legal_basis=["EU VAT e-commerce/IOSS rules"],
        )
    if tx.is_distance_sale and tx.buyer_country:
        return VATDecision(
            treatment="EU distance sale; OSS threshold/scheme eligibility should be assessed",
            compliance_status="REVIEW_REQUIRED",
            reason_codes=["DISTANCE_SALE"],
            legal_basis=["EU VAT e-commerce rules"],
        )
    return None
