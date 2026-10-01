"""Configured exemption/threshold rules."""
from utils.schema import VATDecision


def exemption_for(tx):
    desc = (tx.description or "").lower()
    keywords = {
        "medical": "Potential medical/health exemption; verify national conditions",
        "education": "Potential education exemption; verify national conditions",
        "insurance": "Potential insurance/financial exemption; verify national conditions",
    }
    for keyword, treatment in keywords.items():
        if keyword in desc:
            return VATDecision(
                treatment=treatment,
                compliance_status="REVIEW_REQUIRED",
                reason_codes=[f"POTENTIAL_{keyword.upper()}_EXEMPTION"],
                legal_basis=["Council Directive 2006/112/EC, exemption provisions"],
            )
    return None
