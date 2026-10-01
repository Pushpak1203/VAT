"""Step 4: deterministic VAT rules engine."""
from vat_rules.eu_vat_rules import evaluate_eu_transaction
from vat_rules.uk_vat_rules import evaluate_uk_transaction
from vat_rules.oss_ioss_handler import evaluate_scheme
from vat_rules.exemptions import exemption_for


class VATRulesEngine:
    def evaluate(self, tx, classification):
        scheme_result = evaluate_scheme(tx)
        if scheme_result:
            return scheme_result
        exemption = exemption_for(tx)
        if exemption:
            return exemption
        if tx.seller_country == "GB" or tx.buyer_country == "GB":
            return evaluate_uk_transaction(tx, classification)
        return evaluate_eu_transaction(tx, classification)
