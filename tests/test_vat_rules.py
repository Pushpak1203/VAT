from utils.schema import TransactionInput
from models.deberta_classifier import DebertaClassifier
from vat_rules.eu_vat_rules import evaluate_eu_transaction
from vat_rules.uk_vat_rules import evaluate_uk_transaction
from vat_rules.oss_ioss_handler import evaluate_scheme
from config import Settings


def test_intra_eu_b2b():
    tx = TransactionInput(
        transaction_id="t",
        seller_country="DE",
        buyer_country="FR",
        buyer_vat_id="FR12345678901",
        customer_type="B2B",
        supply_type="goods",
        net_amount=1000,
        vat_amount=0,
        description="goods",
    )
    c = DebertaClassifier(Settings(allow_model_download=False)).classify(tx)
    result = evaluate_eu_transaction(tx, c)
    assert result.reason_codes == ["INTRA_EU_B2B"]
    assert result.applicable_rate == 0.0


def test_uk_import():
    tx = TransactionInput(transaction_id="t", seller_country="DE", buyer_country="GB", net_amount=100)
    c = DebertaClassifier(Settings(allow_model_download=False)).classify(tx)
    result = evaluate_uk_transaction(tx, c)
    assert "UK_IMPORT" in result.reason_codes


def test_oss():
    tx = TransactionInput(transaction_id="t", seller_country="DE", buyer_country="FR", scheme="OSS")
    assert evaluate_scheme(tx).reason_codes == ["OSS"]
