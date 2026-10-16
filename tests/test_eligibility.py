from dpo.config_loader import load_all
from dpo.eligibility import evaluate_route


def _ctx(**overrides):
    base = {
        "transaction_type": "deposit",
        "currency": "USD",
        "amount": 100.0,
        "available_routes": None,
    }
    base.update(overrides)
    return base


def test_square_rejects_jpy_and_withdrawals():
    cfg = load_all()
    ok, reason = evaluate_route("square", _ctx(currency="JPY"), cfg)
    assert not ok and reason
    ok, reason = evaluate_route("square", _ctx(transaction_type="withdrawal"), cfg)
    assert not ok


def test_paypal_amount_cap_and_withdrawal_currency():
    cfg = load_all()
    ok, _ = evaluate_route("paypal", _ctx(amount=12500), cfg)
    assert not ok
    ok, _ = evaluate_route("paypal", _ctx(transaction_type="withdrawal", currency="JPY"), cfg)
    assert not ok
    ok, _ = evaluate_route("paypal", _ctx(transaction_type="withdrawal", currency="USD"), cfg)
    assert ok


def test_merchant_allowlist():
    cfg = load_all()
    ok, reason = evaluate_route("adyen", _ctx(available_routes=["stripe"]), cfg)
    assert not ok
    assert "not enabled" in reason
