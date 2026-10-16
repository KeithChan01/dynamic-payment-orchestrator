from dpo.config_loader import load_all
from dpo.fees import estimate_fee


def test_domestic_fee_is_percent_plus_fixed():
    cfg = load_all()
    fee, rate = estimate_fee(
        "stripe",
        {
            "amount": 100.0,
            "currency": "USD",
            "local_currency": "USD",
            "is_cross_border": 0,
        },
        cfg,
    )
    assert fee == round(100 * 0.029 + 0.30, 4)
    assert rate == fee / 100


def test_fx_fee_applies_when_cross_border():
    cfg = load_all()
    domestic, _ = estimate_fee(
        "wise",
        {"amount": 1000, "currency": "USD", "local_currency": "USD", "is_cross_border": 0},
        cfg,
    )
    fx, _ = estimate_fee(
        "wise",
        {"amount": 1000, "currency": "USD", "local_currency": "EUR", "is_cross_border": 1},
        cfg,
    )
    assert fx > domestic
