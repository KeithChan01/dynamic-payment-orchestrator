from dpo.config_loader import load_all
from dpo.ranking import cascade, rank_routes


def _row(route, success, fee_rate, p90, relevance=0.1):
    return {
        "route": route,
        "success_probability": success,
        "fee_rate": fee_rate,
        "settlement_p90_minutes": p90,
        "relevance": relevance,
    }


def test_cost_policy_prefers_cheaper_route():
    cfg = load_all()["ranking"]
    ranked = rank_routes(
        [
            _row("stripe", 0.93, 0.032, 2),
            _row("bank_transfer", 0.92, 0.002, 400),
        ],
        "cost",
        cfg,
    )
    assert ranked[0]["route"] == "bank_transfer"


def test_speed_policy_prefers_faster_route():
    cfg = load_all()["ranking"]
    ranked = rank_routes(
        [
            _row("stripe", 0.93, 0.032, 2),
            _row("bank_transfer", 0.92, 0.002, 1800),
        ],
        "speed",
        cfg,
    )
    assert ranked[0]["route"] == "stripe"


def test_cascade_follows_score_order():
    cfg = load_all()["ranking"]
    ranked = rank_routes(
        [
            _row("paypal", 0.50, 0.01, 5, 0.9),
            _row("adyen", 0.94, 0.02, 8, 0.2),
            _row("stripe", 0.90, 0.03, 2, 0.1),
        ],
        "balanced",
        cfg,
    )
    order = cascade(ranked, cfg)
    assert order == [row["route"] for row in ranked[: cfg["cascade_size"]]]
