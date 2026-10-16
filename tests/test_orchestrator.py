from dpo.orchestrator import Orchestrator
from dpo.schemas import RouteRequest


def test_evaluate_returns_ranked_eligible_routes(engine: Orchestrator, sample_request: RouteRequest):
    result = engine.evaluate(sample_request)
    assert result.recommended
    assert result.ranked
    assert result.ranked[0].route == result.recommended
    assert all(row.eligible for row in result.ranked)
    assert "Synthetic" in result.disclaimer


def test_jpy_withdrawal_excludes_square_and_revolut(engine: Orchestrator):
    result = engine.evaluate(
        RouteRequest(
            transaction_type="withdrawal",
            amount=500,
            currency="JPY",
            country="Japan",
        )
    )
    blocked = {row.route for row in result.ineligible}
    assert "square" in blocked
    assert "revolut_pay" in blocked
    assert result.recommended not in blocked


def test_compare_changes_with_policy(engine: Orchestrator):
    request = RouteRequest(
        transaction_type="deposit",
        amount=9000,
        currency="USD",
        local_currency="EUR",
        country="United States",
        ip_country="Germany",
        is_cross_border=True,
    )
    compared = engine.compare(request)
    assert set(compared) == {"balanced", "cost", "speed", "approval"}
    winners = {name: result.recommended for name, result in compared.items()}
    assert winners["cost"] is not None
    assert winners["speed"] is not None
