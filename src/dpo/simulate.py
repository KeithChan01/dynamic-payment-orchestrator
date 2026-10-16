from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from dpo.config_loader import load_all
from dpo.orchestrator import Orchestrator
from dpo.schemas import RouteRequest


def random_requests(n: int = 200, seed: int = 7) -> list[RouteRequest]:
    cfg = load_all()
    currencies = cfg["routes"]["currencies"]
    countries = cfg["routes"]["countries"]
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    out: list[RouteRequest] = []
    for i in range(n):
        currency = str(rng.choice(currencies))
        country = str(rng.choice(countries))
        cross = bool(rng.random() < 0.22)
        local = str(rng.choice(currencies)) if cross else currency
        ip_country = str(rng.choice(countries)) if cross else country
        out.append(
            RouteRequest(
                timestamp_utc=start + timedelta(hours=int(rng.integers(0, 24 * 180))),
                transaction_type="withdrawal" if rng.random() < 0.25 else "deposit",
                amount=float(round(10 * (10 ** rng.uniform(0, 3.1)), 2)),
                currency=currency,
                local_currency=local,
                country=country,
                ip_country=ip_country,
                is_cross_border=cross,
            )
        )
    return out


def _pick_baseline(result, strategy: str) -> str | None:
    ranked = result.ranked
    if not ranked:
        return None
    if strategy == "always_stripe":
        for row in ranked:
            if row.route == "stripe":
                return "stripe"
        return ranked[0].route
    if strategy == "cheapest":
        return min(ranked, key=lambda row: row.fee_rate).route
    if strategy == "fastest":
        return min(ranked, key=lambda row: row.settlement_p90_minutes).route
    raise KeyError(strategy)


def summarize(engine: Orchestrator, requests: list[RouteRequest], policy: str = "balanced") -> dict:
    strategies = {
        "orchestrator": None,
        "always_stripe": "always_stripe",
        "cheapest": "cheapest",
        "fastest": "fastest",
    }
    buckets: dict[str, list[tuple[float, float, float]]] = {name: [] for name in strategies}
    no_route = 0
    for req in requests:
        result = engine.evaluate(req.model_copy(update={"policy": policy}))
        by_route = {row.route: row for row in result.ranked}
        if not result.ranked:
            no_route += 1
            continue
        for name, strategy in strategies.items():
            chosen = result.recommended if strategy is None else _pick_baseline(result, strategy)
            if chosen is None or chosen not in by_route:
                continue
            row = by_route[chosen]
            buckets[name].append(
                (row.success_probability, row.fee_amount, row.settlement_p90_minutes)
            )

    def _agg(name: str) -> dict:
        rows = buckets[name]
        if not rows:
            return {"n": 0}
        success, fees, p90 = zip(*rows)
        return {
            "n": len(rows),
            "mean_success": round(float(np.mean(success)), 4),
            "mean_fee": round(float(np.mean(fees)), 4),
            "mean_p90_minutes": round(float(np.mean(p90)), 2),
        }

    return {
        "n": len(requests),
        "policy": policy,
        "no_eligible_route": no_route,
        "model_mode": engine.mode,
        "strategies": {name: _agg(name) for name in strategies},
    }
