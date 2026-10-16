from __future__ import annotations

from typing import Any


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def component_scores(
    success: float,
    fee_rate: float,
    p90_minutes: float,
    relevance: float,
    ranking_cfg: dict[str, Any],
) -> dict[str, float]:
    fee_cap = ranking_cfg["fee_rate_cap"]
    settle_cap = ranking_cfg["settlement_cap_minutes"]
    return {
        "success": _clip01(success),
        "fee": _clip01(1.0 - (fee_rate / fee_cap)),
        "settlement": _clip01(1.0 - (p90_minutes / settle_cap)),
        "relevance": _clip01(relevance),
    }


def weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
    return sum(components[key] * float(weights[key]) for key in weights)


def rank_routes(
    rows: list[dict[str, Any]],
    policy: str,
    ranking_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    weights = ranking_cfg["policies"][policy]
    scored = []
    for row in rows:
        components = component_scores(
            row["success_probability"],
            row["fee_rate"],
            row["settlement_p90_minutes"],
            row["relevance"],
            ranking_cfg,
        )
        item = dict(row)
        item["components"] = components
        item["score"] = round(weighted_score(components, weights), 6)
        scored.append(item)
    return sorted(scored, key=lambda item: item["score"], reverse=True)


def cascade(ranked: list[dict[str, Any]], ranking_cfg: dict[str, Any]) -> list[str]:
    size = int(ranking_cfg["cascade_size"])
    return [row["route"] for row in ranked[:size]]
