from __future__ import annotations

from pathlib import Path
from typing import Any

from dpo.config_loader import load_all, route_label
from dpo.eligibility import eligible_routes
from dpo.features import request_to_context
from dpo.fees import estimate_fee
from dpo.models import ModelBundle
from dpo.ranking import cascade, rank_routes
from dpo.schemas import DISCLAIMER, RouteRequest, RouteResponse, RouteScore


class Orchestrator:
    def __init__(self, config_root: Path | None = None, models_root: Path | None = None):
        root = str(config_root) if config_root else None
        if root:
            load_all.cache_clear()
        self.cfg: dict[str, Any] = load_all(root)
        self.models = ModelBundle.load(self.cfg["priors"], models_root)

    @property
    def mode(self) -> str:
        return self.models.mode

    def evaluate(self, request: RouteRequest | dict) -> RouteResponse:
        req = request if isinstance(request, RouteRequest) else RouteRequest.model_validate(request)
        context = request_to_context(req)
        allowed, blocked = eligible_routes(context, self.cfg)
        relevance = self.models.relevance(context, allowed) if allowed else {}

        scored: list[dict[str, Any]] = []
        for route in allowed:
            fee_amount, fee_rate = estimate_fee(route, context, self.cfg)
            p50, p90 = self.models.settlement(context, route)
            success = self.models.success(context, route)
            scored.append(
                {
                    "route": route,
                    "label": route_label(route, self.cfg),
                    "eligible": True,
                    "relevance": relevance.get(route, 0.0),
                    "success_probability": success,
                    "settlement_p50_minutes": p50,
                    "settlement_p90_minutes": p90,
                    "fee_amount": fee_amount,
                    "fee_rate": fee_rate,
                    "ineligible_reason": None,
                }
            )

        ranked_rows = rank_routes(scored, req.policy, self.cfg["ranking"])
        threshold = float(self.cfg["ranking"]["fallback_success_threshold"])
        for index, row in enumerate(ranked_rows):
            row["fallback_threshold"] = threshold
            if index + 1 < len(ranked_rows):
                row["next_route"] = ranked_rows[index + 1]["label"]
        order = cascade(ranked_rows, self.cfg["ranking"])
        ranked_models = [_to_score(row) for row in ranked_rows]
        ineligible = [
            RouteScore(
                route=route,
                label=route_label(route, self.cfg),
                eligible=False,
                relevance=0.0,
                success_probability=0.0,
                settlement_p50_minutes=0.0,
                settlement_p90_minutes=0.0,
                fee_amount=0.0,
                fee_rate=0.0,
                score=0.0,
                reasons=[reason],
                ineligible_reason=reason,
            )
            for route, reason in blocked.items()
        ]

        recommended = order[0] if order else None
        fallbacks = order[1:] if order else []
        return RouteResponse(
            recommended=recommended,
            recommended_label=route_label(recommended, self.cfg) if recommended else None,
            fallbacks=fallbacks,
            cascade=order,
            ranked=ranked_models,
            ineligible=ineligible,
            policy=req.policy,
            model_mode=self.models.mode,  # type: ignore[arg-type]
            disclaimer=DISCLAIMER,
        )

    def compare(self, request: RouteRequest | dict) -> dict[str, RouteResponse]:
        req = request if isinstance(request, RouteRequest) else RouteRequest.model_validate(request)
        return {
            policy: self.evaluate(req.model_copy(update={"policy": policy}))
            for policy in self.cfg["ranking"]["policies"]
        }


def _to_score(row: dict[str, Any]) -> RouteScore:
    comps = row.get("components", {})
    reasons = [
        f"Predicted success {row['success_probability']:.1%}",
        f"Estimated fee {row['fee_amount']:.2f} ({row['fee_rate']:.2%})",
        f"Typical settlement {row['settlement_p50_minutes']:.1f} min (P90 {row['settlement_p90_minutes']:.1f})",
        f"Relevance {row['relevance']:.1%}",
    ]
    threshold = float(row.get("fallback_threshold") or 0)
    if threshold and row["success_probability"] < threshold:
        reasons.append(
            f"Below reliability threshold ({threshold:.0%}); keep {row.get('next_route', 'next route')} ready as fallback"
        )
    if comps:
        reasons.append(
            "Score mix "
            + ", ".join(f"{key}={value:.2f}" for key, value in comps.items())
        )
    return RouteScore(
        route=row["route"],
        label=row["label"],
        eligible=True,
        relevance=round(row["relevance"], 6),
        success_probability=round(row["success_probability"], 6),
        settlement_p50_minutes=round(row["settlement_p50_minutes"], 4),
        settlement_p90_minutes=round(row["settlement_p90_minutes"], 4),
        fee_amount=round(row["fee_amount"], 4),
        fee_rate=round(row["fee_rate"], 6),
        score=round(row["score"], 6),
        reasons=reasons,
    )
