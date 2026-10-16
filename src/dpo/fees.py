from __future__ import annotations

from typing import Any


def estimate_fee(route_id: str, context: dict, cfg: dict[str, Any]) -> tuple[float, float]:
    spec = cfg["fees"]["routes"][route_id]
    amount = float(context["amount"])
    cross = bool(context.get("is_cross_border")) or context["currency"] != context["local_currency"]
    percent = spec["percent"] / 100.0
    fixed = spec["fixed"]
    fx = (spec["fx_percent"] / 100.0 * amount) if cross else 0.0
    fee = amount * percent + fixed + fx
    rate = fee / amount if amount else 0.0
    return round(fee, 4), round(rate, 6)
