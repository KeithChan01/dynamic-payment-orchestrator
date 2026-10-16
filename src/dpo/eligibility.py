from __future__ import annotations

from typing import Any


def evaluate_route(route_id: str, context: dict, cfg: dict[str, Any]) -> tuple[bool, str | None]:
    rules = cfg["eligibility"]["routes"].get(route_id)
    if rules is None:
        return False, f"{route_id} is not a configured route"

    defaults = cfg["eligibility"]["defaults"]
    deny = rules.get("deny_reason")
    currencies = rules.get("currencies") or defaults["currencies"]
    tx_types = rules.get("transaction_types") or defaults["transaction_types"]
    currency = context["currency"]
    tx_type = context["transaction_type"]
    amount = context["amount"]

    merchant = context.get("available_routes")
    if merchant is not None and route_id not in merchant:
        return False, f"{route_id} is not enabled for this merchant"

    if tx_type not in tx_types:
        return False, deny or f"{route_id} does not accept {tx_type}s"

    if currency not in currencies:
        return False, deny or f"{route_id} does not accept {currency}"

    withdrawal_ccy = rules.get("withdrawal_currencies")
    if tx_type == "withdrawal" and withdrawal_ccy and currency not in withdrawal_ccy:
        return False, deny or f"{route_id} withdrawals are not available in {currency}"

    max_amount = rules.get("max_amount")
    if max_amount is not None and amount > max_amount:
        return False, deny or f"{route_id} amount cap is {max_amount}"

    return True, None


def eligible_routes(context: dict, cfg: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    allowed: list[str] = []
    blocked: dict[str, str] = {}
    for item in cfg["routes"]["routes"]:
        route_id = item["id"]
        ok, reason = evaluate_route(route_id, context, cfg)
        if ok:
            allowed.append(route_id)
        else:
            blocked[route_id] = reason or "ineligible"
    return allowed, blocked
