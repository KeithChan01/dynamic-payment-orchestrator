from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from dpo.schemas import RouteRequest

ROUTE_CATS = ["transaction_type", "country", "ip_country", "jurisdiction", "currency", "local_currency", "app_type", "day_of_week"]
ROUTE_NUMS = ["amount", "log_amount", "hour", "is_weekend", "is_cross_border"]

OUTCOME_CATS = ["transaction_type", "currency", "day_of_week", "payment_type"]
OUTCOME_NUMS = ["amount", "log_amount", "hour", "is_weekend", "is_cross_border"]


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_amount"] = np.log1p(out["amount"].astype(float))
    if "is_weekend" in out.columns:
        out["is_weekend"] = out["is_weekend"].astype(int)
    if "is_cross_border" in out.columns:
        out["is_cross_border"] = out["is_cross_border"].astype(int)
    return out


def banking_hours(hour: int, is_weekend: int) -> int:
    return int((not is_weekend) and 9 <= hour <= 16)


def request_to_context(req: RouteRequest, now: datetime | None = None) -> dict:
    ts = req.timestamp_utc or now or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    hour = ts.hour
    is_weekend = int(ts.weekday() >= 5)
    day_of_week = ts.strftime("%A")
    return {
        "timestamp_utc": ts,
        "transaction_type": req.transaction_type,
        "country": req.country or "Unknown",
        "ip_country": req.ip_country or req.country or "Unknown",
        "jurisdiction": req.jurisdiction or "Unknown",
        "currency": req.currency,
        "local_currency": req.local_currency or req.currency,
        "amount": float(req.amount),
        "log_amount": float(np.log1p(req.amount)),
        "app_type": req.app_type,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_cross_border": int(bool(req.is_cross_border)),
        "available_routes": req.available_routes,
        "policy": req.policy,
        "banking_hours": banking_hours(hour, is_weekend),
    }


def context_frame(context: dict, payment_type: str | None = None) -> pd.DataFrame:
    row = {key: context[key] for key in (*ROUTE_CATS, *ROUTE_NUMS) if key in context}
    if payment_type is not None:
        row["payment_type"] = payment_type
    return pd.DataFrame([row])
