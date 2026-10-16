from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

PolicyName = Literal["balanced", "cost", "speed", "approval"]
TransactionType = Literal["deposit", "withdrawal"]
ModelMode = Literal["trained", "heuristic"]

DISCLAIMER = (
    "Synthetic PayMind reference data. Rankings are not real payment-provider performance "
    "and must not be used as production routing benchmarks."
)


class RouteRequest(BaseModel):
    timestamp_utc: datetime | None = None
    transaction_type: TransactionType
    amount: float = Field(gt=0, le=10_000_000)
    currency: str
    local_currency: str | None = None
    country: str | None = None
    ip_country: str | None = None
    jurisdiction: str | None = None
    app_type: str = "DEMO"
    is_cross_border: bool | None = None
    available_routes: list[str] | None = None
    policy: PolicyName = "balanced"

    @field_validator("currency", "local_currency", mode="before")
    @classmethod
    def _upper_ccy(cls, value: str | None) -> str | None:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _defaults(self) -> RouteRequest:
        if self.local_currency is None:
            self.local_currency = self.currency
        if self.ip_country is None:
            self.ip_country = self.country
        if self.jurisdiction is None and self.country:
            self.jurisdiction = f"{self.country} Demo"
        if self.is_cross_border is None:
            country_mismatch = (
                self.country is not None
                and self.ip_country is not None
                and self.country != self.ip_country
            )
            ccy_mismatch = self.currency != self.local_currency
            self.is_cross_border = bool(country_mismatch or ccy_mismatch)
        return self


class RouteScore(BaseModel):
    route: str
    label: str
    eligible: bool
    relevance: float
    success_probability: float
    settlement_p50_minutes: float
    settlement_p90_minutes: float
    fee_amount: float
    fee_rate: float
    score: float
    reasons: list[str]
    ineligible_reason: str | None = None


class RouteResponse(BaseModel):
    recommended: str | None
    recommended_label: str | None
    fallbacks: list[str]
    cascade: list[str]
    ranked: list[RouteScore]
    ineligible: list[RouteScore]
    policy: PolicyName
    model_mode: ModelMode
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: str
    model_mode: ModelMode
    routes: list[str]


class CompareResponse(BaseModel):
    policies: dict[str, RouteResponse]
    disclaimer: str = DISCLAIMER
