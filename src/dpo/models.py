from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from dpo.features import OUTCOME_CATS, OUTCOME_NUMS, ROUTE_CATS, ROUTE_NUMS, context_frame
from dpo.paths import models_dir


class TabularEncoder:
    def __init__(self, cat_cols: list[str], num_cols: list[str]):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )

    def fit(self, frame: pd.DataFrame) -> TabularEncoder:
        self.encoder.fit(frame[self.cat_cols].astype(str))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        cats = self.encoder.transform(frame[self.cat_cols].astype(str))
        nums = frame[self.num_cols].to_numpy(dtype=float)
        return np.hstack([nums, cats])

    @property
    def categorical_features(self) -> list[int]:
        return list(range(len(self.num_cols), len(self.num_cols) + len(self.cat_cols)))


@dataclass
class TrainedBundle:
    route_encoder: TabularEncoder
    route_model: HistGradientBoostingClassifier
    route_labels: LabelEncoder
    success_encoder: TabularEncoder
    success_model: HistGradientBoostingClassifier
    settle_encoder: TabularEncoder
    settle_p50: HistGradientBoostingRegressor
    settle_p90: HistGradientBoostingRegressor


def heuristic_relevance(context: dict, routes: list[str], priors: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for route in routes:
        score = float(priors["routes"].get(route, {}).get("share", 1.0 / max(len(routes), 1)))
        if context["is_cross_border"]:
            if route == "wise":
                score *= 1.45
            if route == "bank_transfer":
                score *= 0.75
        if context["amount"] >= 5000:
            if route in {"adyen", "bank_transfer", "checkout"}:
                score *= 1.2
            if route in {"paypal", "square"}:
                score *= 0.7
        scores[route] = score
    total = sum(scores.values()) or 1.0
    return {route: value / total for route, value in scores.items()}


def heuristic_success(context: dict, route: str, priors: dict[str, Any]) -> float:
    p = float(priors["routes"][route]["success"])
    if context["is_cross_border"]:
        p *= 0.97 if route != "wise" else 1.0
    if context["is_weekend"] and route == "bank_transfer":
        p *= 0.93
    if context["amount"] >= 8000:
        p *= 0.97
        if route in {"paypal", "square"}:
            p *= 0.96
    return float(np.clip(p, 0.05, 0.99))


def heuristic_settlement(context: dict, route: str, priors: dict[str, Any]) -> tuple[float, float]:
    spec = priors["routes"][route]
    p50 = float(spec["p50"])
    p90 = float(spec["p90"])
    slow = route in {"bank_transfer", "wise"}
    if (context["is_weekend"] or not context["banking_hours"]) and slow:
        p50 *= 1.8
        p90 *= 1.5
    if context["is_cross_border"] and route == "bank_transfer":
        p50 *= 1.3
        p90 *= 1.3
    return p50, p90


class ModelBundle:
    def __init__(self, trained: TrainedBundle | None, priors: dict[str, Any]):
        self.trained = trained
        self.priors = priors
        self.mode = "trained" if trained is not None else "heuristic"

    @classmethod
    def load(cls, priors: dict[str, Any], directory: Path | None = None) -> ModelBundle:
        path = (directory or models_dir()) / "bundle.joblib"
        if path.exists():
            return cls(joblib.load(path), priors)
        return cls(None, priors)

    def relevance(self, context: dict, routes: list[str]) -> dict[str, float]:
        if self.trained is None:
            return heuristic_relevance(context, routes, self.priors)
        frame = context_frame(context)
        x = self.trained.route_encoder.transform(frame)
        proba = self.trained.route_model.predict_proba(x)[0]
        labels = list(self.trained.route_labels.classes_)
        raw = {label: float(prob) for label, prob in zip(labels, proba) if label in routes}
        for route in routes:
            raw.setdefault(route, 0.0)
        total = sum(raw.values()) or 1.0
        return {route: value / total for route, value in raw.items()}

    def success(self, context: dict, route: str) -> float:
        if self.trained is None:
            return heuristic_success(context, route, self.priors)
        frame = context_frame(context, payment_type=route)
        x = self.trained.success_encoder.transform(frame)
        return float(self.trained.success_model.predict_proba(x)[0, 1])

    def settlement(self, context: dict, route: str) -> tuple[float, float]:
        if self.trained is None:
            return heuristic_settlement(context, route, self.priors)
        frame = context_frame(context, payment_type=route)
        x = self.trained.settle_encoder.transform(frame)
        p50 = max(float(self.trained.settle_p50.predict(x)[0]), 0.01)
        p90 = max(float(self.trained.settle_p90.predict(x)[0]), p50)
        return p50, p90
