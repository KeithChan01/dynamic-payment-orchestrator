from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from dpo.data import chronological_split, load_table
from dpo.features import OUTCOME_CATS, OUTCOME_NUMS, ROUTE_CATS, ROUTE_NUMS, add_derived_columns
from dpo.models import TabularEncoder, TrainedBundle
from dpo.paths import models_dir


def _maybe_sample(frame: pd.DataFrame, sample: int | None, rng: np.random.Generator) -> pd.DataFrame:
    if sample is None or sample >= len(frame):
        return frame
    idx = rng.choice(len(frame), size=sample, replace=False)
    return frame.iloc[np.sort(idx)].reset_index(drop=True)


def _fit_classifier(encoder: TabularEncoder, x: np.ndarray, y, *, sample_weight=None, **kwargs):
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=180,
        l2_regularization=0.1,
        categorical_features=encoder.categorical_features,
        random_state=42,
        **kwargs,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model


def _fit_quantile(encoder: TabularEncoder, x: np.ndarray, y: np.ndarray, quantile: float):
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_depth=6,
        learning_rate=0.08,
        max_iter=180,
        l2_regularization=0.1,
        categorical_features=encoder.categorical_features,
        random_state=42,
    )
    model.fit(x, y)
    return model


def _pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    delta = y_true - y_pred
    return float(np.mean(np.maximum(q * delta, (q - 1) * delta)))


def train(sample: int | None = None, output: Path | None = None) -> dict:
    rng = np.random.default_rng(42)
    out_dir = output or models_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    payment_method = add_derived_columns(_maybe_sample(load_table("payment_method"), sample, rng))
    success = add_derived_columns(_maybe_sample(load_table("success"), sample, rng))
    arrival = add_derived_columns(_maybe_sample(load_table("arrival"), sample, rng))

    pm_train, _, pm_test = chronological_split(payment_method)
    su_train, _, su_test = chronological_split(success)
    ar_train, _, ar_test = chronological_split(arrival)

    route_encoder = TabularEncoder(ROUTE_CATS, ROUTE_NUMS).fit(pm_train)
    route_labels = LabelEncoder().fit(pm_train["payment_type"])
    x_route_train = route_encoder.transform(pm_train)
    y_route_train = route_labels.transform(pm_train["payment_type"])
    route_model = _fit_classifier(route_encoder, x_route_train, y_route_train)

    success_encoder = TabularEncoder(OUTCOME_CATS, OUTCOME_NUMS).fit(su_train)
    x_su_train = success_encoder.transform(su_train)
    y_su_train = su_train["success"].to_numpy()
    # Keep natural class balance so P(success) stays calibrated for ranking.
    success_model = _fit_classifier(success_encoder, x_su_train, y_su_train)

    settle_encoder = TabularEncoder(OUTCOME_CATS, OUTCOME_NUMS).fit(ar_train)
    x_ar_train = settle_encoder.transform(ar_train)
    y_ar_train = ar_train["arrival_duration_minutes"].clip(lower=0).to_numpy()
    settle_p50 = _fit_quantile(settle_encoder, x_ar_train, y_ar_train, 0.5)
    settle_p90 = _fit_quantile(settle_encoder, x_ar_train, y_ar_train, 0.9)

    bundle = TrainedBundle(
        route_encoder=route_encoder,
        route_model=route_model,
        route_labels=route_labels,
        success_encoder=success_encoder,
        success_model=success_model,
        settle_encoder=settle_encoder,
        settle_p50=settle_p50,
        settle_p90=settle_p90,
    )
    joblib.dump(bundle, out_dir / "bundle.joblib")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sample": sample,
        "rows": {
            "payment_method": int(len(payment_method)),
            "success": int(len(success)),
            "arrival": int(len(arrival)),
        },
        "route": _eval_route(bundle, pm_test),
        "success": _eval_success(bundle, su_test),
        "settlement": _eval_settlement(bundle, ar_test),
        "split": "chronological 70/15/15; metrics reported on the latest 15%",
        "disclaimer": "Synthetic data. Metrics are not real PSP performance.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": "navk8690/paymind-reference-data",
                "files": ["payment_method copy.csv", "success copy.csv", "arrival copy.csv"],
                "trained_at": metrics["trained_at"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def _eval_route(bundle: TrainedBundle, test: pd.DataFrame) -> dict:
    x = bundle.route_encoder.transform(test)
    y = bundle.route_labels.transform(test["payment_type"])
    proba = bundle.route_model.predict_proba(x)
    pred = bundle.route_model.predict(x)
    top3 = np.argsort(proba, axis=1)[:, -3:]
    top3_acc = float(np.mean([label in row for label, row in zip(y, top3)]))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=np.arange(len(bundle.route_labels.classes_)))),
        "top3_accuracy": top3_acc,
        "majority_baseline_accuracy": float(
            test["payment_type"].value_counts(normalize=True).iloc[0]
        ),
    }


def _eval_success(bundle: TrainedBundle, test: pd.DataFrame) -> dict:
    x = bundle.success_encoder.transform(test)
    y = test["success"].to_numpy()
    proba = bundle.success_model.predict_proba(x)[:, 1]
    baseline = np.full_like(proba, y.mean(), dtype=float)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "brier_baseline": float(brier_score_loss(y, baseline)),
        "positive_rate": float(y.mean()),
    }


def _eval_settlement(bundle: TrainedBundle, test: pd.DataFrame) -> dict:
    x = bundle.settle_encoder.transform(test)
    y = test["arrival_duration_minutes"].clip(lower=0).to_numpy()
    p50 = np.clip(bundle.settle_p50.predict(x), 0, None)
    p90 = np.clip(bundle.settle_p90.predict(x), 0, None)
    median_baseline = np.full_like(y, np.median(y))
    return {
        "mae_p50": float(mean_absolute_error(y, p50)),
        "mae_median_baseline": float(mean_absolute_error(y, median_baseline)),
        "pinball_p50": _pinball(y, p50, 0.5),
        "pinball_p90": _pinball(y, p90, 0.9),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train orchestrator models from Hugging Face")
    parser.add_argument("--sample", type=int, default=None, help="Optional row cap per table")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    metrics = train(sample=args.sample, output=args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
