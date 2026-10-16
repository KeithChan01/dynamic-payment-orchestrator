"""Load the three PayMind tables through the Hugging Face `datasets` library."""

from __future__ import annotations

import argparse

import pandas as pd
from datasets import load_dataset

HF_REPO = "navk8690/paymind-reference-data"
FILES = {
    "payment_method": "payment_method copy.csv",
    "success": "success copy.csv",
    "arrival": "arrival copy.csv",
}

PAYMENT_METHOD_COLUMNS = [
    "event_id",
    "timestamp_utc",
    "transaction_type",
    "country",
    "ip_country",
    "jurisdiction",
    "currency",
    "local_currency",
    "amount",
    "app_type",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_cross_border",
    "payment_type",
]
SUCCESS_COLUMNS = [
    "timestamp_utc",
    "transaction_type",
    "currency",
    "amount",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_cross_border",
    "payment_code",
    "payment_type",
    "success",
]
ARRIVAL_COLUMNS = [
    "timestamp_utc",
    "transaction_type",
    "currency",
    "amount",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_cross_border",
    "payment_code",
    "payment_type",
    "banking_hours_indicator",
    "arrival_duration_minutes",
]
EXPECTED_COLUMNS = {
    "payment_method": PAYMENT_METHOD_COLUMNS,
    "success": SUCCESS_COLUMNS,
    "arrival": ARRIVAL_COLUMNS,
}


class SchemaError(ValueError):
    pass


def load_table(name: str) -> pd.DataFrame:
    """Load one of payment_method / success / arrival from the Hub.

    The Hub repo cannot be loaded as a single dataset: the three CSVs have
    different schemas. Each file is requested explicitly.
    """
    if name not in FILES:
        raise KeyError(f"Unknown table '{name}'. Expected one of {list(FILES)}")

    filename = FILES[name]
    try:
        dataset = load_dataset(
            "csv",
            data_files=f"hf://datasets/{HF_REPO}/{filename}",
            split="train",
        )
    except Exception:
        dataset = load_dataset(HF_REPO, data_files=filename, split="train")

    frame = dataset.to_pandas()
    validate_columns(name, frame)
    return _normalize(frame)


def load_all_tables() -> dict[str, pd.DataFrame]:
    return {name: load_table(name) for name in FILES}


def validate_columns(name: str, frame: pd.DataFrame) -> None:
    expected = EXPECTED_COLUMNS[name]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}. Got {list(frame.columns)}")


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    if out["timestamp_utc"].isna().any():
        raise SchemaError("timestamp_utc contains unparseable values")
    if "amount" in out.columns:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    if "success" in out.columns:
        out["success"] = pd.to_numeric(out["success"], errors="coerce").astype("int64")
    if "arrival_duration_minutes" in out.columns:
        out["arrival_duration_minutes"] = pd.to_numeric(
            out["arrival_duration_minutes"], errors="coerce"
        )
    for flag in ("is_weekend", "is_cross_border", "hour"):
        if flag in out.columns:
            out[flag] = pd.to_numeric(out[flag], errors="coerce").astype("int64")
    return out.sort_values("timestamp_utc").reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("timestamp_utc").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()
    if len(train) and len(val) and train["timestamp_utc"].max() > val["timestamp_utc"].min():
        raise SchemaError("train/val timestamps overlap")
    if len(val) and len(test) and val["timestamp_utc"].max() > test["timestamp_utc"].min():
        raise SchemaError("val/test timestamps overlap")
    return train, val, test


def inspect_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect PayMind reference tables via Hugging Face")
    parser.add_argument("--table", choices=list(FILES), default=None)
    args = parser.parse_args(argv)
    names = [args.table] if args.table else list(FILES)
    for name in names:
        frame = load_table(name)
        print(f"== {name} ==")
        print(f"rows={len(frame):,} cols={list(frame.columns)}")
        print(f"timestamp {frame['timestamp_utc'].min()} → {frame['timestamp_utc'].max()}")
        if "payment_type" in frame.columns:
            print(frame["payment_type"].value_counts().to_string())
        if "success" in frame.columns:
            print("success rate", float(frame["success"].mean()))
        if "arrival_duration_minutes" in frame.columns:
            print(
                "arrival p50/p90",
                float(frame["arrival_duration_minutes"].quantile(0.5)),
                float(frame["arrival_duration_minutes"].quantile(0.9)),
            )
        print()


if __name__ == "__main__":
    inspect_main()
