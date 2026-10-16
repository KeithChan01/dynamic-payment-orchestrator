from __future__ import annotations

import pandas as pd
import pytest

from dpo.data import SchemaError, chronological_split, validate_columns


def test_chronological_split_does_not_leak_future_rows():
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                [
                    "2022-01-01",
                    "2022-06-01",
                    "2023-01-01",
                    "2023-06-01",
                    "2024-01-01",
                    "2024-06-01",
                    "2025-01-01",
                    "2025-06-01",
                    "2026-01-01",
                    "2026-06-01",
                ],
                utc=True,
            ),
            "amount": range(10),
        }
    )
    train, val, test = chronological_split(frame, train_frac=0.6, val_frac=0.2)
    assert train["timestamp_utc"].max() <= val["timestamp_utc"].min()
    assert val["timestamp_utc"].max() <= test["timestamp_utc"].min()
    assert len(train) + len(val) + len(test) == 10


def test_validate_columns_rejects_missing():
    with pytest.raises(SchemaError):
        validate_columns("success", pd.DataFrame({"timestamp_utc": []}))
