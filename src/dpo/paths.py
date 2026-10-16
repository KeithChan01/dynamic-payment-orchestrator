from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    override = os.getenv("DPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return repo_root() / "config"


def models_dir() -> Path:
    return repo_root() / "models"
