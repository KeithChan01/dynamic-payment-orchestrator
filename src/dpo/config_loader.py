from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from dpo.paths import config_dir


def _load(name: str, root: Path | None = None) -> dict[str, Any]:
    path = (root or config_dir()) / name
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=8)
def load_all(root: str | None = None) -> dict[str, Any]:
    cfg_root = Path(root) if root else config_dir()
    return {
        "routes": _load("routes.json", cfg_root),
        "fees": _load("fees.json", cfg_root),
        "eligibility": _load("eligibility.json", cfg_root),
        "ranking": _load("ranking.json", cfg_root),
        "priors": _load("priors.json", cfg_root),
    }


def route_ids(cfg: dict[str, Any] | None = None) -> list[str]:
    data = cfg or load_all()
    return [item["id"] for item in data["routes"]["routes"]]


def route_label(route_id: str, cfg: dict[str, Any] | None = None) -> str:
    data = cfg or load_all()
    for item in data["routes"]["routes"]:
        if item["id"] == route_id:
            return item["label"]
    return route_id
