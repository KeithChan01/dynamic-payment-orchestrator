from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dpo.api import app, get_engine
from dpo.orchestrator import Orchestrator
from dpo.schemas import RouteRequest


@pytest.fixture
def engine() -> Orchestrator:
    return Orchestrator()


@pytest.fixture
def client() -> TestClient:
    get_engine.cache_clear()
    return TestClient(app)


@pytest.fixture
def sample_request() -> RouteRequest:
    return RouteRequest(
        transaction_type="deposit",
        amount=480,
        currency="USD",
        country="United States",
        policy="balanced",
    )
