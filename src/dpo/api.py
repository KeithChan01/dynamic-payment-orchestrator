from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dpo import __version__
from dpo.config_loader import load_all, route_ids
from dpo.orchestrator import Orchestrator
from dpo.schemas import CompareResponse, HealthResponse, RouteRequest, RouteResponse
from dpo.simulate import random_requests, summarize

app = FastAPI(
    title="Dynamic Payment Orchestrator",
    version=__version__,
    description="Ranks payment routes using synthetic PayMind reference data. Not a payment gateway.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_engine() -> Orchestrator:
    return Orchestrator()


class SimulateBody(BaseModel):
    n: int = Field(default=200, ge=10, le=2000)
    policy: str = "balanced"
    seed: int = 7


@app.get("/")
def root() -> dict:
    return {"service": "dynamic-payment-orchestrator", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    engine = get_engine()
    return HealthResponse(status="ok", model_mode=engine.mode, routes=route_ids())  # type: ignore[arg-type]


@app.get("/routes")
def routes() -> dict:
    return load_all()["routes"]


@app.get("/policies")
def policies() -> dict:
    return load_all()["ranking"]["policies"]


@app.post("/v1/route", response_model=RouteResponse)
def route(request: RouteRequest) -> RouteResponse:
    return get_engine().evaluate(request)


@app.post("/v1/compare", response_model=CompareResponse)
def compare(request: RouteRequest) -> CompareResponse:
    return CompareResponse(policies=get_engine().compare(request))


@app.post("/v1/simulate")
def simulate(body: SimulateBody) -> dict:
    cfg = load_all()
    if body.policy not in cfg["ranking"]["policies"]:
        raise HTTPException(status_code=400, detail=f"Unknown policy '{body.policy}'")
    requests = random_requests(n=body.n, seed=body.seed)
    return summarize(get_engine(), requests, policy=body.policy)


def main() -> None:
    import uvicorn

    uvicorn.run("dpo.api:app", host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    main()
