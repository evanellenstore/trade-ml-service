from __future__ import annotations

from fastapi import FastAPI

from app.api.dataset_api import router as dataset_router
from app.api.health_api import router as health_router
from app.config.logging_config import configure_logging

configure_logging()

app = FastAPI(
    title="Trade ML Service",
    version="0.1.0",
    description="Machine learning dataset generation and feature engineering service for trading data.",
)

app.include_router(health_router)
app.include_router(dataset_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "UP", "service": "trade-ml-service"}
