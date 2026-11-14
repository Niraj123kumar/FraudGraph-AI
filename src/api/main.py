from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, validate_api_key_config
from src.auth import single_key_mode_enabled
from src.data.datasets import fail_interrupted_prediction_runs


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if single_key_mode_enabled():
        validate_api_key_config()
    fail_interrupted_prediction_runs()
    yield

app = FastAPI(
    title="FraudGraph AI API",
    description="Machine-learning transaction fraud detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)

app.include_router(router)
