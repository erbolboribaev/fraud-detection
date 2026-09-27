"""FastAPI serving layer for the fraud model.

POST /predict scores one transaction using src.predictor.FraudPredictor,
which reads/updates per-card state in Redis and runs the ONNX model.
"""
import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predictor import FraudPredictor

predictor: FraudPredictor | None = None

LOG_PATH = Path("logs/predictions.jsonl")
_log_lock = threading.Lock()


def log_prediction(features: dict, score: float, is_fraud: bool) -> None:
    """Append one scored transaction's features to a local log, so
    src/drift_report.py has real production traffic to compare against the
    training distribution. Best-effort: a logging failure must never break
    a prediction response.
    """
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
                 **features, "score": score, "is_fraud": is_fraud}
        with _log_lock, open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    # Loaded once at startup: the ONNX session and the Redis connection pool
    # are reused across requests instead of being recreated every call.
    predictor = FraudPredictor()
    yield
    predictor = None


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


class Transaction(BaseModel):
    transaction_id: str
    card_id: str
    amount: float = Field(gt=0)
    merchant_category: str
    location: str
    timestamp: datetime | None = None  # defaults to "now" if omitted


class PredictResponse(BaseModel):
    transaction_id: str
    score: float
    is_fraud: bool
    threshold: float
    features: dict


@app.get("/health")
def health():
    try:
        predictor.redis.ping()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"redis unavailable: {e}")


def to_utc_epoch(dt: datetime | None) -> float:
    """Convert a possibly-naive datetime to a UTC unix timestamp.

    Naive datetimes (no tzinfo) are assumed to already represent UTC, which
    is what the generator/Kafka messages send. Without the explicit
    `replace(tzinfo=utc)`, Python's .timestamp() instead interprets a naive
    datetime as the server's local time zone -- silently shifting every
    feature that depends on wall-clock time depending on where this service
    happens to be deployed.
    """
    dt = dt or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


@app.post("/predict", response_model=PredictResponse)
def predict(txn: Transaction):
    ts = to_utc_epoch(txn.timestamp)

    result = predictor.predict(
        card_id=txn.card_id,
        amount=txn.amount,
        ts=ts,
        location=txn.location,
        merchant_category=txn.merchant_category,
        transaction_id=txn.transaction_id,
    )
    log_prediction(result["features"], result["score"], result["is_fraud"])
    return PredictResponse(
        transaction_id=txn.transaction_id,
        score=result["score"],
        is_fraud=result["is_fraud"],
        threshold=predictor.threshold,
        features=result["features"],
    )
