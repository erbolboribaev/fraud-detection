"""FastAPI serving layer for the fraud model.

POST /predict scores one transaction using src.predictor.FraudPredictor,
which reads/updates per-card state in Redis and runs the ONNX model.

The GET endpoints below (/stats, /transactions/recent, /alerts/recent,
/explain/{id}, /drift/summary) exist to power the dashboard: they read the
same local logs everything else already writes, rather than adding a
separate datastore.
"""
import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.alerts import send_alert
from src.events import build_scored_event
from src.predictor import FraudPredictor
from src.publisher import ScoredEventPublisher
from src.schema import FEATURES

predictor: FraudPredictor | None = None
explainer: "shap.TreeExplainer | None" = None
# Feed for the Spark lakehouse (lakehouse/). Disabled unless KAFKA_BOOTSTRAP is set.
publisher: ScoredEventPublisher | None = None

LOG_PATH = Path("logs/predictions.jsonl")
ALERTS_PATH = Path("logs/alerts.jsonl")
FEEDBACK_PATH = Path("logs/feedback.jsonl")
DRIFT_SUMMARY_PATH = Path("reports/drift_summary.json")
BOOSTER_PATH = "models/fraud_model.txt"
_log_lock = threading.Lock()
_feedback_lock = threading.Lock()


def log_prediction(txn: "Transaction", features: dict, score: float, is_fraud: bool) -> None:
    """Append one scored transaction (its raw fields and computed features)
    to a local log. This is the single source the dashboard, drift_report.py,
    and apply_feedback.py all read from. Best-effort: a logging failure must
    never break a prediction response.
    """
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transaction_id": txn.transaction_id,
            "card_id": txn.card_id,
            "amount": txn.amount,
            "merchant_category": txn.merchant_category,
            "location": txn.location,
            **features,
            "score": score,
            "is_fraud": is_fraud,
        }
        with _log_lock, open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def read_jsonl_tail(path: Path, limit: int) -> list[dict]:
    """Return up to the last `limit` entries of a JSONL log, most recent
    first. Simple full-file read: fine at this project's data volume; a
    higher-traffic system would use a capped ring buffer or a real datastore
    instead of re-reading a growing file on every request.
    """
    if not path.exists():
        return []
    with open(path) as f:
        lines = [line for line in f if line.strip()]
    tail = lines[-limit:]
    return [json.loads(line) for line in reversed(tail)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, explainer, publisher
    # Loaded once at startup: the ONNX session, Redis connection pool, and
    # SHAP explainer are reused across requests instead of being recreated
    # every call.
    predictor = FraudPredictor()
    explainer = shap.TreeExplainer(lgb.Booster(model_file=BOOSTER_PATH))
    publisher = ScoredEventPublisher.from_env()
    yield
    publisher.close()
    predictor = None
    explainer = None
    publisher = None


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

# The dashboard (Vite dev server) runs on a different origin than this API,
# so the browser needs an explicit CORS allow-list. Restricted to local dev
# ports rather than "*", since this API also accepts feedback/writes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves reports/drift_report.html (and anything else dropped in reports/) so
# the dashboard's "Open full report" link has something to point at.
Path("reports").mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")


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
        return {"status": "ok", "scored_events": publisher.stats}
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


def publish_scored_event(txn: "Transaction", ts: float, result: dict) -> None:
    """Hand the scored transaction to the lakehouse feed. Best-effort and
    non-blocking: a Kafka problem must never fail or slow a prediction."""
    if publisher is None or not publisher.enabled:
        return
    publisher.publish(build_scored_event(
        transaction_id=txn.transaction_id,
        card_id=txn.card_id,
        amount=txn.amount,
        merchant_category=txn.merchant_category,
        location=txn.location,
        event_ts=ts,
        features=result["features"],
        score=result["score"],
        is_fraud=result["is_fraud"],
        threshold=predictor.threshold,
    ))


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
    log_prediction(txn, result["features"], result["score"], result["is_fraud"])
    publish_scored_event(txn, ts, result)
    if result["is_fraud"]:
        send_alert(txn.transaction_id, txn.card_id, txn.amount,
                    txn.merchant_category, txn.location, result["score"])
    return PredictResponse(
        transaction_id=txn.transaction_id,
        score=result["score"],
        is_fraud=result["is_fraud"],
        threshold=predictor.threshold,
        features=result["features"],
    )


class FeedbackRequest(BaseModel):
    transaction_id: str
    actual_is_fraud: bool


@app.post("/feedback")
def feedback(fb: FeedbackRequest):
    """Record the true label for a transaction once it's known (e.g. a
    chargeback confirmed fraud, or the cardholder confirmed a purchase).
    src/apply_feedback.py later joins this with the logged features to build
    a corrected dataset for retraining -- this is what "closes the loop"
    between the model's predictions and reality.
    """
    try:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "transaction_id": fb.transaction_id,
            "actual_is_fraud": fb.actual_is_fraud,
            "reported_at": datetime.now(timezone.utc).isoformat(),
        }
        with _feedback_lock, open(FEEDBACK_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"could not record feedback: {e}")
    return {"status": "recorded"}


# ---------------------------------------------------------------------------
# Read endpoints for the dashboard
# ---------------------------------------------------------------------------

@app.get("/stats")
def stats():
    today = datetime.now(timezone.utc).date().isoformat()
    total = 0
    fraud = 0
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("timestamp", "").startswith(today):
                    total += 1
                    if entry.get("is_fraud"):
                        fraud += 1
    return {
        "total_today": total,
        "fraud_today": fraud,
        "fraud_rate": (fraud / total) if total else 0.0,
    }


@app.get("/transactions/recent")
def recent_transactions(limit: int = 40):
    return read_jsonl_tail(LOG_PATH, limit)


@app.get("/alerts/recent")
def recent_alerts(limit: int = 8):
    return read_jsonl_tail(ALERTS_PATH, limit)


@app.get("/explain/{transaction_id}")
def explain(transaction_id: str):
    entry = None
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                candidate = json.loads(line)
                if candidate.get("transaction_id") == transaction_id:
                    entry = candidate  # keep scanning; want the LATEST match
    if entry is None:
        raise HTTPException(status_code=404, detail=f"{transaction_id} not found in logs")

    x = np.array([[entry[f] for f in FEATURES]], dtype=np.float64)
    shap_values = explainer.shap_values(x)[0]
    base_value = float(explainer.expected_value)
    total_log_odds = base_value + float(shap_values.sum())
    probability = 1 / (1 + np.exp(-total_log_odds))

    contributions = sorted(
        (
            {"feature": f, "value": float(v), "raw_value": entry[f]}
            for f, v in zip(FEATURES, shap_values)
        ),
        key=lambda c: -abs(c["value"]),
    )
    return {
        "transaction_id": transaction_id,
        "card_id": entry.get("card_id"),
        "amount": entry.get("amount"),
        "merchant_category": entry.get("merchant_category"),
        "score": entry.get("score"),
        "base_value": base_value,
        "contributions": contributions,
        "probability": float(probability),
    }


@app.get("/drift/summary")
def drift_summary():
    if not DRIFT_SUMMARY_PATH.exists():
        return {"available": False}
    return {"available": True, **json.loads(DRIFT_SUMMARY_PATH.read_text())}
