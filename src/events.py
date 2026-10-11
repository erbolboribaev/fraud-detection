"""The scored-transaction event: the contract between the serving API and the
lakehouse.

Every transaction the API scores is published as one flat JSON event to the
`scored-transactions` Kafka topic. The Spark jobs in `lakehouse/` read that
topic, so this module is the single place that defines the event's shape.
`lakehouse/schema.py` builds its Spark schema from EVENT_FIELDS, and a test
(tests/lakehouse/test_transforms.py) fails if the two ever drift apart.

Stdlib only (plus src.schema), so the API's slim Docker image can import it.
"""
from datetime import datetime, timezone

from src.schema import FEATURES

SCORED_TOPIC = "scored-transactions"
EVENT_VERSION = 1

# field name -> logical type ("string" | "double" | "int" | "boolean").
# Order matters: it is the column order of the lakehouse tables.
EVENT_FIELDS: dict[str, str] = {
    "event_version": "int",
    "transaction_id": "string",
    "card_id": "string",
    "amount": "double",
    "merchant_category": "string",
    "location": "string",
    "event_time": "string",   # ISO-8601 UTC: when the transaction happened
    "scored_at": "string",    # ISO-8601 UTC: when the API scored it
    "score": "double",
    "is_fraud": "boolean",    # the MODEL's flag (score >= threshold), not ground truth
    "threshold": "double",
    **{name: "double" for name in FEATURES},
}


def _iso_utc(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def build_scored_event(
    *,
    transaction_id: str,
    card_id: str,
    amount: float,
    merchant_category: str,
    location: str,
    event_ts: float,
    features: dict,
    score: float,
    is_fraud: bool,
    threshold: float,
    scored_at: datetime | None = None,
) -> dict:
    """Build one event. `event_ts` is a UTC unix timestamp (what
    api.main.to_utc_epoch returns), so naive timestamps can never leak into
    the lake as local time."""
    scored_at = scored_at or datetime.now(timezone.utc)
    event = {
        "event_version": EVENT_VERSION,
        "transaction_id": transaction_id,
        "card_id": card_id,
        "amount": float(amount),
        "merchant_category": merchant_category,
        "location": location,
        "event_time": _iso_utc(event_ts),
        "scored_at": scored_at.astimezone(timezone.utc).isoformat(),
        "score": float(score),
        "is_fraud": bool(is_fraud),
        "threshold": float(threshold),
    }
    for name in FEATURES:
        event[name] = float(features[name])
    return event
