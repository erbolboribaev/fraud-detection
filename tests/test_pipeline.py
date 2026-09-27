"""Tests for the fraud detection pipeline.

Three of these tests exist specifically because manual end-to-end testing
found real bugs; they pin down that behavior so a future change can't
silently reintroduce them:
  - test_secs_since_last_has_sub_second_precision  (dt.total_seconds() bug)
  - test_batch_and_online_features_match           (training-serving skew)
  - test_duplicate_transaction_id_does_not_corrupt_state  (Kafka redelivery)
"""
from datetime import datetime, timedelta, timezone

import fakeredis
import numpy as np
import polars as pl
import pytest

from src.feature_engineering import build_features
from src.schema import CATEGORY_MAP, FEATURES
from src.predictor import FraudPredictor


@pytest.fixture
def redis_client():
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture
def predictor(redis_client):
    return FraudPredictor(redis_client=redis_client)


def make_df(rows):
    """rows: list of (transaction_id, card_id, amount, category, location, timestamp, is_fraud)."""
    return pl.DataFrame(
        rows,
        schema=["transaction_id", "card_id", "amount", "merchant_category",
                "location", "timestamp", "is_fraud"],
        orient="row",
    )


# ---------------------------------------------------------------------------
# feature_engineering.py
# ---------------------------------------------------------------------------

def test_secs_since_last_has_sub_second_precision():
    """Regression test: dt.total_seconds() truncates to whole seconds in this
    Polars version, silently dropping sub-second gaps. secs_since_last must
    keep microsecond precision, or training data disagrees with predictor.py."""
    t0 = datetime(2026, 1, 1, 0, 0, 0)
    t1 = t0 + timedelta(seconds=17, microseconds=562744)
    df = make_df([
        ("t1", "card_a", 10.0, "grocery", "UZ", t0, 0),
        ("t2", "card_a", 20.0, "grocery", "UZ", t1, 0),
    ])
    out = build_features(df)
    gap = out.filter(pl.col("transaction_id") == "t2")["secs_since_last"][0]
    assert gap == pytest.approx(17.562744, abs=1e-4)


def test_amount_ratio_excludes_current_transaction():
    """amount_ratio must compare against the mean of *prior* transactions
    only; including the current one would leak the label into the feature."""
    t0 = datetime(2026, 1, 1)
    df = make_df([
        ("t1", "card_a", 10.0, "grocery", "UZ", t0, 0),
        ("t2", "card_a", 10.0, "grocery", "UZ", t0 + timedelta(minutes=1), 0),
        ("t3", "card_a", 1000.0, "grocery", "UZ", t0 + timedelta(minutes=2), 1),
    ])
    out = build_features(df).sort("transaction_id")
    ratios = out["amount_ratio"].to_list()
    assert ratios[0] == 1.0  # no history: falls back to 1.0
    assert ratios[1] == pytest.approx(1.0)  # 10 / mean([10]) == 1.0
    assert ratios[2] == pytest.approx(100.0)  # 1000 / mean([10, 10]) == 100.0


def test_is_new_location_first_appearance_only():
    t0 = datetime(2026, 1, 1)
    df = make_df([
        ("t1", "card_a", 10.0, "grocery", "UZ", t0, 0),
        ("t2", "card_a", 10.0, "grocery", "UZ", t0 + timedelta(minutes=1), 0),
        ("t3", "card_a", 10.0, "grocery", "DE", t0 + timedelta(minutes=2), 0),
    ])
    out = build_features(df).sort("transaction_id")
    assert out["is_new_location"].to_list() == [1, 0, 1]


# ---------------------------------------------------------------------------
# predictor.py: online (Redis-backed) features must match the batch pipeline
# ---------------------------------------------------------------------------

def test_batch_and_online_features_match(predictor):
    """The single most important invariant in this project: whatever
    feature_engineering.py computes for training must be exactly what
    predictor.py computes online, or the model sees different inputs at
    training time and at serving time."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = [
        ("t1", "card_a", 41.76, "grocery", "UZ", t0),
        ("t2", "card_a", 154.80, "clothing", "UZ", t0 + timedelta(seconds=45)),
        ("t3", "card_a", 3000.0, "crypto", "DE", t0 + timedelta(seconds=50)),
        ("t4", "card_b", 27.35, "restaurant", "TR", t0 + timedelta(seconds=60)),
    ]
    rows = [(tid, card, amt, cat, loc, ts.replace(tzinfo=None), 0)
            for tid, card, amt, cat, loc, ts in events]
    batch = build_features(make_df(rows)).sort("transaction_id")
    batch_arr = batch.select(FEATURES).to_numpy().astype(float)

    online_rows = []
    for tid, card, amt, cat, loc, ts in events:
        feats = predictor._features(card, amt, ts.timestamp(), loc, cat)
        predictor._write_state(card, ts.timestamp(), amt, loc)
        online_rows.append(feats)
    online_arr = np.array(online_rows, dtype=float)

    order = np.argsort([e[0] for e in events])
    assert np.allclose(batch_arr, online_arr[order], atol=1e-6)


def test_new_card_has_expected_defaults(predictor):
    result = predictor.predict("brand_new_card", 45.0, datetime(2026, 1, 1).timestamp(),
                                "UZ", "grocery", transaction_id="tx1")
    f = result["features"]
    assert f["prev_txn_count"] == 0
    assert f["amount_ratio"] == 1.0
    assert f["is_new_location"] == 1
    assert f["secs_since_last"] == pytest.approx(7 * 24 * 3600)


def test_obvious_fraud_pattern_scores_high(predictor):
    """A large, first-ever, cross-border crypto purchase should score high;
    this is a coarse sanity check on the trained model, not a precise bound."""
    predictor.predict("card_x", 40.0, datetime(2026, 1, 1).timestamp(), "UZ", "grocery")
    result = predictor.predict(
        "card_x", 3000.0, datetime(2026, 1, 1, 0, 0, 5).timestamp(), "DE", "crypto"
    )
    assert result["score"] > 0.9


def test_duplicate_transaction_id_does_not_corrupt_state(predictor):
    """Regression test: Kafka's "at least once" delivery means the same
    message can be scored twice (e.g. after a consumer restart before offsets
    were committed). A duplicate must not be folded into the per-card state
    twice, or every later feature for that card is wrong."""
    ts = datetime(2026, 1, 1).timestamp()
    predictor.predict("card_a", 50.0, ts, "UZ", "grocery", transaction_id="dup-1")
    # redelivery of the exact same message
    predictor.predict("card_a", 50.0, ts, "UZ", "grocery", transaction_id="dup-1")

    result = predictor.predict(
        "card_a", 55.0, ts + 10, "UZ", "grocery", transaction_id="dup-2"
    )
    f = result["features"]
    assert f["prev_txn_count"] == 1  # not 2: the duplicate must not have counted
    assert f["amount_ratio"] == pytest.approx(55.0 / 50.0)
    assert f["secs_since_last"] == pytest.approx(10.0)


def test_missing_transaction_id_still_updates_state(predictor):
    """Backward compatibility: transaction_id is optional. Without one, the
    idempotency guard is skipped and state updates unconditionally."""
    ts = datetime(2026, 1, 1).timestamp()
    predictor.predict("card_a", 50.0, ts, "UZ", "grocery")
    result = predictor.predict("card_a", 55.0, ts + 10, "UZ", "grocery")
    assert result["features"]["prev_txn_count"] == 1


# ---------------------------------------------------------------------------
# model artifacts
# ---------------------------------------------------------------------------

def test_model_meta_matches_feature_engineering():
    import json
    from pathlib import Path
    meta = json.loads(Path("models/model_meta.json").read_text())
    assert meta["features"] == FEATURES
    assert meta["category_map"] == CATEGORY_MAP


# ---------------------------------------------------------------------------
# api/main.py
# ---------------------------------------------------------------------------

def test_to_utc_epoch_treats_naive_datetime_as_utc():
    """Regression test: a naive datetime (as parsed from a tz-less ISO string
    like the generator sends) must be treated as UTC, not the server's local
    time zone -- this bug shifted the 'hour' feature by exactly the server's
    UTC offset."""
    from api.main import to_utc_epoch
    naive = datetime(2026, 1, 1, 1, 2, 23)
    ts = to_utc_epoch(naive)
    assert datetime.fromtimestamp(ts, tz=timezone.utc).hour == 1


def test_to_utc_epoch_respects_explicit_timezone():
    from api.main import to_utc_epoch
    aware = datetime(2026, 1, 1, 1, 2, 23, tzinfo=timezone(timedelta(hours=5)))
    ts = to_utc_epoch(aware)
    assert datetime.fromtimestamp(ts, tz=timezone.utc).hour == 20  # one day back, 20:02 UTC
