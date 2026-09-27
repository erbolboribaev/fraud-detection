"""Online feature computation + ONNX inference for a single transaction.

Mirrors src/feature_engineering.py exactly, but computes each feature from a
small amount of per-card state kept in Redis instead of scanning the whole
history. If this file and feature_engineering.py ever disagree, the model
sees different inputs at training time and at serving time (training-serving
skew) and its accuracy silently degrades in production.
"""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import redis

from src.schema import CATEGORY_MAP, FEATURES, NO_HISTORY_GAP

MODEL_PATH = "models/fraud_model.onnx"
META_PATH = "models/model_meta.json"

TEN_MINUTES = 600  # seconds, must match the "10m" window in feature_engineering.py
STATE_TTL = 30 * 24 * 3600  # expire a card's state after 30 days of inactivity


class FraudPredictor:
    def __init__(self, redis_client=None, model_path=MODEL_PATH, meta_path=META_PATH):
        meta = json.loads(Path(meta_path).read_text())
        self.features = meta["features"]
        self.threshold = meta["threshold"]
        self.category_map = meta["category_map"]
        self.no_history_gap = meta["no_history_gap"]
        assert self.features == FEATURES, "model_meta.json is out of sync with feature_engineering.py"

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.prob_name = self.session.get_outputs()[1].name  # output 0 is the label, output 1 the probabilities

        self.redis = redis_client if redis_client is not None else redis.Redis(
            # REDIS_HOST lets this run unchanged locally (defaults to
            # localhost) and inside Docker Compose (set to the "redis"
            # service name there).
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            decode_responses=True,
        )

    @staticmethod
    def _key(card_id, suffix):
        return f"card:{card_id}:{suffix}"

    def _read_state(self, card_id, ts, location):
        """One round trip: read everything needed to compute features for `ts`,
        using only data from *before* this transaction (no leakage)."""
        r = self.redis
        pipe = r.pipeline()
        pipe.get(self._key(card_id, "count"))
        pipe.get(self._key(card_id, "sum"))
        pipe.get(self._key(card_id, "last_ts"))
        pipe.sismember(self._key(card_id, "locations"), location)
        # "(" excludes the right edge, matching rolling_sum_by(..., closed="left")
        # in feature_engineering.py: only transactions strictly before `ts` count.
        pipe.zcount(self._key(card_id, "recent"), ts - TEN_MINUTES, f"({ts}")
        prev_count, prev_sum, last_ts, seen_location, txn_count_10m = pipe.execute()
        return {
            "prev_count": int(prev_count) if prev_count is not None else 0,
            "prev_sum": float(prev_sum) if prev_sum is not None else 0.0,
            "last_ts": float(last_ts) if last_ts is not None else None,
            "seen_location": bool(seen_location),
            "txn_count_10m": int(txn_count_10m),
        }

    def _write_state(self, card_id, ts, amount, location):
        """Fold this transaction into the card's state, for future requests."""
        r = self.redis
        recent_key = self._key(card_id, "recent")
        pipe = r.pipeline()
        pipe.incr(self._key(card_id, "count"))
        pipe.incrbyfloat(self._key(card_id, "sum"), amount)
        pipe.set(self._key(card_id, "last_ts"), ts)
        pipe.sadd(self._key(card_id, "locations"), location)
        # A unique member per event; ts alone could collide if two transactions
        # for the same card land on the exact same timestamp.
        pipe.zadd(recent_key, {f"{ts}:{amount}:{id(object())}": ts})
        pipe.zremrangebyscore(recent_key, 0, ts - TEN_MINUTES)
        for key in (self._key(card_id, "count"), self._key(card_id, "sum"),
                    self._key(card_id, "last_ts"), self._key(card_id, "locations"), recent_key):
            pipe.expire(key, STATE_TTL)
        pipe.execute()

    def _features(self, card_id, amount, ts, location, category):
        s = self._read_state(card_id, ts, location)
        amount_ratio = (
            amount / (s["prev_sum"] / s["prev_count"]) if s["prev_count"] > 0 else 1.0
        )
        secs_since_last = (
            ts - s["last_ts"] if s["last_ts"] is not None else self.no_history_gap
        )
        values = {
            "log_amount": math.log1p(amount),
            "hour": time.gmtime(ts).tm_hour,
            "amount_ratio": amount_ratio,
            "prev_txn_count": s["prev_count"],
            "txn_count_10m": s["txn_count_10m"],
            "secs_since_last": secs_since_last,
            "is_new_location": 0 if s["seen_location"] else 1,
            "category_code": self.category_map.get(category, -1),
        }
        return [values[f] for f in self.features]

    def predict(self, card_id, amount, ts, location, merchant_category, transaction_id=None):
        """ts: unix timestamp (seconds) of the transaction.

        transaction_id is optional but strongly recommended: Kafka gives
        "at least once" delivery, so the same transaction can be redelivered
        (e.g. after a consumer restart before offsets were committed). The
        score itself is a pure function of state and is harmless to
        recompute, but folding the transaction into the per-card state twice
        would corrupt it (it would look like its own preceding transaction).
        This guard makes the state update idempotent per transaction_id.
        """
        feats = self._features(card_id, amount, ts, location, merchant_category)
        if transaction_id is not None:
            is_new = self.redis.set(f"seen:{transaction_id}", "1", nx=True, ex=STATE_TTL)
            if is_new:
                self._write_state(card_id, ts, amount, location)
        else:
            self._write_state(card_id, ts, amount, location)

        x = np.array([feats], dtype=np.float32)
        score = float(self.session.run([self.prob_name], {self.input_name: x})[0][0][1])
        return {
            "score": score,
            "is_fraud": score >= self.threshold,
            "features": dict(zip(self.features, feats)),
        }
