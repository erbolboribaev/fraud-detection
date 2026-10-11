"""Tests for the scored-event contract (src/events.py) and the best-effort
Kafka publisher (src/publisher.py). Neither needs Kafka or Spark."""
import json
from datetime import datetime, timezone

import pytest

from src.events import EVENT_FIELDS, SCORED_TOPIC, build_scored_event
from src.publisher import ScoredEventPublisher
from src.schema import FEATURES


def make_event(**overrides):
    args = dict(
        transaction_id="t1",
        card_id="card_00001",
        amount=125.5,
        merchant_category="electronics",
        location="TR",
        event_ts=datetime(2026, 1, 1, 20, 0, tzinfo=timezone.utc).timestamp(),
        features={name: 1.0 for name in FEATURES},
        score=0.91,
        is_fraud=True,
        threshold=0.5,
        scored_at=datetime(2026, 1, 1, 20, 0, 1, tzinfo=timezone.utc),
    )
    args.update(overrides)
    return build_scored_event(**args)


# ---------------------------------------------------------------------------
# src/events.py
# ---------------------------------------------------------------------------

def test_event_has_exactly_the_declared_fields_in_order():
    assert list(make_event()) == list(EVENT_FIELDS)


def test_event_time_is_utc_regardless_of_server_timezone(monkeypatch):
    """Same class of bug as the naive-datetime one in api/main.py: the lake
    must always hold UTC, whatever TZ the API container runs in."""
    monkeypatch.setenv("TZ", "Asia/Tashkent")
    ev = make_event()
    assert ev["event_time"] == "2026-01-01T20:00:00+00:00"
    assert ev["scored_at"] == "2026-01-01T20:00:01+00:00"


def test_event_is_json_serializable_with_plain_types():
    ev = json.loads(json.dumps(make_event()))
    assert ev["is_fraud"] is True
    assert isinstance(ev["amount"], float)


def test_event_includes_every_model_feature():
    ev = make_event()
    for name in FEATURES:
        assert name in ev


# ---------------------------------------------------------------------------
# src/publisher.py
# ---------------------------------------------------------------------------

class FakeFuture:
    def __init__(self):
        self.errbacks = []

    def add_errback(self, fn):
        self.errbacks.append(fn)


class FakeProducer:
    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []

    def send(self, topic, key=None, value=None):
        if self.fail:
            raise RuntimeError("broker unavailable")
        future = FakeFuture()
        self.sent.append((topic, key, value, future))
        return future

    def flush(self, timeout=None):
        pass

    def close(self, timeout=None):
        pass


def test_publisher_is_disabled_without_bootstrap(monkeypatch):
    monkeypatch.delenv("KAFKA_BOOTSTRAP", raising=False)
    pub = ScoredEventPublisher.from_env()
    assert not pub.enabled
    assert pub.publish(make_event()) is False
    assert pub.stats == {"enabled": False, "sent": 0, "dropped": 0}


def test_publisher_sends_to_topic_keyed_by_card():
    producer = FakeProducer()
    pub = ScoredEventPublisher(producer=producer)
    assert pub.publish(make_event()) is True
    topic, key, value, _ = producer.sent[0]
    assert topic == SCORED_TOPIC
    assert key == "card_00001"
    assert value["transaction_id"] == "t1"
    assert pub.stats["sent"] == 1


def test_async_send_errors_are_counted():
    producer = FakeProducer()
    pub = ScoredEventPublisher(producer=producer)
    pub.publish(make_event())
    producer.sent[0][3].errbacks[0](RuntimeError("delivery failed"))
    assert pub.stats["dropped"] == 1


def test_publish_failure_never_raises_and_opens_circuit_breaker():
    now = [100.0]
    producer = FakeProducer(fail=True)
    pub = ScoredEventPublisher(producer=producer, cooldown_s=30.0, clock=lambda: now[0])

    assert pub.publish(make_event()) is False      # fails, breaker opens
    assert pub.publish(make_event()) is False      # skipped during cooldown
    assert pub.stats["dropped"] == 2
    assert pub.stats["sent"] == 0

    now[0] += 31.0                                  # cooldown over
    producer.fail = False
    assert pub.publish(make_event()) is True
    assert pub.stats["sent"] == 1
