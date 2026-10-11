"""Best-effort publisher of scored-transaction events to Kafka.

Scoring is the product; the lakehouse feed is a side effect. So publishing
must never slow down or break /predict:

  * It is opt-in: with KAFKA_BOOTSTRAP unset (tests, plain local dev, the
    existing Kubernetes manifests) the publisher is disabled and does nothing.
  * KafkaProducer.send() is asynchronous; send errors are only counted.
  * If the broker is unreachable, a short circuit breaker (cooldown) skips
    publishing instead of letting every request wait on a metadata timeout.

Dropped events are counted (see `stats`, exposed by GET /health) so a lossy
feed is visible instead of silent. Delivery to the lake is therefore
at-most-once from the API's side; the Spark silver job is idempotent per
transaction_id, so the lake never double-counts.
"""
import json
import os
import time

from src.events import SCORED_TOPIC


class ScoredEventPublisher:
    def __init__(self, bootstrap=None, topic=SCORED_TOPIC, producer=None,
                 cooldown_s=30.0, clock=time.monotonic):
        self.bootstrap = bootstrap
        self.topic = topic
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._producer = producer  # injectable for tests
        self._injected = producer is not None
        self._skip_until = 0.0
        self.sent = 0
        self.dropped = 0

    @classmethod
    def from_env(cls):
        return cls(
            bootstrap=os.getenv("KAFKA_BOOTSTRAP") or None,
            topic=os.getenv("SCORED_TOPIC", SCORED_TOPIC),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.bootstrap) or self._injected

    @property
    def stats(self) -> dict:
        return {"enabled": self.enabled, "sent": self.sent, "dropped": self.dropped}

    def _get_producer(self):
        if self._producer is None:
            from kafka import KafkaProducer  # imported lazily: only needed when enabled

            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap,
                value_serializer=lambda v: json.dumps(v).encode(),
                key_serializer=lambda k: k.encode(),
                acks=1,
                linger_ms=20,
                max_block_ms=1000,       # never block a request for long
                request_timeout_ms=2000,
                retries=3,
            )
        return self._producer

    def _on_send_error(self, _exc):
        self.dropped += 1

    def publish(self, event: dict) -> bool:
        """Queue one event. Returns True if it was handed to the producer."""
        if not self.enabled:
            return False
        if self._clock() < self._skip_until:
            self.dropped += 1
            return False
        try:
            future = self._get_producer().send(
                self.topic, key=event["card_id"], value=event
            )
            add_errback = getattr(future, "add_errback", None)
            if add_errback:
                add_errback(self._on_send_error)
            self.sent += 1
            return True
        except Exception:
            self.dropped += 1
            self._skip_until = self._clock() + self.cooldown_s
            if not self._injected:
                self._producer = None  # rebuilt on the next attempt after the cooldown
            return False

    def close(self):
        if self._producer is not None:
            try:
                self._producer.flush(timeout=5)
                self._producer.close(timeout=5)
            except Exception:
                pass
