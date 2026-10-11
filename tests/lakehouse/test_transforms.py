"""Tests for lakehouse/transforms.py.

These run on a local Spark session and need Java + PySpark
(requirements-lakehouse.txt). Without them the whole module is skipped, so the
main CI job (which doesn't install Spark) stays green; the dedicated
`lakehouse` CI job runs them for real.

The most important one is test_event_built_by_the_api_passes_validation: it
feeds an event produced by src.events.build_scored_event (what the API
publishes) through the lake's parser, so a change on either side of the
contract fails a test instead of silently filling quarantine.
"""
import json
import shutil
from datetime import date, datetime, timezone

import pytest

pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Java is required to run Spark", allow_module_level=True)

from pyspark.sql import functions as F  # noqa: E402

from lakehouse.schema import SCORED_EVENT_SCHEMA  # noqa: E402
from lakehouse.transforms import (  # noqa: E402
    dedupe_batch,
    gold_fraud_hourly,
    gold_pipeline_health,
    parse_bronze,
    split_valid,
)
from src.events import EVENT_FIELDS, build_scored_event  # noqa: E402
from src.schema import FEATURES  # noqa: E402


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc).timestamp()


def make_event(**overrides) -> dict:
    args = dict(
        transaction_id="t1",
        card_id="card_00001",
        amount=100.0,
        merchant_category="electronics",
        location="TR",
        event_ts=utc(2026, 1, 1, 20, 0),
        features={name: 1.0 for name in FEATURES},
        score=0.9,
        is_fraud=True,
        threshold=0.5,
        scored_at=datetime(2026, 1, 1, 20, 0, 1, tzinfo=timezone.utc),
    )
    args.update(overrides)
    return build_scored_event(**args)


def bronze_df(spark, rows):
    """rows: (raw_value, kafka_partition, kafka_offset, ingested_at 'YYYY-MM-DD HH:MM:SS')."""
    df = spark.createDataFrame(
        rows,
        "raw_value string, kafka_partition int, kafka_offset long, ingested_at string",
    )
    return df.withColumn("ingested_at", F.to_timestamp("ingested_at"))


def silver_from_events(spark, events, ingested_at="2026-01-01 20:05:00"):
    rows = [(json.dumps(e), 0, i, ingested_at) for i, e in enumerate(events)]
    valid, invalid = split_valid(parse_bronze(bronze_df(spark, rows)))
    assert invalid.count() == 0
    return valid


# ---------------------------------------------------------------------------
# schema contract
# ---------------------------------------------------------------------------

def test_spark_schema_matches_event_contract():
    assert [f.name for f in SCORED_EVENT_SCHEMA.fields] == list(EVENT_FIELDS)


# ---------------------------------------------------------------------------
# parse + validate
# ---------------------------------------------------------------------------

def test_event_built_by_the_api_passes_validation(spark):
    rows = [(json.dumps(make_event()), 0, 0, "2026-01-01 20:05:00")]
    valid, invalid = split_valid(parse_bronze(bronze_df(spark, rows)))

    assert invalid.count() == 0
    row = valid.collect()[0]
    assert row["transaction_id"] == "t1"
    assert row["amount"] == 100.0
    assert row["is_fraud"] is True
    assert row["event_date"] == date(2026, 1, 1)
    assert valid.schema["event_time"].dataType.simpleString() == "timestamp"
    # Compare as a formatted string: collect() would convert timestamps to the
    # *Python process's* local time zone, which makes the test machine-dependent.
    event_time = valid.select(F.date_format("event_time", "yyyy-MM-dd HH:mm:ss")).first()[0]
    assert event_time == "2026-01-01 20:00:00"


@pytest.mark.parametrize(
    "raw_value, expected_reason",
    [
        ("{this is not json", "unparseable_json"),
        (json.dumps({**make_event(), "amount": "not-a-number"}), "unparseable_json"),
        (json.dumps({**make_event(), "transaction_id": None}), "missing_key"),
        (json.dumps({**make_event(), "event_time": "yesterday-ish"}), "invalid_event_time"),
        (json.dumps({**make_event(), "amount": -5.0}), "invalid_amount"),
        (json.dumps({**make_event(), "score": 1.7}), "invalid_score"),
    ],
)
def test_bad_records_are_quarantined_with_a_reason(spark, raw_value, expected_reason):
    valid, invalid = split_valid(parse_bronze(bronze_df(spark, [(raw_value, 0, 0, "2026-01-01 20:05:00")])))

    assert valid.count() == 0
    rows = invalid.collect()
    assert len(rows) == 1
    assert rows[0]["reject_reason"] == expected_reason
    assert rows[0]["raw_value"] == raw_value  # original payload kept for debugging/replay


def test_one_bad_record_does_not_affect_good_ones(spark):
    rows = [
        (json.dumps(make_event(transaction_id="good")), 0, 0, "2026-01-01 20:05:00"),
        ("{broken", 0, 1, "2026-01-01 20:05:00"),
    ]
    valid, invalid = split_valid(parse_bronze(bronze_df(spark, rows)))
    assert [r["transaction_id"] for r in valid.collect()] == ["good"]
    assert invalid.count() == 1


# ---------------------------------------------------------------------------
# deduplication
# ---------------------------------------------------------------------------

def test_dedupe_batch_keeps_earliest_ingested_copy(spark):
    ev = json.dumps(make_event(transaction_id="dup"))
    other = json.dumps(make_event(transaction_id="other"))
    rows = [
        (ev, 0, 5, "2026-01-01 20:05:30"),   # redelivered copy, ingested later
        (ev, 0, 2, "2026-01-01 20:05:00"),   # first copy
        (other, 0, 3, "2026-01-01 20:05:10"),
    ]
    valid, _ = split_valid(parse_bronze(bronze_df(spark, rows)))
    out = {r["transaction_id"]: r for r in dedupe_batch(valid).collect()}

    assert set(out) == {"dup", "other"}
    assert out["dup"]["kafka_offset"] == 2


# ---------------------------------------------------------------------------
# gold
# ---------------------------------------------------------------------------

def test_gold_fraud_hourly_aggregates_per_hour_category_and_country(spark):
    events = [
        make_event(transaction_id="a", amount=100.0, score=0.9, is_fraud=True,
                   event_ts=utc(2026, 1, 1, 10, 5)),
        make_event(transaction_id="b", amount=50.0, score=0.1, is_fraud=False,
                   event_ts=utc(2026, 1, 1, 10, 40)),
        make_event(transaction_id="c", amount=20.0, score=0.2, is_fraud=False,
                   event_ts=utc(2026, 1, 1, 11, 1)),
        make_event(transaction_id="d", amount=30.0, score=0.2, is_fraud=False,
                   merchant_category="grocery", event_ts=utc(2026, 1, 1, 10, 7)),
    ]
    gold = gold_fraud_hourly(silver_from_events(spark, events)).withColumn(
        "event_hour", F.date_format("event_hour", "yyyy-MM-dd HH:mm")
    )
    rows = {
        (r["event_hour"], r["merchant_category"], r["location"]): r for r in gold.collect()
    }

    assert len(rows) == 3
    r = rows[("2026-01-01 10:00", "electronics", "TR")]
    assert r["txn_count"] == 2
    assert r["flagged_count"] == 1
    assert r["flagged_rate"] == pytest.approx(0.5)
    assert r["total_amount"] == pytest.approx(150.0)
    assert r["flagged_amount"] == pytest.approx(100.0)
    assert r["avg_score"] == pytest.approx(0.5)
    assert rows[("2026-01-01 11:00", "electronics", "TR")]["flagged_count"] == 0


def test_pipeline_health_reports_reject_rate_per_ingestion_hour(spark):
    silver = silver_from_events(
        spark,
        [make_event(transaction_id="a"), make_event(transaction_id="b")],
        ingested_at="2026-01-01 20:05:00",
    )
    _, quarantine = split_valid(
        parse_bronze(bronze_df(spark, [("{broken", 0, 9, "2026-01-01 20:30:00")]))
    )

    health = (
        gold_pipeline_health(silver, quarantine)
        .withColumn("ingest_hour", F.date_format("ingest_hour", "yyyy-MM-dd HH:mm"))
        .collect()
    )

    assert len(health) == 1
    assert health[0]["ingest_hour"] == "2026-01-01 20:00"
    assert health[0]["valid_count"] == 2
    assert health[0]["rejected_count"] == 1
    assert health[0]["reject_rate"] == pytest.approx(0.3333, abs=1e-4)
