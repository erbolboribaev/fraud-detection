"""Pure DataFrame transformations for the lakehouse layers.

Everything here maps DataFrame -> DataFrame and knows nothing about Kafka,
Delta or S3. That is deliberate: the streaming plumbing lives in
lakehouse/jobs/, while the logic that can be wrong (parsing, validation,
deduplication, aggregation) can be unit-tested against a plain local Spark
session (tests/lakehouse/test_transforms.py).

Layers
  bronze  raw Kafka records, stored as-is (JSON string + Kafka lineage)
  silver  parsed, validated, deduplicated; bad records go to quarantine
  gold    business aggregates and pipeline health, built with Spark SQL
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from lakehouse.schema import SCORED_EVENT_SCHEMA
from src.events import EVENT_FIELDS

EVENT_COLUMNS = list(EVENT_FIELDS)
LINEAGE_COLUMNS = ["kafka_partition", "kafka_offset", "ingested_at"]

# from_json fills this extra column with the original text when a record is
# malformed or has a field of the wrong type (PERMISSIVE mode), which is a
# reliable way to tell "garbage" apart from "valid JSON with a null field".
_CORRUPT = "_corrupt_record"
_PARSE_SCHEMA = StructType(
    SCORED_EVENT_SCHEMA.fields + [StructField(_CORRUPT, StringType(), True)]
)


def parse_bronze(bronze_df: DataFrame) -> DataFrame:
    """Parse bronze's `raw_value` JSON into typed columns.

    Expects: raw_value (string) and the LINEAGE_COLUMNS. Never drops a row:
    records that fail to parse come out with null event columns and
    `_unparseable = true`, so split_valid() can quarantine them with a reason.
    """
    parsed = bronze_df.withColumn(
        "_e",
        F.from_json(F.col("raw_value"), _PARSE_SCHEMA, {"mode": "PERMISSIVE",
                                                        "columnNameOfCorruptRecord": _CORRUPT}),
    )
    return (
        parsed.select(
            *[F.col(f"_e.{c}").alias(c) for c in EVENT_COLUMNS],
            "raw_value",
            *LINEAGE_COLUMNS,
            (F.col("_e").isNull() | F.col(f"_e.{_CORRUPT}").isNotNull()).alias("_unparseable"),
        )
        # ISO-8601 strings -> real timestamps (null if malformed)
        .withColumn("event_time", F.to_timestamp("event_time"))
        .withColumn("scored_at", F.to_timestamp("scored_at"))
    )


def _reject_reason() -> F.Column:
    """First failing rule wins; null means the record is valid."""
    return (
        F.when(F.col("_unparseable"), F.lit("unparseable_json"))
        .when(F.col("transaction_id").isNull() | F.col("card_id").isNull(), F.lit("missing_key"))
        .when(F.col("event_time").isNull(), F.lit("invalid_event_time"))
        .when(F.col("amount").isNull() | (F.col("amount") <= 0), F.lit("invalid_amount"))
        .when(
            F.col("score").isNull() | (F.col("score") < 0) | (F.col("score") > 1),
            F.lit("invalid_score"),
        )
    )


def split_valid(parsed_df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Split parsed records into (valid, invalid).

    valid   typed event columns + event_date (partition column) + lineage
    invalid raw_value + reject_reason + lineage -- kept, never silently lost
    """
    flagged = parsed_df.withColumn("reject_reason", _reject_reason())
    valid = flagged.filter(F.col("reject_reason").isNull()).select(
        *EVENT_COLUMNS,
        F.to_date("event_time").alias("event_date"),
        *LINEAGE_COLUMNS,
    )
    invalid = flagged.filter(F.col("reject_reason").isNotNull()).select(
        "raw_value", "reject_reason", *LINEAGE_COLUMNS
    )
    return valid, invalid


def dedupe_batch(df: DataFrame) -> DataFrame:
    """Keep one row per transaction_id inside a micro-batch (the earliest
    ingested). Kafka is at-least-once, so the same event can appear twice in
    one batch; the silver job's MERGE then handles duplicates across batches."""
    window = Window.partitionBy("transaction_id").orderBy(
        F.col("ingested_at").asc(), F.col("kafka_partition").asc(), F.col("kafka_offset").asc()
    )
    return df.withColumn("_rn", F.row_number().over(window)).filter(F.col("_rn") == 1).drop("_rn")


FRAUD_HOURLY_SQL = """
SELECT
    date_trunc('hour', event_time)                              AS event_hour,
    merchant_category,
    location,
    COUNT(*)                                                    AS txn_count,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)                   AS flagged_count,
    ROUND(SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4) AS flagged_rate,
    ROUND(SUM(amount), 2)                                       AS total_amount,
    ROUND(SUM(CASE WHEN is_fraud THEN amount ELSE 0 END), 2)    AS flagged_amount,
    ROUND(AVG(score), 4)                                        AS avg_score
FROM silver_scored
GROUP BY date_trunc('hour', event_time), merchant_category, location
"""


def gold_fraud_hourly(silver_df: DataFrame) -> DataFrame:
    """Hourly flagged-transaction rate by merchant category and country.

    "Flagged" means the model's decision (score >= threshold), not confirmed
    fraud: ground truth only arrives later through /feedback.
    """
    silver_df.createOrReplaceTempView("silver_scored")
    return silver_df.sparkSession.sql(FRAUD_HOURLY_SQL)


def gold_pipeline_health(silver_df: DataFrame, quarantine_df: DataFrame) -> DataFrame:
    """Hourly data-quality view of the pipeline itself: how many records were
    accepted vs rejected, by ingestion hour. A rising reject_rate means a
    producer changed the contract. (Duplicates are dropped by the silver job
    and are intentionally not counted as rejects.)"""
    hour = F.date_trunc("hour", F.col("ingested_at")).alias("ingest_hour")
    valid = silver_df.groupBy(hour).agg(F.count(F.lit(1)).alias("valid_count"))
    rejected = quarantine_df.groupBy(hour).agg(F.count(F.lit(1)).alias("rejected_count"))
    return (
        valid.join(rejected, "ingest_hour", "full_outer")
        .na.fill(0, ["valid_count", "rejected_count"])
        .withColumn(
            "reject_rate",
            F.round(
                F.col("rejected_count") / (F.col("valid_count") + F.col("rejected_count")), 4
            ),
        )
        .orderBy("ingest_hour")
    )
