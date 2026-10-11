"""Configuration and Spark session factory for the lakehouse jobs.

Everything environment-specific comes from environment variables, so the same
code runs against MinIO (docker compose) and real AWS S3:

  LAKE_ROOT       s3a://fraud-lake            (any s3a:// or file:// root)
  S3_ENDPOINT     http://minio:9000           (set for MinIO; leave unset for AWS S3)
  KAFKA_BOOTSTRAP redpanda:9092
  SCORED_TOPIC    scored-transactions

S3 credentials are not handled here: hadoop-aws's default credential chain
reads AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (MinIO in docker compose) or
an instance/IAM role (AWS), so no secret ever passes through this code.
"""
import os
import time
from dataclasses import dataclass

from src.events import SCORED_TOPIC

# Spark 3.5.x line. hadoop-aws must match the Hadoop version bundled with the
# pip `pyspark` wheel (3.3.4 for pyspark 3.5.x); delta-spark 3.2.x targets Spark 3.5.
SPARK_PACKAGES = [
    "io.delta:delta-spark_2.12:3.2.1",
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3",
    "org.apache.hadoop:hadoop-aws:3.3.4",
]


@dataclass(frozen=True)
class LakeConfig:
    kafka_bootstrap: str = "localhost:19092"
    topic: str = SCORED_TOPIC
    lake_root: str = "s3a://fraud-lake"
    s3_endpoint: str | None = None

    @classmethod
    def from_env(cls) -> "LakeConfig":
        return cls(
            kafka_bootstrap=os.getenv("KAFKA_BOOTSTRAP", cls.kafka_bootstrap),
            topic=os.getenv("SCORED_TOPIC", cls.topic),
            lake_root=os.getenv("LAKE_ROOT", cls.lake_root),
            s3_endpoint=os.getenv("S3_ENDPOINT") or None,
        )

    def path(self, *parts: str) -> str:
        return "/".join([self.lake_root.rstrip("/"), *parts])

    # Medallion layout: bronze (raw) -> silver (clean, deduplicated) -> gold (aggregates)
    @property
    def bronze_path(self) -> str:
        return self.path("bronze", "scored_transactions")

    @property
    def silver_path(self) -> str:
        return self.path("silver", "scored_transactions")

    @property
    def quarantine_path(self) -> str:
        return self.path("silver", "scored_transactions_quarantine")

    @property
    def gold_fraud_hourly_path(self) -> str:
        return self.path("gold", "fraud_hourly")

    @property
    def gold_pipeline_health_path(self) -> str:
        return self.path("gold", "pipeline_health_hourly")

    def checkpoint(self, name: str) -> str:
        return self.path("_checkpoints", name)


def build_spark(app_name: str, cfg: LakeConfig | None = None):
    """SparkSession with Delta Lake, the Kafka source and S3A configured."""
    from pyspark.sql import SparkSession

    cfg = cfg or LakeConfig.from_env()
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.jars.packages", ",".join(SPARK_PACKAGES))
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # All timestamps in the lake are UTC; never depend on the container's TZ.
        .config("spark.sql.session.timeZone", "UTC")
        # Small data volume: the default 200 shuffle partitions would be mostly empty tasks.
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "4"))
        .config("spark.sql.streaming.metricsEnabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )
    if cfg.lake_root.startswith("s3a://"):
        builder = builder.config("spark.hadoop.fs.s3a.impl",
                                 "org.apache.hadoop.fs.s3a.S3AFileSystem")
        if cfg.s3_endpoint:  # MinIO / any S3-compatible store
            builder = (
                builder.config("spark.hadoop.fs.s3a.endpoint", cfg.s3_endpoint)
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .config("spark.hadoop.fs.s3a.connection.ssl.enabled",
                        "true" if cfg.s3_endpoint.startswith("https") else "false")
            )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def wait_for_delta_table(spark, path: str, poll_s: float = 10.0,
                         timeout_s: float | None = None) -> None:
    """Block until a Delta table exists at `path`.

    The silver job streams from the bronze table, which only appears once the
    bronze job has committed its first micro-batch; waiting here lets the
    whole stack start in any order.
    """
    from delta.tables import DeltaTable

    waited = 0.0
    while not DeltaTable.isDeltaTable(spark, path):
        if timeout_s is not None and waited >= timeout_s:
            raise TimeoutError(f"no Delta table appeared at {path} within {timeout_s}s")
        print(f"[wait] no Delta table at {path} yet, retrying in {poll_s:.0f}s ...", flush=True)
        time.sleep(poll_s)
        waited += poll_s
