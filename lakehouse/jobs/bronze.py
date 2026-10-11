"""Bronze job: Kafka `scored-transactions` -> Delta table, stored raw.

Bronze keeps the record exactly as it arrived (the JSON string plus Kafka
topic/partition/offset/timestamp). No parsing happens here, so a bad or
changed producer payload can never make ingestion fail or lose data: it is
parsed -- and, if necessary, quarantined -- one layer later, and bronze can
always be replayed.

    python -m lakehouse.jobs.bronze            # continuous
    python -m lakehouse.jobs.bronze --once     # drain what is in Kafka, then exit
"""
import argparse

from pyspark.sql import functions as F

from lakehouse.config import LakeConfig, build_spark


def run(cfg: LakeConfig, once: bool = False) -> None:
    spark = build_spark("bronze-scored-transactions", cfg)

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", cfg.kafka_bootstrap)
        .option("subscribe", cfg.topic)
        .option("startingOffsets", "earliest")  # only used on the very first run
        .option("failOnDataLoss", "false")
        .load()
    )

    bronze = raw.select(
        F.col("value").cast("string").alias("raw_value"),
        F.col("key").cast("string").alias("kafka_key"),
        F.col("topic").alias("kafka_topic"),
        F.col("partition").alias("kafka_partition"),
        F.col("offset").alias("kafka_offset"),
        F.col("timestamp").alias("kafka_timestamp"),
        F.current_timestamp().alias("ingested_at"),
    ).withColumn("ingest_date", F.to_date("ingested_at"))

    writer = (
        bronze.writeStream.format("delta")
        .outputMode("append")
        .partitionBy("ingest_date")
        .option("checkpointLocation", cfg.checkpoint("bronze"))
        .queryName("bronze")
    )
    writer = writer.trigger(availableNow=True) if once else writer.trigger(processingTime="10 seconds")

    query = writer.start(cfg.bronze_path)
    print(f"[bronze] {cfg.topic} @ {cfg.kafka_bootstrap} -> {cfg.bronze_path}", flush=True)
    query.awaitTermination()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="process available data, then exit")
    args = p.parse_args()
    run(LakeConfig.from_env(), once=args.once)


if __name__ == "__main__":
    main()
