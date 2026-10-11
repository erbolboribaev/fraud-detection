"""Silver job: bronze (raw JSON) -> parsed, validated, deduplicated table.

For every micro-batch read from the bronze Delta table:
  1. parse the JSON into typed columns
  2. send records that fail validation to a quarantine table, with the reason
  3. drop duplicates inside the batch
  4. MERGE into silver, inserting only transaction_ids not already there

Step 4 is what makes the lake idempotent. Kafka (and the API's best-effort
publisher) are at-least-once, and a crashed job replays its last batch;
`MERGE ... WHEN NOT MATCHED THEN INSERT` means a transaction is stored once no
matter how many times it arrives -- the same property predictor.py's SETNX
guard gives the per-card state in Redis.

    python -m lakehouse.jobs.silver
    python -m lakehouse.jobs.silver --once
"""
import argparse

from delta.tables import DeltaTable

from lakehouse import transforms
from lakehouse.config import LakeConfig, build_spark, wait_for_delta_table


def process_batch(batch_df, batch_id: int, cfg: LakeConfig) -> None:
    spark = batch_df.sparkSession

    parsed = transforms.parse_bronze(batch_df).cache()
    try:
        valid, invalid = transforms.split_valid(parsed)

        if not invalid.isEmpty():
            # txnAppId/txnVersion make this append idempotent if the batch is replayed.
            (
                invalid.write.format("delta").mode("append")
                .option("txnAppId", "silver-quarantine")
                .option("txnVersion", batch_id)
                .save(cfg.quarantine_path)
            )

        valid = transforms.dedupe_batch(valid)
        if valid.isEmpty():
            return

        if not DeltaTable.isDeltaTable(spark, cfg.silver_path):
            valid.limit(0).write.format("delta").partitionBy("event_date").save(cfg.silver_path)

        (
            DeltaTable.forPath(spark, cfg.silver_path).alias("t")
            .merge(valid.alias("s"), "t.transaction_id = s.transaction_id")
            .whenNotMatchedInsertAll()
            .execute()
        )
    finally:
        parsed.unpersist()


def run(cfg: LakeConfig, once: bool = False) -> None:
    spark = build_spark("silver-scored-transactions", cfg)
    wait_for_delta_table(spark, cfg.bronze_path)

    bronze_stream = spark.readStream.format("delta").load(cfg.bronze_path)

    writer = (
        bronze_stream.writeStream
        .foreachBatch(lambda df, batch_id: process_batch(df, batch_id, cfg))
        .option("checkpointLocation", cfg.checkpoint("silver"))
        .queryName("silver")
    )
    writer = writer.trigger(availableNow=True) if once else writer.trigger(processingTime="10 seconds")

    query = writer.start()
    print(f"[silver] {cfg.bronze_path} -> {cfg.silver_path} (quarantine: {cfg.quarantine_path})",
          flush=True)
    query.awaitTermination()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="process available data, then exit")
    args = p.parse_args()
    run(LakeConfig.from_env(), once=args.once)


if __name__ == "__main__":
    main()
