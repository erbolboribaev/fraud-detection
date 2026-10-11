"""Gold job (batch): silver -> business aggregates, built with Spark SQL.

  gold/fraud_hourly           hourly flagged rate by merchant category and country
  gold/pipeline_health_hourly accepted vs rejected records per ingestion hour

Each run rebuilds both tables from silver (full overwrite), so re-running is
always safe. At real volume this would become an incremental build over the
affected hours only (Delta `replaceWhere`); at this project's size a full
rebuild is simpler and just as correct.

    python -m lakehouse.jobs.gold
"""
from delta.tables import DeltaTable

from lakehouse import transforms
from lakehouse.config import LakeConfig, build_spark


def run(cfg: LakeConfig) -> None:
    spark = build_spark("gold-fraud-metrics", cfg)

    if not DeltaTable.isDeltaTable(spark, cfg.silver_path):
        raise SystemExit(
            f"No silver table at {cfg.silver_path} yet -- run the bronze and silver jobs first."
        )
    silver = spark.read.format("delta").load(cfg.silver_path)

    if DeltaTable.isDeltaTable(spark, cfg.quarantine_path):
        quarantine = spark.read.format("delta").load(cfg.quarantine_path)
    else:  # nothing was ever rejected
        quarantine = spark.createDataFrame([], "ingested_at timestamp, reject_reason string")

    fraud_hourly = transforms.gold_fraud_hourly(silver)
    (
        fraud_hourly.write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true").save(cfg.gold_fraud_hourly_path)
    )

    health = transforms.gold_pipeline_health(silver, quarantine)
    (
        health.write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true").save(cfg.gold_pipeline_health_path)
    )

    print("\n[gold] fraud_hourly (top 10 by flagged_count):")
    spark.read.format("delta").load(cfg.gold_fraud_hourly_path) \
        .orderBy("flagged_count", ascending=False).show(10, truncate=False)
    print("[gold] pipeline_health_hourly:")
    spark.read.format("delta").load(cfg.gold_pipeline_health_path).show(24, truncate=False)


if __name__ == "__main__":
    run(LakeConfig.from_env())
