"""Feature engineering (batch, Polars).

Golden rule: every feature for a row is computed only from data strictly
before that row's timestamp (no lookahead). Online computation (Redis, in
predictor.py) must match these definitions exactly, or training and serving
disagree (training-serving skew).
"""
import polars as pl

from src.schema import CATEGORY_MAP, FEATURES, NO_HISTORY_GAP


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("card_id", "timestamp")

    df = df.with_columns(
        (pl.col("amount").cum_count().over("card_id") - 1).alias("prev_txn_count"),
        (pl.col("amount").cum_sum().over("card_id") - pl.col("amount")).alias("_prev_sum"),
        pl.lit(1).alias("_one"),
    )
    prev_mean = (
        pl.when(pl.col("prev_txn_count") > 0)
        .then(pl.col("_prev_sum") / pl.col("prev_txn_count"))
    )
    df = df.with_columns(
        pl.col("amount").log1p().alias("log_amount"),
        pl.col("timestamp").dt.hour().alias("hour"),
        (pl.col("amount") / prev_mean).fill_null(1.0).alias("amount_ratio"),
        pl.col("_one")
        .rolling_sum_by("timestamp", window_size="10m", closed="left")
        .over("card_id")
        .fill_null(0)
        .alias("txn_count_10m"),
        # dt.total_seconds() truncates to whole seconds in this Polars version,
        # so go through microseconds to keep full precision (must match predictor.py).
        (pl.col("timestamp").diff().over("card_id").dt.total_microseconds() / 1_000_000)
        .fill_null(NO_HISTORY_GAP).alias("secs_since_last"),
        (pl.col("amount").cum_count().over("card_id", "location") == 1)
        .cast(pl.Int8).alias("is_new_location"),
        pl.col("merchant_category")
        .replace_strict(CATEGORY_MAP, default=-1, return_dtype=pl.Int8)
        .alias("category_code"),
    )
    keep = ["transaction_id", "card_id", "timestamp"] + FEATURES
    if "is_fraud" in df.columns:
        keep.append("is_fraud")
    return df.select(keep).sort("timestamp")


def main():
    df = pl.read_parquet("data/raw/transactions.parquet")
    out = build_features(df)
    out.write_parquet("data/processed/features.parquet")
    print(f"Saved: data/processed/features.parquet  {out.shape}")

    print("\nFeature means (legit vs fraud):")
    summary = out.group_by("is_fraud").agg([pl.col(f).mean().round(2) for f in FEATURES]).sort("is_fraud")
    print(summary)

    print("\nFraud rate by category:")
    cats = (
        df.group_by("merchant_category")
        .agg(pl.col("is_fraud").mean().round(4).alias("fraud_rate"), pl.len().alias("n"))
        .sort("fraud_rate", descending=True)
    )
    print(cats)


if __name__ == "__main__":
    main()
