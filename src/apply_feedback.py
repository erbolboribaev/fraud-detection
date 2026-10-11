"""Joins logged predictions (logs/predictions.jsonl) with reported ground
truth (logs/feedback.jsonl) on transaction_id, producing a corrected
labeled dataset (data/processed/feedback_features.parquet) that can be
concatenated into the next training run. This is what "closes the loop"
between what the model predicted and what actually happened.
"""
import json
from pathlib import Path

import polars as pl

from src.schema import FEATURES

PREDICTIONS_PATH = "logs/predictions.jsonl"
FEEDBACK_PATH = "logs/feedback.jsonl"
OUTPUT_PATH = "data/processed/feedback_features.parquet"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    if not Path(PREDICTIONS_PATH).exists() or not Path(FEEDBACK_PATH).exists():
        print(f"Need both {PREDICTIONS_PATH} and {FEEDBACK_PATH} to exist. "
              f"Send some /predict traffic and at least one /feedback report first.")
        return

    predictions = pl.DataFrame(load_jsonl(PREDICTIONS_PATH))
    feedback = pl.DataFrame(load_jsonl(FEEDBACK_PATH))

    # A transaction can be reported more than once; keep only the latest report.
    feedback = (
        feedback.sort("reported_at")
        .group_by("transaction_id", maintain_order=True)
        .last()
    )

    joined = predictions.join(feedback, on="transaction_id", how="inner")
    if joined.height == 0:
        print("No predictions matched any feedback transaction_id -- nothing to do.")
        return

    disagreements = joined.filter(pl.col("is_fraud") != pl.col("actual_is_fraud"))
    print(f"{joined.height} feedback-labeled transactions found.")
    print(f"{disagreements.height} of them disagree with the model's original call "
          f"(false positives or false negatives) -- these are the most valuable "
          f"examples for the next retrain.")

    corrected = joined.select(
        FEATURES + [pl.col("actual_is_fraud").cast(pl.Int64).alias("is_fraud")]
    )
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    corrected.write_parquet(OUTPUT_PATH)
    print(f"Saved {corrected.height} corrected rows to {OUTPUT_PATH}.")
    print("To retrain with this feedback included, concatenate it with "
          "data/processed/features.parquet before the time split in src/train.py.")


if __name__ == "__main__":
    main()
