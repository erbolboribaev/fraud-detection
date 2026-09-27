"""Compares the model's training-time feature distribution (reference)
against recent production traffic (current) using Evidently, and saves an
HTML report.

Reference: data/processed/features.parquet (what the model was trained on).
Current:   logs/predictions.jsonl (features logged by api/main.py for every
           /predict call). Run this periodically (e.g. daily/weekly) to
           catch data drift before it silently degrades model accuracy.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import polars as pl
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.schema import FEATURES

REFERENCE_PATH = "data/processed/features.parquet"
CURRENT_PATH = "logs/predictions.jsonl"
REPORT_PATH = "reports/drift_report.html"
MIN_CURRENT_ROWS = 30  # below this, drift stats are too noisy to trust


def load_reference(sample_size=5000):
    df = pl.read_parquet(REFERENCE_PATH).select(FEATURES)
    if df.height > sample_size:
        df = df.sample(n=sample_size, seed=42)
    return df.to_pandas()


def load_current():
    rows = []
    with open(CURRENT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=FEATURES)
    return pd.DataFrame(rows)[FEATURES]


def main():
    if not Path(CURRENT_PATH).exists():
        print(f"No production logs yet at {CURRENT_PATH}. "
              f"Send some traffic through /predict first, then rerun this.")
        sys.exit(1)

    reference = load_reference()
    current = load_current()
    if len(current) < MIN_CURRENT_ROWS:
        print(f"Only {len(current)} logged predictions found (need at least "
              f"{MIN_CURRENT_ROWS}). Send more traffic through /predict, then rerun.")
        sys.exit(1)

    print(f"Reference: {len(reference)} rows (training data)")
    print(f"Current:   {len(current)} rows (production traffic)")

    report = Report(metrics=[DataDriftPreset(columns=FEATURES)])
    mapping = ColumnMapping(numerical_features=FEATURES)
    report.run(reference_data=reference, current_data=current, column_mapping=mapping)

    Path("reports").mkdir(exist_ok=True)
    report.save_html(REPORT_PATH)

    result = report.as_dict()
    summary = result["metrics"][0]["result"]
    detail = result["metrics"][1]["result"]["drift_by_columns"]

    print(f"\nOverall dataset drift detected: {summary['dataset_drift']}")
    print(f"Drifted columns: {summary['number_of_drifted_columns']}/{summary['number_of_columns']}")
    print("\nPer-feature drift:")
    for col in FEATURES:
        info = detail[col]
        flag = "DRIFTED" if info["drift_detected"] else "ok"
        # drift_score's meaning depends on which stat test Evidently picked
        # for this column (e.g. a p-value for KS, a distance for Wasserstein)
        # -- only drift_detected (Evidently's own threshold check) is
        # comparable across columns.
        print(f"  {col:<18} {flag:<8} score={info['drift_score']:.4f} ({info['stattest_name']})")

    print(f"\nFull report saved to {REPORT_PATH} -- open it in a browser.")


if __name__ == "__main__":
    main()
