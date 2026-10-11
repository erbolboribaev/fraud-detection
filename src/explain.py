"""SHAP-based explainability for the fraud model.

Runs offline against the native LightGBM model (models/fraud_model.txt),
not the ONNX model served by the API -- onnxruntime sessions don't expose
the tree structure SHAP's TreeExplainer needs. Two things this gives you:

  1. A global summary: which features drive the model's decisions overall,
     and in which direction (e.g. "high amount_ratio pushes toward fraud").
  2. A per-transaction breakdown: for one specific scored transaction (e.g.
     one that triggered a [FRAUD ALERT]), exactly how much each feature
     pushed the score up or down -- useful for justifying an alert to a
     human reviewer or a customer dispute.
"""
import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # no display available in a script/container
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap

from src.schema import FEATURES

# This version's TreeExplainer emits a heads-up about a return-shape change
# that applies to a different code path than the one we use here; verified
# (see the conversation/tests) that shap_values() still returns a plain
# ndarray for this binary LightGBM model, so the warning is not actionable.
warnings.filterwarnings("ignore", message="LightGBM binary classifier")

BOOSTER_PATH = "models/fraud_model.txt"
DATA_PATH = "data/processed/features.parquet"
PREDICTIONS_PATH = "logs/predictions.jsonl"
SUMMARY_PLOT_PATH = "reports/shap_summary.png"


def load_explainer():
    booster = lgb.Booster(model_file=BOOSTER_PATH)
    return booster, shap.TreeExplainer(booster)


def global_summary(sample_size=2000):
    """Save a bar chart of each feature's average impact on the fraud score,
    computed over a sample of the training data."""
    booster, explainer = load_explainer()
    df = pl.read_parquet(DATA_PATH).select(FEATURES)
    if df.height > sample_size:
        df = df.sample(n=sample_size, seed=42)
    x = df.to_pandas()

    shap_values = explainer.shap_values(x)

    Path("reports").mkdir(exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, x, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved global SHAP summary to {SUMMARY_PLOT_PATH}")

    mean_abs = abs(shap_values).mean(axis=0)
    order = mean_abs.argsort()[::-1]
    print("\nAverage impact on fraud score, by feature:")
    for i in order:
        print(f"  {FEATURES[i]:<18} {mean_abs[i]:.4f}")


def explain_one(features: dict):
    """Print a per-feature breakdown of how much each feature pushed the
    score toward or away from fraud, for one transaction's feature dict.

    SHAP values here are in log-odds space (they add up exactly to the
    model's raw margin output), not 0-1 probability space -- that's why a
    single feature's contribution can look large even for a low final
    score. The last line converts the total back to the same 0-1 scale
    /predict reports, via a sigmoid, so it's directly comparable.
    """
    booster, explainer = load_explainer()
    x = np.array([[features[f] for f in FEATURES]], dtype=np.float64)
    shap_values = explainer.shap_values(x)[0]
    base_value = explainer.expected_value

    print(f"Base rate (average model output, log-odds): {base_value:.4f}")
    contributions = sorted(zip(FEATURES, shap_values, x[0]), key=lambda t: -abs(t[1]))
    for name, contribution, value in contributions:
        direction = "toward fraud" if contribution > 0 else "away from fraud"
        print(f"  {name:<18} value={value:<10} pushed {direction} by {contribution:+.4f} (log-odds)")

    total_log_odds = base_value + shap_values.sum()
    probability = 1 / (1 + np.exp(-total_log_odds))
    print(f"\n  Total: {total_log_odds:+.4f} log-odds -> {probability:.4f} probability "
          f"(should match /predict's score for this transaction)")


def explain_transaction_id(transaction_id: str):
    """Look up a logged transaction's features by ID and explain it."""
    with open(PREDICTIONS_PATH) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("transaction_id") == transaction_id:
                print(f"Explaining {transaction_id} (score={entry['score']:.3f}, "
                      f"is_fraud={entry['is_fraud']})")
                explain_one({f: entry[f] for f in FEATURES})
                return
    print(f"transaction_id {transaction_id} not found in {PREDICTIONS_PATH}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transaction-id", help="Explain one logged transaction by ID")
    args = p.parse_args()

    if args.transaction_id:
        explain_transaction_id(args.transaction_id)
    else:
        global_summary()


if __name__ == "__main__":
    main()
