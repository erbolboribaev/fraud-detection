"""Train a LightGBM fraud model and export it to ONNX.

Key decisions:
  * Time-based split (train -> validation -> test), never a random split,
    because in production we always predict the future from the past.
  * Class imbalance is handled with class weights (scale_pos_weight), not SMOTE.
  * The decision threshold is chosen on the validation set only.
    The test set is touched exactly once, for the final report.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnxmltools
import onnxruntime as ort
import polars as pl
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.schema import CATEGORY_MAP, FEATURES, NO_HISTORY_GAP

DATA_PATH = "data/processed/features.parquet"
MODEL_DIR = Path("models")
ONNX_PATH = MODEL_DIR / "fraud_model.onnx"
META_PATH = MODEL_DIR / "model_meta.json"
# Native LightGBM format, kept alongside the ONNX export. Serving (predictor.py)
# only ever loads the ONNX file; this one exists purely so src/explain.py can
# run exact tree-based SHAP explanations, which onnxruntime can't do.
BOOSTER_PATH = MODEL_DIR / "fraud_model.txt"


def time_split(df: pl.DataFrame, train_frac=0.70, val_frac=0.15):
    """Split a time-sorted dataframe into train / validation / test."""
    df = df.sort("timestamp")
    n = df.height
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df[:i_train], df[i_train:i_val], df[i_val:]


def to_xy(df: pl.DataFrame):
    x = df.select(FEATURES).to_numpy().astype(np.float32)
    y = df["is_fraud"].to_numpy().astype(np.int32)
    return x, y


def best_f1_threshold(y_true, scores):
    """Return the score threshold that maximizes F1 (chosen on validation data)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def report(name, y_true, scores, threshold):
    pred = scores >= threshold
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"\n[{name}] threshold={threshold:.3f}")
    print(f"  ROC-AUC={roc_auc_score(y_true, scores):.4f}  "
          f"PR-AUC={average_precision_score(y_true, scores):.4f}")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  TP={tp}  FP={fp}  FN={fn}")


def main():
    df = pl.read_parquet(DATA_PATH)
    train, val, test = time_split(df)
    x_tr, y_tr = to_xy(train)
    x_va, y_va = to_xy(val)
    x_te, y_te = to_xy(test)
    print(f"train={len(y_tr)} val={len(y_va)} test={len(y_te)}")
    print(f"fraud rate: train={y_tr.mean():.2%} val={y_va.mean():.2%} test={y_te.mean():.2%}")

    # Class weights: weight the rare fraud class by (#negatives / #positives)
    scale_pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"scale_pos_weight={scale_pos_weight:.1f}")

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        x_tr, y_tr,
        eval_set=[(x_va, y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    print(f"best iteration: {model.best_iteration_}")

    # Threshold is picked on validation, then applied unchanged to test
    val_scores = model.predict_proba(x_va)[:, 1]
    threshold = best_f1_threshold(y_va, val_scores)
    report("validation", y_va, val_scores, threshold)
    test_scores = model.predict_proba(x_te)[:, 1]
    report("test", y_te, test_scores, threshold)

    print("\nFeature importance (gain, % of total):")
    gain = model.booster_.feature_importance(importance_type="gain")
    for name, g in sorted(zip(FEATURES, gain), key=lambda t: -t[1]):
        print(f"  {name:<18} {100 * g / gain.sum():5.1f}%")

    # ---- Native LightGBM export (for SHAP; see BOOSTER_PATH note above) ---
    MODEL_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(BOOSTER_PATH))

    # ---- ONNX export -------------------------------------------------------
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
        zipmap=False,  # return a plain probability tensor instead of dicts
    )
    ONNX_PATH.write_bytes(onnx_model.SerializeToString())

    # Everything the serving layer needs to reproduce training-time behavior
    META_PATH.write_text(json.dumps({
        "features": FEATURES,
        "threshold": threshold,
        "category_map": CATEGORY_MAP,
        "no_history_gap": NO_HISTORY_GAP,
    }, indent=2))
    print(f"\nSaved {ONNX_PATH}, {META_PATH}, and {BOOSTER_PATH}")

    # ---- Verify ONNX matches LightGBM and measure latency ------------------
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    prob_name = sess.get_outputs()[1].name  # output 0 is the label, output 1 the probabilities
    onnx_scores = sess.run([prob_name], {input_name: x_te})[0][:, 1]
    print(f"max |ONNX - LightGBM| on test: {np.abs(onnx_scores - test_scores).max():.2e}")

    # Pre-build single-row copies so slicing cost is not part of the measurement
    rows = [x_te[i:i + 1].copy() for i in range(2000)]

    def bench(fn):
        start = time.perf_counter()
        for row in rows:
            fn(row)
        return (time.perf_counter() - start) / len(rows) * 1e6  # microseconds per call

    lgb_us = bench(lambda row: model.predict_proba(row))
    onnx_us = bench(lambda row: sess.run([prob_name], {input_name: row}))
    print(f"single-row latency: LightGBM={lgb_us:.0f}us  ONNX={onnx_us:.0f}us  "
          f"speedup={lgb_us / onnx_us:.1f}x")


if __name__ == "__main__":
    main()
