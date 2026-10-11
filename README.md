# Real-Time Fraud Detection Pipeline

[![CI](https://github.com/erbolboribaev/fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/erbolboribaev/fraud-detection/actions)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A real-time credit-card fraud detection system: synthetic transactions stream through Kafka, get scored by a LightGBM model (served as ONNX, ~10-30 µs model inference), and flagged transactions trigger alerts -- all observable through a live React dashboard with SHAP explanations and data-drift monitoring.

Built end-to-end: data generation, feature engineering, model training, low-latency serving, streaming, monitoring, a feedback loop for retraining, and three deployment targets (Docker Compose, Kubernetes, CI/CD).

## Demo

![Dashboard screenshot](docs/dashboard-screenshot.png)

Run it yourself in one command -- see [Quickstart](#quickstart) below.

## What it does

1. A synthetic transaction generator streams card transactions through Kafka (Redpanda), including realistic fraud bursts (stolen-card patterns: rapid, high-value, cross-border purchases in risky categories).
2. A consumer forwards each transaction to a FastAPI `/predict` endpoint, which is the real product boundary -- it doesn't know or care whether the caller is the demo generator or a real payment processor.
3. The endpoint computes 8 features from Redis-backed per-card history (amount ratio vs. the card's average, time since last transaction, velocity in the last 10 minutes, etc.), scores them with an ONNX-exported LightGBM model, and returns a fraud probability.
4. Flagged transactions trigger an alert (console + file, optionally Slack) and are explained via SHAP (which features pushed the score up or down, and by how much).
5. Every scored transaction is logged and compared against the training distribution (Evidently AI) to catch data drift before it silently degrades accuracy.
6. A `/feedback` endpoint lets a human confirm or correct a prediction, closing the loop for future retraining.

## Architecture

```mermaid
flowchart LR
    GEN["Generator<br/>(demo traffic only)"] --> RP["Redpanda<br/>(Kafka topic)"]
    RP --> CONS["Consumer"]
    CONS -->|"POST /predict"| API["FastAPI<br/>(real product boundary)"]
    API <--> REDIS[("Redis<br/>per-card state")]
    API --> ONNX["ONNX model<br/>(LightGBM, ~10-30 µs)"]
    API --> LOG[("predictions.jsonl")]
    API -.->|"if fraud"| ALERT["Alerts"]
    LOG --> DASH["React dashboard<br/>(polls every 2s)"]
    LOG --> DRIFT["Evidently<br/>(drift report)"]
```

The box most people get wrong: **the generator/Redpanda/consumer path exists only for this demo.** A real payment processor would call `POST /predict` directly -- nothing downstream of that line would need to change.

## Results

> **Note:** the model is trained and evaluated on **synthetic, rule-generated data**. These numbers show how well it learned those fraud patterns, not how it would perform on real transactions.

| Metric                                           | Value                                       |
| ------------------------------------------------ | ------------------------------------------- |
| Test ROC-AUC                                     | 0.987                                       |
| Test PR-AUC                                      | 0.945                                       |
| Precision / Recall @ threshold                   | 0.947 / 0.868                               |
| Model inference latency (ONNX, single transaction) | ~10-30 µs                                 |
| Model inference latency (raw LightGBM, for comparison) | ~400-800 µs (~30x slower)             |
| Top features by SHAP impact                      | `amount_ratio` (47%), `category_code` (21%) |
| Tests                                            | 11 passing (pytest + CI)                    |

Latency above is model inference only; the full `/predict` request additionally includes HTTP handling and Redis reads/writes.

## Tech stack

| Layer               | Choice                                            |
| ------------------- | ------------------------------------------------- |
| Streaming           | Kafka (Redpanda)                                  |
| Feature engineering | Polars                                            |
| Model               | LightGBM, exported to ONNX for serving            |
| Serving             | FastAPI + Redis (per-card state)                  |
| Explainability      | SHAP (TreeExplainer on the native LightGBM model) |
| Drift monitoring    | Evidently AI                                      |
| Frontend            | React + Vite                                      |
| Deployment          | Docker Compose, Kubernetes (manifests included)   |
| CI/CD               | GitHub Actions (tests + Docker image builds)      |

## Quickstart

```bash
git clone https://github.com/erbolboribaev/fraud-detection.git
cd fraud-detection
docker compose up --build -d
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173` for the dashboard, `http://localhost:8000/docs` for the API, `http://localhost:8080` for Redpanda Console.

## Project structure

```
fraud-detection/
├── src/
│   ├── data_generator.py       # synthetic transaction stream (Kafka producer)
│   ├── feature_engineering.py  # batch feature computation (training)
│   ├── schema.py               # shared constants (no Polars dependency)
│   ├── predictor.py            # online feature computation + ONNX inference
│   ├── train.py                # LightGBM training, ONNX + native export
│   ├── consumer.py             # Kafka -> API bridge
│   ├── alerts.py               # fraud alert delivery (file/console/Slack)
│   ├── drift_report.py         # Evidently drift report
│   ├── explain.py              # standalone SHAP CLI
│   └── apply_feedback.py       # joins feedback into a retraining dataset
├── api/main.py                 # FastAPI app: /predict, /feedback, dashboard reads
├── frontend/                   # React + Vite dashboard
├── models/                     # trained model artifacts (ONNX + native LightGBM)
├── docker/                     # Dockerfile.api (lean), Dockerfile.stream
├── k8s/                        # Kubernetes manifests
├── tests/                      # pytest suite
├── docs/                       # screenshots
├── docker-compose.yml
├── requirements.txt            # requirements-api.txt / requirements-stream.txt per image
└── .github/workflows/ci.yml
```

## Technical decisions

**ONNX for serving, not the raw LightGBM model.** ~30x faster single-row inference (µs instead of hundreds of µs), which matters because `/predict` is in the hot path of every transaction. The native LightGBM model is still saved alongside it (`models/fraud_model.txt`) purely because SHAP's `TreeExplainer` needs the real tree structure -- `onnxruntime` sessions don't expose it. Two artifacts, two purposes: ONNX for the latency-critical path, native for the offline explainability path.

**Time-based train/val/test split, not random.** A random split lets the model implicitly see "future" transactions during training, which never happens in production -- you always predict forward from the past. Random splitting would have overstated accuracy.

**Class weights (`scale_pos_weight`), not SMOTE.** Several features here are counts or categorical codes (`txn_count_10m`, `category_code`); SMOTE's synthetic interpolation between minority-class points produces nonsensical values for them (e.g. `category_code = 5.4`). Class weighting penalizes the loss function instead of fabricating data, which keeps every training example real.

**Redis for per-card state, not a database.** Each feature (amount ratio, time since last transaction, 10-minute velocity) only needs a small, fast-expiring window of state per card. Redis gives sub-millisecond reads/writes for exactly that access pattern; a relational database would add latency the hot path can't afford.

**An idempotency guard on every state write.** Kafka guarantees *at-least-once* delivery -- after a consumer restart before offsets are committed, the same transaction can be redelivered. Without a guard, the duplicate would double-count itself into the card's own history (e.g. `amount_ratio` computed against itself). Fixed with a Redis `SETNX` keyed on `transaction_id`: harmless to re-score, but only ever counted once.

**`src/schema.py` split out from `feature_engineering.py`.** The API's Docker image doesn't need Polars/DuckDB at all -- but `predictor.py` originally imported shared constants from a module that did `import polars` at the top. Moving those constants into a dependency-free module kept the serving image lean without duplicating the category map by hand.

**`initContainers` in the Kubernetes manifests, not `depends_on`.** Docker Compose's `depends_on: condition: service_healthy` has no equivalent in vanilla Kubernetes -- there's no built-in "wait for this other pod to be ready" primitive. Each dependent Deployment runs a small `initContainer` that polls the dependency's port before the main container starts, which is the standard idiom for this.

## Bugs found during development

These were found by running the system end to end and noticing the output didn't make sense.

1. **Training-serving skew from a silent Polars truncation.** `dt.total_seconds()` truncated to whole seconds in the installed Polars version, so the model trained on `secs_since_last` values rounded to the nearest second, while the online predictor computed the same feature with full microsecond precision. Fixed by going through `dt.total_microseconds() / 1_000_000` instead, and verified by asserting batch and online feature computation match exactly (`tests/test_pipeline.py`).

2. **A timezone bug that only showed up at 20:00 UTC.** FastAPI parsed incoming ISO timestamps (no `Z` suffix) as naive datetimes; Python's `.timestamp()` then interpreted a naive datetime as the *server's local time zone* rather than UTC, silently shifting the `hour` feature by exactly the server's UTC offset. Caught by comparing a transaction's logged hour against its raw timestamp and noticing they didn't match -- fixed by explicitly attaching `tzinfo=UTC` before conversion, with a regression test run under `TZ=Asia/Tashkent` to prove it.

3. **Kafka redelivery corrupting Redis state.** After a consumer restart, a batch of already-processed messages replayed from the last *committed* (not last *processed*) offset. Each replayed message updated Redis again, making a transaction look like its own preceding transaction (`amount_ratio` computed against itself, `secs_since_last = 0`). Fixed with the idempotency guard described above, plus switching the consumer to manual offset commits.

## Limitations / what would break at 100x scale

- **`logs/predictions.jsonl` is a flat file, re-read on every dashboard request.** Fine at demo volume; at real production throughput this needs a proper time-series store (or at minimum a capped ring buffer) instead of scanning a growing file.
- **Single Redis instance is a single point of failure.** Production would need Redis Cluster or a managed equivalent with replication.
- **The model is trained entirely on synthetic data.** The fraud patterns are rule-generated, so the reported metrics describe how well the model learned those rules -- not how it would perform on real transaction data.
- **No authentication on the API.** `/predict` and `/feedback` are open; a real deployment needs request signing or mTLS between the payment processor and this service.

## Testing

```bash
pytest tests/ -v
```

11 tests covering feature-engineering correctness, training/serving parity, idempotency under simulated Kafka redelivery, and the timezone fix (run under a non-UTC `TZ` to prove it).

## License

MIT
