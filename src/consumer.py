"""Bridges the Kafka/Redpanda 'transactions' topic to the FastAPI /predict
endpoint: this is the "Inference API" arrow in the architecture table, wired
up end to end.

Each message on the topic is one JSON transaction (see data_generator.py for
the schema). For every message this script POSTs to /predict and logs the
result; transactions scored above the model's threshold are flagged clearly.
"""
import argparse
import json
import sys

import httpx
from kafka import KafkaConsumer


def run(bootstrap, topic, group_id, api_url):
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=group_id,
        auto_offset_reset="latest",  # only new transactions; use "earliest" to replay history
        enable_auto_commit=False,  # commit manually, right after a message is
        # fully scored, so a restart replays at most the single in-flight
        # message (predictor.py's transaction_id guard makes that safe too)
        value_deserializer=lambda v: json.loads(v.decode()),
        key_deserializer=lambda k: k.decode() if k else None,
    )
    print(f"Listening on '{topic}' at {bootstrap}, scoring via {api_url} ...")

    with httpx.Client(base_url=api_url, timeout=5.0) as client:
        for msg in consumer:
            txn = msg.value
            payload = {
                "transaction_id": txn["transaction_id"],
                "card_id": txn["card_id"],
                "amount": txn["amount"],
                "merchant_category": txn["merchant_category"],
                "location": txn["location"],
                "timestamp": txn["timestamp"],
            }
            try:
                resp = client.post("/predict", json=payload)
                resp.raise_for_status()
                result = resp.json()
            except httpx.HTTPError as e:
                print(f"[error] {txn['transaction_id']}: {e}", file=sys.stderr)
                continue  # not committed: this message will be redelivered

            consumer.commit()  # only after a successful score, not before

            tag = "FRAUD" if result["is_fraud"] else "ok"
            # ground-truth label from the generator, for sanity-checking the
            # model against synthetic data; never available for real traffic
            truth = " truth=FRAUD" if txn.get("is_fraud") else ""
            print(
                f"[{tag:>5}] {result['transaction_id']} "
                f"card={txn['card_id']} amount={txn['amount']:.2f} "
                f"category={txn['merchant_category']} score={result['score']:.3f}"
                f"{truth}"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", default="localhost:19092")
    p.add_argument("--topic", default="transactions")
    p.add_argument("--group-id", default="fraud-scorer")
    p.add_argument("--api-url", default="http://localhost:8000")
    args = p.parse_args()
    run(args.bootstrap, args.topic, args.group_id, args.api_url)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
