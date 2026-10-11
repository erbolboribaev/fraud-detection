"""Fraud alert delivery.

Always logs to logs/alerts.jsonl and prints to the console -- this works
with zero configuration. If SLACK_WEBHOOK_URL is set, the same alert is also
posted to Slack. Alerting must never break a prediction response, so every
failure here is swallowed rather than raised.
"""
import json
import os
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ALERTS_LOG_PATH = Path("logs/alerts.jsonl")
_alerts_lock = threading.Lock()


def send_alert(transaction_id, card_id, amount, category, location, score):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transaction_id": transaction_id,
        "card_id": card_id,
        "amount": amount,
        "merchant_category": category,
        "location": location,
        "score": score,
    }
    message = (f"[FRAUD ALERT] {transaction_id} card={card_id} "
               f"amount={amount:.2f} category={category} location={location} "
               f"score={score:.3f}")
    print(message)

    try:
        ALERTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _alerts_lock, open(ALERTS_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if webhook_url:
        try:
            payload = json.dumps({"text": message}).encode()
            req = urllib.request.Request(
                webhook_url, data=payload, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass  # Slack delivery is best-effort; the file/console log already has it
