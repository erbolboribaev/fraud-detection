"""Sintetik tranzaksiya generatori.

Ikki rejim:
  file  - o'qitish uchun dataset yaratadi (data/raw/transactions.parquet)
  kafka - real vaqtli oqimni Redpanda'dagi 'transactions' topic'iga yuboradi
"""
import argparse
import itertools
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

COUNTRIES = ["UZ", "TR", "AE", "DE", "GB", "US"]
COUNTRY_WEIGHTS = [0.70, 0.08, 0.07, 0.05, 0.05, 0.05]

CATEGORY_WEIGHTS = {
    "grocery": 0.30, "restaurant": 0.20, "transport": 0.15, "fuel": 0.10,
    "clothing": 0.08, "pharmacy": 0.07, "electronics": 0.05,
    "travel": 0.03, "gift_card": 0.01, "crypto": 0.01,
}
RISKY_CATEGORIES = ["electronics", "gift_card", "crypto", "travel"]


@dataclass
class Card:
    card_id: str
    home_country: str
    avg_amount: float


def make_cards(n: int, rng: random.Random) -> list[Card]:
    return [
        Card(
            card_id=f"card_{i:05d}",
            home_country=rng.choices(COUNTRIES, COUNTRY_WEIGHTS)[0],
            avg_amount=rng.lognormvariate(3.7, 0.6),  # medianasi ~ 40
        )
        for i in range(n)
    ]


def _record(rng, card, t, txn_id, amount, country, category, is_fraud):
    return {
        "transaction_id": txn_id,
        "card_id": card.card_id,
        "amount": round(amount, 2),
        "merchant": f"{category}_{rng.randint(1, 50):02d}",
        "merchant_category": category,
        "location": country,
        "timestamp": t.isoformat(),
        "is_fraud": int(is_fraud),
    }


def legit_txn(rng, card, t, txn_id):
    amount = card.avg_amount * rng.lognormvariate(0, 0.5)
    if rng.random() < 0.97:
        country = card.home_country
    else:  # ba'zan oddiy mijoz ham safarda bo'ladi
        country = rng.choice(COUNTRIES)
    category = rng.choices(list(CATEGORY_WEIGHTS), list(CATEGORY_WEIGHTS.values()))[0]
    return _record(rng, card, t, txn_id, amount, country, category, False)


def fraud_txn(rng, card, t, txn_id):
    # Har bir belgi faqat 70% ehtimol bilan "gumonli" bo'ladi
    if rng.random() < 0.7:
        amount = card.avg_amount * rng.uniform(4, 12)
    else:
        amount = card.avg_amount * rng.lognormvariate(0, 0.5)
    if rng.random() < 0.7:
        country = rng.choice([c for c in COUNTRIES if c != card.home_country])
    else:
        country = card.home_country
    if rng.random() < 0.7:
        category = rng.choice(RISKY_CATEGORIES)
    else:
        category = rng.choices(list(CATEGORY_WEIGHTS), list(CATEGORY_WEIGHTS.values()))[0]
    return _record(rng, card, t, txn_id, amount, country, category, True)


def activity(hour: int) -> float:
    """Oddiy tranzaksiyalar tunda kamroq bo'ladi."""
    return 0.25 if hour < 6 else 1.0


def event_stream(cards, fraud_rate=0.02, gap_seconds=30.0, seed=42):
    rng = random.Random(seed)
    t = datetime(2026, 1, 1)
    ids = itertools.count()
    avg_burst = 2.8  # [1,1,3,4,5] ning o'rtachasi
    burst_p = fraud_rate / avg_burst

    while True:
        t += timedelta(seconds=rng.expovariate(1 / gap_seconds))
        card = rng.choice(cards)
        if rng.random() < burst_p:
            for i in range(rng.choice([1, 1, 3, 4, 5])):
                if i:
                    t += timedelta(seconds=rng.uniform(3, 60))
                yield fraud_txn(rng, card, t, f"t{next(ids):09d}")
        elif rng.random() < activity(t.hour):
            yield legit_txn(rng, card, t, f"t{next(ids):09d}")


def run_file(args, stream):
    import polars as pl

    rows = list(itertools.islice(stream, args.n))
    df = pl.DataFrame(rows).with_columns(pl.col("timestamp").str.to_datetime())
    df.write_parquet(args.out)
    print(f"Saqlandi: {args.out}")
    print(f"Qatorlar: {df.height}, firibgarlik ulushi: {df['is_fraud'].mean():.2%}")
    print(df.head(5))


def run_kafka(args, stream):
    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        key_serializer=lambda k: k.encode(),
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    delay, sent = 1 / args.rate, 0
    print(f"Yuborilmoqda: {args.bootstrap} -> '{args.topic}' ({args.rate}/s). To'xtatish: Ctrl+C")
    try:
        for txn in stream:
            producer.send(args.topic, key=txn["card_id"], value=txn)
            sent += 1
            if sent % 100 == 0:
                print(f"{sent} ta yuborildi")
            time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()
        producer.close()
        print(f"Jami yuborildi: {sent}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["file", "kafka"], required=True)
    p.add_argument("--n", type=int, default=200_000, help="file rejimi: qatorlar soni")
    p.add_argument("--out", default="data/raw/transactions.parquet")
    p.add_argument("--rate", type=float, default=20, help="kafka rejimi: sekundiga xabar")
    p.add_argument("--bootstrap", default="localhost:19092")
    p.add_argument("--topic", default="transactions")
    p.add_argument("--cards", type=int, default=2000)
    p.add_argument("--fraud-rate", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cards = make_cards(args.cards, random.Random(args.seed))
    stream = event_stream(cards, fraud_rate=args.fraud_rate, seed=args.seed)
    (run_file if args.mode == "file" else run_kafka)(args, stream)


if __name__ == "__main__":
    main()
