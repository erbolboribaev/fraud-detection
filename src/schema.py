"""Shared constants between the batch feature pipeline (feature_engineering.py,
which needs Polars) and the online serving path (predictor.py / api/main.py,
which don't). Keeping these here means the API's Docker image doesn't need
to install Polars/DuckDB just to read a category-to-code mapping.
"""

CATEGORIES = ["grocery", "restaurant", "transport", "fuel", "clothing",
              "pharmacy", "electronics", "travel", "gift_card", "crypto"]
CATEGORY_MAP = {c: i for i, c in enumerate(CATEGORIES)}

NO_HISTORY_GAP = 7 * 24 * 3600.0  # fallback "secs_since_last" when a card has no prior transaction

FEATURES = [
    "log_amount",        # log(1 + amount)
    "hour",               # hour of day (0-23)
    "amount_ratio",       # amount / mean of this card's prior transactions
    "prev_txn_count",     # number of this card's prior transactions
    "txn_count_10m",      # number of this card's prior transactions in the last 10 minutes
    "secs_since_last",    # seconds since this card's previous transaction
    "is_new_location",    # 1 if this is the first time this card is seen in this location
    "category_code",      # category code (from CATEGORY_MAP)
]
