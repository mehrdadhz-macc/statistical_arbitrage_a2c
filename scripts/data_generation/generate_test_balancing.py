"""
Generate synthetic balancing-market (BAL) settlement prices for the test
split. See generate_train_balancing.py for the full rationale; this mirrors
generate_test_data.py's date range/seed exactly so delivery_start values line
up with the existing test cim_order_book.csv / intraday_auction_curves.csv.

Run from the project root:
    venv/bin/python3 scripts/data_generation/generate_test_balancing.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Configuration (must match generate_test_data.py) ─────────────────────────
SEED       = 123
TIMEZONE   = "Europe/Berlin"
N_DAYS     = 31
START_DATE = "2023-08-01"
N_QUARTERS = 4

OUT_DIR  = Path(__file__).parent.parent.parent / "data" / "test"
BAL_FILE = OUT_DIR / "balancing_prices.csv"

HOURLY_BASE_PRICE = np.array([
    35, 33, 32, 31, 32, 35,
    42, 52, 58, 56, 54, 53,
    52, 51, 52, 54, 58, 62,
    65, 63, 58, 52, 45, 40,
], dtype=float)


def generate_balancing_prices(delivery_days: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for delivery_day in delivery_days:
        is_weekend = delivery_day.dayofweek >= 5

        for hour in range(24):
            delivery_start = delivery_day.replace(
                hour=hour, minute=0, second=0, microsecond=0
            )
            base = HOURLY_BASE_PRICE[hour] * (0.85 if is_weekend else 1.0)
            daily_noise = np.random.normal(0, 3.0)

            for q in range(N_QUARTERS):
                reference = base + daily_noise + np.random.normal(0, 1.5)
                take_premium = np.random.uniform(2.0, 15.0)
                feed_discount = np.random.uniform(2.0, 15.0)
                take_price = round(reference + take_premium, 2)
                feed_price = round(reference - feed_discount, 2)
                rows.append((delivery_start, q, take_price, feed_price))

    return pd.DataFrame(rows, columns=[
        "delivery_start", "quarter_index", "take_price_eur_mwh", "feed_price_eur_mwh",
    ])


if __name__ == "__main__":
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    delivery_days = pd.date_range(START_DATE, periods=N_DAYS, freq="D", tz=TIMEZONE)

    t0 = time.time()
    bal_df = generate_balancing_prices(delivery_days)
    bal_df.to_csv(BAL_FILE, index=False)
    elapsed = time.time() - t0

    assert (bal_df["take_price_eur_mwh"] > bal_df["feed_price_eur_mwh"]).all(), \
        "take price must exceed feed price at every quarter (imbalance asymmetry)"

    print(f"Output file: {BAL_FILE}")
    print(f"Rows: {len(bal_df):,}")
    print(f"Date range: {bal_df['delivery_start'].min()}  ->  {bal_df['delivery_start'].max()}")
    print(f"Validation: PASSED ({elapsed:.1f}s)")
