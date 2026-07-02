"""
Generate synthetic balancing-market (BAL) settlement prices for the training
split -- the third market in Demir et al. (2023)'s DAM-CID-BAL arbitrage
chain. reinforce_threshold_policy's dataset only covers DAM (auction) + CID
(order book) since its paper never trades a balancing market; this script
adds the missing piece so BAL settlement can use a genuinely independent
price series instead of a same-session CID proxy.

Schema: one row per (delivery_start, quarter_index in 0..3) with the
quarter-hourly "take" (cost to cover a short position) and "feed" (revenue
for an excess/long position) prices. The paper's hourly ptake/pfeed
(Nomenclature, §2.2) are the average of the four quarter-hourly prices for
that delivery hour.

Take/feed prices share the SAME hourly diurnal base pattern used by
generate_train_data.py's auction/CIM generators (so all three markets
reflect the same underlying demand/price fundamentals, exactly like real
DAM/CID/BAL prices are correlated via shared fundamentals despite clearing
independently), plus their own independent noise and an asymmetric
imbalance premium/discount: take_price > reference > feed_price, matching
the real-world asymmetry of imbalance settlement (being short costs more
than being long earns).

Must be run against the SAME delivery-day set as generate_train_data.py
(same SEED/START_DATE/N_DAYS/DST exclusion logic, deterministic and
independent of the RNG draws) so delivery_start values line up exactly with
the existing cim_order_book.csv / intraday_auction_curves.csv.

Run from the project root:
    venv/bin/python3 scripts/data_generation/generate_train_balancing.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Configuration (must match generate_train_data.py) ────────────────────────
SEED       = 42
TIMEZONE   = "Europe/Berlin"
N_DAYS     = 181
START_DATE = "2023-01-01"
N_QUARTERS = 4

_PROJECT_ROOT = Path(__file__).parent.parent.parent
BAL_FILE      = _PROJECT_ROOT / "data" / "train" / "balancing_prices.csv"

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


def _is_dst_transition_day(day: pd.Timestamp) -> bool:
    """Identical logic to generate_train_data.py -- keeps the delivery-day set aligned."""
    midnight_start = day.normalize()
    midnight_end = midnight_start + pd.DateOffset(days=1)
    diff_s = (midnight_end.tz_convert("UTC") - midnight_start.tz_convert("UTC")).total_seconds()
    return diff_s != 86400


if __name__ == "__main__":
    np.random.seed(SEED)
    BAL_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_days = pd.date_range(START_DATE, periods=N_DAYS, freq="D", tz=TIMEZONE)
    delivery_days = pd.DatetimeIndex(
        [d for d in all_days if not _is_dst_transition_day(d)]
    )

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
