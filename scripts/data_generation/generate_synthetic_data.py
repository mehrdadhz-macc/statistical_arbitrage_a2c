"""
Generate synthetic DAM-auction, CIM order-book and BAL settlement data for one
data split (train or test). Replaces the four separate scripts previously used
(generate_train_data.py, generate_test_data.py, generate_train_balancing.py,
generate_test_balancing.py).

Why rewritten: the old generators drew the auction mid-price, each CID tick's
mid-price, and the BAL reference price from *independent* noise around the same
hourly base price. That made pdam, the CID price path, and pfeed/ptake all hug
the same value too tightly -- a diagnostic check (rolling out the paper's own
Eq. (8)-(11) behaviour-cloning rules on real training contracts) found that
even the "expert" rules produced a non-HOLD action less than 1% of the time,
because the crossing conditions they check (pa_t < pdam, pb_t > pfeed, etc.)
almost never held. A trained A2C agent had essentially no reward signal to
learn to trade on and collapsed to an always-HOLD policy.

New price model: each hourly delivery contract gets ONE continuous
mean-reverting process (Ornstein-Uhlenbeck / AR(1) around 0, added on top of
the same hourly diurnal base level as before) running from the D-1 15:00
auction gate-closure through physical delivery.
  - The auction "locks in" a price forecast at t=0 (auction time): pdam is a
    small-noise sample of the process's value at that moment.
  - The CID mid-price at each tick is the SAME process, sampled throughout the
    trading session -- so real divergence from pdam opens up and closes again
    as the process wanders away from and back toward its mean, exactly like a
    real intraday price oscillating around (but bounded around) the day-ahead
    anchor rather than drifting off to infinity.
  - The BAL settlement reference is the process's value at the moment of
    delivery, plus the same asymmetric take/feed imbalance premium/discount
    as before (take > reference > feed).
A pure (non-mean-reverting) random walk was tried first and rejected: without
mean reversion a session's price can only ever drift persistently in one
direction, so within any single session only one of BUY/SELL ever becomes
profitable (never both), and the divergence grows unboundedly with session
length instead of settling into a realistic, bounded typical spread. Mean
reversion fixes both: divergence stays within a controlled range, and genuine
back-and-forth crossings of pdam/pfeed/ptake happen within a single session.
All 24 hourly products of a delivery day share one day-level process (a shared
market-wide fundamental), so intraday products co-move realistically instead
of being drawn independently.

Schema (unchanged from the old scripts):
  cim_order_book.csv          : order_id, timestamp, delivery_start,
                                 delivery_end, side, price_eur_mwh, quantity_mwh
  intraday_auction_curves.csv : auction_time, delivery_start, side,
                                 price_eur_mwh, quantity_mwh, level
  balancing_prices.csv        : delivery_start, quarter_index,
                                 take_price_eur_mwh, feed_price_eur_mwh

Split conventions (unchanged from the old scripts):
  train : seed=42,  2023-01-01, 181 days
  test  : seed=123, 2023-08-01, 31 days

Run from the project root:
    venv/bin/python3 scripts/data_generation/generate_synthetic_data.py --split train
    venv/bin/python3 scripts/data_generation/generate_synthetic_data.py --split test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Split configuration ─────────────────────────────────────────────────────

SPLIT_CONFIG = {
    "train": {"seed": 42, "start_date": "2023-01-01", "n_days": 181},
    "test": {"seed": 123, "start_date": "2023-08-01", "n_days": 31},
}

TIMEZONE = "Europe/Berlin"
TICK_FREQ = "min"  # 1-minute ticks (practical synthetic-data default)

N_CIM_LEVELS = 5
N_AUCTION_LEVELS = 10
N_QUARTERS = 4

_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Hourly base price shape (EUR/MWh) -- representative German intraday pattern.
HOURLY_BASE_PRICE = np.array([
    35, 33, 32, 31, 32, 35,   # 00-06: overnight low
    42, 52, 58, 56, 54, 53,   # 06-12: morning ramp
    52, 51, 52, 54, 58, 62,   # 12-18: afternoon plateau
    65, 63, 58, 52, 45, 40,   # 18-24: evening decline
], dtype=float)

# Mean-reverting (AR(1)/Ornstein-Uhlenbeck) fundamental process, in deviation
# from the hourly base price. X[t+1] = PHI * X[t] + STEP_STD * N(0,1), X[0] ~
# N(0, STATIONARY_STD^2). Calibrated empirically so the paper's own Eq.
# (8)-(11) crossing conditions (pa_t < pdam, etc.) fire on ~15-20% of ticks
# (checked across several seeds) -- meaningful, genuinely tradeable
# opportunities without being a near-permanent one (the old, since-deleted
# generators produced <1% -- see the module docstring above).
#
# STATIONARY_STD=0.4 (an earlier calibration attempt) actually only hit a
# ~0.9% crossing rate: the CID bid/ask half-spread (half_spread_t below,
# U(0.5, 2.0) EUR, plus another ~U(0.1, 1.0) first-level step) is on its own
# already ~1.8 EUR wider than the process's typical excursion, so the ask
# price could barely ever wander back down through pdam regardless of the
# process's own dynamics. 1.75 EUR was Monte-Carlo-checked (simulating this
# same AR(1) + spread model directly) to actually deliver ~19-20%.
_HALF_LIFE_MIN = 180.0     # minutes for a divergence to decay by half
_STATIONARY_STD = 1.75     # EUR/MWh, long-run typical divergence from base
PHI = 0.5 ** (1.0 / _HALF_LIFE_MIN)
STEP_STD = _STATIONARY_STD * (1.0 - PHI ** 2) ** 0.5

# Minutes from auction time (D-1 15:00) to each hour h's delivery moment:
# (24 - 15 + h) hours = (9 + h) hours.
_MINUTES_TO_DELIVERY = lambda h: (9 + h) * 60  # noqa: E731


# ═══════════════════════════════════════════════════════════════════════════
# Per-day generation: one shared fundamental walk, sliced per hourly product
# ═══════════════════════════════════════════════════════════════════════════

def _generate_day(delivery_day: pd.Timestamp):
    """Return (auction_rows, cim_df, bal_rows) for all 24 hourly products of one day."""
    trading_open = (delivery_day - pd.Timedelta(days=1)).replace(
        hour=15, minute=0, second=0, microsecond=0
    )
    is_weekend = delivery_day.dayofweek >= 5
    weekend_factor = 0.85 if is_weekend else 1.0

    total_minutes = _MINUTES_TO_DELIVERY(23)  # covers the latest hour's delivery moment
    walk = np.empty(total_minutes + 1)
    walk[0] = np.random.normal(0, _STATIONARY_STD)
    steps = np.random.normal(0, STEP_STD, total_minutes)
    for i in range(total_minutes):
        walk[i + 1] = PHI * walk[i] + steps[i]

    auction_rows = []
    cim_dfs = []
    bal_rows = []

    for h in range(24):
        delivery_start = delivery_day.replace(hour=h, minute=0, second=0, microsecond=0)
        delivery_end = delivery_start + pd.Timedelta(hours=1)
        idx_delivery = _MINUTES_TO_DELIVERY(h)
        idx_close = idx_delivery - 30

        base = HOURLY_BASE_PRICE[h] * weekend_factor

        # ── Auction: locked in at trading_open (walk index 0) ──────────────
        # Idiosyncratic noise here is deliberately kept smaller than the
        # mean-reverting process's own stationary std (_STATIONARY_STD):
        # pdam needs to be a reasonably clean read of walk[0] so that
        # comparing pdam against a trailing-average forecast (src/dam_policy.py's
        # VWAP-BENCH) actually predicts which way the session's CID price will
        # revert -- an earlier version used std=1.0 here (larger than the 0.4
        # signal it was supposed to expose), which buried the one signal that
        # makes the paper's DAM rule ("go long if pdam < pvwap forecast")
        # correlate with anything, so v0's direction ended up ~uncorrelated
        # with the session's actual price path.
        auction_mid = base + walk[0] + np.random.normal(0, 0.1)
        half_spread = np.random.uniform(1.0, 3.0)
        sell_steps = np.cumsum(np.random.uniform(0.5, 2.0, N_AUCTION_LEVELS))
        sell_prices = (auction_mid + half_spread + sell_steps).round(2)
        sell_qtys = np.random.uniform(1.0, 50.0, N_AUCTION_LEVELS).round(2)
        buy_steps = np.cumsum(np.random.uniform(0.5, 2.0, N_AUCTION_LEVELS))
        buy_prices = (auction_mid - half_spread - buy_steps).round(2)
        buy_qtys = np.random.uniform(1.0, 50.0, N_AUCTION_LEVELS).round(2)
        for lvl in range(N_AUCTION_LEVELS):
            auction_rows.append((trading_open, delivery_start, "sell",
                                  sell_prices[lvl], sell_qtys[lvl], lvl + 1))
            auction_rows.append((trading_open, delivery_start, "buy",
                                  buy_prices[lvl], buy_qtys[lvl], lvl + 1))

        # ── CID: the same walk, sampled every tick through the session ─────
        n_ticks = idx_close  # ticks at minute offsets 0 .. idx_close-1
        if n_ticks > 0:
            ticks = trading_open + pd.to_timedelta(np.arange(n_ticks), unit="min")
            cid_mid = base + walk[:n_ticks] + np.random.normal(0, 0.3, n_ticks)

            half_spread_t = np.random.uniform(0.5, 2.0, n_ticks)
            sell_steps_t = np.cumsum(np.random.uniform(0.1, 1.0, (n_ticks, N_CIM_LEVELS)), axis=1)
            sell_prices_t = (cid_mid[:, None] + half_spread_t[:, None] + sell_steps_t).round(2)
            sell_qtys_t = np.random.uniform(0.5, 25.0, (n_ticks, N_CIM_LEVELS)).round(2)
            buy_steps_t = np.cumsum(np.random.uniform(0.1, 1.0, (n_ticks, N_CIM_LEVELS)), axis=1)
            buy_prices_t = (cid_mid[:, None] - half_spread_t[:, None] - buy_steps_t).round(2)
            buy_qtys_t = np.random.uniform(0.5, 25.0, (n_ticks, N_CIM_LEVELS)).round(2)

            ts_rep = np.repeat(ticks, N_CIM_LEVELS)
            n_half = n_ticks * N_CIM_LEVELS
            cim_dfs.append(pd.DataFrame({
                "timestamp": np.concatenate([ts_rep, ts_rep]),
                "delivery_start": delivery_start,
                "delivery_end": delivery_end,
                "side": np.concatenate([np.full(n_half, "sell"), np.full(n_half, "buy")]),
                "price_eur_mwh": np.concatenate([sell_prices_t.ravel(), buy_prices_t.ravel()]),
                "quantity_mwh": np.concatenate([sell_qtys_t.ravel(), buy_qtys_t.ravel()]),
            }))

        # ── BAL: the walk's value at the moment of delivery, plus asymmetric
        #    imbalance premium/discount (take > reference > feed) ──────────
        # Same reasoning as auction_mid above: keep idiosyncratic noise small
        # relative to _STATIONARY_STD so causal BAL-BENCH trailing averages
        # (src/balancing.py's bal_bench, used by the Eq. 10 clone rule) can
        # actually track the underlying process instead of being swamped by
        # noise unrelated to it.
        bal_ref = base + walk[idx_delivery] + np.random.normal(0, 0.1)
        for q in range(N_QUARTERS):
            q_ref = bal_ref + np.random.normal(0, 1.0)
            take_premium = np.random.uniform(2.0, 15.0)
            feed_discount = np.random.uniform(2.0, 15.0)
            bal_rows.append((delivery_start, q, round(q_ref + take_premium, 2),
                              round(q_ref - feed_discount, 2)))

    cim_df = pd.concat(cim_dfs, ignore_index=True) if cim_dfs else pd.DataFrame(
        columns=["timestamp", "delivery_start", "delivery_end", "side", "price_eur_mwh", "quantity_mwh"]
    )
    return auction_rows, cim_df, bal_rows


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

def _check_spread(df: pd.DataFrame, group_cols: list[str], label: str) -> None:
    sell = df[df["side"] == "sell"].groupby(group_cols)["price_eur_mwh"].min().rename("min_sell")
    buy = df[df["side"] == "buy"].groupby(group_cols)["price_eur_mwh"].max().rename("max_buy")
    merged = pd.concat([sell, buy], axis=1).dropna()
    violations = merged[merged["min_sell"] <= merged["max_buy"]]
    if not violations.empty:
        raise ValueError(f"[{label}] Spread violation in {len(violations)} group(s):\n{violations.head(5)}")


def _is_dst_transition_day(day: pd.Timestamp) -> bool:
    """True if the day has a DST gap or fold (wall-clock day != 24 hours)."""
    midnight_start = day.normalize()
    midnight_end = midnight_start + pd.DateOffset(days=1)
    diff_s = (midnight_end.tz_convert("UTC") - midnight_start.tz_convert("UTC")).total_seconds()
    return diff_s != 86400


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=sorted(SPLIT_CONFIG), required=True)
    args = parser.parse_args()

    cfg = SPLIT_CONFIG[args.split]
    np.random.seed(cfg["seed"])

    out_dir = _PROJECT_ROOT / "data" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    cim_file = out_dir / "cim_order_book.csv"
    auction_file = out_dir / "intraday_auction_curves.csv"
    bal_file = out_dir / "balancing_prices.csv"

    all_days = pd.date_range(cfg["start_date"], periods=cfg["n_days"], freq="D", tz=TIMEZONE)
    delivery_days = pd.DatetimeIndex([d for d in all_days if not _is_dst_transition_day(d)])
    skipped = len(all_days) - len(delivery_days)
    if skipped:
        print(f"Skipped {skipped} DST transition day(s): "
              f"{[str(d.date()) for d in all_days if _is_dst_transition_day(d)]}")

    t0 = time.time()
    auction_rows, cim_dfs, bal_rows = [], [], []
    for delivery_day in delivery_days:
        day_auction, day_cim, day_bal = _generate_day(delivery_day)
        auction_rows.extend(day_auction)
        cim_dfs.append(day_cim)
        bal_rows.extend(day_bal)

    auction_df = pd.DataFrame(auction_rows, columns=[
        "auction_time", "delivery_start", "side", "price_eur_mwh", "quantity_mwh", "level",
    ])
    cim_df = pd.concat(cim_dfs, ignore_index=True)
    cim_df.insert(0, "order_id", cim_df.index + 1)
    cim_df = cim_df[["order_id", "timestamp", "delivery_start", "delivery_end",
                      "side", "price_eur_mwh", "quantity_mwh"]]
    bal_df = pd.DataFrame(bal_rows, columns=[
        "delivery_start", "quarter_index", "take_price_eur_mwh", "feed_price_eur_mwh",
    ])

    _check_spread(auction_df, ["delivery_start"], "Auction")
    _check_spread(cim_df, ["timestamp", "delivery_start"], "CIM")
    assert (bal_df["take_price_eur_mwh"] > bal_df["feed_price_eur_mwh"]).all(), \
        "take price must exceed feed price at every quarter (imbalance asymmetry)"

    auction_df.to_csv(auction_file, index=False)
    cim_df.to_csv(cim_file, index=False)
    bal_df.to_csv(bal_file, index=False)
    elapsed = time.time() - t0

    print(f"Output files:\n  {cim_file}\n  {auction_file}\n  {bal_file}")
    print(f"Row counts:\n  CIM     : {len(cim_df):>10,}\n  Auction : {len(auction_df):>10,}\n  Balancing: {len(bal_df):>9,}")
    print(f"Date range: {cim_df['delivery_start'].min()}  ->  {cim_df['delivery_start'].max()}")
    print(f"Validation: PASSED ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
