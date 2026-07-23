"""
Balancing-market (BAL) data access -- the third market in Demir et al.
(2023)'s DAM-CID-BAL arbitrage chain (paper §2.2, Eq. 1's Cbal term).

Real settlement (used only at episode termination, see src/environment.py):
    ptake / pfeed for a contract are the average of its four quarter-hourly
    take/feed prices (scripts/data_generation/generate_synthetic_data.py).
    Since this is offline backtesting, the *current* contract's own realized
    ptake/pfeed are legitimately known upfront (same assumption already made
    for pa_low/pa_high/pb_low/pb_high in src/environment.py) and are used
    only in the terminal-settlement branch of ContractCIDEnv.step().

Causal BAL-BENCH forecasts (used mid-episode, for state features and the
Eq. 10 behaviour-cloning rule): a trading agent cannot see its own contract's
future settlement price while it's still trading, so anything used *before*
termination must only look at *other*, already-elapsed contracts --
`bal_bench` (paper Eq. 13-style trailing average) and
`rolling_neighbor_ptake` (paper Table 1's "1/6 sum" rolling feature).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_TS_COLS = ["delivery_start"]
_PROJECT_ROOT = Path(__file__).parent.parent
_VALID_SPLITS = {"train", "test"}


def load_split_balancing(split: str = "train") -> pd.DataFrame:
    """Load data/<split>/balancing_prices.csv (mirrors data_loader.load_all's path convention)."""
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}; got {split!r}")
    return load_balancing(_PROJECT_ROOT / "data" / split / "balancing_prices.csv")


def load_balancing(path) -> pd.DataFrame:
    """Load and validate the quarter-hourly BAL settlement price file."""
    df = pd.read_csv(path)
    required = {"delivery_start", "quarter_index", "take_price_eur_mwh", "feed_price_eur_mwh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Balancing file missing columns: {missing}")

    df["delivery_start"] = pd.to_datetime(df["delivery_start"], utc=True)
    if df[["take_price_eur_mwh", "feed_price_eur_mwh"]].isnull().any().any():
        raise ValueError("Balancing data contains null prices.")
    if (df["take_price_eur_mwh"] <= df["feed_price_eur_mwh"]).any():
        raise ValueError("Balancing data has take_price <= feed_price for some quarter.")

    return df.sort_values(["delivery_start", "quarter_index"]).reset_index(drop=True)


def precompute_hourly_bal(bal: pd.DataFrame) -> pd.DataFrame:
    """
    Hourly ptake/pfeed = average of the four quarter-hourly prices (paper
    Nomenclature §2.2). Returns a DataFrame indexed by delivery_start with
    columns ["ptake", "pfeed"].
    """
    hourly = bal.groupby("delivery_start")[["take_price_eur_mwh", "feed_price_eur_mwh"]].mean()
    return hourly.rename(columns={"take_price_eur_mwh": "ptake", "feed_price_eur_mwh": "pfeed"})


def contract_bal_prices(hourly_bal: pd.DataFrame, delivery_start: pd.Timestamp) -> tuple[float, float] | tuple[None, None]:
    """Real (ptake, pfeed) for one contract -- terminal settlement use only."""
    if delivery_start not in hourly_bal.index:
        return None, None
    row = hourly_bal.loc[delivery_start]
    return float(row["ptake"]), float(row["pfeed"])


def bal_bench(
    hourly_bal: pd.DataFrame,
    delivery_start: pd.Timestamp,
    lookback_days: int = 2,
) -> tuple[float | None, float | None]:
    """
    Causal trailing average of (ptake, pfeed) for the same delivery hour on
    the previous `lookback_days` calendar days (mirrors src.dam_policy's
    VWAP-BENCH, Eq. 13). Used for state features / Eq. 10 cloning, never for
    the actual settlement of the current contract.
    """
    takes, feeds = [], []
    for d in range(1, lookback_days + 1):
        prior = delivery_start - pd.Timedelta(days=d)
        if prior in hourly_bal.index:
            takes.append(hourly_bal.loc[prior, "ptake"])
            feeds.append(hourly_bal.loc[prior, "pfeed"])
    if not takes:
        return None, None
    return float(np.mean(takes)), float(np.mean(feeds))


def rolling_neighbor_ptake(
    hourly_bal: pd.DataFrame,
    delivery_start: pd.Timestamp,
    day_lookback: int = 3,
    hour_window: int = 2,
) -> float | None:
    """
    Causal trailing average of ptake over the previous `day_lookback` days
    and neighbouring delivery hours [h - hour_window, h] (paper Table 1:
    "spread between pb_t and 1/6 sum_{d-3}^{d-1} sum_{h-2}^{h} ptake" --
    we average over however many of the (day_lookback x (hour_window+1))
    terms are actually available rather than hard-coding the paper's "1/6",
    since the OCR'd term count doesn't divide evenly by 6).
    """
    values = []
    for d in range(1, day_lookback + 1):
        for hh in range(0, hour_window + 1):
            ts = delivery_start - pd.Timedelta(days=d) - pd.Timedelta(hours=hh)
            if ts in hourly_bal.index:
                values.append(hourly_bal.loc[ts, "ptake"])
    if not values:
        return None
    return float(np.mean(values))
