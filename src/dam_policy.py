"""
Rule-based day-ahead market (DAM) position opener.

Implements the DAM stage of Demir, Kok & Paterakis (2023) (paper Section 3 / Eq.
in Section 5.3): a long DAM position vmax is opened if the DAM price forecast is
below the CID volume-weighted-average-price (vwap) forecast; otherwise a short
position vmin is opened.

Scope simplification (see README "Scope & simplifications"):
    The paper trains dedicated neural forecasting models (2NN/2CNN_NN with
    autoencoder/VAE/GAN augmentation for pdam; LASSO/RF/GB/DNN ensembles for
    pvwap) against real ENTSO-E / Scholt Energy data. We don't have that real
    data or a reason to retrain those forecasters against synthetic data, so:

      - pdam is proxied by the D-1 auction MID price for the delivery hour
        (src.data_loader.day_auction_mids), which is exactly the auction-based
        day-ahead-style reference price already present in our dataset.

      - pvwap is proxied using the paper's *own* baseline formula, VWAP-BENCH
        (Eq. 13): a causal trailing average of the realized CID price for the
        same delivery hour on previous days. Our synthetic CIM data records
        resting order-book quotes rather than an executed trade tape, so
        "realized CID price" is itself proxied by the time-averaged order-book
        mid-price ((best_bid + best_ask) / 2) across a session -- see
        `precompute_session_mid_vwap`.
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import day_auction_mids


def pdam_series(auc: pd.DataFrame, delivery_starts: list):
    """Thin wrapper around data_loader.day_auction_mids -- the pdam proxy."""
    return day_auction_mids(auc, delivery_starts)


def precompute_session_mid_vwap(cim: pd.DataFrame) -> pd.Series:
    """
    Compute a VWAP-BENCH-compatible "realized CID price" per delivery hour.

    For each delivery_start, this is the time-average of the order-book
    mid-price ((best_ask + best_bid) / 2) across all ticks of that contract's
    CID session -- a proxy for the volume-weighted average trade price, since
    our synthetic data records resting quotes rather than an executed trade
    tape.

    This is a single vectorised pass over the whole split (train or test);
    call it once at startup and reuse the returned Series.

    Returns:
        pd.Series indexed by delivery_start (UTC timestamp) -> float session mid-vwap.
    """
    sell_best = (
        cim[cim["side"] == "sell"]
        .groupby(["timestamp", "delivery_start"])["price_eur_mwh"]
        .min()
    )
    buy_best = (
        cim[cim["side"] == "buy"]
        .groupby(["timestamp", "delivery_start"])["price_eur_mwh"]
        .max()
    )
    mid = ((sell_best + buy_best) / 2).dropna()
    return mid.groupby(level="delivery_start").mean()


def vwap_bench(
    session_vwap: pd.Series,
    delivery_start: pd.Timestamp,
    lookback_days: int = 2,
) -> float | None:
    """
    Eq. (13): causal trailing average of realized CID price for the same
    delivery hour on the previous `lookback_days` calendar days.

    Only uses days strictly before `delivery_start` (no lookahead). Returns
    None if none of the lookback days are available (e.g. start of the split),
    in which case callers should fall back to a HOLD-equivalent direction.
    """
    values = []
    for d in range(1, lookback_days + 1):
        prior = delivery_start - pd.Timedelta(days=d)
        if prior in session_vwap.index:
            values.append(session_vwap.loc[prior])
    if not values:
        return None
    return float(sum(values) / len(values))


def open_dam_position(
    pdam: float,
    pvwap_hat: float | None,
    vmax: float,
    vmin: float,
) -> tuple[float, float]:
    """
    Rule-based DAM position opener (paper Section 5.3).

    Long vmax if pdam < pvwap_hat, else short vmin. If pvwap_hat is unavailable
    (no lookback data), default to a short position of 0 (HOLD-equivalent --
    no DAM position opened, no CID trading incentive either way).

    Returns:
        (v0, c_dam) -- opened position (MWh, +long/-short) and DAM cash flow
        (Eq. 1: c_dam = -v0 * pdam).
    """
    if pvwap_hat is None:
        return 0.0, 0.0
    v0 = vmax if pdam < pvwap_hat else vmin
    c_dam = -v0 * pdam
    return v0, c_dam
