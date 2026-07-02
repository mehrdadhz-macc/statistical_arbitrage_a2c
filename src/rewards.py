"""
Buy/sell/hold reward functions for continuous intraday market (CID) trading.

Implements Eqs. (5)-(7) of Demir, Kok & Paterakis (2023), "Statistical arbitrage
trading across electricity markets using advantage actor-critic methods".

Notation (paper -> code):
    pa_t            -> pa_t            best ask price at time t
    pb_t            -> pb_t            best bid price at time t
    pdam            -> pdam            day-ahead price (proxy, see src/dam_policy.py)
    pa_high/pa_low  -> pa_high/pa_low  highest/lowest best-ask price across the session
    pb_high/pb_low  -> pb_high/pb_low  highest/lowest best-bid price across the session
    tau_B, tau_S    -> threshold arguments to _f_buy/_f_sell

Both rB_t and rS_t are intrinsically bounded to [-2, 0]; rH_t to [-1, 0].
"""

from __future__ import annotations

_EPS = 1e-9


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Ratio with a numerical guard against a degenerate (zero-width) price range."""
    if abs(denominator) < _EPS:
        return 0.0
    return numerator / denominator


def _f_buy(tau_b: float, pa_t: float, pa_high: float, pa_low: float) -> float:
    """Eq. (5) inner function f^B(tau^B, pa_t)."""
    if pa_t > tau_b:
        return -1.0 - _safe_ratio(pa_t - tau_b, pa_high - tau_b)
    return -_safe_ratio(pa_t - pa_low, tau_b - pa_low)


def _f_sell(tau_s: float, pb_t: float, pb_high: float, pb_low: float) -> float:
    """Eq. (6) inner function f^S(tau^S, pb_t)."""
    if pb_t < tau_s:
        return -2.0 + _safe_ratio(pb_t - pb_low, tau_s - pb_low)
    return -1.0 + _safe_ratio(pb_t - tau_s, pb_high - tau_s)


def buy_reward(
    pa_t: float,
    pdam: float,
    pb_high: float,
    pa_high: float,
    pa_low: float,
) -> float:
    """
    Eq. (5): rB_t = 1/2 * (f^B(pdam, pa_t) + f^B(pb_high, pa_t)), bounded to [-2, 0].

    pdam separates gains/losses between DAM and CID; pb_high separates gains/losses
    within the CID.
    """
    r = 0.5 * (
        _f_buy(pdam, pa_t, pa_high, pa_low)
        + _f_buy(pb_high, pa_t, pa_high, pa_low)
    )
    return max(-2.0, min(0.0, r))


def sell_reward(
    pb_t: float,
    pdam: float,
    pa_low: float,
    pb_high: float,
    pb_low: float,
) -> float:
    """
    Eq. (6): rS_t = 1/2 * (f^S(pdam, pb_t) + f^S(pa_low, pb_t)), bounded to [-2, 0].

    pdam separates gains/losses between DAM and CID; pa_low separates gains/losses
    within the CID.
    """
    r = 0.5 * (
        _f_sell(pdam, pb_t, pb_high, pb_low)
        + _f_sell(pa_low, pb_t, pb_high, pb_low)
    )
    return max(-2.0, min(0.0, r))


def hold_reward(r_buy: float, r_sell: float) -> float:
    """
    Eq. (7): opportunity cost of not trading.

    rH_t = 0                        if rB_t < -1 and rS_t < -1  (both trades a loss)
    rH_t = -1 - max(rB_t, rS_t)     otherwise                    (in [-1, 0))
    """
    if r_buy < -1.0 and r_sell < -1.0:
        return 0.0
    return -1.0 - max(r_buy, r_sell)
