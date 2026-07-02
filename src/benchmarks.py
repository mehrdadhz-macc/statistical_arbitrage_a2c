"""
Rule-based baseline comparators (paper §5.4.4), gated into evaluate.py behind
CLI flags the same way reinforce_threshold_policy gates its RI-LP benchmark.

HOLD  : no CID trades at all; the DAM position is settled entirely via the
        BAL-proxy terminal settlement already built into ContractCIDEnv.step()
        (paper: "closes out all open DAM positions on the BAL").
PRE_BA: Eq. (14) -- trades against a trailing 30-tick average of the best
        ask/bid, highlighting the risk of trading too frequently on the CID.

Both return (pnl, quantity). `quantity` matches paper Table 7's convention,
worked out from its own numbers: HOLD's reported quantity is exactly `vmax`
per contract (17600.00 / 1760 contracts = 10.00 = vmax) despite HOLD making
*zero* CID trades -- so "Quantity" = |v0| (the DAM leg, always a full vmax/
vmin position) + gross CID volume (sum of qa_t + qb_t), not CID volume alone.
"""

from __future__ import annotations

from src.environment import BUY, HOLD, SELL, ContractCIDEnv


def run_hold(env: ContractCIDEnv) -> tuple[float, float]:
    """Play an already-reset episode taking HOLD every tick; return (pnl, quantity)."""
    done = False
    info = None
    while not done:
        _, _, _, done, info = env.step(HOLD)
    return info.pnl, abs(env.v0)


def run_pre_ba(env: ContractCIDEnv, state, mask, window: int = 30) -> tuple[float, float]:
    """
    Eq. (14): buy if pa_t < trailing-window average ask, sell if pb_t >
    trailing-window average bid, else hold. `env` must be freshly reset (not
    yet stepped); `state`/`mask` are its reset() return values. Returns
    (pnl, quantity).
    """
    i = 0
    done = False
    info = None
    traded_qty = 0.0
    while not done:
        pa_t = env.filled_best_ask[i]
        pb_t = env.filled_best_bid[i]
        lo = max(0, i - window + 1)
        avg_ask = sum(env.filled_best_ask[lo:i + 1]) / (i - lo + 1)
        avg_bid = sum(env.filled_best_bid[lo:i + 1]) / (i - lo + 1)

        if pa_t < avg_ask and mask[BUY]:
            action = BUY
        elif pb_t > avg_bid and mask[SELL]:
            action = SELL
        else:
            action = HOLD

        state, mask, _, done, info = env.step(action)
        traded_qty += abs(info.traded_qty)
        i += 1
    return info.pnl, abs(env.v0) + traded_qty
