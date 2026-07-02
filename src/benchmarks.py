"""
Rule-based baseline comparators (paper §5.4.4), gated into evaluate.py behind
CLI flags the same way reinforce_threshold_policy gates its RI-LP benchmark.

HOLD  : no CID trades at all; the DAM position is settled entirely via the
        BAL-proxy terminal settlement already built into ContractCIDEnv.step()
        (paper: "closes out all open DAM positions on the BAL").
PRE_BA: Eq. (14) -- trades against a trailing 30-tick average of the best
        ask/bid, highlighting the risk of trading too frequently on the CID.
"""

from __future__ import annotations

from src.environment import BUY, HOLD, SELL, ContractCIDEnv


def run_hold(env: ContractCIDEnv) -> float:
    """Play an already-reset episode taking HOLD every tick; return final PnL."""
    done = False
    info = None
    while not done:
        _, _, _, done, info = env.step(HOLD)
    return info.pnl


def run_pre_ba(env: ContractCIDEnv, state, mask, window: int = 30) -> float:
    """
    Eq. (14): buy if pa_t < trailing-window average ask, sell if pb_t >
    trailing-window average bid, else hold. `env` must be freshly reset (not
    yet stepped); `state`/`mask` are its reset() return values.
    """
    i = 0
    done = False
    info = None
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
        i += 1
    return info.pnl
