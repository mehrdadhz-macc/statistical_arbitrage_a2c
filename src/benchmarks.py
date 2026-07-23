"""
Rule-based baseline comparators (paper §5.4.4), gated into evaluate.py behind
CLI flags the same way reinforce_threshold_policy gates its RI-LP benchmark.

HOLD  : no CID trades at all; the DAM position is settled entirely via the
        BAL-proxy terminal settlement already built into ContractCIDEnv.step()
        (paper: "closes out all open DAM positions on the BAL").
PRE_BA: Eq. (14) -- trades against a trailing 30-tick average of the *previous*
        best ask/bid (paper §5.4.4: "PRE-BA uses the previous best bid and
        best ask prices to place trades"), highlighting the risk of trading
        too frequently on the CID.

Both return (pnl, quantity). `quantity` matches paper Table 7's convention,
worked out from its own numbers: HOLD's reported quantity is exactly `vmax`
per contract (17600.00 / 1760 contracts = 10.00 = vmax) despite HOLD making
*zero* CID trades -- so "Quantity" = |v0| (the DAM leg, always a full vmax/
vmin position) + CID quantity. For the CID leg we use the paper's own Eq. (2.2)
definition of "total arbitraged quantity", min(total bought, total sold), not
the gross sum of both sides -- the paper explicitly names this quantity
(Section 2.2 / Nomenclature's Sigma-q) as `min{sum(qa_t), sum(qb_t)}`.
"""

from __future__ import annotations

from src.environment import BUY, HOLD, SELL, ContractCIDEnv


def run_hold(env: ContractCIDEnv) -> tuple[float, float]:
    """Play an already-reset episode taking HOLD every tick; return (pnl, quantity)."""
    done = False
    info = None
    while not done:
        _, _, _, done, info = env.step(HOLD)
    return info.pnl, abs(env.v0) + min(env.cum_bought, env.cum_sold)


def run_pre_ba(env: ContractCIDEnv, state, mask, window: int = 30) -> tuple[float, float]:
    """
    Eq. (14): buy if pa_t < trailing-window average of the *previous* asks,
    sell if pb_t > trailing-window average of the *previous* bids, else hold.
    The averaging window looks strictly backwards from t (excludes pa_t/pb_t
    itself) -- consistent with the paper's framing of PRE-BA as trading
    against "the previous best bid and best ask prices", rather than a
    self-referential average that includes the current tick. At the first
    tick, with no history yet, the window falls back to the current price
    (so the comparison is a no-op and the agent holds).

    `env` must be freshly reset (not yet stepped); `state`/`mask` are its
    reset() return values. Returns (pnl, quantity).
    """
    i = 0
    done = False
    info = None
    while not done:
        pa_t = env.filled_best_ask[i]
        pb_t = env.filled_best_bid[i]
        lo = max(0, i - window)
        if i > lo:
            avg_ask = sum(env.filled_best_ask[lo:i]) / (i - lo)
            avg_bid = sum(env.filled_best_bid[lo:i]) / (i - lo)
        else:
            avg_ask, avg_bid = pa_t, pb_t

        if pa_t < avg_ask and mask[BUY]:
            action = BUY
        elif pb_t > avg_bid and mask[SELL]:
            action = SELL
        else:
            action = HOLD

        state, mask, _, done, info = env.step(action)
        i += 1
    return info.pnl, abs(env.v0) + min(env.cum_bought, env.cum_sold)
