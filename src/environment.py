"""
Single-contract continuous intraday market (CID) trading environment.

One episode = one hourly delivery contract's CID trading session (paper §4.2:
t in [1, T], T = last revision of the order book before delivery). This is a
deliberate departure from reinforce_threshold_policy's MultiHourMarketEnv (one
episode = a full 24-hour day, all hours traded jointly) -- Demir et al. define
the RL problem per-contract, with a position and PnL brought forward from a
single DAM decision (src/dam_policy.py).

Action space (paper §4.3, Eq. 4): {BUY, SELL, HOLD}, discrete, masked so that
BUY is only offered while `vt < vmax` and cumulative bought quantity < qhigh,
and SELL only while `vt > vmin` and cumulative sold quantity < qhigh. HOLD is
always available.

Trade execution: a chosen BUY/SELL fills against the single best price level
of the order book (paper §4.3: "the agent buys qa_t <= vmax - vt MWh"), capped
by remaining position headroom and remaining qhigh budget -- not a multi-level
depth walk.

Reward: src.rewards.{buy_reward,sell_reward,hold_reward} (Eqs. 5-7).

Terminal settlement (BAL proxy, see README "Scope & simplifications"): any
leftover position vT is settled at the last available best bid (if long,
vT > 0) or best ask (if short, vT < 0) in the contract's own order book, in
place of a real balancing-market clearing price we don't have data for.

State: assembled per Table 1 (paper §4.5); see `FEATURE_NAMES` for the exact
order. A few Table 1 rows reference genuine forecasts (pfeed/ptake, "average pb
forecast") that require data we don't have -- these are proxied by the causal
VWAP-BENCH value already computed for the DAM stage (src/dam_policy.py), which
is documented there and reused here without recomputation (passed in via
`vwap_hat` / `neighbor_bench`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.rewards import buy_reward, hold_reward, sell_reward

HOLD, BUY, SELL = 0, 1, 2
N_ACTIONS = 3
_LAG_STEPS = 8

# Domain-scale constants for min-max-style state normalisation (paper §5.4.1
# calls for min-max scaling but doesn't publish exact bounds; we use fixed,
# documented scales sized to this dataset rather than a full dataset-wide
# min/max pre-pass).
PRICE_SCALE = 100.0   # EUR/MWh -- typical intraday price spread magnitude
QTY_SCALE = 20.0      # MWh -- typical per-level order-book quantity

FEATURE_NAMES = [
    "minutes_to_end",
    "spread_bid_ask",
    "spread_mid_vs_vwap_hat",
    "spread_bid_vs_pdam",
    "spread_bid_vs_pb_forecast",
    "spread_bid_vs_pfeed_forecast",
    "spread_bid_vs_neighbor_bench",
    *[f"lag_bid_spread_t-{k}" for k in range(1, _LAG_STEPS + 1)],
    "best_bid_qty",
    "n_bid_orders",
    "q3_cum_bid_qty",
    "best_ask_price",
    "q1_ask_price",
    "q2_ask_price",
    "q3_ask_price",
    "spread_ask_vs_neighbor_bench",
    "best_ask_qty",
    "total_ask_qty",
    "n_ask_orders",
    "rule_buy",
    "rule_sell",
    "rule_hold",
    "scaled_bought_qty",
    "scaled_sold_qty",
    "scaled_position",
    "scaled_pnl",
]
STATE_DIM = len(FEATURE_NAMES)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class _Book:
    bids: list[tuple[float, float]]  # [(price, qty), ...] desc by price
    asks: list[tuple[float, float]]  # [(price, qty), ...] asc by price


@dataclass
class StepInfo:
    action: int
    traded_qty: float
    cash_flow: float
    reward: float
    position: float
    pnl: float


class ContractCIDEnv:
    """Episode = one delivery-hour contract's CID trading session."""

    def __init__(
        self,
        vmax: float = 10.0,
        vmin: float = -10.0,
        qhigh: float = 50.0,
        pnl_low: float = -5000.0,
        pnl_high: float = 10000.0,
    ) -> None:
        self.vmax = vmax
        self.vmin = vmin
        self.qhigh = qhigh
        self.pnl_low = pnl_low
        self.pnl_high = pnl_high

        self._ticks: list[pd.Timestamp] = []
        self._books: dict[pd.Timestamp, _Book] = {}
        self._tick_idx = 0
        self._session_start: pd.Timestamp | None = None
        self._session_end: pd.Timestamp | None = None

    # ── Setup ────────────────────────────────────────────────────────────

    def reset(
        self,
        cim_contract: pd.DataFrame,
        pdam: float,
        v0: float,
        c_dam: float,
        vwap_hat: float | None = None,
        neighbor_bench: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Begin a new contract episode.

        Args:
            cim_contract   : all CIM rows for a single delivery_start.
            pdam           : DAM price proxy for this contract (src.dam_policy).
            v0             : opened DAM position (MWh, +long/-short).
            c_dam          : DAM cash flow for v0 (Eq. 1).
            vwap_hat       : VWAP-BENCH forecast used for the DAM rule; reused
                             here as the "average forecast" state features'
                             proxy. Falls back to pdam if None.
            neighbor_bench : trailing neighbouring-hour price bench, proxy for
                             the paper's rolling ptake feature. Falls back to
                             pdam if None.
        """
        self._ticks, self._books = self._build_books(cim_contract)
        if not self._ticks:
            raise ValueError("Contract has no order-book ticks.")
        self._tick_idx = 0
        self._session_start = self._ticks[0]
        self._session_end = self._ticks[-1]

        self.pdam = pdam
        self.vwap_hat = vwap_hat if vwap_hat is not None else pdam
        self.neighbor_bench = neighbor_bench if neighbor_bench is not None else pdam

        self.vt = v0
        self.pnl = c_dam
        self.cum_bought = 0.0
        self.cum_sold = 0.0
        self._bid_history: deque[float] = deque(maxlen=_LAG_STEPS)
        self._last_best_bid = pdam
        self._last_best_ask = pdam

        best_bid_series, best_ask_series = [], []
        for ts in self._ticks:
            book = self._books[ts]
            best_bid_series.append(book.bids[0][0] if book.bids else np.nan)
            best_ask_series.append(book.asks[0][0] if book.asks else np.nan)
        bb = pd.Series(best_bid_series).ffill().bfill()
        ba = pd.Series(best_ask_series).ffill().bfill()
        self.pb_low, self.pb_high = float(bb.min()), float(bb.max())
        self.pa_low, self.pa_high = float(ba.min()), float(ba.max())
        self.filled_best_bid = bb.tolist()
        self.filled_best_ask = ba.tolist()

        state = self._state()
        mask = self._action_mask()
        return state, mask

    # ── Per-step interface ───────────────────────────────────────────────

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, float, bool, StepInfo]:
        pa_t = self.filled_best_ask[self._tick_idx]
        pb_t = self.filled_best_bid[self._tick_idx]
        book = self._books[self._ticks[self._tick_idx]]

        r_buy = buy_reward(pa_t, self.pdam, self.pb_high, self.pa_high, self.pa_low)
        r_sell = sell_reward(pb_t, self.pdam, self.pa_low, self.pb_high, self.pb_low)

        traded_qty, cash_flow = 0.0, 0.0
        if action == BUY:
            qa_t = book.asks[0][1] if book.asks else 0.0
            q = max(0.0, min(qa_t, self.vmax - self.vt, self.qhigh - self.cum_bought))
            if q > 0.0:
                cash_flow = -q * pa_t
                self.vt += q
                self.cum_bought += q
                self.pnl += cash_flow
                traded_qty = q
            reward = r_buy
        elif action == SELL:
            qb_t = book.bids[0][1] if book.bids else 0.0
            q = max(0.0, min(qb_t, self.vt - self.vmin, self.qhigh - self.cum_sold))
            if q > 0.0:
                cash_flow = q * pb_t
                self.vt -= q
                self.cum_sold += q
                self.pnl += cash_flow
                traded_qty = q
            reward = r_sell
        else:
            reward = hold_reward(r_buy, r_sell)

        self._bid_history.append(pb_t)
        self._last_best_bid, self._last_best_ask = pb_t, pa_t

        self._tick_idx += 1
        done = self._tick_idx >= len(self._ticks)

        info = StepInfo(
            action=action, traded_qty=traded_qty, cash_flow=cash_flow,
            reward=reward, position=self.vt, pnl=self.pnl,
        )

        if done:
            leftover = self.vt
            if leftover > 0.0:
                bal_cash = leftover * self._last_best_bid
            elif leftover < 0.0:
                bal_cash = leftover * self._last_best_ask
            else:
                bal_cash = 0.0
            self.pnl += bal_cash
            info.pnl = self.pnl
            next_state = np.zeros(STATE_DIM, dtype=np.float32)
            next_mask = np.array([False, False, True], dtype=bool)
        else:
            next_state = self._state()
            next_mask = self._action_mask()

        return next_state, next_mask, reward, done, info

    def rule_context(self) -> dict:
        """
        Snapshot of values needed by the Eq. (8)-(11) behaviour-cloning rules
        (src.a2c_trainer), read *before* calling step() for the current tick.
        """
        i = self._tick_idx
        return {
            "pa_t": self.filled_best_ask[i],
            "pb_t": self.filled_best_bid[i],
            "pdam": self.pdam,
            "pb_high": self.pb_high,
            "pa_low": self.pa_low,
            "vwap_hat": self.vwap_hat,
            "vt": self.vt,
            "cum_bought": self.cum_bought,
            "cum_sold": self.cum_sold,
            "vmax": self.vmax,
            "vmin": self.vmin,
            "qhigh": self.qhigh,
        }

    # ── Internals ────────────────────────────────────────────────────────

    def _action_mask(self) -> np.ndarray:
        can_buy = (self.vt < self.vmax) and (self.cum_bought < self.qhigh)
        can_sell = (self.vt > self.vmin) and (self.cum_sold < self.qhigh)
        return np.array([True, can_buy, can_sell], dtype=bool)  # [HOLD, BUY, SELL]

    def _state(self) -> np.ndarray:
        i = self._tick_idx
        ts = self._ticks[i]
        book = self._books[ts]
        pa_t = self.filled_best_ask[i]
        pb_t = self.filled_best_bid[i]

        minutes_to_end = (self._session_end - ts).total_seconds() / 60.0
        session_minutes = max(
            1.0, (self._session_end - self._session_start).total_seconds() / 60.0
        )

        lags = list(self._bid_history)
        lags = [pb_t] * (_LAG_STEPS - len(lags)) + lags  # pad with current value
        lag_spreads = [_clip((pb_t - lag) / PRICE_SCALE) for lag in lags[-_LAG_STEPS:]]

        ask_prices = [p for p, _ in book.asks] or [pa_t]
        bid_qtys_cum = np.cumsum([q for _, q in book.bids]) if book.bids else np.array([0.0])
        n_bid_orders = float(len(book.bids))
        n_ask_orders = float(len(book.asks))
        best_bid_qty = book.bids[0][1] if book.bids else 0.0
        best_ask_qty = book.asks[0][1] if book.asks else 0.0
        total_ask_qty = float(sum(q for _, q in book.asks))

        rule = 0  # HOLD-rule
        if pa_t < self.pdam:
            rule = 1  # BUY-rule
        elif pb_t > self.pdam:
            rule = 2  # SELL-rule

        features = [
            _clip(minutes_to_end / session_minutes, 0.0, 1.0),
            _clip((pa_t - pb_t) / PRICE_SCALE, 0.0, 1.0),
            _clip(((pa_t + pb_t) / 2 - self.vwap_hat) / PRICE_SCALE),
            _clip((pb_t - self.pdam) / PRICE_SCALE),
            _clip((pb_t - self.vwap_hat) / PRICE_SCALE),
            _clip((pb_t - self.vwap_hat) / PRICE_SCALE),
            _clip((pb_t - self.neighbor_bench) / PRICE_SCALE),
            *lag_spreads,
            _clip(best_bid_qty / QTY_SCALE, 0.0, 1.0),
            _clip(n_bid_orders / 10.0, 0.0, 1.0),
            _clip(float(np.quantile(bid_qtys_cum, 0.75)) / (QTY_SCALE * 5), 0.0, 1.0),
            _clip(pa_t / PRICE_SCALE, 0.0, 1.0),
            _clip(float(np.quantile(ask_prices, 0.25)) / PRICE_SCALE, 0.0, 1.0),
            _clip(float(np.quantile(ask_prices, 0.50)) / PRICE_SCALE, 0.0, 1.0),
            _clip(float(np.quantile(ask_prices, 0.75)) / PRICE_SCALE, 0.0, 1.0),
            _clip((pa_t - self.neighbor_bench) / PRICE_SCALE),
            _clip(best_ask_qty / QTY_SCALE, 0.0, 1.0),
            _clip(total_ask_qty / (QTY_SCALE * 5), 0.0, 1.0),
            _clip(n_ask_orders / 10.0, 0.0, 1.0),
            1.0 if rule == 1 else 0.0,
            1.0 if rule == 2 else 0.0,
            1.0 if rule == 0 else 0.0,
            _clip(self.cum_bought / self.qhigh, 0.0, 1.0),
            _clip(self.cum_sold / self.qhigh, 0.0, 1.0),
            _clip((self.vt - self.vmin) / (self.vmax - self.vmin), 0.0, 1.0),
            _clip((self.pnl - self.pnl_low) / (self.pnl_high - self.pnl_low), 0.0, 1.0),
        ]
        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def _build_books(cim_contract: pd.DataFrame) -> tuple[list, dict]:
        bid_df = (
            cim_contract[cim_contract["side"] == "buy"]
            .sort_values(["timestamp", "price_eur_mwh"], ascending=[True, False])
        )
        ask_df = (
            cim_contract[cim_contract["side"] == "sell"]
            .sort_values(["timestamp", "price_eur_mwh"], ascending=[True, True])
        )
        bid_agg = bid_df.groupby("timestamp")[["price_eur_mwh", "quantity_mwh"]].agg(list)
        ask_agg = ask_df.groupby("timestamp")[["price_eur_mwh", "quantity_mwh"]].agg(list)

        bid_book = {ts: list(zip(p, q)) for ts, (p, q) in zip(bid_agg.index, bid_agg.values)}
        ask_book = {ts: list(zip(p, q)) for ts, (p, q) in zip(ask_agg.index, ask_agg.values)}

        ticks = sorted(set(bid_book) | set(ask_book))
        books = {
            ts: _Book(bids=bid_book.get(ts, []), asks=ask_book.get(ts, []))
            for ts in ticks
        }
        return ticks, books
