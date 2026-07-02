"""
Diagnostic evaluation plots, adapted from reinforce_threshold_policy's
src/eval_plots.py for the per-contract A2C setting. Six PNGs, all built from
plain per-contract `ContractResult` records (policy-agnostic) plus a handful
of recorded tick traces for a few example contracts.

Plots 2-3 are direct analogues of paper Figs. 9-11 (cumulative PnL across test
contracts; traded-quantity / PnL distribution by delivery hour).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ContractResult:
    delivery_start: pd.Timestamp
    hour: int
    reward: float
    pnl: float
    n_steps: int
    traded_qty: float
    hold_pnl: float | None = None
    hold_qty: float | None = None
    pre_ba_pnl: float | None = None
    pre_ba_qty: float | None = None
    action_counts: dict | None = None


def _to_frame(results: list[ContractResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results]).sort_values("delivery_start")


def plot_performance_distribution(results: list[ContractResult], out_path: Path) -> None:
    df = _to_frame(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["pnl"], bins=40, alpha=0.6, label="A2C")
    if df["hold_pnl"].notna().any():
        ax.hist(df["hold_pnl"].dropna(), bins=40, alpha=0.5, label="HOLD")
    if df["pre_ba_pnl"].notna().any():
        ax.hist(df["pre_ba_pnl"].dropna(), bins=40, alpha=0.5, label="PRE-BA")
    ax.set_xlabel("PnL (EUR)")
    ax.set_ylabel("contract count")
    ax.set_title("Performance distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cumulative_pnl(results: list[ContractResult], out_path: Path) -> None:
    df = _to_frame(results)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(len(df)), df["pnl"].cumsum().values, label="A2C")
    if df["hold_pnl"].notna().all():
        ax.plot(range(len(df)), df["hold_pnl"].cumsum().values, label="HOLD", linestyle="--")
    if df["pre_ba_pnl"].notna().all():
        ax.plot(range(len(df)), df["pre_ba_pnl"].cumsum().values, label="PRE-BA", linestyle=":")
    ax.set_xlabel("test contract index (chronological)")
    ax.set_ylabel("cumulative PnL (EUR)")
    ax.set_title("Cumulative PnL across test contracts (paper Fig. 9)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_traded_qty_by_hour(results: list[ContractResult], out_path: Path) -> None:
    df = _to_frame(results)
    hours = sorted(df["hour"].unique())
    by_hour = [df.loc[df["hour"] == h, "traded_qty"].values for h in hours]
    totals = [v.sum() for v in by_hour]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.boxplot(by_hour, positions=hours, widths=0.6)
    ax1.set_xlabel("delivery hour")
    ax1.set_ylabel("traded quantity per contract (MWh)")
    ax2 = ax1.twinx()
    ax2.bar(hours, totals, alpha=0.25, color="salmon", width=0.8)
    ax2.set_ylabel("total traded quantity (MWh)", color="salmon")
    ax1.set_title("Traded quantity by delivery hour (paper Fig. 10)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pnl_by_hour(results: list[ContractResult], out_path: Path) -> None:
    df = _to_frame(results)
    hours = sorted(df["hour"].unique())
    by_hour = [df.loc[df["hour"] == h, "pnl"].values for h in hours]
    totals = [v.sum() for v in by_hour]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.boxplot(by_hour, positions=hours, widths=0.6)
    ax1.axhline(0.0, color="grey", linewidth=0.8)
    ax1.set_xlabel("delivery hour")
    ax1.set_ylabel("PnL per contract (EUR)")
    ax2 = ax1.twinx()
    ax2.bar(hours, totals, alpha=0.25, color="salmon", width=0.8)
    ax2.set_ylabel("total PnL (EUR)", color="salmon")
    ax1.set_title("PnL by delivery hour (paper Fig. 11)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_example_contracts(examples: dict[str, list[dict]], out_path: Path) -> None:
    n = len(examples)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), squeeze=False)
    for row, (label, ticks) in enumerate(examples.items()):
        ax = axes[row][0]
        t = [rec["t"] for rec in ticks]
        pa = [rec["pa_t"] for rec in ticks]
        pb = [rec["pb_t"] for rec in ticks]
        actions = [rec["action"] for rec in ticks]
        ax.plot(t, pa, label="best ask", color="C0", linewidth=1)
        ax.plot(t, pb, label="best bid", color="C1", linewidth=1)
        for action, color, name in ((1, "green", "BUY"), (2, "red", "SELL")):
            idx = [i for i, a in enumerate(actions) if a == action]
            if idx:
                ax.scatter([t[i] for i in idx], [pb[i] if action == 2 else pa[i] for i in idx],
                           color=color, s=18, zorder=5, label=name)
        ax.set_title(label)
        ax.set_xlabel("tick")
        ax.set_ylabel("EUR/MWh")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_action_distribution(results: list[ContractResult], out_path: Path) -> None:
    totals = {"HOLD": 0, "BUY": 0, "SELL": 0}
    for r in results:
        if r.action_counts:
            for k, v in r.action_counts.items():
                totals[k] += v
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(list(totals.keys()), list(totals.values()), color=["grey", "green", "red"])
    ax.set_ylabel("total actions taken across test set")
    ax.set_title("Action distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_all_plots(
    results: list[ContractResult],
    examples: dict[str, list[dict]],
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_performance_distribution(results, out_dir / "fig_test_1_performance_dist.png")
    plot_cumulative_pnl(results, out_dir / "fig_test_2_cumulative_pnl.png")
    plot_traded_qty_by_hour(results, out_dir / "fig_test_3_traded_qty_by_hour.png")
    plot_pnl_by_hour(results, out_dir / "fig_test_4_pnl_by_hour.png")
    plot_example_contracts(examples, out_dir / "fig_test_5_example_contracts.png")
    plot_action_distribution(results, out_dir / "fig_test_6_action_distribution.png")
    print(f"Saved 6 evaluation plots to {out_dir}")
