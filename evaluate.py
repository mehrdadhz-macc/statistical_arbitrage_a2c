"""
Full evaluation of a trained A2C checkpoint: greedy policy vs HOLD (always)
and PRE-BA (optional, Eq. 14) benchmarks, six diagnostic plots (paper Figs.
9-11 analogues + 3 general diagnostics), and a paper-Table-7-style summary
(traded quantity, PnL, %PnL>0, profit-to-deviation PD, profit-to-trade PT).

Usage:
    venv/bin/python3 evaluate.py --model outputs/runs/<timestamp>/model.pt
    venv/bin/python3 evaluate.py --model outputs/runs/<timestamp>/model.pt --with-pre-ba
    venv/bin/python3 evaluate.py --model outputs/runs/<timestamp>/model.pt --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from src import parallel_worker
from src.balancing import load_split_balancing
from src.contracts import list_contracts
from src.data_loader import load_all
from src.eval_plots import ContractResult, save_all_plots
from src.parallel_worker import eval_episode_worker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--model", type=str, required=True, help="Path to model.pt checkpoint")
    p.add_argument("--days", type=int, default=None, help="Number of days to evaluate (default: all)")
    p.add_argument("--plot-dir", type=str, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--with-pre-ba", action="store_true", help="Also compute the PRE-BA benchmark (Eq. 14)")
    return p.parse_args()


def _resolve_plot_dir(args: argparse.Namespace, model_path: Path) -> Path:
    if args.plot_dir:
        return Path(args.plot_dir)
    if model_path.name == "model.pt":
        return model_path.parent / "eval"
    return Path("outputs/eval_plots")


def _summary_stats(pnls: np.ndarray, qtys: np.ndarray) -> dict:
    pnl_sum, pnl_mean = float(pnls.sum()), float(pnls.mean())
    qty_sum, qty_mean = float(qtys.sum()), float(qtys.mean())
    pct_pos = 100.0 * float((pnls > 0).mean())
    pd_ratio = pnl_sum / pnls.std() if pnls.std() > 0 else float("nan")
    pt_ratio = pnl_sum / qty_sum if qty_sum > 0 else float("nan")
    return {
        "qty_sum": qty_sum, "qty_mean": qty_mean,
        "pnl_sum": pnl_sum, "pnl_mean": pnl_mean,
        "pct_pos": pct_pos, "pd": pd_ratio, "pt": pt_ratio,
    }


def _print_table7(name: str, stats: dict) -> None:
    print(f"{name:>8}  qty_sum={stats['qty_sum']:10.2f}  qty_mean={stats['qty_mean']:6.2f}  "
          f"pnl_sum={stats['pnl_sum']:12.2f}  pnl_mean={stats['pnl_mean']:8.2f}  "
          f"PnL>0={stats['pct_pos']:5.1f}%  PD={stats['pd']:8.2f}  PT={stats['pt']:6.3f}")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    ckpt = torch.load(model_path, weights_only=False)
    plot_dir = _resolve_plot_dir(args, model_path)

    print(f"Loading {args.split} data...")
    cim, auc = load_all(split=args.split)
    bal = load_split_balancing(split=args.split)
    contracts = list_contracts(cim, auc, days=args.days)
    print(f"{len(contracts)} contracts.")

    example_idx = sorted(set([0, len(contracts) // 2, len(contracts) - 1]))
    example_set = {contracts[i] for i in example_idx if 0 <= i < len(contracts)}

    parallel_worker.init_globals(cim, auc, bal)

    state_dict_np = {k: v.numpy() for k, v in ckpt["state_dict"].items()}
    tasks = [{
        "delivery_start": ds,
        "env_kwargs": ckpt["env_kwargs"],
        "lr": ckpt["lr"],
        "ctor_args": ckpt["ctor_args"],
        "state_dict": state_dict_np,
        "n_workers": 1,
        "worker_id": 0,
        "record_ticks": ds in example_set,
        "with_pre_ba": args.with_pre_ba,
        "with_hold": True,
    } for ds in contracts]

    if args.workers > 1:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            raw_results = pool.map(eval_episode_worker, tasks)
    else:
        raw_results = [eval_episode_worker(t) for t in tasks]

    raw_results = [r for r in raw_results if not r["skipped"]]

    results = [
        ContractResult(
            delivery_start=r["delivery_start"],
            hour=r["delivery_start"].tz_convert("Europe/Berlin").hour,
            reward=r["reward"],
            pnl=r["pnl"],
            n_steps=r["n_steps"],
            traded_qty=r["traded_qty"],
            hold_pnl=r.get("hold_pnl"),
            pre_ba_pnl=r.get("pre_ba_pnl"),
            action_counts=r.get("action_counts"),
        )
        for r in raw_results
    ]
    examples = {
        f"contract {i} ({r['delivery_start']})": r["tick_records"]
        for i, r in enumerate(raw_results)
        if r.get("tick_records")
    }

    save_all_plots(results, examples, plot_dir)

    pnls = np.array([r.pnl for r in results])
    qtys = np.array([r.traded_qty for r in results])
    print("\nTest results (paper Table 7 columns):")
    _print_table7("A2C", _summary_stats(pnls, qtys))
    if all(r.hold_pnl is not None for r in results):
        hold_pnls = np.array([r.hold_pnl for r in results])
        _print_table7("HOLD", _summary_stats(hold_pnls, np.zeros_like(hold_pnls)))
    if args.with_pre_ba and all(r.pre_ba_pnl is not None for r in results):
        pre_ba_pnls = np.array([r.pre_ba_pnl for r in results])
        _print_table7("PRE-BA", _summary_stats(pre_ba_pnls, qtys))


if __name__ == "__main__":
    main()
