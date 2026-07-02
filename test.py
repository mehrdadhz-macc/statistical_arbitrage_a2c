"""
Quick greedy evaluation of a trained A2C checkpoint: per-contract reward/PnL
table plus summary stats. For full diagnostic plots and benchmark comparisons,
use evaluate.py instead.

Usage:
    venv/bin/python3 test.py --model outputs/runs/<timestamp>/model.pt
    venv/bin/python3 test.py --model outputs/runs/<timestamp>/model.pt --days 10
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
from src.parallel_worker import eval_episode_worker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True, help="Path to model.pt checkpoint")
    p.add_argument("--days", type=int, default=None, help="Number of test days (default: all)")
    p.add_argument("--workers", type=int, default=1, help="Parallel evaluation workers")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    ckpt = torch.load(model_path, weights_only=False)

    print(f"Loaded checkpoint: days_trained={ckpt.get('days_trained')}  "
          f"rounds={ckpt.get('rounds')}  workers={ckpt.get('workers')}  lr={ckpt.get('lr')}")
    print(f"param_snapshot = {ckpt.get('param_snapshot')}")

    print("Loading test data...")
    cim, auc = load_all(split="test")
    bal = load_split_balancing(split="test")
    contracts = list_contracts(cim, auc, days=args.days)
    print(f"{len(contracts)} test contracts.")

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
        "record_ticks": False,
        "with_pre_ba": False,
        "with_hold": False,
    } for ds in contracts]

    if args.workers > 1:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            results = pool.map(eval_episode_worker, tasks)
    else:
        results = [eval_episode_worker(t) for t in tasks]

    results = [r for r in results if not r["skipped"]]

    print(f"\n{'ep':>4}  {'delivery_start':>25}  {'reward':>10}  {'pnl':>10}  {'steps':>6}")
    for i, r in enumerate(results):
        print(f"{i:4d}  {str(r['delivery_start']):>25}  {r['reward']:10.3f}  {r['pnl']:10.2f}  {r['n_steps']:6d}")

    pnls = np.array([r["pnl"] for r in results])
    rewards = np.array([r["reward"] for r in results])
    summary = (
        f"\nSummary over {len(results)} contracts:\n"
        f"  reward: mean={rewards.mean():.3f}  std={rewards.std():.3f}\n"
        f"  pnl:    mean={pnls.mean():.2f}  std={pnls.std():.2f}  "
        f"min={pnls.min():.2f}  max={pnls.max():.2f}  total={pnls.sum():.2f}\n"
        f"  %PnL>0: {100.0 * (pnls > 0).mean():.1f}%\n"
    )
    print(summary)

    if model_path.name == "model.pt":
        (model_path.parent / "test_summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
