"""
Train the A2C statistical-arbitrage trading agent (Demir et al. 2023, §4-5.4).

One episode = one hourly delivery contract's CID session (src/environment.py).
One round = args.workers synchronous worker episodes, gradient-averaged into a
single optimizer step (paper Algorithm 1; --workers 8 matches the paper's
W=8). Total episodes trained = --rounds * --workers (paper: emax=100, W=8 ->
800 total episodes).

Usage:
    venv/bin/python3 train.py
    venv/bin/python3 train.py --days 200 --rounds 100 --workers 8
    venv/bin/python3 train.py --days 5 --rounds 4 --workers 2   # smoke test
"""

from __future__ import annotations

import argparse
import multiprocessing
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src import dam_policy, parallel_worker
from src.a2c_trainer import A2CAgent, epsilon_schedule, gamma_schedule
from src.contracts import list_contracts
from src.balancing import load_split_balancing
from src.data_loader import load_all
from src.parallel_worker import train_episode_worker
from src.training_logger import TrainingLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=200, help="Training days (default 200)")
    p.add_argument("--rounds", type=int, default=100, help="Synchronisation rounds (paper's emax, default 100)")
    p.add_argument("--workers", type=int, default=8, help="Synchronous workers per round (paper's W, default 8)")
    p.add_argument("--lr", type=float, default=0.003, help="Adam learning rate (paper's beta, Section 5.4.2)")
    p.add_argument("--out", type=str, default=None, help="Checkpoint path (default outputs/runs/<timestamp>/model.pt)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--plot-dir", type=str, default=None, help="Training plot directory")
    p.add_argument("--n1", type=int, default=216, help="Hidden layer 1 size (paper default 216)")
    p.add_argument("--n2", type=int, default=193, help="Hidden layer 2 size (paper default 193)")
    p.add_argument("--vmax", type=float, default=10.0)
    p.add_argument("--vmin", type=float, default=-10.0)
    p.add_argument("--qhigh", type=float, default=50.0)
    p.add_argument("--pnl-low", type=float, default=-5000.0)
    p.add_argument("--pnl-high", type=float, default=10000.0)
    p.add_argument("--eps-start", type=float, default=0.9)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--gamma-start", type=float, default=0.29)
    p.add_argument("--gamma-end", type=float, default=0.9999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--entropy-coef", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=1, help="Print/snapshot every N rounds (default 1 = every round)")
    return p.parse_args()


def _env_kwargs(args: argparse.Namespace) -> dict:
    return {
        "vmax": args.vmax, "vmin": args.vmin, "qhigh": args.qhigh,
        "pnl_low": args.pnl_low, "pnl_high": args.pnl_high,
    }


def _write_hparams(path: Path, args: argparse.Namespace, agent: A2CAgent, n_contracts: int) -> None:
    lines = [f"{k} = {v}" for k, v in sorted(vars(args).items())]
    lines.append(f"n_train_contracts = {n_contracts}")
    lines.append(f"initial_param_snapshot = {agent.param_snapshot()}")
    path.write_text("\n".join(lines) + "\n")


def _resolve_run_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.out:
        out_path = Path(args.out)
        run_dir = out_path.parent
    else:
        run_dir = Path("outputs/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = run_dir / "model.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir) if args.plot_dir else run_dir / "training"
    return run_dir, out_path, plot_dir


def _build_tasks(agent: A2CAgent, sample: list, env_kwargs: dict, args: argparse.Namespace,
                  epsilon: float, gamma: float, is_first_episode: bool, round_idx: int) -> list[dict]:
    state_dict_np = {k: v.detach().cpu().numpy() for k, v in agent.net.state_dict().items()}
    ctor_args = agent.net.ctor_args()
    base_seed = (args.seed or 0) * 1_000_000 + round_idx * 1_000
    tasks = []
    for worker_id, delivery_start in enumerate(sample):
        tasks.append({
            "delivery_start": delivery_start,
            "env_kwargs": env_kwargs,
            "lr": args.lr,
            "ctor_args": ctor_args,
            "state_dict": state_dict_np,
            "n_workers": args.workers,
            "worker_id": worker_id,
            "epsilon": epsilon,
            "gamma": gamma,
            "is_first_episode": is_first_episode,
            "grad_clip": args.grad_clip,
            "entropy_coef": args.entropy_coef,
            "seed": base_seed + worker_id,
        })
    return tasks


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run_dir, out_path, plot_dir = _resolve_run_paths(args)

    print("Loading training data...")
    cim, auc = load_all(split="train")
    bal = load_split_balancing(split="train")
    contracts = list_contracts(cim, auc, days=args.days)
    print(f"{len(contracts)} training contracts across {args.days} days.")

    env_kwargs = _env_kwargs(args)
    agent = A2CAgent(lr=args.lr, n1=args.n1, n2=args.n2)
    _write_hparams(run_dir / "hparams.txt", args, agent, len(contracts))

    print("Precomputing VWAP-BENCH/BAL-BENCH and grouping order book by contract (one-time)...")
    parallel_worker.init_globals(cim, auc, bal)

    pool = None
    if args.workers > 1:
        pool = multiprocessing.get_context("fork").Pool(args.workers)

    logger = TrainingLogger()
    t0 = time.time()
    try:
        for round_idx in range(args.rounds):
            epsilon = epsilon_schedule(round_idx, args.rounds, args.eps_start, args.eps_end)
            gamma = gamma_schedule(round_idx, args.rounds, args.gamma_start, args.gamma_end)
            is_first_episode = round_idx == 0

            sample_idx = rng.integers(0, len(contracts), size=args.workers)
            sample = [contracts[i] for i in sample_idx]
            tasks = _build_tasks(agent, sample, env_kwargs, args, epsilon, gamma, is_first_episode, round_idx)

            if pool is not None:
                results = pool.map(train_episode_worker, tasks)
            else:
                results = [train_episode_worker(t) for t in tasks]

            valid = [r for r in results if not r["skipped"]]
            if not valid:
                continue

            grad_keys = valid[0]["grad_dict"].keys()
            avg_grad = {k: np.mean([r["grad_dict"][k] for r in valid], axis=0) for k in grad_keys}
            agent.set_gradients(avg_grad)
            agent.step_optimizer()

            mean_reward = float(np.mean([r["reward"] for r in valid]))
            mean_actor_loss = float(np.mean([r["actor_loss"] for r in valid]))
            mean_critic_loss = float(np.mean([r["critic_loss"] for r in valid]))
            snapshot = agent.param_snapshot() if round_idx % args.log_every == 0 else None
            logger.log_round(round_idx, mean_reward, mean_actor_loss, mean_critic_loss, epsilon, gamma, snapshot)

            if round_idx % args.log_every == 0 or round_idx == args.rounds - 1:
                elapsed = time.time() - t0
                print(f"round {round_idx:4d}/{args.rounds}  reward={mean_reward:8.2f}  "
                      f"actor_loss={mean_actor_loss:7.4f}  critic_loss={mean_critic_loss:9.4f}  "
                      f"eps={epsilon:.3f}  gamma={gamma:.4f}  elapsed={elapsed:6.1f}s")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    torch.save({
        "state_dict": agent.net.state_dict(),
        "ctor_args": agent.net.ctor_args(),
        "lr": args.lr,
        "env_kwargs": env_kwargs,
        "days_trained": args.days,
        "rounds": args.rounds,
        "workers": args.workers,
        "seed": args.seed,
        "param_snapshot": agent.param_snapshot(),
    }, out_path)
    print(f"Saved model to {out_path}")

    logger.plot(out_dir=plot_dir)


if __name__ == "__main__":
    main()
