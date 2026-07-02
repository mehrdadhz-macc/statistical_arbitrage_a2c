"""
Fork-based synchronous multi-worker training/evaluation, mirroring
reinforce_threshold_policy's src/parallel_worker.py pattern.

The parent sets module-level globals (the CIM order book, grouped once by
delivery_start, and the precomputed VWAP-BENCH series) *before* creating a
`multiprocessing.get_context("fork").Pool` -- children inherit them via
copy-on-write, so the ~GB-scale DataFrames are never pickled or copied.

`train_episode_worker` plays exactly one contract episode using a policy
snapshot supplied in the task dict and returns numpy gradients (no optimizer
step) for the parent to average across workers -- this "collect independently,
average synchronously, one central optimizer step" loop *is* synchronous A2C
(paper §4.6: workers update the global network "synchronously in A2C"), so
`--workers 8` in train.py directly implements the paper's W=8.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch

from src import dam_policy
from src.a2c_trainer import A2CAgent, build_worker_clone_rules
from src.environment import ContractCIDEnv

# ── Fork-inherited globals (set by the parent, read-only in children) ──────

_g_cim_groups = None
_g_auc = None
_g_session_vwap: pd.Series | None = None


def init_globals(cim: pd.DataFrame, auc: pd.DataFrame) -> None:
    """Call once in the parent process before creating the worker Pool."""
    global _g_cim_groups, _g_auc, _g_session_vwap
    _g_cim_groups = cim.groupby("delivery_start")
    _g_auc = auc
    _g_session_vwap = dam_policy.precompute_session_mid_vwap(cim)


def _build_env_for_contract(
    delivery_start: pd.Timestamp,
    env_kwargs: dict,
) -> tuple[ContractCIDEnv | None, np.ndarray | None, np.ndarray | None]:
    """Rebuild a ContractCIDEnv for one delivery_start using the fork-inherited globals."""
    try:
        cim_contract = _g_cim_groups.get_group(delivery_start)
    except KeyError:
        return None, None, None

    pdam_arr = dam_policy.pdam_series(_g_auc, [delivery_start])
    if pdam_arr is None:
        return None, None, None
    pdam = float(pdam_arr[0])

    vwap_hat = dam_policy.vwap_bench(_g_session_vwap, delivery_start)
    v0, c_dam = dam_policy.open_dam_position(
        pdam, vwap_hat, env_kwargs["vmax"], env_kwargs["vmin"]
    )

    env = ContractCIDEnv(**env_kwargs)
    state, mask = env.reset(
        cim_contract, pdam=pdam, v0=v0, c_dam=c_dam,
        vwap_hat=vwap_hat, neighbor_bench=vwap_hat,
    )
    return env, state, mask


def _rebuild_agent(task: dict) -> A2CAgent:
    agent = A2CAgent(lr=task["lr"], **task["ctor_args"])
    state_dict = {k: torch.as_tensor(v) for k, v in task["state_dict"].items()}
    agent.net.load_state_dict(state_dict)
    return agent


def _play_episode(agent: A2CAgent, env: ContractCIDEnv, state, mask, task: dict, greedy: bool):
    rng = random.Random(task.get("seed"))
    clone_rules = build_worker_clone_rules(task["n_workers"])
    clone_rule = clone_rules[task["worker_id"] % len(clone_rules)]

    total_reward = 0.0
    total_traded_qty = 0.0
    n_steps = 0
    tick_records = []
    action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
    action_names = ("HOLD", "BUY", "SELL")
    t = 0
    while True:
        ctx = env.rule_context()
        if greedy:
            action = agent.greedy_action(state, mask)
        else:
            clone_action = clone_rule(ctx)
            action = agent.select_action(
                state, mask, task["epsilon"], clone_action,
                task["is_first_episode"], rng,
            )
        action_counts[action_names[action]] += 1
        if task.get("record_ticks"):
            tick_records.append({**ctx, "action": action, "t": t})

        next_state, next_mask, reward, done, info = env.step(action)
        if not greedy:
            agent.store_reward(reward)
        total_reward += reward
        total_traded_qty += abs(info.traded_qty)
        n_steps += 1
        t += 1
        state, mask = next_state, next_mask
        if done:
            break
    return total_reward, n_steps, info, tick_records, action_counts, total_traded_qty


def train_episode_worker(task: dict) -> dict:
    """
    Plays one contract episode with behaviour cloning/exploration, returns
    gradients (not applied) for the parent to average across workers.
    """
    env, state, mask = _build_env_for_contract(task["delivery_start"], task["env_kwargs"])
    if env is None:
        return {"skipped": True, "delivery_start": task["delivery_start"]}

    agent = _rebuild_agent(task)
    total_reward, n_steps, info, _, _, total_traded_qty = _play_episode(
        agent, env, state, mask, task, greedy=False
    )
    actor_loss, critic_loss, grad_dict = agent.compute_gradients(
        gamma=task["gamma"], grad_clip=task.get("grad_clip", 1.0),
        entropy_coef=task.get("entropy_coef", 0.0),
    )
    return {
        "skipped": False,
        "grad_dict": grad_dict,
        "reward": total_reward,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "n_steps": n_steps,
        "pnl": info.pnl,
        "traded_qty": total_traded_qty,
        "delivery_start": task["delivery_start"],
    }


def eval_episode_worker(task: dict) -> dict:
    """Greedy (deterministic) evaluation of one contract; optionally records ticks."""
    env, state, mask = _build_env_for_contract(task["delivery_start"], task["env_kwargs"])
    if env is None:
        return {"skipped": True, "delivery_start": task["delivery_start"]}

    agent = _rebuild_agent(task)
    total_reward, n_steps, info, tick_records, action_counts, total_traded_qty = _play_episode(
        agent, env, state, mask, task, greedy=True
    )

    result = {
        "skipped": False,
        "delivery_start": task["delivery_start"],
        "reward": total_reward,
        "pnl": info.pnl,
        "n_steps": n_steps,
        "traded_qty": total_traded_qty,
        "action_counts": action_counts,
        "tick_records": tick_records if task.get("record_ticks") else None,
    }

    if task.get("with_pre_ba"):
        from src.benchmarks import run_pre_ba
        env2, state2, mask2 = _build_env_for_contract(task["delivery_start"], task["env_kwargs"])
        result["pre_ba_pnl"] = run_pre_ba(env2, state2, mask2)

    if task.get("with_hold"):
        from src.benchmarks import run_hold
        env3, _, _ = _build_env_for_contract(task["delivery_start"], task["env_kwargs"])
        result["hold_pnl"] = run_hold(env3)

    return result
