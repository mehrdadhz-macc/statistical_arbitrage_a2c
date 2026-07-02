"""
Synchronous advantage actor-critic (A2C) agent (paper §4.6, Algorithm 1).

Mirrors reinforce_threshold_policy's REINFORCEAgent public shape
(select action -> store step -> compute_gradients/update -> greedy_actions ->
param_snapshot) so it plugs into train.py/test.py/evaluate.py/parallel_worker.py
the same way, with an actor-critic net and n-step bootstrapped advantages in
place of REINFORCE's plain undiscounted return.

Also implements the epsilon/gamma annealing schedules and the four
behaviour-cloning rules (Eqs. 8-11) used to spur exploration across the W
synchronous workers.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.actor_critic import ActorCriticNet
from src.environment import BUY, HOLD, N_ACTIONS, SELL, STATE_DIM

# ── Epsilon / gamma annealing (Algorithm 1, lines 4-6) ──────────────────────


def episode_index(e: int, emax: int) -> float:
    """ei = 1 - (e / emax); e is 0-based here, so ei -> 1 at e=0 and -> ~0 at e=emax."""
    return 1.0 - (e / max(1, emax))


def epsilon_schedule(e: int, emax: int, eps_start: float = 0.9, eps_end: float = 0.01) -> float:
    ei = episode_index(e, emax)
    return (eps_start - eps_end) * ei + eps_end


def gamma_schedule(e: int, emax: int, gamma_start: float = 0.29, gamma_end: float = 0.9999) -> float:
    ei = episode_index(e, emax)
    return (gamma_start - gamma_end) * ei + gamma_end


# ── Behaviour-cloning rules (Eqs. 8-11) ─────────────────────────────────────


def clone_rule_dam_cid(ctx: dict) -> int:
    """Worker 1, Eq. (8): arbitrage between the DAM (pdam) and the CID."""
    if ctx["pa_t"] < ctx["pdam"] and ctx["vt"] < ctx["vmax"] and ctx["cum_bought"] < ctx["qhigh"]:
        return BUY
    if ctx["pb_t"] > ctx["pdam"] and ctx["vt"] > ctx["vmin"] and ctx["cum_sold"] < ctx["qhigh"]:
        return SELL
    return HOLD


def clone_rule_within_cid(ctx: dict) -> int:
    """Worker 2, Eq. (9): arbitrage within the CID using session extremes."""
    if ctx["pa_t"] < ctx["pb_high"] and ctx["vt"] < ctx["vmax"] and ctx["cum_bought"] < ctx["qhigh"]:
        return BUY
    if ctx["pb_t"] > ctx["pa_low"] and ctx["vt"] > ctx["vmin"] and ctx["cum_sold"] < ctx["qhigh"]:
        return SELL
    return HOLD


def clone_rule_cid_bal(ctx: dict) -> int:
    """
    Worker 3, Eq. (10): arbitrage between the CID and the balancing market,
    using the synthetic third-market dataset (data/{split}/balancing_prices.csv,
    src/balancing.py). `pfeed_bench`/`ptake_bench` are *causal* trailing
    averages of past contracts' realized settlement prices (src.balancing.
    bal_bench) -- unlike the session extremes used by clone_rule_within_cid,
    this must stay causal (no lookahead into this contract's own settlement).
    """
    if ctx["pa_t"] < ctx["pfeed_bench"] and ctx["vt"] < ctx["vmax"] and ctx["cum_bought"] < ctx["qhigh"]:
        return BUY
    if ctx["pb_t"] > ctx["ptake_bench"] and ctx["vt"] > ctx["vmin"] and ctx["cum_sold"] < ctx["qhigh"]:
        return SELL
    return HOLD


def make_threshold_clone_rule(tau: float):
    """Workers 4-8, Eq. (11): clone using a fixed reward threshold tau in [-1, 0)."""
    from src.rewards import buy_reward, sell_reward

    def rule(ctx: dict) -> int:
        r_buy = buy_reward(ctx["pa_t"], ctx["pdam"], ctx["pb_high"], ctx.get("pa_high", ctx["pa_low"]), ctx["pa_low"])
        r_sell = sell_reward(ctx["pb_t"], ctx["pdam"], ctx["pa_low"], ctx["pb_high"], ctx.get("pb_low", ctx["pb_high"]))
        if r_buy > tau and ctx["vt"] < ctx["vmax"] and ctx["cum_bought"] < ctx["qhigh"]:
            return BUY
        if r_sell > tau and ctx["vt"] > ctx["vmin"] and ctx["cum_sold"] < ctx["qhigh"]:
            return SELL
        return HOLD

    return rule


def build_worker_clone_rules(n_workers: int) -> list:
    """
    Assign a behaviour-cloning rule per worker index (paper §4.6): worker 0 ->
    Eq. 8, worker 1 -> Eq. 9, worker 2 -> Eq. 10 (adapted), workers 3.. -> Eq.
    11 with thresholds evenly spread across [-1, 0).
    """
    rules = [clone_rule_dam_cid, clone_rule_within_cid, clone_rule_cid_bal]
    n_extra = max(0, n_workers - len(rules))
    if n_extra > 0:
        thresholds = np.linspace(-0.9, -0.1, n_extra)
        rules += [make_threshold_clone_rule(float(tau)) for tau in thresholds]
    return rules[:n_workers]


# ── A2C agent ────────────────────────────────────────────────────────────


class A2CAgent:
    def __init__(
        self,
        lr: float = 1e-3,
        n1: int = 216,
        n2: int = 193,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
    ) -> None:
        self.lr = lr
        self.net = ActorCriticNet(state_dim=state_dim, n1=n1, n2=n2, n_actions=n_actions)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self._log_probs: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._entropies: list[torch.Tensor] = []
        self._rewards: list[float] = []

    # ── Episode interaction ─────────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        epsilon: float,
        clone_action: int | None,
        is_first_episode: bool,
        rng: random.Random | None = None,
    ) -> int:
        """
        Algorithm 1, lines 11-22 (simplified: our per-contract episodes are
        short relative to the paper's tmax, so we drop the `trandom` cloning
        *window* and simply clone with probability epsilon each step).

        Whatever action is finally chosen -- cloned, randomly explored, or
        greedily exploited -- its log-prob/value under the *current* policy
        are what get stored and used for the actor-critic update (Algorithm 1
        line 31 uses `log pi(a_i|s_i; theta)` for whatever a_i was taken).
        """
        rng = rng or random
        state_t = torch.as_tensor(state, dtype=torch.float32)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        probs, value = self.net(state_t, mask_t)
        dist = torch.distributions.Categorical(probs)

        if is_first_episode and clone_action is not None:
            action = clone_action
        elif rng.random() < epsilon:
            action = clone_action if clone_action is not None else int(dist.sample().item())
        else:
            action = int(torch.argmax(probs).item())

        action_t = torch.as_tensor(action)
        log_prob = dist.log_prob(action_t)
        entropy = dist.entropy()
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._entropies.append(entropy)
        return action

    def store_reward(self, reward: float) -> None:
        self._rewards.append(reward)

    # ── Learning ─────────────────────────────────────────────────────────

    def compute_gradients(
        self,
        gamma: float,
        grad_clip: float = 1.0,
        entropy_coef: float = 0.0,
    ) -> tuple[float, float, dict[str, np.ndarray]]:
        """
        Algorithm 1, lines 27-33: R=0 bootstrap (episodes always run to a true
        terminal state sT in this environment), reverse-accumulated discounted
        return, advantage = return - V(s), actor loss on the (detached)
        advantage, smooth-L1 critic loss, gradient clipping.

        Returns numpy gradients without stepping the optimizer -- callers
        either step directly (`update`) or average grads across synchronous
        workers first (`src.parallel_worker`).
        """
        returns = []
        R = 0.0
        for r in reversed(self._rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        values_t = torch.stack(self._values)
        log_probs_t = torch.stack(self._log_probs)

        advantages = returns_t - values_t.detach()
        actor_loss = -(log_probs_t * advantages).mean()
        if entropy_coef:
            actor_loss = actor_loss - entropy_coef * torch.stack(self._entropies).mean()
        critic_loss = F.smooth_l1_loss(values_t, returns_t)
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)

        grad_dict = {
            name: (p.grad.detach().numpy().copy() if p.grad is not None else np.zeros_like(p.detach().numpy()))
            for name, p in self.net.named_parameters()
        }
        self._clear_buffers()
        return float(actor_loss.item()), float(critic_loss.item()), grad_dict

    def update(self, gamma: float, grad_clip: float = 1.0, entropy_coef: float = 0.0) -> tuple[float, float]:
        """Sequential (single-worker) convenience path: compute + apply in one call."""
        actor_loss, critic_loss, _ = self.compute_gradients(gamma, grad_clip, entropy_coef)
        self.optimizer.step()
        return actor_loss, critic_loss

    def set_gradients(self, grad_dict: dict[str, np.ndarray]) -> None:
        """Apply externally-averaged gradients (parallel/synchronous path)."""
        for name, p in self.net.named_parameters():
            p.grad = torch.as_tensor(grad_dict[name])

    def step_optimizer(self) -> None:
        self.optimizer.step()

    def _clear_buffers(self) -> None:
        self._log_probs.clear()
        self._values.clear()
        self._entropies.clear()
        self._rewards.clear()

    # ── Evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32)
        mask_t = torch.as_tensor(mask, dtype=torch.bool)
        probs, _ = self.net(state_t, mask_t)
        return int(torch.argmax(probs).item())

    def param_snapshot(self) -> dict:
        snap = {"lr": self.lr}
        snap.update(self.net.param_snapshot())
        return snap
