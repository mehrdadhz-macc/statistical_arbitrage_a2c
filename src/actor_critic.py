"""
Two-headed shared actor-critic network (paper §4.6, Fig. 4).

A shared torso of L=2 tanh hidden layers (n1=216, n2=193 -- paper §5.4.2's
optimised hyperparameters, inherited from Demir, Stappers, Kok & Paterakis
2022) branches into:
  - a policy head: softmax over {HOLD, BUY, SELL}, masked so invalid actions
    (paper Eq. 4, enforced upstream by src.environment.ContractCIDEnv) get
    zero probability -- implemented by setting their logits to -inf before
    the softmax.
  - a value head: scalar state-value estimate V(s), used by the critic.
"""

from __future__ import annotations

import torch
from torch import nn

from src.environment import N_ACTIONS, STATE_DIM


class ActorCriticNet(nn.Module):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n1: int = 216,
        n2: int = 193,
        n_actions: int = N_ACTIONS,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n1 = n1
        self.n2 = n2
        self.n_actions = n_actions

        self.fc1 = nn.Linear(state_dim, n1)
        self.fc2 = nn.Linear(n1, n2)
        self.policy_head = nn.Linear(n2, n_actions)
        self.value_head = nn.Linear(n2, 1)

    def torso(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return x

    def forward(
        self,
        state: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state : (..., state_dim) float tensor.
            mask  : (..., n_actions) bool tensor, True = action allowed.
        Returns:
            (action_probs (..., n_actions), value (...,))
        """
        x = self.torso(state)
        logits = self.policy_head(x)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        value = self.value_head(x).squeeze(-1)
        return probs, value

    def ctor_args(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "n1": self.n1,
            "n2": self.n2,
            "n_actions": self.n_actions,
        }

    def param_snapshot(self) -> dict:
        """Lightweight diagnostic snapshot for hparams.txt / test.py summaries."""
        with torch.no_grad():
            return {
                "fc1_weight_norm": float(self.fc1.weight.norm()),
                "fc2_weight_norm": float(self.fc2.weight.norm()),
                "policy_head_weight_norm": float(self.policy_head.weight.norm()),
                "value_head_weight_norm": float(self.value_head.weight.norm()),
            }
