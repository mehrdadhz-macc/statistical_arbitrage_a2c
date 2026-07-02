"""
Training metrics logging + plots, adapted from reinforce_threshold_policy's
TrainingLogger for the A2C setting.

The sibling project's "episode"/"rep" split existed for its curriculum
training (hourly -> 15-min -> 5-min tick resolution phases). This project
uses a single fixed train/test split with no curriculum (see README "Scope &
simplifications"), so the unit of logging here is one *round* -- one batch of
W synchronous workers each playing one contract episode, averaged into a
single gradient update (paper §4.6, Algorithm 1's outer `while e < emax` loop).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rolling_mean(x: list[float], window: int = 20) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    arr = np.asarray(x, dtype=np.float64)
    kernel = np.ones(min(window, len(arr))) / min(window, len(arr))
    return np.convolve(arr, kernel, mode="valid")


class TrainingLogger:
    def __init__(self) -> None:
        self._rounds: list[int] = []
        self._rewards: list[float] = []
        self._actor_losses: list[float] = []
        self._critic_losses: list[float] = []
        self._epsilons: list[float] = []
        self._gammas: list[float] = []
        self._param_snapshots: list[dict] = []

    def log_round(
        self,
        round_idx: int,
        mean_reward: float,
        actor_loss: float,
        critic_loss: float,
        epsilon: float,
        gamma: float,
        param_snapshot: dict | None = None,
    ) -> None:
        self._rounds.append(round_idx)
        self._rewards.append(mean_reward)
        self._actor_losses.append(actor_loss)
        self._critic_losses.append(critic_loss)
        self._epsilons.append(epsilon)
        self._gammas.append(gamma)
        if param_snapshot is not None:
            self._param_snapshots.append(param_snapshot)

    def plot(self, out_dir: str | Path = "outputs/plots") -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rounds = self._rounds

        # fig1: reward per round + rolling mean
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.scatter(rounds, self._rewards, s=8, alpha=0.4, label="round reward")
        rm = _rolling_mean(self._rewards)
        if len(rm) > 0:
            ax.plot(rounds[-len(rm):], rm, color="C1", label="rolling mean (20)")
        ax.set_xlabel("round")
        ax.set_ylabel("mean reward across workers")
        ax.set_title("Training reward")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fig1_rewards.png", dpi=150)
        plt.close(fig)

        # fig2: actor / critic loss
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(rounds, self._actor_losses, color="C2")
        axes[0].set_ylabel("actor loss")
        axes[1].plot(rounds, self._critic_losses, color="C3")
        axes[1].set_ylabel("critic loss")
        axes[1].set_xlabel("round")
        fig.suptitle("Training losses")
        fig.tight_layout()
        fig.savefig(out_dir / "fig2_losses.png", dpi=150)
        plt.close(fig)

        # fig3: epsilon / gamma annealing schedule
        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(rounds, self._epsilons, color="C4", label="epsilon")
        ax1.set_xlabel("round")
        ax1.set_ylabel("epsilon", color="C4")
        ax2 = ax1.twinx()
        ax2.plot(rounds, self._gammas, color="C5", label="gamma")
        ax2.set_ylabel("gamma", color="C5")
        fig.suptitle("Exploration / discount schedule")
        fig.tight_layout()
        fig.savefig(out_dir / "fig3_schedule.png", dpi=150)
        plt.close(fig)

        # fig4: network weight norms (diagnostic, if param snapshots were logged)
        if self._param_snapshots:
            keys = list(self._param_snapshots[0].keys())
            fig, ax = plt.subplots(figsize=(9, 4))
            for k in keys:
                if k == "lr":
                    continue
                vals = [snap.get(k) for snap in self._param_snapshots]
                ax.plot(range(len(vals)), vals, label=k)
            ax.set_xlabel("logged snapshot index")
            ax.set_ylabel("weight norm")
            ax.set_title("Network weight norms")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "fig4_weight_norms.png", dpi=150)
            plt.close(fig)

        print(f"Saved training plots to {out_dir}")
