"""
Repeated Prisoner's Dilemma simulation and simple evolutionary experiments.

This script implements several classic strategies (ALLC, ALLD, TFT, Grim Trigger,
Generous TFT, Win-Stay-Lose-Shift) and provides tools to:

1. Simulate a single repeated game between two strategies with discounting
   and optional action errors.
2. Sweep the discount factor delta to study long-run payoffs of strategy pairs.
3. Run a simple replicator-dynamics-style evolution of a population of strategies.

You can use this script as the "experiment code" for a game-theoretic course project.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Type

import numpy as np
import matplotlib.pyplot as plt

# ----- Game parameters -----

ACTIONS = ("C", "D")

# Payoff matrix for the Prisoner's Dilemma:
# (my_action, opp_action) -> (my_payoff, opp_payoff)
PAYOFF: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("C", "C"): (3.0, 3.0),  # Reward
    ("C", "D"): (0.0, 5.0),  # Sucker, Temptation
    ("D", "C"): (5.0, 0.0),  # Temptation, Sucker
    ("D", "D"): (1.0, 1.0),  # Punishment
}

R = PAYOFF[("C", "C")][0]
T = PAYOFF[("D", "C")][0]
P = PAYOFF[("D", "D")][0]
S = PAYOFF[("C", "D")][0]


# ----- Strategy definitions -----

@dataclass
class Strategy:
    """
    Base class for strategies in the repeated Prisoner's Dilemma.

    Subclasses should override `first_move` and `next_move`.
    """

    name: str = "BaseStrategy"
    history_self: List[str] = field(default_factory=list)
    history_opp: List[str] = field(default_factory=list)

    def reset(self) -> None:
        self.history_self.clear()
        self.history_opp.clear()

    def first_move(self) -> str:
        """Action in the first round."""
        return "C"

    def next_move(self) -> str:
        """Action in round t > 1; default is tit-for-tat style."""
        if not self.history_opp:
            return self.first_move()
        return self.history_opp[-1]

    def move(self) -> str:
        if not self.history_self:
            return self.first_move()
        return self.next_move()

    def update_history(self, my_action: str, opp_action: str) -> None:
        self.history_self.append(my_action)
        self.history_opp.append(opp_action)


@dataclass
class AlwaysCooperate(Strategy):
    name: str = "ALLC"

    def first_move(self) -> str:
        return "C"

    def next_move(self) -> str:
        return "C"


@dataclass
class AlwaysDefect(Strategy):
    name: str = "ALLD"

    def first_move(self) -> str:
        return "D"

    def next_move(self) -> str:
        return "D"


@dataclass
class TitForTat(Strategy):
    name: str = "TFT"

    def first_move(self) -> str:
        return "C"

    def next_move(self) -> str:
        # Cooperate if opponent cooperated last round, otherwise defect
        return "C" if self.history_opp and self.history_opp[-1] == "C" else "D"


@dataclass
class GrimTrigger(Strategy):
    """
    Grim Trigger: start with C; if opponent ever defects, defect forever.
    """
    name: str = "GRIM"
    triggered: bool = False

    def reset(self) -> None:
        super().reset()
        self.triggered = False

    def first_move(self) -> str:
        return "C"

    def next_move(self) -> str:
        if self.triggered:
            return "D"
        # If opponent has ever defected, trigger punishment
        if "D" in self.history_opp:
            self.triggered = True
            return "D"
        return "C"


@dataclass
class GenerousTitForTat(Strategy):
    """
    Generous Tit-for-Tat: normally copy opponent's last action,
    but forgive a defection with some probability.
    """
    name: str = "GTFT"
    forgiveness_prob: float = 0.1  # probability to cooperate despite last-round D

    def first_move(self) -> str:
        return "C"

    def next_move(self) -> str:
        if not self.history_opp:
            return "C"
        if self.history_opp[-1] == "C":
            return "C"
        # Opponent defected last round; forgive with some probability
        if random.random() < self.forgiveness_prob:
            return "C"
        return "D"


@dataclass
class WinStayLoseShift(Strategy):
    """
    Win-Stay, Lose-Shift (a.k.a. Pavlov):
    - If last round's payoff was R or T (good), repeat the same action.
    - If last round's payoff was P or S (bad), switch action.
    """
    name: str = "WSLS"

    def first_move(self) -> str:
        return "C"

    def next_move(self) -> str:
        last_my = self.history_self[-1]
        last_opp = self.history_opp[-1]
        my_payoff, _ = PAYOFF[(last_my, last_opp)]
        if my_payoff in (R, T):
            # stay with previous action
            return last_my
        else:
            # switch
            return "D" if last_my == "C" else "C"


# Map from strategy name to class for convenience
STRATEGY_CLASSES: Dict[str, Type[Strategy]] = {
    "ALLC": AlwaysCooperate,
    "ALLD": AlwaysDefect,
    "TFT": TitForTat,
    "GRIM": GrimTrigger,
    "GTFT": GenerousTitForTat,
    "WSLS": WinStayLoseShift,
}


# ----- Core simulation functions -----

def maybe_flip_action(action: str, error_prob: float) -> str:
    """
    With probability error_prob, flip C <-> D to model implementation noise.
    """
    if random.random() < error_prob:
        return "D" if action == "C" else "C"
    return action


def play_repeated_game(
    strat1: Strategy,
    strat2: Strategy,
    delta: float = 0.95,
    T_max: int = 200,
    error_prob: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Simulate a repeated PD game between two strategies.

    Parameters
    ----------
    strat1, strat2 : Strategy
        Strategy objects (will be reset at start).
    delta : float
        Discount factor in (0,1).
    T_max : int
        Number of rounds (should be large, e.g. 200).
    error_prob : float
        Probability that a chosen action is flipped (implementation error).

    Returns
    -------
    avg_payoff_1 : float
        Discounted average payoff of player 1.
    avg_payoff_2 : float
        Discounted average payoff of player 2.
    coop_freq : float
        Overall cooperation frequency across both players and all rounds.
    """
    strat1.reset()
    strat2.reset()

    total_payoff_1 = 0.0
    total_payoff_2 = 0.0
    total_weight = 0.0
    coop_count = 0
    total_actions = 0

    for t in range(T_max):
        a1 = strat1.move()
        a2 = strat2.move()

        # Implementation errors
        a1_real = maybe_flip_action(a1, error_prob)
        a2_real = maybe_flip_action(a2, error_prob)

        u1, u2 = PAYOFF[(a1_real, a2_real)]

        weight = delta ** t
        total_payoff_1 += weight * u1
        total_payoff_2 += weight * u2
        total_weight += weight

        coop_count += (1 if a1_real == "C" else 0) + (1 if a2_real == "C" else 0)
        total_actions += 2

        strat1.update_history(a1_real, a2_real)
        strat2.update_history(a2_real, a1_real)

    avg_payoff_1 = total_payoff_1 / total_weight
    avg_payoff_2 = total_payoff_2 / total_weight
    coop_freq = coop_count / total_actions if total_actions > 0 else 0.0
    return avg_payoff_1, avg_payoff_2, coop_freq


def sweep_delta_for_pair(
    strat_name_1: str,
    strat_name_2: str,
    deltas: np.ndarray,
    T_max: int = 200,
    error_prob: float = 0.0,
    n_runs: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a pair of strategies, sweep discount factor delta and estimate
    their average payoffs and cooperation frequency.

    Returns arrays:
    avg_payoff_1[d], avg_payoff_2[d], avg_coop[d]
    """
    cls1 = STRATEGY_CLASSES[strat_name_1]
    cls2 = STRATEGY_CLASSES[strat_name_2]

    avg_payoff_1 = np.zeros_like(deltas, dtype=float)
    avg_payoff_2 = np.zeros_like(deltas, dtype=float)
    avg_coop = np.zeros_like(deltas, dtype=float)

    for i, d in enumerate(deltas):
        p1_list, p2_list, coop_list = [], [], []
        for _ in range(n_runs):
            s1 = cls1()
            s2 = cls2()
            p1, p2, cf = play_repeated_game(
                s1, s2, delta=float(d), T_max=T_max, error_prob=error_prob
            )
            p1_list.append(p1)
            p2_list.append(p2)
            coop_list.append(cf)
        avg_payoff_1[i] = np.mean(p1_list)
        avg_payoff_2[i] = np.mean(p2_list)
        avg_coop[i] = np.mean(coop_list)

    return avg_payoff_1, avg_payoff_2, avg_coop


# ----- Simple evolutionary (replicator) dynamics -----

def payoff_matrix_for_strategies(
    strategy_names: List[str],
    delta: float = 0.95,
    T_max: int = 200,
    error_prob: float = 0.0,
) -> np.ndarray:
    """
    Compute an |S| x |S| payoff matrix A where A[i,j] is the
    discounted average payoff of strategy i against strategy j.
    """
    n = len(strategy_names)
    A = np.zeros((n, n), dtype=float)
    for i, name_i in enumerate(strategy_names):
        for j, name_j in enumerate(strategy_names):
            s_i = STRATEGY_CLASSES[name_i]()
            s_j = STRATEGY_CLASSES[name_j]()
            p_i, p_j, _ = play_repeated_game(
                s_i, s_j, delta=delta, T_max=T_max, error_prob=error_prob
            )
            A[i, j] = p_i
    return A


def replicator_dynamics(
    payoff_matrix: np.ndarray,
    x0: np.ndarray,
    n_generations: int = 50,
    learning_rate: float = 1.0,
) -> np.ndarray:
    """
    Run a simple discrete-time replicator dynamics:
        x_i(t+1) = x_i(t) * exp(eta * (pi_i - avg_pi)),
    followed by normalization.

    This "softmax-like" update keeps all frequencies positive.
    """
    x = x0.astype(float)
    x = x / x.sum()
    n = len(x)
    history = np.zeros((n_generations + 1, n), dtype=float)
    history[0] = x

    for t in range(1, n_generations + 1):
        # payoffs for each strategy given current population x
        pi = payoff_matrix @ x
        avg_pi = float(np.dot(pi, x))
        # Exponentiated replicator step
        growth = np.exp(learning_rate * (pi - avg_pi))
        x = x * growth
        # renormalize
        x = x / x.sum()
        history[t] = x

    return history


# ----- Main demo -----

def main():
    random.seed(0)
    np.random.seed(0)

    # --- Experiment 1: Pairwise payoffs vs delta ---

    deltas = np.linspace(0.1, 0.99, 15)
    pairs = [
        ("TFT", "ALLD"),
        ("GRIM", "ALLD"),
        ("GRIM", "TFT"),
        ("GRIM", "GRIM"),  # 用来验证 δ 阈值条件
        ("TFT", "TFT"),
        ("GTFT", "ALLD"),
        ("WSLS", "ALLD"),
    ]

    for s1, s2 in pairs:
        p1, p2, coop = sweep_delta_for_pair(
            s1, s2, deltas, T_max=200, error_prob=0.01, n_runs=10
        )
        print(f"=== {s1} vs {s2} ===")
        for d, a, b, c in zip(deltas, p1, p2, coop):
            print(
                f"delta={d:.2f}, {s1} payoff={a:.3f}, {s2} payoff={b:.3f}, "
                f"cooperation freq={c:.3f}"
            )
        print()

        # Plot payoffs
        plt.figure()
        plt.title(f"Payoffs vs delta: {s1} vs {s2}")
        plt.plot(deltas, p1, marker="o", label=f"{s1} payoff")
        plt.plot(deltas, p2, marker="s", label=f"{s2} payoff")
        plt.xlabel("Discount factor delta")
        plt.ylabel("Discounted average payoff")
        plt.legend()
        plt.grid(True)

        # Plot cooperation frequency
        plt.figure()
        plt.title(f"Cooperation frequency vs delta: {s1} vs {s2}")
        plt.plot(deltas, coop, marker="o")
        plt.xlabel("Discount factor delta")
        plt.ylabel("Cooperation frequency")
        plt.grid(True)

    # --- Experiment 2: Replicator dynamics among several strategies ---

    strategy_names = ["ALLD", "TFT", "GRIM", "GTFT", "WSLS"]
    delta = 0.95
    error_prob = 0.01

    A = payoff_matrix_for_strategies(
        strategy_names, delta=delta, T_max=200, error_prob=error_prob
    )
    print("Payoff matrix A[i,j] (row i vs column j):")
    print("Strategies:", strategy_names)
    print(A)

    # Initial population: equal frequencies
    x0 = np.ones(len(strategy_names)) / len(strategy_names)
    history = replicator_dynamics(A, x0, n_generations=60, learning_rate=0.5)

    # Plot population shares over generations
    generations = np.arange(history.shape[0])
    plt.figure()
    for i, name in enumerate(strategy_names):
        plt.plot(generations, history[:, i], label=name)
    plt.xlabel("Generation")
    plt.ylabel("Population share")
    plt.title(f"Replicator dynamics (delta={delta}, error_prob={error_prob})")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
