"""Core Monte Carlo engine for option pricing."""

from typing import Optional

import numpy as np


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate asset paths with Geometric Brownian Motion using NumPy."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    Z = rng.standard_normal((n_paths, n_steps))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    cumulative_returns = np.cumsum(log_returns, axis=1)

    paths = np.zeros((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(cumulative_returns)

    return paths


class MonteCarloEngine:
    """Simulate asset price paths under geometric Brownian motion."""

    def __init__(self, S0: float, r: float, sigma: float, T: float, steps: int = 252):
        """Initialize the Monte Carlo option pricing engine."""
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.dt = self.T / self.steps

    def generate_paths(self, simulations: int) -> np.ndarray:
        """
        Generate simulated stock price paths with antithetic variates.

        Args:
            simulations: Total number of paths to simulate
                (if not even, one extra path will be generated without an antithetic pair).

        Returns:
            A 2D numpy array of shape (simulations, steps + 1) of simulated paths.
        """
        if simulations <= 0:
            raise ValueError("simulations must be a positive integer")

        half_sims = simulations // 2
        if half_sims == 0:
            half_sims = 1

        Z = np.random.standard_normal((half_sims, self.steps))
        Z = np.vstack((Z, -Z))

        if simulations != 2 * half_sims:
            extra = np.random.standard_normal((1, self.steps))
            Z = np.vstack((Z, extra))

        paths = np.zeros((simulations, self.steps + 1), dtype=float)
        paths[:, 0] = self.S0

        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        shock_scale = self.sigma * np.sqrt(self.dt)

        for t in range(1, self.steps + 1):
            shock = shock_scale * Z[:, t - 1]
            paths[:, t] = paths[:, t - 1] * np.exp(drift + shock)

        return paths
