"""Plotting helpers for Monte Carlo option pricing."""

import matplotlib.pyplot as plt
import numpy as np

from .engine import MonteCarloEngine


def plot_simulation_results(engine: MonteCarloEngine, K: float, num_paths_to_plot: int = 50, simulations: int = 1000) -> None:
    """Create a two-panel plot of simulated paths and terminal price distribution."""
    paths = engine.generate_paths(simulations)
    time_grid = np.linspace(0, engine.T, engine.steps + 1)
    terminal_prices = paths[:, -1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(time_grid, paths[:num_paths_to_plot].T, lw=1, alpha=0.6)
    ax1.axhline(K, color="black", linestyle="--", label=f"Strike Price (K={K})")
    ax1.set_title(f"GBM: {num_paths_to_plot} Simulated Paths")
    ax1.set_xlabel("Time (Years)")
    ax1.set_ylabel("Asset Price")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.hist(terminal_prices, bins=50, color="skyblue", edgecolor="black", alpha=0.7, density=True)
    ax2.axvline(np.mean(terminal_prices), color="red", linestyle="-", label=f"Mean S_T: {np.mean(terminal_prices):.2f}")
    ax2.axvline(K, color="black", linestyle="--", label="Strike (K)")
    ax2.set_title(f"Terminal Price Distribution (n={simulations})")
    ax2.set_xlabel("Price at Maturity (S_T)")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
