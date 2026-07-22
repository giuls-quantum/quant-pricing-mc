"""Pricing helpers for European and Asian options."""

from typing import Optional

import numpy as np

from .engine import MonteCarloEngine, simulate_gbm_paths


def _validate_option_type(option_type: str) -> str:
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return option_type


def price_european_option(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_steps: int = 100,
    n_paths: int = 100_000,
    seed: Optional[int] = None,
) -> float:
    """Price a European option with Monte Carlo simulation."""
    option_type = _validate_option_type(option_type)
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed)
    ST = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    else:
        payoffs = np.maximum(K - ST, 0.0)

    discount_factor = np.exp(-r * T)
    return float(discount_factor * np.mean(payoffs))


def price_euro_option(
    engine: MonteCarloEngine,
    K: float,
    simulations: int,
    option_type: str = "call",
) -> float:
    """Price a European option using the engine-based API."""
    option_type = _validate_option_type(option_type)
    paths = engine.generate_paths(simulations)
    terminal_prices = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(terminal_prices - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal_prices, 0.0)

    discounted_price = np.exp(-engine.r * engine.T) * np.mean(payoffs)
    return float(discounted_price)


def price_asian_option(
    engine: MonteCarloEngine,
    K: float,
    simulations: int,
    option_type: str = "call",
) -> float:
    """Price an Asian option using the arithmetic average of simulated prices."""
    option_type = _validate_option_type(option_type)

    paths = engine.generate_paths(simulations)
    average_prices = np.mean(paths[:, 1:], axis=1)

    if option_type == "call":
        payoffs = np.maximum(average_prices - K, 0.0)
    else:
        payoffs = np.maximum(K - average_prices, 0.0)

    discounted_price = np.exp(-engine.r * engine.T) * np.mean(payoffs)
    return float(discounted_price)
