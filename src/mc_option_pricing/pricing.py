"""Pricing helpers for European and Asian options."""

import numpy as np

from .engine import MonteCarloEngine


def _validate_option_type(option_type: str) -> str:
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return option_type


def price_euro_option(
        engine: MonteCarloEngine, 
        K: float, 
        simulations: int, 
        option_type: str = "call"
    )-> float:
    """
    Price a standard European option using Monte Carlo simulation.
    
    Args:
        engine: An instance of MonteCarloEngine to generate price paths.
        K: Strike price of the option.
        simulations: Number of Monte Carlo paths to simulate.
        option_type: Type of the option ("call" or "put").

    Returns:
        The estimated price of the European option.
    """
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
        option_type: str = "call"
    ) -> float:
    """
    Price an Asian option using the arithmetic average of simulated prices.
    
    Args:
        engine: An instance of MonteCarloEngine to generate price paths.
        K: Strike price of the option.
        simulations: Number of Monte Carlo paths to simulate.
        option_type: Type of the option ("call" or "put").

    Returns:
        The estimated price of the Asian option.
    """
    option_type = _validate_option_type(option_type)

    paths = engine.generate_paths(simulations)
    average_prices = np.mean(paths[:, 1:], axis=1)

    if option_type == "call":
        payoffs = np.maximum(average_prices - K, 0.0)
    else:
        payoffs = np.maximum(K - average_prices, 0.0)

    discounted_price = np.exp(-engine.r * engine.T) * np.mean(payoffs)
    return float(discounted_price)
