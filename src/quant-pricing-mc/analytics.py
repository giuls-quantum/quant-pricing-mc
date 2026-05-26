"""Analytical benchmarks and pricing diagnostics."""

import numpy as np
from scipy.stats import norm


def analytical_bsm(
        S0: float, 
        K: float, 
        r: float, 
        sigma: float, 
        T: float, 
        option_type: str = "call"
    ) -> float:
    """
    Compute the Black-Scholes-Merton price for a European option.
    
    Args:
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        option_type: Option type ("call" or "put")

    Returns:
        The Black-Scholes-Merton price for the European option.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    return float(price)
