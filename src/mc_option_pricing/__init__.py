"""Compatibility package for the historical mc_option_pricing import path."""

from ..quant_pricing_mc import MonteCarloEngine, simulate_gbm_paths
from ..quant_pricing_mc.pricing import price_asian_option, price_euro_option, price_european_option

__all__ = [
    "MonteCarloEngine",
    "simulate_gbm_paths",
    "price_european_option",
    "price_euro_option",
    "price_asian_option",
]
