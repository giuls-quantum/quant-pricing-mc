"""Compatibility wrapper for the legacy mc_option_pricing engine module."""

from ..quant_pricing_mc.engine import MonteCarloEngine, simulate_gbm_paths

__all__ = ["MonteCarloEngine", "simulate_gbm_paths"]
