"""Compatibility wrapper for the legacy mc_option_pricing pricing module."""

from ..quant_pricing_mc.pricing import price_asian_option, price_euro_option, price_european_option

__all__ = ["price_european_option", "price_euro_option", "price_asian_option"]
