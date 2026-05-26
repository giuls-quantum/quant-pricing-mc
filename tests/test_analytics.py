import pytest
import math

from src.mc_option_pricing.analytics import analytical_bsm


def test_analytical_bsm_matches_manual_formula() -> None:
    S0 = 100.0
    K = 105.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    price = analytical_bsm(S0, K, r, sigma, T, option_type="call")

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    expected = (
        S0 * 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        - K * math.exp(-r * T) * 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    )

    assert math.isfinite(price)
    assert math.isclose(price, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_analytical_bsm_rejects_invalid_option_type() -> None:
    try:
        analytical_bsm(100.0, 105.0, 0.05, 0.2, 1.0, option_type="digital")
    except ValueError:
        assert True
    else:
        raise AssertionError("Expected ValueError for invalid option type")
