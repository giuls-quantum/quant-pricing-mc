import pytest
import math

import numpy as np

from src.mc_option_pricing.engine import MonteCarloEngine
from src.mc_option_pricing.pricing import price_asian_option, price_euro_option


@pytest.fixture
def engine() -> MonteCarloEngine:
    return MonteCarloEngine(S0=100.0, r=0.05, sigma=0.2, T=1.0, steps=5)


def test_price_euro_option_returns_finite_value(engine: MonteCarloEngine) -> None:
    np.random.seed(7)

    price = price_euro_option(engine, K=105.0, simulations=2000, option_type="call")

    assert isinstance(price, float)
    assert math.isfinite(price)
    assert price >= 0.0


def test_price_asian_option_returns_finite_value(engine: MonteCarloEngine) -> None:
    np.random.seed(11)

    price = price_asian_option(engine, K=105.0, simulations=2000, option_type="put")

    assert isinstance(price, float)
    assert math.isfinite(price)
    assert price >= 0.0


@pytest.mark.parametrize("option_type", ["", "digital", "none"])
def test_invalid_option_type_raises(engine: MonteCarloEngine, option_type: str) -> None:
    with pytest.raises(ValueError):
        price_euro_option(engine, K=105.0, simulations=100, option_type=option_type)

    with pytest.raises(ValueError):
        price_asian_option(engine, K=105.0, simulations=100, option_type=option_type)
