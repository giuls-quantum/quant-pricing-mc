import pytest
import numpy as np

from src.mc_option_pricing.engine import MonteCarloEngine


@pytest.fixture
def engine() -> MonteCarloEngine:
    return MonteCarloEngine(S0=100.0, r=0.05, sigma=0.2, T=1.0, steps=5)


def test_generate_paths_returns_expected_shape_and_start_value(engine: MonteCarloEngine) -> None:
    np.random.seed(123)

    paths = engine.generate_paths(4)

    assert paths.shape == (4, 6)
    assert np.allclose(paths[:, 0], engine.S0)
    assert np.all(paths > 0)


def test_generate_paths_rejects_non_positive_simulations(engine: MonteCarloEngine) -> None:
    with pytest.raises(ValueError):
        engine.generate_paths(0)

    with pytest.raises(ValueError):
        engine.generate_paths(-3)
