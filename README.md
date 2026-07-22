# Monte Carlo Option Pricing Engine

## Overview
This repository contains a quantitative finance pricing engine that uses Monte Carlo simulations to calculate the theoretical value of financial derivatives. It is built upon the underlying mathematical assumptions of the **Black-Scholes-Merton (BSM)** framework, specifically utilizing **Geometric Brownian Motion (GBM)** to model asset price paths.

While the analytical BSM formula is highly efficient for standard European options, it cannot price path-dependent derivatives. This engine solves that limitation numerically by simulating discrete price trajectories over time, allowing for the pricing of both standard options and complex derivatives.


## Mathematical Framework & Logic
The engine simulates future stock prices under the risk-neutral measure using the discrete-time Geometric Brownian Motion stochastic differential equation:

$$S_{t+dt} = S_t \exp\left(\left(r - \frac{1}{2}\sigma^2\right)dt + \sigma\sqrt{dt} Z\right)$$

**Where:**
* **$S_t$**: Asset price at the current time step
* **$r$**: Continuous risk-free interest rate
* **$\sigma$**: Annualized volatility of the asset
* **$dt$**: The size of the time step (e.g., 1 day = $1/252$)
* **$Z$**: A random draw from a standard normal distribution ($N(0,1)$), representing market shocks.


The core algorithm executes the following sequence:
1. **Path Generation**: Generates $N$ independent, randomized asset price paths from the valuation date to expiration using the GBM equation.
2. **Payoff Evaluation**: For each individual simulated path, the engine calculates the option's payoff based on its specific contract rules.
3. **Statistical Expectation**: Averages the calculated payoffs across all $N$ simulated paths to find the expected future value.
4. **Discounting**: Discounts the expected future payoff back to present value using the continuous risk-free rate ($e^{-rT}$) to determine the theoretical fair price.


## Features
* **European Option Pricing**: Calculates standard call and put prices to benchmark against the analytical BSM formula.
* **Path-Dependent Engine**: Extensible logic to price **Asian options** (arithmetic average) and other exotics by tracking the asset's journey step-by-step.
* **Optimized Execution**: Utilizes **NumPy array vectorization** to process tens of thousands of simulations in milliseconds.
* **Variance Reduction**: Implements **Antithetic Variates** to significantly reduce computational noise and accelerate convergence to the true analytical price.


## Requirements

Install runtime dependencies with pip:

```bash
python3 -m pip install numpy matplotlib
```

## Testing

Install development dependencies with pip:

```bash
python3 -m pip install pytest
```

Run the test suite with:

```bash
PYTHONPATH=. pytest
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/giuls-quantum/quant-pricing-mc.git
cd quant-pricing-mc
```

2. Run the CLI:

```bash
python3 -m src.quant_pricing_mc.cli
```

The CLI will:

- Run a Monte Carlo simulation for European and Asian call options alongside an analytical Black-Scholes benchmark.
- Print a comparative pricing table to the console detailing the simulated prices, analytical prices, and the absolute MC error.
- Generate visualizations for the simulated asset price trajectories.

### CLI options

```bash
python3 -m src.quant_pricing_mc.cli \
  --S0 100.0 \
  --K 105.0 \
  --T 1.0 \

  --r 0.05 \
  --sigma 0.20 \
  --steps 252 \
  --simulations 100000 \
  --plot-sims 2000 \
  --seed 42 \
  --show
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--S0` | `float` | `100.0` | Initial asset price |
| `--K` | `float` | `105.0` | Strike price |
| `--T` | `float` | `1.0` | Time to maturity in years |
| `--r` | `float` | `0.05` | Risk-free interest rate |
| `--sigma` | `float` | `0.20` | Annualized volatility |
| `--steps` | `int` | `252` | Number of discrete time steps for path generation |
| `--simulations` | `int` | `100000` | Total number of Monte Carlo paths to simulate |
| `--plot-sims` | `int` | `2000` | Number of paths passed to the plotting routine |
| `--seed` | `int` | `None` | Random seed for reproducibility |
| `--show` | `flag` | `None` | Switch to display the generated plot window |

### Programmatic Usage

It is also possible to use the engine programmatically. To do that, import the components into your script, initialize the engine, and run the pricing functions:

```python
import numpy as np
from src.quant_pricing_mc.engine import MonteCarloEngine
from src.quant_pricing_mc.pricing import price_euro_option, price_asian_option

# Set a random seed for reproducibility
np.random.seed(42)

# Initialize the engine with market parameters
engine = MonteCarloEngine(S0=100.0, r=0.05, sigma=0.20, T=1.0, steps=252)

# Price a European Call Option
euro_price = price_euro_option(engine, K=105.0, simulations=100000, option_type='call')
print(f"The simulated European Call option price is: ${euro_price:.4f}")

# Price an Asian Call Option
asian_price = price_asian_option(engine, K=105.0, simulations=100000, option_type='call')
print(f"The simulated Asian Call option price is: ${asian_price:.4f}")
```

## Notes

- The code is designed for **educational and research purposes**, and can be extended to higher dimensions or other MCMC algorithms.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
