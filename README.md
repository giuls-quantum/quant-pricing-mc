# Monte Carlo Option Pricing Engine

## Overview
This repository contains a quantitative finance pricing engine that uses Monte Carlo simulations to calculate the theoretical value of financial derivatives. It is built upon the underlying mathematical assumptions of the **Black-Scholes-Merton (BSM)** framework, specifically utilizing **Geometric Brownian Motion (GBM)** to model asset price paths.

While the analytical BSM formula is highly efficient for standard European options, it cannot price path-dependent derivatives. This engine solves that limitation numerically by simulating discrete price trajectories over time, allowing for the pricing of both standard options and complex derivatives.

---

## The Mathematical Framework
The engine simulates future stock prices under the risk-neutral measure using the discrete-time Geometric Brownian Motion stochastic differential equation:

$$S_{t+dt} = S_t \exp\left(\left(r - \frac{1}{2}\sigma^2\right)dt + \sigma\sqrt{dt} Z\right)$$

**Where:**
* **$S_t$**: Asset price at the current time step
* **$r$**: Continuous risk-free interest rate
* **$\sigma$**: Annualized volatility of the asset
* **$dt$**: The size of the time step (e.g., 1 day = $1/252$)
* **$Z$**: A random draw from a standard normal distribution ($N(0,1)$), representing market shocks.



---

## Implementation Logic
The core algorithm executes the following sequence:
1. **Path Generation**: Generates $N$ independent, randomized asset price paths from the valuation date to expiration using the GBM equation.
2. **Payoff Evaluation**: For each individual simulated path, the engine calculates the option's payoff based on its specific contract rules.
3. **Statistical Expectation**: Averages the calculated payoffs across all $N$ simulated paths to find the expected future value.
4. **Discounting**: Discounts the expected future payoff back to present value using the continuous risk-free rate ($e^{-rT}$) to determine the theoretical fair price.

---

## Key Features
* **European Option Pricing**: Calculates standard call and put prices to benchmark against the analytical BSM formula.
* **Path-Dependent Engine**: Extensible logic to price **Asian options** (arithmetic average) and other exotics by tracking the asset's journey step-by-step.
* **Optimized Execution**: Utilizes **NumPy array vectorization** to process tens of thousands of simulations in milliseconds.
* **Variance Reduction**: Implements **Antithetic Variates** to significantly reduce computational noise and accelerate convergence to the true analytical price.

---

## Installation

Clone the repository and ensure you have the necessary dependencies installed:

```bash
git clone [https://github.com/giuls-quantum/mc_option_pricing.git](https://github.com/giuls-quantum/mc_option_pricing.git)
cd mc-option-pricing
pip install numpy matplotlib scipy
```
(Dependencies primarily include numpy for core calculations and matplotlib for path visualization).

---

## Example Usage

To use the engine, import the class into your script, initialize it with your market parameters, and call the desired pricing method:

```python
from engine import MonteCarloEngine

# Initialize the engine
engine = MonteCarloEngine(S0=100.0, r=0.05, sigma=0.20, T=1.0)

# Price a European Call Option
price = engine.price_euro_option(K=105.0, simulations=100000, option_type='call')
print(f"The simulated option price is: ${price:.4f}")
```
