import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==========================================
# Class Definition and methods
# ==========================================

class MonteCarloEngine:
    def __init__(self, S0: float, r: float, sigma: float, T: float, steps: int = 252):
        """
        Initializes the Monte Carlo Option Pricing Engine.
        
        :param S0: Initial stock price
        :param r: Continuous risk-free rate
        :param sigma: Volatility of the underlying asset
        :param T: Time to maturity in years
        :param steps: Number of time steps (default = 252 for daily trading)
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.dt = self.T / self.steps

    def generate_paths(self, simulations: int) -> np.ndarray:
        """
        Generates simulated stock price paths using Geometric Brownian Motion.
        Implements Antithetic Variates for variance reduction.
        """
        # Simulate an even number of paths for antithetic variates
        half_sims = simulations // 2
        
        # Generate random standard normal matrix for half the simulations
        Z = np.random.standard_normal((half_sims, self.steps))
        
        # Append the negative of the random numbers to reduce variance (Antithetic Variates)
        Z = np.vstack((Z, -Z))
        
        # Initialize the paths array with the starting price
        paths = np.zeros((simulations, self.steps + 1))
        paths[:, 0] = self.S0
        
        # Calculate the drift component
        drift = (self.r - 0.5 * self.sigma ** 2) * self.dt
        
        # Vectorized path generation
        for t in range(1, self.steps + 1):
            shock = self.sigma * np.sqrt(self.dt) * Z[:, t-1]
            paths[:, t] = paths[:, t-1] * np.exp(drift + shock)
            
        return paths

    def price_euro_option(self, K: float, simulations: int, option_type: str = 'call') -> float:
        """
        Prices a standard European option (Call or Put).
        """
        paths = self.generate_paths(simulations)
        terminal_prices = paths[:, -1] # We only care about the final price at expiration
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(terminal_prices - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - terminal_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
            
        # Discount the average payoff back to present value
        expected_payoff = np.mean(payoffs)
        discounted_price = np.exp(-self.r * self.T) * expected_payoff
        
        return discounted_price

    def price_asian_option(self, K: float, simulations: int, option_type: str = 'call') -> float:
        """
        Prices an Asian option where the payoff depends on the average price over the path.
        """
        paths = self.generate_paths(simulations)
        
        # Calculate the arithmetic average price across each path (excluding the start price S0)
        average_prices = np.mean(paths[:, 1:], axis=1)
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(average_prices - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - average_prices, 0)
            
        expected_payoff = np.mean(payoffs)
        discounted_price = np.exp(-self.r * self.T) * expected_payoff
        
        return discounted_price

    def analytical_bsm(self, K: float, option_type: str = 'call') -> float:
        """
        Standard Black-Scholes-Merton formula for benchmarking European options.
        """
        d1 = (np.log(self.S0 / K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if option_type.lower() == 'call':
            price = self.S0 * norm.cdf(d1) - K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
            
        return price

    def plot_simulation_results(self, K: float, num_paths_to_plot: int = 50, simulations: int = 1000):
        """
        Generates a dual-plot dashboard: 
        1. Simulated price paths with Strike Price K.
        2. Histogram of terminal prices (S_T) with the option payoff.
        """
        paths = self.generate_paths(simulations)
        time_grid = np.linspace(0, self.T, self.steps + 1)
        terminal_prices = paths[:, -1]
        
        # Create a figure with two subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # --- Plot 1: Price Paths ---
        # Plotting the first N paths
        ax1.plot(time_grid, paths[:num_paths_to_plot].T, lw=1, alpha=0.6)
        # Add a horizontal line for the Strike Price
        ax1.axhline(K, color='black', linestyle='--', label=f'Strike Price (K={K})')
        
        ax1.set_title(f'GBM: {num_paths_to_plot} Simulated Paths')
        ax1.set_xlabel('Time (Years)')
        ax1.set_ylabel('Asset Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: Terminal Price Distribution ---
        ax2.hist(terminal_prices, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        # Add a vertical line for the mean terminal price
        ax2.axvline(np.mean(terminal_prices), color='red', linestyle='-', label=f'Mean S_T: {np.mean(terminal_prices):.2f}')
        ax2.axvline(K, color='black', linestyle='--', label=f'Strike (K)')
        
        ax2.set_title(f'Terminal Price Distribution (n={simulations})')
        ax2.set_xlabel('Price at Maturity (S_T)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# ==========================================
# Execution
# ------------------------------------------
# 1. Configure market parameters (S0, K, T, r, sigma).
# 2. Run the MC Engine to price European and path-dependent (Asian) options.
# 3. Compare stochastic results against the BSM analytical closed-form solution.
# 4. Visualize the risk distribution and price paths.
# ==========================================

if __name__ == "__main__":
    # Define Market Parameters
    S0 = 100.0          # Initial Asset Price
    K = 105.0           # Strike Price (Option Exercise Price)
    T = 1.0             # Time to Maturity (expressed in Years)
    r = 0.05            # Annual Risk-Free Rate (5%)
    sigma = 0.20        # Annual Volatility (20%)
    
    # Define Computation Settings
    SIMS = 100_000      # Total number of Monte Carlo paths to generate

    # Initialize Engine
    engine = MonteCarloEngine(S0, r, sigma, T, steps=252)

    # Calculate Prices
    mc_call = engine.price_euro_option(K, SIMS, 'call')
    bsm_call = engine.analytical_bsm(K, 'call')
    asian_call = engine.price_asian_option(K, SIMS, 'call')
    
    # Print Results Table
    print(f"{'Method':<20} | {'Price':<10}")
    print("-" * 35)
    print(f"{'Black-Scholes':<20} | ${bsm_call:.4f}")
    print(f"{'Monte Carlo Euro':<20} | ${mc_call:.4f}")
    print(f"{'Asian Call':<20} | ${asian_call:.4f}")
    print(f"{'MC Error vs BSM':<20} | ${abs(mc_call - bsm_call):.6f}")

    # Generate the plots
    print("\nVisualizing simulation results...")
    engine.plot_simulation_results(K, num_paths_to_plot=50, simulations=2000)
