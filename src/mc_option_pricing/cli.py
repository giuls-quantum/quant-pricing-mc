"""Command-line interface for Monte Carlo option pricing."""

import argparse

import numpy as np

from .analytics import analytical_bsm
from .engine import MonteCarloEngine
from .plotting import plot_simulation_results
from .pricing import price_asian_option, price_euro_option


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Monte Carlo option pricing demo.")
    parser.add_argument("--S0", type=float, default=100.0, help="Initial asset price")
    parser.add_argument("--K", type=float, default=105.0, help="Strike price")
    parser.add_argument("--T", type=float, default=1.0, help="Time to maturity in years")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.20, help="Annualized volatility")
    parser.add_argument("--steps", type=int, default=252, help="Number of discrete time steps")
    parser.add_argument("--simulations", type=int, default=100_000, help="Number of Monte Carlo paths")
    parser.add_argument("--plot-sims", type=int, default=2000, help="Number of paths to use for the plots")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--show", action="store_true", help="Display the plot window")
    return parser


def main() -> None:
    """Parse commmand-line arguments, run the Monte Carlo option pricing, and visualize results. """
    parser = build_parser()
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    engine = MonteCarloEngine(S0=args.S0, r=args.r, sigma=args.sigma, T=args.T, steps=args.steps)

    mc_call = price_euro_option(engine, args.K, args.simulations, option_type="call")
    bsm_call = analytical_bsm(args.S0, args.K, args.r, args.sigma, args.T, option_type="call")
    asian_call = price_asian_option(engine, args.K, args.simulations, option_type="call")

    print(f"{'Method':<20} | {'Price':<10}")
    print("-" * 35)
    print(f"{'Black-Scholes':<20} | ${bsm_call:.4f}")
    print(f"{'Monte Carlo Euro':<20} | ${mc_call:.4f}")
    print(f"{'Asian Call':<20} | ${asian_call:.4f}")
    print(f"{'MC Error vs BSM':<20} | ${abs(mc_call - bsm_call):.6f}")

    print("\nVisualizing simulation results...")
    plot_simulation_results(engine, args.K, num_paths_to_plot=50, simulations=args.plot_sims)


if __name__ == "__main__":
    main()
