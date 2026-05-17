"""
Analyze results from pde_BlackScholes_exchange.py runs across seeds and methods.

For each method in {bsde, l2_proj, corr_max} found under --results_dir, aggregate
per-seed results.csv files, compute mean/std across seeds, and compare against
the analytical Margrabe price for the 2-D exchange option.
"""

import argparse
import glob
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import norm


METHODS = ["bsde", "l2_proj", "corr_max"]


def margrabe_price(S0: float, T: float, sigma: float) -> float:
    """
    Closed-form Margrabe price for a 2-D exchange option on assets with identical
    volatility ``sigma``, uncorrelated Brownians, and equal spot ``S0``.

    The price is independent of the risk-free rate (it cancels between the two
    assets), so ``r`` is intentionally not used here.
    """
    sigma_bar = sigma * math.sqrt(2.0)
    d1 = (sigma_bar ** 2 * T / 2.0) / (sigma_bar * math.sqrt(T))
    d2 = d1 - sigma_bar * math.sqrt(T)
    return S0 * (norm.cdf(d1) - norm.cdf(d2))


def aggregate_method(method_dir):
    """Collect per-seed results.csv files and compute mean/std across seeds."""
    csv_paths = sorted(glob.glob(os.path.join(method_dir, "seed*", "results.csv")))
    if not csv_paths:
        return None
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    cols = [
        "discounted_payoff",
        "discounted_payoff_cv",
        "variance_red_factor",
        "var_discounted_payoff",
        "var_discounted_payoff_cv",
    ]
    stats = {"n_seeds": len(df)}
    for c in cols:
        if c in df.columns:
            stats[f"{c}_mean"] = float(df[c].mean())
            stats[f"{c}_std"] = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
        else:
            stats[f"{c}_mean"] = float("nan")
            stats[f"{c}_std"] = float("nan")
    return stats


def fmt_mean_std(mean, std, fmt="{:.4e}"):
    if np.isnan(mean):
        return "N/A"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-seed results.csv files from pde_BlackScholes_exchange.py "
            "and compare against the analytical Margrabe price."
        )
    )
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument(
        "--r",
        type=float,
        default=0.05,
        help="risk-free rate (called --mu in the training script); not used in the "
        "Margrabe formula but accepted for completeness",
    )
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--S0", type=float, default=1.0)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join("numerical_results", "BS", "exchange_2"),
    )
    args = parser.parse_args()

    if args.d != 2:
        print(
            f"warning: the Margrabe closed form is only valid for d=2 (got d={args.d}); "
            "the printed Margrabe value will not be the right benchmark."
        )

    margrabe = margrabe_price(S0=args.S0, T=args.T, sigma=args.sigma)

    rows = []
    for method in METHODS:
        method_dir = os.path.join(args.results_dir, method)
        stats = aggregate_method(method_dir)
        if stats is None:
            rows.append({
                "Method": method,
                "n_seeds": 0,
                "Var(MC) mean": "N/A",
                "Var(MC+CV) mean ± std": "N/A",
                "Variance Reduction Factor mean ± std": "N/A",
                "Mean(MC+CV)": "N/A",
                "Margrabe": f"{margrabe:.6f}",
                "Bias = Mean(MC+CV) - Margrabe": "N/A",
            })
            continue

        bias = stats["discounted_payoff_cv_mean"] - margrabe
        rows.append({
            "Method": method,
            "n_seeds": stats["n_seeds"],
            "Var(MC) mean": f"{stats['var_discounted_payoff_mean']:.4e}",
            "Var(MC+CV) mean ± std": fmt_mean_std(
                stats["var_discounted_payoff_cv_mean"],
                stats["var_discounted_payoff_cv_std"],
            ),
            "Variance Reduction Factor mean ± std": fmt_mean_std(
                stats["variance_red_factor_mean"],
                stats["variance_red_factor_std"],
                fmt="{:.2f}",
            ),
            "Mean(MC+CV)": f"{stats['discounted_payoff_cv_mean']:.6f}",
            "Margrabe": f"{margrabe:.6f}",
            "Bias = Mean(MC+CV) - Margrabe": f"{bias:+.6f}",
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "comparison_table.csv")
    table.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
