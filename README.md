# Deep PDE Solvers — 2D Black-Scholes Exchange Option

Pricing the 2-dimensional Black-Scholes exchange option (Margrabe) by solving
the associated PDE with deep BSDE methods, and comparing against the analytical
Margrabe benchmark.

Forked from [msabvid/Deep-PDE-Solvers](https://github.com/msabvid/Deep-PDE-Solvers)
(see `LICENSE`). This trimmed copy keeps only the exchange-option solver and
adds:

- **`corr_max` training method** (Algorithm 5, Appendix B of the paper) — trains
  the gradient network `dfdx` by maximizing the squared Pearson correlation
  between the discounted terminal payoff and the discrete stochastic-integral
  control variate. Lives alongside the existing `bsde` and `l2_proj` methods in
  `lib/bsde_risk_neutral_measure.py`.
- **`analyze_results.py`** — aggregates per-seed `results.csv` files across
  methods, computes the Margrabe analytical price, and prints a comparison
  table (variance reduction factor, bias, etc.).

Original reference:

```
@misc{vidales2019unbiased,
    title  = {Unbiased deep solvers for parametric PDEs},
    author = {Marc Sabate Vidales and David Siska and Lukasz Szpruch},
    year   = {2019},
    eprint = {1810.05094},
    archivePrefix = {arXiv},
    primaryClass  = {q-fin.CP}
}
```

## Install

Python 3.10 is what this was tested on. For GPU support (recommended; the
default torch wheel on PyPI is CPU-only):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

For CPU-only, change the `torch` line in `requirements.txt` to plain
`torch==2.11.0` and install without the extra index URL.

## Running

Train one or more methods (writes per-seed output under
`numerical_results/BS/exchange_{d}/{method}/seed{N}/`):

```powershell
python pde_BlackScholes_exchange.py --use_cuda --device 0 --d 2 --method bsde
python pde_BlackScholes_exchange.py --use_cuda --device 0 --d 2 --method l2_proj
python pde_BlackScholes_exchange.py --use_cuda --device 0 --d 2 --method corr_max
```

Key flags (defaults in parentheses):
`--T 0.5`, `--mu 0.05`, `--sigma 0.3`, `--d 2`, `--n_steps 50`,
`--batch_size 500`, `--max_updates 5000`, `--n_seeds 10`,
`--method {bsde,l2_proj,corr_max}`.

Then aggregate and compare against the analytical Margrabe price:

```powershell
python analyze_results.py --T 0.5 --r 0.05 --sigma 0.3 --S0 1.0 --d 2
```

This prints a per-method table with columns *Var(MC)*, *Var(MC+CV) ± std*,
*Variance Reduction Factor ± std*, *Mean(MC+CV)*, *Margrabe*, and
*Bias = Mean(MC+CV) − Margrabe*, and also writes
`comparison_table.csv` into the results directory.

## Layout

```
pde_BlackScholes_exchange.py        Training entry point (bsde / l2_proj / corr_max)
analyze_results.py                  Cross-seed aggregation + Margrabe comparison
lib/bsde_risk_neutral_measure.py    FBSDE base class + Black-Scholes subclass
lib/networks.py                     FFN building blocks
lib/options.py                      Option payoffs (Exchange) + Margrabe closed form
lib/utils.py                        Seeding + logging helpers
numerical_results/BS/               Per-seed training output (CSV + checkpoint)
requirements.txt                    Pinned dependencies
```
