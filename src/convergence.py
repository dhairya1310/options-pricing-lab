"""
Step 5: show that the binomial tree and Monte Carlo converge to Black-Scholes.

Run from the project root:
    python -m src.convergence

For a European option all three models should agree, so Black-Scholes acts as
the answer key. We take a real SPY contract (about 30 days out, closest to the
money), price it with more and more tree steps and simulation paths, and plot
how fast each method closes in on the exact value.

Output:
    reports/figures/convergence_binomial.png
    reports/figures/convergence_mc.png
    reports/convergence.csv
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pricing import binomial_price, bs_price, mc_price

PROCESSED_DIR = Path("data/processed")
FIG_DIR = Path("reports/figures")
REPORT_DIR = Path("reports")

TARGET_DAYS = 30
FALLBACK_VOL = 0.20
STEP_COUNTS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
PATH_COUNTS = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]

log = logging.getLogger("convergence")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def pick_contract() -> dict:
    """The SPY call closest to 30 days out and closest to the money."""
    files = sorted(PROCESSED_DIR.glob("SPY_*_ready.parquet"))
    if not files:
        raise FileNotFoundError("no processed SPY file, run src.inputs first")
    df = pd.read_parquet(files[-1])
    calls = df[df["option_type"] == "call"]

    nearest_expiry = calls.loc[(calls["days_to_expiry"] - TARGET_DAYS).abs().idxmin(), "expiry"]
    same_expiry = calls[calls["expiry"] == nearest_expiry]
    row = same_expiry.loc[(same_expiry["strike"] - same_expiry["spot"]).abs().idxmin()]

    # We have not built our own IV solver yet (that is Step 6), so we borrow
    # Yahoo's figure. Any sensible vol works here since we only test convergence.
    vol = row.get("impliedVolatility", np.nan)
    if not (0.05 < vol < 1.5):
        log.warning("Yahoo IV of %s looks off, using %.0f%% instead", vol, FALLBACK_VOL * 100)
        vol = FALLBACK_VOL

    contract = dict(S=float(row["spot"]), K=float(row["strike"]), T=float(row["T"]),
                    r=float(row["r"]), q=float(row["q"]), sigma=float(vol))
    log.info("Test contract: %s, %.1f days, K=%.0f, S=%.2f, vol=%.1f%%",
             row["contractSymbol"], row["days_to_expiry"], contract["K"], contract["S"], vol * 100)
    return contract


def run_binomial(c: dict, exact: float) -> pd.DataFrame:
    rows = []
    for n in STEP_COUNTS:
        price = binomial_price(**c, kind="call", steps=n, american=False)
        rows.append({"method": "binomial", "n": n, "price": price, "error": price - exact})
    return pd.DataFrame(rows)


def run_mc(c: dict, exact: float) -> pd.DataFrame:
    rows = []
    for n in PATH_COUNTS:
        mc = mc_price(**c, kind="call", paths=n)
        rows.append({"method": "monte_carlo", "n": n, "price": mc["price"], "error": mc["price"] - exact,
                     "std_err": mc["std_err"], "ci_low": mc["ci_low"], "ci_high": mc["ci_high"]})
    return pd.DataFrame(rows)


def sanity_check(tree: pd.DataFrame, mc: pd.DataFrame, exact: float) -> None:
    # The spec setting (500 steps) must land within one cent of Black-Scholes
    at_500 = tree.loc[tree["n"] == 500, "error"].abs().iloc[0]
    assert at_500 < 0.01, f"tree is {at_500:.4f} off at 500 steps"

    # More steps should mean a smaller error overall (the tree wobbles step to
    # step, so we compare the coarse end with the fine end, not every point)
    assert tree["error"].abs().iloc[-3:].max() < tree["error"].abs().iloc[:3].min(), \
        "tree error is not shrinking as steps grow"

    # The spec setting (100,000 paths) must have Black-Scholes inside its 95% interval
    at_100k = mc[mc["n"] == 100_000].iloc[0]
    assert at_100k["ci_low"] <= exact <= at_100k["ci_high"], "Black-Scholes falls outside the MC interval"

    # Monte Carlo error should fall like 1 over the square root of paths.
    # Going from 1,000 to 1,000,000 paths is 1000x, so the error should drop about 31.6x.
    ratio = mc["std_err"].iloc[0] / mc["std_err"].iloc[-1]
    assert 25 < ratio < 40, f"MC error fell {ratio:.1f}x, expected about 31.6x"

    # Across all path counts, roughly 95% of intervals should contain the answer.
    # With 10 runs we allow at most 2 misses before calling it suspicious.
    misses = ((mc["ci_low"] > exact) | (mc["ci_high"] < exact)).sum()
    assert misses <= 2, f"{misses} of {len(mc)} MC intervals missed Black-Scholes"

    log.info("sanity checks passed: tree off by %.5f at 500 steps, MC error fell %.1fx", at_500, ratio)


def plot_binomial(tree: pd.DataFrame, exact: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tree["n"], tree["price"], marker="o", label="Binomial tree (European)")
    ax.axhline(exact, color="black", linestyle="--", label=f"Black-Scholes = {exact:.4f}")
    ax.axvline(500, color="grey", linestyle=":", label="500 steps (used in this project)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of steps (log scale)")
    ax.set_ylabel("Call price ($)")
    ax.set_title("Binomial tree converges to Black-Scholes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence_binomial.png", dpi=150)
    plt.close(fig)


def plot_mc(mc: pd.DataFrame, exact: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(mc["n"], mc["ci_low"], mc["ci_high"], alpha=0.25, label="95% confidence interval")
    ax.plot(mc["n"], mc["price"], marker="o", label="Monte Carlo (antithetic)")
    ax.axhline(exact, color="black", linestyle="--", label=f"Black-Scholes = {exact:.4f}")
    ax.axvline(100_000, color="grey", linestyle=":", label="100,000 paths (used in this project)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths (log scale)")
    ax.set_ylabel("Call price ($)")
    ax.set_title("Monte Carlo converges to Black-Scholes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence_mc.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    c = pick_contract()
    exact = float(bs_price(**c, kind="call"))
    log.info("Black-Scholes answer key: %.4f", exact)

    tree = run_binomial(c, exact)
    mc = run_mc(c, exact)
    sanity_check(tree, mc, exact)

    plot_binomial(tree, exact)
    plot_mc(mc, exact)
    pd.concat([tree, mc], ignore_index=True).to_csv(REPORT_DIR / "convergence.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Binomial tree ===")
    print(tree[["n", "price", "error"]].round(5).to_string(index=False))
    print("\n=== Monte Carlo ===")
    print(mc[["n", "price", "error", "std_err", "ci_low", "ci_high"]].round(5).to_string(index=False))
    print(f"\nCharts saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
