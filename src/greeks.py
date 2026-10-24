"""
Step 7: Greeks for every contract that has an implied vol.

Run from the project root:
    python -m src.greeks

Two sets of Greeks per contract, both at the contract's own implied vol:
    bs_*    closed form Black-Scholes (European)
    tree_*  binomial tree with early exercise, delta/gamma/theta read off the
            tree, vega/rho by finite differences

Units are converted to how traders quote them:
    delta   change in option price per $1 move in the stock
    gamma   change in delta per $1 move in the stock
    vega    change in option price per 1 vol point (e.g. 25% to 26%)
    theta   change in option price per calendar day
    rho     change in option price per 1% move in rates

Output:
    data/processed/<TICKER>_<timestamp>_greeks.parquet
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pricing import binomial_greeks, bs_greeks

TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "JPM"]
PROCESSED_DIR = Path("data/processed")
GREEKS = ["delta", "gamma", "vega", "theta", "rho"]

# Raw units out of pricing.py are per 1.00 of vol, per year, per 1.00 of rate
TO_TRADER_UNITS = {"delta": 1, "gamma": 1, "vega": 1 / 100, "theta": 1 / 365, "rho": 1 / 100}

log = logging.getLogger("greeks")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def add_bs_greeks(df: pd.DataFrame) -> pd.DataFrame:
    """Closed form, vectorised over the whole chain at once."""
    df = df.copy()
    for kind in ("call", "put"):
        rows = df["option_type"] == kind
        g = bs_greeks(df.loc[rows, "spot"].values, df.loc[rows, "strike"].values,
                      df.loc[rows, "T"].values, df.loc[rows, "r"].values,
                      df.loc[rows, "q"].values, df.loc[rows, "iv"].values, kind)
        for name in GREEKS:
            df.loc[rows, f"bs_{name}"] = g[name] * TO_TRADER_UNITS[name]
    return df


def add_tree_greeks(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """One tree per contract, so this is the slow part (about 20 ms each)."""
    df = df.copy()
    out = {name: np.empty(len(df)) for name in GREEKS}
    start = time.time()
    for i, row in enumerate(df.itertuples()):
        g = binomial_greeks(row.spot, row.strike, row.T, row.r, row.q, row.iv,
                            row.option_type, steps=500, american=True)
        for name in GREEKS:
            out[name][i] = g[name] * TO_TRADER_UNITS[name]
        if (i + 1) % 1000 == 0:
            log.info("%s: %d of %d trees done", ticker, i + 1, len(df))
    for name in GREEKS:
        df[f"tree_{name}"] = out[name]
    log.info("%s: %d trees in %.0f seconds", ticker, len(df), time.time() - start)
    return df


def sanity_check(df: pd.DataFrame, ticker: str) -> dict:
    cols = [f"{m}_{g}" for m in ("bs", "tree") for g in GREEKS]
    assert np.isfinite(df[cols].values).all(), f"{ticker}: a Greek came out NaN or infinite"

    calls = df[df["option_type"] == "call"]
    puts = df[df["option_type"] == "put"]
    tol = 1e-9

    # Delta's ceiling is e^(-qT), not 1. Since Step 6, q holds the carry implied
    # by put-call parity, which can dip below zero (no dividend plus a little
    # borrow cost or timing noise). Then e^(-qT) sits just above 1 and a deep
    # ITM call can carry a delta of 1.001. That is correct maths, not a bug.
    ceiling = np.maximum(1.0, np.exp(-df["q"] * df["T"]))
    call_cap = ceiling[calls.index]
    put_cap = ceiling[puts.index]

    # Signs and ranges every option must respect, for both models
    for m in ("bs", "tree"):
        assert (calls[f"{m}_delta"] >= -tol).all() and (calls[f"{m}_delta"] <= call_cap + tol).all(), \
            f"{ticker}: {m} call delta outside 0 to e^(-qT)"
        assert (puts[f"{m}_delta"] <= tol).all() and (puts[f"{m}_delta"] >= -put_cap - tol).all(), \
            f"{ticker}: {m} put delta outside -e^(-qT) to 0"
        assert (df[f"{m}_vega"] >= -tol).all(), f"{ticker}: {m} vega negative"
        assert (calls[f"{m}_rho"] >= -tol).all(), f"{ticker}: {m} call rho negative"
        assert (puts[f"{m}_rho"] <= tol).all(), f"{ticker}: {m} put rho positive"
    # Black-Scholes gamma can never be negative. The tree gets a pass on this:
    # where early exercise kicks in, a deep ITM put's value is locked to K - S,
    # so its true gamma is zero and the tree can show tiny negative noise.
    assert (df["bs_gamma"] >= -tol).all(), f"{ticker}: bs gamma negative"
    worst_tree_gamma = df["tree_gamma"].min()
    assert worst_tree_gamma > -1e-3, f"{ticker}: tree gamma {worst_tree_gamma:.4f} is too negative"

    # For OTM contracts with over a week left, early exercise is worth almost
    # nothing, so the American tree and Black-Scholes should agree closely
    clean = df[df["otm"] & (df["days_to_expiry"] > 7)]
    delta_gap = (clean["tree_delta"] - clean["bs_delta"]).abs().median()
    vega_gap = ((clean["tree_vega"] - clean["bs_vega"]).abs() / clean["bs_vega"]).median()
    assert delta_gap < 0.01, f"{ticker}: median OTM delta gap {delta_gap:.4f} is too big"
    assert vega_gap < 0.02, f"{ticker}: median OTM vega gap {vega_gap:.1%} is too big"

    # Gamma should peak near the money on short expiries. On long, high vol
    # expiries the peak drifts well above spot (toward S * e^((r - q + vol^2/2) T)),
    # so we only check under 60 days. Warn only, since a sparse chain can put
    # the peak on the nearest available strike.
    short = df[df["days_to_expiry"] < 60]
    peak = short.loc[short.groupby("expiry")["bs_gamma"].idxmax()]
    far = ((peak["strike"] / peak["spot"] - 1).abs() > 0.10).sum()
    if far:
        log.warning("%s: on %d expiries gamma peaks more than 10%% from spot", ticker, far)

    # Where the models should disagree: deep ITM puts, the early exercise zone
    itm_puts = puts[~puts["otm"]]
    put_delta_gap = (itm_puts["tree_delta"] - itm_puts["bs_delta"]).abs().median() if len(itm_puts) else 0.0

    log.info("%s: sanity checks passed", ticker)
    return {"otm_delta_gap": round(delta_gap, 4), "otm_vega_gap_pct": round(vega_gap * 100, 2),
            "itm_put_delta_gap": round(put_delta_gap, 4),
            "max_call_delta": round(calls["bs_delta"].max(), 4),
            "carry_min_pct": round(df["q"].min() * 100, 2), "carry_max_pct": round(df["q"].max() * 100, 2)}


def main() -> None:
    summary = []
    for ticker in TICKERS:
        files = sorted(PROCESSED_DIR.glob(f"{ticker}_*_iv.parquet"))
        if not files:
            raise FileNotFoundError(f"no IV file for {ticker}, run src.implied_vol first")
        df = pd.read_parquet(files[-1])
        df = df[df["iv_status"] == "ok"].reset_index(drop=True)

        df = add_bs_greeks(df)
        df = add_tree_greeks(df, ticker)
        stats = sanity_check(df, ticker)

        df.to_parquet(PROCESSED_DIR / files[-1].name.replace("_iv", "_greeks"), index=False)
        summary.append({"ticker": ticker, "contracts": len(df), **stats})

    print("\n=== Greeks ===")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\ngap columns: median absolute difference between the tree and Black-Scholes")


if __name__ == "__main__":
    main()
