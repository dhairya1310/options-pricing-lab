"""
Step 6: back out the implied volatility of every contract with Brent's method.

Run from the project root:
    python -m src.implied_vol

Implied vol is the one sigma that makes Black-Scholes return the market price.
There is no formula for it, so we search: Brent's method squeezes a bracket
around the answer until the price matches to a tiny fraction of a cent.

Some contracts have no answer at all. Their price sits outside anything
Black-Scholes can produce for any volatility. We never guess for those,
we log them with the reason.

A note on American options: US single stock options can be exercised early,
Black-Scholes assumes they cannot. For out of the money contracts the early
exercise right is worth almost nothing, so their BS implied vol is clean.
Deep in the money puts carry a real early exercise premium, which shows up
as inflated IV or no solution. That is why the surface (Step 8) is built from
out of the money contracts only, the same convention trading desks use.

Output:
    data/processed/<TICKER>_<timestamp>_iv.parquet
    data/processed/iv_failures.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.pricing import bs_price

TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "JPM"]
PROCESSED_DIR = Path("data/processed")

VOL_LOW, VOL_HIGH = 0.001, 5.0   # search between 0.1% and 500% vol
PRICE_TOLERANCE = 1e-6           # how closely the solved IV must reprice the mid
MIN_PRICE_PER_VOL_POINT = 0.001  # below this the price carries no real vol information

log = logging.getLogger("implied_vol")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def implied_vol(price, S, K, T, r, q, kind):
    """
    Returns (iv, status). status is "ok" when solved, otherwise the reason it failed.

    Black-Scholes price rises steadily with vol, so if the market price lies
    between the price at 0.1% vol and the price at 500% vol, exactly one
    answer exists in between and Brent is guaranteed to find it.
    """
    def gap(sigma):
        return bs_price(S, K, T, r, q, sigma, kind) - price

    low_gap, high_gap = gap(VOL_LOW), gap(VOL_HIGH)
    if low_gap > 0:
        # Even near zero vol the model asks more than the market. For puts this
        # is usually the early exercise premium that BS cannot see.
        return np.nan, "below model minimum"
    if high_gap < 0:
        return np.nan, "above model maximum"

    try:
        iv = brentq(gap, VOL_LOW, VOL_HIGH, xtol=1e-10, maxiter=200)
    except (ValueError, RuntimeError) as e:
        return np.nan, f"solver failed: {e}"

    # Deep in the money options barely react to vol. If moving vol a full point
    # changes the price by less than a tenth of a cent, a whole range of vols
    # fits the price equally well and the "answer" is meaningless.
    if abs(gap(iv + 0.01) - gap(iv)) < MIN_PRICE_PER_VOL_POINT:
        return np.nan, "vega too small"
    return iv, "ok"


def solve_chain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    results = [
        implied_vol(row.mid, row.spot, row.strike, row.T, row.r, row.q, row.option_type)
        for row in df.itertuples()
    ]
    df["iv"] = [r[0] for r in results]
    df["iv_status"] = [r[1] for r in results]

    # Out of the money: calls above spot, puts below. These carry almost no
    # early exercise value, so their BS implied vol is the trustworthy one.
    df["otm"] = ((df["option_type"] == "call") & (df["strike"] >= df["spot"])) | \
                ((df["option_type"] == "put") & (df["strike"] < df["spot"]))
    return df


def atm_vol(df: pd.DataFrame, target_days: float = 30) -> float:
    """Average IV of the two OTM contracts nearest the money, nearest 30 days out."""
    good = df[(df["iv_status"] == "ok") & df["otm"]]
    if good.empty:
        return np.nan
    expiry = good.loc[(good["days_to_expiry"] - target_days).abs().idxmin(), "expiry"]
    one = good[good["expiry"] == expiry].copy()
    one["dist"] = (one["strike"] - one["spot"]).abs()
    return one.nsmallest(2, "dist")["iv"].mean()


def sanity_check(df: pd.DataFrame, ticker: str) -> dict:
    solved = df[df["iv_status"] == "ok"]

    # Hard check 1: every solved IV must reprice its contract back to the mid
    repriced = bs_price(solved["spot"].values, solved["strike"].values, solved["T"].values,
                        solved["r"].values, solved["q"].values, solved["iv"].values, "call")
    is_put = (solved["option_type"] == "put").values
    if is_put.any():
        repriced[is_put] = bs_price(
            solved["spot"].values[is_put], solved["strike"].values[is_put], solved["T"].values[is_put],
            solved["r"].values[is_put], solved["q"].values[is_put], solved["iv"].values[is_put], "put")
    worst = np.abs(repriced - solved["mid"].values).max() if len(solved) else 0.0
    assert worst < PRICE_TOLERANCE, f"{ticker}: solved IV reprices {worst:.2e} away from the mid"

    # Hard check 2: solved vols must sit strictly inside the search range
    assert solved["iv"].between(VOL_LOW, VOL_HIGH, inclusive="neither").all(), \
        f"{ticker}: an IV landed on the edge of the search range"

    # Hard check 3: nearly all OTM contracts should solve, they have no early exercise excuse
    otm = df[df["otm"]]
    otm_rate = (otm["iv_status"] == "ok").mean()
    assert otm_rate > 0.95, f"{ticker}: only {otm_rate:.0%} of OTM contracts solved"

    # Soft check: compare with Yahoo's own IV on the OTM contracts we solved.
    # Yahoo uses its own price snapshot and conventions, so we expect close,
    # not identical. A big gap would point to a units or input problem.
    both = solved[solved["otm"] & solved["impliedVolatility"].between(0.01, 5)]
    yahoo_gap = (both["iv"] - both["impliedVolatility"]).abs().median() if len(both) else np.nan
    if yahoo_gap > 0.05:
        log.warning("%s: median gap to Yahoo IV is %.1f vol points, worth a look", ticker, yahoo_gap * 100)

    atm = atm_vol(df)
    if not 0.05 < atm < 1.5:
        log.warning("%s: 30 day ATM vol of %.1f%% looks unusual", ticker, atm * 100)

    log.info("%s: sanity checks passed, worst repricing error %.1e", ticker, worst)
    return {"otm_solved_pct": round(otm_rate * 100, 1),
            "atm_30d_iv_pct": round(atm * 100, 1),
            "median_gap_to_yahoo_pts": round(yahoo_gap * 100, 2)}


def main() -> None:
    summary, failures = [], []
    for ticker in TICKERS:
        files = sorted(PROCESSED_DIR.glob(f"{ticker}_*_ready.parquet"))
        if not files:
            raise FileNotFoundError(f"no ready file for {ticker}, run src.inputs first")
        df = solve_chain(pd.read_parquet(files[-1]))
        stats = sanity_check(df, ticker)

        out = PROCESSED_DIR / files[-1].name.replace("_ready", "_iv")
        df.to_parquet(out, index=False)

        failed = df[df["iv_status"] != "ok"]
        failures.append(failed[["ticker", "contractSymbol", "option_type", "strike", "expiry",
                                "spot", "mid", "otm", "iv_status"]])
        counts = failed["iv_status"].value_counts()
        summary.append({
            "ticker": ticker,
            "contracts": len(df),
            "solved": int((df["iv_status"] == "ok").sum()),
            "below_min": int(counts.get("below model minimum", 0)),
            "above_max": int(counts.get("above model maximum", 0)),
            "vega_tiny": int(counts.get("vega too small", 0)),
            "other_fail": int(len(failed) - counts.get("below model minimum", 0)
                              - counts.get("above model maximum", 0)
                              - counts.get("vega too small", 0)),
            # Failures should be almost all in the money, where vol barely matters
            "failed_itm": int((~failed["otm"]).sum()),
            "failed_otm": int(failed["otm"].sum()),
            **stats,
        })

    pd.concat(failures, ignore_index=True).to_csv(PROCESSED_DIR / "iv_failures.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== Implied volatility ===")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
