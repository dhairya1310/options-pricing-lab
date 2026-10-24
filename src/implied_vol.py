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
PARITY_STRIKES = 6               # strikes nearest spot used to imply the forward
PARITY_MIN_PAIRS = 2             # below this an expiry keeps the trailing dividend

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
    df = add_implied_carry(df)
    results = [
        implied_vol(row.mid, row.spot, row.strike, row.T, row.r, row.q, row.option_type)
        for row in df.itertuples()
    ]
    df["iv"] = [r[0] for r in results]
    df["iv_status"] = [r[1] for r in results]

    # Out of the money relative to the forward, the price the market expects
    # the stock to trade at on expiry. These carry almost no early exercise
    # value, so their BS implied vol is the trustworthy one.
    df["otm"] = ((df["option_type"] == "call") & (df["strike"] >= df["forward"])) | \
                ((df["option_type"] == "put") & (df["strike"] < df["forward"]))
    return df


def add_implied_carry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Read the forward price from the market instead of computing it.

    Put-call parity says C - P = e^(-rT) * (F - K), so every strike that has
    both a call and a put gives us F = K + e^(rT) * (C - P). We take the median
    over the strikes nearest the money, where quotes are tightest and early
    exercise matters least.

    Why bother: our own forward (spot, rate, trailing dividend) missed in two
    ways. SPY options keep trading until 4:15 PM while our spot is the 4 PM
    close, and a trailing dividend yield is not what the market expects over
    each expiry. Either error shows up as calls and puts implying different
    vols at the same strike, a visible jump in the smile right at the money.

    From the forward we back out an implied carry per expiry and store it in q,
    so every later step (IV, Greeks, surface) uses the market's forward.
    The original trailing yield is kept as q_trailing.
    """
    df = df.copy()
    df["q_trailing"] = df["q"]
    df["forward_source"] = "trailing dividend"
    df["forward"] = df["spot"] * np.exp((df["r"] - df["q"]) * df["T"])

    for expiry, one in df.groupby("expiry"):
        calls = one[one["option_type"] == "call"].set_index("strike")["mid"]
        puts = one[one["option_type"] == "put"].set_index("strike")["mid"]
        pairs = pd.DataFrame({"call": calls, "put": puts}).dropna()
        if len(pairs) < PARITY_MIN_PAIRS:
            continue
        spot, T, r = one["spot"].iloc[0], one["T"].iloc[0], one["r"].iloc[0]

        nearest = pairs.loc[(pairs.index.to_series() - spot).abs().nsmallest(PARITY_STRIKES).index]
        forwards = nearest.index.values + np.exp(r * T) * (nearest["call"] - nearest["put"]).values
        forward = float(np.median(forwards))

        rows = df["expiry"] == expiry
        df.loc[rows, "forward"] = forward
        df.loc[rows, "q"] = r - np.log(forward / spot) / T
        df.loc[rows, "forward_source"] = "put-call parity"
    return df


def atm_vol(df: pd.DataFrame, target_days: float = 30) -> float:
    """Average IV of the two OTM contracts nearest the money, nearest 30 days out."""
    good = df[(df["iv_status"] == "ok") & df["otm"]]
    if good.empty:
        return np.nan
    expiry = good.loc[(good["days_to_expiry"] - target_days).abs().idxmin(), "expiry"]
    one = good[good["expiry"] == expiry].copy()
    one["dist"] = (one["strike"] - one["forward"]).abs()
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

    # The parity forward should sit close to the forward we computed ourselves.
    # A big gap would mean a bad quote dragged the median, not a real carry.
    per_expiry = df.drop_duplicates("expiry")
    own_forward = per_expiry["spot"] * np.exp((per_expiry["r"] - per_expiry["q_trailing"]) * per_expiry["T"])
    forward_gap = ((per_expiry["forward"] / own_forward - 1).abs() * 100)
    if (forward_gap > 2).any():
        log.warning("%s: %d expiries have a parity forward over 2%% from our own",
                    ticker, int((forward_gap > 2).sum()))

    # The test that matters: at the strike nearest the forward, the call and the
    # put should now imply nearly the same vol. This gap is what showed up as a
    # jump in the smile before we used the parity forward.
    gaps = []
    for _, one in solved.groupby("expiry"):
        both = one.pivot_table(index="strike", columns="option_type", values="iv").dropna()
        if both.empty:
            continue
        k_atm = (both.index.to_series() - one["forward"].iloc[0]).abs().idxmin()
        gaps.append(abs(both.loc[k_atm, "call"] - both.loc[k_atm, "put"]) * 100)
    atm_gap = float(np.median(gaps)) if gaps else np.nan
    if atm_gap > 1.0:
        log.warning("%s: calls and puts still disagree by %.2f vol pts at the money", ticker, atm_gap)

    log.info("%s: sanity checks passed, worst repricing error %.1e", ticker, worst)
    return {"otm_solved_pct": round(otm_rate * 100, 1),
            "atm_30d_iv_pct": round(atm * 100, 1),
            "atm_call_put_gap_pts": round(atm_gap, 2),
            "parity_fwd_pct": round((per_expiry["forward_source"] == "put-call parity").mean() * 100, 0),
            "max_fwd_shift_pct": round(forward_gap.max(), 2),
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
