"""
Step 3: attach the remaining pricing inputs to every clean contract.

Run from the project root:
    python -m src.inputs

Adds three columns each pricer needs:
    T  time to expiry in years
    r  risk free rate, from the FRED 3-month Treasury (DGS3MO)
    q  dividend yield, from the raw pull

Output:
    data/processed/<TICKER>_<timestamp>_ready.parquet
    data/processed/market_inputs.csv  (one row per ticker, easy to eyeball)
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "JPM"]
CLEAN_DIR = Path("data/clean")
OUT_DIR = Path("data/processed")

SECONDS_PER_YEAR = 365 * 24 * 3600
BILL_DAYS = 91  # a 3-month T-bill runs about 91 days

log = logging.getLogger("inputs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def latest_clean(ticker: str) -> Path:
    files = sorted(CLEAN_DIR.glob(f"{ticker}_*_clean.parquet"))
    if not files:
        raise FileNotFoundError(f"no clean file for {ticker}, run clean first")
    return files[-1]


def get_risk_free_rate(pull_time_utc: pd.Timestamp) -> tuple[float, pd.Timestamp, float]:
    """
    Latest 3-month Treasury yield published on or before the pull date.

    FRED quotes it in percent on a simple annual basis. Black-Scholes wants a
    continuously compounded rate, so we convert. The gap is small (a few basis
    points) but getting units right is the whole point of this project.
    """
    load_dotenv()
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY missing. Check that .env sits in the project root.")

    pull_date = pull_time_utc.tz_convert("America/New_York").tz_localize(None).normalize()
    series = Fred(api_key=key).get_series(
        "DGS3MO",
        observation_start=pull_date - pd.Timedelta(days=30),
        observation_end=pull_date,
    ).dropna()  # FRED leaves holidays as blanks

    if series.empty:
        raise ValueError("FRED returned no DGS3MO values for the last 30 days")

    rate_date = series.index[-1]
    simple = series.iloc[-1] / 100
    continuous = np.log(1 + simple * BILL_DAYS / 365) * 365 / BILL_DAYS
    return continuous, rate_date, simple


def add_inputs(df: pd.DataFrame, r: float) -> pd.DataFrame:
    df = df.copy()

    # US equity options stop trading at 4 PM Eastern on expiry day,
    # so we measure time to that moment, not to midnight
    expiry_close = ((df["expiry"] + pd.Timedelta(hours=16))
                    .dt.tz_localize("America/New_York")
                    .dt.tz_convert("UTC"))
    df["T"] = (expiry_close - df["pulled_at_utc"]).dt.total_seconds() / SECONDS_PER_YEAR
    df["days_to_expiry"] = df["T"] * 365

    df["r"] = r
    # Same reasoning as the rate: convert the simple trailing yield to continuous
    df["q"] = np.log(1 + df["div_yield"])
    return df


def sanity_check(df: pd.DataFrame, ticker: str) -> None:
    # Hard checks
    assert not df[["T", "r", "q"]].isna().any().any(), f"{ticker}: missing T, r or q"
    assert (df["T"] > 0).all(), f"{ticker}: found a contract with no time left"
    assert (df["T"] < 3.5).all(), f"{ticker}: an expiry more than 3.5 years out looks wrong"
    assert df["r"].between(0, 0.15).all(), f"{ticker}: risk free rate outside 0 to 15%"
    assert df["q"].between(0, 0.15).all(), f"{ticker}: dividend yield outside 0 to 15%"

    # Every contract on the same expiry must share exactly one T
    per_expiry = df.groupby("expiry")["T"].nunique()
    assert (per_expiry == 1).all(), f"{ticker}: contracts on one expiry got different T values"

    # Later expiries must have more time left, never less
    ordered = df.drop_duplicates("expiry").sort_values("expiry")["T"]
    assert ordered.is_monotonic_increasing, f"{ticker}: T does not rise with expiry"

    # Cross-check T against a plain calendar day count, should agree within a day
    pull_day = df["pulled_at_utc"].iloc[0].tz_convert("America/New_York").tz_localize(None).normalize()
    calendar_days = (df["expiry"] - pull_day).dt.days
    assert (abs(df["days_to_expiry"] - calendar_days) < 1).all(), f"{ticker}: T disagrees with calendar"

    log.info("%s: sanity checks passed", ticker)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # All tickers were pulled within seconds of each other, so one rate covers them all
    first = pd.read_parquet(latest_clean(TICKERS[0]))
    pull_time = first["pulled_at_utc"].iloc[0]
    r, rate_date, simple = get_risk_free_rate(pull_time)

    stale_days = (pull_time.tz_convert("America/New_York").tz_localize(None).normalize() - rate_date).days
    if stale_days > 5:
        log.warning("latest DGS3MO is %d days old, FRED may be behind", stale_days)
    log.info("DGS3MO on %s: %.2f%% simple, %.4f continuous", rate_date.date(), simple * 100, r)

    summary = []
    for ticker in TICKERS:
        path = latest_clean(ticker)
        df = add_inputs(pd.read_parquet(path), r)
        sanity_check(df, ticker)
        out = OUT_DIR / path.name.replace("_clean", "_ready")
        df.to_parquet(out, index=False)
        summary.append({
            "ticker": ticker,
            "spot": round(df["spot"].iloc[0], 2),
            "r": round(r, 4),
            "rate_date": rate_date.date(),
            "q": round(df["q"].iloc[0], 4),
            "shortest_days": round(df["days_to_expiry"].min(), 2),
            "longest_years": round(df["T"].max(), 2),
            "contracts": len(df),
        })

    table = pd.DataFrame(summary)
    table.to_csv(OUT_DIR / "market_inputs.csv", index=False)
    print("\n=== Market inputs ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
