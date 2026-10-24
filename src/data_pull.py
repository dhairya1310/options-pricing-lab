"""
Step 1: pull option chains from Yahoo Finance and save each pull as a Parquet snapshot.

Run from the project root:
    python -m src.data_pull
    python -m src.data_pull --tickers SPY AAPL
    python -m src.data_pull --strict --skip-if-closed   (what the daily GitHub job runs)

Every snapshot goes to data/raw/<TICKER>_<UTC timestamp>.parquet so we can
rebuild any past surface later. Nothing gets saved if a hard sanity check fails.
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "JPM"]
RAW_DIR = Path("data/raw")
LOG_DIR = Path("logs")

log = logging.getLogger("data_pull")

# yfinance has not caught up with pandas 3 yet and floods the terminal with
# deprecation notices. They do not affect our data, so we hide them.
warnings.filterwarnings("ignore", module="yfinance")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / f"data_pull_{stamp}.log"),
            logging.StreamHandler(),
        ],
    )


def get_spot(tk: yf.Ticker) -> float:
    """
    Latest traded price. fast_info is our main source because after the close
    Yahoo sometimes returns today's history row with NaN prices.
    We cross-check it against the last valid close so a bad quote gets caught.
    """
    closes = tk.history(period="5d")["Close"].dropna()
    if closes.empty:
        raise ValueError("no valid closing prices in the last 5 days")
    last_close = float(closes.iloc[-1])

    try:
        spot = float(tk.fast_info.last_price)
    except Exception:
        spot = float("nan")

    if pd.isna(spot) or spot <= 0:
        log.warning("fast_info gave no usable price, falling back to last close %.2f", last_close)
        return last_close

    # A one day move above 15% is rare enough that we want a human to look at it
    move = abs(spot / last_close - 1)
    if move > 0.15:
        raise ValueError(f"spot {spot:.2f} is {move:.0%} away from last close {last_close:.2f}")
    return spot


def get_dividend_yield(tk: yf.Ticker, spot: float) -> float:
    """
    Trailing 12 month dividend yield, worked out from actual payments.
    Yahoo's own dividendYield field changed units in 2025 (percent vs decimal),
    so we compute it ourselves to avoid a silent 100x error.
    """
    divs = tk.dividends
    if divs.empty:
        return 0.0
    cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
    return float(divs[divs.index >= cutoff].sum() / spot)


def pull_chain(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    spot = get_spot(tk)
    q = get_dividend_yield(tk, spot)
    expiries = tk.options
    if not expiries:
        raise ValueError("Yahoo returned no expiry dates")

    pulled_at = datetime.now(timezone.utc)
    frames = []
    for exp in expiries:
        chain = tk.option_chain(exp)
        for side, df in (("call", chain.calls), ("put", chain.puts)):
            if df.empty:
                continue
            df = df.copy()
            df["option_type"] = side
            df["expiry"] = pd.Timestamp(exp)
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["ticker"] = ticker
    out["spot"] = spot
    out["div_yield"] = q
    out["pulled_at_utc"] = pulled_at
    log.info("%s: spot=%.2f, div_yield=%.4f, %d expiries, %d contracts",
             ticker, spot, q, len(expiries), len(out))
    return out


def sanity_check(df: pd.DataFrame, ticker: str) -> None:
    """
    Hard failures raise and block the save. Soft issues only warn,
    because real market data is messy and Step 2 cleans most of it.
    """
    # Hard checks: if any of these fail, the data is unusable
    assert not df.empty, f"{ticker}: chain is empty"
    assert (df["spot"] > 0).all(), f"{ticker}: spot price is not positive"
    assert (df["strike"] > 0).all(), f"{ticker}: found a non positive strike"
    assert set(df["option_type"]) == {"call", "put"}, f"{ticker}: missing calls or puts"
    assert not df["contractSymbol"].duplicated().any(), f"{ticker}: duplicate contracts"
    assert df["div_yield"].between(0, 0.15).all(), f"{ticker}: dividend yield looks wrong"
    today = pd.Timestamp.now().normalize()
    assert (df["expiry"] >= today).all(), f"{ticker}: found an expiry in the past"

    # Soft checks: worth knowing, not worth stopping for
    both = (df["bid"] > 0) & (df["ask"] > 0)
    crossed = (df.loc[both, "bid"] > df.loc[both, "ask"]).sum()
    if crossed:
        log.warning("%s: %d contracts have bid above ask", ticker, crossed)

    zero_bid_share = (df["bid"] == 0).mean()
    if zero_bid_share > 0.5:
        log.warning("%s: %.0f%% of contracts show a zero bid. Yahoo often blanks "
                    "quotes outside market hours. Re-pull on a weekday afternoon.",
                    ticker, zero_bid_share * 100)

    spot = df["spot"].iloc[0]
    odd = ~df["strike"].between(0.05 * spot, 20 * spot)
    if odd.any():
        log.warning("%s: %d strikes sit far outside a normal range", ticker, odd.sum())

    log.info("%s: sanity checks passed (zero bid share %.0f%%)", ticker, zero_bid_share * 100)


def save_snapshot(df: pd.DataFrame, ticker: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = df["pulled_at_utc"].iloc[0].strftime("%Y%m%d_%H%M%S")
    path = RAW_DIR / f"{ticker}_{stamp}.parquet"
    df.to_parquet(path, index=False)

    # Read it back and make sure nothing got lost on the way to disk
    check = pd.read_parquet(path)
    assert len(check) == len(df), f"{ticker}: row count changed after saving"
    return path


def market_traded_today() -> bool:
    """
    True if US stocks traded today (New York time). We ask Yahoo for SPY's
    recent daily bars: on a weekend or exchange holiday today has no bar.
    The scheduled job uses this so it never saves a duplicate snapshot of
    the previous session.
    """
    hist = yf.Ticker("SPY").history(period="5d")
    if hist.empty:
        raise ValueError("no SPY history came back, cannot tell if the market was open")
    last_bar = hist.index[-1].tz_convert("America/New_York").date()
    today = datetime.now(ZoneInfo("America/New_York")).date()
    return last_bar == today


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--strict", action="store_true",
                        help="exit with an error if any ticker fails, so the scheduled job shows red")
    parser.add_argument("--skip-if-closed", action="store_true",
                        help="do nothing if the market did not trade today")
    args = parser.parse_args()

    setup_logging()
    if args.skip_if_closed and not market_traded_today():
        log.info("The market did not trade today (weekend or holiday). Nothing to pull.")
        return

    summary = []
    for ticker in args.tickers:
        try:
            df = pull_chain(ticker)
            sanity_check(df, ticker)
            path = save_snapshot(df, ticker)
            summary.append({
                "ticker": ticker,
                "spot": round(df["spot"].iloc[0], 2),
                "div_yield": round(df["div_yield"].iloc[0], 4),
                "expiries": df["expiry"].nunique(),
                "contracts": len(df),
                "zero_bid_pct": round((df["bid"] == 0).mean() * 100, 1),
                "file": path.name,
            })
        except Exception as e:
            log.error("%s FAILED: %s", ticker, e)
            summary.append({"ticker": ticker, "file": f"FAILED: {e}"})

    print("\n=== Pull summary ===")
    print(pd.DataFrame(summary).to_string(index=False))

    failed = [s["ticker"] for s in summary if str(s["file"]).startswith("FAILED")]
    if failed and args.strict:
        log.error("Failed tickers: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
