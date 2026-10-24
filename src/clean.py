"""
Step 2: clean the raw option chains so every contract left is one we can trust.

Run from the project root:
    python -m src.clean

Takes the latest raw snapshot for each ticker, uses the bid/ask midpoint as the
market price, and drops contracts one filter at a time. Every filter logs how
many contracts it removed, so the cleaning is fully traceable.

Output:
    data/clean/<TICKER>_<timestamp>_clean.parquet
    data/clean/filter_log.csv
"""

import logging
from pathlib import Path

import pandas as pd

TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "JPM"]
RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")

MIN_OPEN_INTEREST = 100
MAX_SPREAD_PCT = 0.20  # bid/ask spread as a share of the midpoint
MAX_DAYS_SINCE_TRADE = 7  # older than this and the quote is probably frozen

log = logging.getLogger("clean")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def latest_snapshot(ticker: str) -> Path:
    files = sorted(RAW_DIR.glob(f"{ticker}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no raw snapshot found for {ticker}, run data_pull first")
    return files[-1]  # file names carry a timestamp, so the last one is the newest


def drop_and_log(df, drop_mask, name, ticker, log_rows):
    """Remove the flagged rows and write down exactly how many went."""
    removed = int(drop_mask.sum())
    kept = df[~drop_mask].copy()
    log_rows.append({"ticker": ticker, "filter": name, "removed": removed, "remaining": len(kept)})
    log.info("%s | %-28s removed %5d, %5d left", ticker, name, removed, len(kept))
    return kept


def clean_chain(raw: pd.DataFrame, ticker: str, log_rows: list) -> pd.DataFrame:
    df = raw.copy()
    df["bid"] = df["bid"].fillna(0.0)
    df["ask"] = df["ask"].fillna(0.0)
    df["openInterest"] = df["openInterest"].fillna(0)

    log_rows.append({"ticker": ticker, "filter": "start", "removed": 0, "remaining": len(df)})

    # Contracts that expired today are worthless for pricing. Time to expiry would be zero.
    pull_date = (df["pulled_at_utc"].iloc[0].tz_convert("America/New_York")
                 .normalize().tz_localize(None))
    df = drop_and_log(df, df["expiry"] <= pull_date, "expired on pull date", ticker, log_rows)

    # Yahoo keeps listing some contracts long after anyone stopped trading them.
    # NVDA still shows pre-split strikes last traded in June 2024. A quote with
    # no trade behind it for over a week is not a real market price.
    cutoff = df["pulled_at_utc"].iloc[0] - pd.Timedelta(days=MAX_DAYS_SINCE_TRADE)
    stale = df["lastTradeDate"].isna() | (df["lastTradeDate"] < cutoff)
    df = drop_and_log(df, stale, "no trade in last 7 days", ticker, log_rows)

    # Filters from the project spec
    df = drop_and_log(df, df["bid"] <= 0, "zero bid", ticker, log_rows)
    df = drop_and_log(df, df["ask"] < df["bid"], "ask below bid", ticker, log_rows)
    df = drop_and_log(df, df["openInterest"] < MIN_OPEN_INTEREST, "open interest under 100", ticker, log_rows)

    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
    df = drop_and_log(df, df["spread_pct"] > MAX_SPREAD_PCT, "spread over 20% of mid", ticker, log_rows)

    # If even the ask sits below intrinsic value, the quote is stale.
    # No model can price it and the IV solver would fail on it later.
    intrinsic = (df["spot"] - df["strike"]).clip(lower=0)
    intrinsic = intrinsic.where(df["option_type"] == "call",
                                (df["strike"] - df["spot"]).clip(lower=0))
    df["intrinsic"] = intrinsic
    df = drop_and_log(df, df["ask"] < df["intrinsic"] - 0.01, "ask below intrinsic", ticker, log_rows)

    # The other side of the no-arbitrage range: a call can never be worth more
    # than the stock itself, and a put can never be worth more than its strike
    too_rich = ((df["option_type"] == "call") & (df["mid"] > df["spot"])) | \
               ((df["option_type"] == "put") & (df["mid"] > df["strike"]))
    df = drop_and_log(df, too_rich, "above no-arbitrage max", ticker, log_rows)

    df["moneyness"] = df["strike"] / df["spot"]
    return df.reset_index(drop=True)


def sanity_check(clean: pd.DataFrame, raw_count: int, ticker: str, log_rows: list) -> None:
    # Hard checks: the clean data must actually obey every rule we applied
    assert not clean.empty, f"{ticker}: nothing survived cleaning"
    assert set(clean["option_type"]) == {"call", "put"}, f"{ticker}: lost all calls or all puts"
    assert (clean["bid"] > 0).all(), f"{ticker}: a zero bid slipped through"
    assert (clean["ask"] >= clean["bid"]).all(), f"{ticker}: crossed quote slipped through"
    assert (clean["openInterest"] >= MIN_OPEN_INTEREST).all(), f"{ticker}: low OI slipped through"
    assert (clean["spread_pct"] <= MAX_SPREAD_PCT + 1e-12).all(), f"{ticker}: wide spread slipped through"
    assert clean["mid"].between(clean["bid"], clean["ask"]).all(), f"{ticker}: mid outside bid/ask"
    assert not clean[["strike", "mid", "expiry", "spot"]].isna().any().any(), f"{ticker}: missing values"

    # No-arbitrage upper bounds: a call cannot cost more than the stock,
    # and a put cannot cost more than its strike
    calls = clean[clean["option_type"] == "call"]
    puts = clean[clean["option_type"] == "put"]
    assert (calls["mid"] <= calls["spot"]).all(), f"{ticker}: call priced above the stock"
    assert (puts["mid"] <= puts["strike"]).all(), f"{ticker}: put priced above its strike"

    # Bookkeeping: everything removed plus everything kept must equal what we started with
    removed = sum(r["removed"] for r in log_rows if r["ticker"] == ticker)
    assert removed + len(clean) == raw_count, f"{ticker}: row counts do not add up"

    # Soft checks: the vol surface needs enough points per expiry
    thin = clean.groupby("expiry").size()
    thin = thin[thin < 5]
    if len(thin):
        log.warning("%s: %d expiries have fewer than 5 contracts left, the surface will skip them",
                    ticker, len(thin))
    log.info("%s: sanity checks passed, %d of %d contracts kept (%.0f%%)",
             ticker, len(clean), raw_count, 100 * len(clean) / raw_count)


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    log_rows = []

    for ticker in TICKERS:
        path = latest_snapshot(ticker)
        raw = pd.read_parquet(path)
        clean = clean_chain(raw, ticker, log_rows)
        sanity_check(clean, len(raw), ticker, log_rows)
        out = CLEAN_DIR / f"{path.stem}_clean.parquet"
        clean.to_parquet(out, index=False)

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(CLEAN_DIR / "filter_log.csv", index=False)

    # One table showing what each filter did to each ticker
    order = list(dict.fromkeys(log_df["filter"]))
    table = log_df.pivot(index="ticker", columns="filter", values="removed")[order]
    table["start"] = log_df[log_df["filter"] == "start"].set_index("ticker")["remaining"]
    table["kept"] = log_df.groupby("ticker")["remaining"].last()
    table = table.loc[TICKERS]

    pd.set_option("display.width", 200)
    print("\n=== Contracts removed by each filter ===")
    print(table.to_string())


if __name__ == "__main__":
    main()
