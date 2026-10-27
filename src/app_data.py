"""
Shared data loading for the Streamlit app.

The app never calls Yahoo or FRED. It reads the files the pipeline already
saved, which keeps it fast and means the deployed version needs no API keys.
Everything is cached, so switching pages does not reload anything.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.surface import svi_w

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "JPM"]
SKEW_MAX_DAYS_AWAY = 30  # how far from 30 days we look for an expiry with both skew strikes quoted

# The daily job commits fresh results every evening. Cached data expires after
# an hour, so the live app picks up a new snapshot without a manual restart.
CACHE_SECONDS = 3600


def _latest(pattern: str) -> Path:
    files = sorted(PROCESSED.glob(pattern))
    if not files:
        st.error(f"Missing data file matching {pattern}. Run the pipeline first (see the README).")
        st.stop()
    return files[-1]


@st.cache_data(ttl=CACHE_SECONDS)
def load_chain(ticker: str) -> pd.DataFrame:
    """Every clean contract with its implied vol, forward and carry."""
    return pd.read_parquet(_latest(f"{ticker}_*_iv.parquet"))


@st.cache_data(ttl=CACHE_SECONDS)
def load_svi(ticker: str) -> pd.DataFrame:
    """SVI parameters per expiry, only the ones that fit cleanly."""
    p = pd.read_csv(PROCESSED / f"{ticker}_svi_params.csv", parse_dates=["expiry"])
    p = p[p["status"] == "ok"].copy()
    p["atm_vol"] = np.sqrt(svi_w(0.0, p["a"], p["b"], p["rho"], p["m"], p["sigma"]) / p["T"])
    return p.sort_values("T").reset_index(drop=True)


@st.cache_data(ttl=CACHE_SECONDS)
def load_surface(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{ticker}_surface.parquet")


@st.cache_data(ttl=CACHE_SECONDS)
def load_model_vs_market() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "model_vs_market.parquet")


@st.cache_data(ttl=CACHE_SECONDS)
def load_parity() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "parity_pairs.parquet")


def smile_vol(params: pd.Series, strikes, forward: float) -> np.ndarray:
    """SVI implied vol at the given strikes for one expiry."""
    k = np.log(np.asarray(strikes, dtype=float) / forward)
    w = svi_w(k, params["a"], params["b"], params["rho"], params["m"], params["sigma"])
    return np.sqrt(np.maximum(w, 1e-12) / params["T"])


def skew_near_30d(ticker: str) -> tuple[float, float]:
    """
    Vol at a 95% strike minus a 105% strike, in vol points, and the days to
    expiry it was measured at.

    Uses the expiry closest to 30 days that has real quotes at both strikes.
    The closest one is not always usable: on October 26 JPM's 25 day expiry
    had no quotes up at 105%, so we move to the next expiry rather than
    extrapolate. Callers show the tenor when it is not near 30 days.
    """
    g = load_surface(ticker)
    expiries = g.drop_duplicates("expiry").assign(dist=lambda x: (x["days"] - 30).abs()).sort_values("dist")
    for e in expiries.itertuples():
        if e.dist > SKEW_MAX_DAYS_AWAY:
            break
        one = g[g["expiry"] == e.expiry]
        low = one[np.isclose(one["moneyness"], 0.95)]
        high = one[np.isclose(one["moneyness"], 1.05)]
        if low.empty or high.empty or low["extrapolated"].iloc[0] or high["extrapolated"].iloc[0]:
            continue
        return float((low["iv"].iloc[0] - high["iv"].iloc[0]) * 100), float(e.days)
    return np.nan, np.nan


def skew_30d(ticker: str) -> float:
    return skew_near_30d(ticker)[0]


def skew_label(ticker: str) -> str:
    """Skew for display: '9.4 pts', '3.4 pts (53d)' when not near 30 days, or 'n/a'."""
    value, days = skew_near_30d(ticker)
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.1f} pts" + ("" if abs(days - 30) <= 7 else f" ({days:.0f}d)")


def atm_vol_30d(ticker: str) -> float:
    """SVI vol at the forward, interpolated in total variance to exactly 30 days."""
    p = load_svi(ticker)
    w = p["atm_vol"] ** 2 * p["T"]
    t = 30 / 365
    if t <= p["T"].iloc[0]:
        return float(p["atm_vol"].iloc[0])
    return float(np.sqrt(np.interp(t, p["T"], w) / t))


def as_of() -> str:
    """When the data in the app was pulled, in New York time."""
    stamp = load_chain("SPY")["pulled_at_utc"].iloc[0]
    return pd.Timestamp(stamp).tz_convert("America/New_York").strftime("%b %d, %Y %I:%M %p ET")


def expiry_label(expiry, days) -> str:
    return f"{pd.Timestamp(expiry).date()}  ({days:.0f} days)"
