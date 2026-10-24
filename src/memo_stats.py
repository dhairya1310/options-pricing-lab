"""
Step 14 prep: every number the desk memo and README quote, computed in one place.

Run from the project root:
    python -m src.memo_stats
    python -m src.memo_stats --earnings AAPL=2026-10-29 NVDA=2026-11-18   (if Yahoo has no date)

Nothing in the memo is typed in by hand. It all comes from this output, so
when a new snapshot comes in the memo can be refreshed from the same script.

What it measures per ticker:
    vol level      ATM vol at 7, 30, 90, 180 and 365 days (SVI at the forward,
                   interpolated in total variance, which is the correct way)
    skew           vol at a 95% strike minus a 105% strike, at 30 and 90 days
    term slope     1 year ATM vol minus 30 day ATM vol
    earnings       next earnings date, the ATM vol of the expiries either side,
                   and the earnings move the options imply (method below)
    model error    flat vol BS error on deep puts, tree vs BS on ITM puts
    parity         the funnel and the implied funding rate

Implied earnings move. An expiry after earnings carries normal day to day
variance plus one extra jump on the earnings day. Taking the normal vol from
the last expiry before earnings:
    jump variance = w_after - sigma_before^2 * T_after,   w = vol^2 * T
    implied move  = sqrt(jump variance), a one standard deviation move
This is the standard desk method. It assumes the normal vol is flat between
the two expiries, so it is an estimate, not an exact number.

Output:
    reports/memo_stats.json
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.surface import svi_w

TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "JPM"]
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
TENORS = [7, 30, 90, 180, 365]

log = logging.getLogger("memo_stats")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
warnings.filterwarnings("ignore", module="yfinance")


def load_svi(ticker: str) -> pd.DataFrame:
    p = pd.read_csv(PROCESSED_DIR / f"{ticker}_svi_params.csv", parse_dates=["expiry"])
    p = p[p["status"] == "ok"].sort_values("T").reset_index(drop=True)
    p["atm_vol"] = np.sqrt(svi_w(0.0, p["a"], p["b"], p["rho"], p["m"], p["sigma"]) / p["T"])
    p["w_atm"] = p["atm_vol"] ** 2 * p["T"]
    return p


def atm_at(svi: pd.DataFrame, days: float) -> float:
    """ATM vol at any tenor, interpolating total variance linearly in time."""
    t = days / 365
    if t < svi["T"].iloc[0] or t > svi["T"].iloc[-1]:
        return np.nan  # never extrapolate past the listed expiries
    return float(np.sqrt(np.interp(t, svi["T"], svi["w_atm"]) / t))


def skew_at(ticker: str, days: float) -> float:
    """Vol at a 95% strike minus a 105% strike, in vol points, on the expiry nearest the tenor."""
    g = pd.read_parquet(PROCESSED_DIR / f"{ticker}_surface.parquet")
    expiry = g.loc[(g["days"] - days).abs().idxmin(), "expiry"]
    one = g[g["expiry"] == expiry]
    lo, hi = one[np.isclose(one["moneyness"], 0.95)], one[np.isclose(one["moneyness"], 1.05)]
    if lo.empty or hi.empty or lo["extrapolated"].iloc[0] or hi["extrapolated"].iloc[0]:
        return np.nan
    return float((lo["iv"].iloc[0] - hi["iv"].iloc[0]) * 100)


def earnings_dates(ticker: str, pull_day: pd.Timestamp) -> tuple:
    """Next and last earnings dates from Yahoo. Returns (next, last), either can be None."""
    if ticker == "SPY":
        return None, None
    try:
        import yfinance as yf
        dates = yf.Ticker(ticker).get_earnings_dates(limit=12)
        days = pd.to_datetime(dates.index).tz_localize(None).normalize()
        future = sorted(d for d in days if d >= pull_day)
        past = sorted(d for d in days if d < pull_day)
        return (future[0] if future else None), (past[-1] if past else None)
    except Exception as e:
        log.warning("%s: no earnings dates from Yahoo (%s), pass --earnings to set one", ticker, e)
        return None, None


def atm_by_expiry(svi: pd.DataFrame, chain: pd.DataFrame) -> pd.DataFrame:
    """
    ATM vol for every expiry. SVI where the fit was clean. Elsewhere, often the
    one week expiries right before earnings, which have too few strikes for a
    five parameter fit, the average implied vol of the call and put nearest
    the forward, read straight off the quotes.
    """
    rows = []
    good = chain[(chain["iv_status"] == "ok")]
    for expiry, one in good.groupby("expiry"):
        hit = svi[svi["expiry"] == expiry]
        if len(hit):
            vol, source = float(hit["atm_vol"].iloc[0]), "svi"
        else:
            near = one.assign(dist=(one["strike"] - one["forward"]).abs()).nsmallest(2, "dist")
            vol, source = float(near["iv"].mean()), "quotes"
        T = float(one["T"].iloc[0])
        rows.append({"expiry": expiry, "T": T, "atm_vol": vol, "w_atm": vol ** 2 * T, "source": source})
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def implied_move(atm: pd.DataFrame, event: pd.Timestamp) -> dict:
    """
    Earnings move implied by the expiries either side of the event.

    Assumes the company reports after the close, as AAPL, NVDA and TSLA do. An
    option expiring on the report day stops trading at 4 PM, before the news,
    so it counts as a before expiry.
    """
    before = atm[atm["expiry"] <= event]
    after = atm[atm["expiry"] > event]
    if before.empty or after.empty:
        return {"status": "no expiry on one side of the event"}
    b, a = before.iloc[-1], after.iloc[0]
    jump_var = a["w_atm"] - b["atm_vol"] ** 2 * a["T"]
    out = {
        "expiry_before": str(b["expiry"].date()), "atm_before_pct": round(b["atm_vol"] * 100, 1),
        "expiry_after": str(a["expiry"].date()), "atm_after_pct": round(a["atm_vol"] * 100, 1),
        "vol_source": f"{b['source']} / {a['source']}",
    }
    if len(after) > 1:
        out["atm_next_pct"] = round(after.iloc[1]["atm_vol"] * 100, 1)
    if jump_var <= 0:
        out["status"] = "no earnings premium visible"
        return out
    out["status"] = "ok"
    out["implied_move_pct"] = round(float(np.sqrt(jump_var)) * 100, 2)
    return out


def ticker_stats(ticker: str, overrides: dict, pull_day: pd.Timestamp, mvm, parity) -> dict:
    svi = load_svi(ticker)
    chain = pd.read_parquet(sorted(PROCESSED_DIR.glob(f"{ticker}_*_iv.parquet"))[-1])
    s = {"spot": round(float(chain["spot"].iloc[0]), 2), "expiries_fit": len(svi),
         "median_fit_rmse_pts": round(float(svi["rmse_vol_pts"].median()), 2)}

    for d in TENORS:
        v = atm_at(svi, d)
        s[f"atm_{d}d_pct"] = round(v * 100, 1) if np.isfinite(v) else None
    for d in (30, 90):
        v = skew_at(ticker, d)
        s[f"skew_{d}d_pts"] = round(v, 1) if np.isfinite(v) else None
    if s["atm_30d_pct"] and s["atm_365d_pct"]:
        s["term_slope_pts"] = round(s["atm_365d_pct"] - s["atm_30d_pct"], 1)

    # The single biggest bump: an expiry whose ATM vol beats both neighbours
    v = svi["atm_vol"].values
    # Half a vol point above both neighbours is the bar, below that it is noise
    bumps = [(i, v[i] - (v[i - 1] + v[i + 1]) / 2) for i in range(1, len(v) - 1)
             if v[i] > v[i - 1] and v[i] > v[i + 1] and v[i] - (v[i - 1] + v[i + 1]) / 2 > 0.005]
    if bumps:
        i, size = max(bumps, key=lambda x: x[1])
        s["biggest_bump"] = {"expiry": str(svi["expiry"].iloc[i].date()),
                             "days": round(float(svi["days"].iloc[i]), 1),
                             "atm_pct": round(v[i] * 100, 1), "above_neighbours_pts": round(size * 100, 2)}

    nxt, last = earnings_dates(ticker, pull_day)
    if ticker in overrides:
        nxt = overrides[ticker]
    s["next_earnings"] = str(nxt.date()) if nxt is not None else None
    s["last_earnings"] = str(last.date()) if last is not None else None
    if nxt is not None:
        s["earnings"] = implied_move(atm_by_expiry(svi, chain), nxt)

    m = mvm[mvm["ticker"] == ticker]
    deep = m[(m["option_type"] == "put") & (m["moneyness"] < 0.90)]
    itm = m[(m["option_type"] == "put") & (m["moneyness"] > 1.05) & (m["days_to_expiry"] > 30)
            & ~m["smile_extrapolated"]]
    s["bs_err_deep_puts_pct"] = round(float(deep["bs_rel_err"].median()) * 100, 1) if len(deep) else None
    if len(itm):
        s["itm_puts"] = len(itm)
        s["itm_put_bs_err_usd"] = round(float(itm["bs_smile_err"].abs().median()), 2)
        s["itm_put_tree_err_usd"] = round(float(itm["tree_smile_err"].abs().median()), 2)
        s["tree_wins_pct"] = round(float((itm["tree_smile_err"].abs() < itm["bs_smile_err"].abs()).mean()) * 100, 0)

    p = parity[parity["ticker"] == ticker]
    rates = p.drop_duplicates("expiry").dropna(subset=["rate_implied"])
    s["parity_pairs"] = len(p)
    s["parity_raw_pct"] = round(float(p["beats_cost"].mean()) * 100, 1)
    s["parity_final_pct"] = round(float(p["beats_cost_ex_rate"].mean()) * 100, 1)
    s["true_arb"] = int(p["true_arb"].sum())
    if len(rates):
        s["funding_spread_bp"] = round(float((rates["rate_implied"] - rates["r"]).median()) * 1e4, 0)
    return s


def sanity_check(stats: dict) -> None:
    for t, s in stats["tickers"].items():
        assert s["atm_30d_pct"] and 3 < s["atm_30d_pct"] < 200, f"{t}: 30 day ATM vol looks wrong"
        for d in TENORS:
            v = s.get(f"atm_{d}d_pct")
            assert v is None or 3 < v < 300, f"{t}: {d} day ATM vol {v} looks wrong"
        e = s.get("earnings", {})
        if e.get("status") == "ok":
            assert 0 < e["implied_move_pct"] < 40, f"{t}: implied earnings move {e['implied_move_pct']}% looks wrong"
            assert pd.Timestamp(e["expiry_before"]) <= pd.Timestamp(s["next_earnings"]) < pd.Timestamp(e["expiry_after"]), \
                f"{t}: earnings date is not between the two expiries used"
    log.info("sanity checks passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--earnings", nargs="*", default=[], help="override, for example AAPL=2026-10-29")
    args = parser.parse_args()
    overrides = {k: pd.Timestamp(v) for k, v in (x.split("=") for x in args.earnings)}

    first = pd.read_parquet(sorted(PROCESSED_DIR.glob("SPY_*_iv.parquet"))[-1])
    pulled = pd.Timestamp(first["pulled_at_utc"].iloc[0]).tz_convert("America/New_York")
    pull_day = pulled.tz_localize(None).normalize()
    mvm = pd.read_parquet(PROCESSED_DIR / "model_vs_market.parquet")
    parity = pd.read_parquet(PROCESSED_DIR / "parity_pairs.parquet")

    stats = {
        "as_of": pulled.strftime("%Y-%m-%d %H:%M ET"),
        "risk_free_pct": round(float(first["r"].iloc[0]) * 100, 2),
        "contracts": int(sum(len(pd.read_parquet(sorted(PROCESSED_DIR.glob(f"{t}_*_iv.parquet"))[-1]))
                             for t in TICKERS)),
        "tickers": {t: ticker_stats(t, overrides, pull_day, mvm, parity) for t in TICKERS},
    }
    rates = parity.drop_duplicates(["ticker", "expiry"]).dropna(subset=["rate_implied"])
    long = rates[rates["days_to_expiry"] > 365]
    stats["funding"] = {
        "median_spread_bp": round(float((rates["rate_implied"] - rates["r"]).median()) * 1e4, 0),
        "expiries": len(rates),
        "over_1y_range_pct": [round(float(long["rate_implied"].min()) * 100, 2),
                              round(float(long["rate_implied"].max()) * 100, 2)] if len(long) else None,
    }
    funnel = parity
    stats["parity_funnel"] = {
        "pairs": len(funnel),
        "beat_spread": int(funnel["beats_cost"].sum()),
        "with_market_forward": int((funnel["beats_cost"] & funnel["beats_cost_mkt_fwd"]).sum()),
        "after_early_exercise": int((funnel["beats_cost"] & funnel["beats_cost_mkt_fwd"]
                                     & funnel["beats_cost_ex_early"]).sum()),
        "after_funding_rate": int((funnel["beats_cost"] & funnel["beats_cost_mkt_fwd"]
                                   & funnel["beats_cost_ex_early"] & funnel["beats_cost_ex_rate"]).sum()),
        "true_arbitrage": int(funnel["true_arb"].sum()),
    }
    sanity_check(stats)

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "memo_stats.json").write_text(json.dumps(stats, indent=2))

    rows = []
    for t, s in stats["tickers"].items():
        e = s.get("earnings", {})
        rows.append({"ticker": t, "atm7": s["atm_7d_pct"], "atm30": s["atm_30d_pct"], "atm90": s["atm_90d_pct"],
                     "atm1y": s["atm_365d_pct"], "slope": s.get("term_slope_pts"), "skew30": s["skew_30d_pts"],
                     "skew90": s["skew_90d_pts"], "next_earn": s["next_earnings"],
                     "earn_move%": e.get("implied_move_pct", e.get("status")),
                     "bump": (s.get("biggest_bump") or {}).get("expiry"),
                     "fund_bp": s.get("funding_spread_bp")})
    pd.set_option("display.width", 220)
    print("\n=== Memo stats ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nFull detail in {REPORT_DIR / 'memo_stats.json'}")


if __name__ == "__main__":
    main()
