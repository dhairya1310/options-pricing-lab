"""
Step 8: build a smooth volatility surface by fitting an SVI curve to every expiry.

Run from the project root:
    python -m src.surface

Raw implied vols are noisy and only exist at listed strikes. SVI (Stochastic
Volatility Inspired, Gatheral 2004) is a five parameter curve that trading desks
fit to each expiry to turn those scattered points into a smooth smile:

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

    k  log moneyness, ln(strike / forward)
    w  total implied variance, iv^2 * T

    a      overall level of variance
    b      how steep the wings are
    rho    tilt of the smile (negative means downside puts are pricier, the skew)
    m      where the bottom of the smile sits
    sigma  how rounded the bottom is

Only out of the money contracts go into the fit (see Step 6 for why).

Output:
    data/processed/<TICKER>_svi_params.csv   one row per expiry
    data/processed/<TICKER>_surface.parquet  fitted vol on a moneyness by expiry grid
    reports/figures/<TICKER>_smiles.png      market points vs fitted curves
    reports/figures/<TICKER>_surface.html    interactive 3D preview
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import least_squares, minimize

TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "JPM"]
PROCESSED_DIR = Path("data/processed")
FIG_DIR = Path("reports/figures")

MIN_POINTS = 6             # fewer than this and five parameters would overfit
MAX_RMSE_VOL_PTS = 3.0     # fits worse than this get flagged
MONEYNESS_GRID = np.linspace(0.70, 1.30, 61)  # strike / spot, for the surface output

log = logging.getLogger("surface")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# SVI maths
# ---------------------------------------------------------------------------

def svi_w(k, a, b, rho, m, sigma):
    """Total implied variance at log moneyness k."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_density_check(k, a, b, rho, m, sigma):
    """
    Gatheral's butterfly condition g(k). If g goes negative anywhere, the curve
    implies a negative probability for the stock landing there, which means a
    free butterfly spread. A usable fit keeps g >= 0 over the strikes we show.
    """
    root = np.sqrt((k - m) ** 2 + sigma ** 2)
    w = svi_w(k, a, b, rho, m, sigma)
    w1 = b * (rho + (k - m) / root)          # first derivative of w
    w2 = b * sigma ** 2 / root ** 3          # second derivative of w
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


def _linear_part(k, w, m, sigma):
    """
    The quasi-explicit trick (Zeliade Systems, 2009). Once m and sigma are fixed,
    SVI is a straight line regression in three numbers:
        w = a + d * y + c * sqrt(y^2 + 1),   y = (k - m) / sigma
    with c = b * sigma and d = rho * b * sigma. So we solve those three exactly
    and only search over two parameters, which is fast and stable.
    """
    y = (k - m) / sigma
    root = np.sqrt(y * y + 1)
    X = np.column_stack([np.ones_like(y), y, root])
    a, d, c = np.linalg.lstsq(X, w, rcond=None)[0]
    # Keep b non negative and |rho| below 1, then refit the level to match
    c = max(c, 0.0)
    d = np.clip(d, -0.999 * c, 0.999 * c)
    # Roger Lee's moment bound: far from the money, total variance cannot grow
    # faster than 2 per unit of log moneyness. Here that reads c + |d| <= 2 * sigma.
    # Without it the curve can hug the quotes and explode just outside them.
    if c + abs(d) > 2 * sigma:
        shrink = 2 * sigma / (c + abs(d))
        c, d = c * shrink, d * shrink
    a = np.mean(w - d * y - c * root)
    # The lowest point of the curve is a + sqrt(c^2 - d^2). Keeping that above
    # zero guarantees positive variance at every strike, not just the quoted ones.
    a = max(a, -np.sqrt(c * c - d * d))
    return a, d, c


def fit_svi(k, iv, T):
    """
    Two stage fit.
    Stage 1: search m and sigma with Nelder-Mead, solving the rest exactly.
    Stage 2: polish all five parameters with least squares in vol space,
             so the reported error reads directly in vol points.
    """
    w_mkt = iv ** 2 * T
    k_lo, k_hi = k.min(), k.max()

    def variance_error(p):
        m, log_sigma = p
        sigma = np.exp(log_sigma)  # searching on log keeps sigma positive
        a, d, c = _linear_part(k, w_mkt, m, sigma)
        y = (k - m) / sigma
        error = np.sum((a + d * y + c * np.sqrt(y * y + 1) - w_mkt) ** 2)
        # The bottom of the smile has to sit inside the strikes we actually saw
        outside = max(0.0, k_lo - m) + max(0.0, m - k_hi)
        return error + 1e3 * outside ** 2

    # A small grid of starting points, since the error surface has local minima
    best = None
    for m0 in (k_lo / 2, 0.0, k_hi / 2):
        for sigma0 in (0.02, 0.1, 0.3):
            res = minimize(variance_error, [m0, np.log(sigma0)], method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-14, "maxiter": 2000})
            if best is None or res.fun < best.fun:
                best = res

    m, sigma = best.x[0], np.exp(best.x[1])
    a, d, c = _linear_part(k, w_mkt, m, sigma)
    b = c / sigma
    rho = d / c if c > 0 else 0.0

    # Stage 2 works on a rebuilt set of parameters where both rules hold by
    # construction, so the optimiser cannot break them even slightly:
    #   floor  = lowest total variance on the curve, kept >= 0
    #   wing   = b * (1 + |rho|) / 2, kept between 0 and 1 (Lee's bound)
    # A penalty approach failed on one week SPY options, where variance is so
    # small that the penalty barely registered against the fit error.
    def to_svi(p):
        floor, wing, rho, m, sigma = p
        b = 2 * wing / (1 + abs(rho))
        a = floor - b * sigma * np.sqrt(1 - rho ** 2)
        return a, b, rho, m, sigma

    def vol_residuals(p):
        w = svi_w(k, *to_svi(p))
        return np.sqrt(np.clip(w, 1e-12, None) / T) - iv

    start = np.array([a + b * sigma * np.sqrt(1 - rho ** 2), b * (1 + abs(rho)) / 2, rho, m, sigma])
    lower = [0.0, 0.0, -0.999, k_lo, 1e-4]
    upper = [max(w_mkt), 1.0, 0.999, k_hi, 2.0]
    start = np.clip(start, np.array(lower) + 1e-9, np.array(upper) - 1e-9)
    polish = least_squares(vol_residuals, start, bounds=(lower, upper), max_nfev=500, x_scale="jac")
    return np.array(to_svi(polish.x)), polish


# ---------------------------------------------------------------------------
# Per ticker pipeline
# ---------------------------------------------------------------------------

def load_otm(ticker: str) -> pd.DataFrame:
    files = sorted(PROCESSED_DIR.glob(f"{ticker}_*_iv.parquet"))
    if not files:
        raise FileNotFoundError(f"no IV file for {ticker}, run src.implied_vol first")
    df = pd.read_parquet(files[-1])
    df = df[(df["iv_status"] == "ok") & df["otm"]].copy()
    df["forward"] = df["spot"] * np.exp((df["r"] - df["q"]) * df["T"])
    df["k"] = np.log(df["strike"] / df["forward"])
    return df


def fit_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    rows = []
    for expiry, one in df.groupby("expiry"):
        T = one["T"].iloc[0]
        mny = one["strike"] / one["spot"]
        row = {"expiry": expiry, "days": round(one["days_to_expiry"].iloc[0], 2), "T": T, "points": len(one),
               "quoted_from": mny.min(), "quoted_to": mny.max()}
        if len(one) < MIN_POINTS:
            rows.append({**row, "status": "too few points"})
            continue

        params, res = fit_svi(one["k"].values, one["iv"].values, T)
        a, b, rho, m, sigma = params
        fitted_iv = np.sqrt(svi_w(one["k"].values, *params) / T)
        rmse = np.sqrt(np.mean((fitted_iv - one["iv"].values) ** 2)) * 100

        # Check the butterfly condition over the strikes that were actually quoted
        k_grid = np.linspace(one["k"].min(), one["k"].max(), 200)
        g_min = svi_density_check(k_grid, *params).min()

        status = "ok"
        if a + b * sigma * np.sqrt(1 - rho ** 2) < 0:
            status = "negative variance"
        elif b * (1 + abs(rho)) > 2 + 1e-6:
            status = "wings too steep"
        elif rmse > MAX_RMSE_VOL_PTS:
            status = "poor fit"
        elif g_min < 0:
            status = "butterfly arbitrage"
        rows.append({**row, "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
                     "rmse_vol_pts": round(rmse, 3), "min_density_g": g_min, "status": status})
    return pd.DataFrame(rows)


def build_grid(params: pd.DataFrame, spot: float, r: float, q: float) -> pd.DataFrame:
    """
    Fitted vol on a strike/spot grid for every expiry that fit cleanly.

    A 3 day option only has quotes within a few percent of spot, so most of the
    70% to 130% grid is pure extrapolation for it. We still compute those values
    but flag them, so charts and checks can stick to where real quotes exist.
    """
    good = params[params["status"] == "ok"]
    out = []
    for p in good.itertuples():
        forward = spot * np.exp((r - q) * p.T)
        k = np.log(MONEYNESS_GRID * spot / forward)
        iv = np.sqrt(svi_w(k, p.a, p.b, p.rho, p.m, p.sigma) / p.T)
        outside = (MONEYNESS_GRID < p.quoted_from) | (MONEYNESS_GRID > p.quoted_to)
        out.append(pd.DataFrame({"expiry": p.expiry, "days": p.days, "T": p.T,
                                 "moneyness": MONEYNESS_GRID, "iv": iv, "extrapolated": outside}))
    return pd.concat(out, ignore_index=True)


def calendar_violations(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Total variance (iv^2 * T) must never fall as expiry gets longer at the same
    moneyness, otherwise selling the short option and buying the long one is free
    money. Returns every grid point that breaks that rule, using only points
    backed by real quotes (extrapolated wings are allowed to be wrong).
    """
    g = grid[~grid["extrapolated"]].copy()
    g["w"] = g["iv"] ** 2 * g["T"]
    g = g.sort_values(["moneyness", "T"])
    g["prev_expiry"] = g.groupby("moneyness")["expiry"].shift()
    g["prev_iv"] = g.groupby("moneyness")["iv"].shift()
    g["w_drop"] = -g.groupby("moneyness")["w"].diff()
    breaks = g[g["w_drop"] > 1e-6]
    return breaks[["moneyness", "prev_expiry", "expiry", "prev_iv", "iv", "w_drop"]]


def skew_30d(grid: pd.DataFrame) -> float:
    """
    The skew number desks quote: 30 day implied vol at a 95% strike minus at a
    105% strike, in vol points. Positive means downside puts cost more vol than
    upside calls. Model free, so it does not depend on how SVI split its parameters.
    """
    expiry = grid.loc[(grid["days"] - 30).abs().idxmin(), "expiry"]
    one = grid[grid["expiry"] == expiry]
    low = one[np.isclose(one["moneyness"], 0.95)]
    high = one[np.isclose(one["moneyness"], 1.05)]
    if low.empty or high.empty or low["extrapolated"].iloc[0] or high["extrapolated"].iloc[0]:
        return np.nan
    return (low["iv"].iloc[0] - high["iv"].iloc[0]) * 100


def sanity_check(params: pd.DataFrame, grid: pd.DataFrame, ticker: str) -> dict:
    fitted = params[params["status"] != "too few points"]
    ok = params[params["status"] == "ok"]

    # Hard checks
    assert len(ok) >= 3, f"{ticker}: only {len(ok)} expiries fit cleanly, not enough for a surface"
    assert np.isfinite(grid["iv"]).all(), f"{ticker}: surface has missing values"

    # The range check covers only points backed by real quotes. A 3 day option
    # is quoted a few percent around spot. Carried out to a 70% strike, even a
    # curve inside Lee's bound can reach several hundred percent vol, because
    # total variance spread over 3 days of time is a huge annualised number.
    # Those points are flagged extrapolated and hidden from every chart.
    quoted = grid[~grid["extrapolated"]]
    bad = quoted[~quoted["iv"].between(0.01, 3.0)]
    if len(bad):
        log.error("%s: quoted surface points outside 1%% to 300%%:\n%s", ticker,
                  bad.groupby("expiry").agg(days=("days", "first"), points=("iv", "size"),
                                            lowest=("iv", "min"), highest=("iv", "max")).to_string())
    assert bad.empty, f"{ticker}: quoted surface vol outside 1% to 300%"

    wild = grid[grid["extrapolated"] & ~grid["iv"].between(0.01, 3.0)]
    if len(wild):
        log.info("%s: %d extrapolated grid points beyond 300%% vol (hidden), on expiries of %s days",
                 ticker, len(wild), ", ".join(f"{d:.0f}" for d in sorted(wild["days"].unique())))
    assert (ok["b"] >= 0).all() and ok["rho"].between(-1, 1).all(), f"{ticker}: SVI parameters out of bounds"

    # Soft checks
    bad = fitted[fitted["status"] != "ok"]
    for row in bad.itertuples():
        log.warning("%s %s (%.0f days): %s, rmse %.2f vol pts",
                    ticker, row.expiry.date(), row.days, row.status, row.rmse_vol_pts)
    cal = calendar_violations(grid)
    breaks_file = PROCESSED_DIR / f"{ticker}_calendar_breaks.csv"
    breaks_file.unlink(missing_ok=True)  # clear out any file left by an earlier run
    if len(cal):
        cal.to_csv(breaks_file, index=False)
        pairs = cal.groupby(["prev_expiry", "expiry"]).size()
        for (short_exp, long_exp), n in pairs.items():
            log.warning("%s: %d grid points where %s carries more total variance than the later %s",
                        ticker, n, short_exp.date(), long_exp.date())

    log.info("%s: sanity checks passed", ticker)
    return {"expiries_fit": len(ok), "skipped_thin": int((params["status"] == "too few points").sum()),
            "flagged": len(bad), "median_rmse_vol_pts": round(ok["rmse_vol_pts"].median(), 2),
            "worst_rmse_vol_pts": round(ok["rmse_vol_pts"].max(), 2),
            "skew_30d_pts": round(skew_30d(grid), 2), "calendar_breaks": len(cal)}


# ---------------------------------------------------------------------------
# Charts for eyeballing the fits
# ---------------------------------------------------------------------------

def plot_smiles(df: pd.DataFrame, params: pd.DataFrame, ticker: str) -> None:
    ok = params[params["status"] == "ok"]
    # Show up to 6 expiries spread across the term structure
    pick = ok.iloc[np.unique(np.linspace(0, len(ok) - 1, min(6, len(ok))).astype(int))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    for ax, p in zip(axes.flat, pick.itertuples()):
        pts = df[df["expiry"] == p.expiry]
        k = np.linspace(pts["k"].min(), pts["k"].max(), 200)
        fwd = pts["forward"].iloc[0]
        spot = pts["spot"].iloc[0]
        ax.scatter(pts["strike"] / spot, pts["iv"] * 100, s=12, alpha=0.6, label="market (OTM)")
        ax.plot(fwd * np.exp(k) / spot, np.sqrt(svi_w(k, p.a, p.b, p.rho, p.m, p.sigma) / p.T) * 100,
                color="black", label="SVI fit")
        ax.set_title(f"{p.expiry.date()}  ({p.days:.0f}d)  rmse {p.rmse_vol_pts:.2f}")
        ax.set_xlabel("Strike / spot")
        ax.set_ylabel("Implied vol (%)")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(pick):]:
        ax.axis("off")
    axes.flat[0].legend()
    fig.suptitle(f"{ticker} volatility smiles: market vs SVI", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{ticker}_smiles.png", dpi=130)
    plt.close(fig)


def plot_surface(grid: pd.DataFrame, ticker: str) -> None:
    # Blank out extrapolated points so the chart only shows what the market quoted
    shown = grid.assign(iv=grid["iv"].where(~grid["extrapolated"]))
    pivot = shown.pivot(index="days", columns="moneyness", values="iv") * 100
    fig = go.Figure(go.Surface(x=pivot.columns, y=pivot.index, z=pivot.values, colorscale="Viridis",
                               colorbar={"title": "IV %"}))
    fig.update_layout(title=f"{ticker} implied volatility surface (SVI)",
                      scene={"xaxis_title": "Strike / spot", "yaxis_title": "Days to expiry",
                             "zaxis_title": "Implied vol (%)"},
                      height=700)
    fig.write_html(FIG_DIR / f"{ticker}_surface.html", include_plotlyjs="cdn")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for ticker in TICKERS:
        df = load_otm(ticker)
        params = fit_ticker(df, ticker)
        first = df.iloc[0]
        grid = build_grid(params, first["spot"], first["r"], first["q"])
        stats = sanity_check(params, grid, ticker)

        params.to_csv(PROCESSED_DIR / f"{ticker}_svi_params.csv", index=False)
        grid.to_parquet(PROCESSED_DIR / f"{ticker}_surface.parquet", index=False)
        plot_smiles(df, params, ticker)
        plot_surface(grid, ticker)
        summary.append({"ticker": ticker, **stats})

    pd.set_option("display.width", 200)
    print("\n=== SVI volatility surface ===")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nSmile charts and 3D surfaces saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
