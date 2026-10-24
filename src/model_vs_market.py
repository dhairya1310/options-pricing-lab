"""
Step 9: how well does each model price the real market, and where does it fail?

Run from the project root:
    python -m src.model_vs_market

The test: give every model one volatility per expiry (the at the money vol
from the SVI fit) and price every contract on that expiry. That is the
textbook Black-Scholes world, one vol for all strikes. The market disagrees,
and the errors show exactly how:

    deep OTM puts   the market pays far more than flat vol says (the skew)
    ITM puts        Black-Scholes and Monte Carlo miss early exercise,
                    the American tree should do better
    at the money    all three should be close, by construction

Error is model minus market mid, so negative means the model is too cheap.

Output:
    data/processed/model_vs_market.parquet       every contract, every model
    reports/model_error_summary.csv              median errors per bucket
    reports/figures/model_error_heatmap.png      the picture for the app
"""

import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pricing import binomial_price, bs_price, mc_price
from src.surface import svi_w

TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "JPM"]
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"

MODELS = ["bs", "tree", "mc"]
MODEL_NAMES = {"bs": "Black-Scholes", "tree": "Binomial (American)", "mc": "Monte Carlo"}
MIN_MID = 0.10  # below a dime, percentage errors mean nothing

# Moneyness is strike / forward, so 0.80 is a strike 20% below where the market expects the stock
MONEYNESS_BINS = [0, 0.85, 0.95, 1.05, 1.15, np.inf]
MONEYNESS_LABELS = ["< 0.85", "0.85-0.95", "0.95-1.05", "1.05-1.15", "> 1.15"]
EXPIRY_BINS = [0, 7, 30, 90, 365, np.inf]
EXPIRY_LABELS = ["<= 1w", "1w-1m", "1-3m", "3-12m", "> 1y"]

log = logging.getLogger("model_vs_market")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load(ticker: str) -> pd.DataFrame:
    """Every clean contract, with the flat at the money vol of its expiry attached."""
    files = sorted(PROCESSED_DIR.glob(f"{ticker}_*_iv.parquet"))
    if not files:
        raise FileNotFoundError(f"no IV file for {ticker}, run src.implied_vol first")
    df = pd.read_parquet(files[-1])

    params = pd.read_csv(PROCESSED_DIR / f"{ticker}_svi_params.csv", parse_dates=["expiry"])
    params = params[params["status"] == "ok"].copy()
    # SVI at k = 0 is the vol at the forward, the cleanest at the money number
    params["atm_vol"] = np.sqrt(svi_w(0.0, params["a"], params["b"], params["rho"],
                                      params["m"], params["sigma"]) / params["T"])

    before = len(df)
    keep = ["expiry", "atm_vol", "a", "b", "rho", "m", "sigma", "quoted_from", "quoted_to"]
    df = df.merge(params[keep], on="expiry", how="inner")
    if len(df) < before:
        log.info("%s: %d contracts skipped, their expiry had no clean SVI fit", ticker, before - len(df))
    df = df[df["mid"] >= MIN_MID].reset_index(drop=True)
    df["moneyness"] = df["strike"] / df["forward"]

    # The smile vol: SVI at each contract's own strike. Pricing with this takes
    # the skew out of the comparison, so what is left on ITM puts is early exercise.
    k = np.log(df["strike"] / df["forward"])
    df["smile_vol"] = np.sqrt(svi_w(k, df["a"], df["b"], df["rho"], df["m"], df["sigma"]) / df["T"])
    mny_spot = df["strike"] / df["spot"]
    df["smile_extrapolated"] = (mny_spot < df["quoted_from"]) | (mny_spot > df["quoted_to"])
    return df.drop(columns=["a", "b", "rho", "m", "sigma", "quoted_from", "quoted_to"])


def price_all(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    start = time.time()

    # Black-Scholes in one vectorised pass per option type, at both vols
    for kind in ("call", "put"):
        rows = df["option_type"] == kind
        base = (df.loc[rows, "spot"].values, df.loc[rows, "strike"].values,
                df.loc[rows, "T"].values, df.loc[rows, "r"].values, df.loc[rows, "q"].values)
        df.loc[rows, "bs"] = bs_price(*base, df.loc[rows, "atm_vol"].values, kind)
        df.loc[rows, "bs_smile"] = bs_price(*base, df.loc[rows, "smile_vol"].values, kind)

    tree, tree_eu, tree_smile, mc, mc_err = [], [], [], [], []
    for row in df.itertuples():
        args = (row.spot, row.strike, row.T, row.r, row.q, row.atm_vol, row.option_type)
        tree.append(binomial_price(*args, steps=500, american=True))
        # Same tree without early exercise. Only used for checks: on an identical
        # grid the rounding cancels, so American minus this is pure early exercise value.
        tree_eu.append(binomial_price(*args, steps=500, american=False))
        tree_smile.append(binomial_price(row.spot, row.strike, row.T, row.r, row.q, row.smile_vol,
                                         row.option_type, steps=500, american=True))
        sim = mc_price(*args, paths=100_000)
        mc.append(sim["price"])
        mc_err.append(sim["std_err"])
    df["tree"], df["tree_european"], df["mc"], df["mc_std_err"] = tree, tree_eu, mc, mc_err
    df["tree_smile"] = tree_smile
    log.info("%s: priced %d contracts with 3 models in %.0f seconds", ticker, len(df), time.time() - start)

    # Smile vol errors, Black-Scholes vs the American tree. Monte Carlo is left
    # out here: it prices the same European option as Black-Scholes.
    for m in ("bs_smile", "tree_smile"):
        df[f"{m}_err"] = df[m] - df["mid"]

    for m in MODELS:
        df[f"{m}_err"] = df[m] - df["mid"]                 # dollars
        df[f"{m}_rel_err"] = df[f"{m}_err"] / df["mid"]    # share of the market price
        # Inside the spread means the model price is one you could actually trade at
        df[f"{m}_in_spread"] = df[m].between(df["bid"], df["ask"])

    df["moneyness_bucket"] = pd.cut(df["moneyness"], MONEYNESS_BINS, labels=MONEYNESS_LABELS)
    df["expiry_bucket"] = pd.cut(df["days_to_expiry"], EXPIRY_BINS, labels=EXPIRY_LABELS)
    return df


def sanity_check(df: pd.DataFrame, ticker: str) -> dict:
    # Hard check 1: Monte Carlo and Black-Scholes price the same European
    # option, so they must agree within the simulation's own noise.
    #
    # Half a cent of slack on top of 3 standard errors. Under a flat vol, deep
    # OTM contracts get a theoretical price near $0.000001. Monte Carlo then
    # sees no path finish in the money, so both its price and its standard
    # error come out as 0, and dividing by that zero turns a millionth of a
    # dollar into "infinitely many standard errors". Quotes move in cents, so
    # a gap under half a cent is not a real disagreement.
    gap = (df["mc"] - df["bs"]).abs()
    within_noise = gap < 3 * df["mc_std_err"]
    within_slack = gap < 3 * df["mc_std_err"] + 0.005
    share_ok = within_slack.mean()

    needed_slack = within_slack & ~within_noise
    if needed_slack.any():
        rescued = df[needed_slack]
        log.info("%s: %d contracts only pass thanks to the half cent floor. Median BS price $%.1e, "
                 "median MC standard error $%.1e, largest gap $%.1e",
                 ticker, len(rescued), rescued["bs"].median(), rescued["mc_std_err"].median(),
                 gap[needed_slack].max())

    if share_ok <= 0.98:
        worst = df.assign(gap=gap).nlargest(5, "gap")
        log.error("%s: largest MC vs BS gaps:\n%s", ticker,
                  worst[["contractSymbol", "moneyness", "days_to_expiry", "bs", "mc", "mc_std_err", "gap"]]
                  .to_string(index=False))
    assert share_ok > 0.98, f"{ticker}: only {share_ok:.1%} of MC prices agree with BS"
    share_within_3se = share_ok

    # Hard check 2: early exercise can only add value, never remove it.
    # We compare the American tree with the European tree on the same grid,
    # not with Black-Scholes. An earlier version compared with BS and failed
    # on 48 long dated SPY calls, all from tree rounding: the European tree
    # sat exactly as far below BS, and early exercise value was exactly 0.
    early_exercise = df["tree"] - df["tree_european"]
    assert (early_exercise >= -1e-9).all(), \
        f"{ticker}: {(early_exercise < -1e-9).sum()} contracts show negative early exercise value"

    # Hard check 3: the tree's rounding is small for its size. Rounding scales
    # with how wide the tree spreads (S * vol * sqrt(T)), not with the option's
    # own price, so a cheap OTM call on a $760 stock can still be a few cents off.
    tree_width = df["spot"] * df["atm_vol"] * np.sqrt(df["T"])
    rounding = (df["tree_european"] - df["bs"]).abs() / tree_width
    assert rounding.max() < 1e-3, f"{ticker}: tree rounding reached {rounding.max():.2e} of the tree width"

    # Hard check 4: truly at the money, the at the money vol should reprice the
    # market almost exactly. This tests the plumbing (forward, vol, units).
    # An earlier version used the whole 0.95 to 1.05 bucket and failed on SPY
    # at 7.1%: its skew is so steep that a 96% strike already sits well up the
    # smile. That is the skew itself, the finding of this step, not a bug.
    # So the plumbing test uses strikes within 1% of the forward.
    atm = df[(df["moneyness"].between(0.99, 1.01)) & (df["days_to_expiry"] > 7)]
    atm_err = atm["bs_rel_err"].abs().median()
    assert atm_err < 0.05, f"{ticker}: median at the money BS error is {atm_err:.1%}, should be small"
    # For the record, the wider bucket, where skew already shows
    near = df[(df["moneyness_bucket"] == "0.95-1.05") & (df["days_to_expiry"] > 7)]
    near_err = near["bs_rel_err"].abs().median()

    # The findings, logged rather than asserted, because this is what the market says
    otm_puts = df[(df["option_type"] == "put") & (df["moneyness"] < 0.90)]
    otm_put_err = otm_puts["bs_rel_err"].median() if len(otm_puts) else np.nan
    # Early exercise, tested at the smile vol so skew is out of the way.
    # (With flat vol the tree looked worse than BS on ITM puts: flat ATM vol is
    # too high above the forward, both models overprice, and early exercise
    # value then pushes the tree further off. That was skew, not the tree.)
    itm_puts = df[(df["option_type"] == "put") & (df["moneyness"] > 1.05)
                  & (df["days_to_expiry"] > 30) & ~df["smile_extrapolated"]]
    if len(itm_puts):
        bs_itm = itm_puts["bs_smile_err"].abs().median()
        tree_itm = itm_puts["tree_smile_err"].abs().median()
        tree_wins = (itm_puts["tree_smile_err"].abs() < itm_puts["bs_smile_err"].abs()).mean()
        if tree_itm >= bs_itm:
            log.warning("%s: at the smile vol the tree does not beat BS on ITM puts "
                        "(%.3f vs %.3f), worth a look", ticker, tree_itm, bs_itm)
    else:
        bs_itm = tree_itm = tree_wins = np.nan

    log.info("%s: sanity checks passed, %.1f%% of MC prices agree with BS (3 SE or half a cent)",
             ticker, share_within_3se * 100)
    return {
        "contracts": len(df),
        "atm_bs_err_pct": round(atm_err * 100, 2),
        "near_atm_bs_err_pct": round(near_err * 100, 2),
        "max_tree_rounding": f"{rounding.max():.1e}",
        "max_early_ex_usd": round(early_exercise.max(), 2),
        "otm_put_bs_err_pct": round(otm_put_err * 100, 1),
        "itm_puts": len(itm_puts),
        "itm_put_bs_err_usd": round(bs_itm, 3),
        "itm_put_tree_err_usd": round(tree_itm, 3),
        "tree_wins_pct": round(tree_wins * 100, 1),
        "bs_in_spread_pct": round(df["bs_in_spread"].mean() * 100, 1),
        "tree_in_spread_pct": round(df["tree_in_spread"].mean() * 100, 1),
    }


def summarise(all_rows: pd.DataFrame) -> pd.DataFrame:
    """Median relative error per model, option type, moneyness and expiry bucket."""
    frames = []
    for m in MODELS:
        g = (all_rows.groupby(["ticker", "option_type", "moneyness_bucket", "expiry_bucket"], observed=True)
             .agg(contracts=(f"{m}_rel_err", "size"),
                  median_err_pct=(f"{m}_rel_err", lambda s: s.median() * 100),
                  median_err_usd=(f"{m}_err", "median"),
                  in_spread_pct=(f"{m}_in_spread", lambda s: s.mean() * 100))
             .reset_index())
        g.insert(0, "model", m)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def plot_heatmap(all_rows: pd.DataFrame) -> None:
    """
    One panel per model and option type, all tickers pooled. Colour is the
    median percentage error: blue means the model is too cheap, red too rich.
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for col, m in enumerate(MODELS):
        for row, kind in enumerate(("put", "call")):
            ax = axes[row, col]
            sub = all_rows[all_rows["option_type"] == kind]
            table = (sub.groupby(["expiry_bucket", "moneyness_bucket"], observed=False)[f"{m}_rel_err"]
                     .median().unstack() * 100)
            image = ax.imshow(table.values, cmap="RdBu_r", vmin=-50, vmax=50, aspect="auto")
            for i in range(table.shape[0]):
                for j in range(table.shape[1]):
                    v = table.values[i, j]
                    if np.isfinite(v):
                        label = f"{v:.0f}%".replace("-0%", "0%")  # tiny negatives print as -0 otherwise
                        ax.text(j, i, label, ha="center", va="center", fontsize=9,
                                color="white" if abs(v) > 30 else "black")
            ax.set_xticks(range(len(MONEYNESS_LABELS)), MONEYNESS_LABELS, fontsize=8)
            ax.set_yticks(range(len(EXPIRY_LABELS)), EXPIRY_LABELS, fontsize=8)
            ax.set_xlabel("Strike / forward")
            ax.set_ylabel("Time to expiry")
            ax.set_title(f"{MODEL_NAMES[m]}: {kind}s")
    fig.colorbar(image, ax=axes, shrink=0.6, label="Median error, % of market price (model minus market)")
    fig.suptitle("Pricing error with one flat vol per expiry, all tickers pooled\n"
                 "blue = model too cheap, red = model too rich", fontsize=14)
    fig.savefig(FIG_DIR / "model_error_heatmap.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frames, summary = [], []
    for ticker in TICKERS:
        df = price_all(load(ticker), ticker)
        summary.append({"ticker": ticker, **sanity_check(df, ticker)})
        frames.append(df)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_parquet(PROCESSED_DIR / "model_vs_market.parquet", index=False)
    summarise(all_rows).to_csv(REPORT_DIR / "model_error_summary.csv", index=False)
    plot_heatmap(all_rows)

    pd.set_option("display.width", 220)
    print("\n=== Model vs market (one flat vol per expiry) ===")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\natm_bs_err_pct      median |error| within 1% of the forward, should be small")
    print("near_atm_bs_err_pct same for 95% to 105% of forward, where steep skew already shows")
    print("otm_put_bs_err_pct  median error on puts struck below 90% of forward, negative = BS too cheap")
    print("itm_put_*_err_usd   median |error| on ITM puts over 30 days, priced at the smile vol")
    print("                    so skew is removed and only early exercise is left, the tree should win")
    print("tree_wins_pct       share of those ITM puts where the tree lands closer to the market than BS")
    print(f"\nHeatmap saved to {FIG_DIR}/model_error_heatmap.png")


if __name__ == "__main__":
    main()
