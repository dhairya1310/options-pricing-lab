"""
Step 10: check put-call parity on every strike that has both a call and a put.

Run from the project root:
    python -m src.parity

For European options parity is an exact rule:
    C - P = S * e^(-qT) - K * e^(-rT)
If the market breaks it, in theory you buy the cheap side, sell the rich side,
hedge with the stock, and lock in the gap. In practice almost every gap is
eaten by costs. This step shows that in three levels:

    1. raw gap      call minus put vs textbook parity, mid prices,
                    our own inputs (spot, rate, trailing dividend)
    2. after costs  is the gap bigger than crossing both bid/ask spreads?
    3. true arb     US options are American, so parity loosens into a range:
                        S * e^(-qT) - K  <=  C - P  <=  S - K * e^(-rT)
                    a real arbitrage has to break that range at prices you
                    can actually trade (buy at the ask, sell at the bid)

We also measure how much of each raw gap is just the forward. Step 6 found
that our own forward was off (SPY options trade until 4:15 PM, trailing
dividends are not expected dividends). Re-measuring each gap against the
market's own forward shows what is left once that is removed.

Output:
    data/processed/parity_pairs.parquet
    reports/parity_flagged.csv           pairs whose gap beats the spread
    reports/figures/parity_gaps.png
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pricing import binomial_price

TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "JPM"]
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"

ARB_MIN_PROFIT = 0.01  # a true arbitrage has to clear at least a cent per share
EEP_STEPS = 200        # tree steps for early exercise estimates
SHORT_DAYS = 30        # expiries shorter than this set the implied spot

log = logging.getLogger("parity")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def build_pairs(ticker: str) -> pd.DataFrame:
    """One row per strike and expiry that has both a call and a put after cleaning."""
    files = sorted(PROCESSED_DIR.glob(f"{ticker}_*_iv.parquet"))
    if not files:
        raise FileNotFoundError(f"no IV file for {ticker}, run src.implied_vol first")
    df = pd.read_parquet(files[-1])

    keys = ["expiry", "strike"]
    shared = ["ticker", "spot", "T", "days_to_expiry", "r", "q", "q_trailing", "forward"]
    quote = ["bid", "ask", "mid", "iv"]
    calls = df[df["option_type"] == "call"][keys + shared + quote]
    puts = df[df["option_type"] == "put"][keys + quote]
    pairs = calls.merge(puts, on=keys, suffixes=("_c", "_p"))
    pairs["moneyness"] = pairs["strike"] / pairs["forward"]
    return pairs.reset_index(drop=True)


def early_exercise_values(p: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """American minus European tree value for each leg, on the same grid so rounding cancels."""
    eep_c, eep_p = np.zeros(len(p)), np.zeros(len(p))
    for i, row in enumerate(p.itertuples()):
        # One vol per strike, from the out of the money leg. The in the money
        # leg's own BS vol is inflated by the very early exercise value we are
        # trying to remove, which made deep ITM long dated puts look like
        # parity breaks in testing. Fall back to the other leg if needed.
        clean, other = (row.iv_c, row.iv_p) if row.strike >= row.forward else (row.iv_p, row.iv_c)
        vol = clean if np.isfinite(clean) else other
        if not np.isfinite(vol):
            continue
        for kind, out in (("call", eep_c), ("put", eep_p)):
            args = (row.spot, row.strike, row.T, row.r, row.q, vol, kind)
            out[i] = (binomial_price(*args, steps=EEP_STEPS, american=True)
                      - binomial_price(*args, steps=EEP_STEPS, american=False))
    return eep_c, eep_p


def implied_spot(p: pd.DataFrame) -> float:
    """One spot for the whole chain, backed out of expiries under 30 days."""
    short = p[p["days_to_expiry"] < SHORT_DAYS].drop_duplicates("expiry")
    if short.empty:
        short = p.nsmallest(1, "days_to_expiry")
    spots = short["forward"] * np.exp(-(short["r"] - short["q_trailing"]) * short["T"])
    return float(spots.median())


def measure(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["spot_implied"] = implied_spot(p)
    S, K, T, r = p["spot"], p["strike"], p["T"], p["r"]
    q = p["q_trailing"]
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    market = p["mid_c"] - p["mid_p"]

    # Level 1: raw gap against textbook parity with our own inputs
    p["parity_own"] = S * disc_q - K * disc_r
    p["raw_gap"] = market - p["parity_own"]

    # How much of that is just the forward: same gap against the market forward
    p["parity_market_fwd"] = disc_r * (p["forward"] - K)
    p["residual_gap"] = market - p["parity_market_fwd"]

    # How much of what is left is early exercise. Parity is a European rule,
    # so we strip each leg's early exercise value (tree, American minus
    # European, at that leg's own implied vol) and measure again.
    p["eep_c"], p["eep_p"] = early_exercise_values(p)
    p["residual_ex_early"] = (p["mid_c"] - p["eep_c"]) - (p["mid_p"] - p["eep_p"]) - p["parity_market_fwd"]

    # Level 2: what it costs to trade both legs. Each leg crosses half its
    # spread from the mid, so the pair costs half the call spread plus half
    # the put spread. Stock spread is ignored, it is a cent or two on these names.
    p["cost"] = (p["ask_c"] - p["bid_c"]) / 2 + (p["ask_p"] - p["bid_p"]) / 2
    p["gap_over_cost"] = p["raw_gap"].abs() / p["cost"]
    p["beats_cost"] = p["raw_gap"].abs() > p["cost"]

    # Level 3: does it still beat the spread once we use the market's forward?
    p["beats_cost_mkt_fwd"] = p["residual_gap"].abs() > p["cost"]
    # Level 4: and once early exercise is taken out as well?
    p["beats_cost_ex_early"] = p["residual_ex_early"].abs() > p["cost"]

    # Level 5: true arbitrage against the American range, at tradeable prices.
    #   too cheap: buy the call at the ask, sell the put at the bid, short the stock
    #   too rich:  sell the call at the bid, buy the put at the ask, buy the stock
    #
    # An arbitrage test has to be model free. The market forward depends on our
    # early exercise model, so the bounds use only observables:
    #
    # Spot: our close can be a few cents stale (SPY options close at 4:15 PM),
    # and on a range check a stale spot looks exactly like free money. So we
    # back one spot out of the short expiries, where carry is too small to matter.
    #
    # Carry: the textbook range  S e^(-qT) - K  <=  C - P  <=  S - K e^(-rT)
    # assumes we know the dividend yield q and that everyone funds at r. Neither
    # holds exactly. TSLA pays no dividend yet its options imply a carry of
    # about -1.4%, which means the options market funds at a rate above our
    # 3 month T-bill (rates implied by box spreads do sit above T-bills).
    # With the T-bill rate, the upper bound came out too tight and around 30
    # ordinary TSLA quotes looked like free money.
    #
    # So we give the quotes every benefit of the doubt. q could be anywhere
    # between the lowest and highest of trailing yield, implied carry and zero:
    #   lower bound uses the highest q (smallest S e^(-qT))
    #   upper bound uses the lowest q, S e^(-q_low T) - K e^(-rT), which for a
    #   negative q is the same as allowing a higher funding rate
    # Anything still flagged is a genuine quote problem, not an assumption.
    S_obs = p["spot_implied"]
    q_all = np.vstack([p["q_trailing"].values, p["q"].values, np.zeros(len(p))])
    p["q_high"], p["q_low"] = q_all.max(axis=0), q_all.min(axis=0)
    lower = S_obs * np.exp(-p["q_high"] * T) - K
    upper = S_obs * np.exp(-p["q_low"] * T) - K * disc_r
    p["arb_profit_low"] = lower - (p["ask_c"] - p["bid_p"])
    p["arb_profit_high"] = (p["bid_c"] - p["ask_p"]) - upper
    p["arb_profit"] = p[["arb_profit_low", "arb_profit_high"]].max(axis=1)
    p["true_arb"] = p["arb_profit"] > ARB_MIN_PROFIT

    # Above the forward, American puts carry most of the early exercise value.
    # We report the two sides separately so the effect stays visible.
    p["side"] = np.where(p["strike"] < p["forward"], "below forward", "above forward")
    return p


def sanity_check(p: pd.DataFrame, ticker: str) -> dict:
    # Hard checks
    assert len(p) > 0, f"{ticker}: no strikes with both a call and a put"
    assert (p["cost"] >= 0).all(), f"{ticker}: negative spread cost, a bid sits above an ask"
    cols = ["raw_gap", "residual_gap", "residual_ex_early", "arb_profit"]
    assert np.isfinite(p[cols].values).all(), f"{ticker}: missing values"
    assert (p["eep_c"] >= -1e-9).all() and (p["eep_p"] >= -1e-9).all(), \
        f"{ticker}: negative early exercise value"

    # The American range has to contain European parity at the same spot and
    # yield, otherwise the bounds are wrong
    # The range must contain European parity for every q we allow
    K, T, r, S = p["strike"], p["T"], p["r"], p["spot_implied"]
    for q in (p["q_low"], p["q_high"]):
        euro = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert ((S * np.exp(-p["q_high"] * T) - K) <= euro + 1e-9).all(), f"{ticker}: lower bound above parity"
        assert ((S * np.exp(-p["q_low"] * T) - K * np.exp(-r * T)) >= euro - 1e-9).all(), \
            f"{ticker}: upper bound below parity"

    # The implied spot should sit within a whisker of our own
    spot_shift = abs(p["spot_implied"].iloc[0] / p["spot"].iloc[0] - 1) * 100
    if spot_shift > 0.5:
        log.warning("%s: implied spot is %.2f%% from our spot", ticker, spot_shift)

    # Near the money the market forward came from these very quotes (Step 6,
    # with early exercise stripped), so here the early exercise free residual
    # should sit well inside the spread. If not, the forward and the parity
    # maths disagree somewhere. (The raw residual is not tested here: on long
    # expiries it carries several dollars of put early exercise value, more
    # than the spread, and that is the market being right, not a bug.)
    near = p[(p["moneyness"] - 1).abs() < 0.05]
    if len(near):
        resid_ratio = (near["residual_ex_early"].abs() / near["cost"]).median()
        if resid_ratio >= 1:
            worst = near.assign(ratio=near["residual_ex_early"].abs() / near["cost"]).nlargest(8, "ratio")
            log.error("%s: near the money pairs with the largest early exercise free residual:\n%s", ticker,
                      worst[["expiry", "days_to_expiry", "strike", "mid_c", "mid_p", "eep_c", "eep_p",
                             "residual_gap", "residual_ex_early", "cost", "ratio"]].round(3).to_string(index=False))
        assert resid_ratio < 1, f"{ticker}: near the money, residual is {resid_ratio:.2f}x the spread"
    else:
        resid_ratio = np.nan

    arbs = p[p["true_arb"]]
    if len(arbs):
        log.warning("%s: flagged pairs:\n%s", ticker,
                    arbs[["expiry", "days_to_expiry", "strike", "bid_c", "ask_c", "bid_p", "ask_p",
                          "spot_implied", "arb_profit"]].round(3).to_string(index=False))
        log.warning("%s: %d pairs look like true arbitrage at tradeable prices. Almost always a stale "
                    "quote or our spot not matching the option close, listed in parity_flagged.csv",
                    ticker, len(arbs))

    below = p[p["side"] == "below forward"]
    above = p[p["side"] == "above forward"]

    log.info("%s: sanity checks passed", ticker)
    return {
        "pairs": len(p),
        "median_cost_usd": round(p["cost"].median(), 3),
        "raw_gap": round(p["raw_gap"].abs().median(), 3),
        "mkt_fwd_gap": round(p["residual_gap"].abs().median(), 3),
        "ex_early_gap": round(p["residual_ex_early"].abs().median(), 3),
        "above_fwd_eep_put": round(above["eep_p"].median(), 3) if len(above) else np.nan,
        "beats_cost_pct": round(p["beats_cost"].mean() * 100, 1),
        "ex_early_beats_pct": round(p["beats_cost_ex_early"].mean() * 100, 1),
        "true_arb": len(arbs),
        "spot_shift_pct": round(spot_shift, 3),
        "near_atm_resid_vs_cost": round(resid_ratio, 2),
    }


def plot_gaps(all_pairs: pd.DataFrame) -> None:
    """
    Gap divided by cost per ticker, before (grey) and after (colour) taking out
    the forward and early exercise. Above the dashed line the gap beats the spread.
    """
    fig, axes = plt.subplots(1, len(TICKERS), figsize=(20, 4.8), sharey=True, constrained_layout=True)
    floor = 1e-4  # log scale cannot show zero
    for ax, ticker in zip(axes, TICKERS):
        p = all_pairs[all_pairs["ticker"] == ticker]
        final = (p["residual_ex_early"].abs() / p["cost"]).clip(lower=floor)
        beats = p["beats_cost_ex_early"]
        ax.scatter(p["moneyness"], p["gap_over_cost"].clip(lower=floor), s=5, color="lightgrey",
                   label="raw gap, our own inputs")
        ax.scatter(p.loc[~beats, "moneyness"], final[~beats], s=6, alpha=0.5, label="final gap, inside the spread")
        ax.scatter(p.loc[beats, "moneyness"], final[beats], s=10, color="crimson", label="final gap, beats the spread")
        if p["true_arb"].any():
            a = p["true_arb"]
            ax.scatter(p.loc[a, "moneyness"], final[a], s=50, facecolors="none", edgecolors="black",
                       label="breaks American range")
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel("Strike / forward")
        ax.set_title(f"{ticker}: {p['beats_cost'].mean():.0%} raw, {beats.mean():.1%} final beat the spread",
                     fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("|parity gap| / spread cost (log)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Put-call parity: most gaps vanish once the forward and early exercise are accounted for",
                 fontsize=13)
    fig.savefig(FIG_DIR / "parity_gaps.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frames, summary = [], []
    for ticker in TICKERS:
        p = measure(build_pairs(ticker))
        summary.append({"ticker": ticker, **sanity_check(p, ticker)})
        frames.append(p)

    all_pairs = pd.concat(frames, ignore_index=True)
    all_pairs.to_parquet(PROCESSED_DIR / "parity_pairs.parquet", index=False)

    flagged = all_pairs[all_pairs["beats_cost"]].sort_values("gap_over_cost", ascending=False)
    cols = ["ticker", "expiry", "strike", "side", "moneyness", "days_to_expiry", "mid_c", "mid_p",
            "raw_gap", "residual_gap", "eep_c", "eep_p", "residual_ex_early", "cost", "gap_over_cost",
            "beats_cost_mkt_fwd", "beats_cost_ex_early", "arb_profit", "true_arb"]
    numeric = flagged[cols].select_dtypes("number").columns
    flagged = flagged[cols].copy()
    flagged[numeric] = flagged[numeric].round(4)
    flagged.to_csv(REPORT_DIR / "parity_flagged.csv", index=False)
    plot_gaps(all_pairs)

    a = all_pairs
    total = len(a)
    beat = int(a["beats_cost"].sum())
    beat_mkt = int((a["beats_cost"] & a["beats_cost_mkt_fwd"]).sum())
    beat_ex = int((a["beats_cost"] & a["beats_cost_mkt_fwd"] & a["beats_cost_ex_early"]).sum())
    arb = int(a["true_arb"].sum())
    pd.set_option("display.width", 220)
    print("\n=== Put-call parity ===")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nFunnel: {total} pairs"
          f"\n  -> {beat} beat the spread ({beat / total:.1%})"
          f"\n  -> {beat_mkt} still do with the market forward ({beat_mkt / total:.1%})"
          f"\n  -> {beat_ex} still do once early exercise is removed ({beat_ex / total:.1%})"
          f"\n  -> {arb} break the American range at tradeable prices")
    print("\nraw_gap             median |C - P - parity|, our own spot and trailing dividend")
    print("mkt_fwd_gap         same against the market forward from Step 6")
    print("ex_early_gap        same again after removing each leg's early exercise value")
    print("above_fwd_eep_put   median put early exercise value for strikes above the forward, in dollars")
    print("ex_early_beats_pct  share of pairs whose early exercise free gap still beats the spread")
    print("spot_shift_pct      our spot vs the spot implied by short expiries")
    print(f"\nFlagged pairs saved to {REPORT_DIR}/parity_flagged.csv")


if __name__ == "__main__":
    main()
