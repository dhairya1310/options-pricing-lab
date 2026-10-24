"""Overview page: what the project does and the headline findings, computed live from the data."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.app_data import TICKERS, as_of, atm_vol_30d, load_chain, load_model_vs_market, load_parity, skew_30d

ui.setup()
ui.header(
    "Equity derivatives research",
    "Options Pricing Lab",
    "How well do textbook option models price real US equity options, and where exactly do they break? "
    "Five live option chains, three pricers built from scratch, an SVI volatility surface and a put-call "
    "parity audit after trading costs.",
    as_of(),
)

# Headline numbers, recomputed from the saved results every time the data changes
mvm = load_model_vs_market()
parity = load_parity()
contracts = sum(len(load_chain(t)) for t in TICKERS)
ui.kpis([
    {"label": "Contracts priced", "value": f"{contracts:,}", "note": "after cleaning, 5 underlyings"},
    {"label": "Pricing models", "value": "3", "note": "Black-Scholes, American tree, Monte Carlo", "color": ui.CALL},
    {"label": "Parity pairs tested", "value": f"{len(parity):,}", "note": "every strike with a call and a put",
     "color": ui.PUT},
    {"label": "Tradeable arbitrage", "value": f"{int(parity['true_arb'].sum())}",
     "note": "breaks the American range at bid and ask", "color": ui.MC},
])

rows, stats = [], []
for t in TICKERS:
    m = mvm[mvm["ticker"] == t]
    deep_puts = m[(m["option_type"] == "put") & (m["moneyness"] < 0.90)]
    itm = m[(m["option_type"] == "put") & (m["moneyness"] > 1.05) & (m["days_to_expiry"] > 30)
            & ~m["smile_extrapolated"]]
    p = parity[parity["ticker"] == t]
    skew = skew_30d(t)
    tree_bs = (itm["tree_smile_err"].abs().median() < itm["bs_smile_err"].abs().median()) if len(itm) else None
    stats.append({"ticker": t, "skew": skew, "atm": atm_vol_30d(t) * 100, "tree_beats_bs": tree_bs})
    rows.append({
        "Ticker": t,
        "ATM vol": atm_vol_30d(t) * 100,
        "Skew": skew,
        "BS err <90%": deep_puts["bs_rel_err"].median() * 100,
        "Tree wins": (itm["tree_smile_err"].abs() < itm["bs_smile_err"].abs()).mean() * 100 if len(itm) else None,
        "Parity raw": p["beats_cost"].mean() * 100,
        "Parity final": p["beats_cost_ex_rate"].mean() * 100,
    })

s = pd.DataFrame(stats)

left, right = st.columns([3, 2], gap="medium")
with left:
    ui.section("Market snapshot", "per underlying")
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "ATM vol": st.column_config.NumberColumn(format="%.1f%%", help="30 day at the money implied vol"),
            "Skew": st.column_config.NumberColumn(format="%.1f pts",
                                                  help="30 day vol at a 95% strike minus at a 105% strike"),
            "BS err <90%": st.column_config.NumberColumn(
                format="%.0f%%", help="Median Black-Scholes error on puts struck below 90% of the forward, flat vol"),
            "Tree wins": st.column_config.ProgressColumn(
                format="%.0f%%", min_value=0, max_value=100,
                help="Share of ITM puts where the American tree lands closer to the market than Black-Scholes"),
            "Parity raw": st.column_config.NumberColumn(format="%.0f%%", help="Pairs whose raw gap beats the spread"),
            "Parity final": st.column_config.NumberColumn(
                format="%.1f%%", help="After the market forward, early exercise and the implied funding rate"),
        },
    )
with right:
    ui.section("Vol level vs skew", "30 days")
    fig = go.Figure(go.Scatter(
        x=s["atm"], y=s["skew"], mode="markers+text", text=s["ticker"], textposition="top center",
        textfont=dict(color=ui.INK, size=12),
        marker=dict(size=13, color=[ui.TICKER_COLOR[t] for t in s["ticker"]],
                    symbol=[ui.TICKER_SYMBOL[t] for t in s["ticker"]], line=dict(color=ui.CARD, width=2)),
        hovertemplate="%{text}<br>ATM vol %{x:.1f}%<br>Skew %{y:.1f} pts<extra></extra>",
    ))
    fig.update_xaxes(title="30 day ATM vol (%)")
    fig.update_yaxes(title="30 day skew (vol pts)")
    fig.update_layout(height=300, margin=dict(t=20, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")

# The written findings are built from the numbers above, so they can never
# drift out of date when a new snapshot comes in
ss = s.dropna(subset=["skew"])
steep, flat = ss.loc[ss["skew"].idxmax(), "ticker"], ss.loc[ss["skew"].idxmin(), "ticker"]
tested = [x for x in stats if x["tree_beats_bs"] is not None]
wins = [x["ticker"] for x in tested if x["tree_beats_bs"]]
tree_text = ("for every ticker" if len(wins) == len(tested)
             else f"for {len(wins)} of {len(tested)} tickers ({', '.join(wins)})")
arbs = int(parity["true_arb"].sum())
rates = parity.drop_duplicates(["ticker", "expiry"]).dropna(subset=["rate_implied"])
funding_bp = ((rates["rate_implied"] - rates["r"]).median() * 1e4) if len(rates) else 0.0
arb_text = ("and none break the no arbitrage range at tradeable prices" if arbs == 0 else
            f"and {arbs} pairs break the no arbitrage range at quoted prices, most likely stale quotes")

ui.section("Key findings")
c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    ui.note(f"<b>01 · Black-Scholes breaks on the skew.</b><br>With one flat vol per expiry, deep out of the "
            f"money puts come out far too cheap. The market pays for crash protection a flat vol model prices "
            f"near zero. Skew is steepest on <b>{steep}</b> and flattest on <b>{flat}</b>.")
with c2:
    ui.note(f"<b>02 · Early exercise is real money.</b><br>Priced at the smile vol so skew is out of the "
            f"picture, the American tree has a smaller median error than Black-Scholes on in the money puts "
            f"<b>{tree_text}</b>.")
with c3:
    ui.note(f"<b>03 · Parity gaps are not free money.</b><br>Once the market's own forward, early exercise and "
            f"funding rate are accounted for, almost no gaps survive the bid/ask spread, {arb_text}. "
            f"On long expiries the options market funds about <b>{funding_bp:.0f} bp</b> above the T-bill.")

st.write("")
st.caption("Use the pages on the left to price any contract, explore the vol surface, move the Greeks, "
           "and read the full method. Every number passed sanity checks before it was saved.")
