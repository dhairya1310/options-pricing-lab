"""Parity Check page: the funnel, the implied funding rate, and every pair whose gap survives it."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.app_data import TICKERS, load_parity

ui.setup()
ui.header("Arbitrage audit", "Put-call parity check",
          "For every strike with both a call and a put: is call minus put where parity says it should be, "
          "and if not, is the gap big enough to trade after costs?")

df = load_parity()
ticker = st.selectbox("Ticker", ["All tickers"] + TICKERS)
p = df if ticker == "All tickers" else df[df["ticker"] == ticker]

total = len(p)
stage1 = p["beats_cost"]
stage2 = stage1 & p["beats_cost_mkt_fwd"]
stage3 = stage2 & p["beats_cost_ex_early"]
stage4 = stage3 & p["beats_cost_ex_rate"]
arb = p["true_arb"]

ui.section("The funnel", "each stage removes one explanation that is not free money")
ui.kpis([
    {"label": "Call and put pairs", "value": f"{total:,}", "note": "every strike with both legs", "color": ui.MUTED},
    {"label": "Raw gap beats spread", "value": f"{stage1.sum():,}", "note": f"{stage1.mean():.1%} · our own inputs",
     "color": ui.PUT},
    {"label": "With market forward", "value": f"{stage2.sum():,}", "note": f"{stage2.mean():.1%} · forward from the options",
     "color": ui.PUT},
    {"label": "After early exercise", "value": f"{stage3.sum():,}", "note": f"{stage3.mean():.1%} · European parity",
     "color": ui.CALL},
    {"label": "Market funding rate", "value": f"{stage4.sum():,}", "note": f"{stage4.mean():.1%} · rate implied by parity",
     "color": ui.CALL},
    {"label": "True arbitrage", "value": f"{arb.sum():,}", "note": "breaks the range at bid and ask",
     "color": ui.MC},
])

ui.note(
    """
<b>Why most gaps disappear.</b> The raw gap mixes four things that are not free money. Our own forward
is slightly off (SPY options trade until 4:15 PM, and trailing dividends are not expected dividends).
US options are American, so puts carry early exercise value that textbook parity ignores. On long
expiries the options market funds at a rate above the T-bill, which tilts call minus put in a straight
line across strikes. And every trade has to cross the bid/ask spread on two legs.
"""
)

final = (p["residual_ex_rate"].abs() / p["cost"]).clip(lower=1e-4)
raw = p["gap_over_cost"].clip(lower=1e-4)
fig = go.Figure()
fig.add_scatter(x=p["moneyness"], y=raw, mode="markers", name="raw gap", marker=dict(color="rgba(170,178,197,0.25)", size=5),
                hoverinfo="skip")
fig.add_scatter(x=p["moneyness"], y=final, mode="markers", name="final gap",
                marker=dict(color=np.where(stage4, ui.PUT, ui.CALL), size=6, opacity=0.7),
                customdata=np.c_[p["ticker"], p["strike"], p["days_to_expiry"].round(0)],
                hovertemplate="%{customdata[0]} strike %{customdata[1]}, %{customdata[2]} days"
                              "<br>gap / cost %{y:.2f}<extra></extra>")
fig.add_hline(y=1, line_dash="dash", line_color=ui.INK_2, annotation_text="gap equals the spread cost")
fig.update_yaxes(type="log", title="|parity gap| / spread cost", tickvals=[1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
                 ticktext=["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])
fig.update_xaxes(title="Strike / forward")
fig.update_layout(height=430, margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, width="stretch")
st.caption("Grey: raw gap. Blue: final gap, inside the spread. Orange: final gap that still beats the spread.")

# The funding rate the options market implies, one point per long expiry
rates = p.drop_duplicates(["ticker", "expiry"]).dropna(subset=["rate_implied"])
if len(rates):
    ui.section("Implied funding rate", "the slope of call minus put across strikes, expiries over 90 days")
    fig = go.Figure()
    for i, (t, one) in enumerate(rates.groupby("ticker", sort=False)):
        fig.add_scatter(x=one["days_to_expiry"], y=one["rate_implied"] * 100, mode="markers", name=t,
                        marker=dict(size=11, color=ui.TICKER_COLOR.get(t), symbol=ui.TICKER_SYMBOL.get(t),
                                    line=dict(color=ui.CARD, width=1.5)),
                        hovertemplate=f"{t}" + "<br>%{x:.0f} days<br>implied rate %{y:.2f}%<extra></extra>")
    fig.add_hline(y=rates["r"].iloc[0] * 100, line_dash="dash", line_color=ui.INK_2,
                  annotation_text=f"3 month T-bill {rates['r'].iloc[0] * 100:.2f}%")
    fig.update_xaxes(title="Days to expiry")
    fig.update_yaxes(title="Implied funding rate (%)")
    fig.update_layout(height=340, margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    spread = (rates["rate_implied"] - rates["r"]).median() * 1e4
    long_end = rates[rates["days_to_expiry"] > 365]
    long_text = (f" Past one year the estimates cluster tightly, {long_end['rate_implied'].min() * 100:.1f}% to "
                 f"{long_end['rate_implied'].max() * 100:.1f}%. Under a year they scatter more, because a rate gap "
                 "moves call minus put by less there, so quote noise matters more." if len(long_end) >= 3 else "")
    st.caption(f"Median {spread:.0f} bp over the T-bill.{long_text} Parity says European call minus put is a straight line in "
               "the strike, and its slope is the discount factor the market actually uses. Desks call this the box "
               "spread rate. It sits above T-bills because option positions are funded through brokers, not the "
               "Treasury. Fitted on strikes below the forward, where puts carry little early exercise value.")

ui.section("Pairs that still beat the spread", "after every adjustment")
flagged = p[stage4 | arb].copy()
if flagged.empty:
    st.success("No pair survives all the adjustments. No tradeable parity break in this data.")
else:
    flagged["expiry"] = flagged["expiry"].dt.date
    cols = {"ticker": "Ticker", "expiry": "Expiry", "strike": "Strike", "days_to_expiry": "Days",
            "bid_c": "Call bid", "ask_c": "Call ask", "bid_p": "Put bid", "ask_p": "Put ask",
            "residual_ex_rate": "Final gap $", "cost": "Spread cost $", "true_arb": "True arbitrage"}
    show = flagged[list(cols)].rename(columns=cols).sort_values("Final gap $", key=abs, ascending=False)
    st.dataframe(show.round(3), hide_index=True, width="stretch")
    # Described from the data itself, so the text never claims something the table contradicts
    gap, cost = flagged["residual_ex_rate"].abs().median(), flagged["cost"].median()
    long_share = (flagged["days_to_expiry"] > 365).mean()
    top = flagged["ticker"].value_counts()
    st.caption(
        f"{len(flagged)} pairs, median leftover gap ${gap:.2f} against a median spread cost of ${cost:.2f}. "
        f"{top.index[0]} accounts for {top.iloc[0]} of them, and {long_share:.0%} expire in over a year. "
        "Likely causes: dividends paid in lumps on set dates while the model uses a smooth yield (this is what "
        "makes deep in the money American calls worth exercising just before an ex-dividend date), and stale "
        "quotes on thinly traded strikes in a delayed end of day snapshot."
    )
    st.download_button("Download as CSV", show.to_csv(index=False), "parity_flagged.csv", "text/csv")
