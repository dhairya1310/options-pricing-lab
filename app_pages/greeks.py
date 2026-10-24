"""Greeks page: all five Greeks across strikes, with sliders to move the stock and the clock."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import ui
from src.app_data import TICKERS, expiry_label, load_chain, load_svi, smile_vol
from src.pricing import binomial_greeks, bs_greeks

GREEKS = ["delta", "gamma", "vega", "theta", "rho"]
SCALE = {"delta": 1, "gamma": 1, "vega": 0.01, "theta": 1 / 365, "rho": 0.01}
UNITS = {"delta": "per $1 in the stock", "gamma": "delta change per $1", "vega": "$ per 1 vol point",
         "theta": "$ per calendar day", "rho": "$ per 1% in rates"}
COLORS = {"call": ui.CALL, "put": ui.PUT}

ui.setup()
ui.header("Risk", "Greeks explorer",
          "Black-Scholes Greeks across every strike of one expiry. Move the stock or the clock and watch the "
          "risk profile change, or overlay the American tree to see early exercise.")

c1, c2, c3 = st.columns(3)
ticker = c1.selectbox("Ticker", TICKERS)
svi = load_svi(ticker)
chain = load_chain(ticker)
labels = {expiry_label(e, d): i for i, (e, d) in enumerate(zip(svi["expiry"], svi["days"]))}
pick = svi.iloc[labels[c2.selectbox("Expiry", list(labels), index=int((svi["days"] - 30).abs().argmin()))]]
vol_mode = c3.radio("Vol", ["Smile (SVI per strike)", "Flat (ATM vol)"], horizontal=True,
                    help="Smile keeps each strike at its own SVI vol (sticky strike). Flat uses one vol for all.")

one = chain[chain["expiry"] == pick["expiry"]]
spot0, r, q, fwd0 = one["spot"].iloc[0], one["r"].iloc[0], one["q"].iloc[0], one["forward"].iloc[0]
T0 = pick["T"]

s1, s2, s3 = st.columns([2, 2, 1])
shift = s1.slider("Move the stock (%)", -20, 20, 0, help="Shift the spot price up or down.")
max_days = max(0, int(np.floor(pick["days"] - 0.5)))
passed = s2.slider("Days pass", 0, max_days, 0, help="Move the clock forward toward expiry.") if max_days else 0
show_tree = s3.toggle("Overlay American tree", help="Binomial Greeks on a sample of strikes. Takes a second.")

S = spot0 * (1 + shift / 100)
T = T0 - passed / 365
strikes = np.linspace(one["strike"].min(), one["strike"].max(), 150)

def vols(k):
    if vol_mode.startswith("Flat"):
        return np.full(len(k), pick["atm_vol"])
    return smile_vol(pick, k, fwd0)

fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{g.title()} ({UNITS[g]})" for g in GREEKS],
                    horizontal_spacing=0.07, vertical_spacing=0.16)
spots = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
for kind in ("call", "put"):
    g = bs_greeks(S, strikes, T, r, q, vols(strikes), kind)
    for name, (row, col) in zip(GREEKS, spots):
        fig.add_scatter(x=strikes, y=g[name] * SCALE[name], mode="lines", name=f"{kind}s (BS)",
                        line=dict(color=COLORS[kind], width=2), legendgroup=kind, showlegend=(name == "delta"),
                        row=row, col=col)

if show_tree:
    sample = np.linspace(strikes[0], strikes[-1], 15)
    sample_vols = vols(sample)
    with st.spinner("Running the trees..."):
        for kind in ("call", "put"):
            pts = [binomial_greeks(S, k, T, r, q, v, kind, steps=500, american=True)
                   for k, v in zip(sample, sample_vols)]
            for name, (row, col) in zip(GREEKS, spots):
                fig.add_scatter(x=sample, y=[p[name] * SCALE[name] for p in pts], mode="markers",
                                name=f"{kind}s (American tree)", marker=dict(color=COLORS[kind], symbol="x", size=7),
                                legendgroup=f"{kind}-tree", showlegend=(name == "delta"), row=row, col=col)

for row, col in spots:
    fig.add_vline(x=S, line_dash="dot", line_color=ui.MUTED, row=row, col=col)
fig.update_xaxes(title_text="Strike")
fig.update_annotations(font=dict(color=ui.INK, size=12))
fig.update_layout(height=700, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.08))
st.plotly_chart(fig, width="stretch")

st.caption(
    f"Spot ${S:,.2f} (dotted line), {T * 365:.1f} days left, rate {r * 100:.2f}%, carry {q * 100:.2f}%. "
    "Call and put gamma and vega are identical, so those lines sit on top of each other."
)
ui.note(
    """
<b>What to look for.</b> Gamma and vega peak near the money and shrink as expiry approaches, so slide
<b>Days pass</b> to the right and watch gamma spike into a narrow peak. Theta is most negative near the
money. With the tree overlay on, deep in the money <b>put</b> deltas sit closer to -1 than Black-Scholes:
that is early exercise, the reason US single stock options need an American model.
"""
)
