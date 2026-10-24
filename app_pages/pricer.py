"""Pricer page: pick a real contract or type your own inputs, see all three models side by side."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.app_data import TICKERS, expiry_label, load_chain, load_svi, smile_vol
from src.pricing import binomial_greeks, binomial_price, bs_greeks, bs_price, mc_price

ui.setup()
ui.header("Pricing engine", "Contract pricer",
          "Pick a listed contract to start from live market inputs, then change anything you like. "
          "All three models are built from scratch and run on every change.")

c1, c2, c3, c4 = st.columns(4)
ticker = c1.selectbox("Ticker", TICKERS)
chain = load_chain(ticker)
svi = load_svi(ticker)

expiries = chain[["expiry", "days_to_expiry"]].drop_duplicates().sort_values("expiry")
labels = {expiry_label(e, d): e for e, d in zip(expiries["expiry"], expiries["days_to_expiry"])}
default_exp = int((expiries["days_to_expiry"] - 30).abs().values.argmin())
expiry = labels[c2.selectbox("Expiry", list(labels), index=default_exp)]
kind = c3.radio("Type", ["call", "put"], horizontal=True)

one = chain[(chain["expiry"] == expiry) & (chain["option_type"] == kind)].sort_values("strike")
if one.empty:
    st.warning("No clean quotes for this type on this expiry. Try the other type or another expiry.")
    st.stop()
spot = float(one["spot"].iloc[0])
nearest = int((one["strike"] - one["forward"]).abs().values.argmin())
strike = c4.selectbox("Strike", one["strike"].tolist(), index=nearest)
row = one[one["strike"] == strike].iloc[0]

# Vol to start from: the SVI smile at this strike if the expiry fit cleanly,
# otherwise the contract's own implied vol
fit = svi[svi["expiry"] == expiry]
if len(fit):
    vol0 = float(smile_vol(fit.iloc[0], [strike], row["forward"])[0])
    vol_note = "SVI smile vol at this strike"
else:
    vol0 = float(row["iv"]) if np.isfinite(row["iv"]) else 0.3
    vol_note = "this contract's own implied vol (its expiry has no clean SVI fit)"

ui.section("Inputs", "prefilled from the market, editable")
key = f"{ticker}-{expiry}-{kind}-{strike}"  # new defaults whenever the contract changes
i1, i2, i3, i4, i5, i6 = st.columns(6)
S = i1.number_input("Spot", value=spot, min_value=0.01, format="%.2f", key=f"S{key}")
K = i2.number_input("Strike", value=float(strike), min_value=0.01, format="%.2f", key=f"K{key}")
days = i3.number_input("Days to expiry", value=float(row["days_to_expiry"]), min_value=0.01,
                       format="%.2f", key=f"D{key}")
r = i4.number_input("Rate %", value=float(row["r"]) * 100, format="%.3f", key=f"r{key}") / 100
q = i5.number_input("Carry %", value=float(row["q"]) * 100, format="%.3f", key=f"q{key}",
                    help="Dividend yield plus any implied carry, backed out of put-call parity (Step 6).") / 100
sigma = i6.number_input("Vol %", value=vol0 * 100, min_value=0.1, format="%.2f", key=f"v{key}",
                        help=f"Starts at the {vol_note}.") / 100
T = days / 365

bs = float(bs_price(S, K, T, r, q, sigma, kind))
tree_am = binomial_price(S, K, T, r, q, sigma, kind, steps=500, american=True)
tree_eu = binomial_price(S, K, T, r, q, sigma, kind, steps=500, american=False)
mc = mc_price(S, K, T, r, q, sigma, kind, paths=100_000)

ui.section("Model prices", f"{ticker} {pd.Timestamp(expiry).date()} {strike:g} {kind}")
ui.kpis([
    {"label": "Market mid", "value": f"${row['mid']:.2f}", "note": f"bid ${row['bid']:.2f} · ask ${row['ask']:.2f}",
     "color": ui.MUTED},
    {"label": "Black-Scholes", "value": f"${bs:.2f}", "note": f"{bs - row['mid']:+.2f} vs mid · European",
     "color": ui.CALL},
    {"label": "Binomial tree", "value": f"${tree_am:.2f}", "note": f"{tree_am - row['mid']:+.2f} vs mid · American",
     "color": ui.PUT},
    {"label": "Monte Carlo", "value": f"${mc['price']:.2f}",
     "note": f"95% CI {mc['ci_low']:.2f} to {mc['ci_high']:.2f}", "color": ui.MC},
])

# Dots, not bars: the prices differ by cents, and bars that do not start at
# zero would visually exaggerate that. The shaded band is the market bid to ask.
fig = go.Figure()
fig.add_hrect(y0=row["bid"], y1=row["ask"], fillcolor="rgba(170,178,197,0.12)", opacity=1, line_width=0)
fig.add_hline(y=row["mid"], line_dash="dash", line_color=ui.MUTED,
              annotation_text=f"market mid ${row['mid']:.2f}, shaded band is bid to ask",
              annotation_position="top left")
names = ["Black-Scholes", "Binomial (American)", "Monte Carlo"]
values = [bs, tree_am, mc["price"]]
fig.add_scatter(x=names, y=values, mode="markers+text", text=[f"${v:.2f}" for v in values],
                textposition="middle right", textfont=dict(color=ui.INK, family=ui.MONO),
                marker=dict(size=14, color=ui.SERIES, line=dict(color=ui.CARD, width=2)),
                error_y=dict(type="data", array=[0, 0, 1.96 * mc["std_err"]], visible=True, thickness=2, color=ui.INK_2),
                hovertemplate="%{x}: $%{y:.4f}<extra></extra>")
lo, hi = min(values + [row["bid"]]), max(values + [row["ask"]])
pad = max(0.25 * (hi - lo), 0.02)
fig.update_yaxes(range=[lo - pad, hi + pad], title="Price ($)")
fig.update_xaxes(range=[-0.4, 2.7])  # room for the price labels to the right of each dot
fig.update_layout(height=340, margin=dict(t=30, b=40, l=60, r=20), showlegend=False)
st.plotly_chart(fig, width="stretch")
st.caption("The Monte Carlo error bar is its 95% confidence interval.")

early = tree_am - tree_eu
ui.note(
    f"<b>Early exercise value: ${early:.3f}</b>, the American tree minus the same tree with exercise switched off. "
    + ("It is close to zero here, so all three models agree. "
       if early < 0.01 else "This is the part Black-Scholes and Monte Carlo cannot see. ")
    + f"Monte Carlo standard error is ${mc['std_err']:.4f} with 100,000 antithetic paths."
)
st.write("")

with st.expander("Greeks for this contract"):
    g_bs = bs_greeks(S, K, T, r, q, sigma, kind)
    g_tree = binomial_greeks(S, K, T, r, q, sigma, kind, steps=500, american=True)
    scale = {"delta": 1, "gamma": 1, "vega": 0.01, "theta": 1 / 365, "rho": 0.01}
    units = {"delta": "per $1 in the stock", "gamma": "delta change per $1", "vega": "per 1 vol point",
             "theta": "per calendar day", "rho": "per 1% in rates"}
    st.dataframe(pd.DataFrame({
        "Greek": list(scale),
        "Black-Scholes": [float(g_bs[g]) * scale[g] for g in scale],
        "Binomial (American)": [g_tree[g] * scale[g] for g in scale],
        "Units": [units[g] for g in scale],
    }).round(4), hide_index=True, width="stretch")
