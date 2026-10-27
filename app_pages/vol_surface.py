"""Vol Surface page: the SVI surface in 3D, the smile for any expiry, and the ATM term structure."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.app_data import TICKERS, atm_vol_30d, expiry_label, load_chain, load_surface, load_svi, skew_label, smile_vol

# One hue, dark to light, so higher vol reads as brighter on the dark background
VOL_SCALE = [[0.0, "#0d366b"], [0.25, "#1c5cab"], [0.5, "#3987e5"], [0.75, "#86b6ef"], [1.0, "#e3eefc"]]

ui.setup()
ui.header("Volatility", "Implied volatility surface",
          "Implied vol from every out of the money contract, smoothed per expiry with an SVI fit, the same "
          "parameterisation trading desks use.")

ticker = st.selectbox("Ticker", TICKERS)
grid = load_surface(ticker)
svi = load_svi(ticker)
chain = load_chain(ticker)

ui.kpis([
    {"label": "30 day ATM vol", "value": f"{atm_vol_30d(ticker) * 100:.1f}%", "note": "SVI at the forward"},
    {"label": "30 day skew", "value": skew_label(ticker), "note": "95% strike minus 105% strike",
     "color": ui.PUT},
    {"label": "Expiries fitted", "value": f"{len(svi)}", "note": "clean SVI fits", "color": ui.CALL},
    {"label": "Median fit error", "value": f"{svi['rmse_vol_pts'].median():.2f} pts", "note": "SVI vs market vols",
     "color": ui.MC},
])

tab3d, tab_smile, tab_term = st.tabs(["3D surface", "Smile by expiry", "Term structure"])

with tab3d:
    # Only points backed by real quotes. Extrapolated wings stay blank on purpose.
    d1, d2, d3 = st.columns([2, 2, 1])
    fill = d3.toggle("Fill beyond quotes", help="Draw the SVI curve outside the quoted strikes too. "
                                                "That part is extrapolation, not market data.")
    shown = grid if fill else grid.assign(iv=grid["iv"].where(~grid["extrapolated"]))
    pivot = shown.pivot(index="days", columns="moneyness", values="iv") * 100
    top = int(pivot.index.max()) + 1
    day_range = d1.slider("Expiries between (days)", 0, top, (7, min(400, top)),
                          help="Under a week, quotes cover only a few strikes, so those slices are short "
                               "and spiky. They are hidden by default.")
    mny_range = d2.slider("Strike / spot", 0.70, 1.30, (0.80, 1.20), step=0.05)
    pivot = pivot[(pivot.index >= day_range[0]) & (pivot.index <= day_range[1])]
    pivot = pivot.loc[:, (pivot.columns >= mny_range[0] - 1e-9) & (pivot.columns <= mny_range[1] + 1e-9)]
    if pivot.shape[0] < 2:
        st.info("Widen the day range to show at least two expiries.")
        st.stop()
    fig = go.Figure(go.Surface(x=pivot.columns, y=pivot.index, z=pivot.values, colorscale=VOL_SCALE,
                               colorbar=dict(title="IV %", thickness=12, outlinewidth=0),
                               contours=dict(z=dict(show=True, usecolormap=True, project_z=False, width=1)),
                               hovertemplate="Strike/spot %{x:.2f}<br>%{y:.0f} days<br>IV %{z:.1f}%<extra></extra>"))
    fig.update_layout(height=620, margin=dict(t=10, b=10, l=0, r=0),
                      scene=dict(xaxis_title="Strike / spot", yaxis_title="Days to expiry",
                                 zaxis_title="Implied vol (%)", aspectmode="manual",
                                 aspectratio=dict(x=1.25, y=1.4, z=0.65),
                                 camera=dict(eye=dict(x=1.55, y=-1.75, z=0.95), center=dict(x=0, y=0, z=-0.12))))
    st.plotly_chart(fig, width="stretch")
    st.caption("Drag to rotate, scroll to zoom. The stepped edge is real: short expiries are only quoted near "
               "spot, longer ones over a wider strike range. Blank areas have no quotes unless you turn on the fill.")

with tab_smile:
    labels = {expiry_label(e, d): i for i, (e, d) in enumerate(zip(svi["expiry"], svi["days"]))}
    pick = svi.iloc[labels[st.selectbox("Expiry", list(labels), index=int((svi["days"] - 30).abs().argmin()))]]
    pts = chain[(chain["expiry"] == pick["expiry"]) & (chain["iv_status"] == "ok") & chain["otm"]]
    fwd, spot = pts["forward"].iloc[0], pts["spot"].iloc[0]
    k = np.linspace(pts["strike"].min(), pts["strike"].max(), 200)

    fig = go.Figure()
    for kind, color in (("put", ui.PUT), ("call", ui.CALL)):
        side = pts[pts["option_type"] == kind]
        fig.add_scatter(x=side["strike"] / spot, y=side["iv"] * 100, mode="markers", name=f"market OTM {kind}s",
                        marker=dict(color=color, size=8, opacity=0.85, line=dict(color=ui.CARD, width=1)),
                        customdata=np.c_[side["strike"], side["bid"], side["ask"]],
                        hovertemplate="Strike %{customdata[0]}<br>IV %{y:.2f}%<br>bid %{customdata[1]} ask "
                                      "%{customdata[2]}<extra></extra>")
    fig.add_scatter(x=k / spot, y=smile_vol(pick, k, fwd) * 100, mode="lines", name="SVI fit",
                    line=dict(color=ui.INK, width=2))
    fig.add_vline(x=fwd / spot, line_dash="dot", line_color=ui.MUTED, annotation_text="forward")
    fig.update_layout(height=480, margin=dict(t=40, b=10), legend=dict(orientation="h", y=1.12),
                      title=f"{ticker} {pick['expiry'].date()}: fit error {pick['rmse_vol_pts']:.2f} vol points")
    fig.update_xaxes(title="Strike / spot")
    fig.update_yaxes(title="Implied vol (%)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Puts left of the forward, calls right of it. Each side uses only out of the money contracts, "
               "which carry almost no early exercise value, so their Black-Scholes vol is clean.")

with tab_term:
    fig = go.Figure()
    fig.add_scatter(x=svi["days"], y=svi["atm_vol"] * 100, mode="lines+markers", name="ATM vol",
                    line=dict(color=ui.CALL, width=2), marker=dict(size=8, line=dict(color=ui.CARD, width=2)),
                    hovertemplate="%{x:.0f} days<br>ATM vol %{y:.1f}%<extra></extra>")
    fig.update_layout(height=420, margin=dict(t=30, b=10),
                      title=f"{ticker}: at the money vol by expiry")
    fig.update_xaxes(title="Days to expiry", type="log")
    fig.update_yaxes(title="ATM implied vol (%)")
    st.plotly_chart(fig, width="stretch")
    st.caption("A bump at one expiry, above its neighbours, usually means an event such as earnings falls "
               "before that date. Log scale on days so the short end is readable.")
