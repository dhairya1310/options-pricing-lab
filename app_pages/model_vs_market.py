"""Model vs Market page: where each model misprices the market, by moneyness and expiry."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.app_data import TICKERS, load_model_vs_market
from src.model_vs_market import EXPIRY_LABELS, MODEL_NAMES, MONEYNESS_LABELS

ui.setup()
ui.header("Model risk", "Model vs market",
          "Every model gets one vol per expiry, the at the money vol from the SVI fit, and prices every contract "
          "on that expiry: the textbook one vol fits all assumption. Error is model minus market mid, as a share "
          "of the market price.")

df = load_model_vs_market()
c1, c2, c3 = st.columns(3)
ticker = c1.selectbox("Ticker", ["All tickers"] + TICKERS)
model = c2.selectbox("Model", list(MODEL_NAMES), format_func=MODEL_NAMES.get)
kind = c3.radio("Type", ["put", "call"], horizontal=True)

sub = df if ticker == "All tickers" else df[df["ticker"] == ticker]
sub = sub[sub["option_type"] == kind]

table = (sub.groupby(["expiry_bucket", "moneyness_bucket"], observed=False)[f"{model}_rel_err"]
         .median().unstack().reindex(index=EXPIRY_LABELS, columns=MONEYNESS_LABELS) * 100)
counts = (sub.groupby(["expiry_bucket", "moneyness_bucket"], observed=False).size().unstack()
          .reindex(index=EXPIRY_LABELS, columns=MONEYNESS_LABELS))
# Empty cells stay blank, and tiny negatives print as 0% rather than -0%
text = [[(f"{v:.0f}%".replace("-0%", "0%") if np.isfinite(v) else "") for v in row] for row in table.values]

fig = go.Figure(go.Heatmap(
    z=table.values, x=MONEYNESS_LABELS, y=EXPIRY_LABELS, text=text, texttemplate="%{text}", textfont=dict(family=ui.MONO, size=12, color=ui.INK),
    xgap=2, ygap=2,
    customdata=counts.values, colorscale=ui.DIVERGING, zmid=0, zmin=-50, zmax=50,
    colorbar=dict(title="Median error %"),
    hovertemplate="Strike/forward %{x}<br>Expiry %{y}<br>Median error %{z:.1f}%<br>%{customdata} contracts<extra></extra>",
))
fig.update_layout(height=430, margin=dict(t=30, b=10),
                  title=f"{MODEL_NAMES[model]}, {kind}s, {ticker}: blue means the model is too cheap")
fig.update_xaxes(title="Strike / forward")
fig.update_yaxes(title="Time to expiry", autorange="reversed")
st.plotly_chart(fig, width="stretch")
st.caption("Hover a cell for the contract count. Cells with only a handful of contracts are noisy.")

ui.note(
    """
<b>How to read it.</b> Near the forward (the middle column) errors are small because that is where
the vol came from. Far below the forward, puts come out far too cheap: the market prices crash
risk with a much higher vol than at the money, which is the <b>skew</b>. Black-Scholes and Monte Carlo
make the same errors because they price the same European option. The tree only differs where
early exercise is worth something.
"""
)

ui.section("Does the American tree beat Black-Scholes where early exercise matters?",
           "ITM puts over 30 days, priced at the smile vol so only early exercise is left")
itm = df[(df["option_type"] == "put") & (df["moneyness"] > 1.05) & (df["days_to_expiry"] > 30)
         & ~df["smile_extrapolated"]]
summary = (itm.groupby("ticker")
           .agg(contracts=("ticker", "size"),
                bs=("bs_smile_err", lambda s: s.abs().median()),
                tree=("tree_smile_err", lambda s: s.abs().median()))
           .reindex(TICKERS).dropna().reset_index())
long = summary.melt(id_vars=["ticker", "contracts"], value_vars=["bs", "tree"],
                    var_name="model", value_name="error")
long["model"] = long["model"].map({"bs": "Black-Scholes", "tree": "Binomial (American)"})
bar = px.bar(long, x="ticker", y="error", color="model", barmode="group", text_auto=".2f",
             color_discrete_map={"Black-Scholes": ui.CALL, "Binomial (American)": ui.PUT},
             labels={"error": "Median |error| ($)", "ticker": ""}, hover_data=["contracts"])
bar.update_traces(marker_line_width=0, textfont=dict(family=ui.MONO))
bar.update_layout(height=360, margin=dict(t=20, b=10), legend_title_text="", bargap=0.35, bargroupgap=0.08,
                  legend=dict(orientation="h", y=1.1))
st.plotly_chart(bar, width="stretch")
