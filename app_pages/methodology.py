"""Methodology page: the formulas, the assumptions, the convergence tests and the limitations."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.app_data import REPORTS, as_of

ui.setup()
ui.header("Documentation", "Methodology",
          "Data, formulas, convergence tests and limitations. All three pricers are written from scratch, "
          "no pricing libraries.", as_of())

ui.section("1. Data")
st.markdown(
    """
- **Option chains** for SPY, AAPL, NVDA, TSLA and JPM from Yahoo Finance: strikes, expiries, bid, ask,
  volume, open interest. A GitHub Actions job saves a fresh snapshot every weekday after the close.
- **Risk free rate:** 3 month Treasury yield (FRED series DGS3MO), converted from a simple annual rate
  to continuous compounding.
- **Market price** of each contract is the bid/ask midpoint.
- **Cleaning** drops contracts with a zero bid, open interest under 100, a spread over 20% of the mid,
  no trade in the last 7 days, or a price outside no arbitrage bounds. Every filter logs how many it removed.
- **Time to expiry** runs to 4 PM New York time on expiry day, in years.
"""
)

ui.section("2. The three pricers")
st.subheader("Black-Scholes with a dividend yield (European)")
st.latex(r"C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)")
st.latex(r"d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}")
st.markdown("Greeks are the closed form derivatives of these formulas.")

st.subheader("Binomial tree (American), 500 steps")
st.latex(r"u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{(r-q)\Delta t} - d}{u - d}")
st.latex(r"V_{node} = \max\left(\text{exercise now},\; e^{-r\Delta t}\left[p\,V_{up} + (1-p)\,V_{down}\right]\right)")
st.markdown(
    "Every node checks whether exercising beats holding, which is what makes US puts worth more than "
    "European ones. Delta, gamma and theta are read off the first two layers of the tree. Vega and rho "
    "use central finite differences."
)

st.subheader("Monte Carlo, 100,000 paths with antithetic variates")
st.latex(r"S_T = S_0 \exp\left[(r - q - \tfrac{1}{2}\sigma^2)T \pm \sigma\sqrt{T}\,Z\right], \qquad "
         r"V = e^{-rT}\,\overline{\tfrac{1}{2}\left[\text{payoff}(S_T^{+}) + \text{payoff}(S_T^{-})\right]}")
st.markdown("Each draw Z is paired with minus Z. The 95% confidence interval comes from the pair averages.")

ui.section("3. Convergence tests")
st.markdown(
    "For a European option all three models must agree, so Black-Scholes is the answer key. "
    "These charts use a real SPY call about 30 days out, at the money."
)
conv = pd.read_csv(REPORTS / "convergence.csv")
tree, mc = conv[conv["method"] == "binomial"], conv[conv["method"] == "monte_carlo"]
exact = (tree["price"] - tree["error"]).iloc[0]
c1, c2 = st.columns(2)
fig = go.Figure()
fig.add_scatter(x=tree["n"], y=tree["price"], mode="lines+markers", name="binomial tree",
                line=dict(color=ui.PUT, width=2), marker=dict(size=8, line=dict(color=ui.CARD, width=2)))
fig.add_hline(y=exact, line_dash="dash", line_color=ui.CALL, annotation_text=f"Black-Scholes {exact:.4f}")
fig.add_vline(x=500, line_dash="dot", line_color=ui.MUTED, annotation_text="500 steps used")
fig.update_xaxes(type="log", title="Steps")
fig.update_yaxes(title="Call price ($)")
fig.update_layout(height=360, margin=dict(t=30, b=10), showlegend=False, title="Binomial tree")
c1.plotly_chart(fig, width="stretch")
fig = go.Figure()
fig.add_scatter(x=mc["n"], y=mc["ci_high"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=mc["n"], y=mc["ci_low"], mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor="rgba(25,158,112,0.22)", name="95% interval")
fig.add_scatter(x=mc["n"], y=mc["price"], mode="lines+markers", name="Monte Carlo",
                line=dict(color=ui.MC, width=2), marker=dict(size=8, line=dict(color=ui.CARD, width=2)))
fig.add_hline(y=exact, line_dash="dash", line_color=ui.CALL, annotation_text=f"Black-Scholes {exact:.4f}")
fig.add_vline(x=100_000, line_dash="dot", line_color=ui.MUTED, annotation_text="100,000 used")
fig.update_xaxes(type="log", title="Paths")
fig.update_yaxes(title="Call price ($)")
fig.update_layout(height=360, margin=dict(t=30, b=10), showlegend=False, title="Monte Carlo")
c2.plotly_chart(fig, width="stretch")
at500 = tree.loc[tree["n"] == 500, "error"].abs().iloc[0]
ratio = mc["std_err"].iloc[0] / mc["std_err"].iloc[-1]
st.caption(f"At 500 steps the tree is ${at500:.4f} from Black-Scholes. Going from 1,000 to 1,000,000 paths "
           f"cut the Monte Carlo error {ratio:.1f}x, against 31.6x expected from the square root rule.")

ui.section("4. Implied vol, forward and surface")
st.markdown(
    """
- **Implied vol** for every contract with Brent's method. Contracts with no solution (price outside
  what Black-Scholes can produce at any vol, or too little vega to pin a vol down) are logged, never guessed.
- **The forward** is read from the market through put-call parity on strikes near the money, after
  removing each option's early exercise value with the tree. The market's own forward fixed a visible
  jump between call and put vols at the money that a computed forward left behind.
- **SVI** is fitted to the out of the money vols of each expiry:
"""
)
st.latex(r"w(k) = a + b\left(\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\right), \qquad k = \ln(K/F), \quad w = \sigma_{IV}^2 T")
st.markdown(
    "Fits are constrained so variance stays positive and the wings obey Roger Lee's bound, then checked "
    "for butterfly arbitrage (Gatheral's density test) and calendar arbitrage (total variance must rise with expiry)."
)

ui.section("5. Limitations")
st.markdown(
    """
- **Yahoo data is delayed and end of day.** Quotes for different contracts are not perfectly simultaneous,
  so small inconsistencies are expected, especially on illiquid strikes.
- **Mid price is an approximation.** The true price can be anywhere inside the spread. Some deep in the
  money AAPL calls have no implied vol because their mid sits a few cents under the arbitrage floor
  while their spread is over $3 wide.
- **American options are priced with the binomial tree only.** Black-Scholes and Monte Carlo are European
  by construction, which is the point of comparing them.
- **Dividends are a continuous yield.** Real dividends are paid in lumps on set dates. This is the most
  likely source of the small parity gaps left on SPY.
- **One risk free rate for pricing.** The options market funds at rates above the 3 month T-bill. The
  parity check measures this directly from the slope of call minus put across strikes (the box spread rate),
  but the pricers still discount at the T-bill. It also shows up as a slightly negative implied carry on
  short expiries with no dividend inside them.
- **Thin chains fit worse.** JPM has the fewest liquid contracts, and its shortest expiry has only a few quotes.
"""
)
