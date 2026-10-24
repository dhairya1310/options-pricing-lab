"""
The look of the app in one place: colours, fonts, the chart theme and a few
building blocks (page header, KPI cards, section titles).

Every page calls setup() first. Change a colour here and it changes everywhere.
"""

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Surfaces and ink, a dark trading terminal palette
PAGE = "#0a0e17"
CARD = "#111827"
CARD_EDGE = "#1f2937"
GRID = "#1c2433"
INK = "#e8ebf1"
INK_2 = "#aab2c5"
MUTED = "#6b7489"
ACCENT = "#19c39b"   # brand teal, used for UI chrome only, never as a data series

# Data colours. Validated as a set against the dark card surface with the
# dataviz palette checker: lightness band, chroma, colourblind separation and
# contrast all pass. Fixed roles, so a colour always means the same thing.
CALL = "#3987e5"     # also Black-Scholes
PUT = "#d95926"      # also the American tree
MC = "#199e70"       # Monte Carlo
SERIES = [CALL, PUT, MC]

# One fixed colour and marker shape per ticker, so a ticker looks the same on
# every chart. Five hues validated together on the card surface. On scatter
# plots the shape is the backup channel for colourblind readers.
TICKER_COLOR = {"SPY": "#3987e5", "AAPL": "#d95926", "NVDA": "#199e70", "TSLA": "#c98500", "JPM": "#d55181"}
TICKER_SYMBOL = {"SPY": "circle", "AAPL": "square", "NVDA": "diamond", "TSLA": "triangle-up", "JPM": "x"}

# Diverging scale for pricing error: blue means the model is too cheap,
# red too rich, a neutral grey means spot on
DIVERGING = [[0.0, "#1c5cab"], [0.25, "#3987e5"], [0.5, "#2a2f3a"], [0.75, "#e66767"], [1.0, "#b8322f"]]

FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
MONO = "JetBrains Mono, SFMono-Regular, Menlo, monospace"

_CSS = f"""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap");
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
    font-family: {FONT};
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: #0d1220; border-right: 1px solid {CARD_EDGE}; }}
section[data-testid="stSidebar"] a span {{ font-weight: 500; letter-spacing: .01em; }}

/* headings */
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h2 {{ font-size: 1.15rem !important; font-weight: 600 !important; color: {INK} !important; }}
h3 {{ font-size: 1rem !important; font-weight: 600 !important; }}

/* page header */
.lab-hero {{ border-bottom: 1px solid {CARD_EDGE}; padding-bottom: 1.1rem; margin-bottom: 1.4rem; }}
.lab-eyebrow {{ font: 600 .72rem/1 {MONO}; letter-spacing: .14em; text-transform: uppercase; color: {ACCENT}; }}
.lab-title {{ font-size: 2.05rem; font-weight: 700; color: {INK}; margin: .45rem 0 .35rem; letter-spacing: -0.02em; }}
.lab-sub {{ color: {INK_2}; font-size: .97rem; max-width: 880px; line-height: 1.5; }}
.lab-pill {{ display: inline-flex; align-items: center; gap: .45rem; margin-top: .8rem;
    font: 500 .74rem/1 {MONO}; color: {INK_2}; background: {CARD}; border: 1px solid {CARD_EDGE};
    padding: .38rem .65rem; border-radius: 999px; }}
.lab-dot {{ width: 7px; height: 7px; border-radius: 50%; background: {ACCENT};
    box-shadow: 0 0 0 3px rgba(25,195,155,.18); }}

/* KPI cards */
.lab-kpis {{ display: grid; gap: .8rem; margin: .4rem 0 1.2rem; }}
.lab-kpi {{ background: linear-gradient(180deg, #131b2b 0%, {CARD} 100%); border: 1px solid {CARD_EDGE};
    border-radius: 12px; padding: .95rem 1.05rem; position: relative; overflow: hidden; }}
.lab-kpi::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--bar, {ACCENT}); }}
.lab-kpi-label {{ font: 600 .68rem/1.2 {MONO}; letter-spacing: .1em; text-transform: uppercase; color: {MUTED}; }}
.lab-kpi-value {{ font: 600 1.6rem/1.15 {MONO}; color: {INK}; margin-top: .45rem; font-variant-numeric: tabular-nums; }}
.lab-kpi-note {{ font-size: .78rem; color: {INK_2}; margin-top: .3rem; }}

/* section titles */
.lab-section {{ display: flex; align-items: baseline; gap: .7rem; margin: 1.6rem 0 .5rem; }}
.lab-section-title {{ font-size: 1.08rem; font-weight: 600; color: {INK}; }}
.lab-section-note {{ font-size: .82rem; color: {MUTED}; }}

/* callout */
.lab-note {{ background: {CARD}; border: 1px solid {CARD_EDGE}; border-left: 3px solid {ACCENT};
    border-radius: 10px; padding: .85rem 1rem; color: {INK_2}; font-size: .9rem; line-height: 1.55; }}
.lab-note b {{ color: {INK}; }}

/* streamlit widgets */
div[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {CARD_EDGE}; border-radius: 12px; padding: .8rem 1rem; }}
div[data-testid="stMetricValue"] {{ font-family: {MONO}; font-weight: 600; }}
div[data-testid="stMetricLabel"] p {{ font: 600 .7rem/1.2 {MONO} !important; letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; }}
div[data-testid="stPlotlyChart"] {{ background: {CARD}; border: 1px solid {CARD_EDGE}; border-radius: 12px;
    padding: 0; overflow: hidden; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {CARD_EDGE}; border-radius: 12px; overflow: hidden; }}
div[data-testid="stExpander"] details {{ background: {CARD}; border: 1px solid {CARD_EDGE}; border-radius: 12px; }}
button[data-baseweb="tab"] p {{ font-weight: 600; }}
hr {{ border-color: {CARD_EDGE} !important; }}
</style>
"""


def _register_chart_theme() -> None:
    """One plotly template so every chart matches the app without per chart styling."""
    axis = dict(gridcolor=GRID, zerolinecolor=CARD_EDGE, linecolor=CARD_EDGE,
                tickfont=dict(color=MUTED, family=MONO, size=11), title=dict(font=dict(color=INK_2, size=12)))
    pio.templates["lab"] = go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK_2, size=12),
        title=dict(font=dict(color=INK, size=14, family=FONT), x=0.01, xanchor="left"),
        colorway=SERIES, xaxis=axis, yaxis=axis,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="#0b1220", bordercolor=CARD_EDGE, font=dict(family=MONO, color=INK, size=12)),
        scene=dict(xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID, color=MUTED),
                   yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID, color=MUTED),
                   zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID, color=MUTED)),
        coloraxis=dict(colorbar=dict(tickfont=dict(color=MUTED, family=MONO), outlinewidth=0)),
        annotationdefaults=dict(font=dict(color=INK_2, size=11)),
    ))
    pio.templates.default = "lab"


def setup() -> None:
    """Call once at the top of every page."""
    _register_chart_theme()
    # st.html takes a raw style block as is, st.markdown would print it as text
    st.html(_CSS)


def header(eyebrow: str, title: str, subtitle: str, as_of: str | None = None) -> None:
    pill = (f'<div class="lab-pill"><span class="lab-dot"></span>Data as of {html.escape(as_of)}</div>'
            if as_of else "")
    st.markdown(
        f'<div class="lab-hero"><div class="lab-eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="lab-title">{html.escape(title)}</div>'
        f'<div class="lab-sub">{subtitle}</div>{pill}</div>',
        unsafe_allow_html=True,
    )


def kpis(cards: list[dict]) -> None:
    """cards: [{"label", "value", "note" (optional), "color" (optional bar colour)}]"""
    cells = "".join(
        f'<div class="lab-kpi" style="--bar:{c.get("color", ACCENT)}">'
        f'<div class="lab-kpi-label">{html.escape(c["label"])}</div>'
        f'<div class="lab-kpi-value">{html.escape(str(c["value"]))}</div>'
        + (f'<div class="lab-kpi-note">{c["note"]}</div>' if c.get("note") else "")
        + "</div>"
        for c in cards
    )
    st.markdown(f'<div class="lab-kpis" style="grid-template-columns:repeat({len(cards)},minmax(0,1fr))">'
                f"{cells}</div>", unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    st.markdown(f'<div class="lab-section"><span class="lab-section-title">{html.escape(title)}</span>'
                f'<span class="lab-section-note">{html.escape(note)}</span></div>', unsafe_allow_html=True)


def note(text: str) -> None:
    """A highlighted explanation box. Accepts simple HTML such as <b>."""
    st.markdown(f'<div class="lab-note">{text}</div>', unsafe_allow_html=True)
