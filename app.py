"""
Options Pricing Lab, the Streamlit app.

Run locally from the project root:
    streamlit run app.py

This file only sets up the page list. Each page lives in app_pages/.
"""

import streamlit as st

st.set_page_config(page_title="Options Pricing Lab", page_icon="📈", layout="wide")

pages = [
    st.Page("app_pages/home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("app_pages/pricer.py", title="Pricer", icon=":material/calculate:"),
    st.Page("app_pages/model_vs_market.py", title="Model vs Market", icon=":material/grid_on:"),
    st.Page("app_pages/vol_surface.py", title="Vol Surface", icon=":material/landscape:"),
    st.Page("app_pages/greeks.py", title="Greeks", icon=":material/tune:"),
    st.Page("app_pages/parity.py", title="Parity Check", icon=":material/balance:"),
    st.Page("app_pages/methodology.py", title="Methodology", icon=":material/menu_book:"),
]

st.navigation(pages).run()
