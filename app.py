"""Langt mei Hei? – App-Einstiegspunkt mit benannter Mehrseiten-Navigation.

Start:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

page = st.navigation(
    [
        st.Page("app_pages/0_Langt_mei_Hei.py", title="Langt mei Hei?", icon="🌾", default=True),
        st.Page("app_pages/1_Futterqualität.py", title="Futterqualität", icon="🌱"),
        st.Page("app_pages/2_Tiergesundheit.py", title="Tiergesundheit", icon="🐄"),
        st.Page("app_pages/3_Klimatrend.py", title="Klimatrend", icon="📈"),
    ]
)
page.run()
