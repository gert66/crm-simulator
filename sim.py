# sim.py
import streamlit as st

from core import init_state

st.set_page_config(page_title="CRM Simulator", layout="wide")
init_state()

pg = st.navigation(
    [
        st.Page("pages/1_Essentials.py", title="Essentials"),
        st.Page("pages/2_Playground.py", title="Playground"),
    ]
)
pg.run()
