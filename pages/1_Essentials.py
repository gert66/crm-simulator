# pages/1_Essentials.py
import streamlit as st

from core import init_state, reset_to_defaults

st.set_page_config(page_title="Essentials", layout="wide")
init_state()

st.header("Essentials")

c1, c2, c3 = st.columns([1.1, 1.1, 1.0], gap="large")

with c1:
    st.subheader("Study")
    st.number_input("Target DLT rate", min_value=0.01, max_value=0.99, step=0.01, key="target")
    st.number_input("Start dose level (1-based)", min_value=1, max_value=99, step=1, key="start_dose_level")

with c2:
    st.subheader("Simulation")
    st.number_input("Number of simulated trials", min_value=10, max_value=20000, step=10, key="n_sims")
    st.number_input("Random seed", min_value=0, max_value=10_000_000, step=1, key="seed")

with c3:
    st.subheader("6+3 design")
    st.number_input("Cohort size", min_value=3, max_value=6, step=1, key="cohort_size_6p3")
    st.number_input("Maximum sample size (6+3)", min_value=6, max_value=60, step=3, key="max_n_6p3")

st.divider()

r1, r2 = st.columns([0.9, 2.1], gap="large")

with r1:
    if st.button("Reset to defaults", use_container_width=True):
        reset_to_defaults()
        st.rerun()

with r2:
    st.caption("Reset restores defaults from core.py and clears results.")
