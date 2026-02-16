# pages/1_Essentials.py
import streamlit as st

from core import init_state, reset_to_defaults

st.set_page_config(page_title="Essentials", layout="wide")
init_state()

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
      h1, h2, h3 { margin-top: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("Essentials")

c1, c2, c3 = st.columns([1.2, 1.2, 1.0], gap="large")

with c1:
    st.subheader("Study")
    st.number_input(
        "Target DLT rate",
        min_value=0.01,
        max_value=0.99,
        step=0.01,
        key="target",
        help="Target probability of DLT used by the design.\nR default: 0.15",
    )
    st.number_input(
        "Start dose level (1-based)",
        min_value=1,
        max_value=99,
        step=1,
        key="start_dose_level",
        help="Dose level where enrollment starts (1-based).\nR default: 1",
    )

with c2:
    st.subheader("Simulation")
    st.number_input(
        "Number of simulated trials",
        min_value=10,
        max_value=20000,
        step=10,
        key="n_sims",
        help="Number of simulated trials.\nR default (NREP): 1000",
    )
    st.number_input(
        "Random seed",
        min_value=0,
        max_value=10_000_000,
        step=1,
        key="seed",
        help="Random seed for reproducibility.\nR default: 123",
    )

with c3:
    st.subheader("6+3 design")
    st.number_input(
        "Cohort size",
        min_value=3,
        max_value=6,
        step=1,
        key="cohort_size_6p3",
        help="Patients per cohort.\nR default (CO): 3",
    )
    st.number_input(
        "Maximum sample size (6+3)",
        min_value=6,
        max_value=60,
        step=3,
        key="max_n_6p3",
        help="Maximum number of patients.\nR default (N.patient): 27",
    )

st.divider()

colr1, colr2 = st.columns([1, 3])
with colr1:
    if st.button("Reset to defaults", use_container_width=True, help="Reset all parameters to defaults and clear results."):
        reset_to_defaults()

with colr2:
    st.caption("Reset clears results too.")
