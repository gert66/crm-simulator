# pages/1_Essentials.py
import streamlit as st

from core import init_state, reset_to_defaults, DEFAULTS

st.set_page_config(page_title="Essentials", layout="wide")
init_state()

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 1.2rem; }
      h1, h2, h3 { margin-top: 0rem; padding-top: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Essentials")

c1, c2, c3 = st.columns([1.2, 1.0, 1.2], gap="large")

with c1:
    st.subheader("Study")

    st.number_input(
        "Target DLT rate",
        min_value=0.01,
        max_value=0.50,
        step=0.01,
        key="target",
        help=f"Target toxicity rate used to define the MTD.\nR default: 0.15\nApp default: {DEFAULTS.target}",
    )

    st.number_input(
        "Start dose level (1-based)",
        min_value=1,
        max_value=50,
        step=1,
        key="start_dose_level",
        help=f"Starting dose level for both designs.\nR default: 1\nApp default: {DEFAULTS.start_dose_level}",
    )

    st.number_input(
        "Already treated at start dose (0 DLT)",
        min_value=0,
        max_value=999,
        step=1,
        key="n_prior_start_no_dlt",
        help=(
            "Number of patients already treated at the start dose with 0 DLT.\n"
            "This is added as prior observed data at the starting dose for CRM.\n"
            f"R default: 0\nApp default: {DEFAULTS.n_prior_start_no_dlt}"
        ),
    )

with c2:
    st.subheader("Simulation")

    st.number_input(
        "Number of simulated trials",
        min_value=1,
        max_value=50000,
        step=50,
        key="n_sims",
        help=f"How many simulated trials to run.\nR default: 500\nApp default: {DEFAULTS.n_sims}",
    )

    st.number_input(
        "Random seed",
        min_value=0,
        max_value=10**9,
        step=1,
        key="seed",
        help=f"Seed for reproducibility.\nR default: 123\nApp default: {DEFAULTS.seed}",
    )

with c3:
    st.subheader("Sample size")

    st.number_input(
        "Maximum sample size (6+3)",
        min_value=1,
        max_value=999,
        step=3,
        key="max_n_6p3",
        help=f"Maximum number of patients for the 6+3 design.\nR default: 27\nApp default: {DEFAULTS.max_n_6p3}",
    )

    st.number_input(
        "Maximum sample size (CRM)",
        min_value=1,
        max_value=999,
        step=3,
        key="max_n_crm",
        help=f"Maximum number of patients for the CRM design.\nR default: 27\nApp default: {DEFAULTS.max_n_crm}",
    )

    st.number_input(
        "Cohort size",
        min_value=1,
        max_value=12,
        step=1,
        key="cohort_size",
        help=f"Cohort size used in both designs.\nR default: 3\nApp default: {DEFAULTS.cohort_size}",
    )

st.divider()

if st.button("Reset to defaults", use_container_width=True):
    reset_to_defaults()
