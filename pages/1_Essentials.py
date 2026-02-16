# pages/1_Essentials.py
import streamlit as st

from core import init_state, reset_to_defaults

st.set_page_config(page_title="Essentials", layout="wide")
init_state()

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.0rem; padding-bottom: 1.0rem; }
      h1, h2, h3 { margin-top: 0rem; padding-top: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Essentials")

c1, c2, c3 = st.columns([1.2, 1.1, 1.3], gap="large")

with c1:
    st.subheader("Study")

    st.number_input(
        "Target DLT rate",
        min_value=0.01,
        max_value=0.50,
        step=0.01,
        key="target",
        help="Target probability of DLT.\nR default: 0.15",
    )

    st.number_input(
        "Start dose level (1-based)",
        min_value=1,
        max_value=20,
        step=1,
        key="start_dose_level",
        help="Dose level where escalation starts.\nR default: 1",
    )

    st.number_input(
        "Already treated at start dose (0 DLT)",
        min_value=0,
        max_value=500,
        step=1,
        key="n_prior_start_no_dlt",
        help=(
            "Number of patients already treated at the start dose level with zero DLTs.\n"
            "Used as prior observed data.\nR default: 0"
        ),
    )

with c2:
    st.subheader("Simulation")

    st.number_input(
        "Number of simulated trials",
        min_value=1,
        max_value=20000,
        step=10,
        key="n_sims",
        help="How many simulated trials to run.\nR default: 500",
    )

    st.number_input(
        "Random seed",
        min_value=0,
        max_value=10_000_000,
        step=1,
        key="seed",
        help="Seed for reproducibility.\nR default: 123",
    )

with c3:
    st.subheader("Sample size")

    st.caption("Set the maximum number of patients per design.")

    a, b = st.columns(2, gap="large")
    with a:
        st.number_input(
            "Maximum sample size (6+3)",
            min_value=1,
            max_value=500,
            step=1,
            key="max_n_6p3",
            help="Maximum total patients for 6+3 simulations.\nR default: 27",
        )
        st.number_input(
            "Cohort size (6+3)",
            min_value=1,
            max_value=12,
            step=1,
            key="cohort_size_6p3",
            help="Cohort size for 6+3 design.\nR default: 3",
        )

    with b:
        st.number_input(
            "Maximum sample size (CRM)",
            min_value=1,
            max_value=500,
            step=1,
            key="max_n_crm",
            help="Maximum total patients for CRM simulations.\nR default: 27",
        )

st.divider()

st.button("Reset to defaults", use_container_width=True, on_click=reset_to_defaults)
