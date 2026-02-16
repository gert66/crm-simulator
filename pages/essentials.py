import streamlit as st
from core import (
    init_state, reset_to_defaults,
    K_TARGET, K_START_DOSE, K_MAX_N_SERUM, K_COHORT, K_N_SIMS, K_SEED, K_GAUSS_POINTS,
    DOSE_LABELS,
)

# Compact spacing
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 1.2rem; }
      h1, h2, h3 { margin-bottom: 0.35rem; }
      .stNumberInput, .stSelectbox, .stSlider { margin-bottom: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()

# Keep title small and clean
st.header("Essentials")

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    st.subheader("Study")
    st.slider(
        "Target toxicity",
        min_value=0.05,
        max_value=0.50,
        step=0.01,
        key=K_TARGET,
        help="Target DLT probability (e.g., 0.15).",
    )
    st.selectbox(
        "Start dose level",
        options=list(range(len(DOSE_LABELS))),
        format_func=lambda i: DOSE_LABELS[i].replace("\n", " "),
        key=K_START_DOSE,
        help="Dose level where escalation starts (0-based).",
    )

with c2:
    st.subheader("6+3 design")
    st.number_input(
        "Max sample size (serum)",
        min_value=6,
        max_value=60,
        step=3,
        key=K_MAX_N_SERUM,
        help="Hard cap for the 6+3 design. You wanted default = 27.",
    )
    st.number_input(
        "Cohort size",
        min_value=1,
        max_value=12,
        step=1,
        key=K_COHORT,
        help="Patients per cohort.",
    )

with c3:
    st.subheader("Simulation")
    st.number_input(
        "Number of simulated trials",
        min_value=50,
        max_value=20000,
        step=50,
        key=K_N_SIMS,
        help="Monte Carlo repetitions.",
    )
    st.number_input(
        "Random seed",
        min_value=0,
        max_value=10_000_000,
        step=1,
        key=K_SEED,
        help="Seed for reproducibility.",
    )
    st.number_input(
        "Gaussian quadrature points",
        min_value=5,
        max_value=50,
        step=1,
        key=K_GAUSS_POINTS,
        help="Numerical integration points (used by CRM parts if applicable).",
    )

st.divider()

cA, cB = st.columns([1, 2])
with cA:
    if st.button("Reset essentials to defaults", use_container_width=True):
        reset_to_defaults(scope="essentials")
        st.rerun()

with cB:
    st.caption("Tip: Essentials is deliberately stable. Playground controls live on the Playground tab.")
