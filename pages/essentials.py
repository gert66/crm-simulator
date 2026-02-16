import streamlit as st
from ui_state import ensure_state, reset_all, DOSE_LABELS

ensure_state()

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Essentials")

c1, c2, c3, c4 = st.columns([1.0, 1.2, 1.0, 1.0], gap="small")

with c1:
    st.number_input(
        "Study target (acute)",
        min_value=0.01, max_value=0.50, step=0.01,
        key="target",
        help="Target acute DLT probability used to define the MTD (closest-to-target).",
    )

with c2:
    st.selectbox(
        "Start dose",
        options=list(range(len(DOSE_LABELS))),
        format_func=lambda i: DOSE_LABELS[i],
        key="start_dose",
        help="Dose level where escalation begins.",
    )

with c3:
    st.number_input(
        "Max sample size (CRM)",
        min_value=3, max_value=120, step=3,
        key="crm_max_n",
        help="Maximum total number of patients for CRM simulations.",
    )

with c4:
    st.number_input(
        "CRM cohort size",
        min_value=1, max_value=12, step=1,
        key="crm_cohort",
        help="Number of patients treated per CRM cohort.",
    )

c5, c6, c7, c8 = st.columns([1.0, 1.0, 1.0, 1.0], gap="small")

with c5:
    st.number_input(
        "Max sample size (6+3)",
        min_value=3, max_value=120, step=3,
        key="sixplus3_max_n",
        help="Maximum total number of patients for the 6+3 design. Default is 27.",
    )

with c6:
    st.number_input(
        "Number of simulated trials",
        min_value=50, max_value=5000, step=50,
        key="n_sims",
        help="How many independent simulated trials to run.",
    )

with c7:
    st.number_input(
        "Random seed",
        min_value=0, max_value=999999, step=1,
        key="seed",
        help="Reproducibility seed for the simulator.",
    )

with c8:
    st.button(
        "Reset to defaults",
        on_click=reset_all,
        help="Restore all parameters to built-in defaults.",
        use_container_width=True,
    )

st.info("Go to **Playground** to tune the true curve, priors, CRM knobs, and run simulations.")
