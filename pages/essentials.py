import streamlit as st
from core import init_state, DOSE_LABELS

st.set_page_config(page_title="Essentials", layout="wide")

init_state()

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1.0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.subheader("Essentials")

colA, colB, colC = st.columns([1.1, 1.1, 1.0], gap="large")

with colA:
    st.number_input(
        "Study target (DLT)",
        min_value=0.01, max_value=0.60, step=0.01,
        key="target",
    )
    st.selectbox(
        "Start dose level",
        options=list(range(len(DOSE_LABELS))),
        format_func=lambda i: f"{i}  ({DOSE_LABELS[i].replace(chr(10), ' ')})",
        key="start_dose",
    )
    st.number_input(
        "Cohort size",
        min_value=1, max_value=12, step=1,
        key="cohort_size",
    )

with colB:
    st.number_input(
        "Max sample size (6+3)",
        min_value=6, max_value=120, step=3,
        key="max_n_63",
        help="Default = 27",
    )
    st.number_input(
        "Number of simulated trials",
        min_value=50, max_value=20000, step=50,
        key="n_sims",
    )
    st.number_input(
        "Random seed",
        min_value=0, max_value=10**9, step=1,
        key="seed",
    )

with colC:
    st.number_input(
        "Guardrail: max final MTD level",
        min_value=0, max_value=len(DOSE_LABELS) - 1, step=1,
        key="guardrail_max_mtd",
    )
    st.toggle(
        "Allow dose skipping (CRM step > 1)",
        key="allow_skip",
    )

st.caption("Go to Playground to tune priors and run simulations.")
