# pages/1_Essentials.py
from __future__ import annotations

import streamlit as st
from core import DEFAULTS, init_state, reset_to_defaults

st.set_page_config(page_title="Essentials", layout="wide")

init_state(st, DEFAULTS)

# Compact padding
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Put reset BEFORE widgets so it never triggers "set after widget" issues.
c1, c2, _ = st.columns([1, 1, 3])
with c1:
    if st.button("Reset to defaults", use_container_width=True):
        reset_to_defaults(st, DEFAULTS)
with c2:
    st.caption("Essentials for the simulation run.")

st.subheader("Essentials")

colA, colB, colC, colD = st.columns([1.2, 1.2, 1.2, 1.2])

with colA:
    st.number_input(
        "Study target (acute)",
        min_value=0.01,
        max_value=0.40,
        step=0.01,
        key="target",
        help="Target acute DLT probability used for 'true MTD' and CRM dose selection.",
    )

    st.selectbox(
        "Start dose",
        options=list(range(len(st.session_state["dose_labels"]))),
        format_func=lambda i: st.session_state["dose_labels"][i].replace("\n", " "),
        key="start_dose_idx",
        help="Starting dose level for both 6+3 and CRM.",
    )

with colB:
    st.number_input(
        "Max sample size (CRM)",
        min_value=6,
        max_value=120,
        step=1,
        key="crm_max_n",
        help="Maximum number of patients enrolled under CRM.",
    )
    st.number_input(
        "CRM cohort size",
        min_value=1,
        max_value=12,
        step=1,
        key="crm_cohort",
        help="Patients per CRM cohort.",
    )

with colC:
    st.number_input(
        "Max sample size (6+3)",
        min_value=9,
        max_value=120,
        step=1,
        key="sixplus3_max_n",
        help="Maximum number of patients enrolled under the 6+3 design.",
    )
    st.number_input(
        "Number of simulated trials",
        min_value=50,
        max_value=5000,
        step=50,
        key="n_sims",
        help="Number of Monte Carlo trials.",
    )

with colD:
    st.number_input(
        "Random seed",
        min_value=0,
        max_value=10_000_000,
        step=1,
        key="seed",
        help="Seed for reproducible simulation runs.",
    )
    st.caption("Tip: run simulations from the Playground page.")

st.divider()

with st.expander("Current settings (from code)", expanded=False):
    keys = [
        "target", "start_dose_idx", "crm_max_n", "crm_cohort", "sixplus3_max_n",
        "n_sims", "seed",
        "prior_sigma_theta", "burn_in_until_first_dlt", "enable_ewoc", "ewoc_alpha",
        "skeleton_model", "prior_target", "prior_halfwidth", "prior_mtd_nu", "logistic_intercept",
        "true_p0", "true_p1", "true_p2", "true_p3", "true_p4",
    ]
    st.json({k: st.session_state[k] for k in keys})
