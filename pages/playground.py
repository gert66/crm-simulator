import streamlit as st
import numpy as np

from core import (
    init_state,
    DOSE_LABELS,
    N_DOSES,
    get_true_probs,
    set_true_prob,
    compute_skeleton_from_state,
    run_simulations,
    fig_true_vs_prior,
    fig_select_prob,
    fig_avg_n,
    true_mtd_idx_from_probs,
)

st.set_page_config(page_title="Playground", layout="wide")
init_state()

st.markdown(
    """
    <style>
    .block-container { padding-top: 0.9rem; padding-bottom: 0.9rem; }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) { padding-top: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Geen grote titel: scheelt hoogte.
# st.subheader("Playground")  # bewust weg

col1, col2, col3 = st.columns([1.05, 1.10, 1.15], gap="large")

true_p = get_true_probs()
target = float(st.session_state["target"])
true_mtd_idx = true_mtd_idx_from_probs(true_p, target)

with col1:
    st.markdown("#### True acute DLT curve")
    st.toggle("Edit true curve", key="edit_true_curve")

    editable = bool(st.session_state["edit_true_curve"])
    for i in range(N_DOSES):
        key = f"true_p_{i}"
        # init widget state once
        if key not in st.session_state:
            st.session_state[key] = float(true_p[i])

        val = st.number_input(
            DOSE_LABELS[i].replace("\n", " "),
            min_value=0.0,
            max_value=0.95,
            step=0.01,
            key=key,
            disabled=(not editable),
        )
        if editable:
            set_true_prob(i, float(val))

    # recompute after potential edits
    true_p = get_true_probs()
    true_mtd_idx = true_mtd_idx_from_probs(true_p, target)
    st.caption(f"True MTD (closest to target) = L{true_mtd_idx}")

with col2:
    st.markdown("#### Prior playground")

    st.radio(
        "Skeleton model",
        options=["empiric", "logistic"],
        horizontal=True,
        key="skeleton_model",
    )

    st.slider(
        "Prior target",
        min_value=0.05, max_value=0.40, step=0.01,
        key="prior_target",
    )
    st.slider(
        "Halfwidth (delta)",
        min_value=0.01, max_value=0.25, step=0.01,
        key="delta",
    )
    st.slider(
        "Prior MTD (nu, 1-based)",
        min_value=1, max_value=N_DOSES,
        step=1,
        key="prior_mtd",
        help="Shown as 1..K, internally stored as 0-based.",
    )

    # convert to 0-based consistently
    st.session_state["prior_mtd"] = int(st.session_state["prior_mtd"]) - 1

    st.slider(
        "Logistic intercept",
        min_value=-5.0, max_value=5.0, step=0.1,
        key="logistic_intercept",
        help="Only used when Skeleton model is logistic.",
    )

    # Run button onder middelste kolom
    if st.button("Run simulations", use_container_width=True):
        run_simulations()

    sk = compute_skeleton_from_state()
    st.caption("Skeleton: " + ", ".join([f"{v:.3f}" for v in sk]))

with col3:
    st.markdown("#### CRM knobs + preview")

    st.slider(
        "Prior sigma on theta",
        min_value=0.10, max_value=3.00, step=0.05,
        key="prior_sigma_theta",
    )
    st.toggle("Burn-in until first DLT", key="burn_in_until_first_dlt")
    st.toggle("Enable EWOC overdose control", key="ewoc_enabled")
    st.slider(
        "EWOC alpha",
        min_value=0.05, max_value=0.50, step=0.01,
        key="ewoc_alpha",
        disabled=(not bool(st.session_state["ewoc_enabled"])),
    )

    sk = compute_skeleton_from_state()
    fig = fig_true_vs_prior(true_p=true_p, skeleton=sk, target=target, true_mtd_idx=true_mtd_idx)
    st.pyplot(fig, clear_figure=True)

# ----------------------------
# Results (compact)
# ----------------------------

res = st.session_state.get("results", None)
if res is not None:
    # Geen grote "Results" titel, scheelt hoogte
    r1, r2, r3 = st.columns([1.0, 1.0, 0.55], gap="large")

    with r1:
        st.pyplot(fig_select_prob(res.p_select_63, res.p_select_crm, true_mtd_idx), clear_figure=True)

    with r2:
        st.pyplot(fig_avg_n(res.avg_n_per_dose_63, res.avg_n_per_dose_crm), clear_figure=True)

    with r3:
        st.metric("DLT prob per patient (6+3)", f"{res.dlt_prob_per_patient_63:.3f}")
        st.metric("DLT prob per patient (CRM)", f"{res.dlt_prob_per_patient_crm:.3f}")
        st.caption(f"n_sims={st.session_state['n_sims']} | seed={st.session_state['seed']} | True MTD marker=L{true_mtd_idx}")
else:
    st.caption("Run simulations to populate results.")
