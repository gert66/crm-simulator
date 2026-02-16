# pages/2_Playground.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from core import (
    init_state,
    run_simulations,
    get_skeleton_from_state,
    plot_true_vs_prior,
    dose_labels,
    sync_true_curve_from_widgets,
)

st.set_page_config(page_title="Playground", layout="wide")
init_state()

# Compact top padding a bit
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
      h1, h2, h3 { margin-top: 0.2rem; }
      [data-testid="stVerticalBlock"] { gap: 0.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Helpers ---
def _true_mtd_idx0(true_curve, target):
    arr = np.asarray(true_curve, dtype=float)
    return int(np.argmin(np.abs(arr - float(target))))

def _plot_bar_compare(title, labels, a, b, a_lab="6+3", b_lab="CRM", figsize=(5.2, 3.4)):
    x = np.arange(len(labels))
    width = 0.38
    fig = plt.figure(figsize=figsize, dpi=140)
    ax = fig.add_subplot(111)
    ax.bar(x - width/2, a, width, label=a_lab)
    ax.bar(x + width/2, b, width, label=b_lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}" for i in range(len(labels))], fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig

# --- State ---
true_curve = list(map(float, st.session_state["true_curve"]))
n_dose = len(true_curve)
labels = dose_labels(n_dose)
true_mtd0 = _true_mtd_idx0(true_curve, st.session_state["target"])

col1, col2, col3 = st.columns([1.05, 1.10, 1.15], gap="large")

# ========== COL 1: True acute DLT curve ==========
with col1:
    st.subheader("True acute DLT curve")

    for i, lab in enumerate(labels):
        st.number_input(
            lab,
            min_value=0.0,
            max_value=0.99,
            step=0.01,
            key=f"true_p_{i}",
            on_change=sync_true_curve_from_widgets,
            help=f"True acute DLT probability at dose level L{i}.",
        )

    # Keep session_state["true_curve"] synced in case user edits multiple boxes
    sync_true_curve_from_widgets()
    true_curve = list(map(float, st.session_state["true_curve"]))
    true_mtd0 = _true_mtd_idx0(true_curve, st.session_state["target"])

    st.caption(f"True MTD (closest to target) = L{true_mtd0}")

# ========== COL 2: Prior playground ==========
with col2:
    st.subheader("Prior playground")

    st.radio(
        "Skeleton model",
        options=["empiric", "logistic"],
        horizontal=True,
        key="skeleton_model",
        help="How the prior skeleton is constructed.\n\nR default: empiric.",
    )

    st.slider(
        "Prior target",
        0.01, 0.40,
        step=0.01,
        key="prior_target",
        help="Target DLT used to build the prior skeleton.\n\nR default: 0.15.",
    )

    st.slider(
        "Halfwidth (delta)",
        0.01, 0.25,
        step=0.01,
        key="delta",
        help="Step size between skeleton dose levels for empiric skeleton.\n\nR default: 0.10.",
        disabled=(st.session_state["skeleton_model"] != "empiric"),
    )

    st.slider(
        "Prior MTD (1-based)",
        1, n_dose,
        step=1,
        key="prior_mtd",
        help="Dose level that the prior believes is closest to the target.\n\nR default: 3.",
    )

    st.slider(
        "Logistic intercept",
        -5.0, 5.0,
        step=0.25,
        key="logistic_intercept",
        help="Intercept for logistic skeleton.\n\nR default: 0.0.",
        disabled=(st.session_state["skeleton_model"] != "logistic"),
    )

    run_clicked = st.button("Run simulations", use_container_width=True)

# ========== COL 3: CRM knobs + preview ==========
with col3:
    st.subheader("CRM knobs + preview")

    st.slider(
        "Prior sigma on theta",
        0.10, 3.00,
        step=0.05,
        key="prior_sigma_theta",
        help="Prior SD for the CRM model parameter (theta).\n\nR default: 1.0.",
    )

    st.toggle(
        "Burn-in until first DLT",
        key="burnin_until_first_dlt",
        help="If enabled: run simple escalation until first DLT, then switch to CRM.\n\nR default: ON.",
    )

    st.toggle(
        "Enable EWOC overdose control",
        key="ewoc_enable",
        help="Enable EWOC overdose control.\n\nR default: OFF.",
    )

    st.slider(
        "EWOC alpha",
        0.01, 0.50,
        step=0.01,
        key="ewoc_alpha",
        disabled=(not bool(st.session_state.get("ewoc_enable", False))),
        help="EWOC threshold (smaller = stricter).\n\nR default: 0.25.",
    )

    prior_curve = get_skeleton_from_state(n_dose)
    fig_prev = plot_true_vs_prior(
        true_p=true_curve,
        prior_p=prior_curve,
        target=float(st.session_state["target"]),
        true_mtd_idx0=true_mtd0,
    )
    st.pyplot(fig_prev, use_container_width=True)

# --- Run simulations (do this AFTER widgets exist) ---
if run_clicked:
    run_simulations()
    # no st.rerun() needed; Streamlit reruns automatically on button click

st.divider()

# ========== Results area ==========
results = st.session_state.get("results", None)
meta = st.session_state.get("results_meta", None)
err = st.session_state.get("last_error", None)

if err:
    st.error(err)

if results is None and err is None:
    st.caption("Run simulations to populate results.")
elif isinstance(results, dict):
    # Defensive extraction (works even if keys vary a bit)
    # Expected-ish:
    #  - p_mtd_6p3, p_mtd_crm : list length n_dose
    #  - avg_n_6p3, avg_n_crm : list length n_dose
    #  - dlt_prob_patient_6p3, dlt_prob_patient_crm : floats
    p_mtd_6p3 = results.get("p_mtd_6p3") or results.get("p_select_mtd_6p3")
    p_mtd_crm = results.get("p_mtd_crm") or results.get("p_select_mtd_crm")

    avg_n_6p3 = results.get("avg_n_6p3") or results.get("avg_patients_per_dose_6p3")
    avg_n_crm = results.get("avg_n_crm") or results.get("avg_patients_per_dose_crm")

    dlt_pp_6p3 = results.get("dlt_prob_per_patient_6p3") or results.get("dlt_pp_6p3")
    dlt_pp_crm = results.get("dlt_prob_per_patient_crm") or results.get("dlt_pp_crm")

    # Fallbacks if missing
    if p_mtd_6p3 is None or p_mtd_crm is None or avg_n_6p3 is None or avg_n_crm is None:
        st.warning("Results dict did not contain the expected keys. Showing available keys:")
        st.code(", ".join(sorted(results.keys())))
    else:
        # Ensure arrays
        p_mtd_6p3 = np.asarray(p_mtd_6p3, dtype=float)
        p_mtd_crm = np.asarray(p_mtd_crm, dtype=float)
        avg_n_6p3 = np.asarray(avg_n_6p3, dtype=float)
        avg_n_crm = np.asarray(avg_n_crm, dtype=float)

        r1, r2, r3 = st.columns([1.25, 1.25, 0.8], gap="large")

        with r1:
            fig1 = _plot_bar_compare(
                "P(select dose as MTD)",
                labels,
                p_mtd_6p3,
                p_mtd_crm,
                figsize=(5.4, 3.6),
            )
            st.pyplot(fig1, use_container_width=True)

        with r2:
            fig2 = _plot_bar_compare(
                "Avg patients treated per dose",
                labels,
                avg_n_6p3,
                avg_n_crm,
                figsize=(5.4, 3.6),
            )
            st.pyplot(fig2, use_container_width=True)

        with r3:
            if dlt_pp_6p3 is not None:
                st.caption("DLT prob per patient (6+3)")
                st.markdown(f"## {float(dlt_pp_6p3):.3f}")
            if dlt_pp_crm is not None:
                st.caption("DLT prob per patient (CRM)")
                st.markdown(f"## {float(dlt_pp_crm):.3f}")

            if meta:
                st.caption(f"n_sims={meta.get('n_sims')} | seed={meta.get('seed')} | True MTD marker=L{true_mtd0}")
else:
    st.warning("Unexpected results type. Expected dict.")
