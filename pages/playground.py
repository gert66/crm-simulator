# pages/2_Playground.py
from __future__ import annotations

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from core import (
    DEFAULTS,
    init_state,
    get_true_curve_from_state,
    get_skeleton_from_state,
    run_simulations,
)

st.set_page_config(page_title="Playground", layout="wide")

init_state(st, DEFAULTS)

# Compact padding + slightly smaller headers
st.markdown(
    """
    <style>
    .block-container { padding-top: 0.8rem; padding-bottom: 0.6rem; }
    h2 { margin-top: 0.2rem; margin-bottom: 0.4rem; }
    h3 { margin-top: 0.2rem; margin-bottom: 0.35rem; font-size: 1.08rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

dose_labels = st.session_state["dose_labels"]
x = np.arange(len(dose_labels), dtype=int)


def plot_true_vs_skeleton(true_curve: np.ndarray, skeleton: np.ndarray, target: float, true_mtd_idx: int):
    fig = plt.figure(figsize=(5.3, 2.6), dpi=140)
    ax = fig.add_subplot(111)
    ax.plot(x, true_curve, marker="o", label="True P(DLT)")
    ax.plot(x, skeleton, marker="o", label="Prior (skeleton)")
    ax.axhline(float(target), linewidth=1)
    ax.axvline(int(true_mtd_idx), linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(dose_labels, fontsize=8)
    ax.set_ylabel("Probability")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def bar_prob_mtd(prob6: list[float], probc: list[float], true_mtd_idx: int):
    fig = plt.figure(figsize=(5.2, 2.6), dpi=140)
    ax = fig.add_subplot(111)
    w = 0.38
    ax.bar(x - w / 2, prob6, width=w, label="6+3")
    ax.bar(x + w / 2, probc, width=w, label="CRM")
    ax.axvline(int(true_mtd_idx), linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("\n", " ") for d in dose_labels], fontsize=8)
    ax.set_title("P(select dose as MTD)", fontsize=10)
    ax.set_ylabel("Probability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def bar_avg_n(avg6: list[float], avgc: list[float]):
    fig = plt.figure(figsize=(5.2, 2.6), dpi=140)
    ax = fig.add_subplot(111)
    w = 0.38
    ax.bar(x - w / 2, avg6, width=w, label="6+3")
    ax.bar(x + w / 2, avgc, width=w, label="CRM")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("\n", " ") for d in dose_labels], fontsize=8)
    ax.set_title("Avg patients treated per dose", fontsize=10)
    ax.set_ylabel("Patients")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# -------------------------
# Playground controls (FORM)
# -------------------------
col1, col2, col3 = st.columns([1.05, 1.05, 1.25], gap="large")

with st.form("playground_form", border=False):
    with col1:
        st.subheader("True acute DLT curve")

        st.toggle(
            "Edit true curve",
            key="edit_true_curve",
            help="Turn on editing of the true acute DLT probabilities used to generate outcomes.",
        )

        disabled = not bool(st.session_state["edit_true_curve"])

        st.number_input("L0 5×4 Gy", min_value=0.00, max_value=0.95, step=0.01, key="true_p0", disabled=disabled)
        st.number_input("L1 5×5 Gy", min_value=0.00, max_value=0.95, step=0.01, key="true_p1", disabled=disabled)
        st.number_input("L2 5×6 Gy", min_value=0.00, max_value=0.95, step=0.01, key="true_p2", disabled=disabled)
        st.number_input("L3 5×7 Gy", min_value=0.00, max_value=0.95, step=0.01, key="true_p3", disabled=disabled)
        st.number_input("L4 5×8 Gy", min_value=0.00, max_value=0.95, step=0.01, key="true_p4", disabled=disabled)

        true_curve_preview = get_true_curve_from_state(st)
        true_mtd_idx_preview = int(np.argmin(np.abs(true_curve_preview - float(st.session_state["target"]))))
        st.caption(f"True MTD (closest to target) = {dose_labels[true_mtd_idx_preview].splitlines()[0]}")

    with col2:
        st.subheader("Prior playground")

        st.radio(
            "Skeleton model",
            options=["empiric", "logistic"],
            key="skeleton_model",
            horizontal=True,
            help="Empiric: monotone skeleton anchored at prior target. Logistic: alternative monotone skeleton.",
        )

        st.slider(
            "Prior target",
            min_value=0.05,
            max_value=0.35,
            step=0.01,
            key="prior_target",
            help="Target toxicity level used to anchor the prior skeleton.",
        )
        st.slider(
            "Halfwidth (delta)",
            min_value=0.01,
            max_value=0.25,
            step=0.01,
            key="prior_halfwidth",
            help="Controls how steep the empiric skeleton is around the prior target.",
        )
        st.slider(
            "Prior MTD (nu, 1-based)",
            min_value=1,
            max_value=5,
            step=1,
            key="prior_mtd_nu",
            help="Dose index (1..5) that the skeleton treats as the prior MTD location.",
        )
        st.slider(
            "Logistic intercept",
            min_value=-5.0,
            max_value=5.0,
            step=0.1,
            key="logistic_intercept",
            help="Only used when Skeleton model is logistic.",
        )

        # Put the RUN button under the middle column
        run_clicked = st.button("Run simulations", use_container_width=True)
        if run_clicked:
                run_simulations()


        # compact skeleton preview line
        sk_preview = get_skeleton_from_state(st)
        st.caption("Skeleton: " + ", ".join(f"{v:.3f}" for v in sk_preview))

    with col3:
        st.subheader("CRM knobs + preview")

        st.slider(
            "Prior sigma on theta",
            min_value=0.10,
            max_value=3.00,
            step=0.05,
            key="prior_sigma_theta",
            help="Prior SD for theta in the CRM one-parameter power model.",
        )

        st.toggle(
            "Burn-in until first DLT",
            key="burn_in_until_first_dlt",
            help="If on: CRM treats at the start dose until the first DLT is observed.",
        )

        st.toggle(
            "Enable EWOC overdose control",
            key="enable_ewoc",
            help="If on: restrict doses where posterior P(p > target) is above EWOC alpha.",
        )

        st.slider(
            "EWOC alpha",
            min_value=0.05,
            max_value=0.50,
            step=0.01,
            key="ewoc_alpha",
            disabled=not bool(st.session_state["enable_ewoc"]),
            help="Upper bound for overdose probability under the posterior.",
        )

        # Plot preview (true vs skeleton)
        true_curve = get_true_curve_from_state(st)
        skeleton = get_skeleton_from_state(st)
        target = float(st.session_state["target"])
        true_mtd_idx = int(np.argmin(np.abs(true_curve - target)))
        st.pyplot(plot_true_vs_skeleton(true_curve, skeleton, target, true_mtd_idx), use_container_width=True)

# -------------------------
# Run simulations (only when submitted)
# -------------------------
if run_clicked:
    true_curve = get_true_curve_from_state(st)
    skeleton = get_skeleton_from_state(st)

    res = run_simulations(
        true_curve=true_curve,
        skeleton=skeleton,
        dose_labels=dose_labels,
        target=float(st.session_state["target"]),
        start_idx=int(st.session_state["start_dose_idx"]),
        n_sims=int(st.session_state["n_sims"]),
        seed=int(st.session_state["seed"]),
        six_max_n=int(st.session_state["sixplus3_max_n"]),
        crm_max_n=int(st.session_state["crm_max_n"]),
        crm_cohort=int(st.session_state["crm_cohort"]),
        prior_sigma_theta=float(st.session_state["prior_sigma_theta"]),
        burn_in_until_first_dlt=bool(st.session_state["burn_in_until_first_dlt"]),
        enable_ewoc=bool(st.session_state["enable_ewoc"]),
        ewoc_alpha=float(st.session_state["ewoc_alpha"]),
    )
    st.session_state["last_results"] = res

# -------------------------
# Results (compact, no big title)
# -------------------------
res = st.session_state.get("last_results", None)

if res is None:
    st.caption("Run simulations to populate results.")
else:
    # very small vertical footprint
    r1, r2, r3 = st.columns([1.0, 1.0, 0.8], gap="large")

    with r3:
        st.metric("DLT prob per patient (6+3)", f"{res.dlt_prob_per_patient_six:.3f}")
        st.metric("DLT prob per patient (CRM)", f"{res.dlt_prob_per_patient_crm:.3f}")
        st.caption(f"n_sims={res.n_sims} | seed={res.seed} | True MTD marker={dose_labels[res.true_mtd_idx].splitlines()[0]}")

    with r1:
        st.pyplot(bar_prob_mtd(res.prob_mtd_six, res.prob_mtd_crm, res.true_mtd_idx), use_container_width=True)

    with r2:
        st.pyplot(bar_avg_n(res.avg_n_six, res.avg_n_crm), use_container_width=True)
