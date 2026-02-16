# pages/2_Playground.py
import streamlit as st
import numpy as np

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

true_curve = list(st.session_state["true_curve"])
n_dose = len(true_curve)
labels = dose_labels(n_dose)

col1, col2, col3 = st.columns([1.05, 1.10, 1.25], gap="large")

with col1:
    st.subheader("True acute DLT curve")
    st.toggle("Edit true curve", key="edit_true_curve")

    if st.session_state["edit_true_curve"]:
        for i, lab in enumerate(labels):
            st.number_input(
                lab,
                min_value=0.0,
                max_value=0.99,
                step=0.01,
                key=f"true_p_{i}",
                on_change=sync_true_curve_from_widgets,
            )
        true_curve = list(st.session_state["true_curve"])
    else:
        for i, lab in enumerate(labels):
            st.caption(f"{lab}: **{float(true_curve[i]):.2f}**")

    target = float(st.session_state["target"])
    arr = np.array(true_curve, dtype=float)
    true_mtd_idx0 = int(np.argmin(np.abs(arr - target)))
    st.caption(f"True MTD (closest to target) = L{true_mtd_idx0}")

with col2:
    st.subheader("Prior playground")
    st.radio("Skeleton model", ["empiric", "logistic"], key="skeleton_model", horizontal=True)

    st.slider("Prior target", min_value=0.01, max_value=0.50, step=0.01, key="prior_target")
    st.slider("Halfwidth (delta)", min_value=0.01, max_value=0.30, step=0.01, key="delta")

    st.slider(
        "Prior MTD (1-based)",
        min_value=1,
        max_value=n_dose,
        step=1,
        key="prior_mtd",
    )

    st.slider(
        "Logistic intercept",
        min_value=-5.0,
        max_value=5.0,
        step=0.1,
        key="logistic_intercept",
        disabled=(st.session_state["skeleton_model"] != "logistic"),
    )

    run_disabled = bool(st.session_state.get("is_running", False))
    if st.button("Run simulations", use_container_width=True, disabled=run_disabled):
        with st.spinner("Running simulations..."):
            run_simulations()
        st.rerun()

    sk_preview = get_skeleton_from_state(n_dose)
    st.caption("Skeleton: " + ", ".join(f"{v:.3f}" for v in sk_preview))

    err = st.session_state.get("last_error", None)
    if err:
        st.error(err)

with col3:
    st.subheader("CRM knobs + preview")
    st.slider("Prior sigma on theta", min_value=0.10, max_value=3.00, step=0.05, key="prior_sigma_theta")
    st.toggle("Burn-in until first DLT", key="burnin_until_first_dlt")
    st.toggle("Enable EWOC overdose control", key="ewoc_enable")
    st.slider(
        "EWOC alpha",
        min_value=0.01,
        max_value=0.50,
        step=0.01,
        key="ewoc_alpha",
        disabled=(not st.session_state["ewoc_enable"]),
    )

    prior_curve = get_skeleton_from_state(n_dose)
    fig = plot_true_vs_prior(
        true_p=true_curve,
        prior_p=prior_curve,
        target=float(st.session_state["target"]),
        true_mtd_idx0=true_mtd_idx0,
    )
    st.pyplot(fig, clear_figure=True, use_container_width=True)

res = st.session_state.get("results", None)
meta = st.session_state.get("results_meta", None)

if res is not None:
    st.divider()

    def _get(name, fallback=None):
        return res.get(name, fallback) if isinstance(res, dict) else fallback

    mtd6 = np.array(_get("mtd_probs_6p3", [0] * n_dose), dtype=float)
    mtdc = np.array(_get("mtd_probs_crm", [0] * n_dose), dtype=float)
    n6 = np.array(_get("avg_n_per_dose_6p3", [0] * n_dose), dtype=float)
    nc = np.array(_get("avg_n_per_dose_crm", [0] * n_dose), dtype=float)

    p_dlt6 = float(_get("p_dlt_per_patient_6p3", np.nan))
    p_dltc = float(_get("p_dlt_per_patient_crm", np.nan))

    r1, r2, r3 = st.columns([1.25, 1.25, 0.8], gap="large")

    with r1:
        import matplotlib.pyplot as plt

        fig1 = plt.figure(figsize=(4.8, 2.4), dpi=140)
        ax = fig1.add_subplot(111)
        xs = np.arange(n_dose)
        w = 0.38
        ax.bar(xs - w / 2, mtd6, width=w, label="6+3")
        ax.bar(xs + w / 2, mtdc, width=w, label="CRM")
        ax.axvline(true_mtd_idx0, linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"L{i}" for i in xs], fontsize=8)
        ax.set_title("P(select dose as MTD)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=7)
        fig1.tight_layout()
        st.pyplot(fig1, clear_figure=True, use_container_width=True)

    with r2:
        import matplotlib.pyplot as plt

        fig2 = plt.figure(figsize=(4.8, 2.4), dpi=140)
        ax = fig2.add_subplot(111)
        xs = np.arange(n_dose)
        w = 0.38
        ax.bar(xs - w / 2, n6, width=w, label="6+3")
        ax.bar(xs + w / 2, nc, width=w, label="CRM")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"L{i}" for i in xs], fontsize=8)
        ax.set_title("Avg patients treated per dose", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=7)
        fig2.tight_layout()
        st.pyplot(fig2, clear_figure=True, use_container_width=True)

    with r3:
        st.metric("DLT prob per patient (6+3)", f"{p_dlt6:.3f}" if np.isfinite(p_dlt6) else "—")
        st.metric("DLT prob per patient (CRM)", f"{p_dltc:.3f}" if np.isfinite(p_dltc) else "—")
        if meta:
            st.caption(f"n_sims={meta.get('n_sims')} | seed={meta.get('seed')} | True MTD marker=L{true_mtd_idx0}")
else:
    st.caption("Run simulations to populate results.")
