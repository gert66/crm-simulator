import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from core import (
    init_state, reset_to_defaults, run_simulations,
    DOSE_LABELS, N_DOSES,
    K_TRUE_P, K_EDIT_TRUE,
    K_SKEL_MODEL, K_PRIOR_TARGET, K_PRIOR_DELTA, K_PRIOR_MTD_NU, K_LOGISTIC_INTERCEPT,
    K_PRIOR_SIGMA_THETA, K_BURNIN_FIRST_DLT, K_ENABLE_EWOC, K_EWOC_ALPHA,
    K_TARGET, K_RESULTS,
    get_true_curve, get_skeleton_preview_from_state, true_mtd_index_from_curve,
)

# Compact spacing (big win for your “no scroll” goal)
st.markdown(
    """
    <style>
      .block-container { padding-top: 0.8rem; padding-bottom: 0.9rem; }
      h1, h2, h3 { margin-top: 0.2rem; margin-bottom: 0.25rem; }
      .stSlider, .stNumberInput, .stToggle, .stRadio { margin-bottom: 0.15rem; }
      div[data-testid="stMetric"] { padding: 0.15rem 0.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()

# No big "Playground" title (you asked to remove it)
# st.header("Playground")  # intentionally omitted

# --- TOP: three columns
col1, col2, col3 = st.columns([1.05, 1.15, 1.30], gap="large")

with col1:
    st.subheader("True acute DLT curve")
    st.toggle(
        "Edit true curve",
        key=K_EDIT_TRUE,
        help="Unlocks the per-dose fields below.",
    )

    editable = bool(st.session_state[K_EDIT_TRUE])
    current = list(st.session_state[K_TRUE_P])

    # Keep these compact. Use number_input for precision and small vertical footprint.
    new_vals = []
    for i, lab in enumerate(DOSE_LABELS):
        new_vals.append(
            st.number_input(
                lab.replace("\n", " "),
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=float(current[i]) if i < len(current) else 0.0,
                disabled=(not editable),
                key=f"true_p_{i}",
                help="True per-patient DLT probability at this dose level.",
            )
        )

    # Only write back when editable is ON (prevents surprise overwrites)
    if editable:
        st.session_state[K_TRUE_P] = [float(x) for x in new_vals]

    true_curve = get_true_curve()
    target = float(st.session_state[K_TARGET])
    true_mtd_idx = true_mtd_index_from_curve(true_curve, target)
    st.caption(f"True MTD (closest to target) = L{true_mtd_idx}")

with col2:
    st.subheader("Prior playground")
    st.radio(
        "Skeleton model",
        options=["empiric", "logistic"],
        horizontal=True,
        key=K_SKEL_MODEL,
        help="Controls how the prior skeleton preview is constructed.",
        label_visibility="visible",
    )

    st.slider(
        "Prior target",
        min_value=0.05,
        max_value=0.50,
        step=0.01,
        key=K_PRIOR_TARGET,
        help="Target used to build the prior skeleton preview.",
    )

    st.slider(
        "Halfwidth (delta)",
        min_value=0.01,
        max_value=0.30,
        step=0.01,
        key=K_PRIOR_DELTA,
        help="Only used for the empiric skeleton preview.",
        disabled=(st.session_state[K_SKEL_MODEL] == "logistic"),
    )

    st.slider(
        "Prior MTD (nu, 1-based)",
        min_value=1,
        max_value=N_DOSES,
        step=1,
        key=K_PRIOR_MTD_NU,
        help="Dose index where the prior is anchored (1-based).",
    )

    st.slider(
        "Logistic intercept",
        min_value=-5.0,
        max_value=5.0,
        step=0.1,
        key=K_LOGISTIC_INTERCEPT,
        help="Only used for the logistic skeleton preview.",
        disabled=(st.session_state[K_SKEL_MODEL] != "logistic"),
    )

    # Run button goes under middle column (your request)
    run_clicked = st.button("Run simulations", use_container_width=True)
    if run_clicked:
        # Important: this must not reset any sliders. core.init_state() only sets missing keys.
        run_simulations()
        st.rerun()

    # Skeleton preview line
    sk = get_skeleton_preview_from_state()
    st.caption("Skeleton: " + ", ".join(f"{v:.3f}" for v in sk))

with col3:
    st.subheader("CRM knobs + preview")

    st.slider(
        "Prior sigma on theta",
        min_value=0.10,
        max_value=3.00,
        step=0.05,
        key=K_PRIOR_SIGMA_THETA,
        help="Prior SD for theta (CRM one-parameter power model).",
    )

    st.toggle(
        "Burn-in until first DLT",
        key=K_BURNIN_FIRST_DLT,
        help="If enabled, keep sampling until at least one DLT is observed (burn-in logic).",
    )

    st.toggle(
        "Enable EWOC overdose control",
        key=K_ENABLE_EWOC,
        help="If enabled, applies EWOC overdose control constraint.",
    )

    st.slider(
        "EWOC alpha",
        min_value=0.01,
        max_value=0.50,
        step=0.01,
        key=K_EWOC_ALPHA,
        disabled=(not bool(st.session_state[K_ENABLE_EWOC])),
        help="EWOC feasibility cutoff (alpha).",
    )

    # Compact plot (no giant height)
    true_curve = get_true_curve()
    sk = get_skeleton_preview_from_state()
    target = float(st.session_state[K_TARGET])
    true_mtd_idx = true_mtd_index_from_curve(true_curve, target)

    fig = plt.figure(figsize=(5.4, 2.6), dpi=140)
    ax = fig.add_subplot(111)
    x = np.arange(N_DOSES)
    ax.plot(x, true_curve, marker="o", label="True P(DLT)")
    ax.plot(x, sk, marker="o", label="Prior (skeleton)")
    ax.axhline(target, linewidth=1)
    ax.axvline(true_mtd_idx, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lab.split("\n")[0] for lab in DOSE_LABELS])
    ax.set_ylabel("Probability")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# --- RESULTS (compact, no big header)
res = st.session_state.get(K_RESULTS, None)

if res is not None:
    st.divider()

    # One row: 2 plots + metrics column
    r1, r2, r3 = st.columns([1.15, 1.15, 0.70], gap="large")

    p6 = np.asarray(res["p_select_6p3"], dtype=float)
    pc = np.asarray(res["p_select_crm"], dtype=float)
    n6 = np.asarray(res["avg_n_per_dose_6p3"], dtype=float)
    nc = np.asarray(res["avg_n_per_dose_crm"], dtype=float)

    with r1:
        fig1 = plt.figure(figsize=(5.2, 2.6), dpi=140)
        ax1 = fig1.add_subplot(111)
        x = np.arange(N_DOSES)
        w = 0.38
        ax1.bar(x - w/2, p6, width=w, label="6+3")
        ax1.bar(x + w/2, pc, width=w, label="CRM")
        true_mtd = int(res["meta"]["true_mtd_index"])
        ax1.axvline(true_mtd, linewidth=1)
        ax1.set_title("P(select dose as MTD)", fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"L{i}" for i in range(N_DOSES)])
        ax1.set_ylim(0, max(0.05, float(np.nanmax([p6.max(), pc.max()])) * 1.15))
        ax1.legend(fontsize=8)
        ax1.grid(True, axis="y", alpha=0.25)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with r2:
        fig2 = plt.figure(figsize=(5.2, 2.6), dpi=140)
        ax2 = fig2.add_subplot(111)
        x = np.arange(N_DOSES)
        w = 0.38
        ax2.bar(x - w/2, n6, width=w, label="6+3")
        ax2.bar(x + w/2, nc, width=w, label="CRM")
        ax2.set_title("Avg patients treated per dose", fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"L{i}" for i in range(N_DOSES)])
        ax2.set_ylim(0, max(1.0, float(np.nanmax([n6.max(), nc.max()])) * 1.20))
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="y", alpha=0.25)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with r3:
        st.metric("DLT prob per patient (6+3)", f"{res['p_dlt_per_patient_6p3']:.3f}")
        st.metric("DLT prob per patient (CRM)", f"{res['p_dlt_per_patient_crm']:.3f}")

        meta = res.get("meta", {})
        st.caption(
            f"n_sims={meta.get('n_sims')} | seed={meta.get('seed')} | True MTD marker=L{meta.get('true_mtd_index')}"
        )

    # Small reset below results (optional but handy)
    cR1, cR2 = st.columns([1, 3])
    with cR1:
        if st.button("Reset playground", use_container_width=True):
            reset_to_defaults(scope="playground")
            st.rerun()
    with cR2:
        st.caption("Results update only when you click Run simulations.")
else:
    # No results yet: keep it minimal, no big blank blocks
    st.caption("Click Run simulations to populate results.")
