import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from core import (
    DOSE_LABELS,
    dfcrm_getprior,
    find_true_mtd,
    simulate_many,
)

def _compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)

def _get_true_curve():
    return [float(st.session_state.get(f"true_{i}", 0.0)) for i in range(len(DOSE_LABELS))]

def _get_settings_snapshot():
    # only what simulator needs
    keys = [
        "target","start_level","max_n_crm","cohort_size","n_sims","seed","gh_n","max_step",
        "sigma","burn_in","ewoc_on","ewoc_alpha","guardrail","final_tried_only",
        "max_n_63","accept_rule_63",
    ]
    return {k: st.session_state.get(k) for k in keys}

st.set_page_config(page_title="Playground", layout="wide")

st.title("Playground")
st.caption("Tune true curve + priors + CRM knobs. Run simulations here and view results below without scrolling.")

# ============================================================
# TOP: Playground controls (3 columns like your screenshot)
# ============================================================

col_true, col_prior, col_knobs = st.columns([0.95, 1.05, 1.25], gap="large")

with col_true:
    st.markdown("### True acute DLT curve (compact)")
    edit_true = st.toggle("Edit true curve", value=True, help="Enable/disable editing of the true toxicity probabilities.")

    true_p = []
    for i, lab in enumerate(DOSE_LABELS):
        true_p.append(float(st.number_input(
            f"{lab}",
            min_value=0.0, max_value=1.0, step=0.01,
            key=f"true_{i}",
            disabled=(not edit_true),
            help=f"True P(DLT) used to simulate outcomes at this dose. Default from code is stored in session state."
        )))

    true_mtd = find_true_mtd(true_p, float(st.session_state.get("target", 0.15)))
    st.caption(f"True MTD (closest to target) = L{true_mtd} ({DOSE_LABELS[true_mtd]})")

with col_prior:
    st.markdown("### Prior playground (skeleton)")

    st.radio(
        "Skeleton model",
        options=["empiric", "logistic"],
        horizontal=True,
        key="prior_model",
        help="dfcrm-style skeleton construction."
    )

    st.slider(
        "Prior target",
        min_value=0.05, max_value=0.50, step=0.01,
        key="prior_target",
        help="dfcrm getprior target used to set skeleton around the prior MTD."
    )
    st.slider(
        "Halfwidth (delta)",
        min_value=0.01, max_value=0.30, step=0.01,
        key="halfwidth",
        help="dfcrm getprior halfwidth."
    )
    st.slider(
        "Prior MTD (nu, 1-based)",
        min_value=1, max_value=len(DOSE_LABELS), step=1,
        key="prior_nu",
        help="dfcrm getprior nu (1-based)."
    )
    st.slider(
        "Logistic intercept",
        min_value=0.0, max_value=10.0, step=0.1,
        key="logistic_intcpt",
        help="Only used if logistic skeleton model is selected."
    )

    skeleton = dfcrm_getprior(
        halfwidth=float(st.session_state["halfwidth"]),
        target=float(st.session_state["prior_target"]),
        nu=int(st.session_state["prior_nu"]),
        nlevel=len(DOSE_LABELS),
        model=str(st.session_state["prior_model"]),
        intcpt=float(st.session_state["logistic_intcpt"]),
    )
    st.caption("Skeleton: " + ", ".join([f"{v:.3f}" for v in skeleton]))

with col_knobs:
    st.markdown("### CRM knobs + preview")

    st.slider(
        "Prior sigma on theta",
        min_value=0.2, max_value=5.0, step=0.1,
        key="sigma",
        help="Controls prior spread on theta. Larger = looser prior."
    )
    st.toggle(
        "Burn-in until first DLT",
        key="burn_in",
        help="Escalate cohort-wise until first DLT is seen, then start CRM updates."
    )
    st.toggle(
        "Enable EWOC overdose control",
        key="ewoc_on",
        help="If enabled: allowed doses satisfy P(p_k > target | data) < alpha."
    )
    st.slider(
        "EWOC alpha",
        min_value=0.05, max_value=0.99, step=0.01,
        key="ewoc_alpha",
        disabled=(not st.session_state.get("ewoc_on", False)),
        help="EWOC threshold (only used when EWOC is enabled)."
    )

    # compact preview plot
    fig, ax = plt.subplots(figsize=(5.2, 2.2), dpi=170)
    x = np.arange(len(DOSE_LABELS))
    target_val = float(st.session_state.get("target", 0.15))

    ax.plot(x, true_p, marker="o", linewidth=1.6, label="True P(DLT)")
    ax.plot(x, skeleton, marker="o", linewidth=1.6, label="Prior (skeleton)")
    ax.axhline(target_val, linewidth=1, alpha=0.6)
    ax.axvline(true_mtd, linewidth=1, alpha=0.35)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}\n{DOSE_LABELS[i]}" for i in range(len(DOSE_LABELS))], fontsize=8)
    ax.set_ylabel("Probability", fontsize=9)
    ax.set_ylim(0, min(1.0, max(max(true_p), max(skeleton), target_val) * 1.20 + 0.02))
    _compact_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    st.pyplot(fig, clear_figure=True)

st.divider()

# ============================================================
# Run button + results stored in session_state
# ============================================================

run_col1, run_col2 = st.columns([0.30, 0.70], gap="large")

with run_col1:
    run = st.button(
        "Run simulations",
        type="primary",
        use_container_width=True,
        help="Runs n_sims trials for both designs using current Essentials + Playground settings."
    )

with run_col2:
    with st.expander("Current settings (from code)", expanded=False):
        st.json(_get_settings_snapshot())

if run:
    settings = _get_settings_snapshot()
    results = simulate_many(true_p=true_p, skeleton=skeleton, settings=settings)
    st.session_state["last_results"] = {
        "settings": settings,
        "true_p": true_p,
        "skeleton": skeleton.tolist(),
        "true_mtd": int(true_mtd),
        **{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in results.items()}
    }
    st.success("Results stored. Keep this window open for Playground + Results.")

# ============================================================
# Results (compact)
# ============================================================

if "last_results" in st.session_state:
    R = st.session_state["last_results"]

    st.markdown("## Results (6+3 vs CRM)")

    # Top metrics row (only what you requested)
    m1, m2, m3 = st.columns([0.35, 0.35, 0.30], gap="large")
    with m1:
        st.metric("DLT probability per patient (6+3)", f"{float(R['dlt_prob_63']):.3f}")
    with m2:
        st.metric("DLT probability per patient (CRM)", f"{float(R['dlt_prob_crm']):.3f}")
    with m3:
        st.caption(f"n_sims={R['settings']['n_sims']} | seed={R['settings']['seed']} | True MTD marker at L{R['true_mtd']} ({DOSE_LABELS[R['true_mtd']]})")

    # Two plots, compact height
    p1, p2 = st.columns([1.0, 1.0], gap="large")

    with p1:
        fig, ax = plt.subplots(figsize=(6.2, 2.6), dpi=170)
        xx = np.arange(len(DOSE_LABELS))
        w = 0.38
        ax.bar(xx - w/2, np.asarray(R["p_mtd_63"]), w, label="6+3")
        ax.bar(xx + w/2, np.asarray(R["p_mtd_crm"]), w, label="CRM")
        ax.set_title("Probability of selecting each dose as MTD", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}\n{DOSE_LABELS[i]}" for i in range(len(DOSE_LABELS))], fontsize=8)
        ax.set_ylabel("Probability", fontsize=9)
        ax.axvline(int(R["true_mtd"]), linewidth=1, alpha=0.6)
        _compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with p2:
        fig, ax = plt.subplots(figsize=(6.2, 2.6), dpi=170)
        xx = np.arange(len(DOSE_LABELS))
        w = 0.38
        ax.bar(xx - w/2, np.asarray(R["avg_n_63"]), w, label="6+3")
        ax.bar(xx + w/2, np.asarray(R["avg_n_crm"]), w, label="CRM")
        ax.set_title("Average number of patients treated per dose", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}\n{DOSE_LABELS[i]}" for i in range(len(DOSE_LABELS))], fontsize=8)
        ax.set_ylabel("Patients", fontsize=9)
        _compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)
