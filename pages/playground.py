import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from ui_state import ensure_state, DOSE_LABELS
from core import simulate, pick_true_mtd

ensure_state()

st.markdown(
    """
    <style>
      .block-container { padding-top: 0.6rem; padding-bottom: 0.8rem; }
      h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
      h2 { font-size: 1.1rem; margin-top: 0.4rem; margin-bottom: 0.2rem; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# You asked: remove wasted space. Keep it minimal.
st.title("Playground")

# ---- Top controls: 3 columns ----
colA, colB, colC = st.columns([1.05, 1.05, 1.25], gap="small")

true_p = st.session_state["true_p"]

with colA:
    st.subheader("True acute DLT curve")
    st.toggle("Edit true curve", key="edit_true", help="If off, the curve stays fixed.")

    # Use compact number inputs, but do NOT overwrite values on rerun.
    # Keys are stable, and we only write back to true_p after reading.
    new_p = []
    for i, lab in enumerate(DOSE_LABELS):
        v = st.number_input(
            label=lab.split(" ")[1] if " " in lab else lab,
            min_value=0.0, max_value=1.0, step=0.01,
            key=f"true_p_{i}",
            value=float(true_p[i]),
            disabled=not st.session_state["edit_true"],
            help=f"True acute DLT probability at {lab}.",
        )
        new_p.append(float(v))

    # Write back once
    st.session_state["true_p"] = new_p
    true_mtd = pick_true_mtd(new_p, st.session_state["target"])
    st.caption(f"True MTD (closest to target) = {DOSE_LABELS[true_mtd]}")

with colB:
    st.subheader("Prior playground")
    st.radio(
        "Skeleton model",
        options=["empiric", "logistic"],
        key="skeleton_model",
        horizontal=True,
        help="Empiric: linear skeleton around target. Logistic: logistic-shaped skeleton.",
    )

    st.slider(
        "Prior target",
        min_value=0.01, max_value=0.50, step=0.01,
        key="prior_target",
        help="Target used to build the prior skeleton (can differ from study target).",
    )

    st.slider(
        "Halfwidth (delta)",
        min_value=0.01, max_value=0.30, step=0.01,
        key="delta",
        help="Spacing used in the empiric skeleton.",
        disabled=(st.session_state["skeleton_model"] != "empiric"),
    )

    st.slider(
        "Prior MTD (nu, 1-based)",
        min_value=1, max_value=len(DOSE_LABELS), step=1,
        key="prior_mtd_nu",
        help="Dose index where the prior skeleton is centered (1-based).",
    )

    st.slider(
        "Logistic intercept",
        min_value=-2.0, max_value=6.0, step=0.1,
        key="logit_intercept",
        help="Controls logistic skeleton level (used only if logistic).",
        disabled=(st.session_state["skeleton_model"] != "logistic"),
    )

with colC:
    st.subheader("CRM knobs + preview")
    st.slider(
        "Prior sigma on theta",
        min_value=0.05, max_value=3.0, step=0.05,
        key="sigma_theta",
        help="Controls prior spread for CRM (higher means less informative).",
    )
    st.toggle(
        "Burn-in until first DLT",
        key="burn_in_first_dlt",
        help="If enabled, CRM escalates conservatively until the first DLT is observed.",
    )
    st.toggle(
        "Enable EWOC overdose control",
        key="ewoc",
        help="If enabled, CRM uses an overdose guardrail (placeholder in this simplified engine).",
    )
    st.slider(
        "EWOC alpha",
        min_value=0.05, max_value=0.50, step=0.01,
        key="ewoc_alpha",
        help="Overdose probability cutoff (used if EWOC is enabled).",
        disabled=(not st.session_state["ewoc"]),
    )

    # Preview plot: small and wide
    # We use the current stored skeleton from last sim, or reconstruct quickly via simulate() call later.
    # For preview we keep it simple: show True curve vs a skeleton created by core.simulate on demand.
    # To avoid expensive sims: we build a tiny params dict and call simulate with n_sims=1? No.
    # Instead: rebuild skeleton locally using core.simulate logic by running simulate with n_sims=1 is still fine, but we can keep it minimal.
    from core import skeleton_empiric, skeleton_logistic

    K = len(DOSE_LABELS)
    if st.session_state["skeleton_model"] == "logistic":
        skel = skeleton_logistic(K, st.session_state["prior_target"], st.session_state["prior_mtd_nu"], st.session_state["logit_intercept"])
    else:
        skel = skeleton_empiric(K, st.session_state["prior_target"], st.session_state["prior_mtd_nu"], st.session_state["delta"])

    fig = plt.figure(figsize=(6.8, 2.4))
    ax = fig.add_subplot(111)
    x = np.arange(K)
    ax.plot(x, st.session_state["true_p"], marker="o", label="True P(DLT)")
    ax.plot(x, skel, marker="o", label="Prior (skeleton)")
    ax.axhline(st.session_state["target"], linewidth=1)
    ax.axvline(true_mtd, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([d.split(" ")[0] for d in DOSE_LABELS])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, min(1.0, max(max(skel), max(st.session_state["true_p"])) + 0.1))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig, use_container_width=True)

st.divider()

# ---- Run button + compact settings preview ----
run_col, settings_col = st.columns([1.0, 1.2], gap="small")

with run_col:
    if st.button("Run simulations", type="primary", use_container_width=True):
        params = {
            "target": st.session_state["target"],
            "start_dose": st.session_state["start_dose"],
            "crm_max_n": st.session_state["crm_max_n"],
            "crm_cohort": st.session_state["crm_cohort"],
            "sixplus3_max_n": st.session_state["sixplus3_max_n"],
            "n_sims": st.session_state["n_sims"],
            "seed": st.session_state["seed"],
            "true_p": st.session_state["true_p"],
            "skeleton_model": st.session_state["skeleton_model"],
            "prior_target": st.session_state["prior_target"],
            "delta": st.session_state["delta"],
            "prior_mtd_nu": st.session_state["prior_mtd_nu"],
            "logit_intercept": st.session_state["logit_intercept"],
            "sigma_theta": st.session_state["sigma_theta"],
            "burn_in_first_dlt": st.session_state["burn_in_first_dlt"],
            "ewoc": st.session_state["ewoc"],
            "ewoc_alpha": st.session_state["ewoc_alpha"],
        }
        st.session_state["results"] = simulate(params)

with settings_col:
    with st.expander("Current settings (from code)", expanded=False):
        st.write(
            {
                "target": st.session_state["target"],
                "start_dose": DOSE_LABELS[st.session_state["start_dose"]],
                "n_sims": st.session_state["n_sims"],
                "seed": st.session_state["seed"],
                "crm_max_n": st.session_state["crm_max_n"],
                "crm_cohort": st.session_state["crm_cohort"],
                "sixplus3_max_n": st.session_state["sixplus3_max_n"],
                "burn_in_first_dlt": st.session_state["burn_in_first_dlt"],
                "ewoc": st.session_state["ewoc"],
            }
        )

# ---- Results (compact) ----
res = st.session_state.get("results")
if res is None:
    st.caption("Run simulations to see results.")
    st.stop()

st.subheader("Results (6+3 vs CRM)")

m1, m2, m3 = st.columns([1.0, 1.0, 2.0], gap="small")
with m1:
    st.metric("DLT probability per patient (6+3)", f"{res['p_dlt_per_patient_633']:.3f}")
with m2:
    st.metric("DLT probability per patient (CRM)", f"{res['p_dlt_per_patient_crm']:.3f}")
with m3:
    st.caption(f"n_sims = {st.session_state['n_sims']} | seed = {st.session_state['seed']} | True MTD = {DOSE_LABELS[res['true_mtd']]}")

# Two compact plots
pcol1, pcol2 = st.columns([1.0, 1.0], gap="small")

x = np.arange(len(DOSE_LABELS))

with pcol1:
    fig1 = plt.figure(figsize=(6.8, 2.6))
    ax1 = fig1.add_subplot(111)
    w = 0.38
    ax1.bar(x - w/2, res["p_select_633"], width=w, label="6+3")
    ax1.bar(x + w/2, res["p_select_crm"], width=w, label="CRM")
    ax1.axvline(res["true_mtd"], linewidth=1)
    ax1.set_title("Probability of selecting each dose as MTD", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.split(" ")[0] for d in DOSE_LABELS])
    ax1.set_ylabel("Probability")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)
    st.pyplot(fig1, use_container_width=True)

with pcol2:
    fig2 = plt.figure(figsize=(6.8, 2.6))
    ax2 = fig2.add_subplot(111)
    ax2.bar(x - w/2, res["avg_n_633"], width=w, label="6+3")
    ax2.bar(x + w/2, res["avg_n_crm"], width=w, label="CRM")
    ax2.set_title("Average patients treated per dose", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.split(" ")[0] for d in DOSE_LABELS])
    ax2.set_ylabel("Patients")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    st.pyplot(fig2, use_container_width=True)
