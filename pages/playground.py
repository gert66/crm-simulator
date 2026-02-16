# pages/Playground.py
from __future__ import annotations

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import core


st.set_page_config(layout="wide")


def inject_compact_css() -> None:
    st.markdown(
        """
        <style>
        /* hide main H1 title (Playground) */
        h1 { display:none; }

        /* tighten top padding */
        section.main > div { padding-top: 0.6rem; }

        /* slightly smaller section headers */
        h2, h3 { margin-top: 0.4rem; }

        /* make widgets a bit tighter */
        .stSlider, .stNumberInput, .stSelectbox { margin-bottom: 0.35rem; }

        /* reduce gap between blocks */
        div.block-container { padding-bottom: 0.6rem; }

        /* compact metric cards */
        [data-testid="stMetric"] { padding: 0.2rem 0.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_prior_vs_true(true_p, prior_p, target, true_mtd_idx):
    fig = plt.figure(figsize=(4.8, 2.3), dpi=160)
    ax = fig.add_subplot(111)
    x = np.arange(len(true_p))
    ax.plot(x, true_p, marker="o", label="True P(DLT)")
    ax.plot(x, prior_p, marker="o", label="Prior (skeleton)")
    ax.axhline(target, linewidth=1)
    ax.axvline(true_mtd_idx, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.split("\n")[0] for lbl in core.DOSE_LABELS])
    ax.set_ylabel("Probability")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def plot_bars(title, p6, pc, true_mtd_idx):
    fig = plt.figure(figsize=(5.2, 2.2), dpi=160)
    ax = fig.add_subplot(111)
    x = np.arange(len(p6))
    w = 0.35
    ax.bar(x - w/2, p6, width=w, label="6+3")
    ax.bar(x + w/2, pc, width=w, label="CRM")
    ax.axvline(true_mtd_idx, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.split("\n")[0] for lbl in core.DOSE_LABELS])
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_patients(title, n6, nc):
    fig = plt.figure(figsize=(5.2, 2.2), dpi=160)
    ax = fig.add_subplot(111)
    x = np.arange(len(n6))
    w = 0.35
    ax.bar(x - w/2, n6, width=w, label="6+3")
    ax.bar(x + w/2, nc, width=w, label="CRM")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.split("\n")[0] for lbl in core.DOSE_LABELS])
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def main():
    core.init_state(st)
    inject_compact_css()

    # ---------- Top row: 3 columns ----------
    c1, c2, c3 = st.columns([1.05, 1.05, 1.15], gap="large")

    with c1:
        st.subheader("True acute DLT curve")
        st.toggle("Edit true curve", key="edit_true_curve", help="When off, the true curve is locked.")

        true_p = st.session_state["true_p"]
        new_true = []
        for i, lbl in enumerate(core.DOSE_LABELS):
            new_true.append(
                st.number_input(
                    lbl.replace("\n", " "),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    value=float(true_p[i]),
                    key=f"true_p_{i}",
                    disabled=not st.session_state["edit_true_curve"],
                    help="True probability of acute DLT at this dose (used to simulate outcomes).",
                )
            )
        # write back only if editing enabled
        if st.session_state["edit_true_curve"]:
            st.session_state["true_p"] = [float(x) for x in new_true]

        true_mtd_idx = core.true_mtd_index(st.session_state["true_p"], st.session_state["target"])
        st.caption(f"True MTD (closest to target) = {core.DOSE_LABELS[true_mtd_idx].split()[0]}")

    with c2:
        st.subheader("Prior playground")
        st.radio(
            "Skeleton model",
            ["empiric", "logistic"],
            key="skeleton_model",
            horizontal=True,
            help="Empiric makes a simple ladder around the prior target. Logistic makes a smooth curve.",
        )
        st.slider("Prior target", 0.01, 0.40, key="prior_target", step=0.01, help="Target toxicity used to build the prior skeleton.")
        st.slider("Halfwidth (delta)", 0.01, 0.20, key="delta", step=0.01, help="Controls how steep the empiric skeleton steps are.")
        st.slider("Prior MTD (nu, 1-based)", 1, 5, key="prior_mtd_nu", help="Dose index that the prior believes is closest to target.")
        st.slider("Logistic intercept", -5.0, 5.0, key="logistic_intercept", step=0.5, help="Only used for the logistic skeleton.")

        prior_p = core.build_skeleton_from_controls(st)
        st.caption("Skeleton: " + ", ".join([f"{p:.3f}" for p in prior_p]))

        # Run button moved here (middle column), as you asked
        st.write("")
        if st.button("Run simulations", use_container_width=True):
            st.session_state["last_results"] = core.run_simulations_stub(st)

    with c3:
        st.subheader("CRM knobs + preview")
        st.slider(
            "Prior sigma on theta",
            0.05, 3.0,
            key="sigma_theta",
            step=0.05,
            help="Controls how concentrated the CRM prior is around the skeleton.",
        )
        st.toggle("Burn-in until first DLT", key="burn_in_first_dlt", help="If enabled, constrain early escalation until a DLT is observed.")
        st.toggle("Enable EWOC overdose control", key="ewoc", help="If enabled, apply an overdose control rule.")
        st.slider("EWOC alpha", 0.01, 0.50, key="ewoc_alpha", step=0.01, disabled=not st.session_state["ewoc"], help="Overdose tolerance level.")

        fig = plot_prior_vs_true(
            true_p=np.array(st.session_state["true_p"], dtype=float),
            prior_p=np.array(prior_p, dtype=float),
            target=float(st.session_state["target"]),
            true_mtd_idx=true_mtd_idx,
        )
        st.pyplot(fig, use_container_width=True)

    # ---------- Results (compact) ----------
    if "last_results" in st.session_state:
        res = st.session_state["last_results"]
        st.markdown("### Results (6+3 vs CRM)")

        # 2 plots + metrics column (DLT probs right)
        r1, r2, r3 = st.columns([1.1, 1.1, 0.7], gap="large")

        with r1:
            fig1 = plot_bars(
                "P(select dose as MTD)",
                np.array(res.p_select_6p3),
                np.array(res.p_select_crm),
                true_mtd_idx,
            )
            st.pyplot(fig1, use_container_width=True)

        with r2:
            fig2 = plot_patients(
                "Avg patients treated per dose",
                np.array(res.mean_n_6p3),
                np.array(res.mean_n_crm),
            )
            st.pyplot(fig2, use_container_width=True)

        with r3:
            st.metric("DLT prob per patient (6+3)", f"{res.dlt_prob_6p3:.3f}")
            st.metric("DLT prob per patient (CRM)", f"{res.dlt_prob_crm:.3f}")
            st.caption(f"n_sims={st.session_state['n_sims']} | seed={st.session_state['seed']} | True MTD marker={core.DOSE_LABELS[true_mtd_idx].split()[0]}")

    else:
        st.caption("Run simulations to populate results.")


if __name__ == "__main__":
    main()
