import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)

st.set_page_config(page_title="Results", layout="wide")
st.title("Results (6+3 vs CRM)")

payload = st.session_state.get("results_payload")

if payload is None:
    st.warning("No results found yet. Run simulations on the main page first.")
    st.stop()

dose_labels = payload["dose_labels"]
true_mtd = int(payload["true_mtd"])
n_sims = int(payload["n_sims"])
seed = int(payload["seed"])

p63 = np.array(payload["p_mtd_63"], dtype=float)
pcrm = np.array(payload["p_mtd_crm"], dtype=float)

avg_n63 = np.array(payload["avg_n_63"], dtype=float)
avg_ncrm = np.array(payload["avg_n_crm"], dtype=float)

dlt_rate_63 = float(payload["dlt_rate_63"])
dlt_rate_crm = float(payload["dlt_rate_crm"])

# Top compact summary
m1, m2, m3 = st.columns([1.0, 1.0, 1.2], gap="large")
with m1:
    st.metric("DLT probability per patient (6+3)", f"{dlt_rate_63:.3f}")
with m2:
    st.metric("DLT probability per patient (CRM)", f"{dlt_rate_crm:.3f}")
with m3:
    st.caption(f"n_sims = {n_sims} | seed = {seed} | True MTD marker at L{true_mtd} ({dose_labels[true_mtd]})")

st.divider()

c1, c2 = st.columns([1.0, 1.0], gap="large")

with c1:
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=160)
    xx = np.arange(len(dose_labels))
    w = 0.38
    ax.bar(xx - w/2, p63, w, label="6+3")
    ax.bar(xx + w/2, pcrm, w, label="CRM")
    ax.set_title("Probability of selecting each dose as MTD", fontsize=10)
    ax.set_xticks(xx)
    ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(len(dose_labels))], fontsize=8)
    ax.set_ylabel("Probability", fontsize=9)
    ax.set_ylim(0, max(p63.max(), pcrm.max()) * 1.15 + 1e-6)
    ax.axvline(true_mtd, linewidth=1, alpha=0.6)
    ax.text(true_mtd + 0.05, ax.get_ylim()[1] * 0.92, "True MTD", fontsize=8)
    compact_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    st.pyplot(fig, clear_figure=True)

with c2:
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=160)
    xx = np.arange(len(dose_labels))
    w = 0.38
    ax.bar(xx - w/2, avg_n63, w, label="6+3")
    ax.bar(xx + w/2, avg_ncrm, w, label="CRM")
    ax.set_title("Average number of patients treated per dose", fontsize=10)
    ax.set_xticks(xx)
    ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(len(dose_labels))], fontsize=8)
    ax.set_ylabel("Patients", fontsize=9)
    compact_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    st.pyplot(fig, clear_figure=True)
