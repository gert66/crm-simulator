import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)

st.set_page_config(page_title="Results", layout="wide")

st.title("Results (6+3 vs CRM)")
st.caption("Tip: open this page in a separate browser window and move it to your second monitor.")

payload = st.session_state.get("results_payload", None)

if payload is None:
    st.warning("No results found yet. Go back to the main page, run simulations, then return here.")
    st.page_link("app.py", label="Go to main page", icon="⬅️")
    st.stop()

dose_labels = payload["dose_labels"]
K = len(dose_labels)
x = np.arange(K)

true_mtd = int(payload["true_mtd"])

p63 = np.array(payload["p63"], dtype=float)
pcrm = np.array(payload["pcrm"], dtype=float)
avg63 = np.array(payload["avg63"], dtype=float)
avgcrm = np.array(payload["avgcrm"], dtype=float)
prop63 = np.array(payload["prop63"], dtype=float)
propcrm = np.array(payload["propcrm"], dtype=float)

st.subheader("Key metrics")
m1, m2, m3, m4 = st.columns(4, gap="large")
m1.metric("Mean sample size (6+3)", f"{payload['mean_n63']:.1f}")
m2.metric("Mean sample size (CRM)", f"{payload['mean_ncrm']:.1f}")
m3.metric("Mean total DLTs (6+3)", f"{payload['mean_dlt63']:.2f}")
m4.metric("Mean total DLTs (CRM)", f"{payload['mean_dltcrm']:.2f}")

st.divider()

c1, c2 = st.columns(2, gap="large")

with c1:
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=160)
    w = 0.38
    ax.bar(x - w/2, p63, w, label="6+3")
    ax.bar(x + w/2, pcrm, w, label="CRM")
    ax.set_title("Probability of selecting each dose as MTD", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(K)], fontsize=8)
    ax.set_ylabel("Probability", fontsize=9)
    ax.set_ylim(0, max(p63.max(), pcrm.max()) * 1.15 + 1e-6)
    ax.axvline(true_mtd, linewidth=1, alpha=0.6)
    ax.text(true_mtd + 0.05, ax.get_ylim()[1] * 0.92, "True MTD", fontsize=8)
    compact_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    st.pyplot(fig, clear_figure=True)

with c2:
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=160)
    w = 0.38
    ax.bar(x - w/2, avg63, w, label="6+3")
    ax.bar(x + w/2, avgcrm, w, label="CRM")
    ax.set_title("Average number treated per dose level", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(K)], fontsize=8)
    ax.set_ylabel("Patients", fontsize=9)
    compact_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    st.pyplot(fig, clear_figure=True)

c3, c4 = st.columns(2, gap="large")

with c3:
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=160)
    w = 0.38
    ax.bar(x - w/2, prop63, w, label="6+3")
    ax.bar(x + w/2, propcrm, w, label="CRM")
    ax.set_title("Percentage treated at each dose (mean over trials)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(K)], fontsize=8)
    ax.set_ylabel("Proportion", fontsize=9)
    ax.set_ylim(0, max(prop63.max(), propcrm.max()) * 1.15 + 1e-6)
    compact_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    st.pyplot(fig, clear_figure=True)

with c4:
    st.markdown("**R-like summary vectors**")
    st.write("CRM: [Prob(MTD=1..5), Mean sample size, Mean % treated at dose 1..5]")
    st.code(", ".join([f"{v:.4f}" for v in payload["r_like_crm"]]))
    st.write("6+3: [Prob(MTD=1..5), Mean sample size, Mean % treated at dose 1..5]")
    st.code(", ".join([f"{v:.4f}" for v in payload["r_like_63"]]))

if payload.get("debug_dump", None):
    with st.expander("CRM debug (first simulated trial)", expanded=False):
        for i, row in enumerate(payload["debug_dump"], start=1):
            st.write(
                f"Update {i}: treated L{row['treated_level']} | n={row['cohort_n']} "
                f"| dlts={row['cohort_dlts']} | any_dlt_seen={row['any_dlt_seen']}"
            )
            if "next_level" in row:
                st.write(f"  allowed: {row['allowed']} | next: L{row['next_level']} | highest_tried={row['highest_tried']}")
                st.write(f"  post_mean: {[round(v, 3) for v in row['post_mean']]}")
                st.write(f"  od_prob:   {[round(v, 3) for v in row['od_prob']]}")
