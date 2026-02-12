import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Core utilities
# ------------------------------------------------------------

def find_true_mtd(true_p, target):
    true_p = np.array(true_p, dtype=float)
    return int(np.argmin(np.abs(true_p - target)))

def simulate_bernoulli(n, p, rng):
    return rng.binomial(1, p, size=n)

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)

# ------------------------------------------------------------
# 6+3 design (cohort of 6, optional expansion by 3)
# Rule:
#   0/6 -> escalate
#   1/6 -> expand to 9 at same dose; accept if <= accept_max_dlt out of 9, then escalate
#   >=2/6 -> de-escalate/stop
# MTD selected as last acceptable dose visited
# ------------------------------------------------------------

def run_6plus3(true_p, start_level=0, max_n=36, accept_max_dlt=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_levels = len(true_p)
    level = int(start_level)
    n_per_level = np.zeros(n_levels, dtype=int)
    dlt_per_level = np.zeros(n_levels, dtype=int)

    total_n = 0
    last_acceptable = None

    while total_n < max_n:
        # Treat 6 at current level
        n_add = min(6, max_n - total_n)
        out6 = simulate_bernoulli(n_add, true_p[level], rng)

        n_per_level[level] += n_add
        dlt_per_level[level] += int(out6.sum())
        total_n += n_add

        if n_add < 6:
            break

        d6 = int(out6.sum())

        if d6 == 0:
            last_acceptable = level
            if level < n_levels - 1:
                level += 1
                continue
            break

        if d6 == 1:
            # Expand by 3 at same dose
            n_add2 = min(3, max_n - total_n)
            out3 = simulate_bernoulli(n_add2, true_p[level], rng)

            n_per_level[level] += n_add2
            dlt_per_level[level] += int(out3.sum())
            total_n += n_add2

            if n_add2 < 3:
                break

            d9_cycle = d6 + int(out3.sum())
            if d9_cycle <= accept_max_dlt:
                last_acceptable = level
                if level < n_levels - 1:
                    level += 1
                    continue
                break
            else:
                if level > 0:
                    level -= 1
                break

        # d6 >= 2
        if level > 0:
            level -= 1
        break

    selected = 0 if last_acceptable is None else int(last_acceptable)
    return selected, n_per_level, int(dlt_per_level.sum())

# ------------------------------------------------------------
# Simple CRM implementation (practical and app-friendly)
# - Independent Beta priors per level with mean = skeleton
# - Posterior updated per level
# - Dose choice: posterior mean closest to target among doses satisfying overdose control
# - Overdose control: P(p > target) < alpha_overdose, approximated via Monte Carlo from Beta posterior
# - Escalation restricted to at most +1 and at most -1 per cohort
# ------------------------------------------------------------

def crm_posterior_means(skeleton, prior_strength, n_per_level, dlt_per_level):
    sk = np.array(skeleton, dtype=float)
    m = float(prior_strength)
    a0 = m * sk
    b0 = m * (1 - sk)
    return (a0 + dlt_per_level) / (a0 + b0 + n_per_level + 1e-12)

def run_crm(true_p, target, skeleton, prior_strength=2.0, start_level=0, max_n=36,
            cohort_size=6, alpha_overdose=0.25, rng=None,
            mc_samples=4000, mc_samples_final=7000):

    if rng is None:
        rng = np.random.default_rng()

    n_levels = len(true_p)
    level = int(start_level)
    n_per_level = np.zeros(n_levels, dtype=int)
    dlt_per_level = np.zeros(n_levels, dtype=int)
    total_n = 0

    sk = np.array(skeleton, dtype=float)
    m = float(prior_strength)
    a0 = m * sk
    b0 = m * (1 - sk)

    def overdose_prob(k, nsamp):
        a = a0[k] + dlt_per_level[k]
        b = b0[k] + (n_per_level[k] - dlt_per_level[k])
        samp = rng.beta(a, b, size=nsamp)
        return float(np.mean(samp > target))

    while total_n < max_n:
        n_add = min(cohort_size, max_n - total_n)
        out = simulate_bernoulli(n_add, true_p[level], rng)

        n_per_level[level] += n_add
        dlt_per_level[level] += int(out.sum())
        total_n += n_add

        if n_add < cohort_size:
            break

        post_mean = crm_posterior_means(sk, m, n_per_level, dlt_per_level)

        allowed = []
        for k in range(n_levels):
            if overdose_prob(k, mc_samples) < alpha_overdose:
                allowed.append(k)

        if len(allowed) == 0:
            level = 0
            break

        allowed = np.array(allowed, dtype=int)
        k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

        # Restrict step size
        if k_star > level + 1:
            level = level + 1
        elif k_star < level - 1:
            level = level - 1
        else:
            level = k_star

    # Final selection
    post_mean = crm_posterior_means(sk, m, n_per_level, dlt_per_level)

    allowed_final = []
    for k in range(n_levels):
        if overdose_prob(k, mc_samples_final) < alpha_overdose:
            allowed_final.append(k)

    if len(allowed_final) == 0:
        selected = 0
    else:
        allowed_final = np.array(allowed_final, dtype=int)
        selected = int(allowed_final[np.argmin(np.abs(post_mean[allowed_final] - target))])

    return selected, n_per_level, int(dlt_per_level.sum())

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

st.set_page_config(page_title="6+3 vs CRM Simulator", layout="wide")
st.title("Dose Escalation Simulator: 6+3 vs CRM")

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]
level_labels = [f"L{i}" for i in range(5)]

top1, top2 = st.columns([1.05, 1.0])

with top1:
    st.subheader("Study setup")
    target = st.number_input("Target DLT probability", min_value=0.05, max_value=0.50, value=0.25, step=0.01)
    max_n = st.number_input("Max sample size", min_value=12, max_value=90, value=36, step=3)
    start_level = st.selectbox(
        "Start dose level",
        options=list(range(0, 5)),
        format_func=lambda i: f"Level {i} ({dose_labels[i]})",
        index=0
    )
    n_sims = st.number_input("Number of simulated trials", min_value=200, max_value=20000, value=2000, step=200)
    seed = st.number_input("Random seed", min_value=1, max_value=10_000_000, value=12345, step=1)

with top2:
    st.subheader("True scenario (ware DLT-kansen)")
    default_true = [0.05, 0.10, 0.20, 0.35, 0.55]
    true_p = []
    for i, lab in enumerate(dose_labels):
        true_p.append(
            st.number_input(
                f"True P(DLT) at {lab}",
                min_value=0.0, max_value=1.0,
                value=float(default_true[i]),
                step=0.01,
                key=f"true_{i}"
            )
        )
    true_mtd = find_true_mtd(true_p, target)
    st.write(f"True MTD (closest to target {target:.2f}) = Level {true_mtd} ({dose_labels[true_mtd]})")

st.divider()

mid1, mid2 = st.columns([1.0, 1.0])

with mid1:
    st.subheader("6+3 settings")
    accept_rule = st.selectbox("Acceptance rule after expansion to 9", options=[1, 2], index=0,
                               help="Accept dose if DLTs among 9 patients in that cycle <= this number.")

with mid2:
    st.subheader("CRM settings")
    default_skeleton = [0.05, 0.10, 0.20, 0.35, 0.55]
    skeleton = []
    for i, lab in enumerate(dose_labels):
        skeleton.append(
            st.number_input(
                f"Skeleton prior mean at {lab}",
                min_value=0.0, max_value=1.0,
                value=float(default_skeleton[i]),
                step=0.01,
                key=f"sk_{i}"
            )
        )
    prior_strength = st.number_input("Prior strength (pseudo sample size)", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
    alpha_overdose = st.number_input("Overdose control alpha", min_value=0.05, max_value=0.50, value=0.25, step=0.01)

run = st.button("Run simulations")

if run:
    rng = np.random.default_rng(int(seed))

    sel_6 = np.zeros(5, dtype=int)
    sel_c = np.zeros(5, dtype=int)

    tot_dlt_6 = np.zeros(int(n_sims), dtype=int)
    tot_dlt_c = np.zeros(int(n_sims), dtype=int)

    nmat_6 = np.zeros((int(n_sims), 5), dtype=int)
    nmat_c = np.zeros((int(n_sims), 5), dtype=int)

    for s in range(int(n_sims)):
        chosen6, n6, d6 = run_6plus3(
            true_p,
            start_level=int(start_level),
            max_n=int(max_n),
            accept_max_dlt=int(accept_rule),
            rng=rng
        )
        chosenc, nc, dc = run_crm(
            true_p,
            target=float(target),
            skeleton=skeleton,
            prior_strength=float(prior_strength),
            start_level=int(start_level),
            max_n=int(max_n),
            cohort_size=6,
            alpha_overdose=float(alpha_overdose),
            rng=rng
        )

        sel_6[chosen6] += 1
        sel_c[chosenc] += 1

        tot_dlt_6[s] = d6
        tot_dlt_c[s] = dc

        nmat_6[s, :] = n6
        nmat_c[s, :] = nc

    p_sel_6 = sel_6 / float(n_sims)
    p_sel_c = sel_c / float(n_sims)

    avg_n6 = np.mean(nmat_6, axis=0)
    avg_nc = np.mean(nmat_c, axis=0)

    overshoot6 = float(np.mean((nmat_6[:, true_mtd+1:].sum(axis=1)) > 0)) if true_mtd < 4 else 0.0
    overshootc = float(np.mean((nmat_c[:, true_mtd+1:].sum(axis=1)) > 0)) if true_mtd < 4 else 0.0

    st.subheader("Results")

    x = np.arange(5)
    width = 0.38

    # Row 1: two compact plots
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=140)
        ax.bar(x - width/2, p_sel_6, width, label="6+3")
        ax.bar(x + width/2, p_sel_c, width, label="CRM")
        ax.set_title("Probability of selecting each dose as MTD", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{level_labels[i]}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, max(p_sel_6.max(), p_sel_c.max()) * 1.15 + 1e-6)
        compact_style(ax)

        ax.axvline(true_mtd, linewidth=1, alpha=0.6)
        ax.text(true_mtd + 0.05, ax.get_ylim()[1] * 0.92, "True MTD", fontsize=8)

        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with c2:
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=140)
        ax.bar(x - width/2, avg_n6, width, label="6+3")
        ax.bar(x + width/2, avg_nc, width, label="CRM")
        ax.set_title("Average number treated per dose level", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{level_labels[i]}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Patients", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    # Row 2: compact numeric summaries + optional counts chart
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Trial summary (averages)**")
        st.write(f"Mean total DLTs: 6+3 = {np.mean(tot_dlt_6):.2f} | CRM = {np.mean(tot_dlt_c):.2f}")
        st.write(f"Mean sample size: 6+3 = {np.mean(nmat_6.sum(axis=1)):.1f} | CRM = {np.mean(nmat_c.sum(axis=1)):.1f}")
        st.write(f"True MTD: Level {true_mtd} ({dose_labels[true_mtd]})")

    with c4:
        st.markdown("**Safety signal**")
        st.write(f"P(overshoot above true MTD): 6+3 = {overshoot6:.3f} | CRM = {overshootc:.3f}")

    with st.expander("Optional: counts of selected MTD across simulations"):
        fig, ax = plt.subplots(figsize=(6.2, 2.5), dpi=140)
        ax.bar(x - width/2, sel_6, width, label="6+3")
        ax.bar(x + width/2, sel_c, width, label="CRM")
        ax.set_title("Counts of selected MTD across simulations", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{level_labels[i]}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Count", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)
