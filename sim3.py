import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# Utilities
# ============================================================

def find_true_mtd(true_p, target):
    true_p = np.array(true_p, dtype=float)
    return int(np.argmin(np.abs(true_p - target)))

def simulate_bernoulli(n, p, rng):
    return rng.binomial(1, p, size=n)

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)

def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

def safe_probs(x):
    x = np.array(x, dtype=float)
    return np.clip(x, 1e-6, 1 - 1e-6)

# ============================================================
# 6+3 Design
# ============================================================

def run_6plus3(true_p, start_level=0, max_n=36, accept_max_dlt=1, rng=None):
    """
    Cohorts of 6.
    - If 0/6 DLT: escalate.
    - If 1/6 DLT: expand by 3 at same level (total 9 in that cycle),
      then escalate if DLTs among those 9 <= accept_max_dlt.
    - If >=2/6 DLT: stop / de-escalate (simple stop rule here).
    Final selection = last acceptable dose visited.
    """
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

        # If truncated cohort, stop
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
            # Expand by 3
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
                # too toxic
                if level > 0:
                    level -= 1
                break

        # d6 >= 2
        if level > 0:
            level -= 1
        break

    selected = 0 if last_acceptable is None else int(last_acceptable)
    return selected, n_per_level, int(dlt_per_level.sum())

# ============================================================
# CRM: 1-parameter power model
# p_k(theta) = skeleton_k ^ exp(theta)
# theta ~ Normal(0, sigma^2)
# Posterior computed on a grid
# ============================================================

def crm_power_probs(theta_grid, skeleton):
    sk = safe_probs(skeleton)
    a = np.exp(theta_grid)[:, None]    # (G,1)
    return sk[None, :] ** a            # (G,K)

def normal_prior_logpdf(theta_grid, sigma):
    return -0.5 * (theta_grid / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

def posterior_on_grid(theta_grid, sigma, skeleton, n_per_level, dlt_per_level):
    n = np.array(n_per_level, dtype=float)
    y = np.array(dlt_per_level, dtype=float)

    P = crm_power_probs(theta_grid, skeleton)  # (G,K)
    ll = (y[None, :] * np.log(P) + (n[None, :] - y[None, :]) * np.log(1 - P)).sum(axis=1)

    lp = normal_prior_logpdf(theta_grid, sigma)
    log_post = lp + ll

    lse = logsumexp(log_post)
    w = np.exp(log_post - lse)
    return w, P

def crm_choose_next(theta_grid, sigma, skeleton, n_per_level, dlt_per_level,
                    current_level, target, alpha_overdose, max_step=1):
    w, P = posterior_on_grid(theta_grid, sigma, skeleton, n_per_level, dlt_per_level)

    post_mean = (w[:, None] * P).sum(axis=0)
    overdose_prob = (w[:, None] * (P > target)).sum(axis=0)

    allowed = np.where(overdose_prob < alpha_overdose)[0]
    if allowed.size == 0:
        return 0, post_mean, overdose_prob

    k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

    # Restrict step size
    if k_star > current_level + max_step:
        k_star = current_level + max_step
    if k_star < current_level - max_step:
        k_star = current_level - max_step

    k_star = int(np.clip(k_star, 0, len(skeleton) - 1))
    return k_star, post_mean, overdose_prob

def crm_select_mtd(theta_grid, sigma, skeleton, n_per_level, dlt_per_level, target, alpha_overdose):
    w, P = posterior_on_grid(theta_grid, sigma, skeleton, n_per_level, dlt_per_level)

    post_mean = (w[:, None] * P).sum(axis=0)
    overdose_prob = (w[:, None] * (P > target)).sum(axis=0)

    allowed = np.where(overdose_prob < alpha_overdose)[0]
    if allowed.size == 0:
        return 0

    return int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

def run_crm(true_p, target, skeleton, sigma=1.0, start_level=0, max_n=36,
            cohort_size=6, alpha_overdose=0.25, theta_min=-4.0, theta_max=4.0,
            theta_grid_n=401, max_step=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_levels = len(true_p)
    level = int(start_level)

    n_per_level = np.zeros(n_levels, dtype=int)
    dlt_per_level = np.zeros(n_levels, dtype=int)
    total_n = 0

    theta_grid = np.linspace(theta_min, theta_max, int(theta_grid_n))

    while total_n < max_n:
        n_add = min(int(cohort_size), max_n - total_n)
        out = simulate_bernoulli(n_add, true_p[level], rng)

        n_per_level[level] += n_add
        dlt_per_level[level] += int(out.sum())
        total_n += n_add

        if n_add < cohort_size:
            break

        next_level, _, _ = crm_choose_next(
            theta_grid, sigma, skeleton,
            n_per_level, dlt_per_level,
            current_level=level,
            target=target,
            alpha_overdose=alpha_overdose,
            max_step=max_step
        )
        level = next_level

    selected = crm_select_mtd(
        theta_grid, sigma, skeleton,
        n_per_level, dlt_per_level,
        target=target,
        alpha_overdose=alpha_overdose
    )

    return selected, n_per_level, int(dlt_per_level.sum())

# ============================================================
# App UI
# ============================================================

st.set_page_config(page_title="6+3 vs CRM Simulator", layout="wide")
st.title("Dose Escalation Simulator: 6+3 vs CRM")

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]
level_labels = [f"L{i}" for i in range(5)]

# ------------------------------------------------------------
# Scenario library
# ------------------------------------------------------------

scenario_library = {
    "Default (MTD around level 2)": [0.05, 0.10, 0.20, 0.35, 0.55],
    "Safer overall (pushes higher)": [0.03, 0.06, 0.12, 0.20, 0.30],
    "More toxic overall (lands lower)": [0.08, 0.16, 0.28, 0.42, 0.60],
    "MTD around level 1": [0.05, 0.22, 0.35, 0.50, 0.65],
    "MTD around level 3": [0.03, 0.06, 0.12, 0.24, 0.40],
    "Sharp jump after level 2": [0.05, 0.08, 0.18, 0.40, 0.65],
}

skeleton_defaults = {
    "Match default scenario": [0.05, 0.10, 0.20, 0.35, 0.55],
    "More conservative skeleton": [0.07, 0.12, 0.22, 0.38, 0.58],
    "More optimistic skeleton": [0.03, 0.07, 0.16, 0.30, 0.48],
}

# ------------------------------------------------------------
# Study setup
# ------------------------------------------------------------

colA, colB = st.columns([1.05, 1.0])

with colA:
    st.subheader("Study setup")

    target = st.number_input(
        "Target DLT probability",
        min_value=0.05, max_value=0.50, value=0.25, step=0.01,
        help=(
            "The toxicity rate you aim for at the MTD. "
            "CRM will try to recommend a dose whose estimated DLT probability is closest to this target "
            "while respecting overdose control."
        )
    )

    start_level = st.selectbox(
        "Start dose level",
        options=list(range(0, 5)),
        index=0,
        format_func=lambda i: f"Level {i} ({dose_labels[i]})",
        help="Starting dose for both designs."
    )

    n_sims = st.number_input(
        "Number of simulated trials",
        min_value=200, max_value=20000, value=2000, step=200,
        help="Number of virtual trials. Higher means smoother estimates, slower runtime."
    )

    seed = st.number_input(
        "Random seed",
        min_value=1, max_value=10_000_000, value=12345, step=1,
        help="Fixes randomness so results are reproducible."
    )

with colB:
    st.subheader("True scenario (ground truth)")

    scenario_name = st.selectbox(
        "Scenario library",
        options=list(scenario_library.keys()),
        index=0,
        help="Pick a predefined set of true DLT probabilities as the simulation ground truth."
    )
    scenario_vals = scenario_library[scenario_name]

    manual_edit = st.toggle(
        "Manually edit the true DLT probabilities",
        value=True,
        help="If off, the scenario library values are used as-is. If on, you can tweak them below."
    )

    true_p = []
    for i, lab in enumerate(dose_labels):
        default_val = float(scenario_vals[i])
        if manual_edit:
            val = st.number_input(
                f"True P(DLT) at {lab}",
                min_value=0.0, max_value=1.0,
                value=default_val, step=0.01,
                key=f"true_{i}",
                help="Assumed true DLT probability at this dose level."
            )
        else:
            st.write(f"{lab}: {default_val:.2f}")
            val = default_val
        true_p.append(val)

    true_mtd = find_true_mtd(true_p, target)
    st.write(f"True MTD (closest to target {target:.2f}) = Level {true_mtd} ({dose_labels[true_mtd]})")

st.divider()

# ------------------------------------------------------------
# Design settings
# ------------------------------------------------------------

c1, c2 = st.columns([1.0, 1.0])

with c1:
    st.subheader("6+3 settings")

    max_n_6 = st.number_input(
        "Max sample size (6+3)",
        min_value=12, max_value=120, value=36, step=3,
        help="Maximum number of patients in each simulated 6+3 trial."
    )

    accept_rule = st.selectbox(
        "Acceptance rule after expansion to 9",
        options=[1, 2],
        index=0,
        help=(
            "If 1/6 DLT occurs, treat 3 more at the same dose (total 9 in that cycle). "
            "Escalate only if DLTs among those 9 are <= this number."
        )
    )

    st.info(
        f"""**6+3 decision rules (plain language)**  
- Treat 6 patients at the current dose.  
- If **0/6 DLT**, go up one dose level.  
- If **1/6 DLT**, treat **3 more** at the same dose (total 9 in that cycle).  
  Then go up one level only if **DLTs among those 9 <= {accept_rule}/9**.  
- If **2 or more DLTs in the first 6**, stop escalation (simple rule here).  
- The reported MTD is the **last dose considered acceptable** during the trial."""
    )

with c2:
    st.subheader("CRM settings")

    max_n_crm = st.number_input(
        "Max sample size (CRM)",
        min_value=12, max_value=120, value=36, step=3,
        help="Maximum number of patients in each simulated CRM trial. Independent of 6+3."
    )

    cohort_size = st.number_input(
        "Cohort size (CRM)",
        min_value=1, max_value=12, value=6, step=1,
        help="Patients treated before CRM updates and selects the next dose."
    )

    skeleton_preset = st.selectbox(
        "Skeleton preset",
        options=list(skeleton_defaults.keys()),
        index=0,
        help="Starting dose-toxicity shape for CRM. The model calibrates this shape using trial data."
    )

    manual_skeleton = st.toggle(
        "Manually edit the skeleton",
        value=True,
        help="If off, the selected skeleton preset is used as-is. If on, you can tweak it below."
    )

    sk_vals = skeleton_defaults[skeleton_preset]
    skeleton = []
    for i, lab in enumerate(dose_labels):
        default_val = float(sk_vals[i])
        if manual_skeleton:
            val = st.number_input(
                f"Skeleton prior mean at {lab}",
                min_value=0.01, max_value=0.99,
                value=default_val, step=0.01,
                key=f"sk_{i}",
                help="Prior mean DLT probability at this dose level. Must be increasing overall."
            )
        else:
            st.write(f"{lab}: {default_val:.2f}")
            val = default_val
        skeleton.append(val)

    sigma = st.number_input(
        "Prior sigma on theta",
        min_value=0.2, max_value=5.0, value=1.0, step=0.1,
        help=(
            "Uncertainty on the CRM curve calibration parameter. "
            "Higher sigma means more uncertainty and more learning from data."
        )
    )

    alpha_overdose = st.number_input(
        "Overdose control alpha",
        min_value=0.05, max_value=0.50, value=0.25, step=0.01,
        help=(
            "A dose is allowed only if the posterior probability that its DLT rate exceeds the target "
            "is less than this alpha. Smaller means more conservative."
        )
    )

    max_step = st.selectbox(
        "Max dose step per cohort",
        options=[1, 2],
        index=0,
        help="Limits how quickly CRM can move between dose levels per cohort."
    )

    st.info(
        f"""**CRM decision rules (plain language)**  
- Treat **{cohort_size}** patients at the current dose, then update the model.  
- The model assumes toxicity increases with dose (monotone curve).  
- Among doses that pass the safety filter, CRM chooses the dose whose estimated DLT probability is closest to **{target:.2f}**.  
- **Overdose control:** a dose is allowed only if **P(DLT > {target:.2f}) < {alpha_overdose:.2f}** under the posterior.  
- The next cohort can move by at most **±{max_step}** dose level(s)."""
    )

run = st.button("Run simulations")

# ============================================================
# Run simulations and show results
# ============================================================

if run:
    rng = np.random.default_rng(int(seed))
    ns = int(n_sims)

    sel_6 = np.zeros(5, dtype=int)
    sel_c = np.zeros(5, dtype=int)

    tot_dlt_6 = np.zeros(ns, dtype=int)
    tot_dlt_c = np.zeros(ns, dtype=int)

    nmat_6 = np.zeros((ns, 5), dtype=int)
    nmat_c = np.zeros((ns, 5), dtype=int)

    for s in range(ns):
        chosen6, n6, d6 = run_6plus3(
            true_p,
            start_level=int(start_level),
            max_n=int(max_n_6),
            accept_max_dlt=int(accept_rule),
            rng=rng
        )

        chosenc, nc, dc = run_crm(
            true_p,
            target=float(target),
            skeleton=skeleton,
            sigma=float(sigma),
            start_level=int(start_level),
            max_n=int(max_n_crm),
            cohort_size=int(cohort_size),
            alpha_overdose=float(alpha_overdose),
            max_step=int(max_step),
            rng=rng
        )

        sel_6[chosen6] += 1
        sel_c[chosenc] += 1

        tot_dlt_6[s] = d6
        tot_dlt_c[s] = dc

        nmat_6[s, :] = n6
        nmat_c[s, :] = nc

    p_sel_6 = sel_6 / float(ns)
    p_sel_c = sel_c / float(ns)

    avg_n6 = np.mean(nmat_6, axis=0)
    avg_nc = np.mean(nmat_c, axis=0)

    overshoot6 = float(np.mean((nmat_6[:, true_mtd + 1:].sum(axis=1)) > 0)) if true_mtd < 4 else 0.0
    overshootc = float(np.mean((nmat_c[:, true_mtd + 1:].sum(axis=1)) > 0)) if true_mtd < 4 else 0.0

    st.subheader("Results")

    x = np.arange(5)
    width = 0.38

    r1, r2 = st.columns(2)

    with r1:
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=140)
        ax.bar(x - width/2, p_sel_6, width, label="6+3")
        ax.bar(x + width/2, p_sel_c, width, label="CRM")
        ax.set_title("Probability of selecting each dose as MTD", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, max(p_sel_6.max(), p_sel_c.max()) * 1.15 + 1e-6)
        compact_style(ax)
        ax.axvline(true_mtd, linewidth=1, alpha=0.6)
        ax.text(true_mtd + 0.05, ax.get_ylim()[1] * 0.92, "True MTD", fontsize=8)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with r2:
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=140)
        ax.bar(x - width/2, avg_n6, width, label="6+3")
        ax.bar(x + width/2, avg_nc, width, label="CRM")
        ax.set_title("Average number treated per dose level", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Patients", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    r3, r4 = st.columns(2)

    with r3:
        st.markdown("**Trial summary (averages)**")
        st.write(f"Mean total DLTs: 6+3 = {np.mean(tot_dlt_6):.2f} | CRM = {np.mean(tot_dlt_c):.2f}")
        st.write(f"Mean sample size: 6+3 = {np.mean(nmat_6.sum(axis=1)):.1f} | CRM = {np.mean(nmat_c.sum(axis=1)):.1f}")
        st.write(f"True MTD: Level {true_mtd} ({dose_labels[true_mtd]})")

    with r4:
        st.markdown("**Safety signal**")
        st.write(f"P(overshoot above true MTD): 6+3 = {overshoot6:.3f} | CRM = {overshootc:.3f}")

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------
    st.divider()
    st.subheader("Export")

    # Build a CSV string without pandas
    header = [
        "design", "dose_level", "dose_label",
        "select_prob", "avg_n_treated",
        "true_p_dlt", "skeleton_p_dlt"
    ]

    lines = [",".join(header)]
    for k in range(5):
        lines.append(",".join([
            "6+3", str(k), dose_labels[k],
            f"{p_sel_6[k]:.6f}",
            f"{avg_n6[k]:.6f}",
            f"{true_p[k]:.6f}",
            f"{skeleton[k]:.6f}",
        ]))
    for k in range(5):
        lines.append(",".join([
            "CRM", str(k), dose_labels[k],
            f"{p_sel_c[k]:.6f}",
            f"{avg_nc[k]:.6f}",
            f"{true_p[k]:.6f}",
            f"{skeleton[k]:.6f}",
        ]))

    # Add a small footer block (as extra lines)
    lines.append("")
    lines.append("summary_item,value")
    lines.append(f"target,{target:.6f}")
    lines.append(f"true_mtd_level,{true_mtd}")
    lines.append(f"n_sims,{ns}")
    lines.append(f"seed,{int(seed)}")
    lines.append(f"max_n_6plus3,{int(max_n_6)}")
    lines.append(f"max_n_crm,{int(max_n_crm)}")
    lines.append(f"cohort_size_crm,{int(cohort_size)}")
    lines.append(f"overdose_alpha,{alpha_overdose:.6f}")
    lines.append(f"sigma_theta,{sigma:.6f}")
    lines.append(f"max_step_crm,{int(max_step)}")
    lines.append(f"overshoot_6plus3,{overshoot6:.6f}")
    lines.append(f"overshoot_crm,{overshootc:.6f}")
    lines.append(f"mean_total_dlts_6plus3,{np.mean(tot_dlt_6):.6f}")
    lines.append(f"mean_total_dlts_crm,{np.mean(tot_dlt_c):.6f}")

    csv_data = "\n".join(lines)

    st.download_button(
        label="Download results as CSV",
        data=csv_data.encode("utf-8"),
        file_name="dose_escalation_sim_results.csv",
        mime="text/csv",
        help="Downloads per-dose results for both designs and a small summary block."
    )
