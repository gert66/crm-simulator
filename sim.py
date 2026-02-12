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

# ============================================================
# 6+3 design
# ============================================================

def run_6plus3(true_p, start_level=0, max_n=36, accept_max_dlt=1, rng=None):
    """
    Cohorts of 6. If 0/6 DLT -> escalate.
    If 1/6 DLT -> add 3 at same dose (total 9 in that cycle).
        If DLTs among those 9 <= accept_max_dlt -> escalate else stop/de-escalate.
    If >=2/6 DLT -> stop/de-escalate.
    Select MTD as last acceptable dose visited.
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
            # expand by 3 at same dose
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
# CRM: 1-parameter power model (O'Quigley style)
# p_k(theta) = skeleton_k ^ exp(theta)
# theta ~ Normal(0, sigma^2)
# Posterior computed on a grid.
# ============================================================

def crm_power_probs(theta_grid, skeleton):
    """
    Returns matrix P of shape (len(theta_grid), K) with p_k(theta).
    """
    sk = np.array(skeleton, dtype=float)
    # Ensure skeleton in (0,1)
    sk = np.clip(sk, 1e-6, 1 - 1e-6)
    a = np.exp(theta_grid)[:, None]  # (G,1)
    return sk[None, :] ** a          # (G,K)

def normal_prior_logpdf(theta_grid, sigma):
    return -0.5 * (theta_grid / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

def posterior_on_grid(theta_grid, sigma, skeleton, n_per_level, dlt_per_level):
    """
    Computes posterior weights on theta_grid given binomial likelihood across levels.
    """
    n = np.array(n_per_level, dtype=float)
    y = np.array(dlt_per_level, dtype=float)

    P = crm_power_probs(theta_grid, skeleton)  # (G,K)
    # Binomial log-likelihood up to an additive constant:
    # sum_k [ y_k log p_k + (n_k - y_k) log(1 - p_k) ]
    ll = (y[None, :] * np.log(P) + (n[None, :] - y[None, :]) * np.log(1 - P)).sum(axis=1)

    lp = normal_prior_logpdf(theta_grid, sigma)
    log_post_unnorm = lp + ll

    lse = logsumexp(log_post_unnorm)
    w = np.exp(log_post_unnorm - lse)  # normalized weights
    return w, P

def crm_choose_dose(theta_grid, sigma, skeleton, n_per_level, dlt_per_level,
                    current_level, target, alpha_overdose, max_step=1):
    """
    Choose next dose:
    - compute posterior mean toxicity per dose
    - apply overdose control: Pr(p_k > target) < alpha_overdose
    - among allowed, pick dose with posterior mean closest to target
    - restrict step size to +- max_step
    """
    w, P = posterior_on_grid(theta_grid, sigma, skeleton, n_per_level, dlt_per_level)
    post_mean = (w[:, None] * P).sum(axis=0)

    # overdose probability per level
    overdose_prob = (w[:, None] * (P > target)).sum(axis=0)

    allowed = np.where(overdose_prob < alpha_overdose)[0]
    if allowed.size == 0:
        # fall back to lowest dose
        chosen = 0
        return chosen, post_mean, overdose_prob

    # choose closest to target among allowed
    k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

    # restrict step size
    if k_star > current_level + max_step:
        k_star = current_level + max_step
    if k_star < current_level - max_step:
        k_star = current_level - max_step

    k_star = int(np.clip(k_star, 0, len(skeleton) - 1))
    return k_star, post_mean, overdose_prob

def crm_select_mtd(theta_grid, sigma, skeleton, n_per_level, dlt_per_level,
                   target, alpha_overdose):
    """
    Final MTD selection using the same posterior:
    - apply overdose control
    - among allowed, pick closest posterior mean to target
    """
    w, P = posterior_on_grid(theta_grid, sigma, skeleton, n_per_level, dlt_per_level)
    post_mean = (w[:, None] * P).sum(axis=0)
    overdose_prob = (w[:, None] * (P > target)).sum(axis=0)

    allowed = np.where(overdose_prob < alpha_overdose)[0]
    if allowed.size == 0:
        return 0

    chosen = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])
    return chosen

def run_crm(true_p, target, skeleton, sigma=1.0, start_level=0, max_n=36,
            cohort_size=6, alpha_overdose=0.25, theta_min=-4.0, theta_max=4.0, theta_grid_n=401,
            max_step=1, rng=None):
    """
    Simulate CRM trial:
    - treat cohorts
    - update posterior on theta grid
    - choose next dose based on overdose control and closeness to target
    """
    if rng is None:
        rng = np.random.default_rng()

    n_levels = len(true_p)
    level = int(start_level)

    n_per_level = np.zeros(n_levels, dtype=int)
    dlt_per_level = np.zeros(n_levels, dtype=int)
    total_n = 0

    theta_grid = np.linspace(theta_min, theta_max, int(theta_grid_n))

    while total_n < max_n:
        n_add = min(cohort_size, max_n - total_n)
        out = simulate_bernoulli(n_add, true_p[level], rng)

        n_per_level[level] += n_add
        dlt_per_level[level] += int(out.sum())
        total_n += n_add

        if n_add < cohort_size:
            break

        next_level, _, _ = crm_choose_dose(
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
# Streamlit App
# ============================================================

st.set_page_config(page_title="Dose Escalation Simulator: 6+3 vs CRM", layout="wide")
st.title("Dose Escalation Simulator: 6+3 vs CRM")

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]
level_labels = [f"L{i}" for i in range(5)]

col1, col2 = st.columns([1.05, 1.0])

with col1:
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

with col2:
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

c3, c4 = st.columns([1.0, 1.0])

with c3:
    st.subheader("6+3 settings")
    accept_rule = st.selectbox(
        "Acceptance rule after expansion to 9",
        options=[1, 2],
        index=0,
        help="Accept dose if DLTs among the 9 patients in that cycle are <= this number."
    )

with c4:
    st.subheader("CRM settings")
    default_skeleton = [0.05, 0.10, 0.20, 0.35, 0.55]
    skeleton = []
    for i, lab in enumerate(dose_labels):
        skeleton.append(
            st.number_input(
                f"Skeleton prior mean at {lab}",
                min_value=0.01, max_value=0.99,
                value=float(default_skeleton[i]),
                step=0.01,
                key=f"sk_{i}"
            )
        )

    sigma = st.number_input(
        "Prior sigma on theta (larger = more uncertainty)",
        min_value=0.2, max_value=5.0, value=1.0, step=0.1
    )

    alpha_overdose = st.number_input(
        "Overdose control alpha",
        min_value=0.05, max_value=0.50, value=0.25, step=0.01
    )

    max_step = st.selectbox("Max dose step per cohort", options=[1, 2], index=0)

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
            sigma=float(sigma),
            start_level=int(start_level),
            max_n=int(max_n),
            cohort_size=6,
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

    p_sel_6 = sel_6 / float(n_sims)
    p_sel_c = sel_c / float(n_sims)

    avg_n6 = np.mean(nmat_6, axis=0)
    avg_nc = np.mean(nmat_c, axis=0)

    overshoot6 = float(np.mean((nmat_6[:, true_mtd+1:].sum(axis=1)) > 0)) if true_mtd < 4 else 0.0
    overshootc = float(np.mean((nmat_c[:, true_mtd+1:].sum(axis=1)) > 0)) if true_mtd < 4 else 0.0

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
        ax.set_xticklabels([f"{level_labels[i]}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
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
        ax.set_xticklabels([f"{level_labels[i]}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
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
