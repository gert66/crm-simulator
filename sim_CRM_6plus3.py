import numpy as np
import streamlit as st

# -----------------------------
# Helpers
# -----------------------------

def clip01(x):
    return min(max(x, 0.0), 1.0)

def find_true_mtd(true_p, target):
    true_p = np.array(true_p)
    return int(np.argmin(np.abs(true_p - target)))

def simulate_bernoulli(n, p, rng):
    return rng.binomial(1, p, size=n)

# -----------------------------
# 6+3 design simulation
# -----------------------------
def run_6plus3(true_p, start_level=0, max_n=36, accept_max_dlt=1, rng=None):
    """
    Cohorts of 6. If 0/6 DLT -> escalate.
    If 1/6 DLT -> add 3 more at same dose (total 9).
        If <= accept_max_dlt/9 -> escalate, else de-escalate/stop.
    If >=2/6 DLT -> de-escalate/stop.
    Returns: selected_mtd_level, n_per_level, total_dlts
    """
    if rng is None:
        rng = np.random.default_rng()

    n_levels = len(true_p)
    level = start_level
    n_per_level = np.zeros(n_levels, dtype=int)
    dlt_per_level = np.zeros(n_levels, dtype=int)
    total_n = 0

    last_acceptable = None

    while total_n < max_n:
        # Treat cohort of 6 at current level
        n_add = min(6, max_n - total_n)
        outcomes = simulate_bernoulli(n_add, true_p[level], rng)
        n_per_level[level] += n_add
        dlt_per_level[level] += int(outcomes.sum())
        total_n += n_add

        # Decision based on first 6 (or fewer if max_n truncation)
        # If truncated <6, stop and pick last acceptable
        if n_add < 6:
            break

        d6 = int(outcomes.sum())

        if d6 == 0:
            last_acceptable = level
            if level < n_levels - 1:
                level += 1
                continue
            else:
                break

        if d6 == 1:
            # Expand by 3 at same level (if possible)
            n_add2 = min(3, max_n - total_n)
            out2 = simulate_bernoulli(n_add2, true_p[level], rng)
            n_per_level[level] += n_add2
            dlt_per_level[level] += int(out2.sum())
            total_n += n_add2

            if n_add2 < 3:
                break

            d9 = dlt_per_level[level]  # at this level, but careful: could include earlier patients at same level
            # We want DLTs among the 9 just enrolled at this dose in this "cycle".
            # Simpler: approximate by using d6 + sum(out2), because we just treated 9 in this cycle.
            d9_cycle = d6 + int(out2.sum())

            if d9_cycle <= accept_max_dlt:
                last_acceptable = level
                if level < n_levels - 1:
                    level += 1
                    continue
                else:
                    break
            else:
                # too toxic
                if level > 0:
                    level -= 1
                    # stop and select previous acceptable
                    break
                else:
                    break

        # d6 >= 2
        if d6 >= 2:
            if level > 0:
                level -= 1
            break

    # Select MTD as last acceptable level encountered
    if last_acceptable is None:
        selected = 0
    else:
        selected = last_acceptable

    return selected, n_per_level, int(dlt_per_level.sum())

# -----------------------------
# CRM simulation (simple Bayesian beta-binomial per level)
# -----------------------------
def crm_posterior_means(skeleton, prior_strength, n_per_level, dlt_per_level):
    """
    Independent beta priors per dose level.
    Prior mean = skeleton[k], prior strength = m implies alpha=m*mean, beta=m*(1-mean).
    Posterior mean = (alpha + dlt) / (alpha + beta + n)
    """
    sk = np.array(skeleton, dtype=float)
    m = float(prior_strength)
    alpha0 = m * sk
    beta0 = m * (1 - sk)

    post_mean = (alpha0 + dlt_per_level) / (alpha0 + beta0 + n_per_level + 1e-12)
    return post_mean

def run_crm(true_p, target, skeleton, prior_strength=2.0, start_level=0, max_n=36,
            cohort_size=6, alpha_overdose=0.25, rng=None):
    """
    Very practical CRM:
    - Independent beta priors per level with means given by skeleton.
    - After each cohort, compute posterior means.
    - Choose dose with posterior mean closest to target among doses satisfying overdose control.
    - Overdose control: approximate using posterior beta to compute P(p > target) < alpha_overdose.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_levels = len(true_p)
    level = start_level
    n_per_level = np.zeros(n_levels, dtype=int)
    dlt_per_level = np.zeros(n_levels, dtype=int)
    total_n = 0

    # Precompute prior alpha/beta
    sk = np.array(skeleton, dtype=float)
    m = float(prior_strength)
    alpha0 = m * sk
    beta0 = m * (1 - sk)

    def overdose_prob(k):
        # P(p > target) under Beta(alpha, beta)
        # Use Monte Carlo for simplicity and robustness
        a = alpha0[k] + dlt_per_level[k]
        b = beta0[k] + (n_per_level[k] - dlt_per_level[k])
        samples = rng.beta(a, b, size=4000)
        return float(np.mean(samples > target))

    while total_n < max_n:
        n_add = min(cohort_size, max_n - total_n)
        out = simulate_bernoulli(n_add, true_p[level], rng)
        n_per_level[level] += n_add
        dlt_per_level[level] += int(out.sum())
        total_n += n_add

        if n_add < cohort_size:
            break

        # Compute posterior means
        post_mean = crm_posterior_means(sk, m, n_per_level, dlt_per_level)

        # Identify allowable doses under overdose control
        allowed = []
        for k in range(n_levels):
            if overdose_prob(k) < alpha_overdose:
                allowed.append(k)

        if len(allowed) == 0:
            # if everything violates, go to lowest dose and stop
            level = 0
            break

        # Choose closest to target among allowed
        allowed = np.array(allowed, dtype=int)
        k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

        # Restrict escalation to max +1 level
        if k_star > level + 1:
            level = level + 1
        elif k_star < level - 1:
            level = level - 1
        else:
            level = k_star

    # Select final dose as closest to target among allowed using final posterior
    post_mean = crm_posterior_means(sk, m, n_per_level, dlt_per_level)
    allowed_final = []
    for k in range(n_levels):
        # reuse overdose prob with current post
        a = alpha0[k] + dlt_per_level[k]
        b = beta0[k] + (n_per_level[k] - dlt_per_level[k])
        samples = rng.beta(a, b, size=6000)
        if float(np.mean(samples > target)) < alpha_overdose:
            allowed_final.append(k)

    if len(allowed_final) == 0:
        selected = 0
    else:
        allowed_final = np.array(allowed_final, dtype=int)
        selected = int(allowed_final[np.argmin(np.abs(post_mean[allowed_final] - target))])

    return selected, n_per_level, int(dlt_per_level.sum())

# -----------------------------
# Main app
# -----------------------------
st.set_page_config(page_title="6+3 vs CRM Simulator", layout="wide")
st.title("Dose Escalation Simulator: 6+3 vs CRM")

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]

colA, colB = st.columns(2)

with colA:
    st.subheader("Study setup")
    target = st.number_input("Target DLT probability", min_value=0.05, max_value=0.50, value=0.25, step=0.01)
    max_n = st.number_input("Max sample size", min_value=12, max_value=90, value=36, step=3)
    start_level = st.selectbox("Start dose level", options=list(range(0, 5)), format_func=lambda i: f"Level {i} ({dose_labels[i]})", index=0)
    n_sims = st.number_input("Number of simulated trials", min_value=200, max_value=20000, value=2000, step=200)

with colB:
    st.subheader("True scenario (ware DLT-kansen)")
    default_true = [0.05, 0.10, 0.20, 0.35, 0.55]
    true_p = []
    for i, lab in enumerate(dose_labels):
        true_p.append(st.number_input(f"True P(DLT) at {lab}", min_value=0.0, max_value=1.0, value=float(default_true[i]), step=0.01))
    true_mtd = find_true_mtd(true_p, target)
    st.write(f"True MTD (closest to target {target:.2f}) = Level {true_mtd} ({dose_labels[true_mtd]})")

st.divider()

c1, c2 = st.columns(2)

with c1:
    st.subheader("6+3 settings")
    accept_max_dlt = st.selectbox("Acceptance rule in expansion cohort (out of 9)", options=[1, 2], index=0,
                                  help="1 means accept <=1/9. 2 means accept <=2/9.")
with c2:
    st.subheader("CRM settings")
    default_skeleton = [0.05, 0.10, 0.20, 0.35, 0.55]
    skeleton = []
    for i, lab in enumerate(dose_labels):
        skeleton.append(st.number_input(f"Skeleton prior mean at {lab}", min_value=0.0, max_value=1.0, value=float(default_skeleton[i]), step=0.01, key=f"sk_{i}"))
    prior_strength = st.number_input("Prior strength (pseudo sample size)", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
    alpha_overdose = st.number_input("Overdose control alpha", min_value=0.05, max_value=0.50, value=0.25, step=0.01)

run = st.button("Run simulations")

if run:
    rng = np.random.default_rng(12345)

    sel_6 = np.zeros(5, dtype=int)
    sel_c = np.zeros(5, dtype=int)

    tot_dlt_6 = []
    tot_dlt_c = []

    nmat_6 = np.zeros((n_sims, 5), dtype=int)
    nmat_c = np.zeros((n_sims, 5), dtype=int)

    for s in range(int(n_sims)):
        chosen6, n6, d6 = run_6plus3(true_p, start_level=start_level, max_n=int(max_n), accept_max_dlt=int(accept_max_dlt), rng=rng)
        chosenc, nc, dc = run_crm(true_p, target=target, skeleton=skeleton, prior_strength=prior_strength,
                                  start_level=start_level, max_n=int(max_n), cohort_size=6,
                                  alpha_overdose=alpha_overdose, rng=rng)

        sel_6[chosen6] += 1
        sel_c[chosenc] += 1
        tot_dlt_6.append(d6)
        tot_dlt_c.append(dc)
        nmat_6[s, :] = n6
        nmat_c[s, :] = nc

    p_sel_6 = sel_6 / n_sims
    p_sel_c = sel_c / n_sims

    st.subheader("Results")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Selection probability per dose level**")
        st.write("6+3:", {f"Level {i}": float(p_sel_6[i]) for i in range(5)})
        st.write("CRM:", {f"Level {i}": float(p_sel_c[i]) for i in range(5)})

    with r2:
        st.markdown("**Average trial outcomes**")
        st.write(f"6+3 mean total DLTs: {np.mean(tot_dlt_6):.2f}")
        st.write(f"CRM mean total DLTs: {np.mean(tot_dlt_c):.2f}")
        st.write(f"6+3 mean sample size: {np.mean(nmat_6.sum(axis=1)):.1f}")
        st.write(f"CRM mean sample size: {np.mean(nmat_c.sum(axis=1)):.1f}")

    st.markdown("**Average number treated per dose**")
    avg_n6 = np.mean(nmat_6, axis=0)
    avg_nc = np.mean(nmat_c, axis=0)
    st.write("6+3:", {dose_labels[i]: float(avg_n6[i]) for i in range(5)})
    st.write("CRM:", {dose_labels[i]: float(avg_nc[i]) for i in range(5)})

    # Overshoot probability: any patient treated above true MTD
    overshoot6 = float(np.mean((nmat_6[:, true_mtd+1:].sum(axis=1)) > 0)) if true_mtd < 4 else 0.0
    overshootc = float(np.mean((nmat_c[:, true_mtd+1:].sum(axis=1)) > 0)) if true_mtd < 4 else 0.0
    st.markdown("**Safety signals**")
    st.write(f"P(overshoot above true MTD): 6+3 = {overshoot6:.3f}, CRM = {overshootc:.3f}")
