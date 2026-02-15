import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# Utilities
# ============================================================

def safe_probs(x):
    x = np.array(x, dtype=float)
    return np.clip(x, 1e-6, 1 - 1e-6)

def simulate_bernoulli(n, p, rng):
    return rng.binomial(1, p, size=n)

def find_true_mtd(true_p, target):
    true_p = np.array(true_p, dtype=float)
    return int(np.argmin(np.abs(true_p - target)))

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)

# ============================================================
# dfcrm-style skeleton calibration (getprior) | 1:1 translation
# ============================================================

def dfcrm_getprior(halfwidth, target, nu, nlevel, model="empiric", intcpt=3.0):
    halfwidth = float(halfwidth)
    target = float(target)
    nu = int(nu)
    nlevel = int(nlevel)
    intcpt = float(intcpt)

    if not (0 < target < 1):
        raise ValueError("target must be in (0, 1).")
    if halfwidth <= 0:
        raise ValueError("halfwidth must be > 0.")
    if (target - halfwidth) <= 0 or (target + halfwidth) >= 1:
        raise ValueError("halfwidth too large: target±halfwidth must stay within (0,1).")
    if not (1 <= nu <= nlevel):
        raise ValueError("nu must be between 1 and nlevel (inclusive).")

    dosescaled = np.full(nlevel, np.nan, dtype=float)

    if model == "empiric":
        dosescaled[nu - 1] = target

        for k in range(nu, 1, -1):
            b_k = np.log(np.log(target + halfwidth) / np.log(dosescaled[k - 1]))
            dosescaled[k - 2] = np.exp(np.log(target - halfwidth) / np.exp(b_k))

        for k in range(nu, nlevel):
            b_k1 = np.log(np.log(target - halfwidth) / np.log(dosescaled[k - 1]))
            dosescaled[k] = np.exp(np.log(target + halfwidth) / np.exp(b_k1))

        return dosescaled

    if model == "logistic":
        dosescaled[nu - 1] = np.log(target / (1 - target)) - intcpt

        for k in range(nu, 1, -1):
            b_k = np.log((np.log((target + halfwidth) / (1 - target - halfwidth)) - intcpt) / dosescaled[k - 1])
            dosescaled[k - 2] = (np.log((target - halfwidth) / (1 - target + halfwidth)) - intcpt) / np.exp(b_k)

        for k in range(nu, nlevel):
            b_k1 = np.log((np.log((target - halfwidth) / (1 - target + halfwidth)) - intcpt) / dosescaled[k - 1])
            dosescaled[k] = (np.log((target + halfwidth) / (1 - target - halfwidth)) - intcpt) / np.exp(b_k1)

        prior = (1 + np.exp(-intcpt - dosescaled)) ** (-1)
        return prior

    raise ValueError('model must be "empiric" or "logistic".')

# ============================================================
# CRM (acute-only): Gauss–Hermite quadrature
# ============================================================

def posterior_via_gh(sigma, skeleton, n_per_level, dlt_per_level, gh_n=61):
    """
    Model:
      p_k(theta) = skeleton_k ^ exp(theta)
      theta ~ Normal(0, sigma^2)
    """
    sk = safe_probs(skeleton)
    n = np.asarray(n_per_level, dtype=float)
    y = np.asarray(dlt_per_level, dtype=float)

    x, w = np.polynomial.hermite.hermgauss(int(gh_n))
    theta = float(sigma) * np.sqrt(2.0) * x

    P = sk[None, :] ** np.exp(theta)[:, None]
    P = safe_probs(P)

    ll = (y[None, :] * np.log(P) + (n[None, :] - y[None, :]) * np.log(1 - P)).sum(axis=1)

    log_unnorm = np.log(w) + ll
    m = np.max(log_unnorm)
    unnorm = np.exp(log_unnorm - m)
    post_w = unnorm / np.sum(unnorm)

    return post_w, P

def crm_posterior_summaries(sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=61):
    post_w, P = posterior_via_gh(sigma, skeleton, n_per_level, dlt_per_level, gh_n=gh_n)
    post_mean = (post_w[:, None] * P).sum(axis=0)
    overdose_prob = (post_w[:, None] * (P > target)).sum(axis=0)
    return post_mean, overdose_prob

def crm_choose_next(
    sigma, skeleton, n_per_level, dlt_per_level,
    current_level, target, alpha_overdose,
    max_step=1, gh_n=61,
    enforce_highest_tried_plus_one=True,
    highest_tried=None,
):
    post_mean, overdose_prob = crm_posterior_summaries(
        sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=gh_n
    )

    allowed = np.where(overdose_prob < alpha_overdose)[0]
    if allowed.size == 0:
        return 0, post_mean, overdose_prob

    k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

    # Step limit
    k_star = int(np.clip(k_star, current_level - max_step, current_level + max_step))

    # Guardrail: never recommend beyond highest tried + 1
    if enforce_highest_tried_plus_one and highest_tried is not None:
        k_star = int(min(k_star, highest_tried + 1))

    k_star = int(np.clip(k_star, 0, len(skeleton) - 1))
    return k_star, post_mean, overdose_prob

def crm_select_mtd(
    sigma, skeleton, n_per_level, dlt_per_level,
    target, alpha_overdose, gh_n=61,
    restrict_to_tried=True
):
    post_mean, overdose_prob = crm_posterior_summaries(
        sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=gh_n
    )

    allowed = np.where(overdose_prob < alpha_overdose)[0]
    if allowed.size == 0:
        return 0

    if restrict_to_tried:
        tried = np.where(np.asarray(n_per_level) > 0)[0]
        if tried.size > 0:
            allowed = np.intersect1d(allowed, tried)
            if allowed.size == 0:
                return int(tried[0])

    return int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

def run_crm(
    true_p, target, skeleton,
    sigma=1.0, start_level=0, max_n=36,
    cohort_size=6, alpha_overdose=0.25, max_step=1,
    already_n0=0, gh_n=61, rng=None,
    enforce_highest_tried_plus_one=True,
    restrict_final_mtd_to_tried=True,
    stop_if_dose0_likely_overdosing=False,
):
    if rng is None:
        rng = np.random.default_rng()

    n_levels = len(true_p)
    level = int(start_level)

    n_per_level = np.zeros(n_levels, dtype=int)
    dlt_per_level = np.zeros(n_levels, dtype=int)
    total_n = 0
    dose_path = []

    highest_tried = -1

    # Carry-in
    if already_n0 > 0:
        add = int(already_n0)
        n_per_level[level] += add
        total_n += add
        dose_path.extend([level] * add)
        highest_tried = max(highest_tried, level)

    while total_n < max_n:
        n_add = min(int(cohort_size), max_n - total_n)
        dose_path.extend([level] * n_add)

        out = simulate_bernoulli(n_add, true_p[level], rng)
        n_per_level[level] += n_add
        dlt_per_level[level] += int(out.sum())
        total_n += n_add
        highest_tried = max(highest_tried, level)

        if n_add < cohort_size:
            break

        if stop_if_dose0_likely_overdosing:
            _, od = crm_posterior_summaries(sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=gh_n)
            if od[0] >= alpha_overdose:
                level = 0
                break

        next_level, _, _ = crm_choose_next(
            sigma=sigma,
            skeleton=skeleton,
            n_per_level=n_per_level,
            dlt_per_level=dlt_per_level,
            current_level=level,
            target=target,
            alpha_overdose=alpha_overdose,
            max_step=max_step,
            gh_n=gh_n,
            enforce_highest_tried_plus_one=enforce_highest_tried_plus_one,
            highest_tried=highest_tried
        )
        level = next_level

    selected = crm_select_mtd(
        sigma=sigma,
        skeleton=skeleton,
        n_per_level=n_per_level,
        dlt_per_level=dlt_per_level,
        target=target,
        alpha_overdose=alpha_overdose,
        gh_n=gh_n,
        restrict_to_tried=restrict_final_mtd_to_tried
    )

    total_dlts = int(dlt_per_level.sum())
    return selected, n_per_level, total_dlts, dose_path

# ============================================================
# 6+3 (simple)
# ============================================================

def run_6plus3(true_p, start_level=0, max_n=36, accept_max_dlt=1, already_n0=0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    n_levels = len(true_p)
    level = int(start_level)

    n_per_level = np.zeros(n_levels, dtype=int)
    dlt_per_level = np.zeros(n_levels, dtype=int)
    total_n = 0
    last_acceptable = None
    dose_path = []

    if already_n0 > 0:
        add = int(already_n0)
        n_per_level[level] += add
        total_n += add
        dose_path.extend([level] * add)

    while total_n < max_n:
        n_add = min(6, max_n - total_n)
        dose_path.extend([level] * n_add)
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
            n_add2 = min(3, max_n - total_n)
            dose_path.extend([level] * n_add2)
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

        if level > 0:
            level -= 1
        break

    selected = 0 if last_acceptable is None else int(last_acceptable)
    total_dlts = int(dlt_per_level.sum())
    return selected, n_per_level, total_dlts, dose_path

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Dose Escalation Simulator: 6+3 vs CRM", layout="centered")
st.title("Dose Escalation Simulator: 6+3 vs CRM")
st.caption("Acute-only CRM (power model). Guardrail is implemented correctly: next ≤ highest tried + 1.")

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]

scenario_library = {
    "Default (MTD around level 2)": [0.05, 0.10, 0.20, 0.35, 0.55],
    "Safer overall (pushes higher)": [0.03, 0.06, 0.12, 0.20, 0.30],
    "More toxic overall (lands lower)": [0.08, 0.16, 0.28, 0.42, 0.60],
}

colA, colB = st.columns([1.05, 1.0])

with colA:
    st.subheader("Study setup")

    target = st.number_input("Target DLT probability", 0.05, 0.50, 0.25, 0.01)
    start_level = st.selectbox(
        "Start dose level", options=list(range(0, 5)), index=0,
        format_func=lambda i: f"Level {i} ({dose_labels[i]})"
    )
    already_n0 = st.number_input("Carry-in (already treated at start dose, 0 DLT)", 0, 20, 0, 1)

    n_sims = st.number_input("Number of simulated trials", 50, 5000, 200, 50)
    seed = st.number_input("Random seed", 1, 10_000_000, 12345, 1)

with colB:
    st.subheader("True scenario (ground truth)")
    scenario_name = st.selectbox("Scenario library", options=list(scenario_library.keys()), index=0)
    scenario_vals = scenario_library[scenario_name]

    manual_true = st.toggle("Manually edit the true DLT probabilities", value=True)
    true_p = []
    for i, lab in enumerate(dose_labels):
        default_val = float(scenario_vals[i])
        if manual_true:
            val = st.number_input(f"True P(DLT) at {lab}", 0.0, 1.0, default_val, 0.01, key=f"true_{i}")
        else:
            st.write(f"{lab}: {default_val:.2f}")
            val = default_val
        true_p.append(float(val))

    true_mtd = find_true_mtd(true_p, float(target))
    st.write(f"True MTD (closest to target {target:.2f}) = Level {true_mtd} ({dose_labels[true_mtd]})")

st.divider()

c1, c2 = st.columns([1.0, 1.0])

with c1:
    st.subheader("6+3 settings")
    max_n_6 = st.number_input("Max sample size (6+3)", 12, 120, 36, 3)
    accept_rule = st.selectbox("Acceptance rule after expansion to 9", options=[1, 2], index=0)

with c2:
    st.subheader("CRM settings")
    max_n_crm = st.number_input("Max sample size (CRM)", 12, 120, 36, 3)
    cohort_size = st.number_input("Cohort size (CRM)", 1, 12, 6, 1)

    prior_mode = st.radio("Skeleton mode", ["Auto (dfcrm getprior)", "Manual"], index=0)

    prior_mtd_idx = None
    if prior_mode == "Auto (dfcrm getprior)":
        prior_target = st.number_input("Prior target (for skeleton calibration)", 0.05, 0.50, float(target), 0.01)
        prior_halfwidth = st.number_input("Halfwidth (delta)", 0.01, 0.30, 0.10, 0.01)
        prior_nu = st.selectbox("Prior MTD dose level (nu, 1-based)", options=[1, 2, 3, 4, 5], index=2)
        prior_model = st.selectbox("Working model for getprior", options=["empiric", "logistic"], index=0)
        prior_intcpt = st.number_input("Logistic intercept (intcpt)", 0.0, 10.0, 3.0, 0.1)

        skeleton = dfcrm_getprior(
            halfwidth=float(prior_halfwidth),
            target=float(prior_target),
            nu=int(prior_nu),
            nlevel=5,
            model=str(prior_model),
            intcpt=float(prior_intcpt),
        ).tolist()
        prior_mtd_idx = int(prior_nu) - 1
        st.caption("Auto skeleton: " + ", ".join([f"{v:.3f}" for v in skeleton]))
    else:
        skeleton = []
        for i, lab in enumerate(dose_labels):
            v = st.number_input(f"Skeleton prior mean at {lab}", 0.01, 0.99, float(true_p[i]), 0.01, key=f"sk_{i}")
            skeleton.append(float(v))

    sigma = st.number_input("Prior sigma on theta", 0.2, 5.0, 1.0, 0.1)
    alpha_overdose = st.number_input("Overdose control alpha", 0.05, 0.50, 0.25, 0.01)
    max_step = st.selectbox("Max dose step per update", options=[1, 2], index=0)
    gh_n = st.selectbox("Gauss–Hermite points", options=[31, 41, 61, 81], index=2)

    enforce_guardrail = st.toggle("Guardrail: next dose ≤ highest tried + 1", value=True)
    restrict_final_mtd = st.toggle("Final MTD must be among tried doses", value=True)
    stop_if_dose0 = st.toggle("Stop if dose 0 likely overdosing (optional)", value=False)

st.subheader("Input curves (True vs Prior)")
fig, ax = plt.subplots(figsize=(3.8, 2.0), dpi=160)
x = np.arange(5)
ax.plot(x, true_p, marker="o", linewidth=1.4, label="True P(DLT)")
ax.plot(x, skeleton, marker="o", linewidth=1.4, label="Prior (skeleton)")
ax.axhline(float(target), linewidth=1, alpha=0.6)
ax.text(0.05, float(target) + 0.01, f"Target = {float(target):.2f}", fontsize=8)
ax.axvline(true_mtd, linewidth=1, alpha=0.25)
ax.text(true_mtd + 0.05, 0.90, "True MTD", fontsize=8, transform=ax.get_xaxis_transform())
if prior_mtd_idx is not None:
    ax.axvline(prior_mtd_idx, linewidth=1, alpha=0.25)
    ax.text(prior_mtd_idx + 0.05, 0.82, "Prior MTD", fontsize=8, transform=ax.get_xaxis_transform())
ax.set_xticks(x)
ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
ax.set_ylabel("Probability", fontsize=9)
ymax = max(max(true_p), max(skeleton), float(target))
ax.set_ylim(0, min(1.0, ymax * 1.25 + 0.02))
compact_style(ax)
ax.legend(fontsize=8, frameon=False, loc="upper left")
st.pyplot(fig, clear_figure=True)

st.divider()

run = st.button("Run simulations")

if run:
    rng = np.random.default_rng(int(seed))
    ns = int(n_sims)

    sel_6 = np.zeros(5, dtype=int)
    sel_c = np.zeros(5, dtype=int)
    nmat_6 = np.zeros((ns, 5), dtype=int)
    nmat_c = np.zeros((ns, 5), dtype=int)

    for s in range(ns):
        chosen6, n6, _, _ = run_6plus3(
            true_p=true_p,
            start_level=int(start_level),
            max_n=int(max_n_6),
            accept_max_dlt=int(accept_rule),
            already_n0=int(already_n0),
            rng=rng
        )
        chosenc, nc, _, _ = run_crm(
            true_p=true_p,
            target=float(target),
            skeleton=skeleton,
            sigma=float(sigma),
            start_level=int(start_level),
            max_n=int(max_n_crm),
            cohort_size=int(cohort_size),
            alpha_overdose=float(alpha_overdose),
            max_step=int(max_step),
            already_n0=int(already_n0),
            gh_n=int(gh_n),
            rng=rng,
            enforce_highest_tried_plus_one=bool(enforce_guardrail),
            restrict_final_mtd_to_tried=bool(restrict_final_mtd),
            stop_if_dose0_likely_overdosing=bool(stop_if_dose0),
        )

        sel_6[chosen6] += 1
        sel_c[chosenc] += 1
        nmat_6[s, :] = n6
        nmat_c[s, :] = nc

    p_sel_6 = sel_6 / float(ns)
    p_sel_c = sel_c / float(ns)
    avg_n6 = np.mean(nmat_6, axis=0)
    avg_nc = np.mean(nmat_c, axis=0)

    st.subheader("Results")
    xx = np.arange(5)
    width = 0.38
    r1, r2 = st.columns(2)

    with r1:
        fig, ax = plt.subplots(figsize=(4.2, 2.2), dpi=160)
        ax.bar(xx - width/2, p_sel_6, width, label="6+3")
        ax.bar(xx + width/2, p_sel_c, width, label="CRM")
        ax.set_title("Probability of selecting each dose as MTD", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, max(p_sel_6.max(), p_sel_c.max()) * 1.15 + 1e-6)
        compact_style(ax)
        ax.axvline(true_mtd, linewidth=1, alpha=0.6)
        ax.text(true_mtd + 0.05, ax.get_ylim()[1] * 0.92, "True MTD", fontsize=8)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with r2:
        fig, ax = plt.subplots(figsize=(4.2, 2.2), dpi=160)
        ax.bar(xx - width/2, avg_n6, width, label="6+3")
        ax.bar(xx + width/2, avg_nc, width, label="CRM")
        ax.set_title("Average number treated per dose level", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Patients", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    st.caption("If CRM still sticks at level 0, toggle OFF the guardrail and re-run once. If it then escalates, the problem was your guardrail logic.")
