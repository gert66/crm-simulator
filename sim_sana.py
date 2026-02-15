import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# Helpers
# ============================================================

def safe_probs(x):
    x = np.asarray(x, dtype=float)
    return np.clip(x, 1e-6, 1 - 1e-6)

def simulate_bernoulli(n, p, rng):
    return rng.binomial(1, p, size=int(n))

def find_true_mtd(true_p, target):
    true_p = np.asarray(true_p, dtype=float)
    return int(np.argmin(np.abs(true_p - target)))

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)

# ============================================================
# dfcrm getprior port
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
# CRM posterior via Gauss–Hermite quadrature (acute-only)
# p_k(theta) = skeleton_k ^ exp(theta), theta ~ N(0, sigma^2)
# ============================================================

def posterior_via_gh(sigma, skeleton, n_per_level, dlt_per_level, gh_n=61):
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
    current_level, target,
    ewoc_alpha=None,
    max_step=1, gh_n=61,
    enforce_highest_tried_plus_one=True,
    highest_tried=None,
):
    post_mean, overdose_prob = crm_posterior_summaries(
        sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=gh_n
    )

    if ewoc_alpha is None:
        allowed = np.arange(len(skeleton))
    else:
        allowed = np.where(overdose_prob < float(ewoc_alpha))[0]

    if allowed.size == 0:
        return 0, post_mean, overdose_prob, allowed

    k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

    # Step limiting
    k_star = int(np.clip(k_star, current_level - int(max_step), current_level + int(max_step)))

    # Guardrail
    if enforce_highest_tried_plus_one and highest_tried is not None:
        k_star = int(min(k_star, int(highest_tried) + 1))

    k_star = int(np.clip(k_star, 0, len(skeleton) - 1))
    return k_star, post_mean, overdose_prob, allowed

def crm_select_mtd(
    sigma, skeleton, n_per_level, dlt_per_level,
    target, ewoc_alpha=None, gh_n=61,
    restrict_to_tried=True
):
    post_mean, overdose_prob = crm_posterior_summaries(
        sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=gh_n
    )

    if ewoc_alpha is None:
        allowed = np.arange(len(skeleton))
    else:
        allowed = np.where(overdose_prob < float(ewoc_alpha))[0]

    if allowed.size == 0:
        return 0

    if restrict_to_tried:
        tried = np.where(np.asarray(n_per_level) > 0)[0]
        if tried.size > 0:
            allowed = np.intersect1d(allowed, tried)
            if allowed.size == 0:
                return int(tried[0])

    return int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

def run_crm_trial(
    true_p, target, skeleton,
    sigma=1.0,
    start_level=1,
    max_n=27,
    cohort_size=3,
    max_step=1,
    gh_n=61,
    enforce_guardrail=True,
    restrict_final_mtd_to_tried=True,
    ewoc_on=False,
    ewoc_alpha=0.25,
    burn_in_until_first_dlt=True,
    rng=None,
    debug=False
):
    if rng is None:
        rng = np.random.default_rng()

    true_p = np.asarray(true_p, dtype=float)
    n_levels = len(true_p)

    level = int(start_level)
    n_per = np.zeros(n_levels, dtype=int)
    y_per = np.zeros(n_levels, dtype=int)
    dose_path = []

    highest_tried = -1
    any_dlt_seen = False
    debug_rows = []

    while int(n_per.sum()) < int(max_n):
        n_add = min(int(cohort_size), int(max_n) - int(n_per.sum()))
        out = simulate_bernoulli(n_add, true_p[level], rng)

        n_per[level] += n_add
        y_per[level] += int(out.sum())
        dose_path.extend([level] * n_add)
        highest_tried = max(highest_tried, level)

        if int(out.sum()) > 0:
            any_dlt_seen = True

        if debug:
            debug_rows.append({
                "treated_level": level,
                "cohort_n": int(n_add),
                "cohort_dlts": int(out.sum()),
                "any_dlt_seen": bool(any_dlt_seen),
            })

        if n_add < int(cohort_size):
            break

        # R-like burn-in (simplified): escalate until first DLT
        if burn_in_until_first_dlt and (not any_dlt_seen):
            if level < n_levels - 1:
                level = level + 1
            continue

        ewoc_alpha_eff = float(ewoc_alpha) if ewoc_on else None
        next_level, post_mean, od_prob, allowed = crm_choose_next(
            sigma=sigma,
            skeleton=skeleton,
            n_per_level=n_per,
            dlt_per_level=y_per,
            current_level=level,
            target=target,
            ewoc_alpha=ewoc_alpha_eff,
            max_step=max_step,
            gh_n=gh_n,
            enforce_highest_tried_plus_one=enforce_guardrail,
            highest_tried=highest_tried
        )

        if debug:
            debug_rows[-1].update({
                "next_level": int(next_level),
                "allowed_levels": ",".join([str(int(a)) for a in allowed]),
                "post_mean": [float(x) for x in post_mean],
                "od_prob": [float(x) for x in od_prob],
                "highest_tried": int(highest_tried),
            })

        level = int(next_level)

    ewoc_alpha_eff = float(ewoc_alpha) if ewoc_on else None
    selected = crm_select_mtd(
        sigma=sigma,
        skeleton=skeleton,
        n_per_level=n_per,
        dlt_per_level=y_per,
        target=target,
        ewoc_alpha=ewoc_alpha_eff,
        gh_n=gh_n,
        restrict_to_tried=restrict_final_mtd_to_tried
    )

    return int(selected), n_per, int(y_per.sum()), dose_path, debug_rows

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="6+3 vs CRM (R-aligned defaults)", layout="wide")
st.title("Dose Escalation Simulator: 6+3 vs CRM (acute-only)")
st.caption("UI is organized around a prior playground. Advanced knobs are tucked away.")

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]

# R-aligned defaults (acute)
R_N_PATIENT = 27
R_COHORT = 3
PY_START_LEVEL = 1       # R p <- 2 (1-based) -> Python 1 (0-based)
R_TARGET_ACUTE = 0.15    # from your R example
R_PRIOR_TARGET = 0.15
R_HALF_WIDTH = 0.10
R_PRIOR_NU = 3
DEFAULT_SIGMA = 1.0      # dfcrm default not explicit in your script; baseline

default_true_p = [0.01, 0.02, 0.12, 0.20, 0.35]

# ------------------------------------------------------------
# Top row: essentials
# ------------------------------------------------------------
st.subheader("Essentials (R defaults)")

e1, e2, e3, e4 = st.columns([1.0, 1.0, 1.0, 1.0], gap="large")

with e1:
    target = st.number_input(
        "Study target (acute)",
        min_value=0.05, max_value=0.50, value=float(R_TARGET_ACUTE), step=0.01,
        help=(
            "Target toxicity used by CRM.\n\n"
            "R reference: target.acute = 0.15 in your example."
        )
    )

with e2:
    start_level = st.selectbox(
        "Start dose",
        options=list(range(0, 5)),
        index=int(PY_START_LEVEL),
        format_func=lambda i: f"Level {i} ({dose_labels[i]})",
        help=(
            "Starting dose.\n\n"
            "R reference: p <- 2 (1-based), so default is Level 1 (0-based) = 5×5."
        )
    )

with e3:
    max_n = st.number_input(
        "Max sample size",
        min_value=12, max_value=200, value=int(R_N_PATIENT), step=3,
        help="R reference: N.patient = 27."
    )

with e4:
    cohort_size = st.number_input(
        "CRM cohort size",
        min_value=1, max_value=12, value=int(R_COHORT), step=1,
        help="R reference: CO = 3."
    )

st.divider()

# ------------------------------------------------------------
# Main workbench: True curve + Prior playground + Plot
# ------------------------------------------------------------
left, right = st.columns([1.05, 1.25], gap="large")

with left:
    st.subheader("True acute DLT curve (data generating)")
    manual_true = st.toggle("Edit true curve", value=True)
    true_p = []
    for i, lab in enumerate(dose_labels):
        val = st.number_input(
            f"True P(DLT) at {lab}",
            min_value=0.0, max_value=1.0,
            value=float(default_true_p[i]),
            step=0.01,
            disabled=(not manual_true),
            key=f"true_{i}",
            help="Ground truth used to simulate Bernoulli DLT outcomes."
        )
        true_p.append(float(val))

    true_mtd = find_true_mtd(true_p, float(target))
    st.info(f"True MTD (closest to target {float(target):.2f}) = Level {true_mtd} ({dose_labels[true_mtd]})")

    st.subheader("Key CRM knobs (play with these)")
    sigma = st.slider(
        "Prior sigma on theta",
        min_value=0.2, max_value=5.0, value=float(DEFAULT_SIGMA), step=0.1,
        help=(
            "Controls prior strength. Higher sigma makes the prior looser.\n\n"
            "R reference: sigma not explicitly set in your script (package default)."
        )
    )

    burn_in = st.toggle(
        "Burn-in until first DLT (R-like)",
        value=True,
        help=(
            "Simplified mimic of the R script burning phase.\n"
            "We escalate cohort-wise until the first DLT is observed, then CRM decisions start."
        )
    )

    ewoc_on = st.toggle(
        "Enable EWOC overdose control",
        value=False,
        help=(
            "Admissibility rule: P(p_k > target | data) < alpha.\n\n"
            "R reference: no explicit overdose alpha is shown in your titecrm call."
        )
    )

    ewoc_alpha = st.slider(
        "EWOC alpha",
        min_value=0.05, max_value=0.99, value=0.25, step=0.01,
        disabled=(not ewoc_on),
        help="Only used if EWOC is enabled."
    )

with right:
    st.subheader("Prior playground (dfcrm getprior)")

    p1, p2 = st.columns([1.0, 1.0], gap="medium")

    with p1:
        prior_model = st.radio(
            "Skeleton model",
            options=["empiric", "logistic"],
            index=0,
            horizontal=True,
            help=(
                "How the skeleton is generated.\n\n"
                "R reference: getprior(...) is called without explicit model in your snippet. "
                "We default to empiric here for stability."
            )
        )

        prior_target = st.slider(
            "Prior target (skeleton calibration)",
            min_value=0.05, max_value=0.50, value=float(R_PRIOR_TARGET), step=0.01,
            help="R reference: prior.target.acute = 0.15."
        )

        halfwidth = st.slider(
            "Halfwidth (delta)",
            min_value=0.01, max_value=0.30, value=float(R_HALF_WIDTH), step=0.01,
            help="R reference: getprior(halfwidth = 0.1, ...)."
        )

        prior_nu = st.slider(
            "Prior MTD (nu, 1-based)",
            min_value=1, max_value=5, value=int(R_PRIOR_NU), step=1,
            help="R reference: prior.MTD.acute = 3 (1-based)."
        )

        logistic_intcpt = st.slider(
            "Logistic intercept (only if logistic)",
            min_value=0.0, max_value=10.0, value=3.0, step=0.1,
            help="Only relevant if you choose logistic."
        )

        skeleton = dfcrm_getprior(
            halfwidth=float(halfwidth),
            target=float(prior_target),
            nu=int(prior_nu),
            nlevel=5,
            model=str(prior_model),
            intcpt=float(logistic_intcpt),
        ).tolist()

        st.caption("Skeleton: " + ", ".join([f"{v:.3f}" for v in skeleton]))

    with p2:
        st.markdown("**Preview (True vs Prior)**")
        fig, ax = plt.subplots(figsize=(5.4, 2.6), dpi=160)
        x = np.arange(5)
        ax.plot(x, true_p, marker="o", linewidth=1.6, label="True P(DLT)")
        ax.plot(x, skeleton, marker="o", linewidth=1.6, label="Prior (skeleton)")
        ax.axhline(float(target), linewidth=1, alpha=0.6)
        ax.text(0.05, float(target) + 0.01, f"Target = {float(target):.2f}", fontsize=8)
        ax.axvline(true_mtd, linewidth=1, alpha=0.35)
        ax.text(true_mtd + 0.05, 0.92, "True MTD", fontsize=8, transform=ax.get_xaxis_transform())
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, min(1.0, max(max(true_p), max(skeleton), float(target)) * 1.25 + 0.02))
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        st.pyplot(fig, clear_figure=True)

st.divider()

# ------------------------------------------------------------
# Advanced settings tucked away
# ------------------------------------------------------------
with st.expander("Advanced settings", expanded=False):
    a1, a2, a3 = st.columns([1.0, 1.0, 1.0], gap="large")

    with a1:
        n_sims = st.number_input(
            "Number of simulated trials",
            min_value=50, max_value=5000, value=500, step=50,
            help="More trials = smoother estimates, slower runtime."
        )
        seed = st.number_input(
            "Random seed",
            min_value=1, max_value=10_000_000, value=123, step=1,
            help="Controls reproducibility."
        )

    with a2:
        gh_n = st.selectbox(
            "Gauss–Hermite points",
            options=[31, 41, 61, 81],
            index=2,
            help="Accuracy/speed trade-off for posterior integration."
        )
        max_step = st.selectbox(
            "Max dose step per update",
            options=[1, 2],
            index=0,
            help="Dose movement limit per CRM update."
        )

    with a3:
        enforce_guardrail = st.toggle(
            "Guardrail: next dose ≤ highest tried + 1",
            value=True,
            help="Prevents skipping over untried doses."
        )
        restrict_final_mtd = st.toggle(
            "Final MTD must be among tried doses",
            value=True,
            help="Final selection is limited to doses that were treated."
        )

    show_debug = st.toggle(
        "Show CRM decision debug (first simulated trial)",
        value=False,
        help="Prints admissible set and posterior summaries per update for the first trial."
    )

# ------------------------------------------------------------
# Run simulations
# ------------------------------------------------------------
run = st.button("Run simulations")

if run:
    rng = np.random.default_rng(int(seed))
    ns = int(n_sims)

    sel_c = np.zeros(5, dtype=int)
    nmat_c = np.zeros((ns, 5), dtype=int)

    debug_dump = None

    for s in range(ns):
        debug_flag = bool(show_debug and s == 0)
        chosenc, nc, _, _, dbg = run_crm_trial(
            true_p=true_p,
            target=float(target),
            skeleton=skeleton,
            sigma=float(sigma),
            start_level=int(start_level),
            max_n=int(max_n),
            cohort_size=int(cohort_size),
            max_step=int(max_step),
            gh_n=int(gh_n),
            enforce_guardrail=bool(enforce_guardrail),
            restrict_final_mtd_to_tried=bool(restrict_final_mtd),
            ewoc_on=bool(ewoc_on),
            ewoc_alpha=float(ewoc_alpha),
            burn_in_until_first_dlt=bool(burn_in),
            rng=rng,
            debug=debug_flag
        )

        sel_c[chosenc] += 1
        nmat_c[s, :] = nc

        if debug_flag:
            debug_dump = dbg

    p_sel_c = sel_c / float(ns)
    avg_nc = np.mean(nmat_c, axis=0)

    st.subheader("Results (CRM)")
    r1, r2 = st.columns([1.0, 1.0], gap="large")

    with r1:
        fig, ax = plt.subplots(figsize=(5.6, 2.7), dpi=160)
        xx = np.arange(5)
        ax.bar(xx, p_sel_c)
        ax.set_title("Probability of selecting each dose as MTD", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, p_sel_c.max() * 1.15 + 1e-6)
        ax.axvline(true_mtd, linewidth=1, alpha=0.6)
        ax.text(true_mtd + 0.05, ax.get_ylim()[1] * 0.92, "True MTD", fontsize=8)
        compact_style(ax)
        st.pyplot(fig, clear_figure=True)

    with r2:
        fig, ax = plt.subplots(figsize=(5.6, 2.7), dpi=160)
        xx = np.arange(5)
        ax.bar(xx, avg_nc)
        ax.set_title("Average number treated per dose level", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Patients", fontsize=9)
        compact_style(ax)
        st.pyplot(fig, clear_figure=True)

    if show_debug and debug_dump:
        st.subheader("CRM debug (first simulated trial)")
        for i, row in enumerate(debug_dump, start=1):
            st.write(f"Update {i}: treated L{row['treated_level']} | n={row['cohort_n']} | dlts={row['cohort_dlts']} | any_dlt_seen={row['any_dlt_seen']}")
            if "next_level" in row:
                st.write(f"  allowed: {row['allowed_levels']} | next: L{row['next_level']} | highest_tried={row['highest_tried']}")
                st.write(f"  post_mean: {[round(v,3) for v in row['post_mean']]}")
                st.write(f"  od_prob:   {[round(v,3) for v in row['od_prob']]}")
