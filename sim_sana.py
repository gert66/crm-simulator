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
# dfcrm-style skeleton calibration (getprior) | close translation
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
# Ping-pong / oscillation metrics
# ============================================================

def switch_rate(path):
    if path is None or len(path) < 2:
        return 0.0
    path = np.asarray(path, dtype=int)
    return float(np.mean(path[1:] != path[:-1]))

def mean_step_size(path):
    if path is None or len(path) < 2:
        return 0.0
    path = np.asarray(path, dtype=int)
    return float(np.mean(np.abs(path[1:] - path[:-1])))

def oscillation_index(path):
    """
    Fraction of length-3 windows that look like A -> B -> A (with A != B).
    Example: 0,1,0 counts as oscillation.
    """
    if path is None or len(path) < 3:
        return 0.0
    path = np.asarray(path, dtype=int)
    a = path[:-2]
    b = path[1:-1]
    c = path[2:]
    osc = (a == c) & (a != b)
    return float(np.mean(osc))

# ============================================================
# DLT burden metrics per trial
# ============================================================

def observed_dlt_rate(total_dlts, n_total):
    if n_total <= 0:
        return 0.0
    return float(total_dlts) / float(n_total)

def expected_dlt_risk(n_per_level, true_p):
    """
    Expected DLT probability for a random patient in this trial,
    using true_p as ground truth.
    """
    n_per_level = np.asarray(n_per_level, dtype=float)
    true_p = np.asarray(true_p, dtype=float)
    n_total = float(n_per_level.sum())
    if n_total <= 0:
        return 0.0
    return float(np.sum(n_per_level * true_p) / n_total)

# ============================================================
# 6+3 (simple)
# ============================================================

def run_6plus3(true_p, start_level=1, max_n=36, accept_max_dlt=1, already_n0=0, rng=None):
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
# CRM (acute-only): Gauss–Hermite quadrature
# Power CRM model: p_k(theta) = skeleton_k ** exp(theta)
# theta ~ N(0, sigma^2)
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
    current_level, target, alpha_overdose,
    max_step=1, gh_n=61,
    enforce_highest_tried_plus_one=True,
    highest_tried=None,
):
    """
    Decision rule:
    1) compute posterior mean toxicity per dose
    2) compute posterior overdose probability per dose: P(p_k > target | data)
    3) admissible doses satisfy overdose_prob < alpha_overdose
    4) apply guardrail and max_step as filters on the admissible set
    5) pick dose with posterior mean closest to target (tie-break to lower dose)
    """
    post_mean, overdose_prob = crm_posterior_summaries(
        sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=gh_n
    )

    K = len(skeleton)
    doses = np.arange(K)

    admiss = doses[overdose_prob < alpha_overdose]

    if enforce_highest_tried_plus_one and highest_tried is not None:
        admiss = admiss[admiss <= min(K - 1, int(highest_tried) + 1)]

    if max_step is not None and max_step >= 0:
        lo = max(0, int(current_level) - int(max_step))
        hi = min(K - 1, int(current_level) + int(max_step))
        admiss = admiss[(admiss >= lo) & (admiss <= hi)]

    if admiss.size == 0:
        k_star = int(np.clip(current_level, 0, K - 1))
        return k_star, post_mean, overdose_prob, admiss

    dist = np.abs(post_mean[admiss] - target)
    k_star = int(admiss[np.lexsort((admiss, dist))[0]])
    return k_star, post_mean, overdose_prob, admiss

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
    sigma=1.0, start_level=1, max_n=36,
    cohort_size=3, alpha_overdose=0.25, max_step=1,
    already_n0=0, gh_n=61, rng=None,
    enforce_highest_tried_plus_one=True,
    restrict_final_mtd_to_tried=True,
    stop_if_dose0_likely_overdosing=False,
    stop_threshold=0.90,
    debug=False,
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
    debug_log = []

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
            if float(od[0]) >= float(stop_threshold):
                if debug:
                    debug_log.append({
                        "cohort_end_n": int(total_n),
                        "current_level": int(level),
                        "decision": "STOP (dose 0 very likely overdosing)",
                        "overdose_prob_dose0": float(od[0]),
                    })
                level = 0
                break

        next_level, post_mean, overdose_prob, admiss = crm_choose_next(
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

        if debug:
            debug_log.append({
                "cohort_end_n": int(total_n),
                "current_level": int(level),
                "highest_tried": int(highest_tried),
                "next_level": int(next_level),
                "post_mean": post_mean.tolist(),
                "overdose_prob": overdose_prob.tolist(),
                "admissible_after_constraints": admiss.tolist(),
            })

        level = int(next_level)

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
    return selected, n_per_level, total_dlts, dose_path, debug_log

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Dose Escalation Simulator: 6+3 vs CRM", layout="centered")
st.title("Dose Escalation Simulator: 6+3 vs CRM")
st.caption("Acute-only CRM (power model). Includes oscillation and DLT burden metrics.")

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]

scenario_library = {
    "Default (MTD around level 2)": [0.05, 0.10, 0.20, 0.35, 0.55],
    "Safer overall (pushes higher)": [0.03, 0.06, 0.12, 0.20, 0.30],
    "More toxic overall (lands lower)": [0.08, 0.16, 0.28, 0.42, 0.60],
}

colA, colB = st.columns([1.05, 1.0])

with colA:
    st.subheader("Study setup")

    target = st.number_input(
        "Target DLT probability",
        0.05, 0.50, 0.25, 0.01,
        help="The target acute DLT rate that defines the MTD. CRM selects the dose with posterior mean toxicity closest to this target, while applying overdose control."
    )

    start_level = st.selectbox(
        "Start dose level",
        options=list(range(0, 5)),
        index=1,
        format_func=lambda i: f"Level {i} ({dose_labels[i]})",
        help="Starting dose level for both designs. Default is Level 1 (5×5 Gy) to align better with the R script's initial dose choice."
    )

    already_n0 = st.number_input(
        "Carry-in (already treated at start dose, 0 DLT)",
        0, 20, 0, 1,
        help="Adds patients at the starting dose with assumed 0 DLTs. Useful to mimic run-in patients already treated before the simulated trial starts."
    )

    n_sims = st.number_input(
        "Number of simulated trials",
        50, 5000, 200, 50,
        help="Number of independent trials to simulate. Results are Monte Carlo estimates across these trials."
    )

    seed = st.number_input(
        "Random seed",
        1, 10_000_000, 12345, 1,
        help="Seed for the random number generator to make simulations reproducible."
    )

with colB:
    st.subheader("True scenario (ground truth)")

    scenario_name = st.selectbox(
        "Scenario library",
        options=list(scenario_library.keys()),
        index=0,
        help="Select a preset true toxicity curve. You can also edit values manually below."
    )
    scenario_vals = scenario_library[scenario_name]

    manual_true = st.toggle(
        "Manually edit the true DLT probabilities",
        value=True,
        help="If enabled, you can set the true P(DLT) per dose. These are used to generate DLT outcomes in simulation."
    )

    true_p = []
    for i, lab in enumerate(dose_labels):
        default_val = float(scenario_vals[i])
        if manual_true:
            val = st.number_input(
                f"True P(DLT) at {lab}",
                0.0, 1.0, default_val, 0.01, key=f"true_{i}",
                help="Ground truth acute DLT probability at this dose level."
            )
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

    max_n_6 = st.number_input(
        "Max sample size (6+3)",
        12, 120, 36, 3,
        help="Maximum number of patients allowed in the 6+3 design."
    )

    accept_rule = st.selectbox(
        "Acceptance rule after expansion to 9",
        options=[1, 2], index=0,
        help="If 1 DLT in first 6, treat 3 more. Accept the dose if total DLTs among 9 is ≤ this value."
    )

with c2:
    st.subheader("CRM settings")

    max_n_crm = st.number_input(
        "Max sample size (CRM)",
        12, 120, 36, 3,
        help="Maximum number of patients allowed in the CRM trial."
    )

    cohort_size = st.number_input(
        "Cohort size (CRM)",
        1, 12, 3, 1,
        help="Number of patients treated per CRM update. R script uses CO=3, so default is 3."
    )

    prior_mode = st.radio(
        "Skeleton mode",
        ["Auto (dfcrm getprior)", "Manual"],
        index=0,
        help="Auto uses a dfcrm-style skeleton calibration. Manual lets you specify skeleton probabilities directly."
    )

    prior_mtd_idx = None
    if prior_mode == "Auto (dfcrm getprior)":
        prior_target = st.number_input(
            "Prior target (for skeleton calibration)",
            0.05, 0.50, float(target), 0.01,
            help="Target toxicity used only to build the skeleton curve (prior means) via getprior."
        )
        prior_halfwidth = st.number_input(
            "Halfwidth (delta)",
            0.01, 0.30, 0.10, 0.01,
            help="Controls spread of the skeleton around the prior target. Smaller means a tighter skeleton curve."
        )
        prior_nu = st.selectbox(
            "Prior MTD dose level (nu, 1-based)",
            options=[1, 2, 3, 4, 5], index=2,
            help="The dose level that is assumed to be near the target in the prior skeleton calibration."
        )
        prior_model = st.selectbox(
            "Working model for getprior",
            options=["empiric", "logistic"], index=0,
            help="Skeleton calibration model. Empiric is commonly used with dfcrm; logistic uses an intercept plus scaled doses."
        )
        prior_intcpt = st.number_input(
            "Logistic intercept (intcpt)",
            0.0, 10.0, 3.0, 0.1,
            help="Only used when getprior model is logistic."
        )

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
            v = st.number_input(
                f"Skeleton prior mean at {lab}",
                0.01, 0.99, float(true_p[i]), 0.01, key=f"sk_{i}",
                help="Prior mean toxicity for this dose in the CRM model. Must be between 0 and 1."
            )
            skeleton.append(float(v))

    sigma = st.number_input(
        "Prior sigma on theta",
        0.2, 5.0, 1.0, 0.1,
        help="Prior SD for theta in the power CRM: theta ~ N(0, sigma^2). Larger sigma gives a less informative prior."
    )

    alpha_overdose = st.number_input(
        "Overdose control alpha",
        0.05, 0.50, 0.25, 0.01,
        help="Admissibility rule: dose k is allowed if P(p_k > target | data) < alpha."
    )

    max_step = st.selectbox(
        "Max dose step per update",
        options=[1, 2], index=0,
        help="Limits how far the next cohort can move from the current dose in number of dose levels."
    )

    gh_n = st.selectbox(
        "Gauss–Hermite points",
        options=[31, 41, 61, 81], index=2,
        help="Number of Gauss–Hermite quadrature points used for posterior integration. More points increases accuracy but also runtime."
    )

    enforce_guardrail = st.toggle(
        "Guardrail: next dose ≤ highest tried + 1",
        value=True,
        help="Prevents skipping ahead to never-tried doses. Next dose cannot exceed the highest tried dose plus one."
    )

    restrict_final_mtd = st.toggle(
        "Final MTD must be among tried doses",
        value=True,
        help="If enabled, the final MTD is selected only among doses that were actually given to at least one patient."
    )

    stop_if_dose0 = st.toggle(
        "Stop if dose 0 very likely overdosing (optional)",
        value=False,
        help="Optional early safety stop. Stops if dose 0 has very high posterior probability of exceeding the target."
    )

    stop_threshold = st.number_input(
        "Safety stop threshold at dose 0",
        0.50, 0.99, 0.90, 0.01,
        help="Only used if the safety stop is enabled. Trial stops if P(p0 > target | data) >= this value."
    )

    show_crm_debug = st.toggle(
        "Show CRM decision debug (first simulated trial)",
        value=False,
        help="If enabled, shows posterior mean toxicity, overdose probabilities, and admissible dose set after each cohort in the first simulated CRM trial."
    )

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

show_quality_plots = st.toggle(
    "Show quality distribution plots (oscillation, DLT burden)",
    value=True,
    help="If enabled, shows histograms for oscillation and DLT burden metrics across simulated trials."
)

run = st.button("Run simulations")

if run:
    rng = np.random.default_rng(int(seed))
    ns = int(n_sims)

    sel_6 = np.zeros(5, dtype=int)
    sel_c = np.zeros(5, dtype=int)

    nmat_6 = np.zeros((ns, 5), dtype=int)
    nmat_c = np.zeros((ns, 5), dtype=int)

    tot_dlt_6 = np.zeros(ns, dtype=int)
    tot_dlt_c = np.zeros(ns, dtype=int)

    # Quality metrics
    ping_sw_6 = np.zeros(ns, dtype=float)
    ping_osc_6 = np.zeros(ns, dtype=float)
    ping_step_6 = np.zeros(ns, dtype=float)

    ping_sw_c = np.zeros(ns, dtype=float)
    ping_osc_c = np.zeros(ns, dtype=float)
    ping_step_c = np.zeros(ns, dtype=float)

    obs_rate_6 = np.zeros(ns, dtype=float)
    obs_rate_c = np.zeros(ns, dtype=float)

    exp_risk_6 = np.zeros(ns, dtype=float)
    exp_risk_c = np.zeros(ns, dtype=float)

    first_debug_log = None

    for s in range(ns):
        chosen6, n6, d6, path6 = run_6plus3(
            true_p=true_p,
            start_level=int(start_level),
            max_n=int(max_n_6),
            accept_max_dlt=int(accept_rule),
            already_n0=int(already_n0),
            rng=rng
        )

        chosenc, nc, dc, pathc, debug_log = run_crm(
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
            stop_threshold=float(stop_threshold),
            debug=(show_crm_debug and s == 0),
        )

        if show_crm_debug and s == 0:
            first_debug_log = debug_log

        sel_6[chosen6] += 1
        sel_c[chosenc] += 1

        nmat_6[s, :] = n6
        nmat_c[s, :] = nc

        tot_dlt_6[s] = d6
        tot_dlt_c[s] = dc

        ping_sw_6[s] = switch_rate(path6)
        ping_osc_6[s] = oscillation_index(path6)
        ping_step_6[s] = mean_step_size(path6)

        ping_sw_c[s] = switch_rate(pathc)
        ping_osc_c[s] = oscillation_index(pathc)
        ping_step_c[s] = mean_step_size(pathc)

        n6_total = int(np.sum(n6))
        nc_total = int(np.sum(nc))

        obs_rate_6[s] = observed_dlt_rate(d6, n6_total)
        obs_rate_c[s] = observed_dlt_rate(dc, nc_total)

        exp_risk_6[s] = expected_dlt_risk(n6, true_p)
        exp_risk_c[s] = expected_dlt_risk(nc, true_p)

    # Aggregates
    p_sel_6 = sel_6 / float(ns)
    p_sel_c = sel_c / float(ns)

    avg_n6 = np.mean(nmat_6, axis=0)
    avg_nc = np.mean(nmat_c, axis=0)

    mean_total_dlt_6 = float(np.mean(tot_dlt_6))
    mean_total_dlt_c = float(np.mean(tot_dlt_c))

    mean_n_6 = float(np.mean(nmat_6.sum(axis=1)))
    mean_n_c = float(np.mean(nmat_c.sum(axis=1)))

    mean_obs_6 = float(np.mean(obs_rate_6))
    mean_obs_c = float(np.mean(obs_rate_c))

    mean_exp_6 = float(np.mean(exp_risk_6))
    mean_exp_c = float(np.mean(exp_risk_c))

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

    r3, r4 = st.columns(2)
    with r3:
        fig, ax = plt.subplots(figsize=(4.2, 2.1), dpi=160)
        labels = ["6+3", "CRM"]
        obs = [mean_obs_6, mean_obs_c]
        expv = [mean_exp_6, mean_exp_c]
        xi = np.arange(len(labels))
        w = 0.38
        ax.bar(xi - w/2, obs, w, label="Observed DLT rate")
        ax.bar(xi + w/2, expv, w, label="Expected DLT risk")
        ax.set_title("Average DLT burden per patient", fontsize=10)
        ax.set_xticks(xi)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Rate", fontsize=9)
        ax.set_ylim(0, max(max(obs), max(expv)) * 1.20 + 1e-6)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with r4:
        st.markdown("**Summary (averages over simulated trials)**")
        st.write(f"Mean sample size: 6+3 = {mean_n_6:.1f} | CRM = {mean_n_c:.1f}")
        st.write(f"Mean total DLTs: 6+3 = {mean_total_dlt_6:.2f} | CRM = {mean_total_dlt_c:.2f}")
        st.write(f"Mean observed DLT rate: 6+3 = {mean_obs_6:.3f} | CRM = {mean_obs_c:.3f}")
        st.write(f"Mean expected DLT risk: 6+3 = {mean_exp_6:.3f} | CRM = {mean_exp_c:.3f}")
        st.write("")
        st.write(f"Switch rate: 6+3 = {np.mean(ping_sw_6):.3f} | CRM = {np.mean(ping_sw_c):.3f}")
        st.write(f"Oscillation index (A→B→A): 6+3 = {np.mean(ping_osc_6):.3f} | CRM = {np.mean(ping_osc_c):.3f}")
        st.write(f"Mean step size: 6+3 = {np.mean(ping_step_6):.3f} | CRM = {np.mean(ping_step_c):.3f}")

    if show_quality_plots:
        st.subheader("Quality distributions")

        bins01 = np.linspace(0, 1, 21)

        cA, cB = st.columns(2)
        with cA:
            fig, ax = plt.subplots(figsize=(4.2, 2.0), dpi=160)
            ax.hist(ping_osc_6, bins=bins01, alpha=0.9)
            ax.set_title("6+3 oscillation index", fontsize=10)
            ax.set_xlabel("Oscillation index", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            compact_style(ax)
            st.pyplot(fig, clear_figure=True)

        with cB:
            fig, ax = plt.subplots(figsize=(4.2, 2.0), dpi=160)
            ax.hist(ping_osc_c, bins=bins01, alpha=0.9)
            ax.set_title("CRM oscillation index", fontsize=10)
            ax.set_xlabel("Oscillation index", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            compact_style(ax)
            st.pyplot(fig, clear_figure=True)

        cC, cD = st.columns(2)
        with cC:
            fig, ax = plt.subplots(figsize=(4.2, 2.0), dpi=160)
            ax.hist(obs_rate_6, bins=bins01, alpha=0.9)
            ax.set_title("6+3 observed DLT rate", fontsize=10)
            ax.set_xlabel("DLT rate", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            compact_style(ax)
            st.pyplot(fig, clear_figure=True)

        with cD:
            fig, ax = plt.subplots(figsize=(4.2, 2.0), dpi=160)
            ax.hist(obs_rate_c, bins=bins01, alpha=0.9)
            ax.set_title("CRM observed DLT rate", fontsize=10)
            ax.set_xlabel("DLT rate", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            compact_style(ax)
            st.pyplot(fig, clear_figure=True)

    if show_crm_debug and first_debug_log is not None:
        st.subheader("CRM decision debug (first simulated trial)")
        st.caption("Each row corresponds to a cohort decision point, after outcomes were observed for that cohort.")
        for row in first_debug_log:
            st.markdown(f"**After N={row.get('cohort_end_n')} patients**")
            if "decision" in row:
                st.write(row["decision"])
                st.write(f"P(over target | data) at dose 0 = {row.get('overdose_prob_dose0', np.nan):.3f}")
                st.divider()
                continue

            st.write(
                f"Current level: {row['current_level']} ({dose_labels[row['current_level']]}) | "
                f"Highest tried: {row['highest_tried']} | "
                f"Next level: {row['next_level']} ({dose_labels[row['next_level']]})"
            )

            pm = np.array(row["post_mean"])
            od = np.array(row["overdose_prob"])
            adm = row["admissible_after_constraints"]

            tbl = []
            for i in range(5):
                tbl.append({
                    "Dose": f"L{i} ({dose_labels[i]})",
                    "Posterior mean P(DLT)": float(pm[i]),
                    "P(P(DLT)>target)": float(od[i]),
                })
            st.dataframe(tbl, use_container_width=True)
            st.write(f"Admissible after constraints: {adm}")
            st.divider()
