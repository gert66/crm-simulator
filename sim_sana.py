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
    """
    dfcrm::getprior translation (Lee & Cheung calibration).

    Parameters
    ----------
    halfwidth : float
        Halfwidth delta of the indifference interval around target:
        interval = [target - delta, target + delta].
        Smaller delta -> tighter spacing of skeleton around target.
    target : float
        Target DLT probability used to define the MTD.
    nu : int
        Prior MTD dose level (1-based, like dfcrm). For 5 levels: nu in {1,2,3,4,5}.
    nlevel : int
        Number of dose levels.
    model : str
        "empiric" (power/CRM) or "logistic" working model.
    intcpt : float
        Logistic intercept (used only if model="logistic"). dfcrm default is 3.

    Returns
    -------
    np.ndarray
        Skeleton vector of length nlevel (prior mean DLT per dose level).
    """
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

        # k = nu, nu-1, ..., 2 (1-based)
        for k in range(nu, 1, -1):
            b_k = np.log(np.log(target + halfwidth) / np.log(dosescaled[k - 1]))
            if nu > 1:
                dosescaled[k - 2] = np.exp(np.log(target - halfwidth) / np.exp(b_k))

        if nu < nlevel:
            for k in range(nu, nlevel):  # k = nu, ..., nlevel-1 (1-based)
                b_k1 = np.log(np.log(target - halfwidth) / np.log(dosescaled[k - 1]))
                dosescaled[k] = np.exp(np.log(target + halfwidth) / np.exp(b_k1))

        return dosescaled

    if model == "logistic":
        dosescaled[nu - 1] = np.log(target / (1 - target)) - intcpt

        for k in range(nu, 1, -1):
            b_k = np.log((np.log((target + halfwidth) / (1 - target - halfwidth)) - intcpt) / dosescaled[k - 1])
            if nu > 1:
                dosescaled[k - 2] = (np.log((target - halfwidth) / (1 - target + halfwidth)) - intcpt) / np.exp(b_k)

        if nu < nlevel:
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
    if path is None or len(path) < 3:
        return 0.0
    path = np.asarray(path, dtype=int)
    a = path[:-2]
    b = path[1:-1]
    c = path[2:]
    osc = (a == c) & (a != b)
    return float(np.mean(osc))

# ============================================================
# 6+3 Design (acute-only, with carry-in)
# ============================================================

def run_6plus3(true_p, start_level=0, max_n=36, accept_max_dlt=1, already_n0=0, rng=None):
    """
    Cohorts of 6.
    - 0/6 DLT -> escalate
    - 1/6 DLT -> expand by 3 at same level; escalate if DLTs among those 9 <= accept_max_dlt
    - >=2/6 DLT -> stop/de-escalate (simple stop rule)

    already_n0:
        Number of patients already treated at the start level with 0 acute DLT (carry-in).
        These count toward sample size and inform decision-making.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_levels = len(true_p)
    level = int(start_level)

    n_per_level = np.zeros(n_levels, dtype=int)
    dlt_per_level = np.zeros(n_levels, dtype=int)
    total_n = 0
    last_acceptable = None
    dose_path = []

    # Carry-in
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
# CRM (acute-only, no TITE): dfcrm-like numerical integration
# Using Gauss–Hermite quadrature for posterior summaries
# ============================================================

def posterior_via_gh(sigma, skeleton, n_per_level, dlt_per_level, gh_n=61):
    """
    Posterior weights and dose probabilities using Gauss–Hermite quadrature.

    Model:
      p_k(theta) = skeleton_k ^ exp(theta)
      theta ~ Normal(0, sigma^2)

    Returns
    -------
    post_w : (G,) posterior weights over quadrature nodes (normalized)
    P      : (G,K) p_k(theta_g) for each node and dose
    """
    sk = safe_probs(skeleton)
    n = np.asarray(n_per_level, dtype=float)
    y = np.asarray(dlt_per_level, dtype=float)

    # Integrate under exp(-x^2): ∫ exp(-x^2) f(x) dx ≈ Σ w_i f(x_i)
    x, w = np.polynomial.hermite.hermgauss(int(gh_n))

    # Transform to theta for N(0, sigma^2)
    theta = float(sigma) * np.sqrt(2.0) * x

    P = sk[None, :] ** np.exp(theta)[:, None]

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

def crm_choose_next(sigma, skeleton, n_per_level, dlt_per_level,
                    current_level, target, alpha_overdose, max_step=1, gh_n=61):
    post_mean, overdose_prob = crm_posterior_summaries(
        sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=gh_n
    )

    allowed = np.where(overdose_prob < alpha_overdose)[0]
    if allowed.size == 0:
        return 0, post_mean, overdose_prob

    k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

    if k_star > current_level + max_step:
        k_star = current_level + max_step
    if k_star < current_level - max_step:
        k_star = current_level - max_step

    k_star = int(np.clip(k_star, 0, len(skeleton) - 1))
    return k_star, post_mean, overdose_prob

def crm_select_mtd(sigma, skeleton, n_per_level, dlt_per_level, target, alpha_overdose, gh_n=61):
    post_mean, overdose_prob = crm_posterior_summaries(
        sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=gh_n
    )

    allowed = np.where(overdose_prob < alpha_overdose)[0]
    if allowed.size == 0:
        return 0
    return int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

def run_crm(true_p, target, skeleton, sigma=1.0, start_level=0, max_n=36,
            cohort_size=6, alpha_overdose=0.25, max_step=1, already_n0=0,
            gh_n=61, rng=None):
    """
    CRM simulation (acute-only, no TITE, no subacute).

    Parameters
    ----------
    true_p : list[float]
        True acute DLT probability per dose level (ground truth) used for simulation.
    target : float
        Target DLT probability for MTD definition.
    skeleton : list[float]
        Prior mean DLT per dose (CRM skeleton).
    sigma : float
        Prior SD for theta in Normal(0, sigma^2). Higher sigma => weaker prior.
    start_level : int
        0-based dose index to start treatment.
    max_n : int
        Max sample size (includes carry-in).
    cohort_size : int
        Patients per CRM update.
    alpha_overdose : float
        Overdose control threshold. Allow dose k only if posterior P(p_k > target) < alpha.
    max_step : int
        Max dose level movement per update (1 or 2).
    already_n0 : int
        Carry-in patients already treated at start dose with 0 DLT.
    gh_n : int
        Number of Gauss–Hermite nodes. Higher => more accurate, slower.
    rng : np.random.Generator
        Random number generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_levels = len(true_p)
    level = int(start_level)

    n_per_level = np.zeros(n_levels, dtype=int)
    dlt_per_level = np.zeros(n_levels, dtype=int)
    total_n = 0
    dose_path = []

    # Carry-in
    if already_n0 > 0:
        add = int(already_n0)
        n_per_level[level] += add
        total_n += add
        dose_path.extend([level] * add)

    while total_n < max_n:
        n_add = min(int(cohort_size), max_n - total_n)
        dose_path.extend([level] * n_add)

        out = simulate_bernoulli(n_add, true_p[level], rng)
        n_per_level[level] += n_add
        dlt_per_level[level] += int(out.sum())
        total_n += n_add

        if n_add < cohort_size:
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
            gh_n=gh_n
        )
        level = next_level

    selected = crm_select_mtd(
        sigma=sigma,
        skeleton=skeleton,
        n_per_level=n_per_level,
        dlt_per_level=dlt_per_level,
        target=target,
        alpha_overdose=alpha_overdose,
        gh_n=gh_n
    )

    total_dlts = int(dlt_per_level.sum())
    return selected, n_per_level, total_dlts, dose_path

# ============================================================
# DLT burden metrics per trial
# ============================================================

def observed_dlt_rate(total_dlts, n_total):
    if n_total <= 0:
        return 0.0
    return float(total_dlts) / float(n_total)

def expected_dlt_risk(n_per_level, true_p):
    n_per_level = np.asarray(n_per_level, dtype=float)
    true_p = np.asarray(true_p, dtype=float)
    n_total = float(n_per_level.sum())
    if n_total <= 0:
        return 0.0
    return float(np.sum(n_per_level * true_p) / n_total)

# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="Dose Escalation Simulator: 6+3 vs CRM", layout="centered")
st.title("Dose Escalation Simulator: 6+3 vs CRM")
st.caption("Acute-only CRM. No TITE. No subacute.")

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]

scenario_library = {
    "Default (MTD around level 2)": [0.05, 0.10, 0.20, 0.35, 0.55],
    "Safer overall (pushes higher)": [0.03, 0.06, 0.12, 0.20, 0.30],
    "More toxic overall (lands lower)": [0.08, 0.16, 0.28, 0.42, 0.60],
    "MTD around level 1": [0.05, 0.22, 0.35, 0.50, 0.65],
    "MTD around level 3": [0.03, 0.06, 0.12, 0.24, 0.40],
    "Sharp jump after level 2": [0.05, 0.08, 0.18, 0.40, 0.65],
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
        help="Target toxicity level used to define the MTD."
    )

    start_level = st.selectbox(
        "Start dose level",
        options=list(range(0, 5)),
        index=0,
        format_func=lambda i: f"Level {i} ({dose_labels[i]})",
        help="Starting dose level for both designs."
    )

    already_n0 = st.number_input(
        "Already treated at start dose with 0 acute DLT (carry-in)",
        min_value=0, max_value=20, value=0, step=1,
        help="Adds these patients at the starting dose with 0 DLT before the simulation begins."
    )

    n_sims = st.number_input(
        "Number of simulated trials",
        min_value=200, max_value=20000, value=2000, step=200,
        help="How many virtual trials to run."
    )

    seed = st.number_input(
        "Random seed",
        min_value=1, max_value=10_000_000, value=12345, step=1,
        help="Fix randomness so results are reproducible."
    )

with colB:
    st.subheader("True scenario (ground truth)")

    scenario_name = st.selectbox(
        "Scenario library",
        options=list(scenario_library.keys()),
        index=0,
        help="Pick a predefined true dose-toxicity scenario."
    )
    scenario_vals = scenario_library[scenario_name]

    manual_true = st.toggle(
        "Manually edit the true DLT probabilities",
        value=True,
        help="If on, you can tweak the true probabilities below."
    )

    true_p = []
    for i, lab in enumerate(dose_labels):
        default_val = float(scenario_vals[i])
        if manual_true:
            val = st.number_input(
                f"True P(DLT) at {lab}",
                min_value=0.0, max_value=1.0,
                value=default_val, step=0.01,
                key=f"true_{i}",
                help="Assumed true acute DLT probability at this dose level."
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
        help="Maximum number of patients in each simulated 6+3 trial (includes carry-in)."
    )

    accept_rule = st.selectbox(
        "Acceptance rule after expansion to 9",
        options=[1, 2],
        index=0,
        help="After 1/6 DLT, expand by 3. Escalate only if DLTs among those 9 are <= this number."
    )

    st.info(
        f"""**6+3 decision rules (acute DLT)**  
- Treat 6 patients at the current dose.  
- If 0/6 DLT, go up one level.  
- If 1/6 DLT, treat 3 more at the same dose. Escalate only if DLTs among those 9 <= {accept_rule}/9.  
- If 2+ DLT among the first 6, stop escalation (simple rule)."""
    )

with c2:
    st.subheader("CRM settings")

    max_n_crm = st.number_input(
        "Max sample size (CRM)",
        min_value=12, max_value=120, value=36, step=3,
        help="Maximum number of patients in each simulated CRM trial (includes carry-in)."
    )

    cohort_size = st.number_input(
        "Cohort size (CRM)",
        min_value=1, max_value=12, value=6, step=1,
        help="Patients treated before CRM updates and selects the next dose."
    )

    st.markdown("**Prior (skeleton) mode**")
    prior_mode = st.radio(
        "Skeleton mode",
        options=["Auto (dfcrm getprior)", "Manual (edit each dose)"],
        index=0,
        help="Auto uses dfcrm-style getprior calibration. Manual lets you set the skeleton directly."
    )

    prior_mtd_idx = None  # for plot annotation

    if prior_mode == "Auto (dfcrm getprior)":
        prior_target = st.number_input(
            "Prior target (used to calibrate the skeleton)",
            min_value=0.05, max_value=0.50, value=float(target), step=0.01,
            help="Used only for generating the skeleton."
        )
        prior_halfwidth = st.number_input(
            "Halfwidth (delta)",
            min_value=0.01, max_value=0.30, value=0.10, step=0.01,
            help="Indifference interval is [target - delta, target + delta]."
        )
        prior_nu = st.selectbox(
            "Prior MTD dose level (nu, 1-based)",
            options=[1, 2, 3, 4, 5],
            index=2,
            help="Dose you believe is closest to the MTD before seeing any data (1-based, like dfcrm)."
        )
        prior_model = st.selectbox(
            "Working model for getprior",
            options=["empiric", "logistic"],
            index=0,
            help="Empiric corresponds to the power CRM skeleton calibration."
        )
        prior_intcpt = st.number_input(
            "Logistic intercept (intcpt)",
            min_value=0.0, max_value=10.0, value=3.0, step=0.1,
            help='Only used if model="logistic". dfcrm default is 3.'
        )

        skeleton = dfcrm_getprior(
            halfwidth=float(prior_halfwidth),
            target=float(prior_target),
            nu=int(prior_nu),
            nlevel=5,
            model=str(prior_model),
            intcpt=float(prior_intcpt)
        ).tolist()

        prior_mtd_idx = int(prior_nu) - 1
        st.caption("Auto skeleton values: " + ", ".join([f"{v:.3f}" for v in skeleton]))

    else:
        skeleton = []
        for i, lab in enumerate(dose_labels):
            val = st.number_input(
                f"Skeleton prior mean at {lab}",
                min_value=0.01, max_value=0.99,
                value=float(scenario_vals[i]), step=0.01,
                key=f"sk_{i}",
                help="Prior mean DLT probability at this dose level (skeleton)."
            )
            skeleton.append(val)

    sigma = st.number_input(
        "Prior sigma on theta",
        min_value=0.2, max_value=5.0, value=float(np.sqrt(1.34)), step=0.1,
        help="SD of Normal(0, sigma²) prior on theta. Higher sigma weakens the prior."
    )

    alpha_overdose = st.number_input(
        "Overdose control alpha",
        min_value=0.05, max_value=0.50, value=0.25, step=0.01,
        help="Allow dose k only if posterior P(DLT > target) < alpha."
    )

    max_step = st.selectbox(
        "Max dose step per update",
        options=[1, 2],
        index=0,
        help="Limits how far CRM can move between updates."
    )

    gh_n = st.selectbox(
        "Posterior integration accuracy (Gauss–Hermite points)",
        options=[31, 41, 61, 81],
        index=2,
        help="Higher values are more accurate but slower. 61 is a good default."
    )

    st.info(
        f"""**CRM decision rules (acute-only)**  
- Treat {cohort_size} patients at the current dose, then update the model.  
- Among doses that pass overdose control, pick the dose whose estimated DLT probability is closest to {target:.2f}.  
- Overdose control: allow dose k only if P(DLT > {target:.2f}) < {alpha_overdose:.2f}.  
- Next dose can move at most ±{max_step} level(s)."""
    )

# ------------------------------------------------------------
# Input curves plot (True vs Prior)
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Run controls
# ------------------------------------------------------------

show_ping_plot = st.toggle(
    "Show ping-pong distribution plot",
    value=True,
    help="Shows a compact histogram of the oscillation index across simulated trials."
)

run = st.button("Run simulations")

# ============================================================
# Run simulations
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

    for s in range(ns):
        chosen6, n6, d6, path6 = run_6plus3(
            true_p=true_p,
            start_level=int(start_level),
            max_n=int(max_n_6),
            accept_max_dlt=int(accept_rule),
            already_n0=int(already_n0),
            rng=rng
        )

        chosenc, nc, dc, pathc = run_crm(
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
            rng=rng
        )

        sel_6[chosen6] += 1
        sel_c[chosenc] += 1

        tot_dlt_6[s] = d6
        tot_dlt_c[s] = dc

        nmat_6[s, :] = n6
        nmat_c[s, :] = nc

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

    overshoot6 = float(np.mean((nmat_6[:, true_mtd + 1:].sum(axis=1)) > 0)) if true_mtd < 4 else 0.0
    overshootc = float(np.mean((nmat_c[:, true_mtd + 1:].sum(axis=1)) > 0)) if true_mtd < 4 else 0.0

    mean_total_dlt_6 = float(np.mean(tot_dlt_6))
    mean_total_dlt_c = float(np.mean(tot_dlt_c))

    mean_n_6 = float(np.mean(nmat_6.sum(axis=1)))
    mean_n_c = float(np.mean(nmat_c.sum(axis=1)))

    mean_obs_6 = float(np.mean(obs_rate_6))
    mean_obs_c = float(np.mean(obs_rate_c))

    mean_exp_6 = float(np.mean(exp_risk_6))
    mean_exp_c = float(np.mean(exp_risk_c))

    st.subheader("Results")

    x = np.arange(5)
    width = 0.38

    r1, r2 = st.columns(2)

    with r1:
        fig, ax = plt.subplots(figsize=(4.2, 2.2), dpi=160)
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
        fig, ax = plt.subplots(figsize=(4.2, 2.2), dpi=160)
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
        fig, ax = plt.subplots(figsize=(4.2, 2.1), dpi=160)
        labels = ["6+3", "CRM"]
        obs = [mean_obs_6, mean_obs_c]
        exp = [mean_exp_6, mean_exp_c]
        xi = np.arange(len(labels))
        w = 0.38
        ax.bar(xi - w/2, obs, w, label="Observed DLT rate")
        ax.bar(xi + w/2, exp, w, label="Expected DLT risk")
        ax.set_title("Average DLT burden per patient", fontsize=10)
        ax.set_xticks(xi)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Rate", fontsize=9)
        ax.set_ylim(0, max(max(obs), max(exp)) * 1.20 + 1e-6)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with r4:
        st.markdown("**Summary (averages over simulated trials)**")
        st.write(f"Carry-in at start dose: {int(already_n0)} patients with 0 DLT")
        st.write(f"Mean sample size: 6+3 = {mean_n_6:.1f} | CRM = {mean_n_c:.1f}")
        st.write(f"Mean total DLTs: 6+3 = {mean_total_dlt_6:.2f} | CRM = {mean_total_dlt_c:.2f}")
        st.write(f"Mean observed DLT rate: 6+3 = {mean_obs_6:.3f} | CRM = {mean_obs_c:.3f}")
        st.write(f"Mean expected DLT risk: 6+3 = {mean_exp_6:.3f} | CRM = {mean_exp_c:.3f}")
        st.write(f"P(overshoot above true MTD): 6+3 = {overshoot6:.3f} | CRM = {overshootc:.3f}")
        st.write("")
        st.write(f"Switch rate: 6+3 = {np.mean(ping_sw_6):.3f} | CRM = {np.mean(ping_sw_c):.3f}")
        st.write(f"Oscillation index (A→B→A): 6+3 = {np.mean(ping_osc_6):.3f} | CRM = {np.mean(ping_osc_c):.3f}")
        st.write(f"Mean step size: 6+3 = {np.mean(ping_step_6):.3f} | CRM = {np.mean(ping_step_c):.3f}")

    if show_ping_plot:
        st.subheader("Ping-pong distribution (oscillation index)")

        cA, cB = st.columns(2)
        bins = np.linspace(0, 1, 21)

        with cA:
            fig, ax = plt.subplots(figsize=(4.2, 2.0), dpi=160)
            ax.hist(ping_osc_6, bins=bins, alpha=0.9)
            ax.set_title("6+3 oscillation index", fontsize=10)
            ax.set_xlabel("Oscillation index", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            compact_style(ax)
            st.pyplot(fig, clear_figure=True)

        with cB:
            fig, ax = plt.subplots(figsize=(4.2, 2.0), dpi=160)
            ax.hist(ping_osc_c, bins=bins, alpha=0.9)
            ax.set_title("CRM oscillation index", fontsize=10)
            ax.set_xlabel("Oscillation index", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            compact_style(ax)
            st.pyplot(fig, clear_figure=True)
