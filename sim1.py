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
# dfcrm getprior port (empiric + logistic)
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
# 6+3 design (simple)
# ============================================================

def run_6plus3(true_p, start_level=1, max_n=27, accept_max_dlt=1, rng=None):
    """
    Simple 6+3:
      - Treat 6 at current level.
      - If 0/6 DLT: escalate.
      - If 1/6 DLT: expand by 3 (total 9); accept if <= accept_max_dlt.
      - If >=2/6 DLT OR expansion fails: de-escalate and stop.
    Returns:
      selected_level (0-based), n_per_level, total_dlts
    """
    if rng is None:
        rng = np.random.default_rng()

    true_p = np.asarray(true_p, dtype=float)
    n_levels = len(true_p)

    level = int(start_level)
    n_per = np.zeros(n_levels, dtype=int)
    y_per = np.zeros(n_levels, dtype=int)

    total_n = 0
    last_acceptable = None

    while total_n < int(max_n):
        n_add = min(6, int(max_n) - total_n)
        out6 = simulate_bernoulli(n_add, true_p[level], rng)

        n_per[level] += n_add
        y_per[level] += int(out6.sum())
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
            n_add2 = min(3, int(max_n) - total_n)
            out3 = simulate_bernoulli(n_add2, true_p[level], rng)

            n_per[level] += n_add2
            y_per[level] += int(out3.sum())
            total_n += n_add2

            if n_add2 < 3:
                break

            d9 = d6 + int(out3.sum())
            if d9 <= int(accept_max_dlt):
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
    return selected, n_per, int(y_per.sum())

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

    # Safe fallback if EWOC makes allowed empty:
    # go to the lowest dose (clinically conservative), but still obey step/guardrail later.
    if allowed.size == 0:
        allowed = np.array([0], dtype=int)

    k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

    # step limiting
    k_star = int(np.clip(k_star, current_level - int(max_step), current_level + int(max_step)))

    # guardrail
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
        # If EWOC is extremely tight, be conservative.
        return 0

    if restrict_to_tried:
        tried = np.where(np.asarray(n_per_level) > 0)[0]
        if tried.size > 0:
            allowed2 = np.intersect1d(allowed, tried)
            if allowed2.size > 0:
                allowed = allowed2
            else:
                # If none of the tried are allowed, choose lowest tried
                return int(tried.min())

    return int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

def run_crm_trial(
    true_p, target, skeleton,
    sigma=1.0,
    start_level=1,
    already_treated_start=0,
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
    """
    already_treated_start:
      - number of patients already treated at the start dose with 0 DLT (acute),
        added to the data BEFORE any new cohorts are simulated.
    """
    if rng is None:
        rng = np.random.default_rng()

    true_p = np.asarray(true_p, dtype=float)
    n_levels = len(true_p)

    level = int(start_level)
    n_per = np.zeros(n_levels, dtype=int)
    y_per = np.zeros(n_levels, dtype=int)

    # Inject already-treated safe patients at start dose
    already_treated_start = int(max(0, already_treated_start))
    if already_treated_start > 0:
        n_per[level] += already_treated_start
        # y_per[level] += 0  # explicit

    highest_tried = level if (already_treated_start > 0) else -1
    any_dlt_seen = False
    debug_rows = []

    # Burn-in is only meaningful when we have no prior treated patients at start
    burn_in_active = bool(burn_in_until_first_dlt and already_treated_start == 0)

    while int(n_per.sum()) < int(max_n):
        n_add = min(int(cohort_size), int(max_n) - int(n_per.sum()))
        out = simulate_bernoulli(n_add, true_p[level], rng)

        n_per[level] += n_add
        y_per[level] += int(out.sum())
        highest_tried = max(highest_tried, level)

        if int(out.sum()) > 0:
            any_dlt_seen = True

        if debug:
            debug_rows.append({
                "treated_level": level,
                "cohort_n": int(n_add),
                "cohort_dlts": int(out.sum()),
                "any_dlt_seen": bool(any_dlt_seen),
                "n_per": [int(x) for x in n_per],
                "y_per": [int(x) for x in y_per],
            })

        if n_add < int(cohort_size):
            break

        if burn_in_active and (not any_dlt_seen):
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

    return int(selected), n_per, int(y_per.sum()), debug_rows

# ============================================================
# Defaults aligned to Sama's R code
# ============================================================

st.set_page_config(page_title="6+3 vs CRM (Playground)", layout="wide")

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]

# Sama R: P.acute.real <- c(0.01,0.02,0.12, 0.2, 0.35)
DEFAULT_TRUE_P = [0.01, 0.02, 0.12, 0.20, 0.35]

R_DEFAULTS = {
    # Essentials (R)
    "target": 0.15,                 # R: prior.target.acute (also used as target in your earlier example)
    "start_level_1b": 2,            # R: burning phase p <- 2 (1-based). This equals 0-based level 1.
    "already_treated_start": 0,     # extra feature requested
    "n_sims": 1000,                 # R: NREP <- 1000
    "seed": 123,                    # R: set.seed(123)
    "max_n_63": 27,                 # R: N.patient = 27 (using same cap for 6+3 by default)
    "max_n_crm": 27,                # R: N.patient = 27
    "cohort_size": 3,               # R: CO = 3

    # Prior playground (R)
    "prior_model": "empiric",       # R: getprior() default is power/empiric-style skeleton
    "prior_target": 0.15,           # R: prior.target.acute
    "halfwidth": 0.10,              # R: getprior halfwidth = 0.1
    "prior_nu": 3,                  # R: prior.MTD.acute = 3 (1-based)
    "logistic_intcpt": 3.0,

    # CRM knobs (sigma not explicit in Sama R snippet)
    "sigma": 1.0,
    "burn_in": True,
    "ewoc_on": False,
    "ewoc_alpha": 0.25,

    # Advanced
    "gh_n": 61,
    "max_step": 1,
    "enforce_guardrail": True,
    "restrict_final_mtd": True,
    "show_debug": False,

    # 6+3 advanced
    "accept_rule_63": 1,
}

def init_state():
    for k, v in R_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    for i in range(5):
        st.session_state.setdefault(f"true_{i}", float(DEFAULT_TRUE_P[i]))

def reset_all_defaults():
    for k, v in R_DEFAULTS.items():
        st.session_state[k] = v
    # True curve intentionally NOT reset by default.

init_state()

# ============================================================
# Header
# ============================================================

st.title("Dose Escalation Simulator (acute-only): 6+3 vs CRM")
st.caption("Single-window app: Essentials (top) + Playground + Results below.")

# ============================================================
# Essentials (top expander)
# ============================================================

with st.expander("Essentials", expanded=True):
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("### Study")
        st.number_input(
            "Target DLT rate",
            min_value=0.05, max_value=0.50, step=0.01,
            key="target",
            help=(
                "This is the target toxicity probability used to define the MTD.\n\n"
                "R mapping: in your scripts you use target.acute / prior.target.acute. "
                "Here it drives CRM dose selection and the 'True MTD' marker."
            )
        )
        st.number_input(
            "Start dose level (1-based)",
            min_value=1, max_value=5, step=1,
            key="start_level_1b",
            help=(
                "Dose level where the trial starts (1-based indexing).\n\n"
                "R mapping: burning phase uses p <- 2 (1-based) in Sama’s code."
            )
        )
        st.number_input(
            "Already treated at start dose (0 DLT)",
            min_value=0, max_value=500, step=1,
            key="already_treated_start",
            help=(
                "Adds N patients at the start dose with 0 acute DLT to the CRM data before simulation.\n\n"
                "R mapping: this does not exist as a direct parameter in Sama’s code. "
                "It is a practical extension for 'we already treated x patients safely at the starting dose'."
            )
        )

    with c2:
        st.markdown("### Simulation")
        st.number_input(
            "Number of simulated trials",
            min_value=50, max_value=5000, step=50,
            key="n_sims",
            help=(
                "How many simulated trials to run.\n\n"
                "R mapping: NREP <- 1000 in Sama’s Final.Fun()."
            )
        )
        st.number_input(
            "Random seed",
            min_value=1, max_value=10_000_000, step=1,
            key="seed",
            help=(
                "Controls reproducibility.\n\n"
                "R mapping: set.seed(123) in Sama’s code."
            )
        )

    with c3:
        st.markdown("### Sample size")
        st.number_input(
            "Maximum sample size (6+3)",
            min_value=6, max_value=200, step=3,
            key="max_n_63",
            help=(
                "Hard cap for the 6+3 design.\n\n"
                "R mapping: Sama’s acute/subacute simulation uses N.patient = 27."
            )
        )
        st.number_input(
            "Maximum sample size (CRM)",
            min_value=6, max_value=200, step=3,
            key="max_n_crm",
            help=(
                "Hard cap for the CRM trial.\n\n"
                "R mapping: N.patient = 27 in Sama’s code."
            )
        )
        st.number_input(
            "Cohort size",
            min_value=1, max_value=12, step=1,
            key="cohort_size",
            help=(
                "Patients treated per CRM cohort update.\n\n"
                "R mapping: CO = 3 in Sama’s code."
            )
        )

    st.write("")
    if st.button("Reset to defaults"):
        reset_all_defaults()
        st.rerun()

# ============================================================
# Playground expander
# ============================================================

with st.expander("Playground", expanded=True):
    left, mid, right = st.columns([1.0, 1.0, 1.2], gap="large")

    # ---- True curve
    with left:
        st.markdown("### True acute DLT curve")
        true_p = []
        for i, lab in enumerate(dose_labels):
            true_p.append(float(st.number_input(
                f"L{i} {lab}",
                min_value=0.0, max_value=1.0, step=0.01,
                key=f"true_{i}",
                help="Ground truth acute DLT probability used to simulate Bernoulli outcomes."
            )))
        target_val = float(st.session_state["target"])
        true_mtd = find_true_mtd(true_p, target_val)
        st.caption(f"True MTD (closest to target) = L{true_mtd}")

    # ---- Prior playground
    with mid:
        st.markdown("### Prior playground")
        st.radio(
            "Skeleton model",
            options=["empiric", "logistic"],
            horizontal=True,
            key="prior_model",
            help=(
                "Controls how the skeleton is generated.\n\n"
                "R mapping: dfcrm::getprior() generates a prior skeleton; this mimics that behavior."
            )
        )
        st.slider(
            "Prior target",
            min_value=0.05, max_value=0.50, step=0.01,
            key="prior_target",
            help="R mapping: prior.target.acute (Sama code: prior.target.acute = 0.15)."
        )
        st.slider(
            "Halfwidth (delta)",
            min_value=0.01, max_value=0.30, step=0.01,
            key="halfwidth",
            help="R mapping: getprior(halfwidth = 0.1)."
        )
        st.slider(
            "Prior MTD (1-based)",
            min_value=1, max_value=5, step=1,
            key="prior_nu",
            help="R mapping: prior.MTD.acute = 3 (1-based)."
        )
        st.slider(
            "Logistic intercept",
            min_value=-10.0, max_value=10.0, step=0.1,
            key="logistic_intcpt",
            help="Only used for logistic skeleton generation."
        )

        skeleton = dfcrm_getprior(
            halfwidth=float(st.session_state["halfwidth"]),
            target=float(st.session_state["prior_target"]),
            nu=int(st.session_state["prior_nu"]),
            nlevel=5,
            model=str(st.session_state["prior_model"]),
            intcpt=float(st.session_state["logistic_intcpt"]),
        ).tolist()

    # ---- CRM knobs + preview
    with right:
        st.markdown("### CRM knobs + preview")
        st.slider(
            "Prior sigma on theta",
            min_value=0.2, max_value=5.0, step=0.1,
            key="sigma",
            help=(
                "Prior SD for theta in theta ~ N(0, sigma^2).\n\n"
                "R mapping: sigma is not shown explicitly in Sama’s snippet; dfcrm has default priors. "
                "Here it controls how strongly the model can move away from the skeleton."
            )
        )
        st.toggle(
            "Burn-in until first DLT",
            key="burn_in",
            help=(
                "If enabled: escalate cohort-wise until the first observed acute DLT, then use CRM.\n\n"
                "R mapping: Sama’s script has a burning phase before running TITE-CRM.\n\n"
                "Note: if you set 'Already treated at start dose' > 0, burn-in is skipped automatically."
            )
        )
        st.toggle(
            "Enable EWOC overdose control",
            key="ewoc_on",
            help=(
                "EWOC rule: allowed doses satisfy P(p_k > target | data) < alpha.\n\n"
                "R mapping: overdose control is an optional concept; your Python CRM implements it directly."
            )
        )
        st.slider(
            "EWOC alpha",
            min_value=0.01, max_value=0.99, step=0.01,
            key="ewoc_alpha",
            disabled=(not st.session_state["ewoc_on"]),
            help="Only used if EWOC is enabled."
        )

        fig, ax = plt.subplots(figsize=(5.4, 2.8), dpi=160)
        x = np.arange(5)
        ax.plot(x, true_p, marker="o", linewidth=1.6, label="True P(DLT)")
        ax.plot(x, skeleton, marker="o", linewidth=1.6, label="Prior (skeleton)")
        ax.axhline(target_val, linewidth=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, min(1.0, max(max(true_p), max(skeleton), target_val) * 1.25 + 0.02))
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        st.pyplot(fig, clear_figure=True)

        st.write("")
        run = st.button("Run simulations", use_container_width=True)

# ============================================================
# Advanced settings (collapsed, below playground)
# ============================================================

with st.expander("Advanced settings", expanded=False):
    a1, a2, a3 = st.columns([1.0, 1.0, 1.0], gap="large")

    with a1:
        st.selectbox(
            "Gauss–Hermite points",
            options=[31, 41, 61, 81],
            index=[31, 41, 61, 81].index(int(st.session_state["gh_n"])),
            key="gh_n",
            help="Posterior integration accuracy vs speed."
        )
        st.selectbox(
            "Max dose step per CRM update",
            options=[1, 2],
            index=[1, 2].index(int(st.session_state["max_step"])),
            key="max_step",
            help="Limits how far the CRM can move per cohort update."
        )

    with a2:
        st.toggle(
            "Guardrail: next dose ≤ highest tried + 1",
            key="enforce_guardrail",
            help="Prevents skipping over untried dose levels."
        )
        st.toggle(
            "Final MTD must be among tried doses",
            key="restrict_final_mtd",
            help="Restricts final MTD selection to doses that were actually treated."
        )

    with a3:
        st.toggle(
            "Show CRM decision debug (first simulated trial)",
            key="show_debug",
            help="Shows admissible set and posterior summaries for the first trial."
        )

    st.markdown("---")
    st.markdown("**6+3 design settings**")
    st.selectbox(
        "Acceptance rule after expansion to 9",
        options=[1, 2],
        index=[1, 2].index(int(st.session_state["accept_rule_63"])),
        key="accept_rule_63",
        help="If 1/6 DLT, expand to 9; accept if total DLTs <= this threshold."
    )

# ============================================================
# Results
# ============================================================

if "run" in locals() and run:
    rng = np.random.default_rng(int(st.session_state["seed"]))
    ns = int(st.session_state["n_sims"])

    start_0b = int(st.session_state["start_level_1b"]) - 1
    start_0b = int(np.clip(start_0b, 0, 4))

    sel_63 = np.zeros(5, dtype=int)
    sel_crm = np.zeros(5, dtype=int)

    nmat_63 = np.zeros((ns, 5), dtype=int)
    nmat_crm = np.zeros((ns, 5), dtype=int)

    dlt_63 = np.zeros(ns, dtype=int)
    dlt_crm = np.zeros(ns, dtype=int)

    debug_dump = None

    for s in range(ns):
        chosen63, n63, y63 = run_6plus3(
            true_p=true_p,
            start_level=start_0b,
            max_n=int(st.session_state["max_n_63"]),
            accept_max_dlt=int(st.session_state["accept_rule_63"]),
            rng=rng
        )

        debug_flag = bool(st.session_state["show_debug"] and s == 0)

        chosenc, nc, yc, dbg = run_crm_trial(
            true_p=true_p,
            target=float(st.session_state["target"]),
            skeleton=skeleton,
            sigma=float(st.session_state["sigma"]),
            start_level=start_0b,
            already_treated_start=int(st.session_state["already_treated_start"]),
            max_n=int(st.session_state["max_n_crm"]),
            cohort_size=int(st.session_state["cohort_size"]),
            max_step=int(st.session_state["max_step"]),
            gh_n=int(st.session_state["gh_n"]),
            enforce_guardrail=bool(st.session_state["enforce_guardrail"]),
            restrict_final_mtd_to_tried=bool(st.session_state["restrict_final_mtd"]),
            ewoc_on=bool(st.session_state["ewoc_on"]),
            ewoc_alpha=float(st.session_state["ewoc_alpha"]),
            burn_in_until_first_dlt=bool(st.session_state["burn_in"]),
            rng=rng,
            debug=debug_flag
        )

        sel_63[chosen63] += 1
        sel_crm[chosenc] += 1

        nmat_63[s, :] = n63
        nmat_crm[s, :] = nc

        dlt_63[s] = y63
        dlt_crm[s] = yc

        if debug_flag:
            debug_dump = dbg

    p63 = sel_63 / float(ns)
    pcrm = sel_crm / float(ns)

    avg63 = np.mean(nmat_63, axis=0)
    avgcrm = np.mean(nmat_crm, axis=0)

    mean_n63 = float(np.mean(nmat_63.sum(axis=1)))
    mean_ncrm = float(np.mean(nmat_crm.sum(axis=1)))

    # DLT probability per patient (average DLTs / average N)
    dlt_prob_63 = float(np.mean(dlt_63) / max(1e-9, mean_n63))
    dlt_prob_crm = float(np.mean(dlt_crm) / max(1e-9, mean_ncrm))

    st.markdown("---")
    st.subheader("Results")

    r1, r2, r3 = st.columns([1.1, 1.1, 0.8], gap="large")

    with r1:
        fig, ax = plt.subplots(figsize=(6.2, 2.8), dpi=160)
        xx = np.arange(5)
        w = 0.38
        ax.bar(xx - w/2, p63, w, label="6+3")
        ax.bar(xx + w/2, pcrm, w, label="CRM")
        ax.set_title("P(select dose as MTD)", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, max(p63.max(), pcrm.max()) * 1.15 + 1e-6)
        ax.axvline(true_mtd, linewidth=1, alpha=0.6)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with r2:
        fig, ax = plt.subplots(figsize=(6.2, 2.8), dpi=160)
        xx = np.arange(5)
        w = 0.38
        ax.bar(xx - w/2, avg63, w, label="6+3")
        ax.bar(xx + w/2, avgcrm, w, label="CRM")
        ax.set_title("Avg patients treated per dose", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Patients", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with r3:
        st.metric("DLT prob per patient (6+3)", f"{dlt_prob_63:.3f}")
        st.metric("DLT prob per patient (CRM)", f"{dlt_prob_crm:.3f}")
        st.caption(
            f"n_sims={ns} | seed={int(st.session_state['seed'])} | "
            f"True MTD marker=L{true_mtd}"
        )

    if st.session_state["show_debug"] and debug_dump:
        st.subheader("CRM debug (first simulated trial)")
        for i, row in enumerate(debug_dump, start=1):
            st.write(
                f"Update {i}: treated L{row['treated_level']} | n={row['cohort_n']} "
                f"| dlts={row['cohort_dlts']} | any_dlt_seen={row['any_dlt_seen']}"
            )
            if "next_level" in row:
                st.write(f"  allowed: {row['allowed_levels']} | next: L{row['next_level']} | highest_tried={row['highest_tried']}")
                st.write(f"  post_mean: {[round(v,3) for v in row['post_mean']]}")
                st.write(f"  od_prob:   {[round(v,3) for v in row['od_prob']]}")
