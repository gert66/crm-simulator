import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# ============================================================
# 0) Fixed sizing knobs (ONE place to tune)
# ============================================================
PREVIEW_W_PX = 240
RESULT_W_PX  = 460

PREVIEW_W_IN, PREVIEW_H_IN, PREVIEW_DPI = 3.4, 2.2, 170
RESULT_W_IN,  RESULT_H_IN,  RESULT_DPI  = 6.0, 4.4, 170

# ============================================================
# Helpers
# ============================================================

def safe_probs(x):
    x = np.asarray(x, dtype=float)
    return np.clip(x, 1e-6, 1 - 1e-6)

def simulate_bernoulli(n, p, rng):
    return rng.binomial(1, p, size=int(n))

def find_true_safe_dose(true_acute, true_subacute, target_acute, target_subacute):
    """Highest dose level where both true rates are at or below their targets."""
    safe = [d for d in range(len(true_acute))
            if true_acute[d] <= target_acute and true_subacute[d] <= target_subacute]
    return max(safe) if safe else None

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)

def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ============================================================
# dfcrm getprior port (unchanged from original)
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
# Dual-endpoint 6+3 design
#
# After 6 patients at dose d:
#   escalate  if acute_6 == 0  AND subacute_6 <= 1
#   stop      if acute_6 >= 2  OR  subacute_6 >= 3
#   else expand to 9 patients
# After 9 patients:
#   escalate  if acute_9 <= 1  AND subacute_9 <= 3
#   else stop (de-escalate to previous dose)
# ============================================================

def run_dual_6plus3(true_acute, true_subacute, start_level=0, max_n=27, rng=None, debug=False):
    if rng is None:
        rng = np.random.default_rng()

    true_acute    = np.asarray(true_acute,    dtype=float)
    true_subacute = np.asarray(true_subacute, dtype=float)
    n_levels = len(true_acute)

    level = int(start_level)
    n_per  = np.zeros(n_levels, dtype=int)
    yA_per = np.zeros(n_levels, dtype=int)   # acute DLTs per level
    yS_per = np.zeros(n_levels, dtype=int)   # subacute DLTs per level

    total_n = 0
    last_acceptable = None
    debug_rows = []

    while total_n < int(max_n):
        n_add = min(6, int(max_n) - total_n)
        outA6 = simulate_bernoulli(n_add, true_acute[level],    rng)
        outS6 = simulate_bernoulli(n_add, true_subacute[level], rng)

        n_per[level]  += n_add
        yA_per[level] += int(outA6.sum())
        yS_per[level] += int(outS6.sum())
        total_n       += n_add

        if n_add < 6:
            break

        a6 = int(outA6.sum())
        s6 = int(outS6.sum())

        dbg = {"level": level, "phase": "6pt",
               "acute_dlts": a6, "subacute_dlts": s6} if debug else None

        # escalate: 0 acute AND ≤1 subacute
        if a6 == 0 and s6 <= 1:
            last_acceptable = level
            if dbg is not None:
                dbg["decision"] = "escalate"
                debug_rows.append(dbg)
            if level < n_levels - 1:
                level += 1
                continue
            break

        # stop unsafe: ≥2 acute OR ≥3 subacute
        if a6 >= 2 or s6 >= 3:
            if dbg is not None:
                dbg["decision"] = "stop_unsafe"
                debug_rows.append(dbg)
            if level > 0:
                level -= 1
            break

        # otherwise: add 3 more patients
        n_add2 = min(3, int(max_n) - total_n)
        outA3 = simulate_bernoulli(n_add2, true_acute[level],    rng)
        outS3 = simulate_bernoulli(n_add2, true_subacute[level], rng)

        n_per[level]  += n_add2
        yA_per[level] += int(outA3.sum())
        yS_per[level] += int(outS3.sum())
        total_n       += n_add2

        if n_add2 < 3:
            break

        a9 = a6 + int(outA3.sum())
        s9 = s6 + int(outS3.sum())

        if dbg is not None:
            dbg["phase"] = "9pt"
            dbg["acute_dlts"] = a9
            dbg["subacute_dlts"] = s9

        # escalate: ≤1 acute AND ≤3 subacute
        if a9 <= 1 and s9 <= 3:
            last_acceptable = level
            if dbg is not None:
                dbg["decision"] = "escalate"
                debug_rows.append(dbg)
            if level < n_levels - 1:
                level += 1
                continue
            break

        # stop unsafe
        if dbg is not None:
            dbg["decision"] = "stop_unsafe"
            debug_rows.append(dbg)
        if level > 0:
            level -= 1
        break

    selected = 0 if last_acceptable is None else int(last_acceptable)
    return selected, n_per, int(yA_per.sum()), int(yS_per.sum()), debug_rows

# ============================================================
# CRM posterior via Gauss–Hermite quadrature
# Unchanged — reused independently for each endpoint.
# ============================================================

def posterior_via_gh(sigma, skeleton, n_per_level, dlt_per_level, gh_n=61):
    sk = safe_probs(skeleton)
    n  = np.asarray(n_per_level,  dtype=float)
    y  = np.asarray(dlt_per_level, dtype=float)

    x, w = np.polynomial.hermite.hermgauss(int(gh_n))
    theta = float(sigma) * np.sqrt(2.0) * x

    P  = sk[None, :] ** np.exp(theta)[:, None]
    P  = safe_probs(P)

    ll = (y[None, :] * np.log(P) + (n[None, :] - y[None, :]) * np.log(1 - P)).sum(axis=1)

    log_unnorm = np.log(w) + ll
    m      = np.max(log_unnorm)
    unnorm = np.exp(log_unnorm - m)
    post_w = unnorm / np.sum(unnorm)
    return post_w, P

def crm_posterior_summaries(sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=61):
    post_w, P    = posterior_via_gh(sigma, skeleton, n_per_level, dlt_per_level, gh_n=gh_n)
    post_mean    = (post_w[:, None] * P).sum(axis=0)
    overdose_prob = (post_w[:, None] * (P > target)).sum(axis=0)
    return post_mean, overdose_prob

# ============================================================
# Dual CRM decision functions
#
# Joint safety rule: a dose is allowed only if BOTH
#   P(acute(d) > target_acute | data) < ewoc_alpha
#   AND
#   P(subacute(d) > target_subacute | data) < ewoc_alpha
#
# Among allowed doses, choose the HIGHEST (maximise dose
# intensity subject to joint safety), then apply guardrails.
# ============================================================

def crm_choose_next_dual(
    sigma, skeleton_acute, skeleton_subacute,
    n_per_level, yA_per_level, yS_per_level,
    current_level, target_acute, target_subacute,
    ewoc_alpha=None, max_step=1, gh_n=61,
    enforce_highest_tried_plus_one=True, highest_tried=None,
):
    pmA, odA = crm_posterior_summaries(
        sigma, skeleton_acute,    n_per_level, yA_per_level, target_acute,    gh_n=gh_n)
    pmS, odS = crm_posterior_summaries(
        sigma, skeleton_subacute, n_per_level, yS_per_level, target_subacute, gh_n=gh_n)

    n_levels = len(skeleton_acute)
    if ewoc_alpha is None:
        allowed = np.arange(n_levels)
    else:
        # joint safety: both endpoints must satisfy the overdose criterion
        allowed = np.where((odA < float(ewoc_alpha)) & (odS < float(ewoc_alpha)))[0]

    if allowed.size == 0:
        allowed = np.array([0], dtype=int)

    # choose highest jointly safe dose, then apply guardrails
    k_star = int(allowed.max())
    k_star = int(np.clip(k_star, current_level - int(max_step), current_level + int(max_step)))

    if enforce_highest_tried_plus_one and highest_tried is not None:
        k_star = int(min(k_star, int(highest_tried) + 1))

    k_star = int(np.clip(k_star, 0, n_levels - 1))
    return k_star, pmA, pmS, odA, odS, allowed

def crm_select_mtd_dual(
    sigma, skeleton_acute, skeleton_subacute,
    n_per_level, yA_per_level, yS_per_level,
    target_acute, target_subacute,
    ewoc_alpha=None, gh_n=61, restrict_to_tried=True,
):
    pmA, odA = crm_posterior_summaries(
        sigma, skeleton_acute,    n_per_level, yA_per_level, target_acute,    gh_n=gh_n)
    pmS, odS = crm_posterior_summaries(
        sigma, skeleton_subacute, n_per_level, yS_per_level, target_subacute, gh_n=gh_n)

    n_levels = len(skeleton_acute)
    if ewoc_alpha is None:
        allowed = np.arange(n_levels)
    else:
        allowed = np.where((odA < float(ewoc_alpha)) & (odS < float(ewoc_alpha)))[0]

    if allowed.size == 0:
        return 0

    if restrict_to_tried:
        tried = np.where(np.asarray(n_per_level) > 0)[0]
        if tried.size > 0:
            allowed2 = np.intersect1d(allowed, tried)
            if allowed2.size > 0:
                allowed = allowed2
            else:
                return int(tried.min())

    return int(allowed.max())   # highest jointly safe (tried) dose

# ============================================================
# Dual CRM trial runner
# ============================================================

def run_dual_crm_trial(
    true_acute, true_subacute,
    target_acute, target_subacute,
    skeleton_acute, skeleton_subacute,
    sigma=1.0, start_level=0, already_treated_start=0,
    max_n=27, cohort_size=3, max_step=1, gh_n=61,
    enforce_guardrail=True, restrict_final_mtd_to_tried=True,
    ewoc_on=True, ewoc_alpha=0.25,
    burn_in_until_first_dlt=True,
    rng=None, debug=False,
):
    if rng is None:
        rng = np.random.default_rng()

    true_acute    = np.asarray(true_acute,    dtype=float)
    true_subacute = np.asarray(true_subacute, dtype=float)
    n_levels = len(true_acute)

    level  = int(start_level)
    n_per  = np.zeros(n_levels, dtype=int)
    yA_per = np.zeros(n_levels, dtype=int)
    yS_per = np.zeros(n_levels, dtype=int)

    already_treated_start = int(max(0, already_treated_start))
    if already_treated_start > 0:
        # Pre-treated patients: n patients at start dose, 0 DLTs of either type
        n_per[level] += already_treated_start

    highest_tried  = level if already_treated_start > 0 else -1
    any_dlt_seen   = False
    debug_rows     = []

    # Burn-in: escalate one level at a time until any DLT (either endpoint) is seen.
    # Disabled if there are already pre-treated patients.
    burn_in_active = bool(burn_in_until_first_dlt and already_treated_start == 0)

    ewoc_alpha_eff = float(ewoc_alpha) if ewoc_on else None

    while int(n_per.sum()) < int(max_n):
        n_add = min(int(cohort_size), int(max_n) - int(n_per.sum()))
        outA  = simulate_bernoulli(n_add, true_acute[level],    rng)
        outS  = simulate_bernoulli(n_add, true_subacute[level], rng)

        n_per[level]  += n_add
        yA_per[level] += int(outA.sum())
        yS_per[level] += int(outS.sum())
        highest_tried  = max(highest_tried, level)

        if int(outA.sum()) > 0 or int(outS.sum()) > 0:
            any_dlt_seen = True

        if debug:
            debug_rows.append({
                "treated_level":   level,
                "cohort_n":        int(n_add),
                "acute_dlts":      int(outA.sum()),
                "subacute_dlts":   int(outS.sum()),
                "any_dlt_seen":    bool(any_dlt_seen),
            })

        if n_add < int(cohort_size):
            break

        # Burn-in: keep escalating until first DLT of either type
        if burn_in_active and (not any_dlt_seen):
            if level < n_levels - 1:
                level += 1
                continue

        next_level, pmA, pmS, odA, odS, allowed = crm_choose_next_dual(
            sigma=sigma,
            skeleton_acute=skeleton_acute,
            skeleton_subacute=skeleton_subacute,
            n_per_level=n_per,
            yA_per_level=yA_per,
            yS_per_level=yS_per,
            current_level=level,
            target_acute=target_acute,
            target_subacute=target_subacute,
            ewoc_alpha=ewoc_alpha_eff,
            max_step=max_step,
            gh_n=gh_n,
            enforce_highest_tried_plus_one=enforce_guardrail,
            highest_tried=highest_tried,
        )

        if debug:
            debug_rows[-1].update({
                "next_level":          int(next_level),
                "allowed_levels":      ",".join([str(int(a)) for a in allowed]),
                "highest_tried":       int(highest_tried),
                "post_mean_acute":     [float(v) for v in pmA],
                "post_mean_subacute":  [float(v) for v in pmS],
                "od_prob_acute":       [float(v) for v in odA],
                "od_prob_subacute":    [float(v) for v in odS],
            })

        level = int(next_level)

    selected = crm_select_mtd_dual(
        sigma=sigma,
        skeleton_acute=skeleton_acute,
        skeleton_subacute=skeleton_subacute,
        n_per_level=n_per,
        yA_per_level=yA_per,
        yS_per_level=yS_per,
        target_acute=target_acute,
        target_subacute=target_subacute,
        ewoc_alpha=ewoc_alpha_eff,
        gh_n=gh_n,
        restrict_to_tried=restrict_final_mtd_to_tried,
    )

    return int(selected), n_per, int(yA_per.sum()), int(yS_per.sum()), debug_rows

# ============================================================
# Streamlit config + CSS
# ============================================================

st.set_page_config(
    page_title="Dual-endpoint 6+3 vs CRM",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      [data-testid="stSidebar"]       { display: none; }
      [data-testid="stSidebarNav"]    { display: none; }
      [data-testid="collapsedControl"]{ display: none; }

      /* push content down so top expander is not clipped */
      .block-container { padding-top: 2.8rem; padding-bottom: 0.9rem; }

      .element-container { margin-bottom: 0.20rem; }

      /* fixed-size images: do not let browser rescale them */
      [data-testid="stImage"] img {
        max-width: none !important;
        width: auto !important;
        height: auto !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Defaults
# ============================================================

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]

DEFAULT_TRUE_ACUTE    = [0.01, 0.02, 0.12, 0.20, 0.35]
DEFAULT_TRUE_SUBACUTE = [0.02, 0.05, 0.15, 0.25, 0.40]

R_DEFAULTS = {
    # Study
    "target_acute":            0.15,
    "target_subacute":         0.20,
    "start_level_1b":          2,
    "already_treated_start":   0,
    # Simulation
    "n_sims":                  200,
    "seed":                    123,
    # Sample size
    "max_n_63":                27,
    "max_n_crm":               27,
    "cohort_size":             3,
    # Priors – shared
    "prior_model":             "empiric",
    "logistic_intcpt":         3.0,
    # Priors – acute
    "prior_target_acute":      0.15,
    "halfwidth_acute":         0.10,
    "prior_nu_acute":          3,
    # Priors – subacute
    "prior_target_subacute":   0.20,
    "halfwidth_subacute":      0.10,
    "prior_nu_subacute":       3,
    # CRM knobs
    "sigma":                   1.0,
    "burn_in":                 True,
    "ewoc_on":                 True,
    "ewoc_alpha":              0.25,
    # CRM integration
    "gh_n":                    61,
    "max_step":                1,
    # CRM safety / selection
    "enforce_guardrail":       True,
    "restrict_final_mtd":      True,
    "show_debug":              False,
    # Figure sizing
    "preview_w_px":            PREVIEW_W_PX,
    "result_w_px":             RESULT_W_PX,
}

TRUE_ACUTE_KEYS    = [f"true_acute_{i}"    for i in range(5)]
TRUE_SUBACUTE_KEYS = [f"true_subacute_{i}" for i in range(5)]

# ============================================================
# Reset logic
#
# Reset is triggered by setting _do_reset=True then rerunning.
# It runs at the TOP of the script, before any widget is created,
# so overwriting widget-backed keys here is safe (no widget exists yet).
# It also clears stored results so stale charts don't linger.
# ============================================================

if st.session_state.get("_do_reset", False):
    for k, v in R_DEFAULTS.items():
        st.session_state[k] = v
    for i in range(5):
        st.session_state[TRUE_ACUTE_KEYS[i]]    = float(DEFAULT_TRUE_ACUTE[i])
        st.session_state[TRUE_SUBACUTE_KEYS[i]] = float(DEFAULT_TRUE_SUBACUTE[i])
    st.session_state["_results"]  = None
    st.session_state["_do_reset"] = False
    st.rerun()

def init_state():
    for k, v in R_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    for i in range(5):
        st.session_state.setdefault(TRUE_ACUTE_KEYS[i],    float(DEFAULT_TRUE_ACUTE[i]))
        st.session_state.setdefault(TRUE_SUBACUTE_KEYS[i], float(DEFAULT_TRUE_SUBACUTE[i]))
    st.session_state.setdefault("_results", None)

init_state()

# ============================================================
# Help text helpers
#
# h(key, meaning) looks up the default from R_DEFAULTS and
# appends it so every tooltip shows the R-aligned default.
# New widgets follow the same pattern.
# ============================================================

def h(key, meaning, r_name=None):
    r_def  = R_DEFAULTS.get(key, None)
    r_bits = []
    if r_name:
        r_bits.append(f"R: {r_name}")
    if r_def is not None:
        r_bits.append(f"Default: {r_def}")
    suffix = (" | " + " | ".join(r_bits)) if r_bits else ""
    return f"{meaning}{suffix}"

def h_true_acute(i):
    return (f"True acute DLT probability at dose level L{i}. "
            f"Default: {DEFAULT_TRUE_ACUTE[i]}")

def h_true_subacute(i):
    return (f"True subacute DLT probability at dose level L{i}. "
            f"Default: {DEFAULT_TRUE_SUBACUTE[i]}")

# ============================================================
# Essentials
# ============================================================

with st.expander("Essentials", expanded=False):
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("#### Study")
        st.number_input(
            "Target acute DLT rate",
            min_value=0.05, max_value=0.50, step=0.01, key="target_acute",
            help=h("target_acute",
                   "Target acute DLT probability used for MTD definition and CRM joint safety rule.",
                   r_name="target.acute")
        )
        st.number_input(
            "Target subacute DLT rate",
            min_value=0.05, max_value=0.50, step=0.01, key="target_subacute",
            help=h("target_subacute",
                   "Target subacute DLT probability used for MTD definition and CRM joint safety rule.",
                   r_name="target.subacute")
        )
        st.number_input(
            "Start dose level (1-based)",
            min_value=1, max_value=5, step=1, key="start_level_1b",
            help=h("start_level_1b",
                   "Starting dose level for both designs (entered as 1-based level).",
                   r_name="p (start dose index, 1-based)")
        )
        st.number_input(
            "Already treated at start dose (0 DLT)",
            min_value=0, max_value=500, step=1, key="already_treated_start",
            help=h("already_treated_start",
                   "Adds N patients treated at the start dose with 0 DLTs of either type before CRM begins updating.",
                   r_name="alreadytreated / pretreated at start dose")
        )

    with c2:
        st.markdown("#### Simulation")
        st.number_input(
            "Number of simulated trials",
            min_value=50, max_value=5000, step=50, key="n_sims",
            help=h("n_sims",
                   "Number of simulated trials (replicates) used to estimate operating characteristics.",
                   r_name="NREP")
        )
        st.number_input(
            "Random seed",
            min_value=1, max_value=10_000_000, step=1, key="seed",
            help=h("seed",
                   "Random seed for reproducibility.",
                   r_name="set.seed()")
        )

        st.markdown("#### CRM integration")
        st.selectbox(
            "Gauss–Hermite points",
            options=[31, 41, 61, 81], key="gh_n",
            help=h("gh_n",
                   "Number of Gauss–Hermite quadrature points for integrating the CRM posterior. Higher is more accurate but slower.",
                   r_name="gh.n / quadrature points")
        )
        st.selectbox(
            "Max dose step per update",
            options=[1, 2], key="max_step",
            help=h("max_step",
                   "Maximum number of dose levels the CRM can move up or down per cohort update.",
                   r_name="step.size / maxstep")
        )

    with c3:
        st.markdown("#### Sample size")
        st.number_input(
            "Maximum sample size (6+3)",
            min_value=6, max_value=200, step=3, key="max_n_63",
            help=h("max_n_63",
                   "Maximum total number of patients enrolled under the dual 6+3 design.",
                   r_name="N.patient (6+3)")
        )
        st.number_input(
            "Maximum sample size (CRM)",
            min_value=6, max_value=200, step=3, key="max_n_crm",
            help=h("max_n_crm",
                   "Maximum total number of patients enrolled under dual CRM.",
                   r_name="N.patient (CRM)")
        )
        st.number_input(
            "Cohort size",
            min_value=1, max_value=12, step=1, key="cohort_size",
            help=h("cohort_size",
                   "Number of patients per cohort update in CRM.",
                   r_name="CO")
        )

        st.markdown("#### CRM safety / selection")
        st.toggle(
            "Guardrail: next dose ≤ highest tried + 1",
            key="enforce_guardrail",
            help=h("enforce_guardrail",
                   "Prevents skipping untried dose levels by limiting escalation to at most one level above the highest tried dose.",
                   r_name="guardrail / no skipping")
        )
        st.toggle(
            "Final MTD must be among tried doses",
            key="restrict_final_mtd",
            help=h("restrict_final_mtd",
                   "Restricts final MTD selection to dose levels that were actually treated (n > 0).",
                   r_name="final.mtd.restrict.to.tried")
        )
        st.toggle(
            "Show debug (first simulated trial)",
            key="show_debug",
            help=h("show_debug",
                   "Shows detailed internals for the first simulated trial only, for both designs.",
                   r_name="debug")
        )

    # Figure sizing — inline section, no nested expander (Streamlit forbids nesting)
    st.markdown("#### Figure sizing")
    fs1, fs2 = st.columns(2, gap="large")
    with fs1:
        st.number_input(
            "Preview plot width (px)",
            min_value=180, max_value=500, step=10, key="preview_w_px",
            help=h("preview_w_px",
                   "Fixed pixel width for the small preview plot in the CRM knobs panel.",
                   r_name="(UI only)")
        )
    with fs2:
        st.number_input(
            "Results plot width (px)",
            min_value=220, max_value=600, step=10, key="result_w_px",
            help=h("result_w_px",
                   "Fixed pixel width for each results histogram.",
                   r_name="(UI only)")
        )

    st.write("")
    if st.button("Reset to defaults",
                 help="Resets all inputs — Essentials, Priors, CRM knobs, and true DLT curves — back to defaults."):
        st.session_state["_do_reset"] = True
        st.rerun()

# ============================================================
# Playground
# ============================================================

with st.expander("Playground", expanded=True):
    left, mid, right = st.columns([1.05, 1.10, 1.05], gap="large")

    # ---- Left: true DLT curves + Run button
    with left:
        st.markdown("#### True DLT probabilities")

        # Header row
        hL, hA, hS = st.columns([0.40, 0.30, 0.30], gap="small")
        with hA:
            st.markdown("<div style='font-size:0.80rem; font-weight:600;'>Acute</div>",
                        unsafe_allow_html=True)
        with hS:
            st.markdown("<div style='font-size:0.80rem; font-weight:600;'>Subacute</div>",
                        unsafe_allow_html=True)

        true_acute    = []
        true_subacute = []

        for i, lab in enumerate(dose_labels):
            rL, rA, rS = st.columns([0.40, 0.30, 0.30], gap="small")
            with rL:
                st.markdown(
                    f"<div style='font-size:0.84rem; padding-top:0.25rem;'>L{i} {lab}</div>",
                    unsafe_allow_html=True)
            with rA:
                va = st.number_input(
                    f"Acute L{i}", min_value=0.0, max_value=1.0, step=0.01,
                    key=TRUE_ACUTE_KEYS[i],
                    label_visibility="collapsed",
                    help=h_true_acute(i))
                true_acute.append(float(va))
            with rS:
                vs = st.number_input(
                    f"Subacute L{i}", min_value=0.0, max_value=1.0, step=0.01,
                    key=TRUE_SUBACUTE_KEYS[i],
                    label_visibility="collapsed",
                    help=h_true_subacute(i))
                true_subacute.append(float(vs))

        target_acute_val    = float(st.session_state["target_acute"])
        target_subacute_val = float(st.session_state["target_subacute"])
        true_safe = find_true_safe_dose(true_acute, true_subacute,
                                        target_acute_val, target_subacute_val)
        if true_safe is not None:
            st.caption(f"Highest jointly safe dose = L{true_safe}")
        else:
            st.caption("No dose satisfies both targets simultaneously.")

        st.write("")
        run = st.button("Run simulations", use_container_width=True)

    # ---- Mid: Priors (separate for each endpoint; shared model)
    with mid:
        st.markdown("#### Priors")

        # Shared prior model — acceptable per spec
        st.radio(
            "Skeleton model",
            options=["empiric", "logistic"],
            horizontal=True,
            key="prior_model",
            help=h("prior_model",
                   "Method used to generate both prior skeletons with dfcrm_getprior().",
                   r_name="skeleton model (empiric/logistic)")
        )

        prior_model_val  = str(st.session_state["prior_model"])
        intcpt_val       = float(st.session_state["logistic_intcpt"])

        # ---- Acute prior inputs
        st.markdown("**Acute prior**")

        st.slider(
            "Prior target (acute)",
            min_value=0.05, max_value=0.50, step=0.01,
            key="prior_target_acute",
            help=h("prior_target_acute",
                   "Target probability used when building the acute prior skeleton.",
                   r_name="prior.target.acute")
        )

        # Halfwidth clamping: done BEFORE the halfwidth_acute widget is created.
        # This is the only safe moment to write to a widget-backed session_state key.
        _pt_A    = float(st.session_state["prior_target_acute"])
        _max_hwA = max(0.01, min(0.30, _pt_A - 0.001, 1.0 - _pt_A - 0.001))
        if float(st.session_state["halfwidth_acute"]) > _max_hwA:
            st.session_state["halfwidth_acute"] = float(_max_hwA)

        st.slider(
            "Halfwidth (acute)",
            min_value=0.01, max_value=float(_max_hwA), step=0.01,
            key="halfwidth_acute",
            help=h("halfwidth_acute",
                   "Controls how steep the acute skeleton is. Must satisfy prior_target_acute ± halfwidth within (0,1).",
                   r_name="halfwidth.acute")
        )
        st.slider(
            "Prior MTD level (acute, 1-based)",
            min_value=1, max_value=5, step=1,
            key="prior_nu_acute",
            help=h("prior_nu_acute",
                   "Dose level assumed a priori to be closest to the acute target — anchor for the acute skeleton.",
                   r_name="prior.MTD.acute")
        )

        # ---- Subacute prior inputs
        st.markdown("**Subacute prior**")

        st.slider(
            "Prior target (subacute)",
            min_value=0.05, max_value=0.50, step=0.01,
            key="prior_target_subacute",
            help=h("prior_target_subacute",
                   "Target probability used when building the subacute prior skeleton.",
                   r_name="prior.target.subacute")
        )

        # Halfwidth clamping for subacute — again BEFORE its widget is created.
        _pt_S    = float(st.session_state["prior_target_subacute"])
        _max_hwS = max(0.01, min(0.30, _pt_S - 0.001, 1.0 - _pt_S - 0.001))
        if float(st.session_state["halfwidth_subacute"]) > _max_hwS:
            st.session_state["halfwidth_subacute"] = float(_max_hwS)

        st.slider(
            "Halfwidth (subacute)",
            min_value=0.01, max_value=float(_max_hwS), step=0.01,
            key="halfwidth_subacute",
            help=h("halfwidth_subacute",
                   "Controls how steep the subacute skeleton is. Must satisfy prior_target_subacute ± halfwidth within (0,1).",
                   r_name="halfwidth.subacute")
        )
        st.slider(
            "Prior MTD level (subacute, 1-based)",
            min_value=1, max_value=5, step=1,
            key="prior_nu_subacute",
            help=h("prior_nu_subacute",
                   "Dose level assumed a priori to be closest to the subacute target — anchor for the subacute skeleton.",
                   r_name="prior.MTD.subacute")
        )

        # Logistic intercept — shared, shown only when logistic model is selected
        if prior_model_val == "logistic":
            st.slider(
                "Logistic intercept",
                min_value=-10.0, max_value=10.0, step=0.1,
                key="logistic_intcpt",
                help=h("logistic_intcpt",
                       "Intercept for logistic skeleton construction in dfcrm_getprior(), shared for both endpoints.",
                       r_name="intcpt")
            )
            intcpt_val = float(st.session_state["logistic_intcpt"])

        # ---- Build skeletons (use local hw vars; never mutate widget keys here)
        hw_A_eff = float(st.session_state["halfwidth_acute"])
        try:
            skeleton_acute = dfcrm_getprior(
                halfwidth=hw_A_eff,
                target=_pt_A,
                nu=int(st.session_state["prior_nu_acute"]),
                nlevel=5, model=prior_model_val, intcpt=intcpt_val,
            ).tolist()
        except ValueError as e:
            st.warning(f"Acute skeleton: {e}")
            hw_A_eff       = float(min(0.10, _max_hwA))
            skeleton_acute = dfcrm_getprior(
                halfwidth=hw_A_eff, target=_pt_A,
                nu=int(st.session_state["prior_nu_acute"]),
                nlevel=5, model=prior_model_val, intcpt=intcpt_val,
            ).tolist()

        hw_S_eff = float(st.session_state["halfwidth_subacute"])
        try:
            skeleton_subacute = dfcrm_getprior(
                halfwidth=hw_S_eff,
                target=_pt_S,
                nu=int(st.session_state["prior_nu_subacute"]),
                nlevel=5, model=prior_model_val, intcpt=intcpt_val,
            ).tolist()
        except ValueError as e:
            st.warning(f"Subacute skeleton: {e}")
            hw_S_eff          = float(min(0.10, _max_hwS))
            skeleton_subacute = dfcrm_getprior(
                halfwidth=hw_S_eff, target=_pt_S,
                nu=int(st.session_state["prior_nu_subacute"]),
                nlevel=5, model=prior_model_val, intcpt=intcpt_val,
            ).tolist()

        # Numeric skeleton values for inspection
        with st.expander("Skeleton values", expanded=False):
            st.caption("Acute:    " + "  ".join(f"L{i}={v:.3f}" for i, v in enumerate(skeleton_acute)))
            st.caption("Subacute: " + "  ".join(f"L{i}={v:.3f}" for i, v in enumerate(skeleton_subacute)))

    # ---- Right: CRM knobs + preview plot
    with right:
        st.markdown("#### CRM knobs")

        st.slider(
            "Prior sigma on theta",
            min_value=0.2, max_value=5.0, step=0.1,
            key="sigma",
            help=h("sigma",
                   "Standard deviation of theta in the CRM prior: theta ~ Normal(0, sigma^2). "
                   "Shared for both endpoints. Larger sigma = weaker prior.",
                   r_name="prior.sigma / sigma")
        )
        st.toggle(
            "Burn-in until first DLT",
            key="burn_in",
            help=h("burn_in",
                   "If enabled, keep escalating one level at a time until the first DLT of either "
                   "endpoint is observed, then switch to CRM updates.",
                   r_name="burnin / burning.phase")
        )
        st.toggle(
            "Enable EWOC joint overdose control",
            key="ewoc_on",
            help=h("ewoc_on",
                   "If enabled, restricts dose choices to those where BOTH acute and subacute "
                   "posterior overdose probabilities are below EWOC alpha (joint safety rule).",
                   r_name="EWOC on/off")
        )
        st.slider(
            "EWOC alpha",
            min_value=0.01, max_value=0.99, step=0.01,
            key="ewoc_alpha",
            disabled=(not st.session_state["ewoc_on"]),
            help=h("ewoc_alpha",
                   "EWOC threshold applied to BOTH endpoints: dose k is allowed only if "
                   "P(acute_k > target_acute | data) < alpha AND "
                   "P(subacute_k > target_subacute | data) < alpha.",
                   r_name="alpha")
        )

        # Preview plot — fixed size, 4 curves (acute + subacute, true + skeleton)
        # plus horizontal reference lines for both targets.
        fig, ax = plt.subplots(figsize=(PREVIEW_W_IN, PREVIEW_H_IN), dpi=PREVIEW_DPI)
        x = np.arange(5)
        ax.plot(x, true_acute,     marker="o", linewidth=1.6, color="tab:blue",
                label="True acute")
        ax.plot(x, skeleton_acute, marker="o", linewidth=1.6, color="tab:blue",
                linestyle="--", label="Skel acute")
        ax.plot(x, true_subacute,     marker="s", linewidth=1.6, color="tab:orange",
                label="True sub")
        ax.plot(x, skeleton_subacute, marker="s", linewidth=1.6, color="tab:orange",
                linestyle="--", label="Skel sub")
        ax.axhline(target_acute_val,    linewidth=1, alpha=0.55, color="tab:blue")
        ax.axhline(target_subacute_val, linewidth=1, alpha=0.55, color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        _ymax = max(max(true_acute), max(skeleton_acute),
                    max(true_subacute), max(skeleton_subacute),
                    target_acute_val, target_subacute_val)
        ax.set_ylim(0, min(1.0, _ymax * 1.25 + 0.02))
        compact_style(ax)
        ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2)
        st.image(fig_to_png_bytes(fig), width=int(st.session_state["preview_w_px"]))

# ============================================================
# Run simulations (store results in session_state for persistence
# across reruns that don't press the button again)
# ============================================================

if "run" in locals() and run:
    rng = np.random.default_rng(int(st.session_state["seed"]))
    ns  = int(st.session_state["n_sims"])

    start_0b = int(np.clip(int(st.session_state["start_level_1b"]) - 1, 0, 4))

    sel_63  = np.zeros(5, dtype=int)
    sel_crm = np.zeros(5, dtype=int)

    nmat_63  = np.zeros((ns, 5), dtype=int)
    nmat_crm = np.zeros((ns, 5), dtype=int)

    # separate DLT counters per endpoint
    yA_63  = np.zeros(ns, dtype=int)
    yS_63  = np.zeros(ns, dtype=int)
    yA_crm = np.zeros(ns, dtype=int)
    yS_crm = np.zeros(ns, dtype=int)

    debug_dump_63  = None
    debug_dump_crm = None

    for s in range(ns):
        debug_flag = bool(st.session_state["show_debug"] and s == 0)

        chosen63, n63, yA63, yS63, dbg63 = run_dual_6plus3(
            true_acute=true_acute,
            true_subacute=true_subacute,
            start_level=start_0b,
            max_n=int(st.session_state["max_n_63"]),
            rng=rng,
            debug=debug_flag,
        )

        chosenc, nc, yAc, ySc, dbgc = run_dual_crm_trial(
            true_acute=true_acute,
            true_subacute=true_subacute,
            target_acute=float(st.session_state["target_acute"]),
            target_subacute=float(st.session_state["target_subacute"]),
            skeleton_acute=skeleton_acute,
            skeleton_subacute=skeleton_subacute,
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
            debug=debug_flag,
        )

        sel_63[chosen63]  += 1
        sel_crm[chosenc]  += 1
        nmat_63[s, :]      = n63
        nmat_crm[s, :]     = nc
        yA_63[s]           = yA63
        yS_63[s]           = yS63
        yA_crm[s]          = yAc
        yS_crm[s]          = ySc

        if debug_flag:
            debug_dump_63  = dbg63
            debug_dump_crm = dbgc

    mean_n63  = float(np.mean(nmat_63.sum(axis=1)))
    mean_ncrm = float(np.mean(nmat_crm.sum(axis=1)))

    st.session_state["_results"] = {
        "p63":                 sel_63  / float(ns),
        "pcrm":                sel_crm / float(ns),
        "avg63":               np.mean(nmat_63,  axis=0),
        "avgcrm":              np.mean(nmat_crm, axis=0),
        "mean_n63":            mean_n63,
        "mean_ncrm":           mean_ncrm,
        "acute_dlt_prob_63":   float(np.mean(yA_63)  / max(1e-9, mean_n63)),
        "subacute_dlt_prob_63":float(np.mean(yS_63)  / max(1e-9, mean_n63)),
        "acute_dlt_prob_crm":  float(np.mean(yA_crm) / max(1e-9, mean_ncrm)),
        "subacute_dlt_prob_crm":float(np.mean(yS_crm)/ max(1e-9, mean_ncrm)),
        "true_safe":           true_safe,
        "ns":                  ns,
        "seed":                int(st.session_state["seed"]),
        "show_debug":          bool(st.session_state["show_debug"]),
        "debug_dump_63":       debug_dump_63,
        "debug_dump_crm":      debug_dump_crm,
    }

# ============================================================
# Results — read from session_state so they survive reruns
# ============================================================

res = st.session_state.get("_results")
if res is not None:
    p63    = res["p63"]
    pcrm   = res["pcrm"]
    avg63  = res["avg63"]
    avgcrm = res["avgcrm"]
    ts     = res["true_safe"]   # int or None

    st.write("")
    r1, r2, r3 = st.columns([1.05, 1.05, 0.90], gap="large")

    with r1:
        # Fixed-size plot: P(select dose as MTD)
        fig, ax = plt.subplots(figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
        xx = np.arange(5)
        w  = 0.38
        ax.bar(xx - w/2, p63,  w, label="Dual 6+3")
        ax.bar(xx + w/2, pcrm, w, label="Dual CRM")
        ax.set_title("P(select dose as MTD)", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, max(p63.max(), pcrm.max()) * 1.15 + 1e-6)
        if ts is not None:
            ax.axvline(ts, linewidth=1, alpha=0.6, label=f"True safe=L{ts}")
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.image(fig_to_png_bytes(fig), width=int(st.session_state["result_w_px"]))

    with r2:
        # Fixed-size plot: avg patients per dose
        fig, ax = plt.subplots(figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
        xx = np.arange(5)
        w  = 0.38
        ax.bar(xx - w/2, avg63,  w, label="Dual 6+3")
        ax.bar(xx + w/2, avgcrm, w, label="Dual CRM")
        ax.set_title("Avg patients treated per dose", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Patients", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.image(fig_to_png_bytes(fig), width=int(st.session_state["result_w_px"]))

    with r3:
        st.metric("Acute DLT / patient (6+3)",   f"{res['acute_dlt_prob_63']:.3f}")
        st.metric("Acute DLT / patient (CRM)",   f"{res['acute_dlt_prob_crm']:.3f}")
        st.metric("Subacute DLT / patient (6+3)",f"{res['subacute_dlt_prob_63']:.3f}")
        st.metric("Subacute DLT / patient (CRM)",f"{res['subacute_dlt_prob_crm']:.3f}")
        st.caption(
            f"n_sims={res['ns']} | seed={res['seed']}"
            + (f" | True safe dose=L{ts}" if ts is not None else " | No jointly safe dose")
        )

    # ---- Debug output for first simulated trial
    if res["show_debug"]:
        if res["debug_dump_63"]:
            st.subheader("Dual 6+3 debug (first simulated trial)")
            for row in res["debug_dump_63"]:
                st.write(
                    f"L{row['level']} | {row['phase']} | "
                    f"acute={row['acute_dlts']} subacute={row['subacute_dlts']} "
                    f"→ {row['decision']}"
                )

        if res["debug_dump_crm"]:
            st.subheader("Dual CRM debug (first simulated trial)")
            for i, row in enumerate(res["debug_dump_crm"], start=1):
                st.write(
                    f"Update {i}: L{row['treated_level']} | n={row['cohort_n']} "
                    f"| acute_dlts={row['acute_dlts']} subacute_dlts={row['subacute_dlts']} "
                    f"| any_dlt={row['any_dlt_seen']}"
                )
                if "next_level" in row:
                    st.write(
                        f"  allowed: {row['allowed_levels']} | next: L{row['next_level']} "
                        f"| highest_tried={row['highest_tried']}"
                    )
                    st.write(f"  post_mean_acute:    {[round(v,3) for v in row['post_mean_acute']]}")
                    st.write(f"  post_mean_subacute: {[round(v,3) for v in row['post_mean_subacute']]}")
                    st.write(f"  od_prob_acute:      {[round(v,3) for v in row['od_prob_acute']]}")
                    st.write(f"  od_prob_subacute:   {[round(v,3) for v in row['od_prob_subacute']]}")
