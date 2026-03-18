import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# ============================================================
# 0) Fixed sizing knobs (ONE place to tune)
#    Figures are rendered at fixed inch size + dpi, then shown
#    at a fixed pixel width via st.image.
# ============================================================
PREVIEW_W_PX = 260
RESULT_W_PX = 460

PREVIEW_W_IN, PREVIEW_H_IN, PREVIEW_DPI = 3.8, 2.6, 170
RESULT_W_IN, RESULT_H_IN, RESULT_DPI = 6.0, 4.4, 170

# ============================================================
# Helpers
# ============================================================

def safe_probs(x):
    x = np.asarray(x, dtype=float)
    return np.clip(x, 1e-6, 1 - 1e-6)

def simulate_bernoulli(n, p, rng):
    return rng.binomial(1, p, size=int(n))

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)

def fig_to_png_bytes(fig):
    """
    Save fig as PNG bytes using the figure's fixed size + dpi.
    This keeps plot geometry stable across browser sizes.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def compute_halfwidth_upper_bound(target):
    """
    Maximum allowed halfwidth so that target ± halfwidth stays
    strictly inside (0, 1). Used for graceful clamping.
    """
    eps = 1e-3
    return max(0.001, min(0.30, float(target) - eps, 1.0 - float(target) - eps))

def clamp_halfwidth(hw, target):
    lower = 0.01
    upper = compute_halfwidth_upper_bound(target)
    clamped = min(max(float(hw), lower), upper)
    changed = not np.isclose(float(hw), clamped)
    return clamped, changed

def highest_true_safe_dose(true_acute, true_subacute, target_acute, target_subacute):
    """
    Reference dose used for plotting:
    highest dose whose true acute and true subacute risks are both
    within their respective targets.
    """
    true_acute = np.asarray(true_acute, dtype=float)
    true_subacute = np.asarray(true_subacute, dtype=float)
    safe = np.where(
        (true_acute <= float(target_acute)) &
        (true_subacute <= float(target_subacute))
    )[0]
    return int(safe.max()) if safe.size > 0 else 0

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
            b_k = np.log(
                (np.log((target + halfwidth) / (1 - target - halfwidth)) - intcpt)
                / dosescaled[k - 1]
            )
            dosescaled[k - 2] = (
                np.log((target - halfwidth) / (1 - target + halfwidth)) - intcpt
            ) / np.exp(b_k)
        for k in range(nu, nlevel):
            b_k1 = np.log(
                (np.log((target - halfwidth) / (1 - target + halfwidth)) - intcpt)
                / dosescaled[k - 1]
            )
            dosescaled[k] = (
                np.log((target + halfwidth) / (1 - target - halfwidth)) - intcpt
            ) / np.exp(b_k1)
        prior = (1 + np.exp(-intcpt - dosescaled)) ** (-1)
        return prior

    raise ValueError('model must be "empiric" or "logistic".')

# ============================================================
# Dual 6+3 design
# Protocol rule interpreted from the provided flowchart:
#   After first 6 at a dose:
#     escalate if acute == 0 and subacute <= 1
#     stop      if acute >= 2 or subacute >= 3
#     else add 3 more at same dose
#
#   After 9 total at a dose:
#     escalate if acute <= 1 and subacute <= 3
#     stop      if acute >= 2 or subacute >= 4
# ============================================================

def run_6plus3_dual(
    true_acute,
    true_subacute,
    start_level=1,
    max_n=27,
    rng=None,
    debug=False,
):
    if rng is None:
        rng = np.random.default_rng()

    true_acute = np.asarray(true_acute, dtype=float)
    true_subacute = np.asarray(true_subacute, dtype=float)
    n_levels = len(true_acute)

    level = int(start_level)
    n_per = np.zeros(n_levels, dtype=int)
    yA_per = np.zeros(n_levels, dtype=int)
    yS_per = np.zeros(n_levels, dtype=int)

    total_n = 0
    last_acceptable = None
    stop_early = False
    debug_rows = []

    while total_n < int(max_n):
        # Stage 1: treat first 6 at current dose
        n_add = min(6, int(max_n) - total_n)
        outA = simulate_bernoulli(n_add, true_acute[level], rng)
        outS = simulate_bernoulli(n_add, true_subacute[level], rng)

        n_per[level] += n_add
        yA_per[level] += int(outA.sum())
        yS_per[level] += int(outS.sum())
        total_n += n_add

        if debug:
            debug_rows.append({
                "dose_level": int(level),
                "stage": "first_6",
                "n_added": int(n_add),
                "acute_added": int(outA.sum()),
                "subacute_added": int(outS.sum()),
                "acute_total_here": int(yA_per[level]),
                "subacute_total_here": int(yS_per[level]),
                "n_total_here": int(n_per[level]),
            })

        if n_add < 6:
            break

        acute6 = int(yA_per[level])
        sub6 = int(yS_per[level])

        if acute6 == 0 and sub6 <= 1:
            last_acceptable = int(level)
            if level < n_levels - 1:
                if debug:
                    debug_rows[-1]["decision"] = "escalate_after_6"
                level += 1
                continue
            if debug:
                debug_rows[-1]["decision"] = "top_dose_acceptable_after_6"
            break

        if acute6 >= 2 or sub6 >= 3:
            stop_early = True
            if debug:
                debug_rows[-1]["decision"] = "stop_after_6"
            if level > 0:
                level -= 1
            break

        # Borderline zone: add 3 more at same dose
        n_add2 = min(3, int(max_n) - total_n)
        outA2 = simulate_bernoulli(n_add2, true_acute[level], rng)
        outS2 = simulate_bernoulli(n_add2, true_subacute[level], rng)

        n_per[level] += n_add2
        yA_per[level] += int(outA2.sum())
        yS_per[level] += int(outS2.sum())
        total_n += n_add2

        if debug:
            debug_rows.append({
                "dose_level": int(level),
                "stage": "add_3_to_9",
                "n_added": int(n_add2),
                "acute_added": int(outA2.sum()),
                "subacute_added": int(outS2.sum()),
                "acute_total_here": int(yA_per[level]),
                "subacute_total_here": int(yS_per[level]),
                "n_total_here": int(n_per[level]),
            })

        if n_add2 < 3:
            break

        acute9 = int(yA_per[level])
        sub9 = int(yS_per[level])

        if acute9 <= 1 and sub9 <= 3:
            last_acceptable = int(level)
            if level < n_levels - 1:
                if debug:
                    debug_rows[-1]["decision"] = "escalate_after_9"
                level += 1
                continue
            if debug:
                debug_rows[-1]["decision"] = "top_dose_acceptable_after_9"
            break

        stop_early = True
        if debug:
            debug_rows[-1]["decision"] = "stop_after_9"
        if level > 0:
            level -= 1
        break

    selected = 0 if last_acceptable is None else int(last_acceptable)

    return {
        "selected": selected,
        "n_per": n_per,
        "acute_total": int(yA_per.sum()),
        "subacute_total": int(yS_per.sum()),
        "acute_per": yA_per,
        "subacute_per": yS_per,
        "debug_rows": debug_rows,
        "stopped_early": bool(stop_early),
    }

# ============================================================
# CRM posterior via Gauss-Hermite quadrature (single endpoint)
# ============================================================

def posterior_via_gh(sigma, skeleton, n_per_level, dlt_per_level, gh_n=61):
    sk = safe_probs(skeleton)
    n = np.asarray(n_per_level, dtype=float)
    y = np.asarray(dlt_per_level, dtype=float)

    x, w = np.polynomial.hermite.hermgauss(int(gh_n))
    theta = float(sigma) * np.sqrt(2.0) * x

    P = sk[None, :] ** np.exp(theta)[:, None]
    P = safe_probs(P)

    ll = (
        y[None, :] * np.log(P)
        + (n[None, :] - y[None, :]) * np.log(1 - P)
    ).sum(axis=1)

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

# ============================================================
# Dual-endpoint CRM
# Rule:
#   a dose is allowed only if BOTH endpoints are safe
#   among allowed doses, choose the highest safe dose
#   then apply max-step and highest-tried+1 guardrail
# ============================================================

def crm_choose_next_dual(
    sigma,
    skeleton_acute,
    skeleton_subacute,
    n_per_level,
    yA_per_level,
    yS_per_level,
    current_level,
    target_acute,
    target_subacute,
    ewoc_alpha,
    max_step=1,
    gh_n=61,
    enforce_highest_tried_plus_one=True,
    highest_tried=None,
):
    if len(n_per_level) != len(skeleton_acute) or len(n_per_level) != len(skeleton_subacute):
        raise ValueError("Mismatch in number of dose levels.")

    post_mean_A, od_A = crm_posterior_summaries(
        sigma, skeleton_acute, n_per_level, yA_per_level, target_acute, gh_n=gh_n
    )
    post_mean_S, od_S = crm_posterior_summaries(
        sigma, skeleton_subacute, n_per_level, yS_per_level, target_subacute, gh_n=gh_n
    )

    if ewoc_alpha is None:
        allowed = np.arange(len(skeleton_acute))
    else:
        allowed = np.where((od_A < float(ewoc_alpha)) & (od_S < float(ewoc_alpha)))[0]

    if allowed.size == 0:
        allowed = np.array([0], dtype=int)

    k_star = int(np.max(allowed))
    k_star = int(np.clip(k_star, current_level - int(max_step), current_level + int(max_step)))

    if enforce_highest_tried_plus_one and highest_tried is not None:
        k_star = int(min(k_star, int(highest_tried) + 1))

    k_star = int(np.clip(k_star, 0, len(skeleton_acute) - 1))

    return k_star, post_mean_A, post_mean_S, od_A, od_S, allowed

def crm_select_mtd_dual(
    sigma,
    skeleton_acute,
    skeleton_subacute,
    n_per_level,
    yA_per_level,
    yS_per_level,
    target_acute,
    target_subacute,
    ewoc_alpha=None,
    gh_n=61,
    restrict_to_tried=True,
):
    post_mean_A, od_A = crm_posterior_summaries(
        sigma, skeleton_acute, n_per_level, yA_per_level, target_acute, gh_n=gh_n
    )
    post_mean_S, od_S = crm_posterior_summaries(
        sigma, skeleton_subacute, n_per_level, yS_per_level, target_subacute, gh_n=gh_n
    )

    if ewoc_alpha is None:
        allowed = np.arange(len(skeleton_acute))
    else:
        allowed = np.where((od_A < float(ewoc_alpha)) & (od_S < float(ewoc_alpha)))[0]

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

    return int(np.max(allowed))

def run_crm_trial_dual(
    true_acute,
    true_subacute,
    target_acute,
    target_subacute,
    skeleton_acute,
    skeleton_subacute,
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
    burn_in_until_first_tox=True,
    rng=None,
    debug=False,
):
    if rng is None:
        rng = np.random.default_rng()

    true_acute = np.asarray(true_acute, dtype=float)
    true_subacute = np.asarray(true_subacute, dtype=float)
    n_levels = len(true_acute)

    level = int(start_level)
    n_per = np.zeros(n_levels, dtype=int)
    yA_per = np.zeros(n_levels, dtype=int)
    yS_per = np.zeros(n_levels, dtype=int)

    already_treated_start = int(max(0, already_treated_start))
    if already_treated_start > 0:
        n_per[level] += already_treated_start

    highest_tried = level if already_treated_start > 0 else -1
    any_tox_seen = False
    debug_rows = []

    burn_in_active = bool(burn_in_until_first_tox and already_treated_start == 0)

    while int(n_per.sum()) < int(max_n):
        n_add = min(int(cohort_size), int(max_n) - int(n_per.sum()))
        outA = simulate_bernoulli(n_add, true_acute[level], rng)
        outS = simulate_bernoulli(n_add, true_subacute[level], rng)

        n_per[level] += n_add
        yA_per[level] += int(outA.sum())
        yS_per[level] += int(outS.sum())
        highest_tried = max(highest_tried, level)

        if int(outA.sum()) > 0 or int(outS.sum()) > 0:
            any_tox_seen = True

        if debug:
            debug_rows.append({
                "treated_level": int(level),
                "cohort_n": int(n_add),
                "acute_dlts": int(outA.sum()),
                "subacute_dlts": int(outS.sum()),
                "any_tox_seen": bool(any_tox_seen),
                "n_per": [int(x) for x in n_per],
                "yA_per": [int(x) for x in yA_per],
                "yS_per": [int(x) for x in yS_per],
            })

        if n_add < int(cohort_size):
            break

        if burn_in_active and (not any_tox_seen):
            if level < n_levels - 1:
                if debug:
                    debug_rows[-1]["decision"] = "burn_in_escalate"
                level += 1
                continue

        ewoc_alpha_eff = float(ewoc_alpha) if ewoc_on else None
        next_level, post_mean_A, post_mean_S, od_A, od_S, allowed = crm_choose_next_dual(
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
                "next_level": int(next_level),
                "allowed_levels": ",".join([str(int(a)) for a in allowed]),
                "highest_tried": int(highest_tried),
                "post_mean_acute": [float(x) for x in post_mean_A],
                "post_mean_subacute": [float(x) for x in post_mean_S],
                "od_acute": [float(x) for x in od_A],
                "od_subacute": [float(x) for x in od_S],
                "decision": "crm_update",
            })

        level = int(next_level)

    ewoc_alpha_eff = float(ewoc_alpha) if ewoc_on else None
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

    return {
        "selected": int(selected),
        "n_per": n_per,
        "acute_total": int(yA_per.sum()),
        "subacute_total": int(yS_per.sum()),
        "acute_per": yA_per,
        "subacute_per": yS_per,
        "debug_rows": debug_rows,
    }

# ============================================================
# Streamlit config + CSS
# ============================================================

st.set_page_config(
    page_title="Dual-tox 6+3 vs CRM",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none; }
      [data-testid="stSidebarNav"] { display: none; }
      [data-testid="collapsedControl"] { display: none; }

      .block-container { padding-top: 2.8rem; padding-bottom: 0.9rem; }
      .element-container { margin-bottom: 0.20rem; }

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

DEFAULT_TRUE_ACUTE = [0.01, 0.03, 0.08, 0.14, 0.24]
DEFAULT_TRUE_SUBACUTE = [0.05, 0.10, 0.18, 0.28, 0.40]

R_DEFAULTS = {
    "target_acute": 0.10,
    "target_subacute": 0.25,

    "start_level_1b": 2,
    "already_treated_start": 0,

    "n_sims": 200,
    "seed": 123,

    "max_n_63": 27,
    "max_n_crm": 27,
    "cohort_size": 3,

    "prior_model": "empiric",

    "prior_target_acute": 0.10,
    "prior_target_subacute": 0.25,

    "halfwidth_acute": 0.08,
    "halfwidth_subacute": 0.10,

    "prior_nu_acute": 3,
    "prior_nu_subacute": 3,

    "logistic_intcpt": 3.0,

    "sigma": 1.0,
    "burn_in": True,
    "ewoc_on": False,
    "ewoc_alpha": 0.25,

    "gh_n": 61,
    "max_step": 1,
    "enforce_guardrail": True,
    "restrict_final_mtd": True,
    "show_debug": False,

    "preview_w_px": PREVIEW_W_PX,
    "result_w_px": RESULT_W_PX,
}

TRUE_A_KEYS = [f"trueA_{i}" for i in range(5)]
TRUE_S_KEYS = [f"trueS_{i}" for i in range(5)]

RESULTS_STATE_KEY = "_last_results"
NOTICE_STATE_KEY = "_ui_notice"
RESET_FLAG_KEY = "_do_reset"

def init_state():
    for k, v in R_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    for i in range(5):
        st.session_state.setdefault(TRUE_A_KEYS[i], float(DEFAULT_TRUE_ACUTE[i]))
        st.session_state.setdefault(TRUE_S_KEYS[i], float(DEFAULT_TRUE_SUBACUTE[i]))
    st.session_state.setdefault(RESULTS_STATE_KEY, None)
    st.session_state.setdefault(NOTICE_STATE_KEY, [])

def reset_all_state():
    """
    Rock-solid reset logic.
    Resets widgets, derived values, stored results, and UI notices.
    Executed before widgets are created to avoid session_state issues.
    """
    for k, v in R_DEFAULTS.items():
        st.session_state[k] = v
    for i in range(5):
        st.session_state[TRUE_A_KEYS[i]] = float(DEFAULT_TRUE_ACUTE[i])
        st.session_state[TRUE_S_KEYS[i]] = float(DEFAULT_TRUE_SUBACUTE[i])

    st.session_state[RESULTS_STATE_KEY] = None
    st.session_state[NOTICE_STATE_KEY] = []
    st.session_state[RESET_FLAG_KEY] = False

init_state()

if st.session_state.get(RESET_FLAG_KEY, False):
    reset_all_state()
    st.rerun()

# ============================================================
# Help text helpers
# Preserving help tooltips is mandatory.
# ============================================================

def h(key, meaning, r_name=None):
    r_def = R_DEFAULTS.get(key, None)
    r_bits = []
    if r_name:
        r_bits.append(f"R-style name: {r_name}")
    if r_def is not None:
        r_bits.append(f"Default: {r_def}")
    suffix = (" | " + " | ".join(r_bits)) if r_bits else ""
    return f"{meaning}{suffix}"

def h_true_acute(i):
    return f"True acute toxicity probability at dose level L{i}. Default scenario: {DEFAULT_TRUE_ACUTE[i]}"

def h_true_subacute(i):
    return f"True subacute toxicity probability at dose level L{i}. Default scenario: {DEFAULT_TRUE_SUBACUTE[i]}"

# ============================================================
# Essentials
# ============================================================

with st.expander("Essentials", expanded=False):
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("#### Study")
        st.number_input(
            "Target acute toxicity",
            min_value=0.05, max_value=0.50, step=0.01, key="target_acute",
            help=h(
                "target_acute",
                "Target acute toxicity probability used in the dual-endpoint CRM safety rule.",
                r_name="target acute"
            )
        )
        st.number_input(
            "Target subacute toxicity",
            min_value=0.05, max_value=0.60, step=0.01, key="target_subacute",
            help=h(
                "target_subacute",
                "Target subacute toxicity probability used in the dual-endpoint CRM safety rule.",
                r_name="target subacute"
            )
        )
        st.number_input(
            "Start dose level (1-based)",
            min_value=1, max_value=5, step=1, key="start_level_1b",
            help=h(
                "start_level_1b",
                "Starting dose level for both designs, entered as a 1-based level.",
                r_name="p (start dose index, 1-based)"
            )
        )
        st.number_input(
            "Already treated at start dose (0 acute, 0 subacute)",
            min_value=0, max_value=500, step=1, key="already_treated_start",
            help=h(
                "already_treated_start",
                "Adds N patients treated at the CRM start dose with 0 acute and 0 subacute toxicity before CRM begins updating.",
                r_name="alreadytreated / pretreated at start dose"
            )
        )

    with c2:
        st.markdown("#### Simulation")
        st.number_input(
            "Number of simulated trials",
            min_value=50, max_value=5000, step=50, key="n_sims",
            help=h(
                "n_sims",
                "Number of simulated trials used to estimate operating characteristics.",
                r_name="NREP"
            )
        )
        st.number_input(
            "Random seed",
            min_value=1, max_value=10_000_000, step=1, key="seed",
            help=h(
                "seed",
                "Random seed for reproducibility.",
                r_name="set.seed()"
            )
        )

        st.markdown("#### CRM integration")
        st.selectbox(
            "Gauss-Hermite points",
            options=[31, 41, 61, 81], key="gh_n",
            help=h(
                "gh_n",
                "Number of Gauss-Hermite quadrature points for integrating each CRM posterior. Higher is more accurate but slower.",
                r_name="gh.n / quadrature points"
            )
        )
        st.selectbox(
            "Max dose step per update",
            options=[1, 2], key="max_step",
            help=h(
                "max_step",
                "Maximum number of dose levels the CRM can move up or down per cohort update.",
                r_name="step.size / maxstep"
            )
        )

    with c3:
        st.markdown("#### Sample size")
        st.number_input(
            "Maximum sample size (dual 6+3)",
            min_value=6, max_value=200, step=3, key="max_n_63",
            help=h(
                "max_n_63",
                "Maximum total number of patients enrolled under the dual-endpoint 6+3 design.",
                r_name="N.patient (6+3)"
            )
        )
        st.number_input(
            "Maximum sample size (dual CRM)",
            min_value=6, max_value=200, step=3, key="max_n_crm",
            help=h(
                "max_n_crm",
                "Maximum total number of patients enrolled under the dual-endpoint CRM design.",
                r_name="N.patient (CRM)"
            )
        )
        st.number_input(
            "Cohort size (CRM)",
            min_value=1, max_value=12, step=1, key="cohort_size",
            help=h(
                "cohort_size",
                "Number of patients per cohort update in CRM.",
                r_name="CO"
            )
        )

        st.markdown("#### CRM safety / selection")
        st.toggle(
            "Guardrail: next dose ≤ highest tried + 1",
            key="enforce_guardrail",
            help=h(
                "enforce_guardrail",
                "Prevents skipping untried dose levels by limiting escalation to at most one level above the highest tried dose.",
                r_name="guardrail / no skipping"
            )
        )
        st.toggle(
            "Final selected dose must be among tried doses",
            key="restrict_final_mtd",
            help=h(
                "restrict_final_mtd",
                "Restricts final dual-CRM selected dose to levels that were actually treated.",
                r_name="final.mtd.restrict.to.tried"
            )
        )
        st.toggle(
            "Show debug for first simulated trial",
            key="show_debug",
            help=h(
                "show_debug",
                "Shows detailed internals for the first simulated trial only, including both endpoints and CRM safety sets.",
                r_name="debug"
            )
        )

    with st.expander("Figure sizing", expanded=False):
        st.number_input(
            "Preview plot width (px)",
            min_value=180, max_value=520, step=10, key="preview_w_px",
            help=h(
                "preview_w_px",
                "Fixed pixel width for the preview plot.",
                r_name="(UI only)"
            )
        )
        st.number_input(
            "Results plot width (px)",
            min_value=220, max_value=650, step=10, key="result_w_px",
            help=h(
                "result_w_px",
                "Fixed pixel width for each results histogram.",
                r_name="(UI only)"
            )
        )

    st.write("")
    if st.button(
        "Reset to defaults",
        help="Resets Essentials, Priors, CRM knobs, and both true toxicity curves back to the defaults defined in this script."
    ):
        st.session_state[RESET_FLAG_KEY] = True
        st.rerun()

# ============================================================
# Playground
# ============================================================

with st.expander("Playground", expanded=True):
    left, mid, right = st.columns([1.12, 1.05, 1.10], gap="large")

    # ---- Left: true toxicity curves + run button
    with left:
        st.markdown("#### True toxicity")

        hdrL, hdrA, hdrS = st.columns([0.38, 0.31, 0.31], gap="small")
        with hdrA:
            st.markdown("**Acute**")
        with hdrS:
            st.markdown("**Subacute**")

        true_acute = []
        true_subacute = []

        for i, lab in enumerate(dose_labels):
            rL, rA, rS = st.columns([0.38, 0.31, 0.31], gap="small")
            with rL:
                st.markdown(
                    f"<div style='font-size:0.86rem; padding-top:0.2rem;'>L{i} {lab}</div>",
                    unsafe_allow_html=True
                )
            with rA:
                a = st.number_input(
                    f"A{i}",
                    min_value=0.0, max_value=1.0, step=0.01,
                    key=f"trueA_{i}",
                    label_visibility="collapsed",
                    help=h_true_acute(i)
                )
                true_acute.append(float(a))
            with rS:
                s = st.number_input(
                    f"S{i}",
                    min_value=0.0, max_value=1.0, step=0.01,
                    key=f"trueS_{i}",
                    label_visibility="collapsed",
                    help=h_true_subacute(i)
                )
                true_subacute.append(float(s))

        ref_dose = highest_true_safe_dose(
            true_acute,
            true_subacute,
            st.session_state["target_acute"],
            st.session_state["target_subacute"],
        )
        st.caption(
            f"True reference dose = L{ref_dose} "
            f"(highest dose with true acute ≤ target acute and true subacute ≤ target subacute)"
        )

        st.write("")
        run = st.button(
            "Run simulations",
            use_container_width=True,
            help="Runs simulated trials for both the dual 6+3 design and the dual-endpoint CRM using the current inputs, then updates the results below."
        )

    # ---- Mid: priors
    with mid:
        st.markdown("#### Priors")

        st.radio(
            "Skeleton model",
            options=["empiric", "logistic"],
            horizontal=True,
            key="prior_model",
            help=h(
                "prior_model",
                "Method used to generate both endpoint-specific prior skeletons with dfcrm_getprior().",
                r_name="skeleton model (empiric/logistic)"
            )
        )

        st.markdown("**Acute prior**")
        st.slider(
            "Prior target acute",
            min_value=0.05, max_value=0.50, step=0.01,
            key="prior_target_acute",
            help=h(
                "prior_target_acute",
                "Target probability used when building the acute prior skeleton.",
                r_name="prior.target.acute"
            )
        )

        acute_hw, acute_changed = clamp_halfwidth(
            st.session_state["halfwidth_acute"],
            st.session_state["prior_target_acute"]
        )
        if acute_changed:
            st.session_state["halfwidth_acute"] = float(acute_hw)
            st.session_state[NOTICE_STATE_KEY].append(
                f"Acute halfwidth was adjusted to {acute_hw:.2f} so that prior target acute ± halfwidth stays within (0, 1)."
            )

        st.slider(
            "Halfwidth acute",
            min_value=0.01,
            max_value=float(compute_halfwidth_upper_bound(st.session_state["prior_target_acute"])),
            step=0.01,
            key="halfwidth_acute",
            help=h(
                "halfwidth_acute",
                "Controls how steep the acute prior skeleton is around the acute prior MTD. Gracefully clamped if target ± halfwidth would leave (0, 1).",
                r_name="halfwidth acute"
            )
        )
        st.slider(
            "Prior MTD acute (1-based)",
            min_value=1, max_value=5, step=1,
            key="prior_nu_acute",
            help=h(
                "prior_nu_acute",
                "Dose level assumed, a priori, to be closest to the acute target.",
                r_name="prior.MTD.acute"
            )
        )

        st.markdown("**Subacute prior**")
        st.slider(
            "Prior target subacute",
            min_value=0.05, max_value=0.60, step=0.01,
            key="prior_target_subacute",
            help=h(
                "prior_target_subacute",
                "Target probability used when building the subacute prior skeleton.",
                r_name="prior.target.subacute"
            )
        )

        sub_hw, sub_changed = clamp_halfwidth(
            st.session_state["halfwidth_subacute"],
            st.session_state["prior_target_subacute"]
        )
        if sub_changed:
            st.session_state["halfwidth_subacute"] = float(sub_hw)
            st.session_state[NOTICE_STATE_KEY].append(
                f"Subacute halfwidth was adjusted to {sub_hw:.2f} so that prior target subacute ± halfwidth stays within (0, 1)."
            )

        st.slider(
            "Halfwidth subacute",
            min_value=0.01,
            max_value=float(compute_halfwidth_upper_bound(st.session_state["prior_target_subacute"])),
            step=0.01,
            key="halfwidth_subacute",
            help=h(
                "halfwidth_subacute",
                "Controls how steep the subacute prior skeleton is around the subacute prior MTD. Gracefully clamped if target ± halfwidth would leave (0, 1).",
                r_name="halfwidth subacute"
            )
        )
        st.slider(
            "Prior MTD subacute (1-based)",
            min_value=1, max_value=5, step=1,
            key="prior_nu_subacute",
            help=h(
                "prior_nu_subacute",
                "Dose level assumed, a priori, to be closest to the subacute target.",
                r_name="prior.MTD.subacute"
            )
        )

        if st.session_state["prior_model"] == "logistic":
            st.slider(
                "Logistic intercept",
                min_value=-10.0, max_value=10.0, step=0.1,
                key="logistic_intcpt",
                help=h(
                    "logistic_intcpt",
                    "Intercept used only for logistic skeleton construction in dfcrm_getprior().",
                    r_name="intcpt"
                )
            )

        notices = st.session_state.get(NOTICE_STATE_KEY, [])
        if notices:
            for msg in notices:
                st.info(msg)
            st.session_state[NOTICE_STATE_KEY] = []

        # Final safety net before building skeletons
        st.session_state["halfwidth_acute"] = float(
            clamp_halfwidth(st.session_state["halfwidth_acute"], st.session_state["prior_target_acute"])[0]
        )
        st.session_state["halfwidth_subacute"] = float(
            clamp_halfwidth(st.session_state["halfwidth_subacute"], st.session_state["prior_target_subacute"])[0]
        )

        try:
            skeleton_acute = dfcrm_getprior(
                halfwidth=float(st.session_state["halfwidth_acute"]),
                target=float(st.session_state["prior_target_acute"]),
                nu=int(st.session_state["prior_nu_acute"]),
                nlevel=5,
                model=str(st.session_state["prior_model"]),
                intcpt=float(st.session_state["logistic_intcpt"]),
            ).tolist()

            skeleton_subacute = dfcrm_getprior(
                halfwidth=float(st.session_state["halfwidth_subacute"]),
                target=float(st.session_state["prior_target_subacute"]),
                nu=int(st.session_state["prior_nu_subacute"]),
                nlevel=5,
                model=str(st.session_state["prior_model"]),
                intcpt=float(st.session_state["logistic_intcpt"]),
            ).tolist()

        except ValueError as e:
            st.warning(str(e))
            st.session_state["halfwidth_acute"] = float(
                clamp_halfwidth(0.10, st.session_state["prior_target_acute"])[0]
            )
            st.session_state["halfwidth_subacute"] = float(
                clamp_halfwidth(0.10, st.session_state["prior_target_subacute"])[0]
            )

            skeleton_acute = dfcrm_getprior(
                halfwidth=float(st.session_state["halfwidth_acute"]),
                target=float(st.session_state["prior_target_acute"]),
                nu=int(st.session_state["prior_nu_acute"]),
                nlevel=5,
                model=str(st.session_state["prior_model"]),
                intcpt=float(st.session_state["logistic_intcpt"]),
            ).tolist()

            skeleton_subacute = dfcrm_getprior(
                halfwidth=float(st.session_state["halfwidth_subacute"]),
                target=float(st.session_state["prior_target_subacute"]),
                nu=int(st.session_state["prior_nu_subacute"]),
                nlevel=5,
                model=str(st.session_state["prior_model"]),
                intcpt=float(st.session_state["logistic_intcpt"]),
            ).tolist()

    # ---- Right: CRM knobs + preview
    with right:
        st.markdown("#### CRM knobs")

        st.slider(
            "Prior sigma on theta",
            min_value=0.2, max_value=5.0, step=0.1,
            key="sigma",
            help=h(
                "sigma",
                "Standard deviation of theta in both CRM priors: theta ~ Normal(0, sigma^2). Larger sigma means weaker priors around the skeletons.",
                r_name="prior.sigma / sigma"
            )
        )
        st.toggle(
            "Burn-in until first observed toxicity",
            key="burn_in",
            help=h(
                "burn_in",
                "If enabled, keep escalating one level at a time until the first observed acute or subacute toxicity, then switch to CRM updates.",
                r_name="burnin / burning.phase"
            )
        )
        st.toggle(
            "Enable EWOC safety filter",
            key="ewoc_on",
            help=h(
                "ewoc_on",
                "If enabled, a dose is allowed only if both endpoint-specific posterior overdose probabilities are below EWOC alpha.",
                r_name="EWOC on/off"
            )
        )
        st.slider(
            "EWOC alpha",
            min_value=0.01, max_value=0.99, step=0.01,
            key="ewoc_alpha",
            disabled=(not st.session_state["ewoc_on"]),
            help=h(
                "ewoc_alpha",
                "Shared EWOC threshold used for both endpoints: allow dose k only if P(acute_k > target_acute | data) < alpha and P(subacute_k > target_subacute | data) < alpha.",
                r_name="alpha"
            )
        )

        fig, ax = plt.subplots(figsize=(PREVIEW_W_IN, PREVIEW_H_IN), dpi=PREVIEW_DPI)
        x = np.arange(5)
        ax.plot(x, true_acute, marker="o", linewidth=1.6, label="True acute")
        ax.plot(x, true_subacute, marker="o", linewidth=1.6, label="True subacute")
        ax.plot(x, skeleton_acute, marker="o", linestyle="--", linewidth=1.3, label="Prior acute")
        ax.plot(x, skeleton_subacute, marker="o", linestyle="--", linewidth=1.3, label="Prior subacute")
        ax.axhline(float(st.session_state["target_acute"]), linewidth=1, alpha=0.6)
        ax.axhline(float(st.session_state["target_subacute"]), linewidth=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ymax = max(
            max(true_acute),
            max(true_subacute),
            max(skeleton_acute),
            max(skeleton_subacute),
            float(st.session_state["target_acute"]),
            float(st.session_state["target_subacute"]),
        )
        ax.set_ylim(0, min(1.0, ymax * 1.20 + 0.03))
        compact_style(ax)
        ax.legend(fontsize=7.7, frameon=False, loc="upper left")
        st.image(
            fig_to_png_bytes(fig),
            width=int(st.session_state["preview_w_px"]),
            use_container_width=False,
        )

# ============================================================
# Simulation wrapper
# ============================================================

def run_all_simulations(
    true_acute,
    true_subacute,
    skeleton_acute,
    skeleton_subacute,
    ref_dose,
):
    rng = np.random.default_rng(int(st.session_state["seed"]))
    ns = int(st.session_state["n_sims"])

    start_0b = int(st.session_state["start_level_1b"]) - 1
    start_0b = int(np.clip(start_0b, 0, 4))

    sel_63 = np.zeros(5, dtype=int)
    sel_crm = np.zeros(5, dtype=int)

    nmat_63 = np.zeros((ns, 5), dtype=int)
    nmat_crm = np.zeros((ns, 5), dtype=int)

    acute_63 = np.zeros(ns, dtype=int)
    acute_crm = np.zeros(ns, dtype=int)
    sub_63 = np.zeros(ns, dtype=int)
    sub_crm = np.zeros(ns, dtype=int)

    debug_dump_63 = None
    debug_dump_crm = None

    for s in range(ns):
        debug_flag = bool(st.session_state["show_debug"] and s == 0)

        out63 = run_6plus3_dual(
            true_acute=true_acute,
            true_subacute=true_subacute,
            start_level=start_0b,
            max_n=int(st.session_state["max_n_63"]),
            rng=rng,
            debug=debug_flag,
        )

        outcrm = run_crm_trial_dual(
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
            burn_in_until_first_tox=bool(st.session_state["burn_in"]),
            rng=rng,
            debug=debug_flag,
        )

        sel_63[out63["selected"]] += 1
        sel_crm[outcrm["selected"]] += 1

        nmat_63[s, :] = out63["n_per"]
        nmat_crm[s, :] = outcrm["n_per"]

        acute_63[s] = out63["acute_total"]
        acute_crm[s] = outcrm["acute_total"]
        sub_63[s] = out63["subacute_total"]
        sub_crm[s] = outcrm["subacute_total"]

        if debug_flag:
            debug_dump_63 = out63["debug_rows"]
            debug_dump_crm = outcrm["debug_rows"]

    p63 = sel_63 / float(ns)
    pcrm = sel_crm / float(ns)

    avg63 = np.mean(nmat_63, axis=0)
    avgcrm = np.mean(nmat_crm, axis=0)

    mean_n63 = float(np.mean(nmat_63.sum(axis=1)))
    mean_ncrm = float(np.mean(nmat_crm.sum(axis=1)))

    acute_prob_63 = float(np.mean(acute_63) / max(1e-9, mean_n63))
    acute_prob_crm = float(np.mean(acute_crm) / max(1e-9, mean_ncrm))
    sub_prob_63 = float(np.mean(sub_63) / max(1e-9, mean_n63))
    sub_prob_crm = float(np.mean(sub_crm) / max(1e-9, mean_ncrm))

    return {
        "p63": p63,
        "pcrm": pcrm,
        "avg63": avg63,
        "avgcrm": avgcrm,
        "acute_prob_63": acute_prob_63,
        "acute_prob_crm": acute_prob_crm,
        "sub_prob_63": sub_prob_63,
        "sub_prob_crm": sub_prob_crm,
        "debug_dump_63": debug_dump_63,
        "debug_dump_crm": debug_dump_crm,
        "ns": ns,
        "ref_dose": int(ref_dose),
    }

if run:
    st.session_state[RESULTS_STATE_KEY] = run_all_simulations(
        true_acute=true_acute,
        true_subacute=true_subacute,
        skeleton_acute=skeleton_acute,
        skeleton_subacute=skeleton_subacute,
        ref_dose=ref_dose,
    )

# ============================================================
# Results
# No extra page, no extra wasted header.
# Results render directly below after clicking Run simulations.
# ============================================================

results = st.session_state.get(RESULTS_STATE_KEY, None)

if results is not None:
    st.write("")

    r1, r2, r3 = st.columns([1.05, 1.05, 0.90], gap="large")

    with r1:
        fig, ax = plt.subplots(figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
        xx = np.arange(5)
        w = 0.38
        ax.bar(xx - w / 2, results["p63"], w, label="Dual 6+3")
        ax.bar(xx + w / 2, results["pcrm"], w, label="Dual CRM")
        ax.set_title("P(select dose as final dose)", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, max(results["p63"].max(), results["pcrm"].max()) * 1.15 + 1e-6)
        ax.axvline(results["ref_dose"], linewidth=1, alpha=0.6)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.image(
            fig_to_png_bytes(fig),
            width=int(st.session_state["result_w_px"]),
            use_container_width=False,
        )

    with r2:
        fig, ax = plt.subplots(figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
        xx = np.arange(5)
        w = 0.38
        ax.bar(xx - w / 2, results["avg63"], w, label="Dual 6+3")
        ax.bar(xx + w / 2, results["avgcrm"], w, label="Dual CRM")
        ax.set_title("Avg patients treated per dose", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Patients", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.image(
            fig_to_png_bytes(fig),
            width=int(st.session_state["result_w_px"]),
            use_container_width=False,
        )

    with r3:
        st.metric("Acute prob per patient (6+3)", f"{results['acute_prob_63']:.3f}")
        st.metric("Acute prob per patient (CRM)", f"{results['acute_prob_crm']:.3f}")
        st.metric("Subacute prob per patient (6+3)", f"{results['sub_prob_63']:.3f}")
        st.metric("Subacute prob per patient (CRM)", f"{results['sub_prob_crm']:.3f}")
        st.caption(
            f"n_sims={results['ns']} | seed={int(st.session_state['seed'])} | "
            f"Reference marker=L{results['ref_dose']}"
        )

    if st.session_state["show_debug"]:
        if results["debug_dump_63"]:
            st.subheader("Dual 6+3 debug (first simulated trial)")
            for i, row in enumerate(results["debug_dump_63"], start=1):
                st.write(
                    f"Step {i}: dose L{row['dose_level']} | stage={row['stage']} | "
                    f"n_added={row['n_added']} | acute_added={row['acute_added']} | "
                    f"subacute_added={row['subacute_added']} | "
                    f"acute_total_here={row['acute_total_here']} | "
                    f"subacute_total_here={row['subacute_total_here']} | "
                    f"n_total_here={row['n_total_here']}"
                )
                if "decision" in row:
                    st.write(f"  decision: {row['decision']}")

        if results["debug_dump_crm"]:
            st.subheader("Dual CRM debug (first simulated trial)")
            for i, row in enumerate(results["debug_dump_crm"], start=1):
                st.write(
                    f"Update {i}: treated L{row['treated_level']} | n={row['cohort_n']} | "
                    f"acute={row['acute_dlts']} | subacute={row['subacute_dlts']} | "
                    f"any_tox_seen={row['any_tox_seen']}"
                )
                if "next_level" in row:
                    st.write(
                        f"  allowed: {row['allowed_levels']} | next: L{row['next_level']} | "
                        f"highest_tried={row['highest_tried']}"
                    )
                    st.write(f"  post_mean_acute: {[round(v, 3) for v in row['post_mean_acute']]}")
                    st.write(f"  post_mean_subacute: {[round(v, 3) for v in row['post_mean_subacute']]}")
                    st.write(f"  od_acute: {[round(v, 3) for v in row['od_acute']]}")
                    st.write(f"  od_subacute: {[round(v, 3) for v in row['od_subacute']]}")
