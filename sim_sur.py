"""
sim_sur.py — Conditional dual-endpoint dose-escalation simulator
=================================================================
Tox1 = acute toxicity   (observed in ALL treated patients)
Tox2 = subacute toxicity (observed ONLY in patients who undergo surgery)

Surgery probability
-------------------
A single global parameter p_surgery applies equally to all dose levels.
It represents the probability that a patient proceeds to surgery,
independent of dose assignment.

Critical modelling rule
-----------------------
Non-surgery patients are NOT counted as subacute = 0.
They are simply not evaluable for the subacute endpoint.
The CRM subacute model therefore uses:
    n_sub_per[d]  = number of patients at dose d who had surgery
    y_sub_per[d]  = subacute events among those patients
NOT the total number of treated patients.

Modified 6+3 comparator (NOT the standard protocol)
----------------------------------------------------
Standard 6+3 evaluates both endpoints on fixed cohorts of 6 (then 9)
patients.  That is not possible here because subacute toxicity is only
observable in surgery patients.  This design uses a HOLD rule instead.

HOLD rule
  Phase 1 fires once n_treated >= 6 AND n_surgery >= 6.
  Phase 2 fires once n_treated >= 9 AND n_surgery >= 9.
  Both acute and subacute counts accumulate continuously during the hold.

KEY DESIGN CONSEQUENCE — intentional and verified
  If p_surgery is low, reaching the surgery threshold may require far
  more than 6 (or 9) treated patients.  The acute DLT counts
  (y_acute_per[level]) at the time of evaluation therefore reflect ALL
  treated patients, not a fixed cohort of 6 or 9.
  Example: p_surgery = 0.40 → on average ~15 patients needed to obtain
  6 surgery-evaluable patients.  The phase-1 acute thresholds
  (a6_stop_min, a6_esc_max) are then applied to y_acute over 15 pts,
  not 6.  This is consistent with the HOLD rationale: the design
  prioritises subacute evaluability over cohort-size rigidity.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# ============================================================
# 0) Fixed sizing knobs (ONE place to tune)
#    Fixed pixel widths prevent plots from resizing with browser width.
# ============================================================
# Preview: larger defaults so the Playground preview is easier to read.
# Burn-in and EWOC controls moved to Essentials, freeing vertical space in
# the CRM panel and allowing a taller/wider preview figure.
PREVIEW_W_PX = 310
RESULT_W_PX  = 460

PREVIEW_W_IN, PREVIEW_H_IN, PREVIEW_DPI = 4.2, 5.0, 150
RESULT_W_IN,  RESULT_H_IN,  RESULT_DPI  = 6.0, 4.4, 170

# ============================================================
# Helpers
# ============================================================

def safe_probs(x):
    x = np.asarray(x, dtype=float)
    return np.clip(x, 1e-6, 1 - 1e-6)

def simulate_cohort(n, p_acute, p_surgery, p_sub_gs, rng):
    """
    Simulate n patients at one dose level.

    Returns (acute_events, surgery_count, subacute_events_in_surgery_pts).

    Non-surgery patients are NOT assigned subacute = 0.
    They are simply not included in the subacute count.
    The subacute binomial is only drawn for the surgery sub-group.
    """
    n = int(n)
    acute   = int(rng.binomial(n, float(p_acute),   1)[0])
    n_surg  = int(rng.binomial(n, float(p_surgery), 1)[0])
    y_sub   = int(rng.binomial(n_surg, float(p_sub_gs), 1)[0]) if n_surg > 0 else 0
    return acute, n_surg, y_sub

def find_true_safe_dose(true_acute, true_sub_gs, target_acute, target_subacute):
    """
    Highest dose where true_acute <= target_acute
    AND true_subacute_given_surgery <= target_subacute.
    Returns None if no such dose exists.
    """
    safe = [d for d in range(len(true_acute))
            if true_acute[d] <= target_acute and true_sub_gs[d] <= target_subacute]
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
# dfcrm getprior port (unchanged from sim.py)
# ============================================================

def dfcrm_getprior(halfwidth, target, nu, nlevel, model="empiric", intcpt=3.0):
    halfwidth = float(halfwidth); target = float(target)
    nu = int(nu); nlevel = int(nlevel); intcpt = float(intcpt)
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
        return (1 + np.exp(-intcpt - dosescaled)) ** (-1)
    raise ValueError('model must be "empiric" or "logistic".')

# ============================================================
# 6+3 comparator — acute toxicity decisions only
#
# Surgery and subacute are tracked descriptively but do NOT
# drive escalation/de-escalation decisions.  See module docstring
# for rationale.
# ============================================================

def run_6plus3_sur(
    true_acute, p_surgery, true_sub_gs,
    start_level=0, max_n=36,
    a6_esc_max=0,  a6_stop_min=2, a9_esc_max=1,
    s6_esc_max=1,  s6_stop_min=3, s9_esc_max=3, s9_stop_min=4,
    rng=None, debug=False,
):
    """
    Dual 6+3 with joint acute + subacute decision rules.

    Separate denominators
    ---------------------
    acute   : all treated patients at current dose  (n_per)
    subacute: surgery-evaluable patients only        (n_sub_per)
    Non-surgery patients are NOT counted as subacute = 0.

    HOLD rule — verified behaviour
    --------------------------------
    _accrue_until(min_treated, min_surgery) adds one patient at a time
    until BOTH n_treated >= min_treated AND n_surgery >= min_surgery.
    Both acute and subacute counts accumulate during the wait.

    Phase 1 decision: triggered once n_treated >= 6  AND n_surgery >= 6.
    Phase 2 decision: triggered once n_treated >= 9  AND n_surgery >= 9.

    CONSEQUENCE: if p_surgery is low, n_treated at evaluation time can
    be much larger than 6 (or 9).  The acute thresholds a6_stop_min and
    a6_esc_max are applied to the cumulative y_acute over ALL treated
    patients at that dose, not a fixed cohort of 6.  This is intentional:
    the design prioritises subacute evaluability over fixed cohort sizes.

    Decision rules (cumulative counts at evaluation; all thresholds inclusive)
    --------------------------------------------------------------------------
    Phase 1
      stop    : y_acute >= a6_stop_min  OR  (n_surgery>=6 AND y_sub >= s6_stop_min)
      escalate: y_acute <= a6_esc_max  AND  (n_surgery>=6 AND y_sub <= s6_esc_max)
      expand  : neither of the above

    Phase 2 (only reached if phase 1 → expand)
      escalate: y_acute <= a9_esc_max  AND  (n_surgery>=9 AND y_sub <= s9_esc_max)
      stop    : otherwise  (includes y_sub >= s9_stop_min)

    Conservative rule when HOLD hits max_n without reaching threshold:
      subacute criterion is treated as not satisfied → no escalation,
      acute-only stopping still applies.
    """
    if rng is None:
        rng = np.random.default_rng()
    true_acute  = np.asarray(true_acute,  dtype=float)
    true_sub_gs = np.asarray(true_sub_gs, dtype=float)
    p_surg      = float(p_surgery)
    n_levels    = len(true_acute)

    level = int(start_level)
    n_per       = np.zeros(n_levels, dtype=int)   # all treated
    y_acute_per = np.zeros(n_levels, dtype=int)   # acute DLTs
    n_sub_per   = np.zeros(n_levels, dtype=int)   # surgery-evaluable patients
    y_sub_per   = np.zeros(n_levels, dtype=int)   # subacute DLTs (surgery pts only)

    total_n         = 0
    last_acceptable = None
    debug_rows      = []

    def _treat_one():
        """Treat one patient at current level; accumulate all counters."""
        nonlocal total_n
        a, ns, ys = simulate_cohort(
            1, true_acute[level], p_surg, true_sub_gs[level], rng)
        n_per[level]       += 1
        y_acute_per[level] += a
        n_sub_per[level]   += ns
        y_sub_per[level]   += ys
        total_n            += 1

    def _accrue_until(min_treated, min_surgery):
        """
        Accrue patients one-by-one until BOTH n_treated >= min_treated
        AND n_surgery >= min_surgery, or max_n is exhausted (HOLD rule).

        HOLD consequence: if p_surgery is low, n_per[level] at the time
        this function returns may be substantially larger than min_treated.
        All accumulated y_acute_per[level] counts feed into the subsequent
        phase decision — the acute thresholds are therefore applied to a
        potentially larger-than-expected number of acute observations.
        This is by design: subacute evaluability takes precedence.
        """
        while total_n < int(max_n):
            if (n_per[level] >= min_treated
                    and n_sub_per[level] >= min_surgery):
                break
            _treat_one()

    while total_n < int(max_n):

        # ── Phase 1: accrue until ≥6 treated AND ≥6 surgery-evaluable ─────
        _accrue_until(min_treated=6, min_surgery=6)

        ya  = y_acute_per[level]
        ys  = y_sub_per[level]
        nt  = n_per[level]
        nsg = n_sub_per[level]

        dbg = {
            "level": level, "phase": "p1",
            "n_treated": nt, "n_surgery": nsg,
            "y_acute": ya, "y_sub": ys,
        } if debug else None

        # Subacute evaluable flag for phase 1
        sub_eval_p1 = nsg >= 6

        # Phase 1 stop: EITHER acute OR (evaluable) subacute exceeds threshold
        stop_p1 = (ya >= int(a6_stop_min)
                   or (sub_eval_p1 and ys >= int(s6_stop_min)))

        # Phase 1 escalate: BOTH acute AND (evaluable) subacute below threshold
        esc_p1 = (nt >= 6 and sub_eval_p1
                  and ya <= int(a6_esc_max) and ys <= int(s6_esc_max))

        if stop_p1:
            if dbg is not None:
                dbg["decision"] = "stop_p1"; debug_rows.append(dbg)
            if level > 0:
                level -= 1
            break

        if esc_p1:
            last_acceptable = level
            if dbg is not None:
                dbg["decision"] = "escalate_p1"; debug_rows.append(dbg)
            if level < n_levels - 1:
                level += 1; continue
            break

        # ── Phase 2: accrue until ≥9 treated AND ≥9 surgery-evaluable ─────
        _accrue_until(min_treated=9, min_surgery=9)

        ya  = y_acute_per[level]
        ys  = y_sub_per[level]
        nt  = n_per[level]
        nsg = n_sub_per[level]

        if dbg is not None:
            dbg["phase"] = "p2"
            dbg["n_treated"] = nt; dbg["n_surgery"] = nsg
            dbg["y_acute"] = ya;   dbg["y_sub"] = ys

        # Subacute evaluable flag for phase 2
        sub_eval_p2 = nsg >= 9

        # Phase 2 escalate: BOTH acute AND (evaluable) subacute below threshold
        esc_p2 = (sub_eval_p2
                  and ya <= int(a9_esc_max) and ys <= int(s9_esc_max))

        # Phase 2 explicit stop: EITHER acute OR subacute exceeds stop threshold
        # (when both thresholds are complementary this is equivalent to "else stop")
        stop_p2 = (not esc_p2
                   and (not sub_eval_p2
                        or ys >= int(s9_stop_min)
                        or ya > int(a9_esc_max)))

        if esc_p2:
            last_acceptable = level
            if dbg is not None:
                dbg["decision"] = "escalate_p2"; debug_rows.append(dbg)
            if level < n_levels - 1:
                level += 1; continue
            break

        # Phase 2 stop (explicit or default)
        if dbg is not None:
            dbg["decision"] = "stop_p2"; debug_rows.append(dbg)
        if level > 0:
            level -= 1
        break

    selected = 0 if last_acceptable is None else int(last_acceptable)
    return selected, n_per, y_acute_per, n_sub_per, y_sub_per, debug_rows

# ============================================================
# CRM posterior via Gauss–Hermite quadrature
# Unchanged — reused independently for each endpoint.
# The key is that for the subacute endpoint the caller passes
#   n_per_level  = n_sub_per  (surgery patients only)
#   dlt_per_level = y_sub_per
# rather than total treated.  The likelihood function is identical;
# only the denominators differ.
# ============================================================

def posterior_via_gh(sigma, skeleton, n_per_level, dlt_per_level, gh_n=61):
    sk = safe_probs(skeleton)
    n  = np.asarray(n_per_level,   dtype=float)
    y  = np.asarray(dlt_per_level, dtype=float)
    x, w = np.polynomial.hermite.hermgauss(int(gh_n))
    theta = float(sigma) * np.sqrt(2.0) * x
    P  = sk[None, :] ** np.exp(theta)[:, None]
    P  = safe_probs(P)
    ll = (y[None, :] * np.log(P) + (n[None, :] - y[None, :]) * np.log(1 - P)).sum(axis=1)
    log_unnorm = np.log(w) + ll
    m = np.max(log_unnorm)
    unnorm = np.exp(log_unnorm - m)
    post_w = unnorm / np.sum(unnorm)
    return post_w, P

def crm_posterior_summaries(sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=61):
    post_w, P     = posterior_via_gh(sigma, skeleton, n_per_level, dlt_per_level, gh_n=gh_n)
    post_mean     = (post_w[:, None] * P).sum(axis=0)
    overdose_prob = (post_w[:, None] * (P > target)).sum(axis=0)
    return post_mean, overdose_prob

# ============================================================
# Conditional CRM decision functions
#
# Joint safety rule: dose d is allowed only if BOTH
#   P(acute(d) > target_acute | all-patient data)   < ewoc_alpha
#   P(subacute(d) > target_sub | surgery-pt data)   < ewoc_alpha
#
# Among allowed doses: choose the HIGHEST (maximise dose intensity
# subject to joint safety), then apply guardrails.
# ============================================================

def crm_choose_next_sur(
    sigma, skeleton_acute, skeleton_subacute,
    n_acute_per, y_acute_per,       # denominators: all patients
    n_sub_per,   y_sub_per,         # denominators: surgery patients only
    current_level, target_acute, target_subacute,
    ewoc_alpha=None, max_step=1, gh_n=61,
    enforce_highest_tried_plus_one=True, highest_tried=None,
):
    pmA, odA = crm_posterior_summaries(
        sigma, skeleton_acute,    n_acute_per, y_acute_per, target_acute,    gh_n=gh_n)
    pmS, odS = crm_posterior_summaries(
        sigma, skeleton_subacute, n_sub_per,   y_sub_per,   target_subacute, gh_n=gh_n)

    n_levels = len(skeleton_acute)
    if ewoc_alpha is None:
        allowed = np.arange(n_levels)
    else:
        allowed = np.where((odA < float(ewoc_alpha)) & (odS < float(ewoc_alpha)))[0]

    if allowed.size == 0:
        allowed = np.array([0], dtype=int)

    k_star = int(allowed.max())
    k_star = int(np.clip(k_star, current_level - int(max_step), current_level + int(max_step)))
    if enforce_highest_tried_plus_one and highest_tried is not None:
        k_star = int(min(k_star, int(highest_tried) + 1))
    k_star = int(np.clip(k_star, 0, n_levels - 1))
    return k_star, pmA, pmS, odA, odS, allowed

def crm_select_mtd_sur(
    sigma, skeleton_acute, skeleton_subacute,
    n_acute_per, y_acute_per,
    n_sub_per,   y_sub_per,
    target_acute, target_subacute,
    ewoc_alpha=None, gh_n=61, restrict_to_tried=True,
):
    pmA, odA = crm_posterior_summaries(
        sigma, skeleton_acute,    n_acute_per, y_acute_per, target_acute,    gh_n=gh_n)
    pmS, odS = crm_posterior_summaries(
        sigma, skeleton_subacute, n_sub_per,   y_sub_per,   target_subacute, gh_n=gh_n)

    n_levels = len(skeleton_acute)
    if ewoc_alpha is None:
        allowed = np.arange(n_levels)
    else:
        allowed = np.where((odA < float(ewoc_alpha)) & (odS < float(ewoc_alpha)))[0]

    if allowed.size == 0:
        return 0

    if restrict_to_tried:
        tried = np.where(np.asarray(n_acute_per) > 0)[0]
        if tried.size > 0:
            allowed2 = np.intersect1d(allowed, tried)
            if allowed2.size > 0:
                allowed = allowed2
            else:
                return int(tried.min())

    return int(allowed.max())

# ============================================================
# Conditional CRM trial runner
# ============================================================

def run_crm_sur(
    true_acute, p_surgery, true_sub_gs,
    target_acute, target_subacute,
    skeleton_acute, skeleton_subacute,
    sigma=1.0, start_level=0, already_treated_start=0,
    max_n=27, cohort_size=3, max_step=1, gh_n=61,
    enforce_guardrail=True, restrict_final_mtd_to_tried=True,
    ewoc_on=True, ewoc_alpha=0.25,
    burn_in_until_first_dlt=True,
    rng=None, debug=False,
):
    """
    Dual CRM with conditional subacute endpoint.

    p_surgery: scalar probability of surgery, same for all doses.

    Denominators:
      acute   model: n_acute_per[d] = all patients treated at dose d
      subacute model: n_sub_per[d]  = surgery patients at dose d only
    Non-surgery patients are excluded from the subacute likelihood.

    Burn-in: escalate one level at a time until the first acute DLT
    (surgery/subacute status does not drive burn-in termination).

    already_treated_start: pre-treated patients assumed to have 0 acute
    DLTs and unknown surgery status (not added to n_sub_per).
    """
    if rng is None:
        rng = np.random.default_rng()
    true_acute  = np.asarray(true_acute,  dtype=float)
    true_sub_gs = np.asarray(true_sub_gs, dtype=float)
    p_surg      = float(p_surgery)
    n_levels    = len(true_acute)

    level       = int(start_level)
    n_acute_per = np.zeros(n_levels, dtype=int)
    y_acute_per = np.zeros(n_levels, dtype=int)
    n_sub_per   = np.zeros(n_levels, dtype=int)   # surgery patients only
    y_sub_per   = np.zeros(n_levels, dtype=int)

    # Pre-treated patients: counted for acute (0 DLTs), no surgery data.
    already_treated_start = int(max(0, already_treated_start))
    if already_treated_start > 0:
        n_acute_per[level] += already_treated_start

    highest_tried  = level if already_treated_start > 0 else -1
    any_acute_dlt  = False
    debug_rows     = []

    burn_in_active = bool(burn_in_until_first_dlt and already_treated_start == 0)
    ewoc_alpha_eff = float(ewoc_alpha) if ewoc_on else None

    while int(n_acute_per.sum()) < int(max_n):
        n_add = min(int(cohort_size), int(max_n) - int(n_acute_per.sum()))

        a_tot = s_n = s_y = 0
        for _ in range(n_add):
            a, ns, ys = simulate_cohort(1, true_acute[level], p_surg,
                                        true_sub_gs[level], rng)
            a_tot += a; s_n += ns; s_y += ys

        n_acute_per[level] += n_add
        y_acute_per[level] += a_tot
        n_sub_per[level]   += s_n
        y_sub_per[level]   += s_y
        highest_tried       = max(highest_tried, level)

        if a_tot > 0:
            any_acute_dlt = True

        if debug:
            debug_rows.append({
                "treated_level":  level,
                "cohort_n":       int(n_add),
                "acute_dlts":     int(a_tot),
                "n_surg":         int(s_n),
                "sub_dlts":       int(s_y),
                "any_acute_dlt":  bool(any_acute_dlt),
            })

        if n_add < int(cohort_size):
            break

        # Burn-in: escalate until first acute DLT
        if burn_in_active and (not any_acute_dlt):
            if level < n_levels - 1:
                level += 1; continue

        next_level, pmA, pmS, odA, odS, allowed = crm_choose_next_sur(
            sigma=sigma,
            skeleton_acute=skeleton_acute,
            skeleton_subacute=skeleton_subacute,
            n_acute_per=n_acute_per, y_acute_per=y_acute_per,
            n_sub_per=n_sub_per,     y_sub_per=y_sub_per,
            current_level=level,
            target_acute=target_acute,
            target_subacute=target_subacute,
            ewoc_alpha=ewoc_alpha_eff,
            max_step=max_step, gh_n=gh_n,
            enforce_highest_tried_plus_one=enforce_guardrail,
            highest_tried=highest_tried,
        )

        if debug:
            debug_rows[-1].update({
                "next_level":          int(next_level),
                "allowed_levels":      ",".join(str(int(a)) for a in allowed),
                "highest_tried":       int(highest_tried),
                "post_mean_acute":     [float(v) for v in pmA],
                "post_mean_subacute":  [float(v) for v in pmS],
                "od_prob_acute":       [float(v) for v in odA],
                "od_prob_subacute":    [float(v) for v in odS],
                "n_sub_cumul":         [int(v) for v in n_sub_per],
            })

        level = int(next_level)

    selected = crm_select_mtd_sur(
        sigma=sigma,
        skeleton_acute=skeleton_acute,
        skeleton_subacute=skeleton_subacute,
        n_acute_per=n_acute_per, y_acute_per=y_acute_per,
        n_sub_per=n_sub_per,     y_sub_per=y_sub_per,
        target_acute=target_acute, target_subacute=target_subacute,
        ewoc_alpha=ewoc_alpha_eff, gh_n=gh_n,
        restrict_to_tried=restrict_final_mtd_to_tried,
    )
    return int(selected), n_acute_per, y_acute_per, n_sub_per, y_sub_per, debug_rows

# ============================================================
# Streamlit config + CSS
# ============================================================

st.set_page_config(
    page_title="Surgery-conditional dual-endpoint simulator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      [data-testid="stSidebar"]       { display: none; }
      [data-testid="stSidebarNav"]    { display: none; }
      [data-testid="collapsedControl"]{ display: none; }

      /* Tighter page padding — moves content slightly lower so the Essentials
         expander is fully visible without being clipped at the viewport top */
      .block-container { padding-top: 2.6rem; padding-bottom: 0.5rem; }
      .element-container { margin-bottom: 0.12rem; }

      /* Compact metric cards so they stack without excess whitespace */
      [data-testid="stMetric"]           { padding: 0.15rem 0 0.05rem 0 !important; }
      [data-testid="metric-container"]   { gap: 0 !important; }
      [data-testid="stMetricLabel"]      { font-size: 0.78rem !important; }
      [data-testid="stMetricValue"]      { font-size: 1.05rem !important; line-height: 1.2 !important; }

      /* Fixed-size images don't stretch with browser width */
      [data-testid="stImage"] img {
        max-width: none !important;
        width: auto  !important;
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
DEFAULT_TRUE_SUB_GS   = [0.02, 0.05, 0.15, 0.25, 0.40]  # subacute given surgery

R_DEFAULTS = {
    # Study
    "target_acute":            0.15,
    "target_subacute":         0.20,   # conditional on surgery
    "p_surgery":               0.80,   # global surgery probability (dose-independent)
    "start_level_1b":          2,
    "already_treated_start":   0,
    # Simulation
    "n_sims":                  200,
    "seed":                    123,
    # Sample sizes
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
    # Priors – subacute (conditional on surgery)
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
    # 6+3 thresholds — acute (denominator: all treated patients)
    "a6_esc_max":              0,    # max acute in ≥6 treated to escalate
    "a6_stop_min":             2,    # min acute in ≥6 treated to stop
    "a9_esc_max":              1,    # max acute in ≥9 treated to escalate
    # 6+3 thresholds — subacute (denominator: surgery-evaluable patients only)
    # HOLD rule: design waits at current dose until surgery-evaluable count
    # reaches the phase threshold (6 or 9) before applying subacute rules.
    "s6_esc_max":              1,    # max subacute in ≥6 surgery-eval to escalate
    "s6_stop_min":             3,    # min subacute in ≥6 surgery-eval to stop
    "s9_esc_max":              3,    # max subacute in ≥9 surgery-eval to escalate
    "s9_stop_min":             4,    # min subacute in ≥9 surgery-eval to stop
    # Playground prior tab
    "prior_endpoint_tab":      "Acute",
    # Figure sizing
    "preview_w_px":            PREVIEW_W_PX,
    "result_w_px":             RESULT_W_PX,
}

TRUE_ACUTE_KEYS    = [f"true_acute_{i}"    for i in range(5)]
TRUE_SUB_GS_KEYS   = [f"true_sub_gs_{i}"   for i in range(5)]

# ============================================================
# Reset logic
#
# Runs at the TOP of the script before any widget is created,
# so overwriting widget-backed keys is always safe here.
# Also clears stored results so stale charts don't linger.
# ============================================================

if st.session_state.get("_do_reset", False):
    for k, v in R_DEFAULTS.items():
        st.session_state[k] = v
    for i in range(5):
        st.session_state[TRUE_ACUTE_KEYS[i]]  = float(DEFAULT_TRUE_ACUTE[i])
        st.session_state[TRUE_SUB_GS_KEYS[i]] = float(DEFAULT_TRUE_SUB_GS[i])
    st.session_state["_results"]  = None
    st.session_state["_do_reset"] = False
    st.rerun()

def init_state():
    for k, v in R_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    for i in range(5):
        st.session_state.setdefault(TRUE_ACUTE_KEYS[i],  float(DEFAULT_TRUE_ACUTE[i]))
        st.session_state.setdefault(TRUE_SUB_GS_KEYS[i], float(DEFAULT_TRUE_SUB_GS[i]))
    st.session_state.setdefault("_results", None)

init_state()

# ============================================================
# Help text helpers (same pattern as sim.py)
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

def h_acute(i):
    return f"True acute DLT probability at L{i}. Default: {DEFAULT_TRUE_ACUTE[i]}"

def h_sub_gs(i):
    return (f"True subacute DLT probability GIVEN surgery at L{i}. "
            f"Non-surgery patients are NOT evaluated for this endpoint. "
            f"Default: {DEFAULT_TRUE_SUB_GS[i]}")

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
                   "Target acute DLT probability for MTD definition and CRM joint safety rule.",
                   r_name="target.acute")
        )
        st.number_input(
            "Target subacute DLT rate (given surgery)",
            min_value=0.05, max_value=0.50, step=0.01, key="target_subacute",
            help=h("target_subacute",
                   "Target subacute DLT probability CONDITIONAL on surgery. "
                   "Applied only to surgery patients in the CRM likelihood.",
                   r_name="target.subacute.given.surgery")
        )
        st.number_input(
            "Probability of surgery",
            min_value=0.0, max_value=1.0, step=0.01, key="p_surgery",
            help=(
                "Probability that a patient proceeds to surgery. "
                "Subacute toxicity is only observed in these patients. "
                "This probability is the same at every dose level. "
                f"Default: {R_DEFAULTS['p_surgery']}"
            )
        )
        st.number_input(
            "Start dose level (1-based)",
            min_value=1, max_value=5, step=1, key="start_level_1b",
            help=h("start_level_1b",
                   "Starting dose level for both designs (1-based).",
                   r_name="p (start dose index, 1-based)")
        )
        st.number_input(
            "Already treated at start dose (0 acute DLT)",
            min_value=0, max_value=500, step=1, key="already_treated_start",
            help=h("already_treated_start",
                   "Pre-treated patients at start dose with 0 acute DLTs. "
                   "Surgery status is unknown so these patients are NOT added "
                   "to the subacute CRM denominator.",
                   r_name="alreadytreated")
        )

    with c2:
        st.markdown("#### Simulation")
        st.number_input(
            "Number of simulated trials",
            min_value=50, max_value=5000, step=50, key="n_sims",
            help=h("n_sims",
                   "Number of simulated trials (replicates).",
                   r_name="NREP")
        )
        st.number_input(
            "Random seed",
            min_value=1, max_value=10_000_000, step=1, key="seed",
            help=h("seed", "Random seed for reproducibility.", r_name="set.seed()")
        )
        st.markdown("#### CRM integration")
        st.selectbox(
            "Gauss–Hermite points",
            options=[31, 41, 61, 81], key="gh_n",
            help=h("gh_n",
                   "Quadrature points for CRM posterior. Higher = more accurate, slower.",
                   r_name="gh.n")
        )
        st.selectbox(
            "Max dose step per update",
            options=[1, 2], key="max_step",
            help=h("max_step",
                   "Max dose levels the CRM can move up or down per cohort update.",
                   r_name="step.size / maxstep")
        )

    with c3:
        st.markdown("#### Sample size")
        st.number_input(
            "Maximum sample size (6+3)",
            min_value=6, max_value=200, step=3, key="max_n_63",
            help=h("max_n_63",
                   "Max patients under the 6+3 design.",
                   r_name="N.patient (6+3)")
        )
        st.number_input(
            "Maximum sample size (CRM)",
            min_value=6, max_value=200, step=3, key="max_n_crm",
            help=h("max_n_crm",
                   "Max patients under the conditional CRM design.",
                   r_name="N.patient (CRM)")
        )
        st.number_input(
            "Cohort size (CRM)",
            min_value=1, max_value=12, step=1, key="cohort_size",
            help=h("cohort_size",
                   "Patients per CRM cohort update.",
                   r_name="CO")
        )
        st.markdown("#### CRM safety / selection")
        st.toggle(
            "Guardrail: next dose ≤ highest tried + 1",
            key="enforce_guardrail",
            help=h("enforce_guardrail",
                   "Prevent skipping untried dose levels.",
                   r_name="guardrail / no skipping")
        )
        st.toggle(
            "Final MTD must be among tried doses",
            key="restrict_final_mtd",
            help=h("restrict_final_mtd",
                   "Restrict final MTD selection to doses with n > 0.",
                   r_name="final.mtd.restrict.to.tried")
        )
        st.toggle(
            "Show debug (first simulated trial)",
            key="show_debug",
            help=h("show_debug",
                   "Show detailed CRM internals for the first simulated trial.",
                   r_name="debug")
        )
        # CRM behaviour toggles moved here from Playground to free preview space
        st.markdown("#### CRM behaviour")
        st.toggle(
            "Burn-in until first acute DLT",
            key="burn_in",
            help=h("burn_in",
                   "Escalate one level at a time until the first acute DLT, "
                   "then switch to CRM updates. Surgery/subacute events do not "
                   "trigger burn-in termination.",
                   r_name="burnin / burning.phase")
        )
        st.toggle(
            "Enable EWOC joint overdose control",
            key="ewoc_on",
            help=h("ewoc_on",
                   "Restrict doses to those where BOTH acute OD prob and "
                   "conditional subacute OD prob are < EWOC alpha.",
                   r_name="EWOC on/off")
        )
        st.number_input(
            "EWOC alpha",
            min_value=0.01, max_value=0.99, step=0.01, key="ewoc_alpha",
            disabled=(not bool(st.session_state["ewoc_on"])),
            help=h("ewoc_alpha",
                   "EWOC threshold applied to both endpoints independently. "
                   "Disabled when EWOC is off.",
                   r_name="alpha")
        )

    # 6+3 dual stopping rules (acute + subacute)
    st.markdown("#### 6+3 stopping rules — modified design")
    st.info(
        "**This is a modified 6+3, not the standard protocol.**\n\n"
        "The standard 6+3 evaluates both endpoints on fixed cohorts of 6 (then 9) "
        "patients. That is not possible here because subacute toxicity is only "
        "observable in patients who undergo surgery.\n\n"
        "**How it works:**\n"
        "- Acute endpoint: evaluated on **all treated patients** (denominator = all treated).\n"
        "- Subacute endpoint: evaluated only on **surgery-evaluable patients** "
        "(denominator = patients who had surgery).\n"
        "- **HOLD state**: the design keeps accruing patients at the current dose "
        "until the required surgery-evaluable count is reached "
        "(≥ 6 for phase 1, ≥ 9 for phase 2).\n\n"
        "**Important design consequence:** if the surgery probability is low, "
        "reaching the surgery threshold may require far more than 6 (or 9) treated patients. "
        "The acute DLT count at evaluation therefore reflects all those extra patients — "
        "so there may be **more than 6 (or 9) acute observations** before the "
        "first (or second) subacute-based decision is made. "
        "This is intentional: subacute evaluability takes precedence over fixed cohort sizes.\n\n"
        "Escalation requires **BOTH** acute AND subacute criteria to be met. "
        "Stopping is triggered by **EITHER** criterion.",
        icon="ℹ️",
    )

    st.markdown(
        "<div style='font-size:0.80rem;font-weight:600;color:#555;'>"
        "Acute thresholds (denominator: all treated patients)</div>",
        unsafe_allow_html=True,
    )
    ar1, ar2, ar3 = st.columns(3, gap="large")
    with ar1:
        st.number_input(
            "After ≥6 treated — escalate if acute ≤",
            min_value=0, max_value=5, step=1, key="a6_esc_max",
            help=h("a6_esc_max",
                   "Phase 1 acute criterion: escalate if cumulative acute DLTs "
                   "≤ this value (requires BOTH acute and subacute criteria).")
        )
    with ar2:
        st.number_input(
            "After ≥6 treated — stop if acute ≥",
            min_value=1, max_value=6, step=1, key="a6_stop_min",
            help=h("a6_stop_min",
                   "Phase 1 acute criterion: stop if cumulative acute DLTs "
                   "≥ this value (EITHER criterion triggers stop).")
        )
    with ar3:
        st.number_input(
            "After ≥9 treated — escalate if acute ≤",
            min_value=0, max_value=8, step=1, key="a9_esc_max",
            help=h("a9_esc_max",
                   "Phase 2 acute criterion: escalate if cumulative acute DLTs "
                   "≤ this value (requires BOTH criteria; else stop).")
        )

    st.markdown(
        "<div style='font-size:0.80rem;font-weight:600;color:#555;margin-top:0.4rem;'>"
        "Subacute thresholds (denominator: surgery-evaluable patients; "
        "HOLD until ≥6 or ≥9 surgery patients at current dose)</div>",
        unsafe_allow_html=True,
    )
    sr1, sr2, sr3, sr4 = st.columns(4, gap="small")
    with sr1:
        st.number_input(
            "≥6 surgery-eval — escalate if sub ≤",
            min_value=0, max_value=6, step=1, key="s6_esc_max",
            help=h("s6_esc_max",
                   "Phase 1 subacute criterion: escalate if subacute DLTs among "
                   "the first ≥6 surgery-evaluable patients ≤ this value "
                   "(combined with acute escalation criterion).")
        )
    with sr2:
        st.number_input(
            "≥6 surgery-eval — stop if sub ≥",
            min_value=1, max_value=6, step=1, key="s6_stop_min",
            help=h("s6_stop_min",
                   "Phase 1 subacute criterion: stop if subacute DLTs among "
                   "the first ≥6 surgery-evaluable patients ≥ this value "
                   "(EITHER criterion triggers stop).")
        )
    with sr3:
        st.number_input(
            "≥9 surgery-eval — escalate if sub ≤",
            min_value=0, max_value=9, step=1, key="s9_esc_max",
            help=h("s9_esc_max",
                   "Phase 2 subacute criterion: escalate if subacute DLTs among "
                   "the first ≥9 surgery-evaluable patients ≤ this value "
                   "(combined with acute criterion; else stop).")
        )
    with sr4:
        st.number_input(
            "≥9 surgery-eval — stop if sub ≥",
            min_value=1, max_value=9, step=1, key="s9_stop_min",
            help=h("s9_stop_min",
                   "Phase 2 subacute criterion: stop if subacute DLTs among "
                   "the first ≥9 surgery-evaluable patients ≥ this value.")
        )

    # Figure sizing — inline (no nested expander; Streamlit forbids nesting)
    st.markdown("#### Figure sizing")
    fs1, fs2 = st.columns(2, gap="large")
    with fs1:
        st.number_input(
            "Preview plot width (px)",
            min_value=180, max_value=500, step=10, key="preview_w_px",
            help=h("preview_w_px",
                   "Fixed pixel width for the preview plot.",
                   r_name="(UI only)")
        )
    with fs2:
        st.number_input(
            "Results plot width (px)",
            min_value=220, max_value=600, step=10, key="result_w_px",
            help=h("result_w_px",
                   "Fixed pixel width for results histograms.",
                   r_name="(UI only)")
        )

    st.write("")
    if st.button(
        "Reset to defaults",
        help="Resets all inputs — Essentials, Priors, CRM knobs, and true curves — to defaults."
    ):
        st.session_state["_do_reset"] = True
        st.rerun()

# ============================================================
# Playground
# ============================================================

# Playground column ratios: right column is slightly wider to accommodate
# the enlarged preview now that burn-in and EWOC toggles live in Essentials.
with st.expander("Playground", expanded=True):
    left, mid, right = st.columns([1.00, 1.02, 1.12], gap="large")

    # ---- Left: true curves + Run button
    with left:
        st.markdown("#### True probabilities by dose")

        # Column headers
        hL, hA, hSub = st.columns([0.35, 0.30, 0.35], gap="small")
        with hA:
            st.markdown("<div style='font-size:0.79rem;font-weight:600;'>Acute</div>",
                        unsafe_allow_html=True)
        with hSub:
            st.markdown("<div style='font-size:0.79rem;font-weight:600;'>Sub|surg</div>",
                        unsafe_allow_html=True)

        true_acute    = []
        true_sub_gs   = []

        for i, lab in enumerate(dose_labels):
            rL, rA, rSub = st.columns([0.35, 0.30, 0.35], gap="small")
            with rL:
                st.markdown(
                    f"<div style='font-size:0.83rem;padding-top:0.25rem;'>L{i} {lab}</div>",
                    unsafe_allow_html=True)
            with rA:
                va = st.number_input(f"Acute L{i}", 0.0, 1.0, step=0.01,
                                     key=TRUE_ACUTE_KEYS[i],
                                     label_visibility="collapsed",
                                     help=h_acute(i))
                true_acute.append(float(va))
            with rSub:
                vg = st.number_input(f"SubGS L{i}", 0.0, 1.0, step=0.01,
                                     key=TRUE_SUB_GS_KEYS[i],
                                     label_visibility="collapsed",
                                     help=h_sub_gs(i))
                true_sub_gs.append(float(vg))

        p_surgery_val       = float(st.session_state["p_surgery"])
        target_acute_val    = float(st.session_state["target_acute"])
        target_subacute_val = float(st.session_state["target_subacute"])
        true_safe = find_true_safe_dose(
            true_acute, true_sub_gs, target_acute_val, target_subacute_val)
        if true_safe is not None:
            st.caption(f"Highest jointly safe dose = L{true_safe}")
        else:
            st.caption("No dose satisfies both targets.")

        st.write("")
        run = st.button("Run simulations", use_container_width=True)

    # ---- Mid: Priors (Acute / Subacute toggle)
    with mid:
        st.markdown("#### Priors")

        st.radio(
            "Skeleton model",
            options=["empiric", "logistic"],
            horizontal=True,
            key="prior_model",
            help=h("prior_model",
                   "Skeleton generation method, shared for both endpoints.",
                   r_name="skeleton model (empiric/logistic)")
        )

        prior_model_val = str(st.session_state["prior_model"])
        intcpt_val      = float(st.session_state["logistic_intcpt"])

        # Compute max halfwidths (rounded to step precision) and force-write
        # all six prior keys BEFORE any slider widget is created.
        # This guards against Streamlit versions that reset conditional-branch
        # sliders to min_value on first render (Streamlit safety rule).
        _pt_A    = float(st.session_state["prior_target_acute"])
        _max_hwA = round(max(0.01, min(0.30, _pt_A - 0.01, 1.0 - _pt_A - 0.01)), 2)

        _pt_S    = float(st.session_state["prior_target_subacute"])
        _max_hwS = round(max(0.01, min(0.30, _pt_S - 0.01, 1.0 - _pt_S - 0.01)), 2)

        _prior_init = [
            ("prior_target_acute",    float, 0.05, 0.50,    R_DEFAULTS["prior_target_acute"]),
            ("halfwidth_acute",       float, 0.01, _max_hwA, R_DEFAULTS["halfwidth_acute"]),
            ("prior_nu_acute",        int,   1,    5,        R_DEFAULTS["prior_nu_acute"]),
            ("prior_target_subacute", float, 0.05, 0.50,    R_DEFAULTS["prior_target_subacute"]),
            ("halfwidth_subacute",    float, 0.01, _max_hwS, R_DEFAULTS["halfwidth_subacute"]),
            ("prior_nu_subacute",     int,   1,    5,        R_DEFAULTS["prior_nu_subacute"]),
        ]
        for _k, _typ, _lo, _hi, _def in _prior_init:
            _v = _typ(st.session_state.get(_k, _def))
            st.session_state[_k] = _typ(np.clip(_v, _lo, _hi))

        ep_tab = st.radio(
            "Endpoint",
            options=["Acute", "Subacute (given surgery)"],
            horizontal=True,
            key="prior_endpoint_tab",
            help="Switch between acute and subacute prior parameter sets. Both are used in every simulation run.",
        )

        if ep_tab == "Acute":
            st.slider(
                "Prior target (acute)",
                min_value=0.05, max_value=0.50, step=0.01,
                key="prior_target_acute",
                help=h("prior_target_acute",
                       "Target probability for the acute skeleton.",
                       r_name="prior.target.acute")
            )
            st.slider(
                "Halfwidth (acute)",
                min_value=0.01, max_value=float(_max_hwA), step=0.01,
                key="halfwidth_acute",
                help=h("halfwidth_acute",
                       "Skeleton steepness parameter. prior_target_acute ± halfwidth must stay in (0,1).",
                       r_name="halfwidth.acute")
            )
            st.slider(
                "Prior MTD level (acute, 1-based)",
                min_value=1, max_value=5, step=1,
                key="prior_nu_acute",
                help=h("prior_nu_acute",
                       "Dose level a priori closest to the acute target — anchor for the acute skeleton.",
                       r_name="prior.MTD.acute")
            )
        else:
            st.slider(
                "Prior target (subacute | surgery)",
                min_value=0.05, max_value=0.50, step=0.01,
                key="prior_target_subacute",
                help=h("prior_target_subacute",
                       "Target probability for the subacute-given-surgery skeleton.",
                       r_name="prior.target.subacute")
            )
            st.slider(
                "Halfwidth (subacute)",
                min_value=0.01, max_value=float(_max_hwS), step=0.01,
                key="halfwidth_subacute",
                help=h("halfwidth_subacute",
                       "Skeleton steepness. prior_target_subacute ± halfwidth must stay in (0,1).",
                       r_name="halfwidth.subacute")
            )
            st.slider(
                "Prior MTD level (subacute, 1-based)",
                min_value=1, max_value=5, step=1,
                key="prior_nu_subacute",
                help=h("prior_nu_subacute",
                       "Dose level a priori closest to the conditional subacute target.",
                       r_name="prior.MTD.subacute")
            )

        if prior_model_val == "logistic":
            st.slider(
                "Logistic intercept",
                min_value=-10.0, max_value=10.0, step=0.1,
                key="logistic_intcpt",
                help=h("logistic_intcpt",
                       "Intercept for logistic skeleton construction, shared for both endpoints.",
                       r_name="intcpt")
            )
            intcpt_val = float(st.session_state["logistic_intcpt"])

        # Build skeletons (local hw vars only; never mutate widget keys here)
        hw_A_eff = float(st.session_state["halfwidth_acute"])
        try:
            skeleton_acute = dfcrm_getprior(
                halfwidth=hw_A_eff, target=_pt_A,
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
                halfwidth=hw_S_eff, target=_pt_S,
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

    # ---- Right: CRM prior sigma + preview plots
    # Burn-in and EWOC toggles moved to Essentials → more vertical space here
    # for the preview plots (taller figures, easier to read skeletons).
    with right:
        st.markdown("#### CRM prior + preview")

        st.slider(
            "Prior sigma on theta",
            min_value=0.2, max_value=5.0, step=0.1, key="sigma",
            help=h("sigma",
                   "SD of theta in the CRM prior (shared). Larger = weaker prior.",
                   r_name="prior.sigma / sigma")
        )

        # Preview: two stacked mini-plots for clarity
        # Top: acute true + skeleton + target
        # Bottom: subacute|surgery true + skeleton + target + global p_surgery (horizontal)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(PREVIEW_W_IN, PREVIEW_H_IN), dpi=PREVIEW_DPI)
        x = np.arange(5)

        ax1.plot(x, true_acute,     "o-",  color="tab:blue",   lw=1.5, label="True acute")
        ax1.plot(x, skeleton_acute, "o--", color="tab:blue",   lw=1.5, label="Skel acute")
        ax1.axhline(target_acute_val, lw=1, alpha=0.55, color="tab:blue")
        ax1.set_ylabel("P(acute)", fontsize=8)
        ax1.set_xticks(x); ax1.set_xticklabels([f"L{i}" for i in range(5)], fontsize=8)
        _y1 = max(max(true_acute), max(skeleton_acute), target_acute_val)
        ax1.set_ylim(0, min(1.0, _y1 * 1.3 + 0.02))
        ax1.legend(fontsize=7, frameon=False, loc="upper left")
        compact_style(ax1)

        ax2.plot(x, true_sub_gs,       "s-",  color="tab:orange", lw=1.5, label="True sub|surg")
        ax2.plot(x, skeleton_subacute, "s--", color="tab:orange", lw=1.5, label="Skel sub")
        ax2.axhline(p_surgery_val, lw=1.2, alpha=0.70, color="tab:green",
                    linestyle="--", label=f"P(surgery)={p_surgery_val:.2f}")
        ax2.axhline(target_subacute_val, lw=1, alpha=0.55, color="tab:orange")
        ax2.set_ylabel("Probability", fontsize=8)
        ax2.set_xticks(x); ax2.set_xticklabels([f"L{i}" for i in range(5)], fontsize=8)
        _y2 = max(max(true_sub_gs), max(skeleton_subacute), target_subacute_val, p_surgery_val)
        ax2.set_ylim(0, min(1.0, _y2 * 1.25 + 0.02))
        ax2.legend(fontsize=7, frameon=False, loc="upper left")
        compact_style(ax2)

        fig.tight_layout(pad=0.5)
        st.image(fig_to_png_bytes(fig), width=int(st.session_state["preview_w_px"]))

# ============================================================
# Run simulations — results stored in session_state for
# persistence across widget-interaction reruns.
# ============================================================

if "run" in locals() and run:
    rng = np.random.default_rng(int(st.session_state["seed"]))
    ns  = int(st.session_state["n_sims"])
    start_0b = int(np.clip(int(st.session_state["start_level_1b"]) - 1, 0, 4))

    sel_63  = np.zeros(5, dtype=int)
    sel_crm = np.zeros(5, dtype=int)

    nmat_63  = np.zeros((ns, 5), dtype=int)   # total patients per dose
    nmat_crm = np.zeros((ns, 5), dtype=int)

    nsurg_63  = np.zeros((ns, 5), dtype=int)  # surgery patients per dose
    nsurg_crm = np.zeros((ns, 5), dtype=int)

    ya_63  = np.zeros(ns, dtype=int)   # total acute DLTs
    ya_crm = np.zeros(ns, dtype=int)
    ys_63  = np.zeros(ns, dtype=int)   # total subacute DLTs (surgery pts)
    ys_crm = np.zeros(ns, dtype=int)
    ns_63  = np.zeros(ns, dtype=int)   # total surgery patients
    ns_crm = np.zeros(ns, dtype=int)

    dbg_63 = dbg_crm = None

    for s in range(ns):
        flag = bool(st.session_state["show_debug"] and s == 0)

        sel63, n63, ya63, nsg63, ysb63, d63 = run_6plus3_sur(
            true_acute=true_acute,
            p_surgery=float(st.session_state["p_surgery"]),
            true_sub_gs=true_sub_gs,
            start_level=start_0b,
            max_n=int(st.session_state["max_n_63"]),
            a6_esc_max=int(st.session_state["a6_esc_max"]),
            a6_stop_min=int(st.session_state["a6_stop_min"]),
            a9_esc_max=int(st.session_state["a9_esc_max"]),
            s6_esc_max=int(st.session_state["s6_esc_max"]),
            s6_stop_min=int(st.session_state["s6_stop_min"]),
            s9_esc_max=int(st.session_state["s9_esc_max"]),
            s9_stop_min=int(st.session_state["s9_stop_min"]),
            rng=rng, debug=flag,
        )

        selc, nc, yac, nsgc, ysbc, dc = run_crm_sur(
            true_acute=true_acute,
            p_surgery=float(st.session_state["p_surgery"]),
            true_sub_gs=true_sub_gs,
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
            rng=rng, debug=flag,
        )

        sel_63[sel63]  += 1
        sel_crm[selc]  += 1
        nmat_63[s, :]   = n63
        nmat_crm[s, :]  = nc
        nsurg_63[s, :]  = nsg63
        nsurg_crm[s, :] = nsgc
        ya_63[s]  = int(ya63.sum());  ya_crm[s]  = int(yac.sum())
        ys_63[s]  = int(ysb63.sum()); ys_crm[s]  = int(ysbc.sum())
        ns_63[s]  = int(nsg63.sum()); ns_crm[s]  = int(nsgc.sum())

        if flag:
            dbg_63 = d63; dbg_crm = dc

    mean_n63  = float(np.mean(nmat_63.sum(axis=1)))
    mean_ncrm = float(np.mean(nmat_crm.sum(axis=1)))
    mean_ns63  = float(np.mean(ns_63))
    mean_nscrm = float(np.mean(ns_crm))

    st.session_state["_results"] = {
        "p63":              sel_63  / float(ns),
        "pcrm":             sel_crm / float(ns),
        "p_surgery":        float(st.session_state["p_surgery"]),
        "avg_n63":          np.mean(nmat_63,  axis=0),
        "avg_ncrm":         np.mean(nmat_crm, axis=0),
        "avg_nsurg63":      np.mean(nsurg_63,  axis=0),
        "avg_nsurgcrm":     np.mean(nsurg_crm, axis=0),
        "mean_n63":         mean_n63,
        "mean_ncrm":        mean_ncrm,
        "acute_rate_63":    float(np.mean(ya_63)  / max(1e-9, mean_n63)),
        "acute_rate_crm":   float(np.mean(ya_crm) / max(1e-9, mean_ncrm)),
        "surgery_rate_63":  float(mean_ns63  / max(1e-9, mean_n63)),
        "surgery_rate_crm": float(mean_nscrm / max(1e-9, mean_ncrm)),
        # subacute rate denominator = surgery patients (primary)
        "sub_gs_rate_63":   float(np.mean(ys_63)  / max(1e-9, mean_ns63)),
        "sub_gs_rate_crm":  float(np.mean(ys_crm) / max(1e-9, mean_nscrm)),
        # subacute rate per ALL treated (secondary, clearly labelled)
        "sub_all_rate_63":  float(np.mean(ys_63)  / max(1e-9, mean_n63)),
        "sub_all_rate_crm": float(np.mean(ys_crm) / max(1e-9, mean_ncrm)),
        "true_safe":        true_safe,
        "ns":               ns,
        "seed":             int(st.session_state["seed"]),
        "show_debug":       bool(st.session_state["show_debug"]),
        "dbg_63":           dbg_63,
        "dbg_crm":          dbg_crm,
    }

# ============================================================
# Results — read from session_state for persistence
# ============================================================

res = st.session_state.get("_results")
if res is not None:
    p63    = res["p63"]
    pcrm   = res["pcrm"]
    ts     = res["true_safe"]

    st.write("")
    r1, r2, r3 = st.columns([1.05, 1.05, 0.90], gap="large")

    with r1:
        fig, ax = plt.subplots(figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
        xx = np.arange(5); w = 0.38
        ax.bar(xx - w/2, p63,  w, label="6+3 (dual)")
        ax.bar(xx + w/2, pcrm, w, label="Conditional CRM")
        ax.set_title("P(select dose as MTD)", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, max(p63.max(), pcrm.max()) * 1.15 + 1e-6)
        if ts is not None:
            ax.axvline(ts, lw=1, alpha=0.6, label=f"True safe=L{ts}")
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.image(fig_to_png_bytes(fig), width=int(st.session_state["result_w_px"]))

    with r2:
        fig, (ax_n, ax_s) = plt.subplots(2, 1,
            figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
        xx = np.arange(5); w = 0.38

        ax_n.bar(xx - w/2, res["avg_n63"],   w, label="6+3")
        ax_n.bar(xx + w/2, res["avg_ncrm"],  w, label="CRM")
        ax_n.set_title("Avg patients (all)", fontsize=9)
        ax_n.set_xticks(xx)
        ax_n.set_xticklabels([f"L{i}" for i in range(5)], fontsize=8)
        ax_n.set_ylabel("Patients", fontsize=8)
        compact_style(ax_n)
        ax_n.legend(fontsize=7, frameon=False)

        ax_s.bar(xx - w/2, res["avg_nsurg63"],  w, label="6+3")
        ax_s.bar(xx + w/2, res["avg_nsurgcrm"], w, label="CRM")
        ax_s.set_title("Avg surgery patients", fontsize=9)
        ax_s.set_xticks(xx)
        ax_s.set_xticklabels([f"L{i}" for i in range(5)], fontsize=8)
        ax_s.set_ylabel("Patients", fontsize=8)
        compact_style(ax_s)
        ax_s.legend(fontsize=7, frameon=False)

        fig.tight_layout(pad=0.4)
        st.image(fig_to_png_bytes(fig), width=int(st.session_state["result_w_px"]))

    # Compact metrics: 2-column layout halves the height of the metrics panel
    # so it better matches the figure heights beside it.
    with r3:
        p_surg_input = res.get("p_surgery", float(st.session_state["p_surgery"]))
        mc1, mc2 = st.columns(2, gap="small")

        with mc1:
            st.metric("Acute/pt (6+3)",
                      f"{res['acute_rate_63']:.3f}",
                      help="Acute DLT rate per treated patient — 6+3 design")
            st.metric("Surgery rate (6+3)",
                      f"{res['surgery_rate_63']:.3f}",
                      help=f"Expected ≈ {p_surg_input:.2f} (global p_surgery)")
            st.metric("Sub/surg pt (6+3)",
                      f"{res['sub_gs_rate_63']:.3f}",
                      help="Subacute DLT rate per surgery-evaluable patient — 6+3 design")

        with mc2:
            st.metric("Acute/pt (CRM)",
                      f"{res['acute_rate_crm']:.3f}",
                      help="Acute DLT rate per treated patient — conditional CRM")
            st.metric("Surgery rate (CRM)",
                      f"{res['surgery_rate_crm']:.3f}",
                      help=f"Expected ≈ {p_surg_input:.2f} (global p_surgery)")
            st.metric("Sub/surg pt (CRM)",
                      f"{res['sub_gs_rate_crm']:.3f}",
                      help="Subacute DLT rate per surgery-evaluable patient — CRM")

        st.caption(
            f"Sub / all treated: 6+3={res['sub_all_rate_63']:.3f}  "
            f"CRM={res['sub_all_rate_crm']:.3f}  (secondary)"
        )
        st.caption(
            f"n_sims={res['ns']} | seed={res['seed']}"
            + (f" | True safe=L{ts}" if ts is not None else " | No jointly safe dose")
        )

    # Debug output for first simulated trial
    if res["show_debug"]:
        if res["dbg_63"]:
            st.subheader("6+3 debug (first simulated trial)")
            for row in res["dbg_63"]:
                st.write(
                    f"L{row['level']} | {row['phase']} | "
                    f"n_treated={row['n_treated']} n_surgery={row['n_surgery']} | "
                    f"y_acute={row['y_acute']} y_sub={row['y_sub']} "
                    f"→ {row['decision']}"
                )

        if res["dbg_crm"]:
            st.subheader("Conditional CRM debug (first simulated trial)")
            for i, row in enumerate(res["dbg_crm"], start=1):
                st.write(
                    f"Update {i}: L{row['treated_level']} | n={row['cohort_n']} "
                    f"| acute_dlts={row['acute_dlts']} | n_surg={row['n_surg']} "
                    f"| sub_dlts={row['sub_dlts']} | any_acute_dlt={row['any_acute_dlt']}"
                )
                if "next_level" in row:
                    st.write(
                        f"  allowed: {row['allowed_levels']} | next: L{row['next_level']} "
                        f"| highest_tried={row['highest_tried']}"
                    )
                    st.write(f"  n_sub cumul:        {row['n_sub_cumul']}")
                    st.write(f"  post_mean_acute:    {[round(v,3) for v in row['post_mean_acute']]}")
                    st.write(f"  post_mean_subacute: {[round(v,3) for v in row['post_mean_subacute']]}")
                    st.write(f"  od_prob_acute:      {[round(v,3) for v in row['od_prob_acute']]}")
                    st.write(f"  od_prob_subacute:   {[round(v,3) for v in row['od_prob_subacute']]}")
