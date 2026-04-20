"""
sim_core.py — TITE dual-endpoint dose-escalation simulator — computational core
No Streamlit dependencies.  All simulation, CRM, and chart logic lives here.
"""

import base64
import datetime
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO

# ── Fixed sizing (one place to tune) ──────────────────────────────────────────
PREVIEW_W_PX = 310
RESULT_W_PX  = 460
PREVIEW_W_IN, PREVIEW_H_IN, PREVIEW_DPI = 4.2, 5.0, 150
RESULT_W_IN,  RESULT_H_IN,  RESULT_DPI  = 6.0, 4.4, 170

MONTH = 30.0   # days per month (accrual unit conversion)

# ==============================================================================
# Defaults
# ==============================================================================

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]

DEFAULT_TRUE_T1 = [0.01, 0.02, 0.12, 0.20, 0.35]
DEFAULT_TRUE_T2 = [0.02, 0.05, 0.15, 0.25, 0.40]

R_DEFAULTS = {
    # Study
    "target_t1":          0.15,
    "target_t2":          0.33,
    "p_surgery":          0.80,
    "start_level_1b":     2,
    # Simulation
    "n_sims":             200,
    "seed":               123,
    # Accrual
    "accrual_per_month":  1.5,
    # Timing (days)
    "incl_to_rt":         21,
    "rt_dur":             14,
    "rt_to_surg":         84,
    "tox2_win":           30,
    # Sample sizes
    "max_n_63":           27,
    "max_n_crm":          27,
    "cohort_size":        3,
    # Priors — shared model
    "prior_model":        "empiric",
    "logistic_intcpt":    3.0,
    # Priors — tox1
    "prior_target_t1":    0.15,
    "halfwidth_t1":       0.10,
    "prior_nu_t1":        3,
    # Priors — tox2
    "prior_target_t2":    0.33,
    "halfwidth_t2":       0.10,
    "prior_nu_t2":        3,
    # CRM knobs
    "sigma":              1.0,
    "burn_in":            True,
    "ewoc_on":            True,
    "ewoc_alpha":         0.25,
    # CRM integration
    "gh_n":               61,
    "max_step":           1,
    # CRM safety
    "enforce_guardrail":      True,
    "restrict_final_mtd":     True,
    # 6+3 acute thresholds
    "a6_esc_max":         0,
    "a6_stop_min":        2,
    "a9_esc_max":         1,
    # 6+3 subacute thresholds
    "s6_esc_max":         1,
    "s6_stop_min":        3,
    "s9_esc_max":         3,
    "s9_stop_min":        4,
    # Figure sizing
    "preview_w_px":       PREVIEW_W_PX,
    "result_w_px":        RESULT_W_PX,
    # Decision trace (first CRM trial only)
    "show_crm_trace":     True,
    # Playground prior-endpoint tab
    "prior_ep_tab":       "Tox1 (acute)",
    # Playground prior scenario selector
    "prior_scenario":     "Custom",
}

TRUE_T1_KEYS = [f"true_t1_L{i}" for i in range(5)]
TRUE_T2_KEYS = [f"true_t2_L{i}" for i in range(5)]

_TRUE_DEFAULTS = {
    **{TRUE_T1_KEYS[i]: DEFAULT_TRUE_T1[i] for i in range(5)},
    **{TRUE_T2_KEYS[i]: DEFAULT_TRUE_T2[i] for i in range(5)},
}

# Design Exploration defaults (not in R_DEFAULTS to keep it clean)
DE_DEFAULTS = {
    "de_param_name":  "sigma",
    "de_sig_min":     0.3,
    "de_sig_max":     2.0,
    "de_sig_pts":     8,
    "de_ea_min":      0.05,
    "de_ea_max":      0.60,
    "de_ea_pts":      8,
    "de_inc_off":     True,
    "de_max_n_vals":  [12, 15, 18, 21, 24, 27, 30, 33, 36],
    "de_nu1_vals":    [1, 2, 3, 4, 5],
    "de_nu2_vals":    [1, 2, 3, 4, 5],
    "de_cohort_vals": [1, 2, 3, 4],
    "de_n_sim":       200,
    "de_seed":        42,
    "de_speed_mode":  False,
}

# ==============================================================================
# Prior scenario presets
# ==============================================================================

_PRIOR_SCENARIOS = {
    "Neutral": {
        "description": (
            "Balanced starting point. No strong prior belief about which dose level "
            "is the MTD — the skeleton is centred at the middle level (L3) with "
            "moderate uncertainty about the dose-toxicity curve shape."
        ),
        "prior_target_t1": 0.15, "halfwidth_t1": 0.10, "prior_nu_t1": 3,
        "prior_target_t2": 0.33, "halfwidth_t2": 0.10, "prior_nu_t2": 3,
    },
    "Lower-dose prior": {
        "description": (
            "Assumes the MTD is more likely to be at a lower dose level (around L2). "
            "Suitable when prior clinical data suggest toxicity rises early across "
            "the tested dose range."
        ),
        "prior_target_t1": 0.15, "halfwidth_t1": 0.10, "prior_nu_t1": 2,
        "prior_target_t2": 0.33, "halfwidth_t2": 0.10, "prior_nu_t2": 2,
    },
    "Higher-dose prior": {
        "description": (
            "Assumes the MTD is more likely to be at a higher dose level (around L4). "
            "Suitable when existing data suggest the drug is well tolerated at the "
            "lower levels and toxicity is expected only at the upper end."
        ),
        "prior_target_t1": 0.15, "halfwidth_t1": 0.10, "prior_nu_t1": 4,
        "prior_target_t2": 0.33, "halfwidth_t2": 0.10, "prior_nu_t2": 4,
    },
    "Conservative": {
        "description": (
            "Cautious prior: assumes the dose-toxicity curve rises steeply and the "
            "MTD is near lower levels. The skeleton is more concentrated (narrow "
            "halfwidth), limiting escalation to higher doses unless the data clearly "
            "support it."
        ),
        "prior_target_t1": 0.15, "halfwidth_t1": 0.06, "prior_nu_t1": 2,
        "prior_target_t2": 0.33, "halfwidth_t2": 0.06, "prior_nu_t2": 2,
    },
    "Optimistic": {
        "description": (
            "Permissive prior: assumes the dose-toxicity curve is relatively flat and "
            "the MTD is toward higher levels. The skeleton is wider (larger halfwidth), "
            "allowing the model to explore upper doses more readily when early outcomes "
            "are benign."
        ),
        "prior_target_t1": 0.15, "halfwidth_t1": 0.13, "prior_nu_t1": 4,
        "prior_target_t2": 0.33, "halfwidth_t2": 0.13, "prior_nu_t2": 4,
    },
    "Custom": {
        "description": (
            "Manually configure all prior parameters. Recommended for users familiar "
            "with the dfcrm skeleton parameterisation (target rate, halfwidth, and "
            "prior MTD level for each endpoint)."
        ),
    },
}

# ==============================================================================
# Helpers
# ==============================================================================

def safe_probs(x):
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)

_DARK_BG  = "#1a1a2e"
_DARK_AX  = "#16213e"
_DARK_FG  = "#e0e0e0"
_DARK_GRD = "#2a2a4a"

_MTD_LINE_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]

def _apply_dark_fig(fig, *axes):
    fig.patch.set_facecolor(_DARK_BG)
    for ax in axes:
        ax.set_facecolor(_DARK_AX)
        ax.tick_params(colors=_DARK_FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(_DARK_GRD)
        ax.xaxis.label.set_color(_DARK_FG)
        ax.yaxis.label.set_color(_DARK_FG)
        ax.title.set_color(_DARK_FG)

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3, color=_DARK_GRD)

def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def fig_to_b64(fig):
    """Encode a matplotlib figure as a base64 PNG data-URI."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def find_true_safe_dose(true_t1, true_t2, target1, target2):
    safe = [d for d in range(len(true_t1))
            if true_t1[d] <= target1 and true_t2[d] <= target2]
    return max(safe) if safe else None

# ==============================================================================
# dfcrm getprior (ported from sim_sur.py)
# ==============================================================================

def dfcrm_getprior(halfwidth, target, nu, nlevel, model="empiric", intcpt=3.0):
    halfwidth = float(halfwidth); target = float(target)
    nu = int(nu); nlevel = int(nlevel); intcpt = float(intcpt)
    if not (0 < target < 1):
        raise ValueError("target must be in (0, 1).")
    if halfwidth <= 0:
        raise ValueError("halfwidth must be > 0.")
    if (target - halfwidth) <= 0 or (target + halfwidth) >= 1:
        raise ValueError("halfwidth too large: target ± halfwidth must stay in (0,1).")
    if not (1 <= nu <= nlevel):
        raise ValueError("nu must be between 1 and nlevel.")
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

# ==============================================================================
# Patient timeline simulation
# ==============================================================================

def make_patient(rng, dose, arrival_day,
                 true_t1, p_surgery, true_t2,
                 incl_to_rt, rt_dur, rt_to_surg, tox1_win, tox2_win,
                 is_bridging=False):
    """
    Simulate one patient's complete outcomes and event timeline.

    Tox1 event time: Uniform(rt_start, rt_start + tox1_win) if tox1 occurs.
    Tox2 event time: Uniform(surgery_day, surgery_day + tox2_win) if tox2 occurs.
    Non-surgery patients have no tox2 information (not counted as tox2 = 0).
    """
    rt_start     = arrival_day + float(incl_to_rt)
    rt_end       = rt_start    + float(rt_dur)
    tox1_win_end = rt_start    + float(tox1_win)

    has_tox1 = bool(rng.random() < float(true_t1[dose]))
    tox1_day = float(rt_start + rng.uniform(0.0, float(tox1_win))) if has_tox1 else None

    has_surgery  = bool(rng.random() < float(p_surgery))
    surgery_day  = float(rt_end + float(rt_to_surg)) if has_surgery else None
    tox2_win_end = float(surgery_day + float(tox2_win)) if has_surgery else None
    has_tox2     = bool(has_surgery and rng.random() < float(true_t2[dose]))
    tox2_day     = float(surgery_day + rng.uniform(0.0, float(tox2_win))) if has_tox2 else None

    return {
        "dose":         int(dose),
        "arrival":      float(arrival_day),
        "rt_start":     rt_start,
        "tox1_win_end": tox1_win_end,
        "has_tox1":     has_tox1,
        "tox1_day":     tox1_day,
        "has_surgery":  has_surgery,
        "surgery_day":  surgery_day,
        "tox2_win_end": tox2_win_end,
        "has_tox2":     has_tox2,
        "tox2_day":     tox2_day,
        "is_bridging":  bool(is_bridging),
    }

def patient_follow_up_end(pt):
    """Latest day on which all follow-up for this patient is complete."""
    last = pt["tox1_win_end"]
    if pt["has_surgery"] and pt["tox2_win_end"] is not None:
        last = max(last, pt["tox2_win_end"])
    return float(last)

# ==============================================================================
# TITE weight computation
# ==============================================================================

def tite_weights(patients, current_day, tox1_win, tox2_win, n_levels):
    """
    Compute fractional TITE weights for all enrolled patients at current_day.

    Returns four float arrays of shape (n_levels,):
      n1, y1   — effective n and events for tox1 model (all patients)
      n2, y2   — effective n and events for tox2 model (surgery patients only)

    Weights are capped at 1.  Observed events always contribute weight = 1.
    Patients who have not yet entered their relevant window contribute 0.
    """
    n1 = np.zeros(n_levels, dtype=float)
    y1 = np.zeros(n_levels, dtype=float)
    n2 = np.zeros(n_levels, dtype=float)
    y2 = np.zeros(n_levels, dtype=float)
    t  = float(current_day)

    for p in patients:
        d = p["dose"]

        # ── tox1 weight ──────────────────────────────────────────────────────
        if t < p["rt_start"]:
            w1 = 0.0
        elif p["has_tox1"] and p["tox1_day"] is not None and p["tox1_day"] <= t:
            w1 = 1.0
        elif t >= p["tox1_win_end"]:
            w1 = 1.0
        else:
            w1 = (t - p["rt_start"]) / float(tox1_win)
        w1 = float(np.clip(w1, 0.0, 1.0))
        n1[d] += w1
        if p["has_tox1"] and p["tox1_day"] is not None and p["tox1_day"] <= t:
            y1[d] += 1.0

        # ── tox2 weight (surgery patients only) ──────────────────────────────
        if p["has_surgery"] and p["surgery_day"] is not None:
            sd = p["surgery_day"]
            if t < sd:
                w2 = 0.0
            elif p["has_tox2"] and p["tox2_day"] is not None and p["tox2_day"] <= t:
                w2 = 1.0
            elif p["tox2_win_end"] is not None and t >= p["tox2_win_end"]:
                w2 = 1.0
            else:
                w2 = (t - sd) / float(tox2_win)
            w2 = float(np.clip(w2, 0.0, 1.0))
            n2[d] += w2
            if p["has_tox2"] and p["tox2_day"] is not None and p["tox2_day"] <= t:
                y2[d] += 1.0

    return n1, y1, n2, y2

# ==============================================================================
# CRM posterior via Gauss-Hermite quadrature
# Accepts fractional n (TITE weights); log-likelihood form is standard.
# ==============================================================================

def posterior_via_gh(sigma, skeleton, n_per, dlt_per, gh_n=61):
    sk = safe_probs(skeleton)
    n  = np.asarray(n_per,   dtype=float)
    y  = np.asarray(dlt_per, dtype=float)
    x, w = np.polynomial.hermite.hermgauss(int(gh_n))
    theta = float(sigma) * np.sqrt(2.0) * x
    P  = sk[None, :] ** np.exp(theta)[:, None]
    P  = safe_probs(P)
    ll = (y[None, :] * np.log(P) + (n[None, :] - y[None, :]) * np.log(1 - P)).sum(axis=1)
    log_unnorm = np.log(w) + ll
    m          = np.max(log_unnorm)
    unnorm     = np.exp(log_unnorm - m)
    post_w     = unnorm / np.sum(unnorm)
    return post_w, P

def crm_posterior_summaries(sigma, skeleton, n_per, dlt_per, target, gh_n=61):
    post_w, P     = posterior_via_gh(sigma, skeleton, n_per, dlt_per, gh_n=gh_n)
    post_mean     = (post_w[:, None] * P).sum(axis=0)
    overdose_prob = (post_w[:, None] * (P > float(target))).sum(axis=0)
    return post_mean, overdose_prob

def crm_choose_next(sigma, skel1, skel2,
                    n1, y1, n2, y2,
                    current_level, target1, target2,
                    ewoc_alpha=None, max_step=1, gh_n=61,
                    enforce_guardrail=True, highest_tried=-1, n_levels=5):
    """
    Select the next dose level for the upcoming cohort.

    EWOC ON  (ewoc_alpha is not None):
      Admissible set = doses where P(tox1>target1) < alpha AND P(tox2>target2) < alpha.
      Among admissible doses, pick the HIGHEST (maximise dose subject to joint safety).

    EWOC OFF (ewoc_alpha is None):
      No overdose-probability filter is applied.
      Among all doses (subject to step and guardrail constraints), pick the dose whose
      posterior mean P(tox1) is CLOSEST to target1.  This is the standard CRM
      "argmin |pm − target|" rule — it is coherent with the target-based design and
      does NOT blindly seek the highest dose regardless of posterior evidence.
    """
    pm1, od1 = crm_posterior_summaries(sigma, skel1, n1, y1, target1, gh_n=gh_n)
    _,   od2 = crm_posterior_summaries(sigma, skel2, n2, y2, target2, gh_n=gh_n)

    if ewoc_alpha is None:
        candidates = np.arange(n_levels)
    else:
        candidates = np.where((od1 < float(ewoc_alpha)) & (od2 < float(ewoc_alpha)))[0]

    if candidates.size == 0:
        candidates = np.array([0], dtype=int)

    if ewoc_alpha is not None:
        k = int(candidates.max())
    else:
        dist = np.abs(pm1[candidates] - float(target1))
        k = int(candidates[int(np.argmin(dist))])

    k = int(np.clip(k, current_level - int(max_step), current_level + int(max_step)))
    if enforce_guardrail and highest_tried >= 0:
        k = int(min(k, int(highest_tried) + 1))
    return int(np.clip(k, 0, n_levels - 1))

def crm_select_mtd(sigma, skel1, skel2,
                   n1, y1, n2, y2,
                   target1, target2,
                   ewoc_alpha=None, gh_n=61, restrict_to_tried=True):
    """
    Select the final MTD from the completed trial data.

    EWOC ON  (ewoc_alpha is not None):
      Admissible set = doses where both OD probs < alpha (joint safety filter).
      Among admissible (and tried) doses, pick the HIGHEST.

    EWOC OFF (ewoc_alpha is None):
      No overdose-probability filter.  Among tried doses, pick the one whose
      posterior mean P(tox1) is closest to target1 (standard CRM rule).
      This prevents the selector from defaulting to the highest tried dose
      regardless of posterior evidence.
    """
    pm1, od1 = crm_posterior_summaries(sigma, skel1, n1, y1, target1, gh_n=gh_n)
    _,   od2 = crm_posterior_summaries(sigma, skel2, n2, y2, target2, gh_n=gh_n)
    n_levels = len(skel1)

    if ewoc_alpha is None:
        candidates = np.arange(n_levels)
    else:
        candidates = np.where((od1 < float(ewoc_alpha)) & (od2 < float(ewoc_alpha)))[0]

    if candidates.size == 0:
        return 0

    if restrict_to_tried:
        tried = np.where(np.asarray(n1) > 0)[0]
        if tried.size > 0:
            candidates2 = np.intersect1d(candidates, tried)
            candidates = candidates2 if candidates2.size > 0 else tried

    if ewoc_alpha is not None:
        return int(candidates.max())
    else:
        dist = np.abs(pm1[candidates] - float(target1))
        return int(candidates[int(np.argmin(dist))])

# ==============================================================================
# TITE-CRM trial runner
# ==============================================================================

def run_tite_crm(
    true_t1, p_surgery, true_t2,
    target1, target2,
    skel1, skel2,
    sigma=1.0, start_level=0, max_n=27, cohort_size=3,
    accrual_per_month=1.5,
    incl_to_rt=21, rt_dur=14, rt_to_surg=84, tox1_win=84, tox2_win=30,
    max_step=1, gh_n=61,
    enforce_guardrail=True, restrict_final_to_tried=True,
    ewoc_on=True, ewoc_alpha=0.25,
    burn_in=True, rng=None,
    collect_trace=False,
):
    """
    TITE-CRM trial simulation.

    Accrual model: Poisson process via exponential inter-arrival times.
    Decision timing: after each cohort's last patient arrives, compute TITE
      weights for ALL enrolled patients at that calendar time, then update.
      No waiting between cohorts — partial weights carry safety information.

    Burn-in: escalate one level at a time until the first observed tox1 event
      (observed = tox1_day <= decision_day).  Then hand off to CRM.

    collect_trace: when True, record a decision-level trace dict for every
      cohort update.  Adds negligible runtime; used only for the first trial.

    Returns (selected_level, patients_list, study_days, trace).
    """
    if rng is None:
        rng = np.random.default_rng()
    true_t1  = np.asarray(true_t1, dtype=float)
    true_t2  = np.asarray(true_t2, dtype=float)
    n_levels = len(true_t1)
    rate_per_day = float(accrual_per_month) / MONTH

    level         = int(start_level)
    patients      = []
    highest_tried = -1
    current_day   = 0.0
    burn_active   = bool(burn_in)
    ewoc_eff      = float(ewoc_alpha) if ewoc_on else None
    trace         = []
    cohort_step   = 0

    while len(patients) < int(max_n):
        n_add        = min(int(cohort_size), int(max_n) - len(patients))
        cohort_start = len(patients)

        for _ in range(n_add):
            current_day += rng.exponential(1.0 / rate_per_day)
            pt = make_patient(rng, level, current_day,
                              true_t1, p_surgery, true_t2,
                              incl_to_rt, rt_dur, rt_to_surg, tox1_win, tox2_win)
            patients.append(pt)

        decision_day  = current_day
        highest_tried = max(highest_tried, level)

        n1, y1, n2, y2 = tite_weights(
            patients, decision_day, tox1_win, tox2_win, n_levels)

        burn_was_active = burn_active

        if burn_active:
            obs_any_dlt = any(
                p["has_tox1"] and p["tox1_day"] is not None
                and p["tox1_day"] <= decision_day
                for p in patients
            )
            if obs_any_dlt:
                burn_active = False

        if burn_active:
            next_level = min(level + 1, n_levels - 1)
            if next_level == n_levels - 1:
                burn_active = False
        else:
            next_level = crm_choose_next(
                sigma, skel1, skel2,
                n1, y1, n2, y2,
                level, target1, target2,
                ewoc_alpha=ewoc_eff, max_step=max_step, gh_n=gh_n,
                enforce_guardrail=enforce_guardrail,
                highest_tried=highest_tried, n_levels=n_levels,
            )

        if collect_trace:
            pm1, od1 = crm_posterior_summaries(
                sigma, skel1, n1, y1, target1, gh_n=gh_n)
            pm2, od2 = crm_posterior_summaries(
                sigma, skel2, n2, y2, target2, gh_n=gh_n)

            ewoc_mode = "OFF" if ewoc_eff is None else f"ON (α={ewoc_eff:.2f})"

            if ewoc_eff is None:
                allowed_arr = list(range(n_levels))
            else:
                allowed_arr = [int(d) for d in
                               np.where((od1 < ewoc_eff) & (od2 < ewoc_eff))[0]]

            if burn_was_active:
                reason = (f"Burn-in: escalate one level (no tox1 DLT "
                          f"observed yet → L{next_level})")
            elif not allowed_arr:
                reason = "No dose within joint safety bounds → fallback to L0"
            elif ewoc_eff is None:
                cands     = np.arange(n_levels)
                dist      = np.abs(pm1[cands] - float(target1))
                k_target  = int(cands[int(np.argmin(dist))])
                k_step    = int(np.clip(k_target,
                                        level - int(max_step),
                                        level + int(max_step)))
                k_guard   = (int(min(k_step, highest_tried + 1))
                             if enforce_guardrail and highest_tried >= 0
                             else k_step)
                parts = [f"EWOC OFF → argmin|pm1−target1| = L{k_target}"]
                if k_step != k_target:
                    parts.append(f"step-limit → L{k_step}")
                if k_guard != k_step:
                    parts.append(f"guardrail → L{k_guard}")
                reason = f"L{next_level}: " + "; ".join(parts)
            else:
                k_safe  = int(np.array(allowed_arr).max())
                k_step  = int(np.clip(k_safe,
                                      level - int(max_step),
                                      level + int(max_step)))
                k_guard = (int(min(k_step, highest_tried + 1))
                           if enforce_guardrail and highest_tried >= 0
                           else k_step)
                if k_guard < k_step:
                    reason = (f"L{next_level}: guardrail limited "
                              f"(highest tried = L{highest_tried}; "
                              f"max allowed = L{highest_tried + 1})")
                elif k_step < k_safe:
                    reason = (f"L{next_level}: step-size limited "
                              f"(max step = {max_step}; "
                              f"jointly safe optimum was L{k_safe})")
                else:
                    reason = (f"L{next_level}: highest dose satisfying "
                              f"joint tox1 & tox2 safety rule")

            trace.append({
                "step":          cohort_step + 1,
                "decision_day":  decision_day,
                "n_enrolled":    len(patients),
                "current_dose":  level,
                "next_dose":     next_level,
                "highest_tried": highest_tried,
                "burn_in":       burn_was_active,
                "ewoc_mode":     ewoc_mode,
                "n1":   n1.tolist(), "y1": y1.tolist(),
                "n2":   n2.tolist(), "y2": y2.tolist(),
                "pm1":  [round(float(v), 3) for v in pm1],
                "od1":  [round(float(v), 3) for v in od1],
                "pm2":  [round(float(v), 3) for v in pm2],
                "od2":  [round(float(v), 3) for v in od2],
                "allowed":       allowed_arr,
                "reason":        reason,
                "obs_t1":        int(round(y1.sum())),
                "obs_t2":        int(round(y2.sum())),
                "n_surgery":     sum(1 for p in patients if p["has_surgery"]),
                "n1_sum":        float(n1.sum()),
                "n2_sum":        float(n2.sum()),
                "cohort_pts":    list(range(cohort_start, len(patients))),
            })

        cohort_step += 1
        level = next_level

    if patients:
        study_days = max(patient_follow_up_end(p) for p in patients)
        n1f, y1f, n2f, y2f = tite_weights(
            patients, study_days, tox1_win, tox2_win, n_levels)
    else:
        study_days = 0.0
        n1f = y1f = n2f = y2f = np.zeros(n_levels)

    selected = crm_select_mtd(
        sigma, skel1, skel2,
        n1f, y1f, n2f, y2f,
        target1, target2,
        ewoc_alpha=ewoc_eff, gh_n=gh_n,
        restrict_to_tried=restrict_final_to_tried,
    )
    return int(selected), patients, float(study_days), trace

# ==============================================================================
# 6+3 TITE runner with lower-dose bridging
# ==============================================================================

def _tox1_evaluable(pt, day):
    """True once the patient's full tox1 follow-up window has elapsed."""
    return float(day) >= pt["tox1_win_end"]

def _fully_evaluable(pt, day):
    """
    True when tox1 window is complete AND (if surgery) tox2 window is complete.
    Non-surgery patients are fully evaluable once their tox1 window ends.
    """
    if not _tox1_evaluable(pt, day):
        return False
    if pt["has_surgery"] and pt["tox2_win_end"] is not None:
        return float(day) >= pt["tox2_win_end"]
    return True

def run_tite_6plus3(
    true_t1, p_surgery, true_t2,
    start_level=0, max_n=27,
    accrual_per_month=1.5,
    incl_to_rt=21, rt_dur=14, rt_to_surg=84, tox1_win=84, tox2_win=30,
    a6_esc_max=0, a6_stop_min=2, a9_esc_max=1,
    s6_esc_max=1, s6_stop_min=3, s9_esc_max=3, s9_stop_min=4,
    rng=None,
):
    """
    Modified 6+3 in TITE setting with lower-dose bridging.

    Returns (selected, all_patients, study_days, n_bridging_total).
    """
    if rng is None:
        rng = np.random.default_rng()
    true_t1  = np.asarray(true_t1, dtype=float)
    true_t2  = np.asarray(true_t2, dtype=float)
    n_levels = len(true_t1)
    rate_per_day = float(accrual_per_month) / MONTH

    all_patients    = []
    current_day     = 0.0
    last_acceptable = None
    study_days      = 0.0
    eval_dose       = int(start_level)

    def _arrive():
        nonlocal current_day
        current_day += rng.exponential(1.0 / rate_per_day)
        return current_day

    def _enroll(dose, bridging):
        day = _arrive()
        pt  = make_patient(rng, dose, day,
                           true_t1, p_surgery, true_t2,
                           incl_to_rt, rt_dur, rt_to_surg, tox1_win, tox2_win,
                           is_bridging=bridging)
        all_patients.append(pt)
        return pt

    while len(all_patients) < int(max_n):
        eval_cohort = []
        safe_dose   = max(0, eval_dose - 1)

        while len(all_patients) < int(max_n):
            pt = _enroll(eval_dose, bridging=False)
            eval_cohort.append(pt)
            n_t = len(eval_cohort)
            n_s = sum(1 for p in eval_cohort if p["has_surgery"])
            if n_t >= 6 and n_s >= 6:
                break

        while (len(all_patients) < int(max_n) and
               not all(_fully_evaluable(p, current_day) for p in eval_cohort)):
            _enroll(safe_dose, bridging=True)

        if not all(_fully_evaluable(p, current_day) for p in eval_cohort):
            current_day = max(patient_follow_up_end(p) for p in eval_cohort)
        study_days = max(study_days, current_day)

        nt   = len(eval_cohort)
        nsg  = sum(1 for p in eval_cohort if p["has_surgery"])
        ya   = sum(1 for p in eval_cohort if p["has_tox1"])
        ys   = sum(1 for p in eval_cohort if p["has_tox2"])

        sub_eval_p1 = nsg >= 6
        a6_esc_adj  = int(np.floor(nt * int(a6_esc_max)  / 6.0))
        a6_stop_adj = int(np.ceil( nt * int(a6_stop_min) / 6.0))

        stop_p1 = (ya >= a6_stop_adj or
                   (sub_eval_p1 and ys >= int(s6_stop_min)))
        esc_p1  = (nt >= 6 and sub_eval_p1 and
                   ya <= a6_esc_adj and ys <= int(s6_esc_max))

        if stop_p1:
            if eval_dose > 0:
                eval_dose -= 1
            last_acceptable = eval_dose
            break

        if esc_p1:
            last_acceptable = eval_dose
            if eval_dose < n_levels - 1:
                eval_dose += 1
                continue
            break

        while len(all_patients) < int(max_n):
            n_t = len(eval_cohort)
            n_s = sum(1 for p in eval_cohort if p["has_surgery"])
            if n_t >= 9 and n_s >= 9:
                break
            pt = _enroll(eval_dose, bridging=False)
            eval_cohort.append(pt)

        while (len(all_patients) < int(max_n) and
               not all(_fully_evaluable(p, current_day) for p in eval_cohort)):
            _enroll(safe_dose, bridging=True)

        if not all(_fully_evaluable(p, current_day) for p in eval_cohort):
            current_day = max(patient_follow_up_end(p) for p in eval_cohort)
        study_days = max(study_days, current_day)

        nt   = len(eval_cohort)
        nsg  = sum(1 for p in eval_cohort if p["has_surgery"])
        ya   = sum(1 for p in eval_cohort if p["has_tox1"])
        ys   = sum(1 for p in eval_cohort if p["has_tox2"])

        sub_eval_p2 = nsg >= 9
        a9_esc_adj  = int(np.floor(nt * int(a9_esc_max) / 9.0))

        esc_p2  = (sub_eval_p2 and
                   ya <= a9_esc_adj and ys <= int(s9_esc_max))
        stop_p2 = (not esc_p2 and
                   (not sub_eval_p2 or ys >= int(s9_stop_min) or ya > a9_esc_adj))

        if esc_p2:
            last_acceptable = eval_dose
            if eval_dose < n_levels - 1:
                eval_dose += 1
                continue
            break

        if eval_dose > 0:
            eval_dose -= 1
        last_acceptable = eval_dose
        break

    selected   = 0 if last_acceptable is None else int(last_acceptable)
    n_bridging = sum(1 for p in all_patients if p["is_bridging"])
    return int(selected), all_patients, float(study_days), int(n_bridging)


# ==============================================================================
# Timeline figure
# ==============================================================================

def _draw_timeline(incl_to_rt, rt_dur, rt_to_surg, tox2_win):
    """
    Two-row timeline: solid phase bars on the bottom row, toxicity-window
    arrows on the top row.  No overlapping fills.

    Tox1 window: RT start → Surgery (ends exactly at surgery).
    Tox2 window: Surgery  → Surgery + tox2_win.
    """
    _BG = _DARK_BG
    _FG = _DARK_FG

    fig, ax = plt.subplots(figsize=(9.0, 1.6), dpi=120)
    fig.patch.set_facecolor(_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(_BG)

    rt_start = incl_to_rt
    rt_end   = rt_start + rt_dur
    surg     = rt_end   + rt_to_surg
    t1_end   = surg                    # tox1 window ends at surgery
    t2_end   = surg     + tox2_win
    total    = t2_end * 1.05

    def x(d): return float(d) / total

    # ── Bottom row: solid phase bars (no tox overlays) ─────────────────────
    y0, h_bar = 0.14, 0.32
    phases = [
        (0,        rt_start, "#64b5f6", 0.90, "Incl → RT start"),
        (rt_start, rt_end,   "#1976d2", 0.95, "RT"),
        (rt_end,   surg,     "#78909c", 0.85, "RT end → Surgery"),
        (surg,     t2_end,   "#4a4a6a", 0.70, "Post-surgery"),
    ]
    for x0_, x1_, col, alpha, _lbl in phases:
        bar = mpatches.FancyBboxPatch(
            (x(x0_), y0), x(x1_) - x(x0_), h_bar,
            boxstyle="square,pad=0",
            facecolor=col, alpha=alpha,
            edgecolor="#bdbdbd", linewidth=0.5,
        )
        ax.add_patch(bar)

    # ── Top row: toxicity-window double-headed arrows ───────────────────────
    # Tox1 spans RT start → Surgery, Tox2 spans Surgery → Done.
    # They are adjacent (non-overlapping) so they share the same y row.
    y_arr     = 0.74   # arrow shaft y
    y_arr_lbl = 0.88   # label above shaft

    tox_spans = [
        (rt_start, t1_end, "#ffb300", "Tox1 window"),   # amber
        (surg,     t2_end, "#ef5350", "Tox2 window"),   # red
    ]
    for xa0, xa1, col, lbl in tox_spans:
        xp0, xp1 = x(xa0), x(xa1)
        # Double-headed arrow
        ax.annotate(
            "", xy=(xp1, y_arr), xytext=(xp0, y_arr),
            arrowprops=dict(
                arrowstyle="<->", color=col,
                lw=1.8, mutation_scale=11,
            ),
        )
        # Faint shading behind arrow to make span clearer
        ax.axvspan(xp0, xp1, ymin=0.64, ymax=0.80,
                   color=col, alpha=0.08)
        # Label centred above arrow
        ax.text(
            (xp0 + xp1) / 2, y_arr_lbl, lbl,
            ha="center", va="bottom",
            fontsize=7.5, color=col, fontweight="bold",
        )

    # ── Milestone tick-marks and labels ────────────────────────────────────
    markers = [
        (0,        "Incl"),
        (rt_start, "RT\nstart"),
        (rt_end,   "RT\nend"),
        (surg,     "Surgery"),
        (t2_end,   "Done"),
    ]
    for d, lbl in markers:
        xp = x(d)
        ax.axvline(xp, ymin=0.10, ymax=0.60,
                   color=_FG, lw=0.8, alpha=0.55)
        ax.text(xp, 0.01, lbl, ha="center", va="bottom",
                fontsize=7.0, color=_FG, fontweight="bold")

    fig.tight_layout(pad=0.10)
    return fig


# ==============================================================================
# Quality / optimality helpers
# ==============================================================================

def _quality_score(selected, true_t1, true_t2, target1, target2):
    """Asymmetric exponential loss: penalises overdose more than underdose."""
    d1 = float(true_t1[selected]) - float(target1)
    d2 = float(true_t2[selected]) - float(target2)
    bd = max(d1, d2)                          # binding (worst) endpoint
    w  = 1.8 if bd > 0 else 1.0              # w_over=1.8, w_under=1.0
    return float(np.exp(-6.0 * w * abs(bd)))


def _true_optimal(true_t1, true_t2, target1, target2):
    """Dose with highest quality score under the same asymmetric loss used in
    _quality_score (overdose penalised 1.8×, underdose 1.0×).
    Consistent with the quality metric so 'correct selection' and 'quality
    score' agree on what the best dose is."""
    scores = [_quality_score(d, true_t1, true_t2, target1, target2)
              for d in range(len(true_t1))]
    return int(np.argmax(scores))


# ==============================================================================
# Parameter sweep runners
# ==============================================================================

def run_parameter_sweep(param_name, param_values, base_ss,
                        true_t1, true_t2, skel_t1, skel_t2,
                        n_sim, seed):
    """
    Run TITE-CRM simulations sweeping one parameter across *param_values*.

    Parameters
    ----------
    param_name   : "sigma" | "ewoc_alpha" | "max_n" | "cohort_size"
    param_values : list of values; use None in the list for EWOC OFF
    base_ss      : dict of fixed scenario settings (see keys below)
    true_t1/t2   : array-like, true tox rates per dose level
    skel_t1/t2   : array-like, CRM skeletons
    n_sim        : int, replications per grid point
    seed         : int, base RNG seed (each grid point gets seed + idx*1000)

    Required base_ss keys
    ---------------------
    target_tox1, target_tox2, p_surgery,
    sigma, ewoc_on, ewoc_alpha,
    max_n, cohort_size, start_level,
    accrual_per_month, incl_to_rt, rt_dur, rt_to_surg, tox2_win,
    max_step, gh_n, burn_in, enforce_guardrail, restrict_final_to_tried

    Returns
    -------
    pd.DataFrame with columns:
        param_label, param_raw, quality_score, pct_correct_selection,
        overdose_rate
    """
    true_t1 = np.asarray(true_t1, dtype=float)
    true_t2 = np.asarray(true_t2, dtype=float)
    t1      = float(base_ss["target_tox1"])
    t2      = float(base_ss["target_tox2"])
    optimal = _true_optimal(true_t1, true_t2, t1, t2)

    # Fixed kwargs that never change across the sweep.
    # tox1_win is derived from rt_dur + rt_to_surg (matching simulator convention).
    base_kw = dict(
        true_t1=true_t1, p_surgery=float(base_ss["p_surgery"]), true_t2=true_t2,
        target1=t1, target2=t2, skel1=skel_t1, skel2=skel_t2,
        sigma=float(base_ss["sigma"]),
        start_level=int(base_ss["start_level"]),
        max_n=int(base_ss["max_n"]),
        cohort_size=int(base_ss["cohort_size"]),
        accrual_per_month=float(base_ss["accrual_per_month"]),
        incl_to_rt=int(base_ss["incl_to_rt"]),
        rt_dur=int(base_ss["rt_dur"]),
        rt_to_surg=int(base_ss["rt_to_surg"]),
        tox1_win=int(base_ss["rt_dur"]) + int(base_ss["rt_to_surg"]),
        tox2_win=int(base_ss["tox2_win"]),
        max_step=int(base_ss["max_step"]),
        gh_n=int(base_ss["gh_n"]),
        burn_in=bool(base_ss["burn_in"]),
        enforce_guardrail=bool(base_ss["enforce_guardrail"]),
        restrict_final_to_tried=bool(base_ss["restrict_final_to_tried"]),
        ewoc_on=bool(base_ss["ewoc_on"]),
        ewoc_alpha=float(base_ss["ewoc_alpha"]),
    )

    rows = []
    for idx, pv in enumerate(param_values):
        kw = dict(base_kw)  # shallow copy — scalars only

        if param_name == "sigma":
            kw["sigma"] = float(pv)
            label = f"{float(pv):.3g}"
        elif param_name == "ewoc_alpha":
            if pv is None:
                kw["ewoc_on"] = False
                label = "OFF"
            else:
                kw["ewoc_on"]    = True
                kw["ewoc_alpha"] = float(pv)
                label = f"{float(pv):.2f}"
        elif param_name == "max_n":
            kw["max_n"] = int(pv)
            label = str(int(pv))
        elif param_name == "cohort_size":
            kw["cohort_size"] = int(pv)
            label = str(int(pv))
        elif param_name == "prior_nu_t1":
            kw["skel1"] = dfcrm_getprior(
                float(base_ss["prior_hw1"]), float(base_ss["prior_pt1"]),
                int(pv), len(true_t1),
                model=str(base_ss.get("prior_model_str", "empiric")),
                intcpt=float(base_ss.get("logistic_intcpt", 3.0)),
            )
            label = f"L{int(pv)}"
        elif param_name == "prior_nu_t2":
            kw["skel2"] = dfcrm_getprior(
                float(base_ss["prior_hw2"]), float(base_ss["prior_pt2"]),
                int(pv), len(true_t2),
                model=str(base_ss.get("prior_model_str", "empiric")),
                intcpt=float(base_ss.get("logistic_intcpt", 3.0)),
            )
            label = f"L{int(pv)}"
        elif param_name == "enforce_guardrail":
            kw["enforce_guardrail"] = bool(pv)
            label = "ON" if bool(pv) else "OFF"
        elif param_name == "restrict_final_mtd":
            kw["restrict_final_to_tried"] = bool(pv)
            label = "ON" if bool(pv) else "OFF"
        elif param_name == "burn_in":
            kw["burn_in"] = bool(pv)
            label = "ON" if bool(pv) else "OFF"
        else:
            raise ValueError(f"Unknown param_name: {param_name!r}")

        rng = np.random.default_rng(int(seed) + idx * 1000)
        scores, correct, overdosed = [], [], []
        for _ in range(int(n_sim)):
            # run_tite_crm returns (selected, patients, study_days, trace)
            sel, *_ = run_tite_crm(**kw, rng=rng)
            scores.append(_quality_score(sel, true_t1, true_t2, t1, t2))
            correct.append(int(sel == optimal))
            overdosed.append(int(max(float(true_t1[sel]) - t1,
                                     float(true_t2[sel]) - t2) > 0))

        rows.append(dict(
            param_label=label,
            param_raw=pv,
            n_patients=int(kw["max_n"]),          # actual max-N for this point
            quality_score=float(np.mean(scores)),
            pct_correct_selection=float(np.mean(correct)) * 100.0,
            overdose_rate=float(np.mean(overdosed)) * 100.0,
        ))

    return pd.DataFrame(rows)


def run_prior_nu_sweep(nu1_values, nu2_values, base_ss,
                       prior_pt1, prior_hw1, prior_pt2, prior_hw2,
                       prior_model_str, logistic_intcpt,
                       true_t1, true_t2, n_sim, seed):
    """
    Sweep over all (prior_nu_t1, prior_nu_t2) combinations.

    For each (nu1, nu2) pair the CRM skeleton is recomputed via dfcrm_getprior,
    then run_tite_crm is called n_sim times.  All other settings come from
    base_ss (same format as run_parameter_sweep).

    Returns a pd.DataFrame with columns:
        prior_nu_t1, prior_nu_t2, param_label,
        quality_score, pct_correct_selection, overdose_rate
    """
    true_t1  = np.asarray(true_t1, dtype=float)
    true_t2  = np.asarray(true_t2, dtype=float)
    t1       = float(base_ss["target_tox1"])
    t2       = float(base_ss["target_tox2"])
    optimal  = _true_optimal(true_t1, true_t2, t1, t2)
    n_levels = len(true_t1)

    # Fixed kwargs (everything except the skeletons, which vary per row)
    base_kw = dict(
        true_t1=true_t1, p_surgery=float(base_ss["p_surgery"]), true_t2=true_t2,
        target1=t1, target2=t2,
        sigma=float(base_ss["sigma"]),
        start_level=int(base_ss["start_level"]),
        max_n=int(base_ss["max_n"]),
        cohort_size=int(base_ss["cohort_size"]),
        accrual_per_month=float(base_ss["accrual_per_month"]),
        incl_to_rt=int(base_ss["incl_to_rt"]),
        rt_dur=int(base_ss["rt_dur"]),
        rt_to_surg=int(base_ss["rt_to_surg"]),
        tox1_win=int(base_ss["rt_dur"]) + int(base_ss["rt_to_surg"]),
        tox2_win=int(base_ss["tox2_win"]),
        max_step=int(base_ss["max_step"]),
        gh_n=int(base_ss["gh_n"]),
        burn_in=bool(base_ss["burn_in"]),
        enforce_guardrail=bool(base_ss["enforce_guardrail"]),
        restrict_final_to_tried=bool(base_ss["restrict_final_to_tried"]),
        ewoc_on=bool(base_ss["ewoc_on"]),
        ewoc_alpha=float(base_ss["ewoc_alpha"]),
    )

    rows = []
    for idx1, nu1 in enumerate(nu1_values):
        skel1 = dfcrm_getprior(
            prior_hw1, prior_pt1, int(nu1), n_levels,
            model=prior_model_str, intcpt=logistic_intcpt,
        )
        for idx2, nu2 in enumerate(nu2_values):
            skel2 = dfcrm_getprior(
                prior_hw2, prior_pt2, int(nu2), n_levels,
                model=prior_model_str, intcpt=logistic_intcpt,
            )
            kw = dict(base_kw, skel1=skel1, skel2=skel2)
            rng = np.random.default_rng(int(seed) + idx1 * 100 + idx2)
            scores, correct, overdosed = [], [], []
            for _ in range(int(n_sim)):
                sel, *_ = run_tite_crm(**kw, rng=rng)
                scores.append(_quality_score(sel, true_t1, true_t2, t1, t2))
                correct.append(int(sel == optimal))
                overdosed.append(int(max(float(true_t1[sel]) - t1,
                                         float(true_t2[sel]) - t2) > 0))
            rows.append(dict(
                prior_nu_t1=int(nu1),
                prior_nu_t2=int(nu2),
                param_label=f"ν₁=L{nu1} / ν₂=L{nu2}",
                quality_score=float(np.mean(scores)),
                pct_correct_selection=float(np.mean(correct)) * 100.0,
                overdose_rate=float(np.mean(overdosed)) * 100.0,
            ))

    return pd.DataFrame(rows)


# ==============================================================================
# DE parameter helpers and batch-report generators
# ==============================================================================

_DE_PARAM_DISPLAY_NAMES = {
    "sigma":       "Prior sigma (σ)",
    "ewoc_alpha":  "EWOC α — overdose threshold",
    "max_n":       "Maximum sample size (max N)",
    "cohort_size": "Cohort size — patients per dose decision",
    "prior_nu_t1": "Prior MTD level — tox1 (acute)",
    "prior_nu_t2": "Prior MTD level — tox2 (subacute / surgery)",
}

_DE_ALL_PARAMS = [
    "sigma", "ewoc_alpha", "max_n", "cohort_size", "prior_nu_t1", "prior_nu_t2",
    "enforce_guardrail", "restrict_final_mtd", "burn_in",
]


def _de_pv_for_param(param_name, ss, speed=False):
    """Return (pv_list, display_label) for *param_name* using session-state values.

    Falls back to safe defaults for any key that has not yet been set.
    """
    if param_name == "sigma":
        pv = np.linspace(
            float(ss.get("de_sig_min", 0.3)),
            float(ss.get("de_sig_max", 2.0)),
            int(ss.get("de_sig_pts", 8)),
        ).tolist()
        if speed:
            pv = pv[:8]
        return pv, "σ (prior sigma)"

    if param_name == "ewoc_alpha":
        inc_off  = bool(ss.get("de_inc_off", True))
        pv_num   = np.linspace(
            float(ss.get("de_ea_min", 0.05)),
            float(ss.get("de_ea_max", 0.60)),
            int(ss.get("de_ea_pts", 8)),
        ).tolist()
        combined = ([None] if inc_off else []) + pv_num
        if speed:
            combined = combined[:8]
        return combined, "EWOC α"

    if param_name == "max_n":
        pv = list(ss.get("de_max_n_vals",
                         [12, 15, 18, 21, 24, 27, 30, 33, 36]))
        if speed:
            pv = pv[:3]
        return pv, "Maximum total patients (max N)"

    if param_name == "cohort_size":
        pv = list(ss.get("de_cohort_vals", [1, 2, 3, 4]))
        if speed:
            pv = pv[:3]
        return pv, "Cohort size (patients per dose decision)"

    if param_name == "prior_nu_t1":
        pv = list(ss.get("de_nu1_vals", [1, 2, 3, 4, 5]))
        if speed:
            pv = pv[:3]
        return pv, "Prior MTD level — tox1 (acute)"

    if param_name == "prior_nu_t2":
        pv = list(ss.get("de_nu2_vals", [1, 2, 3, 4, 5]))
        if speed:
            pv = pv[:3]
        return pv, "Prior MTD level — tox2 (subacute / surgery)"

    _bool_param_labels = {
        "enforce_guardrail":  "Guardrail (next dose ≤ highest tried + 1)",
        "restrict_final_mtd": "Final MTD restricted to tried doses",
        "burn_in":            "Burn-in until first tox1 DLT",
    }
    if param_name in _bool_param_labels:
        return [False, True], _bool_param_labels[param_name]

    raise ValueError(f"Unknown param_name: {param_name!r}")


def _generate_de_html_report(param_name, param_label, result_df,
                              base_ss, n_sim, seed, pv_list,
                              fig_b64, ts_str, run_label=""):
    """Build a self-contained HTML Design Exploration batch report."""
    param_display = _DE_PARAM_DISPLAY_NAMES.get(param_name, param_name)
    title = f"Design Exploration — {param_display}"
    if run_label:
        title += f"  ·  {run_label}"

    ewoc_off_included = (param_name == "ewoc_alpha" and None in pv_list)

    def _fv(v):
        if v is None:
            return "EWOC OFF"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    values_str = ", ".join(_fv(v) for v in pv_list)
    yn = lambda b: "Yes" if b else "No"

    cfg_rows = [
        ("Target tox1 (acute)",           f"{base_ss['target_tox1']:.3f}"),
        ("Target tox2 (surgery/subacute)", f"{base_ss['target_tox2']:.3f}"),
        ("Prior sigma (σ)",                f"{base_ss['sigma']:.3g}"),
        ("EWOC enabled",                   yn(base_ss["ewoc_on"])),
        ("EWOC α",  f"{base_ss['ewoc_alpha']:.3g}" if base_ss["ewoc_on"] else "N/A"),
        ("Max patients (max N)",           str(base_ss["max_n"])),
        ("Cohort size",                    str(base_ss["cohort_size"])),
        ("Start level",                    f"L{base_ss['start_level'] + 1}"),
        ("Accrual per month",              f"{base_ss['accrual_per_month']:.1f}"),
        ("RT duration (days)",             str(base_ss["rt_dur"])),
        ("RT → surgery gap (days)",        str(base_ss["rt_to_surg"])),
        ("Tox2 window (days)",             str(base_ss["tox2_win"])),
        ("Burn-in until first DLT",        yn(base_ss["burn_in"])),
        ("Guardrail (≤ highest tried + 1)",yn(base_ss["enforce_guardrail"])),
        ("Final MTD restricted to tried",  yn(base_ss["restrict_final_to_tried"])),
        ("Simulations per point",          str(n_sim)),
        ("Random seed",                    str(seed)),
    ]
    cfg_html = "\n".join(
        f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in cfg_rows
    )

    sweep_rows = [
        ("Parameter swept",     param_display),
        ("Sweep points",        str(len(pv_list))),
        ("Values",              values_str),
    ]
    if param_name == "ewoc_alpha":
        sweep_rows.append(("EWOC OFF included as a point", yn(ewoc_off_included)))
    sweep_html = "\n".join(
        f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in sweep_rows
    )

    df_disp = result_df[["param_label", "n_patients", "quality_score",
                          "pct_correct_selection", "overdose_rate"]].copy()
    df_disp.columns = [param_label, "N patients",
                       "Quality score", "% Correct selection", "Overdose rate (%)"]
    df_disp["Quality score"]        = df_disp["Quality score"].round(4)
    df_disp["% Correct selection"]  = df_disp["% Correct selection"].round(1)
    df_disp["Overdose rate (%)"]    = df_disp["Overdose rate (%)"].round(1)
    results_html = df_disp.to_html(index=False, border=0,
                                   classes="results-tbl", justify="left")

    css = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     max-width:1100px;margin:0 auto;padding:28px 44px;color:#2c3e50;background:#fafafa}
h1{color:#1a237e;border-bottom:3px solid #3949ab;padding-bottom:10px;margin-bottom:4px}
h2{color:#283593;margin-top:2.2em;border-left:4px solid #7986cb;padding-left:12px}
.meta{color:#666;font-size:.88em;margin-bottom:2em}
table{border-collapse:collapse;width:100%;margin:1em 0;background:#fff;
      box-shadow:0 1px 4px rgba(0,0,0,.08);border-radius:6px;overflow:hidden}
th{background:#3949ab;color:#fff;padding:9px 14px;text-align:left;
   font-weight:600;font-size:.91em}
td{padding:8px 14px;border-bottom:1px solid #e8eaf6;font-size:.92em}
tr:last-child td{border-bottom:none}
tr:nth-child(even) td{background:#f5f7ff}
img{max-width:100%;border:1px solid #ddd;border-radius:6px;
    box-shadow:0 2px 8px rgba(0,0,0,.10);margin:1em 0;display:block}
.footer{color:#aaa;font-size:.78em;margin-top:3em;
        border-top:1px solid #eee;padding-top:12px}
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {ts_str}&nbsp;&nbsp;|&nbsp;&nbsp;{len(pv_list)} sweep
points&nbsp;&nbsp;|&nbsp;&nbsp;{n_sim:,} simulations per point</p>

<h2>Base Configuration</h2>
<table><tbody>{cfg_html}</tbody></table>

<h2>Sweep Definition</h2>
<table><tbody>{sweep_html}</tbody></table>

<h2>Results</h2>
{results_html}

<h2>Figures</h2>
<img src="{fig_b64}" alt="Sweep results chart">

<p class="footer">CRM Simulator — Design Exploration batch report</p>
</body></html>"""
    return html


def _generate_de_all_html_report(results_list, base_ss, n_sim, seed,
                                  ts_str, run_label=""):
    """Build a single self-contained HTML report covering all swept parameters."""
    title = "Design Exploration — All Parameters"
    if run_label:
        title += f"  ·  {run_label}"

    yn = lambda b: "Yes" if b else "No"

    def _fv(v):
        if v is None:
            return "EWOC OFF"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    cfg_rows = [
        ("Target tox1 (acute)",            f"{base_ss['target_tox1']:.3f}"),
        ("Target tox2 (surgery/subacute)",  f"{base_ss['target_tox2']:.3f}"),
        ("Prior sigma (σ)",                 f"{base_ss['sigma']:.3g}"),
        ("EWOC enabled",                    yn(base_ss["ewoc_on"])),
        ("EWOC α",
         f"{base_ss['ewoc_alpha']:.3g}" if base_ss["ewoc_on"] else "N/A"),
        ("Max patients (max N)",            str(base_ss["max_n"])),
        ("Cohort size",                     str(base_ss["cohort_size"])),
        ("Start level",                     f"L{base_ss['start_level'] + 1}"),
        ("Accrual per month",               f"{base_ss['accrual_per_month']:.1f}"),
        ("RT duration (days)",              str(base_ss["rt_dur"])),
        ("RT → surgery gap (days)",         str(base_ss["rt_to_surg"])),
        ("Tox2 window (days)",              str(base_ss["tox2_win"])),
        ("Burn-in until first DLT",         yn(base_ss["burn_in"])),
        ("Guardrail (≤ highest tried + 1)", yn(base_ss["enforce_guardrail"])),
        ("Final MTD restricted to tried",   yn(base_ss["restrict_final_to_tried"])),
        ("Simulations per point",           str(n_sim)),
        ("Random seed",                     str(seed)),
    ]
    cfg_html = "\n".join(
        f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in cfg_rows
    )

    sum_rows = ""
    for r in results_list:
        df      = r["result_df"]
        best    = df.loc[df["quality_score"].idxmax()]
        note    = (" (EWOC OFF included)"
                   if r["param_name"] == "ewoc_alpha" and None in r["pv_list"]
                   else "")
        sum_rows += (
            f"<tr>"
            f"<td>{r['param_label']}{note}</td>"
            f"<td>{len(r['pv_list'])}</td>"
            f"<td>{best['param_label']}</td>"
            f"<td>{best['quality_score']:.4f}</td>"
            f"<td>{best['pct_correct_selection']:.1f}%</td>"
            f"<td>{best['overdose_rate']:.1f}%</td>"
            f"</tr>\n"
        )
    summary_html = (
        "<table><thead><tr>"
        "<th>Parameter</th><th>Points</th><th>Best value</th>"
        "<th>Quality score</th><th>% Correct selection</th>"
        "<th>Overdose rate</th></tr></thead>"
        f"<tbody>{sum_rows}</tbody></table>"
    )

    sections = ""
    for r in results_list:
        pname  = r["param_name"]
        plabel = r["param_label"]
        pv     = r["pv_list"]
        df     = r["result_df"]

        ewoc_row = ""
        if pname == "ewoc_alpha":
            ewoc_row = (
                f"<tr><td>EWOC OFF included as a point</td>"
                f"<td><b>{yn(None in pv)}</b></td></tr>"
            )

        sweep_tbl = (
            f"<table><tbody>"
            f"<tr><td>Sweep points</td><td><b>{len(pv)}</b></td></tr>"
            f"<tr><td>Values</td>"
            f"<td><b>{', '.join(_fv(v) for v in pv)}</b></td></tr>"
            f"{ewoc_row}"
            f"</tbody></table>"
        )

        df_d = df[["param_label", "n_patients", "quality_score",
                   "pct_correct_selection", "overdose_rate"]].copy()
        df_d.columns = [plabel, "N patients", "Quality score",
                        "% Correct selection", "Overdose rate (%)"]
        df_d["Quality score"]        = df_d["Quality score"].round(4)
        df_d["% Correct selection"]  = df_d["% Correct selection"].round(1)
        df_d["Overdose rate (%)"]    = df_d["Overdose rate (%)"].round(1)
        res_tbl = df_d.to_html(index=False, border=0,
                               classes="results-tbl", justify="left")

        ctx_img = (
            f'<img src="{r["context_fig_b64"]}" style="max-width:520px"'
            f' alt="True toxicity vs prior MTD levels — {plabel}">'
            if r.get("context_fig_b64") else ""
        )
        sections += (
            f'<hr class="param-sep">'
            f"<h2>{plabel}</h2>"
            f"{sweep_tbl}"
            f"{res_tbl}"
            f'<img src="{r["fig_b64"]}" alt="Sweep — {plabel}">'
            f"{ctx_img}"
        )

    css = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     max-width:1100px;margin:0 auto;padding:28px 44px;color:#2c3e50;background:#fafafa}
h1{color:#1a237e;border-bottom:3px solid #3949ab;padding-bottom:10px;margin-bottom:4px}
h2{color:#283593;margin-top:2.2em;border-left:4px solid #7986cb;padding-left:12px}
.meta{color:#666;font-size:.88em;margin-bottom:2em}
table{border-collapse:collapse;width:100%;margin:1em 0;background:#fff;
      box-shadow:0 1px 4px rgba(0,0,0,.08);border-radius:6px;overflow:hidden}
th{background:#3949ab;color:#fff;padding:9px 14px;text-align:left;
   font-weight:600;font-size:.91em}
td{padding:8px 14px;border-bottom:1px solid #e8eaf6;font-size:.92em}
tr:last-child td{border-bottom:none}
tr:nth-child(even) td{background:#f5f7ff}
img{max-width:100%;border:1px solid #ddd;border-radius:6px;
    box-shadow:0 2px 8px rgba(0,0,0,.10);margin:1em 0;display:block}
hr.param-sep{border:none;border-top:2px solid #e8eaf6;margin:3em 0 0}
.footer{color:#aaa;font-size:.78em;margin-top:3em;
        border-top:1px solid #eee;padding-top:12px}
"""
    total_pts = sum(len(r["pv_list"]) for r in results_list)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {ts_str}&nbsp;&nbsp;|&nbsp;&nbsp;\
{len(results_list)} parameters&nbsp;&nbsp;|&nbsp;&nbsp;\
{total_pts} total sweep points&nbsp;&nbsp;|&nbsp;&nbsp;\
{n_sim:,} simulations per point</p>

<h2>Base Configuration</h2>
<table><tbody>{cfg_html}</tbody></table>

<h2>Summary — Best Configuration per Parameter</h2>
{summary_html}

{sections}

<p class="footer">CRM Simulator — Design Exploration batch report (all parameters)</p>
</body></html>"""
    return html


def _plot_sweep_results(df, param_label, param_name="", param_info=None):
    """Render the three-panel sweep results chart (dark theme for app UI)."""
    param_info = param_info or {}
    df = df.sort_values("param_raw", ascending=True).reset_index(drop=True)
    fig, axes  = plt.subplots(1, 3, figsize=(13, 3.6))
    x          = np.arange(len(df))

    if param_name == "max_n":
        xtick_labels = [str(v) for v in df["n_patients"].tolist()]
        xlabel = "Maximum total patients in trial (max N)"
    elif param_name == "cohort_size":
        n_fixed = int(df["n_patients"].iloc[0])
        xtick_labels = df["param_label"].tolist()
        xlabel = f"Cohort size — patients per dose decision  (max N = {n_fixed})"
    else:
        n_fixed = int(df["n_patients"].iloc[0])
        xtick_labels = df["param_label"].tolist()
        xlabel = f"{param_label}  (N = {n_fixed} pts)"

    _apply_dark_fig(fig, *axes)
    specs = [
        ("quality_score",         "Quality score",       "#4a9eff"),
        ("pct_correct_selection", "% Correct selection", "#44dd88"),
        ("overdose_rate",         "Overdose rate (%)",   "#ff6666"),
    ]
    for ax, (col, ylabel, color) in zip(axes, specs):
        ax.bar(x, df[col].values, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels,
                           rotation=35 if len(x) > 6 else 0,
                           ha="right", fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", lw=0.5, alpha=0.3, color=_DARK_GRD)
    fig.tight_layout(pad=1.5)
    return fig


def _plot_sweep_results_light(df, param_label, param_name="", param_info=None):
    """Light-themed version of _plot_sweep_results for embedded HTML reports."""
    param_info = param_info or {}
    df = df.sort_values("param_raw", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(len(df))

    if param_name == "max_n":
        xtick_labels = [str(v) for v in df["n_patients"].tolist()]
        xlabel = "Maximum total patients in trial (max N)"
    elif param_name == "cohort_size":
        n_fixed = int(df["n_patients"].iloc[0])
        xtick_labels = df["param_label"].tolist()
        xlabel = f"Cohort size — patients per dose decision  (max N = {n_fixed})"
    else:
        n_fixed = int(df["n_patients"].iloc[0])
        xtick_labels = df["param_label"].tolist()
        xlabel = f"{param_label}  (N = {n_fixed} pts)"

    fig.patch.set_facecolor("white")
    specs = [
        ("quality_score",         "Quality score",       "#2563eb"),
        ("pct_correct_selection", "% Correct selection", "#16a34a"),
        ("overdose_rate",         "Overdose rate (%)",   "#dc2626"),
    ]
    for ax, (col, ylabel, color) in zip(axes, specs):
        ax.set_facecolor("white")
        ax.bar(x, df[col].values, color=color, alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels,
                           rotation=35 if len(x) > 6 else 0,
                           ha="right", fontsize=8, color="#333333")
        ax.set_xlabel(xlabel, fontsize=9, color="#444444")
        ax.set_title(ylabel, fontsize=10, fontweight="bold", color="#111111")
        ax.tick_params(colors="#333333", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_edgecolor("#cccccc")
        ax.spines["bottom"].set_edgecolor("#cccccc")
        ax.grid(axis="y", lw=0.5, alpha=0.6, color="#e0e0e0")
        ax.yaxis.label.set_color("#444444")
    fig.tight_layout(pad=1.5)
    return fig


def _plot_prior_mtd_context(true_tox, pv_list, tox_label, title,
                            prior_target, prior_halfwidth,
                            model="empiric", intcpt=3.0, light=False):
    """Compact chart: true toxicity bars + one CRM skeleton line per prior MTD level."""
    true_tox = list(true_tox)
    n_levels = len(true_tox)
    x        = np.arange(n_levels)
    x_labels = [f"L{i + 1}" for i in range(n_levels)]

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=130)

    if light:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        bar_color   = "#2563eb"
        grid_color  = "#e0e0e0"
        spine_color = "#cccccc"
        ax.tick_params(colors="#333333", labelsize=9)
        ax.xaxis.label.set_color("#444444")
        ax.yaxis.label.set_color("#444444")
        ax.title.set_color("#111111")
        for sp in ax.spines.values():
            sp.set_edgecolor(spine_color)
        legend_kw = dict(fontsize=8, framealpha=0.90,
                         facecolor="white", edgecolor="#cccccc")
    else:
        _apply_dark_fig(fig, ax)
        bar_color  = "#4a9eff"
        grid_color = _DARK_GRD
        legend_kw  = dict(fontsize=8, framealpha=0.80,
                          facecolor=_DARK_AX, edgecolor=_DARK_GRD,
                          labelcolor=_DARK_FG)

    ax.bar(x, true_tox, color=bar_color, alpha=0.55, label="True toxicity",
           zorder=2)

    for lv in sorted(set(pv_list)):
        idx = lv - 1
        if 0 <= idx < n_levels:
            try:
                sk = dfcrm_getprior(prior_halfwidth, prior_target, lv,
                                    n_levels, model=model, intcpt=intcpt)
                color = _MTD_LINE_COLORS[idx % len(_MTD_LINE_COLORS)]
                ax.plot(x, sk, color=color, linewidth=1.8,
                        marker="o", markersize=3.5,
                        label=f"Skeleton L{lv}", zorder=3)
            except Exception:
                pass

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("Dose level", fontsize=9)
    ax.set_ylabel(f"True {tox_label} prob", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.5, color=grid_color)
    ax.legend(**legend_kw)
    fig.tight_layout(pad=1.2)
    return fig


# ==============================================================================
# Chart generators — return base64 data-URI strings for embedding in HTML
# ==============================================================================

def generate_preview_chart(true_t1, skel_t1, target_t1,
                           true_t2, skel_t2, target_t2):
    """Dose-risk preview: two stacked bar charts (tox1 / tox2) with skeleton overlay."""
    true_t1 = list(true_t1); skel_t1 = list(skel_t1)
    true_t2 = list(true_t2); skel_t2 = list(skel_t2)
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(PREVIEW_W_IN, PREVIEW_H_IN),
                                   dpi=PREVIEW_DPI)
    _apply_dark_fig(fig, ax1, ax2)
    x = np.arange(5); bw = 0.38

    ax1.bar(x - bw/2, true_t1, bw, color="#4a9eff", label="True tox1")
    ax1.bar(x + bw/2, skel_t1, bw, color="#4a9eff", alpha=0.5,
            label="Skel tox1", hatch="//")
    ax1.axhline(target_t1, lw=1.2, alpha=0.85, color="#80c0ff", ls="--")
    ax1.set_ylabel("P(tox1)", fontsize=10, color=_DARK_FG)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L{i}" for i in range(5)], fontsize=10)
    _y1 = max(max(true_t1), max(skel_t1), target_t1)
    ax1.set_ylim(0, min(1.0, _y1 * 1.3 + 0.02))
    ax1.legend(fontsize=10, frameon=True, loc="upper left",
               labelcolor=_DARK_FG, facecolor=_DARK_AX, edgecolor=_DARK_GRD)
    compact_style(ax1)

    ax2.bar(x - bw/2, true_t2, bw, color="#ffaa44", label="True tox2")
    ax2.bar(x + bw/2, skel_t2, bw, color="#ffaa44", alpha=0.5,
            label="Skel tox2", hatch="//")
    ax2.axhline(target_t2, lw=1.2, alpha=0.85, color="#ffd080", ls="--")
    ax2.set_ylabel("P(tox2)", fontsize=10, color=_DARK_FG)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{i}" for i in range(5)], fontsize=10)
    _y2 = max(max(true_t2), max(skel_t2), target_t2)
    ax2.set_ylim(0, min(1.0, _y2 * 1.25 + 0.02))
    ax2.legend(fontsize=10, frameon=True, loc="upper left",
               labelcolor=_DARK_FG, facecolor=_DARK_AX, edgecolor=_DARK_GRD)
    compact_style(ax2)

    fig.tight_layout(pad=0.5)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def generate_mtd_chart(p63, pcrm, true_safe):
    """Bar chart: P(select dose as MTD) for 6+3 vs CRM side by side."""
    p63  = np.asarray(p63,  dtype=float)
    pcrm = np.asarray(pcrm, dtype=float)
    fig, ax = plt.subplots(figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
    _apply_dark_fig(fig, ax)
    xx = np.arange(5); w = 0.38
    ax.bar(xx - w/2, p63,  w, color="#4a9eff", label="TITE 6+3")
    ax.bar(xx + w/2, pcrm, w, color="#ffaa44", label="TITE-CRM")
    ax.set_title("P(select dose as MTD)", fontsize=10)
    ax.set_xticks(xx)
    ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
    ax.set_ylabel("Probability", fontsize=9)
    ax.set_ylim(0, max(p63.max(), pcrm.max()) * 1.15 + 1e-6)
    if true_safe is not None:
        ax.axvline(true_safe, lw=1, alpha=0.6, color="#80ff80",
                   label=f"True safe=L{true_safe}")
    compact_style(ax)
    ax.legend(fontsize=8, frameon=False, labelcolor=_DARK_FG)
    fig.tight_layout(pad=0.4)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def generate_patients_chart(res):
    """Two-panel bar chart: avg patients treated and avg surgery patients per dose."""
    fig, (ax_n, ax_s) = plt.subplots(2, 1,
                                      figsize=(RESULT_W_IN, RESULT_H_IN),
                                      dpi=RESULT_DPI)
    _apply_dark_fig(fig, ax_n, ax_s)
    xx = np.arange(5); w = 0.38
    ax_n.bar(xx - w/2, res["avg_n63"],   w, color="#4a9eff", label="6+3")
    ax_n.bar(xx + w/2, res["avg_ncrm"],  w, color="#ffaa44", label="CRM")
    ax_n.set_title("Avg patients treated", fontsize=9)
    ax_n.set_xticks(xx)
    ax_n.set_xticklabels([f"L{i}" for i in range(5)], fontsize=8)
    ax_n.set_ylabel("Patients", fontsize=8)
    compact_style(ax_n)
    ax_n.legend(fontsize=7, frameon=False, labelcolor=_DARK_FG)

    ax_s.bar(xx - w/2, res["avg_nsurg63"],  w, color="#4a9eff", label="6+3")
    ax_s.bar(xx + w/2, res["avg_nsurgcrm"], w, color="#ffaa44", label="CRM")
    ax_s.set_title("Avg surgery patients", fontsize=9)
    ax_s.set_xticks(xx)
    ax_s.set_xticklabels([f"L{i}" for i in range(5)], fontsize=8)
    ax_s.set_ylabel("Patients", fontsize=8)
    compact_style(ax_s)
    ax_s.legend(fontsize=7, frameon=False, labelcolor=_DARK_FG)

    fig.tight_layout(pad=0.4)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def generate_posterior_tracking_chart(crm_trace):
    """Two-panel line chart: posterior mean tox1/tox2 per dose level over cohort steps."""
    decs      = crm_trace.get("decisions", [])
    if not decs:
        return None

    pt_steps  = [d["step"]  for d in decs]
    pt_pm1    = np.array([d["pm1"] for d in decs])
    pt_pm2    = np.array([d["pm2"] for d in decs])
    n_lvls    = pt_pm1.shape[1]

    sigma_pt   = float(crm_trace["sigma"])
    skel_t1_pt = np.asarray(crm_trace["skel_t1"], dtype=float)
    skel_t2_pt = np.asarray(crm_trace["skel_t2"], dtype=float)
    gh_n_pt    = int(crm_trace["gh_n"])
    tgt1       = float(crm_trace["target_t1"])
    tgt2       = float(crm_trace["target_t2"])
    ewoc_alpha = float(crm_trace["ewoc_alpha"])

    n_steps = len(decs)
    lo1 = np.zeros((n_steps, n_lvls)); hi1 = np.zeros((n_steps, n_lvls))
    lo2 = np.zeros((n_steps, n_lvls)); hi2 = np.zeros((n_steps, n_lvls))

    def _weighted_pct(vals, weights, pcts):
        idx = np.argsort(vals)
        sv  = vals[idx]; sw = weights[idx]
        cum = np.cumsum(sw); cum = cum / cum[-1]
        return np.interp(pcts, cum, sv)

    for si, d in enumerate(decs):
        n1a = np.asarray(d["n1"], dtype=float)
        y1a = np.asarray(d["y1"], dtype=float)
        n2a = np.asarray(d["n2"], dtype=float)
        y2a = np.asarray(d["y2"], dtype=float)
        pw1, P1 = posterior_via_gh(sigma_pt, skel_t1_pt, n1a, y1a, gh_n=gh_n_pt)
        pw2, P2 = posterior_via_gh(sigma_pt, skel_t2_pt, n2a, y2a, gh_n=gh_n_pt)
        for lvl in range(n_lvls):
            lo1[si, lvl], hi1[si, lvl] = _weighted_pct(P1[:, lvl], pw1, [0.05, 0.95])
            lo2[si, lvl], hi2[si, lvl] = _weighted_pct(P2[:, lvl], pw2, [0.05, 0.95])

    dose_colors = ["#4a9eff", "#ffaa44", "#ff6677", "#55dd99", "#cc88ff"]

    fig, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(11.0, 3.8), dpi=150)
    _apply_dark_fig(fig, ax_p1, ax_p2)

    for lvl in range(n_lvls):
        ax_p1.plot(pt_steps, pt_pm1[:, lvl], "o-",
                   color=dose_colors[lvl], lw=1.8, ms=4, label=f"L{lvl}")
        ax_p1.fill_between(pt_steps, lo1[:, lvl], hi1[:, lvl],
                           alpha=0.15, color=dose_colors[lvl], linewidth=0)
        ax_p2.plot(pt_steps, pt_pm2[:, lvl], "o-",
                   color=dose_colors[lvl], lw=1.8, ms=4, label=f"L{lvl}")
        ax_p2.fill_between(pt_steps, lo2[:, lvl], hi2[:, lvl],
                           alpha=0.15, color=dose_colors[lvl], linewidth=0)

    ax_p1.axhline(tgt1, lw=1.5, ls="--", color="#80ff80",
                  alpha=0.8, label=f"Target ({tgt1:.2f})")
    ax_p1.axhline(ewoc_alpha, lw=1.5, ls="--", color="#ff9944",
                  alpha=0.8, label=f"EWOC α={ewoc_alpha:.2f}")
    ax_p2.axhline(tgt2, lw=1.5, ls="--", color="#80ff80",
                  alpha=0.8, label=f"Target ({tgt2:.2f})")
    ax_p2.axhline(ewoc_alpha, lw=1.5, ls="--", color="#ff9944",
                  alpha=0.8, label=f"EWOC α={ewoc_alpha:.2f}")

    ax_p1.set_title("Tox1 posterior mean per dose level", fontsize=10)
    ax_p1.set_xlabel("Cohort decision step", fontsize=9)
    ax_p1.set_ylabel("Posterior mean P(tox1)", fontsize=9)
    ax_p1.set_ylim(0, min(1.0, max(float(hi1.max()), tgt1, ewoc_alpha) * 1.18 + 0.05))
    ax_p1.legend(fontsize=8, frameon=False, labelcolor=_DARK_FG, ncol=2)
    compact_style(ax_p1)

    ax_p2.set_title("Tox2 posterior mean per dose level", fontsize=10)
    ax_p2.set_xlabel("Cohort decision step", fontsize=9)
    ax_p2.set_ylabel("Posterior mean P(tox2)", fontsize=9)
    ax_p2.set_ylim(0, min(1.0, max(float(hi2.max()), tgt2, ewoc_alpha) * 1.18 + 0.05))
    ax_p2.legend(fontsize=8, frameon=False, labelcolor=_DARK_FG, ncol=2)
    compact_style(ax_p2)

    fig.tight_layout(pad=0.5)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def generate_study_end_posteriors_chart(crm_trace):
    """Ridge plot: study-end posterior distributions for tox1 and tox2 per dose."""
    decs = crm_trace.get("decisions", [])
    if not decs:
        return None

    sigma_pt   = float(crm_trace["sigma"])
    skel_t1_pt = np.asarray(crm_trace["skel_t1"], dtype=float)
    skel_t2_pt = np.asarray(crm_trace["skel_t2"], dtype=float)
    gh_n_pt    = int(crm_trace["gh_n"])
    tgt1       = float(crm_trace["target_t1"])
    tgt2       = float(crm_trace["target_t2"])
    ewoc_alpha = float(crm_trace["ewoc_alpha"])
    n_lvls     = len(skel_t1_pt)

    n1e, y1e, n2e, y2e = tite_weights(
        crm_trace["patients"],
        float(crm_trace["study_days"]),
        int(crm_trace["tox1_win"]),
        int(crm_trace["tox2_win"]),
        n_lvls,
    )
    pw1f, P1f = posterior_via_gh(sigma_pt, skel_t1_pt, n1e, y1e, gh_n=gh_n_pt)
    pw2f, P2f = posterior_via_gh(sigma_pt, skel_t2_pt, n2e, y2e, gh_n=gh_n_pt)

    od1_final = np.array([float(np.sum(pw1f[P1f[:, d] > tgt1])) for d in range(n_lvls)])
    od2_final = np.array([float(np.sum(pw2f[P2f[:, d] > tgt2])) for d in range(n_lvls)])

    xgrid  = np.linspace(0.0, 1.0, 400)
    od_mask1 = xgrid >= tgt1
    od_mask2 = xgrid >= tgt2

    def _weighted_kde(p_col, weights, xgrid):
        mu    = np.sum(weights * p_col)
        var   = np.sum(weights * (p_col - mu) ** 2)
        std   = np.sqrt(var) if var > 1e-12 else 0.01
        eff_n = 1.0 / max(np.sum(weights ** 2), 1e-30)
        bw    = max(1.06 * std * eff_n ** (-0.2), 0.005)
        diff  = xgrid[:, None] - p_col[None, :]
        kde   = (weights[None, :] * np.exp(-0.5 * (diff / bw) ** 2)).sum(axis=1)
        return kde

    dose_colors = ["#4a9eff", "#ffaa44", "#ff6677", "#55dd99", "#cc88ff"]
    fig2, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
    _apply_dark_fig(fig2, ax_d1, ax_d2)

    x_ann1 = min(0.94, tgt1 + 0.05)
    x_ann2 = min(0.94, tgt2 + 0.05)

    for lvl in range(n_lvls):
        kde1 = _weighted_kde(P1f[:, lvl], pw1f, xgrid)
        kde2 = _weighted_kde(P2f[:, lvl], pw2f, xgrid)
        k1 = 0.85 / max(float(kde1.max()), 1e-12)
        k2 = 0.85 / max(float(kde2.max()), 1e-12)
        y1c = lvl + kde1 * k1
        y2c = lvl + kde2 * k2

        ax_d1.fill_between(xgrid, lvl, y1c,
                           color=dose_colors[lvl], alpha=0.38, linewidth=0)
        ax_d1.plot(xgrid, y1c, color=dose_colors[lvl], lw=1.3)
        ax_d2.fill_between(xgrid, lvl, y2c,
                           color=dose_colors[lvl], alpha=0.38, linewidth=0)
        ax_d2.plot(xgrid, y2c, color=dose_colors[lvl], lw=1.3)

        ax_d1.fill_between(xgrid, lvl, y1c, where=od_mask1,
                           color=(1.0, 0.39, 0.39), alpha=0.45, linewidth=0)
        ax_d2.fill_between(xgrid, lvl, y2c, where=od_mask2,
                           color=(1.0, 0.39, 0.39), alpha=0.45, linewidth=0)

        ax_d1.text(x_ann1, lvl + 0.08, f"OD: {od1_final[lvl] * 100:.1f}%",
                   color="#e0e0e0", fontsize=7, ha="left", va="bottom", fontweight="bold")
        ax_d2.text(x_ann2, lvl + 0.08, f"OD: {od2_final[lvl] * 100:.1f}%",
                   color="#e0e0e0", fontsize=7, ha="left", va="bottom", fontweight="bold")

    for ax, tgt in ((ax_d1, tgt1), (ax_d2, tgt2)):
        ax.axvline(tgt, lw=1.5, ls="--", color="#80ff80",
                   alpha=0.85, label=f"Target ({tgt:.2f})")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.2, n_lvls - 0.1)
        ax.set_yticks(range(n_lvls))
        ax.set_yticklabels([f"L{i}" for i in range(n_lvls)], fontsize=8)
        ax.set_xlabel("P(tox)", fontsize=9)
        ax.set_ylabel("Dose level", fontsize=9)
        ax.legend(fontsize=8, frameon=False, labelcolor=_DARK_FG)
        compact_style(ax)

    ax_d1.set_title("Tox1 — study-end posterior per dose level", fontsize=10)
    ax_d2.set_title("Tox2 — study-end posterior per dose level", fontsize=10)
    fig2.tight_layout(pad=0.5)
    b64 = fig_to_b64(fig2)
    plt.close(fig2)
    return b64


def generate_dose_eligibility_chart(crm_trace):
    """Heatmap grid: dose eligibility (green/amber/red) across cohort decision steps."""
    decs = crm_trace.get("decisions", [])
    if not decs:
        return None

    n_lvls      = len(crm_trace["skel_t1"])
    ewoc_alpha  = float(crm_trace["ewoc_alpha"])
    elig_border = ewoc_alpha - 0.05
    final_mtd   = crm_trace.get("final_mtd")
    n_steps     = len(decs)

    fig3, ax3 = plt.subplots(
        figsize=(max(8.0, n_steps * 0.90 + 1.5), 4.2), dpi=150)
    _apply_dark_fig(fig3, ax3)
    ax3.set_facecolor(_DARK_BG)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    for si, step_d in enumerate(decs):
        od1_row = step_d["od1"]
        od2_row = step_d["od2"]
        curr    = step_d["current_dose"]

        for lvl in range(n_lvls):
            o1 = float(od1_row[lvl])
            o2 = float(od2_row[lvl])
            if o1 > ewoc_alpha or o2 > ewoc_alpha:
                fc = "#7b241c"
            elif o1 >= elig_border or o2 >= elig_border:
                fc = "#7e5109"
            else:
                fc = "#1a6835"

            ax3.add_patch(mpatches.Rectangle(
                (si, lvl), 1, 1,
                facecolor=fc, edgecolor="#2c2c4c", linewidth=0.8, zorder=1))
            ax3.text(si + 0.5, lvl + 0.64, f"{o1:.2f}",
                     color="white",   fontsize=6.5, ha="center", va="center",
                     fontweight="bold", zorder=3)
            ax3.text(si + 0.5, lvl + 0.34, f"{o2:.2f}",
                     color="#cccccc", fontsize=6.5, ha="center", va="center",
                     fontweight="bold", zorder=3)

        ax3.add_patch(mpatches.Rectangle(
            (si, curr), 1, 1,
            fill=False, edgecolor="#b8b8b8", linewidth=2.0, zorder=4))

    if final_mtd is not None:
        ax3.add_patch(mpatches.Rectangle(
            (n_steps - 1, final_mtd), 1, 1,
            fill=False, edgecolor="#ffd700", linewidth=3.5, zorder=5))

    ax3.set_xlim(0, n_steps)
    ax3.set_ylim(0, n_lvls)
    ax3.set_xticks([s + 0.5 for s in range(n_steps)])
    ax3.set_xticklabels([str(d["step"]) for d in decs], fontsize=8)
    ax3.set_yticks([i + 0.5 for i in range(n_lvls)])
    ax3.set_yticklabels([f"L{i}" for i in range(n_lvls)], fontsize=9)
    ax3.set_xlabel("Cohort decision step", fontsize=9)
    ax3.set_ylabel("Dose level", fontsize=9)
    ax3.set_title("Dose eligibility trajectory across trial", fontsize=10)
    ax3.tick_params(length=0)

    elig_handles = [
        mpatches.Patch(facecolor="#1a6835", edgecolor="#555", linewidth=0.8,
                       label="Allowed — both OD < α"),
        mpatches.Patch(facecolor="#7e5109", edgecolor="#555", linewidth=0.8,
                       label=f"Borderline — either OD ∈ [{elig_border:.2f}, α]"),
        mpatches.Patch(facecolor="#7b241c", edgecolor="#555", linewidth=0.8,
                       label=f"Excluded — either OD > α ({ewoc_alpha:.2f})"),
        mpatches.Patch(fill=False, edgecolor="#b8b8b8", linewidth=2.0,
                       label="Active dose this cohort"),
        mpatches.Patch(fill=False, edgecolor="#ffd700", linewidth=3.0,
                       label="Final MTD selected"),
    ]
    ax3.legend(handles=elig_handles, fontsize=7, frameon=True, ncol=3,
               loc="upper left", bbox_to_anchor=(0.0, -0.20),
               labelcolor=_DARK_FG, facecolor=_DARK_AX, edgecolor=_DARK_GRD)

    fig3.tight_layout(rect=[0, 0.17, 1, 1], pad=0.4)
    b64 = fig_to_b64(fig3)
    plt.close(fig3)
    return b64
