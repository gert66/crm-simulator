"""
sim.py — TITE dual-endpoint dose-escalation simulator
===========================================================
Tox1 = acute toxicity   (window starts at RT start; all treated patients)
Tox2 = subacute toxicity (window starts at surgery; surgery patients only)

Calendar-time model
-------------------
Patients arrive via a Poisson process (exponential inter-arrivals at rate
accrual_per_month / 30 per day).  Each patient has a personal timeline:

  arrival
    └─► [incl_to_rt days] ──► RT start
                                ├─► tox1 window (rt_dur + rt_to_surg days → ends at surgery)
                                │     Event time ~ Uniform(0, tox1_win) if tox1 occurs
                                └─► [rt_dur] ──► RT end
                                                  └─► [rt_to_surg] ──► surgery (if surgery)
                                                                          └─► tox2 window (tox2_win days)
                                                                                Event time ~ Uniform(0, tox2_win)

TITE-CRM (partial follow-up weighting)
---------------------------------------
CRM decisions are made after each cohort finishes enrolling (no waiting).
Fractional weight for each patient at decision time t:

  tox1 weight:  0               if t < rt_start
                1               if event observed OR t >= tox1_win_end
                (t-rt_start) / tox1_win   otherwise

  tox2 weight (surgery patients only):
                0               if t < surgery_day
                1               if event observed OR t >= tox2_win_end
                (t-surgery_day) / tox2_win  otherwise

Both CRM models use the same GH-quadrature posterior; fractional n is valid
because the likelihood decomposes by patient.

Modified 6+3 with lower-dose bridging (full evaluability required)
-------------------------------------------------------------------
Decisions require ALL enrolled patients in the evaluation cohort to have
completed their full relevant follow-up.  While waiting:
  - new arrivals go to safe_dose = max(0, eval_dose - 1)
  - these bridging patients count toward max_n but NOT toward the
    formal evaluation cohort at eval_dose
Max sample size = total enrolled patients (eval + bridging).
Rate-based acute thresholds (from sim_sur.py) preserved.
"""

import base64
import datetime
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as _stcv1
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
# Helpers
# ==============================================================================

def safe_probs(x):
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)

_DARK_BG  = "#1a1a2e"
_DARK_AX  = "#16213e"
_DARK_FG  = "#e0e0e0"
_DARK_GRD = "#2a2a4a"

# ColorBrewer Set1 — 5 clearly distinguishable colours for per-level MTD lines
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

def crm_stopping_prob(sigma, skel1, n1, y1, target1, rec_dose, gh_n=61):
    """
    Posterior probability that rec_dose minimises |P(tox1|d) - target1| over
    all doses — i.e. P(rec_dose is the optimal MTD | current data).

    Computed by integrating over the Gauss-Hermite quadrature posterior: for
    each quadrature point the dose closest to target1 is identified, and the
    weights for points where that dose equals rec_dose are summed.
    """
    post_w, P = posterior_via_gh(sigma, skel1, n1, y1, gh_n=gh_n)
    dist = np.abs(P - float(target1))      # shape (n_quad, n_levels)
    best = np.argmin(dist, axis=1)         # shape (n_quad,): best dose per point
    return float(np.sum(post_w[best == int(rec_dose)]))

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
        # EWOC OFF: all doses are candidates (no overdose-probability filter)
        candidates = np.arange(n_levels)
    else:
        # EWOC ON: jointly admissible doses only
        candidates = np.where((od1 < float(ewoc_alpha)) & (od2 < float(ewoc_alpha)))[0]

    if candidates.size == 0:
        candidates = np.array([0], dtype=int)

    if ewoc_alpha is not None:
        # EWOC ON: highest admissible dose
        k = int(candidates.max())
    else:
        # EWOC OFF: closest to tox1 target by posterior mean (standard CRM rule)
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
        # EWOC OFF: all doses are candidates (no overdose-probability filter)
        candidates = np.arange(n_levels)
    else:
        # EWOC ON: jointly admissible doses only
        candidates = np.where((od1 < float(ewoc_alpha)) & (od2 < float(ewoc_alpha)))[0]

    if candidates.size == 0:
        return 0

    if restrict_to_tried:
        tried = np.where(np.asarray(n1) > 0)[0]
        if tried.size > 0:
            candidates2 = np.intersect1d(candidates, tried)
            candidates = candidates2 if candidates2.size > 0 else tried

    if ewoc_alpha is not None:
        # EWOC ON: highest admissible (and tried) dose
        return int(candidates.max())
    else:
        # EWOC OFF: closest to tox1 target by posterior mean (standard CRM rule)
        dist = np.abs(pm1[candidates] - float(target1))
        return int(candidates[int(np.argmin(dist))])


def crm_mtd_posterior_probs(sigma, skel1, skel2,
                             n1, y1, n2, y2,
                             target1, target2,
                             ewoc_alpha=None, gh_n=61,
                             restrict_to_tried=True):
    """
    Compute the posterior probability that each dose level would be selected
    as the final MTD, integrating over the joint tox1/tox2 posterior.

    Uses the same dual-endpoint selection rule as crm_select_mtd:
    - EWOC ON  (ewoc_alpha is not None): highest jointly admissible dose.
    - EWOC OFF (ewoc_alpha is None):     dose whose posterior mean P(tox1) is
                                         closest to target1.

    For each Gauss-Hermite quadrature point:
      1. Compute P(tox1) and P(tox2) at all dose levels.
      2. Apply joint safety filter (if ewoc_alpha is not None).
      3. Apply restrict_to_tried mask (same as crm_select_mtd).
      4. Identify the selected dose under that quadrature point.
      5. Sum posterior weights by selected dose.

    Returns
    -------
    probs : np.ndarray, shape (n_levels,)
        Posterior probability that each dose level is selected as the MTD.
        Sums to 1.0 (approximately, to floating-point precision).
    """
    # Use the tox1 posterior only (same marginals as crm_select_mtd) for speed.
    # The dual-endpoint EWOC filter is approximated using point-wise OD probs:
    # a dose is "point-admissible" if P1[i,d] <= target1 AND P2[j,d] <= target2.
    # This is consistent with the EWOC posterior-OD logic in crm_select_mtd.
    post_w1, P1 = posterior_via_gh(sigma, skel1, n1, y1, gh_n=gh_n)
    post_w2, P2 = posterior_via_gh(sigma, skel2, n2, y2, gh_n=gh_n)
    n_levels = len(skel1)

    # tried-dose mask
    tried_mask = np.ones(n_levels, dtype=bool)
    if restrict_to_tried:
        tried = np.where(np.asarray(n1) > 0)[0]
        if tried.size > 0:
            tried_mask[:] = False
            tried_mask[tried] = True

    # Outer-product weight matrix: shape (n_q1, n_q2)
    W = post_w1[:, None] * post_w2[None, :]   # (n_q1, n_q2)

    # For each (i, j) quad point: determine selected dose.
    # P1: (n_q1, n_levels), P2: (n_q2, n_levels)
    # Vectorised: compute per (i,j) admissibility and selection.
    # Shape of P1 broadcast: (n_q1, 1, n_levels)
    #                  P2  : (1, n_q2, n_levels)
    P1_3d = P1[:, None, :]          # (n_q1, 1,    n_levels)
    P2_3d = P2[None, :, :]          # (1,    n_q2, n_levels)

    if ewoc_alpha is not None:
        # Admissible = point-wise tox below target (conservative per-point OD proxy)
        admissible = (P1_3d <= float(target1)) & (P2_3d <= float(target2))
        # shape: (n_q1, n_q2, n_levels)
    else:
        admissible = np.ones((len(post_w1), len(post_w2), n_levels), dtype=bool)

    # Apply tried restriction
    admissible &= tried_mask[None, None, :]

    # Fallback: if no admissible dose for a grid point, allow all tried doses
    any_admissible = admissible.any(axis=2, keepdims=True)  # (n_q1, n_q2, 1)
    fallback = tried_mask[None, None, :]
    admissible = np.where(any_admissible, admissible, fallback)

    # Final fallback: if still nothing tried, admit dose 0
    any_adm2 = admissible.any(axis=2, keepdims=True)
    dose0 = np.zeros((1, 1, n_levels), dtype=bool)
    dose0[0, 0, 0] = True
    admissible = np.where(any_adm2, admissible, dose0)

    if ewoc_alpha is not None:
        # Highest admissible dose
        # Replace non-admissible with -1, then argmax
        dose_idx = np.arange(n_levels, dtype=float)[None, None, :]
        dose_idx_masked = np.where(admissible, dose_idx, -1.0)
        selected_grid = dose_idx_masked.max(axis=2).astype(int)  # (n_q1, n_q2)
    else:
        # Closest posterior-mean tox1 to target1
        dist = np.abs(P1_3d - float(target1))      # (n_q1, 1, n_levels)
        dist_masked = np.where(admissible, dist, np.inf)
        selected_grid = dist_masked.argmin(axis=2).astype(int)   # (n_q1, n_q2)

    # Accumulate weighted probability per selected dose
    probs = np.zeros(n_levels, dtype=float)
    for d in range(n_levels):
        probs[d] = float(W[selected_grid == d].sum())

    total = probs.sum()
    if total > 0:
        probs /= total
    return probs

# ==============================================================================
# TITE-CRM trial runner
# ==============================================================================

# EWOC application modes — where the EWOC joint overdose filter is applied.
EWOC_APP_BOTH  = "Dose assignment + final MTD"   # current/default behaviour
EWOC_APP_FINAL = "Final MTD only"
EWOC_APP_OFF   = "Off"
EWOC_APP_OPTIONS = [EWOC_APP_BOTH, EWOC_APP_FINAL, EWOC_APP_OFF]


def ewoc_effective_alphas(ewoc_application, ewoc_alpha, ewoc_on=True):
    """Map an EWOC application mode to (ewoc_decision_eff, ewoc_final_eff).

    ewoc_decision_eff — alpha passed to crm_choose_next() during the trial
                        (None = no EWOC filter for dose assignment)
    ewoc_final_eff    — alpha passed to crm_select_mtd() at study end
                        (None = no EWOC filter for final MTD selection)

    Backward compatibility: ewoc_on=False (the legacy boolean) forces both
    to None regardless of ewoc_application, so older callers that only pass
    ewoc_on/ewoc_alpha keep their previous behaviour.
    """
    app = str(ewoc_application)
    if not ewoc_on or app == EWOC_APP_OFF:
        return None, None
    if app == EWOC_APP_FINAL:
        return None, float(ewoc_alpha)
    # EWOC_APP_BOTH (and any unrecognised value defaults to current behaviour)
    return float(ewoc_alpha), float(ewoc_alpha)


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
    ewoc_application=EWOC_APP_BOTH,
    burn_in=True, rng=None,
    collect_trace=False,
    n_safe_d1=0,
    p_stop=1.0,
    require_full_tox1_fu_before_escalation=True,
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
      cohort update (posteriors, weights, allowed doses, decision reason).
      Adds negligible runtime; used only for the first simulated trial.

    ewoc_application: where the EWOC joint overdose filter is applied.
      - "Dose assignment + final MTD" (default): EWOC used both by
        crm_choose_next() during the trial and by crm_select_mtd() at the end
        (preserves the historical ewoc_on=True behaviour).
      - "Final MTD only": dose assignment runs WITHOUT EWOC; only the final
        MTD selection applies the EWOC filter.
      - "Off": EWOC is never applied.
      ewoc_on=False (legacy) forces "Off" regardless of ewoc_application.

    n_safe_d1: number of patients already safely treated at L1 with complete follow-up
      before the trial opens (day 0).  Their records are pre-loaded into the patient
      list so the CRM sees them as fully-weighted observations at L1 (dose index 1)
      with no DLTs.  Surgery status for these patients is drawn from p_surgery.
      These patients count toward max_n — if you want max_n new patients in addition
      to the pre-treated cohort, increase max_n by n_safe_d1.

    require_full_tox1_fu_before_escalation: when True, burn-in escalation from
      the current dose Lx to Lx+1 is only allowed if at least cohort_size patients
      treated at Lx have completed full acute tox1 follow-up (tox1_win_end defined
      as pt["rt_start"] + tox1_win) with no observed tox1 DLTs.  The check is
      evaluated at the next patient's RT start day (arrival + incl_to_rt), not at
      inclusion.  Has no effect when burn_in=False or after burn-in has ended.

    p_stop: early-stopping threshold (0 < p_stop <= 1).  After each CRM cohort
      decision (burn-in excluded), the posterior probability that the recommended
      dose minimises |P(tox1|d) - target1| over all doses is computed via
      crm_stopping_prob.  If this probability >= p_stop the trial stops early and
      the current recommended dose is declared the MTD.  Set p_stop=1.0 (default)
      to disable early stopping — a probability of exactly 1.0 is unreachable.

    Returns (selected_level, patients_list, study_days, trace, stopped_early).
      stopped_early is True when the p_stop rule fired before max_n was reached.
      trace is a list of dicts (one per cohort decision) when collect_trace=True,
      otherwise an empty list.  Each trace dict includes 'p_stop_prob' — the
      stopping probability computed at that step (0.0 during burn-in).
    """
    if rng is None:
        rng = np.random.default_rng()
    true_t1  = np.asarray(true_t1, dtype=float)
    true_t2  = np.asarray(true_t2, dtype=float)
    n_levels = len(true_t1)
    rate_per_day = float(accrual_per_month) / MONTH

    level         = int(np.clip(int(start_level), 0, n_levels - 1))
    patients      = []
    highest_tried = -1
    current_day   = 0.0
    burn_active   = bool(burn_in)
    ewoc_decision_eff, ewoc_final_eff = ewoc_effective_alphas(
        ewoc_application, ewoc_alpha, ewoc_on=ewoc_on)
    trace         = []
    cohort_step   = 0
    stopped_early = False
    _p_stop       = float(p_stop)

    # Pre-populate with historically safe patients at L1 (dose index 1).
    # Arrivals are placed far enough before day 0 that every follow-up window —
    # including the tox2 window for surgery patients — is already closed.
    if n_safe_d1 > 0:
        _total_fu = (float(incl_to_rt) + float(rt_dur)
                     + float(rt_to_surg) + float(tox2_win))
        for _i in range(int(n_safe_d1)):
            _arr      = -(_total_fu + 1.0 + _i)
            _rt_start = _arr + float(incl_to_rt)
            _rt_end   = _rt_start + float(rt_dur)
            _t1w_end  = _rt_start + float(tox1_win)
            _has_surg = bool(rng.random() < float(p_surgery))
            _surg_day = float(_rt_end + float(rt_to_surg)) if _has_surg else None
            _t2w_end  = float(_surg_day + float(tox2_win)) if _has_surg else None
            patients.append({
                "dose":         1,
                "arrival":      float(_arr),
                "rt_start":     _rt_start,
                "tox1_win_end": _t1w_end,
                "has_tox1":     False,
                "tox1_day":     None,
                "has_surgery":  _has_surg,
                "surgery_day":  _surg_day,
                "tox2_win_end": _t2w_end,
                "has_tox2":     False,
                "tox2_day":     None,
                "is_bridging":  False,
            })
        highest_tried = 1

    _req_fu = bool(require_full_tox1_fu_before_escalation)

    while len(patients) < int(max_n):
        n_add        = min(int(cohort_size), int(max_n) - len(patients))
        cohort_start = len(patients)

        # Enroll cohort: each patient arrives after an Exp(1/rate) inter-arrival.
        # When require_full_tox1_fu_before_escalation is ON and burn-in is active,
        # the dose for each new patient is evaluated at their RT start day so that
        # the full-follow-up check can use the correct calendar time.
        for _ in range(n_add):
            current_day += rng.exponential(1.0 / rate_per_day)
            _assign_level = level

            if _req_fu and burn_active:
                # Evaluate the escalation decision at next patient's RT start day,
                # not at inclusion.  tox1 full follow-up means current_day + incl_to_rt
                # >= pt["tox1_win_end"], i.e. rt_start + tox1_win has elapsed.
                _next_rt_start = current_day + float(incl_to_rt)

                # Check whether any tox1 DLT has been observed by RT start
                _obs_dlt_at_rt = any(
                    p["has_tox1"] and p["tox1_day"] is not None
                    and p["tox1_day"] <= _next_rt_start
                    for p in patients
                )
                if _obs_dlt_at_rt:
                    # DLT observed — burn-in will end; stay at current dose
                    _assign_level = level
                else:
                    # Count fully tox1-evaluable patients at current dose with no DLT.
                    # Full evaluability: tox1_win_end (= rt_start + tox1_win) <= check day.
                    _fu_safe = sum(
                        1 for p in patients
                        if p["dose"] == level
                        and float(_next_rt_start) >= p["tox1_win_end"]
                        and not p["has_tox1"]
                    )
                    _can_esc = _fu_safe >= int(cohort_size)
                    if _can_esc:
                        _assign_level = min(level + 1, n_levels - 1)
                    else:
                        _assign_level = level
            elif burn_active:
                # Standard burn-in: dose assigned at inclusion, escalation checked
                # after the cohort (handled below in the post-cohort block).
                _assign_level = level

            pt = make_patient(rng, _assign_level, current_day,
                              true_t1, p_surgery, true_t2,
                              incl_to_rt, rt_dur, rt_to_surg, tox1_win, tox2_win)
            patients.append(pt)
            # When the strict FU mode escalates mid-cohort, carry the new level forward
            if _req_fu and burn_active:
                level = _assign_level

        # Decision time = calendar day when last patient in cohort arrived
        decision_day  = current_day
        highest_tried = max(highest_tried, level)

        # Compute fractional TITE weights for all enrolled patients
        n1, y1, n2, y2 = tite_weights(
            patients, decision_day, tox1_win, tox2_win, n_levels)

        burn_was_active = burn_active

        # Burn-in: check if any tox1 event has been observed by decision day
        if burn_active:
            obs_any_dlt = any(
                p["has_tox1"] and p["tox1_day"] is not None
                and p["tox1_day"] <= decision_day
                for p in patients
            )
            if obs_any_dlt:
                burn_active = False

        if burn_active:
            if _req_fu:
                # Strict FU mode: escalation was already handled per-patient above.
                # Post-cohort: keep level as-is (already updated inside the loop).
                next_level = level
            else:
                # Standard burn-in: escalate one level
                next_level = min(level + 1, n_levels - 1)
            if next_level == n_levels - 1:
                burn_active = False   # reached top, switch to CRM next round
        else:
            next_level = crm_choose_next(
                sigma, skel1, skel2,
                n1, y1, n2, y2,
                level, target1, target2,
                ewoc_alpha=ewoc_decision_eff, max_step=max_step, gh_n=gh_n,
                enforce_guardrail=enforce_guardrail,
                highest_tried=highest_tried, n_levels=n_levels,
            )

        # ── Early-stopping check (CRM phase only) ────────────────────────────
        # Compute P(next_level is the optimal MTD | current data) and stop if
        # it meets the threshold.  Skipped during burn-in (posterior not yet
        # guiding dose selection) and when p_stop is effectively disabled (>=1).
        if not burn_was_active and _p_stop < 1.0:
            _stop_prob = crm_stopping_prob(
                sigma, skel1, n1, y1, target1, next_level, gh_n=gh_n)
            if _stop_prob >= _p_stop:
                stopped_early = True
        else:
            _stop_prob = 0.0

        # ── Collect trace for this decision (first trial only) ────────────────
        if collect_trace:
            pm1, od1 = crm_posterior_summaries(
                sigma, skel1, n1, y1, target1, gh_n=gh_n)
            pm2, od2 = crm_posterior_summaries(
                sigma, skel2, n2, y2, target2, gh_n=gh_n)

            # EWOC mode label for the trace — describes dose ASSIGNMENT only.
            # In "Final MTD only" mode, make explicit that EWOC was not used
            # for this per-cohort decision (it applies only at final selection).
            if ewoc_decision_eff is not None:
                ewoc_mode = f"ON (α={ewoc_decision_eff:.2f})"
            elif ewoc_final_eff is not None:
                ewoc_mode = "OFF for dose assignment (EWOC at final MTD only)"
            else:
                ewoc_mode = "OFF"

            # Which doses pass the joint EWOC safety filter (dose assignment)?
            if ewoc_decision_eff is None:
                # EWOC OFF for dose assignment: all doses are candidates
                allowed_arr = list(range(n_levels))
            else:
                allowed_arr = [int(d) for d in
                               np.where((od1 < ewoc_decision_eff)
                                        & (od2 < ewoc_decision_eff))[0]]

            # Human-readable reason for the dose selected
            if burn_was_active:
                if _req_fu:
                    # Count evaluable patients at previous level for trace annotation
                    _fu_cnt = sum(
                        1 for p in patients[:cohort_start]
                        if p["dose"] == (next_level - 1) and not p["has_tox1"]
                        and float(decision_day) >= p["tox1_win_end"]
                    )
                    if next_level > level or (next_level == level and cohort_start > 0
                                              and patients[cohort_start - 1]["dose"] == level):
                        # Escalation happened (or was attempted)
                        _prev = next_level - 1 if next_level > 0 else 0
                        if _fu_cnt >= int(cohort_size):
                            reason = (
                                f"Burn-in escalation allowed (strict FU mode): "
                                f"{_fu_cnt} patients at L{_prev} have complete "
                                f"tox1 follow-up with 0 tox1 DLTs → L{next_level}. "
                                f"Dose decision evaluated at RT start."
                            )
                        else:
                            reason = (
                                f"Burn-in escalation blocked (strict FU mode): "
                                f"fewer than {int(cohort_size)} patients at L{_prev} "
                                f"have complete tox1 follow-up — staying at L{next_level}. "
                                f"Dose decision evaluated at RT start."
                            )
                    else:
                        reason = (
                            f"Burn-in: no tox1 DLT observed yet → L{next_level} "
                            f"(strict FU mode; dose decision evaluated at RT start)"
                        )
                else:
                    reason = (f"Burn-in: escalate one level (no tox1 DLT "
                              f"observed yet → L{next_level})")
            elif not allowed_arr:
                reason = "No dose within joint safety bounds → fallback to L0"
            elif ewoc_decision_eff is None:
                # EWOC OFF for dose assignment: closest-to-target1 rule
                cands     = np.arange(n_levels)
                dist      = np.abs(pm1[cands] - float(target1))
                k_target  = int(cands[int(np.argmin(dist))])
                k_step    = int(np.clip(k_target,
                                        level - int(max_step),
                                        level + int(max_step)))
                k_guard   = (int(min(k_step, highest_tried + 1))
                             if enforce_guardrail and highest_tried >= 0
                             else k_step)
                _ewoc_off_lbl = ("EWOC OFF for dose assignment"
                                 if ewoc_final_eff is not None else "EWOC OFF")
                parts = [f"{_ewoc_off_lbl} → argmin|pm1−target1| = L{k_target}"]
                if k_step != k_target:
                    parts.append(f"step-limit → L{k_step}")
                if k_guard != k_step:
                    parts.append(f"guardrail → L{k_guard}")
                reason = f"L{next_level}: " + "; ".join(parts)
            else:
                # EWOC ON: highest jointly admissible dose
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
                "p_stop_prob":   round(_stop_prob, 4),
                "stopped_early": stopped_early,
            })

        cohort_step += 1
        level = next_level

        if stopped_early:
            break

    # Final MTD selection using full follow-up weights
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
        ewoc_alpha=ewoc_final_eff, gh_n=gh_n,
        restrict_to_tried=restrict_final_to_tried,
    )
    return int(selected), patients, float(study_days), trace, stopped_early

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

    Evaluability rules (full follow-up required):
      - tox1 evaluable  : current_day >= tox1_win_end
      - fully evaluable : tox1 evaluable AND (if surgery) current_day >= tox2_win_end

    Enrollment states per dose:
      ENROLL: add patients to eval_cohort at eval_dose until HOLD criteria met
              (n_treated >= req AND n_surgery >= req; same HOLD logic as sim_sur.py)
      WAIT  : once HOLD criteria met, new arrivals → safe_dose (bridging)
              Wait until all eval_cohort patients are fully evaluable.
      DECIDE: apply phase 1 (or phase 2) decision rules.

    Rate-based acute thresholds preserved from sim_sur.py:
      a6_esc_adj  = floor(nt * a6_esc_max  / 6);  a6_stop_adj = ceil(nt * a6_stop_min / 6)
      a9_esc_adj  = floor(nt * a9_esc_max  / 9)

    After escalation, loop restarts with new eval_dose.
    After stop (or at top level), loop exits.

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
        """Advance calendar by one Poisson inter-arrival; return new current_day."""
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

    # ── Outer loop: one iteration per dose level evaluated ────────────────────
    while len(all_patients) < int(max_n):
        eval_cohort = []
        safe_dose   = max(0, eval_dose - 1)

        # ── ENROLL: add patients at eval_dose until n_treated>=6, n_surgery>=6 ─
        while len(all_patients) < int(max_n):
            pt = _enroll(eval_dose, bridging=False)
            eval_cohort.append(pt)
            n_t = len(eval_cohort)
            n_s = sum(1 for p in eval_cohort if p["has_surgery"])
            if n_t >= 6 and n_s >= 6:
                break

        # ── WAIT (phase 1): bridging until all eval_cohort fully evaluable ────
        while (len(all_patients) < int(max_n) and
               not all(_fully_evaluable(p, current_day) for p in eval_cohort)):
            _enroll(safe_dose, bridging=True)

        # If max_n hit before full evaluability, advance virtual clock to when
        # the last eval patient completes follow-up (no new accrual needed).
        if not all(_fully_evaluable(p, current_day) for p in eval_cohort):
            current_day = max(patient_follow_up_end(p) for p in eval_cohort)
        study_days = max(study_days, current_day)

        # ── DECIDE phase 1 ────────────────────────────────────────────────────
        nt   = len(eval_cohort)
        nsg  = sum(1 for p in eval_cohort if p["has_surgery"])
        ya   = sum(1 for p in eval_cohort if p["has_tox1"])
        ys   = sum(1 for p in eval_cohort if p["has_tox2"])

        sub_eval_p1 = nsg >= 6
        # Rate-based acute thresholds: preserve protocol ratios when nt > 6
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
                continue   # restart outer loop at new dose
            break           # already at top; done

        # ── ENROLL phase 2: grow eval_cohort until n_treated>=9, n_surgery>=9 ─
        while len(all_patients) < int(max_n):
            n_t = len(eval_cohort)
            n_s = sum(1 for p in eval_cohort if p["has_surgery"])
            if n_t >= 9 and n_s >= 9:
                break
            pt = _enroll(eval_dose, bridging=False)
            eval_cohort.append(pt)

        # ── WAIT (phase 2): bridging until all eval_cohort fully evaluable ────
        while (len(all_patients) < int(max_n) and
               not all(_fully_evaluable(p, current_day) for p in eval_cohort)):
            _enroll(safe_dose, bridging=True)

        if not all(_fully_evaluable(p, current_day) for p in eval_cohort):
            current_day = max(patient_follow_up_end(p) for p in eval_cohort)
        study_days = max(study_days, current_day)

        # ── DECIDE phase 2 ────────────────────────────────────────────────────
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
                continue   # restart outer loop at new dose
            break           # at top; done

        # stop_p2 or default
        if eval_dose > 0:
            eval_dose -= 1
        last_acceptable = eval_dose
        break

    selected   = 0 if last_acceptable is None else int(last_acceptable)
    n_bridging = sum(1 for p in all_patients if p["is_bridging"])
    return int(selected), all_patients, float(study_days), int(n_bridging)

# ==============================================================================
# Streamlit config + CSS
# ==============================================================================

st.set_page_config(
    page_title="TITE dual-endpoint simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* ── Force dark theme ── */
  html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
  [data-testid="stHeader"], [data-testid="stToolbar"],
  .main, .block-container, section[data-testid="stSidebar"] {
    background-color: #1a1a2e !important;
    color: #e0e0e0 !important;
  }
  section[data-testid="stSidebar"] { background-color: #16213e !important; }
  section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
  [data-testid="stWidgetLabel"] p,
  .stMarkdown p, .stMarkdown h4, label { color: #e0e0e0 !important; }
  [data-testid="stNumberInput"] input,
  [data-testid="stTextInput"] input,
  [data-testid="stSelectbox"] select,
  div[data-baseweb="select"] { background-color: #0f3460 !important; color: #ffffff !important; }
  /* ── Slider dark styling ── */
  [data-testid="stSlider"] { color: #e0e0e0 !important; }
  [data-testid="stSlider"] p { color: #e0e0e0 !important; }
  /* Track background */
  [data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"] {
    background-color: #4a9eff !important;
  }
  [data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
    background-color: #2a2a4a !important;
  }
  /* Thumb/handle */
  [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #4a9eff !important;
    border-color: #80c0ff !important;
  }
  /* Value label above thumb */
  [data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"],
  [data-testid="stSlider"] [data-baseweb="slider"] div[data-testid] {
    background-color: #0f3460 !important;
    color: #e0e0e0 !important;
    border-color: #4a9eff !important;
  }
  /* Min/max tick labels */
  [data-testid="stSlider"] [data-baseweb="slider"] ul li {
    color: #a0a0c0 !important;
  }
  [data-testid="stMetric"]       { background-color: #16213e !important; border-radius: 6px; padding: 0.4rem !important; }
  [data-testid="stMetricLabel"]  { color: #a0a0c0 !important; }
  [data-testid="stMetricValue"]  { color: #e0e0e0 !important; }

  /* ── Layout tweaks ── */
  .block-container { padding-top: 2.6rem; padding-bottom: 0.5rem; }
  .element-container { margin-bottom: 0.12rem; }

  [data-testid="stMetric"]           { padding: 0.15rem 0 0.05rem 0 !important; }
  [data-testid="metric-container"]   { gap: 0 !important; }
  [data-testid="stMetricLabel"]      { font-size: 0.78rem !important; }
  [data-testid="stMetricValue"]      { font-size: 1.05rem !important; line-height: 1.2 !important; }

  [data-testid="stImage"] img {
    max-width: none !important;
    width: auto  !important;
    height: auto !important;
  }

  /* ── Primary action buttons — red ── */
  [data-testid="stButton"] button[data-testid="baseButton-primary"] {
    background-color: #b02a2a !important;
    border-color:     #b02a2a !important;
    color: #ffffff !important;
  }
  [data-testid="stButton"] button[data-testid="baseButton-primary"]:hover {
    background-color: #d93a3a !important;
    border-color:     #d93a3a !important;
  }
  [data-testid="stButton"] button[data-testid="baseButton-primary"]:disabled {
    background-color: #6b1f1f !important;
    border-color:     #6b1f1f !important;
    opacity: 0.55 !important;
  }

  /* ── Radio button labels ── */
  [data-testid="stRadio"] label,
  [data-testid="stRadio"] label span,
  [data-testid="stRadio"] div[data-baseweb="radio"] ~ div,
  [data-testid="stRadio"] [class*="st-"] { color: #e0e0e0 !important; }

  /* ── Streamlit toolbar / header visibility ── */
  [data-testid="stToolbar"] button,
  [data-testid="stToolbar"] button svg { color: #b0bcd0 !important; fill: #b0bcd0 !important; }
  [data-testid="stToolbar"] button:hover,
  [data-testid="stToolbar"] button:hover svg { color: #e0e8f8 !important; fill: #e0e8f8 !important; }
  [data-testid="stStatusWidget"],
  [data-testid="stStatusWidget"] * { color: #b0bcd0 !important; }
  [data-testid="stStatusWidget"] svg { fill: #4a9eff !important; }

  /* ── Spinner visibility ── */
  [data-testid="stSpinner"] p,
  [data-testid="stSpinner"] span { color: #c0c8d8 !important; }
  [data-testid="stSpinner"] svg { stroke: #4a9eff !important; }

  /* ── Help / tooltip icons — outlined style, no fill ── */
  /* Button wrapper */
  [data-testid="stTooltipIcon"],
  [data-testid="stTooltipHoverTarget"],
  .stTooltipIcon,
  button[data-testid="stTooltipHoverTarget"] {
    opacity: 1 !important;
    color: #e5e7eb !important;
    background: transparent !important;
    cursor: pointer !important;
  }
  /* SVG container: no fill, let stroke do the drawing */
  [data-testid="stTooltipIcon"] svg,
  [data-testid="stTooltipHoverTarget"] svg,
  .stTooltipIcon svg,
  button[data-testid="stTooltipHoverTarget"] svg,
  svg[aria-label="Help"],
  svg[aria-label="Info"],
  label span svg {
    fill: none !important;
    stroke: #e5e7eb !important;
    color: #e5e7eb !important;
    opacity: 1 !important;
  }
  /* All SVG child shapes: outlined, not filled */
  [data-testid="stTooltipIcon"] svg *,
  [data-testid="stTooltipHoverTarget"] svg *,
  .stTooltipIcon svg *,
  button[data-testid="stTooltipHoverTarget"] svg *,
  svg[aria-label="Help"] *,
  svg[aria-label="Info"] *,
  label span svg * {
    fill: none !important;
    stroke: #e5e7eb !important;
    color: #e5e7eb !important;
  }
  /* Hover: brighter white, slight scale */
  [data-testid="stTooltipIcon"]:hover,
  [data-testid="stTooltipHoverTarget"]:hover,
  .stTooltipIcon:hover,
  button[data-testid="stTooltipHoverTarget"]:hover {
    color: #ffffff !important;
    cursor: pointer !important;
    transform: scale(1.1);
  }
  [data-testid="stTooltipIcon"]:hover svg,
  [data-testid="stTooltipIcon"]:hover svg *,
  [data-testid="stTooltipHoverTarget"]:hover svg,
  [data-testid="stTooltipHoverTarget"]:hover svg *,
  .stTooltipIcon:hover svg,
  .stTooltipIcon:hover svg *,
  button[data-testid="stTooltipHoverTarget"]:hover svg,
  button[data-testid="stTooltipHoverTarget"]:hover svg * {
    fill: none !important;
    stroke: #ffffff !important;
    color: #ffffff !important;
  }
  /* Tooltip popup box */
  [data-baseweb="tooltip"] {
    background-color: #1e3a5f !important;
    border: 1px solid #4a9eff !important;
    border-radius: 6px !important;
    max-width: 320px !important;
  }
  [data-baseweb="tooltip"] p,
  [data-baseweb="tooltip"] span,
  [data-baseweb="tooltip"] div,
  [data-baseweb="tooltip"] li {
    color: #d0e8ff !important;
    font-size: 0.87rem !important;
    line-height: 1.5 !important;
  }
  /* Also target any Streamlit-internal tooltip wrapper */
  [data-testid="stTooltipContent"] {
    background-color: #1e3a5f !important;
    color: #d0e8ff !important;
    border: 1px solid #4a9eff !important;
    border-radius: 6px !important;
  }

  /* ── Download report button — prominent red ── */
  [data-testid="stDownloadButton"] button {
    background-color: #b02a2a !important;
    border-color:     #b02a2a !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
  }
  [data-testid="stDownloadButton"] button:hover {
    background-color: #d93a3a !important;
    border-color:     #d93a3a !important;
  }

  /* ── Sidebar collapse/expand chevron arrows ── */
  [data-testid="collapsedControl"] {
    color: #c0d0e0 !important;
  }
  [data-testid="collapsedControl"] svg,
  [data-testid="collapsedControl"] svg * {
    color: #c0d0e0 !important;
    fill:  #c0d0e0 !important;
  }
  [data-testid="collapsedControl"]:hover {
    color: #ffffff !important;
  }
  [data-testid="collapsedControl"]:hover svg,
  [data-testid="collapsedControl"]:hover svg * {
    color: #ffffff !important;
    fill:  #ffffff !important;
  }
</style>
""", unsafe_allow_html=True)

# Second CSS block injected last so it wins any cascade tie with Streamlit internals
st.markdown("""
<style>
  [data-testid="stTooltipIcon"],
  [data-testid="stTooltipHoverTarget"],
  button[data-testid="stTooltipHoverTarget"] {
    color: #e5e7eb !important;
    background: transparent !important;
    opacity: 1 !important;
    cursor: pointer !important;
  }
  [data-testid="stTooltipIcon"] svg,
  [data-testid="stTooltipHoverTarget"] svg,
  button[data-testid="stTooltipHoverTarget"] svg {
    fill: none !important;
    stroke: #e5e7eb !important;
    color: #e5e7eb !important;
    opacity: 1 !important;
  }
  [data-testid="stTooltipIcon"] svg *,
  [data-testid="stTooltipHoverTarget"] svg *,
  button[data-testid="stTooltipHoverTarget"] svg * {
    fill: none !important;
    stroke: #e5e7eb !important;
  }
  [data-testid="stTooltipIcon"]:hover,
  [data-testid="stTooltipHoverTarget"]:hover,
  button[data-testid="stTooltipHoverTarget"]:hover {
    color: #ffffff !important;
    cursor: pointer !important;
  }
  [data-testid="stTooltipIcon"]:hover svg,
  [data-testid="stTooltipIcon"]:hover svg *,
  [data-testid="stTooltipHoverTarget"]:hover svg,
  [data-testid="stTooltipHoverTarget"]:hover svg *,
  button[data-testid="stTooltipHoverTarget"]:hover svg,
  button[data-testid="stTooltipHoverTarget"]:hover svg * {
    fill: none !important;
    stroke: #ffffff !important;
    color: #ffffff !important;
  }
</style>
""", unsafe_allow_html=True)

# Third CSS block: white text inside all input controls (late injection wins cascade)
st.markdown("""
<style>
  /* Typed text in number/text inputs */
  input, textarea {
    color: #ffffff !important;
  }
  /* Streamlit-specific input wrappers */
  [data-testid="stNumberInput"] input,
  [data-testid="stTextInput"] input,
  [data-testid="stTextArea"] textarea {
    color: #ffffff !important;
  }
  /* Selectbox / dropdown selected value */
  [data-baseweb="select"] span,
  [data-baseweb="select"] div[class*="ValueContainer"] span,
  [data-baseweb="select"] div[class*="singleValue"],
  [data-testid="stSelectbox"] span {
    color: #ffffff !important;
  }
  /* Dropdown option list items */
  [data-baseweb="menu"] [role="option"],
  [data-baseweb="menu"] li {
    color: #ffffff !important;
  }
  /* Placeholder text — slightly dimmer so it reads as hint */
  input::placeholder, textarea::placeholder {
    color: #9ca3af !important;
    opacity: 1;
  }
</style>
""", unsafe_allow_html=True)

# Fourth CSS block: force dark-theme on BaseWeb select inner control + open menu.
# The outer [data-baseweb="select"] wrapper was already dark, but BaseWeb sets its own
# background on the inner ControlContainer child divs — those must also be overridden.
st.markdown("""
<style>
  /* ── Closed selectbox control: inner ControlContainer ── */
  [data-baseweb="select"] > div,
  [data-baseweb="select"] > div > div {
    background-color: #0f3460 !important;
    color: #ffffff !important;
    border-color: #2d5986 !important;
  }
  /* Selected value text and placeholder inside the control */
  [data-baseweb="select"] > div span,
  [data-baseweb="select"] > div div[class*="placeholder"],
  [data-baseweb="select"] > div div[class*="singleValue"],
  [data-baseweb="select"] > div div[class*="value"] {
    color: #ffffff !important;
  }
  /* Dropdown arrow / chevron SVG */
  [data-baseweb="select"] svg {
    fill: #e5e7eb !important;
    stroke: none !important;
  }
  /* ── Open dropdown menu (popover) ── */
  [data-baseweb="popover"] [data-baseweb="menu"],
  [data-baseweb="popover"] ul,
  [data-baseweb="popover"] [role="listbox"] {
    background-color: #0f3460 !important;
    border: 1px solid #2d5986 !important;
  }
  /* Menu option items */
  [data-baseweb="popover"] [role="option"],
  [data-baseweb="popover"] li {
    background-color: #0f3460 !important;
    color: #ffffff !important;
  }
  /* Hovered option */
  [data-baseweb="popover"] [role="option"]:hover {
    background-color: #1a4a7a !important;
    color: #ffffff !important;
  }
  /* Currently selected option */
  [data-baseweb="popover"] [aria-selected="true"],
  [data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background-color: #1e3a5f !important;
    color: #4a9eff !important;
  }
</style>
""", unsafe_allow_html=True)

# Fifth CSS block: unify borders across ALL input types.
# The white border on number inputs wraps the OUTER div ([data-testid="stNumberInput"] > div)
# which contains both the value field and the +/- steppers.  Targeting only
# [data-baseweb="input"] (the inner text-only sub-div) missed that outer wrapper entirely.
st.markdown("""
<style>
  /* ═══════════════════════════════════════════════════════════
     NUMBER INPUTS — outer container owns the visible border
     (it wraps the value field AND the +/- stepper buttons)
     ═══════════════════════════════════════════════════════════ */
  [data-testid="stNumberInput"] > div {
    border: 1px solid #2f4f6f !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    outline: none !important;
  }
  /* Strip inner borders so they don't double-up with the outer one */
  [data-testid="stNumberInput"] > div > div,
  [data-testid="stNumberInput"] [data-baseweb="input"] {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
  }
  /* Stepper buttons: no independent border; divider line from parent background */
  [data-testid="stNumberInputStepDown"],
  [data-testid="stNumberInputStepUp"] {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
  }
  /* Focus: brighter blue ring on the outer container, no white glow */
  [data-testid="stNumberInput"] > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: none !important;
  }

  /* ═══════════════════════════════════════════════════════════
     TEXT INPUTS
     ═══════════════════════════════════════════════════════════ */
  [data-testid="stTextInput"] > div,
  [data-testid="stTextArea"] > div {
    border: 1px solid #2f4f6f !important;
    border-radius: 4px !important;
    box-shadow: none !important;
  }
  [data-testid="stTextInput"] [data-baseweb="input"],
  [data-testid="stTextArea"] textarea {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
  }
  [data-testid="stTextInput"] > div:focus-within,
  [data-testid="stTextArea"] > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: none !important;
  }

  /* ═══════════════════════════════════════════════════════════
     STANDALONE [data-baseweb="input"] — catches any input not
     wrapped in a stNumberInput/stTextInput testid
     ═══════════════════════════════════════════════════════════ */
  [data-baseweb="input"] {
    border: 1px solid #2f4f6f !important;
    box-shadow: none !important;
  }
  [data-baseweb="input"]:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: none !important;
  }
  /* Raw <input> and <textarea> elements never own a visible border */
  input, textarea {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
  }

  /* ═══════════════════════════════════════════════════════════
     SELECTBOXES
     ═══════════════════════════════════════════════════════════ */
  [data-baseweb="select"] {
    border: none !important;
    box-shadow: none !important;
  }
  [data-baseweb="select"] > div {
    border: 1px solid #2f4f6f !important;
    box-shadow: none !important;
  }
  [data-baseweb="select"]:focus-within > div {
    border-color: #3b82f6 !important;
    box-shadow: none !important;
  }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Defaults
# ==============================================================================

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]

DEFAULT_TRUE_T1  = [0.01, 0.02, 0.12, 0.20, 0.35]
DEFAULT_TRUE_T2  = [0.02, 0.05, 0.15, 0.25, 0.40]

R_DEFAULTS = {
    # Study
    "target_t1":          0.15,
    "target_t2":          0.33,
    "p_surgery":          0.80,
    "start_level_1b":     2,    # L-level (0-based): L2 default when pre-treated patients exist (n_safe_d1=6)
    # Simulation
    "n_sims":             200,
    "seed":               123,
    # Accrual
    "accrual_per_month":  1.5,
    # Timing (days)
    "incl_to_rt":         21,
    "rt_dur":             14,
    "rt_to_surg":         42,
    "tox2_win":           30,
    # Sample sizes
    "max_n_63":           27,
    "max_n_crm":          27,
    "cohort_size":        3,
    # Pre-treated patients at L1 (dose index 1)
    "n_safe_d1":          6,
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
    "require_full_tox1_fu": True,
    "ewoc_on":            True,
    "ewoc_alpha":         0.25,
    "ewoc_application":   "Dose assignment + final MTD",
    # Early stopping
    "early_stop_on":      False,
    "p_stop":             0.80,
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
    # Playground prior-endpoint tab (must be initialised so the slider
    # conditional never sees an undefined key on first Playground load)
    "prior_ep_tab":       "Tox1 (acute)",
    # Playground prior scenario selector
    "prior_scenario":     "Custom",
    # Design Exploration — stress test settings
    "de_expl_type":      "Design parameter sweep",
    "de_st_method":      "Scale probabilities",
    "de_st_mode":        "Both endpoints",
    "de_st_n_scenarios": 5,
    "de_st_scale_spread": 0.5,
    "de_st_shift_spread": 1.0,
    "de_st_scale_str":   "",
    "de_st_shift_str":   "",
    "de_st_n_sim":       200,
    "de_st_seed":        42,
}

TRUE_T1_KEYS  = [f"true_t1_L{i}"  for i in range(5)]
TRUE_T2_KEYS  = [f"true_t2_L{i}"  for i in range(5)]
# Widget-layer keys for the true-tox number_inputs (wl_ prefix = never canonical)
WL_TRUE_T1_KEYS = [f"wl_true_t1_L{i}" for i in range(5)]
WL_TRUE_T2_KEYS = [f"wl_true_t2_L{i}" for i in range(5)]

# Single merged defaults registry — the ONE source of all default values.
# true_t1/t2 ARE included here so init_state() seeds the canonical keys and
# navigation no longer resets them.  Widgets use WL_TRUE_T*_KEYS (wl_ prefix)
# as their key= argument so there is no Streamlit ≥1.31 session_state + value=
# conflict.  Pre-writes read from the canonical keys; value= is not used.
_ALL_DEFAULTS: dict = {
    **R_DEFAULTS,
    **{TRUE_T1_KEYS[i]: DEFAULT_TRUE_T1[i] for i in range(5)},
    **{TRUE_T2_KEYS[i]: DEFAULT_TRUE_T2[i] for i in range(5)},
}

# Kept for backward-compatibility (get_config_value fallback chain).
_TRUE_DEFAULTS: dict = {
    **{TRUE_T1_KEYS[i]: DEFAULT_TRUE_T1[i] for i in range(5)},
    **{TRUE_T2_KEYS[i]: DEFAULT_TRUE_T2[i] for i in range(5)},
}

# All canonical config keys in one ordered list (used by helpers below)
_ALL_CONFIG_KEYS = list(_ALL_DEFAULTS.keys())

# ==============================================================================
# Prior scenario presets
# Each entry maps to concrete prior parameter values.  "Custom" has no preset
# values — it exposes the raw sliders instead.
# Keys used: prior_target_t1, halfwidth_t1, prior_nu_t1 (and _t2 equivalents).
# ==============================================================================

_PRIOR_SCENARIOS: dict = {
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
# Single-source-of-truth state management
# ==============================================================================

_STATE_VERSION = "2026-07-06a"

def init_state() -> None:
    """Seed EVERY canonical config key exactly once per session.

    This is the single-source state contract.  All views (Essentials,
    Playground, Design Exploration), the simulation runner, and the debug
    panel may safely read any key from session_state (or via _cfg) regardless
    of which view was visited first.

    Design rule: widgets use NO explicit value= argument.  Streamlit reads
    the pre-seeded session_state value, avoiding the Streamlit ≥1.31
    StreamlitAPIException ("widget value cannot be set via st.session_state
    AND the widget's value= argument simultaneously").  Halfwidth clamping
    when prior_target sliders change is handled by on_change callbacks.

    Version check: if _STATE_VERSION has changed (e.g. after a code update),
    ALL canonical keys are force-reset to factory defaults so stale session
    state from crashes or previous test sessions never persists.
    """
    if st.session_state.get("_state_version") != _STATE_VERSION:
        for k, v in _ALL_DEFAULTS.items():
            st.session_state[k] = v
        st.session_state["_state_version"] = _STATE_VERSION
        st.rerun()  # widgets re-read fresh session_state on next run
    else:
        for k, v in _ALL_DEFAULTS.items():
            if k not in st.session_state:
                st.session_state[k] = v


def get_config_value(key: str):
    """Read *key* from the canonical session_state store.

    Falls back to ``_ALL_DEFAULTS`` then ``_TRUE_DEFAULTS`` so callers never
    see a KeyError or None, even before any widget for that key has rendered.
    """
    return st.session_state.get(
        key, _ALL_DEFAULTS.get(key, _TRUE_DEFAULTS.get(key))
    )


def set_config_value(key: str, value) -> None:
    """Write *value* into the canonical session_state store.

    Only use this from callbacks or one-off logic, not on every rerun.
    """
    st.session_state[key] = value


def _get_all_config() -> dict:
    """Return a snapshot dict of every shared parameter from session_state.

    Falls back to _ALL_DEFAULTS for keys whose widget has not yet rendered
    (e.g. true_t1_L* before Playground is first visited).
    Read-only — never mutates session_state.
    """
    return {k: st.session_state.get(k, _ALL_DEFAULTS.get(k, "—"))
            for k in _ALL_CONFIG_KEYS}


def h(key, desc, r_name=None):
    txt = desc
    if r_name:
        txt += f"\n\n*R equivalent: `{r_name}`*"
    return txt

# Initialise every shared key exactly once per session
init_state()

# ── Reset-to-defaults button ──────────────────────────────────────────────────
def _do_reset():
    """Restore every canonical key to its factory default."""
    for k, v in _ALL_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["_state_version"] = _STATE_VERSION


# _VALID_RANGES: lightweight type/range info for consistency validation.
# Maps key → (type_fn, min, max) or (type_fn, None, None) for no-range check.
_VALID_RANGES: dict = {
    "target_t1":      (float, 0.05, 0.50),
    "target_t2":      (float, 0.05, 0.50),
    "p_surgery":      (float, 0.0,  1.0),
    "prior_target_t1":(float, 0.05, 0.50),
    "prior_target_t2":(float, 0.05, 0.50),
    "halfwidth_t1":   (float, 0.01, 0.49),
    "halfwidth_t2":   (float, 0.01, 0.49),
    "sigma":          (float, 0.2,  5.0),
    "ewoc_alpha":     (float, 0.01, 0.99),
}


def validate_state_contract() -> tuple[list[str], list[str]]:
    """Check that the full state contract is satisfied.

    Returns (errors, warnings) where:
      errors   — problems that would cause crashes or wrong results
      warnings — informational notes (e.g. a key only has its default value)

    Checks performed:
      1. CRITICAL: Every _ALL_CONFIG_KEYS key must be present in session_state.
         Because init_state() seeds all keys, any missing key means init_state()
         was bypassed or a new key was added without updating _ALL_DEFAULTS.
      2. RANGE:    Keys with _VALID_RANGES entries must be within bounds.
      3. TYPE:     Keys with _VALID_RANGES entries must be the correct type.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    for k in _ALL_CONFIG_KEYS:
        default = _ALL_DEFAULTS.get(k)
        raw     = st.session_state.get(k)

        if raw is None:
            errors.append(
                f"CONTRACT VIOLATION [{k}] — absent from session_state. "
                f"init_state() should have seeded this to {default!r}."
            )
            continue

        if k in _VALID_RANGES:
            type_fn, lo, hi = _VALID_RANGES[k]
            try:
                v = type_fn(raw)
            except (TypeError, ValueError):
                errors.append(f"TYPE ERR [{k}] = {raw!r} (expected {type_fn.__name__})")
                continue
            if lo is not None and not (lo <= v <= hi):
                errors.append(
                    f"OUT OF RANGE [{k}] = {v} not in [{lo}, {hi}] "
                    f"(default {default!r})"
                )

    return errors, warnings


# Legacy alias — keep so existing callers (debug expander) still work
def validate_shared_state_consistency() -> list[str]:
    errors, _ = validate_state_contract()
    return errors


def _cfg(key: str):
    """Convenience alias: typed canonical read with _ALL_DEFAULTS fallback.
    Raises ValueError with a clear message if the key is not in _ALL_DEFAULTS,
    which would indicate a programming error (not a user error).
    """
    if key not in _ALL_DEFAULTS:
        raise ValueError(f"_cfg(): unknown config key {key!r} — add to _ALL_DEFAULTS")
    return st.session_state.get(key, _ALL_DEFAULTS[key])


# ── on_change callbacks for prior-target sliders ──────────────────────────────
# These clamp halfwidth when the user moves a prior_target slider, preventing
# the halfwidth slider from getting out of the dynamic valid range.
# Using callbacks (not script-body writes) is safe: Streamlit runs the callback
# BEFORE the next render, so no "widget default vs Session State API" warning.

def _clamp_halfwidth_t1() -> None:
    """Sync target slider→config and clamp halfwidth_t1 when prior_target_t1 changes."""
    pt = float(st.session_state.get("sl_prior_target_t1", R_DEFAULTS["prior_target_t1"]))
    st.session_state["prior_target_t1"] = pt
    max_hw = max(0.01, round(min(pt - 0.01, 1.0 - pt - 0.01), 2))
    hw = float(st.session_state.get("halfwidth_t1", R_DEFAULTS["halfwidth_t1"]))
    if hw > max_hw:
        st.session_state["halfwidth_t1"] = max_hw
        st.session_state["sl_halfwidth_t1"] = max_hw


def _clamp_halfwidth_t2() -> None:
    """Sync target slider→config and clamp halfwidth_t2 when prior_target_t2 changes."""
    pt = float(st.session_state.get("sl_prior_target_t2", R_DEFAULTS["prior_target_t2"]))
    st.session_state["prior_target_t2"] = pt
    max_hw = max(0.01, round(min(pt - 0.01, 1.0 - pt - 0.01), 2))
    hw = float(st.session_state.get("halfwidth_t2", R_DEFAULTS["halfwidth_t2"]))
    if hw > max_hw:
        st.session_state["halfwidth_t2"] = max_hw
        st.session_state["sl_halfwidth_t2"] = max_hw


def _sync_halfwidth_t1() -> None:
    """Sync halfwidth_t1 widget → canonical config when the user moves that slider."""
    st.session_state["halfwidth_t1"] = float(
        st.session_state.get("sl_halfwidth_t1", R_DEFAULTS["halfwidth_t1"])
    )


def _sync_halfwidth_t2() -> None:
    """Sync halfwidth_t2 widget → canonical config when the user moves that slider."""
    st.session_state["halfwidth_t2"] = float(
        st.session_state.get("sl_halfwidth_t2", R_DEFAULTS["halfwidth_t2"])
    )


def _sync_prior_nu_t1() -> None:
    """Sync prior_nu_t1 widget → canonical config when the user moves that slider."""
    st.session_state["prior_nu_t1"] = int(
        st.session_state.get("sl_prior_nu_t1", R_DEFAULTS["prior_nu_t1"])
    )


def _sync_prior_nu_t2() -> None:
    """Sync prior_nu_t2 widget → canonical config when the user moves that slider."""
    st.session_state["prior_nu_t2"] = int(
        st.session_state.get("sl_prior_nu_t2", R_DEFAULTS["prior_nu_t2"])
    )


# ── Generic sync-callback factory ─────────────────────────────────────────────
# Every Essentials widget uses the pre-write / post-read (wl_) pattern:
#   1. pre-write:  session_state["wl_X"] = canonical_config_value  (before render)
#   2. render:     st.<widget>(key="wl_X", on_change=_sync_X)
#   3. on_change:  session_state["X"] = widget_value               (before next run)
#   4. post-read:  session_state["X"] = session_state["wl_X"]      (after render)
#
# The on_change callback is critical: without it, step 1 reads the STALE
# canonical value (not yet updated from the widget), overwrites the user's
# pending interaction, and the widget snaps back.  With on_change, the
# canonical is updated BEFORE the next script run, so step 1 reads the
# correct new value.

def _make_sync(canonical_key: str, type_fn, wl_key: str):
    """Return an on_change callback that commits wl_key → canonical_key."""
    def _cb():
        st.session_state[canonical_key] = type_fn(
            st.session_state.get(wl_key, _ALL_DEFAULTS.get(canonical_key))
        )
    _cb.__name__ = f"_sync_{canonical_key}"
    return _cb


# Essentials left column
_sync_target_t1      = _make_sync("target_t1",       float, "wl_target_t1")
_sync_target_t2      = _make_sync("target_t2",       float, "wl_target_t2")
_sync_p_surgery      = _make_sync("p_surgery",       float, "wl_p_surgery")
_sync_start_level_1b = _make_sync("start_level_1b",  int,   "wl_start_level_1b")
_sync_n_sims         = _make_sync("n_sims",          int,   "wl_n_sims")
_sync_seed           = _make_sync("seed",            int,   "wl_seed")
_sync_accrual        = _make_sync("accrual_per_month", float, "wl_accrual_per_month")
# Essentials middle column
_sync_incl_to_rt     = _make_sync("incl_to_rt",      int,   "wl_incl_to_rt")
_sync_rt_dur         = _make_sync("rt_dur",          int,   "wl_rt_dur")
_sync_rt_to_surg     = _make_sync("rt_to_surg",      int,   "wl_rt_to_surg")
_sync_tox2_win       = _make_sync("tox2_win",        int,   "wl_tox2_win")
_sync_max_n_63       = _make_sync("max_n_63",        int,   "wl_max_n_63")
_sync_max_n_crm      = _make_sync("max_n_crm",       int,   "wl_max_n_crm")
_sync_cohort_size    = _make_sync("cohort_size",     int,   "wl_cohort_size")
def _sync_n_safe_d1():
    """Sync n_safe_d1 widget → canonical, and auto-adjust start_level_1b default."""
    new_val = int(st.session_state.get("wl_n_safe_d1", R_DEFAULTS["n_safe_d1"]))
    st.session_state["n_safe_d1"] = new_val
    # Auto-adjust start_level_1b (L-level, 0-based): if user hasn't set it away from
    # the base defaults (L1 when no pretreated, L2 when pretreated), keep it in sync.
    cur = int(st.session_state.get("start_level_1b", R_DEFAULTS["start_level_1b"]))
    if new_val > 0 and cur == 1:
        st.session_state["start_level_1b"] = 2
    elif new_val == 0 and cur == 2:
        st.session_state["start_level_1b"] = 1
# Essentials right column
_sync_gh_n              = _make_sync("gh_n",              int,   "wl_gh_n")
_sync_max_step          = _make_sync("max_step",          int,   "wl_max_step")
_sync_sigma             = _make_sync("sigma",             float, "sl_sigma")
_sync_enforce_guardrail = _make_sync("enforce_guardrail", bool,  "wl_enforce_guardrail")
_sync_restrict_final_mtd= _make_sync("restrict_final_mtd",bool,  "wl_restrict_final_mtd")
_sync_burn_in              = _make_sync("burn_in",              bool, "wl_burn_in")
_sync_require_full_tox1_fu = _make_sync("require_full_tox1_fu", bool, "wl_require_full_tox1_fu")
_sync_ewoc_alpha        = _make_sync("ewoc_alpha",        float, "wl_ewoc_alpha")
def _sync_ewoc_application():
    """Sync EWOC application selectbox → canonical, keeping the legacy
    ewoc_on boolean consistent (True unless the mode is Off) so helper code
    and exported configs that still read ewoc_on keep working."""
    app = str(st.session_state.get("wl_ewoc_application",
                                   R_DEFAULTS["ewoc_application"]))
    st.session_state["ewoc_application"] = app
    st.session_state["ewoc_on"] = (app != EWOC_APP_OFF)
_sync_early_stop_on     = _make_sync("early_stop_on",     bool,  "wl_early_stop_on")
_sync_p_stop            = _make_sync("p_stop",            float, "wl_p_stop")
_sync_show_crm_trace    = _make_sync("show_crm_trace",    bool,  "wl_show_crm_trace")
# 6+3 thresholds
_sync_a6_esc_max  = _make_sync("a6_esc_max",  int, "wl_a6_esc_max")
_sync_a6_stop_min = _make_sync("a6_stop_min", int, "wl_a6_stop_min")
_sync_a9_esc_max  = _make_sync("a9_esc_max",  int, "wl_a9_esc_max")
_sync_s6_esc_max  = _make_sync("s6_esc_max",  int, "wl_s6_esc_max")
_sync_s6_stop_min = _make_sync("s6_stop_min", int, "wl_s6_stop_min")
_sync_s9_esc_max  = _make_sync("s9_esc_max",  int, "wl_s9_esc_max")
_sync_s9_stop_min = _make_sync("s9_stop_min", int, "wl_s9_stop_min")
_sync_prior_scenario = _make_sync("prior_scenario", str, "wl_prior_scenario")
# Playground true-tox probabilities
_sync_true_t1 = [_make_sync(TRUE_T1_KEYS[i], float, WL_TRUE_T1_KEYS[i]) for i in range(5)]
_sync_true_t2 = [_make_sync(TRUE_T2_KEYS[i], float, WL_TRUE_T2_KEYS[i]) for i in range(5)]
# Playground skeleton model and prior endpoint tab
_sync_prior_model   = _make_sync("prior_model",   str, "wl_prior_model")
_sync_prior_ep_tab  = _make_sync("prior_ep_tab",  str, "wl_prior_ep_tab")
# Playground widget sync callbacks — mirror the Essentials wl_/canonical pattern so
# that navigating away from Playground and returning preserves user-entered values.
# Streamlit ≥1.28 deletes widget-backed session_state keys when their widgets are
# not rendered, so every Playground widget that must survive navigation needs its own
# non-widget canonical key (set via on_change, not by the widget directly).
_sync_prior_model  = _make_sync("prior_model",   str, "wl_prior_model")
_sync_prior_ep_tab = _make_sync("prior_ep_tab",  str, "wl_prior_ep_tab")


def _make_true_t_sync(canon_key: str, wl_key: str):
    """Return an on_change callback that commits wl_key → canon_key (float)."""
    def _cb():
        st.session_state[canon_key] = float(st.session_state.get(wl_key, 0.0))
    _cb.__name__ = f"_sync_{canon_key}"
    return _cb


_sync_true_t1 = [
    _make_true_t_sync(TRUE_T1_KEYS[i], f"wl_{TRUE_T1_KEYS[i]}") for i in range(5)
]
_sync_true_t2 = [
    _make_true_t_sync(TRUE_T2_KEYS[i], f"wl_{TRUE_T2_KEYS[i]}") for i in range(5)
]


def _apply_prior_scenario() -> None:
    """on_change for the Prior scenario selectbox.

    Commits the selected scenario name to the canonical config, then — for
    every non-Custom scenario — overwrites the underlying prior parameters
    (prior_target, halfwidth, prior_nu for both endpoints) so that the
    skeleton computation and any Custom-mode sliders always start from the
    scenario's preset values.

    Streamlit fires on_change BEFORE the next script run, so the pre-writes
    that follow will read the already-updated canonical values.
    """
    scenario = str(st.session_state.get("wl_prior_scenario", "Custom"))
    st.session_state["prior_scenario"] = scenario

    if scenario == "Custom":
        return  # keep whatever values the user previously set

    preset = _PRIOR_SCENARIOS.get(scenario, {})
    for key in ("prior_target_t1", "halfwidth_t1", "prior_nu_t1",
                "prior_target_t2", "halfwidth_t2", "prior_nu_t2"):
        if key in preset:
            st.session_state[key] = preset[key]
    # Mirror into slider widget keys so pre-writes on next render are
    # consistent (avoids a one-frame lag where sliders show stale values).
    for sl_key, cfg_key in [
        ("sl_prior_target_t1", "prior_target_t1"),
        ("sl_halfwidth_t1",    "halfwidth_t1"),
        ("sl_prior_nu_t1",     "prior_nu_t1"),
        ("sl_prior_target_t2", "prior_target_t2"),
        ("sl_halfwidth_t2",    "halfwidth_t2"),
        ("sl_prior_nu_t2",     "prior_nu_t2"),
    ]:
        if cfg_key in preset:
            st.session_state[sl_key] = preset[cfg_key]


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
# Configuration import / export
# ==============================================================================

# (key, type-coerce-fn) for every section.
# These are the *canonical* session-state keys that _cfg() / get_config_value()
# reads from.  The wl_* / sl_* widget aliases are pre-written from canonical on
# every render, so importing into canonical keys is all that is needed.

_CFG_ESSENTIALS_KEYS: list[tuple[str, type]] = [
    # Study endpoints
    ("target_t1",           float),
    ("target_t2",           float),
    ("p_surgery",           float),
    ("start_level_1b",      int),
    # Simulation
    ("n_sims",              int),
    ("seed",                int),
    ("accrual_per_month",   float),
    # Timing (days)
    ("incl_to_rt",          int),
    ("rt_dur",              int),
    ("rt_to_surg",          int),
    ("tox2_win",            int),
    # Sample sizes
    ("max_n_63",            int),
    ("max_n_crm",           int),
    ("cohort_size",         int),
    ("n_safe_d1",           int),
    # CRM integration
    ("gh_n",                int),
    ("max_step",            int),
    ("sigma",               float),
    # CRM safety / selection
    ("enforce_guardrail",   bool),
    ("restrict_final_mtd",  bool),
    # CRM behaviour
    ("burn_in",             bool),
    ("require_full_tox1_fu", bool),
    ("ewoc_on",             bool),
    ("ewoc_alpha",          float),
    ("ewoc_application",    str),
    ("early_stop_on",       bool),
    ("p_stop",              float),
    # CRM decision trace
    ("show_crm_trace",      bool),
    # 6+3 stopping rules
    ("a6_esc_max",          int),
    ("a6_stop_min",         int),
    ("a9_esc_max",          int),
    ("s6_esc_max",          int),
    ("s6_stop_min",         int),
    ("s9_esc_max",          int),
    ("s9_stop_min",         int),
]

_CFG_PLAYGROUND_PRIOR_KEYS: list[tuple[str, type]] = [
    ("prior_model",         str),
    ("prior_scenario",      str),
    ("prior_target_t1",     float),
    ("halfwidth_t1",        float),
    ("prior_nu_t1",         int),
    ("prior_target_t2",     float),
    ("halfwidth_t2",        float),
    ("prior_nu_t2",         int),
    ("logistic_intcpt",     float),
]

_CFG_DE_KEYS: list[tuple[str, type]] = [
    ("de_param_name",   str),
    ("de_sig_min",      float),
    ("de_sig_max",      float),
    ("de_sig_pts",      int),
    ("de_ea_min",       float),
    ("de_ea_max",       float),
    ("de_ea_pts",       int),
    ("de_inc_off",      bool),
    ("de_max_n_vals",   list),
    ("de_nu1_vals",     list),
    ("de_nu2_vals",     list),
    ("de_cohort_vals",  list),
    ("de_n_sim",        int),
    ("de_seed",         int),
    ("de_speed_mode",   bool),
    ("de_expl_type",       str),
    ("de_st_method",       str),
    ("de_st_mode",         str),
    ("de_st_n_scenarios",  int),
    ("de_st_scale_spread", float),
    ("de_st_shift_spread", float),
    ("de_st_scale_str",    str),
    ("de_st_shift_str",    str),
    ("de_st_n_sim",        int),
    ("de_st_seed",         int),
]

# Default values used when a DE key is absent from session state at export time.
_CFG_DE_DEFAULTS: dict = {
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
    "de_expl_type":       "Design parameter sweep",
    "de_st_method":       "Scale probabilities",
    "de_st_mode":         "Both endpoints",
    "de_st_n_scenarios":  5,
    "de_st_scale_spread": 0.5,
    "de_st_shift_spread": 1.0,
    "de_st_scale_str":    "",
    "de_st_shift_str":    "",
    "de_st_n_sim":        200,
    "de_st_seed":         42,
}

_CFG_SCHEMA_VERSION = 1


def _build_config_dict() -> dict:
    """Return a JSON-serialisable dict of every user-configurable parameter.

    Structure::

        {
          "_meta":              { app, schema_version, state_version, exported_at },
          "essentials":         { ... canonical Essentials keys ... },
          "playground": {
              "true_t1":        [float × 5],
              "true_t2":        [float × 5],
              ... canonical prior keys ...
          },
          "design_exploration": { ... de_* keys ... },
        }
    """
    ss = st.session_state

    essentials = {k: get_config_value(k) for k, _ in _CFG_ESSENTIALS_KEYS}

    playground: dict = {
        "true_t1": [float(get_config_value(f"true_t1_L{i}")) for i in range(5)],
        "true_t2": [float(get_config_value(f"true_t2_L{i}")) for i in range(5)],
    }
    for k, _ in _CFG_PLAYGROUND_PRIOR_KEYS:
        playground[k] = get_config_value(k)

    design_exploration = {
        k: ss.get(k, _CFG_DE_DEFAULTS[k])
        for k, _ in _CFG_DE_KEYS
    }

    return {
        "_meta": {
            "app": "CRM Simulator",
            "schema_version": _CFG_SCHEMA_VERSION,
            "state_version": _STATE_VERSION,
            "exported_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "essentials": essentials,
        "playground": playground,
        "design_exploration": design_exploration,
    }


def _apply_imported_config(cfg: dict) -> list[str]:
    """Write values from *cfg* into session state.  Returns a list of warning strings.

    Only writes keys that are present in the imported file; missing keys keep
    their current / default values.  Type errors are caught and reported as
    warnings rather than crashing.

    Because every Essentials and Playground widget uses the pre-write pattern::

        st.session_state["wl_KEY"] = _cfg("canonical_key")   # runs each render
        st.widget(..., key="wl_KEY", ...)
        st.session_state["canonical_key"] = st.session_state["wl_KEY"]

    writing to the canonical key is sufficient; the wl_/sl_ aliases are updated
    automatically on the next Streamlit re-render.
    """
    ss = st.session_state
    warns: list[str] = []

    # ── Schema version check ──────────────────────────────────────────────────
    meta = cfg.get("_meta", {})
    imported_sv = meta.get("schema_version", 1)
    if not isinstance(imported_sv, int) or imported_sv > _CFG_SCHEMA_VERSION:
        warns.append(
            f"Config schema version {imported_sv} is newer than this app "
            f"(expected ≤{_CFG_SCHEMA_VERSION}). Some values may not import correctly."
        )

    # ── Helper ────────────────────────────────────────────────────────────────
    def _set(key: str, raw_val, coerce: type) -> None:
        try:
            if coerce is bool:
                # json.loads already yields bool; also handle "true"/"false" strings
                if isinstance(raw_val, str):
                    ss[key] = raw_val.lower() == "true"
                else:
                    ss[key] = bool(raw_val)
            elif coerce is list:
                ss[key] = list(raw_val)
            else:
                ss[key] = coerce(raw_val)
        except (ValueError, TypeError) as exc:
            warns.append(f"Could not import '{key}': {exc} — keeping current value.")

    # ── Essentials ────────────────────────────────────────────────────────────
    ess = cfg.get("essentials", {})
    for key, coerce in _CFG_ESSENTIALS_KEYS:
        if key in ess:
            _set(key, ess[key], coerce)

    # ── Playground — true tox arrays ─────────────────────────────────────────
    pg = cfg.get("playground", {})
    for tox_key, arr_name in (("true_t1_L", "true_t1"), ("true_t2_L", "true_t2")):
        if arr_name in pg:
            arr = pg[arr_name]
            if isinstance(arr, list) and len(arr) == 5:
                for i, v in enumerate(arr):
                    _set(f"{tox_key}{i}", v, float)
            else:
                warns.append(
                    f"playground.{arr_name} must be a list of 5 numbers — skipped."
                )

    # ── Playground — prior settings ───────────────────────────────────────────
    for key, coerce in _CFG_PLAYGROUND_PRIOR_KEYS:
        if key in pg:
            _set(key, pg[key], coerce)

    # ── Design Exploration ────────────────────────────────────────────────────
    de = cfg.get("design_exploration", {})
    for key, coerce in _CFG_DE_KEYS:
        if key in de:
            _set(key, de[key], coerce)

    return warns


# ==============================================================================
# Navigation (sidebar)
# ==============================================================================

view = st.sidebar.radio(
    "View",
    options=["Essentials", "Playground", "Design Exploration"],
    key="nav_view",
    label_visibility="collapsed",
)

# ── Import / Export configuration ─────────────────────────────────────────────
_ss = st.session_state   # convenience alias used below (re-used by view sections)

with st.sidebar.expander("⚙ Configuration", expanded=False):
    # ── Export ────────────────────────────────────────────────────────────────
    _cfg_json_str = json.dumps(_build_config_dict(), indent=2)
    _cfg_fname    = (
        "crm_config_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    st.download_button(
        "⬇ Export config",
        data=_cfg_json_str.encode("utf-8"),
        file_name=_cfg_fname,
        mime="application/json",
        key="cfg_export_btn",
        use_container_width=True,
        help="Download all current settings (Essentials, Playground, Design Exploration) as a JSON file.",
    )

    st.divider()

    # ── Import ────────────────────────────────────────────────────────────────
    _uploaded = st.file_uploader(
        "Import config",
        type=["json"],
        key="cfg_import_uploader",
        help="Upload a previously exported CRM Simulator config (.json) to restore all settings.",
    )
    if _uploaded is not None:
        # Track by (name, size) so the same file isn't re-applied on every rerun.
        _file_id = (_uploaded.name, _uploaded.size)
        if _ss.get("_cfg_last_import_id") != _file_id:
            # New upload — parse and apply
            try:
                _raw = _uploaded.read().decode("utf-8")
                _cfg_data = json.loads(_raw)
                if not isinstance(_cfg_data, dict):
                    raise ValueError("Config file must be a JSON object.")
                _import_warns = _apply_imported_config(_cfg_data)
                _ss["_cfg_last_import_id"]   = _file_id
                _ss["_cfg_import_ok"]        = True
                _ss["_cfg_import_warnings"]  = _import_warns
            except json.JSONDecodeError as _je:
                _ss["_cfg_last_import_id"]   = _file_id
                _ss["_cfg_import_ok"]        = False
                _ss["_cfg_import_warnings"]  = [f"Invalid JSON: {_je}"]
            except Exception as _ie:
                _ss["_cfg_last_import_id"]   = _file_id
                _ss["_cfg_import_ok"]        = False
                _ss["_cfg_import_warnings"]  = [f"Import failed: {_ie}"]
            st.rerun()
        else:
            # Same file already applied — show persisted result
            if _ss.get("_cfg_import_ok"):
                st.success("✅ Config loaded.")
                for _w in _ss.get("_cfg_import_warnings", []):
                    st.warning(_w)
            else:
                for _msg in _ss.get("_cfg_import_warnings", ["Import failed."]):
                    st.error(_msg)

# ==============================================================================
# ESSENTIALS VIEW
# ==============================================================================

if view == "Essentials":
    _ec1, _ec2, _ec3 = st.columns(3, gap="large")

    with _ec1:
        st.markdown("#### Study endpoints")

        st.session_state["wl_target_t1"] = float(_cfg("target_t1"))
        st.number_input(
            "Target tox1 (acute) rate",
            min_value=0.05, max_value=0.50, step=0.01, key="wl_target_t1",
            on_change=_sync_target_t1,
            help=h("target_t1", "Target acute DLT probability for MTD definition.")
        )
        st.session_state["target_t1"] = st.session_state["wl_target_t1"]

        st.session_state["wl_target_t2"] = float(_cfg("target_t2"))
        st.number_input(
            "Target tox2 (subacute | surgery) rate",
            min_value=0.05, max_value=0.50, step=0.01, key="wl_target_t2",
            on_change=_sync_target_t2,
            help=h("target_t2",
                   "Target subacute DLT probability conditional on surgery. "
                   "Only surgery patients contribute to the tox2 model.")
        )
        st.session_state["target_t2"] = st.session_state["wl_target_t2"]

        st.session_state["wl_p_surgery"] = float(_cfg("p_surgery"))
        st.number_input(
            "Probability of surgery",
            min_value=0.0, max_value=1.0, step=0.01, key="wl_p_surgery",
            on_change=_sync_p_surgery,
            help=h("p_surgery",
                   "Global probability that a patient proceeds to surgery. "
                   "Dose-independent. Subacute toxicity only observed in these patients.")
        )
        st.session_state["p_surgery"] = st.session_state["wl_p_surgery"]

        # Smart default: L2 when pre-treated patients exist, L1 otherwise.
        # Only applied when start_level_1b has not yet been stored in session state.
        if "start_level_1b" not in st.session_state:
            _n_pretreated_init = int(_cfg("n_safe_d1"))
            st.session_state["start_level_1b"] = 2 if _n_pretreated_init > 0 else 1
        _dose_opts = ["L0", "L1", "L2", "L3", "L4"]
        st.session_state["wl_start_level_1b"] = int(_cfg("start_level_1b"))
        st.selectbox(
            "Start dose level",
            options=list(range(5)),
            format_func=lambda i: _dose_opts[i],
            index=int(_cfg("start_level_1b")),
            key="wl_start_level_1b",
            on_change=_sync_start_level_1b,
            help=h("start_level_1b",
                   "CRM starting dose level (L0 = lowest, L4 = highest). "
                   "Default is L1 when no pre-treated patients exist; "
                   "auto-adjusts to L2 when pre-treated patients are present at L1, "
                   "since L1 is already established as safe.")
        )
        st.session_state["start_level_1b"] = st.session_state["wl_start_level_1b"]

        st.markdown("#### Simulation")

        st.session_state["wl_n_sims"] = int(_cfg("n_sims"))
        st.number_input(
            "Number of simulated trials",
            min_value=50, max_value=5000, step=50, key="wl_n_sims",
            on_change=_sync_n_sims,
            help=h("n_sims", "Replicates for the simulation study.")
        )
        st.session_state["n_sims"] = st.session_state["wl_n_sims"]

        st.session_state["wl_seed"] = int(_cfg("seed"))
        st.number_input(
            "Random seed",
            min_value=1, max_value=10_000_000, step=1, key="wl_seed",
            on_change=_sync_seed,
            help=h("seed", "Random seed for reproducibility.")
        )
        st.session_state["seed"] = st.session_state["wl_seed"]

        st.session_state["wl_accrual_per_month"] = float(_cfg("accrual_per_month"))
        st.number_input(
            "Avg patients per month",
            min_value=0.1, max_value=20.0, step=0.1, key="wl_accrual_per_month",
            on_change=_sync_accrual,
            help=h("accrual_per_month",
                   "Average accrual rate. Arrivals simulated as a Poisson process "
                   "(exponential inter-arrival times at this rate).")
        )
        st.session_state["accrual_per_month"] = st.session_state["wl_accrual_per_month"]

    with _ec2:
        st.markdown("#### Timing (days)")

        st.session_state["wl_incl_to_rt"] = int(_cfg("incl_to_rt"))
        st.number_input(
            "Inclusion to RT start",
            min_value=0, max_value=180, step=1, key="wl_incl_to_rt",
            on_change=_sync_incl_to_rt,
            help=h("incl_to_rt",
                   "Days from enrolment to start of radiotherapy. "
                   "Tox1 window begins at RT start. Default ≈ 3 weeks.")
        )
        st.session_state["incl_to_rt"] = st.session_state["wl_incl_to_rt"]

        st.session_state["wl_rt_dur"] = int(_cfg("rt_dur"))
        st.number_input(
            "Radiotherapy duration",
            min_value=1, max_value=60, step=1, key="wl_rt_dur",
            on_change=_sync_rt_dur,
            help=h("rt_dur",
                   "Duration of radiotherapy in days. Default ≈ 2 weeks (10 fractions).")
        )
        st.session_state["rt_dur"] = st.session_state["wl_rt_dur"]

        st.session_state["wl_rt_to_surg"] = int(_cfg("rt_to_surg"))
        st.number_input(
            "RT end to surgery",
            min_value=1, max_value=365, step=1, key="wl_rt_to_surg",
            on_change=_sync_rt_to_surg,
            help=h("rt_to_surg",
                   "Days from end of radiotherapy to surgery. Default 42 days ≈ 6 weeks. "
                   "The tox1 (acute) follow-up window is derived as RT duration + this value, "
                   "so it always extends from RT start to the moment of surgery.")
        )
        st.session_state["rt_to_surg"] = st.session_state["wl_rt_to_surg"]

        st.session_state["wl_tox2_win"] = int(_cfg("tox2_win"))
        st.number_input(
            "Tox2 follow-up window (days)",
            min_value=7, max_value=180, step=1, key="wl_tox2_win",
            on_change=_sync_tox2_win,
            help=h("tox2_win",
                   "Post-surgery window for subacute toxicity assessment. Default 30 days.")
        )
        st.session_state["tox2_win"] = st.session_state["wl_tox2_win"]

        st.markdown("#### Sample size")

        st.session_state["wl_max_n_63"] = int(_cfg("max_n_63"))
        st.number_input(
            "Max sample size (6+3)",
            min_value=6, max_value=200, step=3, key="wl_max_n_63",
            on_change=_sync_max_n_63,
            help=h("max_n_63",
                   "Maximum total enrolled patients in the 6+3 arm, including "
                   "bridging patients treated at lower doses while awaiting evaluability.")
        )
        st.session_state["max_n_63"] = st.session_state["wl_max_n_63"]

        st.session_state["wl_max_n_crm"] = int(_cfg("max_n_crm"))
        st.number_input(
            "Max sample size (CRM)",
            min_value=6, max_value=200, step=3, key="wl_max_n_crm",
            on_change=_sync_max_n_crm,
            help=h("max_n_crm", "Maximum total enrolled patients in the TITE-CRM arm.")
        )
        st.session_state["max_n_crm"] = st.session_state["wl_max_n_crm"]

        st.session_state["wl_cohort_size"] = int(_cfg("cohort_size"))
        st.number_input(
            "Cohort size (CRM)",
            min_value=1, max_value=12, step=1, key="wl_cohort_size",
            on_change=_sync_cohort_size,
            help=h("cohort_size",
                   "Number of patients per CRM cohort. CRM updates after each "
                   "cohort is fully enrolled, using TITE weights at that moment.")
        )
        st.session_state["cohort_size"] = st.session_state["wl_cohort_size"]

        st.session_state["wl_n_safe_d1"] = int(_cfg("n_safe_d1"))
        st.number_input(
            "Pre-treated patients at L1",
            min_value=0, max_value=50, step=1, key="wl_n_safe_d1",
            on_change=_sync_n_safe_d1,
            help=h("n_safe_d1",
                   "Patients already safely treated at L1 (0 DLTs, complete follow-up) "
                   "before the trial opens. Pre-loaded into the CRM as fully weighted "
                   "observations at L1 with no toxicities. They count toward Max sample "
                   "size (CRM).")
        )
        st.session_state["n_safe_d1"] = st.session_state["wl_n_safe_d1"]

    with _ec3:
        st.markdown("#### CRM integration")

        # ── gh_n ──────────────────────────────────────────────────────────
        # Pre-write canonical → widget so the selectbox always shows the
        # correct value even after a simulation rerun with Essentials absent.
        st.session_state["wl_gh_n"] = int(_cfg("gh_n"))
        st.selectbox(
            "Gauss–Hermite points",
            options=[31, 41, 61, 81], key="wl_gh_n",
            on_change=_sync_gh_n,
            help=h("gh_n",
                   "Quadrature points for CRM posterior. Higher = more accurate, slower.")
        )
        st.session_state["gh_n"] = st.session_state["wl_gh_n"]

        # ── max_step ──────────────────────────────────────────────────────
        st.session_state["wl_max_step"] = int(_cfg("max_step"))
        st.selectbox(
            "Max dose step per update",
            options=[1, 2], key="wl_max_step",
            on_change=_sync_max_step,
            help=h("max_step",
                   "Max dose levels the CRM can move per cohort update.")
        )
        st.session_state["max_step"] = st.session_state["wl_max_step"]

        # ── sigma ─────────────────────────────────────────────────────────
        st.session_state["sl_sigma"] = float(_cfg("sigma"))
        st.slider(
            "Prior sigma on theta",
            min_value=0.2, max_value=5.0, step=0.1, key="sl_sigma",
            on_change=_sync_sigma,
            help=h("sigma",
                   "SD of theta in the CRM prior (shared for both endpoints). "
                   "Larger = more diffuse prior.",
                   r_name="prior.sigma / sigma")
        )
        st.session_state["sigma"] = st.session_state["sl_sigma"]

        st.markdown("#### CRM safety / selection")

        # ── enforce_guardrail ─────────────────────────────────────────────
        st.session_state["wl_enforce_guardrail"] = bool(_cfg("enforce_guardrail"))
        st.toggle(
            "Guardrail: next dose ≤ highest tried + 1",
            key="wl_enforce_guardrail",
            on_change=_sync_enforce_guardrail,
            help=h("enforce_guardrail", "Prevent skipping untried dose levels.")
        )
        st.session_state["enforce_guardrail"] = st.session_state["wl_enforce_guardrail"]

        # ── restrict_final_mtd ────────────────────────────────────────────
        st.session_state["wl_restrict_final_mtd"] = bool(_cfg("restrict_final_mtd"))
        st.toggle(
            "Final MTD must be among tried doses",
            key="wl_restrict_final_mtd",
            on_change=_sync_restrict_final_mtd,
            help=h("restrict_final_mtd",
                   "Restrict final MTD selection to doses where n > 0.")
        )
        st.session_state["restrict_final_mtd"] = st.session_state["wl_restrict_final_mtd"]

        st.markdown("#### CRM behaviour")

        # ── burn_in ───────────────────────────────────────────────────────
        st.session_state["wl_burn_in"] = bool(_cfg("burn_in"))
        st.toggle(
            "Burn-in until first tox1 DLT",
            key="wl_burn_in",
            on_change=_sync_burn_in,
            help=h("burn_in",
                   "Escalate one level at a time until the first observed acute DLT, "
                   "then switch to CRM updates.")
        )
        st.session_state["burn_in"] = st.session_state["wl_burn_in"]

        # ── require_full_tox1_fu ──────────────────────────────────────────
        st.session_state["wl_require_full_tox1_fu"] = bool(_cfg("require_full_tox1_fu"))
        st.toggle(
            "Require full tox1 follow-up before burn-in escalation",
            key="wl_require_full_tox1_fu",
            on_change=_sync_require_full_tox1_fu,
            help=h("require_full_tox1_fu",
                   "When enabled, escalation during burn-in is only allowed after the "
                   "current dose level has at least one full cohort (cohort_size patients) "
                   "with complete acute tox1 follow-up (RT start + tox1 window) and no "
                   "observed tox1 DLT. The escalation check is made at the next patient's "
                   "RT start, not at inclusion.")
        )
        st.session_state["require_full_tox1_fu"] = st.session_state["wl_require_full_tox1_fu"]

        # ── ewoc_application ──────────────────────────────────────────────
        # Reconcile with legacy state/imports: an old config may carry
        # ewoc_on=False without an ewoc_application key, or an invalid string.
        _ewoc_app_cur = str(_cfg("ewoc_application"))
        if _ewoc_app_cur not in EWOC_APP_OPTIONS:
            _ewoc_app_cur = R_DEFAULTS["ewoc_application"]
        if not bool(_cfg("ewoc_on")) and _ewoc_app_cur == EWOC_APP_BOTH:
            _ewoc_app_cur = EWOC_APP_OFF   # legacy "EWOC disabled" checkbox
        st.session_state["ewoc_application"] = _ewoc_app_cur

        st.session_state["wl_ewoc_application"] = _ewoc_app_cur
        st.selectbox(
            "EWOC application",
            options=EWOC_APP_OPTIONS,
            key="wl_ewoc_application",
            on_change=_sync_ewoc_application,
            help=h("ewoc_application",
                   "Choose whether EWOC is used during cohort-by-cohort dose "
                   "assignment, only when selecting the final MTD, or not at all. "
                   "EWOC restricts doses to those where BOTH P(tox1 OD) and "
                   "P(tox2 OD) < EWOC alpha.")
        )
        # Post-read immediately so ewoc_alpha disabled= sees the updated value;
        # keep the legacy ewoc_on boolean consistent for helper code.
        st.session_state["ewoc_application"] = st.session_state["wl_ewoc_application"]
        st.session_state["ewoc_on"] = (
            st.session_state["ewoc_application"] != EWOC_APP_OFF)

        # ── ewoc_alpha ────────────────────────────────────────────────────
        st.session_state["wl_ewoc_alpha"] = float(_cfg("ewoc_alpha"))
        st.number_input(
            "EWOC alpha",
            min_value=0.01, max_value=0.99, step=0.01, key="wl_ewoc_alpha",
            on_change=_sync_ewoc_alpha,
            disabled=(str(_cfg("ewoc_application")) == EWOC_APP_OFF),
            help=h("ewoc_alpha",
                   "EWOC threshold applied to both endpoints independently.")
        )
        st.session_state["ewoc_alpha"] = st.session_state["wl_ewoc_alpha"]

        # ── early_stop_on ─────────────────────────────────────────────────────
        st.session_state["wl_early_stop_on"] = bool(_cfg("early_stop_on"))
        st.toggle(
            "Enable early stopping",
            key="wl_early_stop_on",
            on_change=_sync_early_stop_on,
            help=h("early_stop_on",
                   "Stop enrolling as soon as the posterior probability that the "
                   "recommended dose is the optimal MTD exceeds the threshold below.")
        )
        st.session_state["early_stop_on"] = st.session_state["wl_early_stop_on"]

        # ── p_stop ────────────────────────────────────────────────────────────
        st.session_state["wl_p_stop"] = float(_cfg("p_stop"))
        st.number_input(
            "Early stopping confidence threshold (p_stop)",
            min_value=0.50, max_value=0.99, step=0.01, key="wl_p_stop",
            on_change=_sync_p_stop,
            disabled=(not bool(_cfg("early_stop_on"))),
            help=h("p_stop",
                   "Stop the trial when P(recommended dose is the MTD | data) "
                   "≥ this value. Active only when early stopping is enabled.",
                   r_name="p.stop")
        )
        st.session_state["p_stop"] = st.session_state["wl_p_stop"]

        st.markdown("#### CRM decision trace")

        # ── show_crm_trace ────────────────────────────────────────────────
        st.session_state["wl_show_crm_trace"] = bool(_cfg("show_crm_trace"))
        st.toggle(
            "Explain first CRM trial",
            key="wl_show_crm_trace",
            on_change=_sync_show_crm_trace,
            help=h("show_crm_trace",
                   "When ON, shows a detailed walkthrough for the first simulated "
                   "CRM trial only: which dose each patient received, what follow-up "
                   "data were available at each decision point, how the model judged "
                   "safety for each dose level, and why the next dose was chosen. "
                   "Has no effect on the summary results across all simulated trials.")
        )
        st.session_state["show_crm_trace"] = st.session_state["wl_show_crm_trace"]

    st.markdown("---")
    st.markdown("#### 6+3 stopping rules")
    st.markdown(
        '<div style="background:#1e3a5f;border-left:4px solid #4a9eff;'
        'padding:12px 16px;border-radius:0 4px 4px 0;margin:8px 0;">'
        '<span style="color:#d0e8ff;font-size:0.95em;">'
        'ℹ️ <strong>Modified 6+3 (TITE version) — full evaluability required.</strong>'
        '</span><br><br>'
        '<span style="color:#c0d8f0;font-size:0.91em;">'
        'Decisions are only made once ALL enrolled patients in the evaluation '
        'cohort have completed their relevant follow-up windows.'
        '</span><br><br>'
        '<span style="color:#c0d8f0;font-size:0.91em;">'
        '<strong style="color:#d0e8ff;">Bridging rule:</strong> '
        'while waiting for evaluability at the current dose, new arrivals are '
        'assigned to the next lower dose (<em>safe dose</em>). '
        'These bridging patients count toward the trial total but not toward '
        'the formal evaluation cohort.'
        '</span><br><br>'
        '<span style="color:#c0d8f0;font-size:0.91em;">'
        '<strong style="color:#d0e8ff;">Rate-based acute thresholds:</strong> '
        'if the HOLD rule causes more than 6 (or 9) patients to be enrolled at '
        'eval dose, the acute threshold is scaled proportionally to preserve '
        'the original protocol ratio.'
        '</span></div>',
        unsafe_allow_html=True,
        # replaced st.info() — icon parameter removed
    )
    st.markdown(
        "<div style='font-size:0.79rem;font-weight:600;color:#a0a0c0;'>"
        "Acute thresholds</div>",
        unsafe_allow_html=True,
    )
    _ar1, _ar2, _ar3 = st.columns(3, gap="small")
    with _ar1:
        st.session_state["wl_a6_esc_max"] = int(_cfg("a6_esc_max"))
        st.number_input("≥6 — esc if tox1 ≤", min_value=0, max_value=5,
                        step=1, key="wl_a6_esc_max", on_change=_sync_a6_esc_max,
                        help=h("a6_esc_max", "Phase 1 acute escalation threshold."))
        st.session_state["a6_esc_max"] = st.session_state["wl_a6_esc_max"]
    with _ar2:
        st.session_state["wl_a6_stop_min"] = int(_cfg("a6_stop_min"))
        st.number_input("≥6 — stop if tox1 ≥", min_value=1, max_value=6,
                        step=1, key="wl_a6_stop_min", on_change=_sync_a6_stop_min,
                        help=h("a6_stop_min", "Phase 1 acute stopping threshold."))
        st.session_state["a6_stop_min"] = st.session_state["wl_a6_stop_min"]
    with _ar3:
        st.session_state["wl_a9_esc_max"] = int(_cfg("a9_esc_max"))
        st.number_input("≥9 — esc if tox1 ≤", min_value=0, max_value=8,
                        step=1, key="wl_a9_esc_max", on_change=_sync_a9_esc_max,
                        help=h("a9_esc_max", "Phase 2 acute escalation threshold."))
        st.session_state["a9_esc_max"] = st.session_state["wl_a9_esc_max"]

    st.markdown(
        "<div style='font-size:0.79rem;font-weight:600;color:#a0a0c0;margin-top:0.3rem;'>"
        "Subacute thresholds</div>",
        unsafe_allow_html=True,
    )
    _sr1, _sr2, _sr3, _sr4 = st.columns(4, gap="small")
    with _sr1:
        st.session_state["wl_s6_esc_max"] = int(_cfg("s6_esc_max"))
        st.number_input("≥6 surg — esc if tox2 ≤", min_value=0, max_value=6,
                        step=1, key="wl_s6_esc_max", on_change=_sync_s6_esc_max,
                        help=h("s6_esc_max", "Phase 1 subacute escalation threshold."))
        st.session_state["s6_esc_max"] = st.session_state["wl_s6_esc_max"]
    with _sr2:
        st.session_state["wl_s6_stop_min"] = int(_cfg("s6_stop_min"))
        st.number_input("≥6 surg — stop if tox2 ≥", min_value=1, max_value=6,
                        step=1, key="wl_s6_stop_min", on_change=_sync_s6_stop_min,
                        help=h("s6_stop_min", "Phase 1 subacute stopping threshold."))
        st.session_state["s6_stop_min"] = st.session_state["wl_s6_stop_min"]
    with _sr3:
        st.session_state["wl_s9_esc_max"] = int(_cfg("s9_esc_max"))
        st.number_input("≥9 surg — esc if tox2 ≤", min_value=0, max_value=9,
                        step=1, key="wl_s9_esc_max", on_change=_sync_s9_esc_max,
                        help=h("s9_esc_max", "Phase 2 subacute escalation threshold."))
        st.session_state["s9_esc_max"] = st.session_state["wl_s9_esc_max"]
    with _sr4:
        st.session_state["wl_s9_stop_min"] = int(_cfg("s9_stop_min"))
        st.number_input("≥9 surg — stop if tox2 ≥", min_value=1, max_value=9,
                        step=1, key="wl_s9_stop_min", on_change=_sync_s9_stop_min,
                        help=h("s9_stop_min", "Phase 2 subacute stopping threshold."))
        st.session_state["s9_stop_min"] = st.session_state["wl_s9_stop_min"]

    st.write("")
    st.button("Reset to defaults", on_click=_do_reset, type="primary")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.78rem;font-weight:600;color:#a0a0c0;margin-top:0.3rem;'>"
        "Patient timeline (based on current timing settings)</div>",
        unsafe_allow_html=True,
    )
    _tl_fig = _draw_timeline(
        int(_cfg("incl_to_rt")),
        int(_cfg("rt_dur")),
        int(_cfg("rt_to_surg")),
        int(_cfg("tox2_win")),
    )
    st.image(fig_to_png_bytes(_tl_fig), use_container_width=True)

# ==============================================================================
# PLAYGROUND VIEW
# ==============================================================================

elif view == "Playground":
    left, mid, right = st.columns([1.00, 1.02, 1.12], gap="large")

    # ── Left: true probabilities ──────────────────────────────────────────────
    with left:
        st.markdown("#### True probabilities by dose")
        hL, hT1, hT2 = st.columns([0.33, 0.33, 0.33], gap="small")
        with hT1:
            st.markdown("<div style='font-size:0.79rem;font-weight:600;'>Tox1</div>",
                        unsafe_allow_html=True)
        with hT2:
            st.markdown("<div style='font-size:0.79rem;font-weight:600;'>Tox2</div>",
                        unsafe_allow_html=True)

        true_t1 = []
        true_t2 = []
        for i, lab in enumerate(dose_labels):
            rL, rT1, rT2 = st.columns([0.33, 0.33, 0.33], gap="small")
            with rL:
                st.markdown(
                    f"<div style='font-size:0.83rem;padding-top:0.25rem;'>L{i} {lab}</div>",
                    unsafe_allow_html=True)
            with rT1:
                # Pre-write from canonical key so the widget restores the user's
                # value after navigating away and back (Streamlit ≥1.28 clears
                # widget-backed keys when those widgets are not rendered).
                # TRUE_T1_KEYS[i] is the canonical (non-widget) key; its
                # value persists because it is written by the on_change callback,
                # not directly by the widget.
                _wk1 = f"wl_{TRUE_T1_KEYS[i]}"
                st.session_state[_wk1] = float(
                    st.session_state.get(TRUE_T1_KEYS[i], DEFAULT_TRUE_T1[i])
                )
                v1 = st.number_input(
                    f"T1 L{i}",
                    min_value=0.0, max_value=1.0, step=0.01,
                    key=_wk1,
                    on_change=_sync_true_t1[i],
                    label_visibility="collapsed",
                    help=f"True probability of acute toxicity at dose L{i}.",
                )
                st.session_state[TRUE_T1_KEYS[i]] = float(st.session_state[_wk1])
                true_t1.append(float(v1))
            with rT2:
                _wk2 = f"wl_{TRUE_T2_KEYS[i]}"
                st.session_state[_wk2] = float(
                    st.session_state.get(TRUE_T2_KEYS[i], DEFAULT_TRUE_T2[i])
                )
                v2 = st.number_input(
                    f"T2 L{i}",
                    min_value=0.0, max_value=1.0, step=0.01,
                    key=_wk2,
                    on_change=_sync_true_t2[i],
                    label_visibility="collapsed",
                    help=f"True probability of subacute toxicity given surgery at L{i}.",
                )
                st.session_state[TRUE_T2_KEYS[i]] = float(st.session_state[_wk2])
                true_t2.append(float(v2))

        target_t1_val = float(_cfg("target_t1"))
        target_t2_val = float(_cfg("target_t2"))
        p_surg_val    = float(_cfg("p_surgery"))
        true_safe = find_true_safe_dose(true_t1, true_t2, target_t1_val, target_t2_val)
        if true_safe is not None:
            st.caption(f"Highest jointly safe dose = L{true_safe}")
        else:
            st.caption("No dose satisfies both targets.")
        st.write("")
        run = st.button("Run simulations", type="primary", use_container_width=True)

    # ── Mid: Priors ───────────────────────────────────────────────────────────
    with mid:
        st.markdown("#### Priors")

        # ── Skeleton model ────────────────────────────────────────────────
        st.session_state["wl_prior_model"] = str(_cfg("prior_model"))
        st.radio(
            "Skeleton model",
            options=["empiric", "logistic"],
            horizontal=True, key="wl_prior_model",
            on_change=_sync_prior_model,
            help=h("prior_model", "Skeleton generation method, shared for both endpoints.")
        )
        st.session_state["prior_model"] = st.session_state["wl_prior_model"]
        prior_model_val = str(_cfg("prior_model"))
        intcpt_val      = float(_cfg("logistic_intcpt"))

        # ── Prior scenario selector ───────────────────────────────────────
        st.session_state["wl_prior_scenario"] = str(_cfg("prior_scenario"))
        st.selectbox(
            "Prior scenario",
            options=list(_PRIOR_SCENARIOS.keys()),
            key="wl_prior_scenario",
            on_change=_apply_prior_scenario,
            help=(
                "Choose a pre-configured prior belief about the dose-toxicity "
                "relationship. Select **Custom** to set parameters manually."
            ),
        )
        st.session_state["prior_scenario"] = st.session_state["wl_prior_scenario"]
        _scen_val = str(_cfg("prior_scenario"))

        # Description under the selector
        _scen_desc = _PRIOR_SCENARIOS.get(_scen_val, {}).get("description", "")
        if _scen_desc:
            st.caption(_scen_desc)

        # ── Scenario content ──────────────────────────────────────────────
        if _scen_val == "Custom":
            # Show the Endpoint tab and raw editable sliders
            st.session_state["wl_prior_ep_tab"] = str(_cfg("prior_ep_tab"))
            ep_tab = st.radio(
                "Endpoint",
                options=["Tox1 (acute)", "Tox2 (subacute | surgery)"],
                horizontal=True, key="wl_prior_ep_tab",
                on_change=_sync_prior_ep_tab,
                help="Switch between tox1 and tox2 prior parameter sets.",
            )
            st.session_state["prior_ep_tab"] = st.session_state["wl_prior_ep_tab"]

            if ep_tab == "Tox1 (acute)":
                # ── Prior target ──────────────────────────────────────────
                st.session_state["sl_prior_target_t1"] = float(_cfg("prior_target_t1"))
                st.slider("Prior target (tox1)", 0.05, 0.50, step=0.01,
                          key="sl_prior_target_t1",
                          on_change=_clamp_halfwidth_t1,
                          help=h("prior_target_t1",
                                 "Reference DLT probability at which the skeleton is "
                                 "anchored. Usually matches the Essentials target."))
                st.session_state["prior_target_t1"] = st.session_state["sl_prior_target_t1"]

                _pt1     = float(_cfg("prior_target_t1"))
                _max_hw1 = max(0.01, round(min(_pt1 - 0.01, 1.0 - _pt1 - 0.01), 2))

                # ── Halfwidth ──────────────────────────────────────────────
                _hw1_clamped = min(float(_cfg("halfwidth_t1")), _max_hw1)
                st.session_state["sl_halfwidth_t1"] = _hw1_clamped
                st.slider("Halfwidth (tox1)", 0.01, float(_max_hw1), step=0.01,
                          key="sl_halfwidth_t1",
                          on_change=_sync_halfwidth_t1,
                          help=h("halfwidth_t1",
                                 "Controls skeleton steepness. Larger = flatter curve, "
                                 "smaller = more peaked around the target level. "
                                 "target ± halfwidth must stay within (0, 1)."))
                st.session_state["halfwidth_t1"] = st.session_state["sl_halfwidth_t1"]

                # ── Prior MTD level ────────────────────────────────────────
                st.session_state["sl_prior_nu_t1"] = int(_cfg("prior_nu_t1"))
                st.slider("Prior MTD level (tox1)", 1, 5, step=1,
                          key="sl_prior_nu_t1",
                          on_change=_sync_prior_nu_t1,
                          help=h("prior_nu_t1",
                                 "Dose level that is a priori closest to the tox1 target. "
                                 "1 = most cautious (L0), 5 = most optimistic (L4)."))
                st.session_state["prior_nu_t1"] = st.session_state["sl_prior_nu_t1"]

            else:
                # ── Prior target (tox2) ────────────────────────────────────
                st.session_state["sl_prior_target_t2"] = float(_cfg("prior_target_t2"))
                st.slider("Prior target (tox2)", 0.05, 0.50, step=0.01,
                          key="sl_prior_target_t2",
                          on_change=_clamp_halfwidth_t2,
                          help=h("prior_target_t2",
                                 "Reference DLT probability for the subacute tox2 skeleton."))
                st.session_state["prior_target_t2"] = st.session_state["sl_prior_target_t2"]

                _pt2     = float(_cfg("prior_target_t2"))
                _max_hw2 = max(0.01, round(min(_pt2 - 0.01, 1.0 - _pt2 - 0.01), 2))

                # ── Halfwidth (tox2) ───────────────────────────────────────
                _hw2_clamped = min(float(_cfg("halfwidth_t2")), _max_hw2)
                st.session_state["sl_halfwidth_t2"] = _hw2_clamped
                st.slider("Halfwidth (tox2)", 0.01, float(_max_hw2), step=0.01,
                          key="sl_halfwidth_t2",
                          on_change=_sync_halfwidth_t2,
                          help=h("halfwidth_t2",
                                 "Controls tox2 skeleton steepness. "
                                 "target ± halfwidth must stay within (0, 1)."))
                st.session_state["halfwidth_t2"] = st.session_state["sl_halfwidth_t2"]

                # ── Prior MTD level (tox2) ─────────────────────────────────
                st.session_state["sl_prior_nu_t2"] = int(_cfg("prior_nu_t2"))
                st.slider("Prior MTD level (tox2)", 1, 5, step=1,
                          key="sl_prior_nu_t2",
                          on_change=_sync_prior_nu_t2,
                          help=h("prior_nu_t2",
                                 "Dose level a priori closest to the tox2 conditional target."))
                st.session_state["prior_nu_t2"] = st.session_state["sl_prior_nu_t2"]

        else:
            # ── Non-custom: show compact read-only summary ─────────────────
            _p = _PRIOR_SCENARIOS[_scen_val]
            _s1, _s2 = st.columns(2)
            with _s1:
                st.markdown(
                    "<span style='font-size:0.82em; color:#aac8e0; "
                    "font-weight:600;'>Tox1 (acute)</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:0.82em; line-height:1.65;'>"
                    f"Reference rate: <b>{_p['prior_target_t1']:.2f}</b><br>"
                    f"Halfwidth: <b>{_p['halfwidth_t1']:.2f}</b><br>"
                    f"Prior MTD level: <b>L{_p['prior_nu_t1']}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with _s2:
                st.markdown(
                    "<span style='font-size:0.82em; color:#aac8e0; "
                    "font-weight:600;'>Tox2 (subacute)</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:0.82em; line-height:1.65;'>"
                    f"Reference rate: <b>{_p['prior_target_t2']:.2f}</b><br>"
                    f"Halfwidth: <b>{_p['halfwidth_t2']:.2f}</b><br>"
                    f"Prior MTD level: <b>L{_p['prior_nu_t2']}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Refresh _max_hw1/2 for use in skeleton computation below (based on
        # the canonical values that were just updated by post-reads above).
        _pt1     = float(_cfg("prior_target_t1"))
        _max_hw1 = max(0.01, round(min(_pt1 - 0.01, 1.0 - _pt1 - 0.01), 2))
        _pt2     = float(_cfg("prior_target_t2"))
        _max_hw2 = max(0.01, round(min(_pt2 - 0.01, 1.0 - _pt2 - 0.01), 2))

        # Compute skeletons for preview and simulation.
        # Pre-clamp hw_eff to guarantee dfcrm_getprior receives a strictly
        # valid halfwidth even when session_state diverges from slider display
        # (e.g. during the render cycle after a force-reset).
        _t1 = float(_cfg("prior_target_t1"))
        _safe_hw1 = max(1e-4, min(_t1 - 1e-4, 1.0 - _t1 - 1e-4))
        hw1_eff = min(float(_cfg("halfwidth_t1")), _safe_hw1)
        try:
            skel_t1 = dfcrm_getprior(
                halfwidth=hw1_eff, target=_t1,
                nu=int(_cfg("prior_nu_t1")),
                nlevel=5, model=prior_model_val, intcpt=intcpt_val,
            ).tolist()
        except ValueError as e:
            st.warning(f"Tox1 skeleton: {e}")
            hw1_eff = min(0.10, _max_hw1)
            skel_t1 = dfcrm_getprior(
                halfwidth=hw1_eff, target=_t1,
                nu=int(_cfg("prior_nu_t1")),
                nlevel=5, model=prior_model_val, intcpt=intcpt_val,
            ).tolist()

        _t2 = float(_cfg("prior_target_t2"))
        _safe_hw2 = max(1e-4, min(_t2 - 1e-4, 1.0 - _t2 - 1e-4))
        hw2_eff = min(float(_cfg("halfwidth_t2")), _safe_hw2)
        try:
            skel_t2 = dfcrm_getprior(
                halfwidth=hw2_eff, target=_t2,
                nu=int(_cfg("prior_nu_t2")),
                nlevel=5, model=prior_model_val, intcpt=intcpt_val,
            ).tolist()
        except ValueError as e:
            st.warning(f"Tox2 skeleton: {e}")
            hw2_eff = min(0.10, _max_hw2)
            skel_t2 = dfcrm_getprior(
                halfwidth=hw2_eff, target=_t2,
                nu=int(_cfg("prior_nu_t2")),
                nlevel=5, model=prior_model_val, intcpt=intcpt_val,
            ).tolist()

    # ── Right: dose-risk preview ──────────────────────────────────────────────
    with right:
        st.markdown("#### Dose-risk preview")

        fig, (ax1, ax2) = plt.subplots(2, 1,
                                       figsize=(PREVIEW_W_IN, PREVIEW_H_IN),
                                       dpi=PREVIEW_DPI)
        _apply_dark_fig(fig, ax1, ax2)
        x = np.arange(5); bw = 0.38

        ax1.bar(x - bw/2, true_t1, bw, color="#4a9eff", label="True tox1")
        ax1.bar(x + bw/2, skel_t1, bw, color="#4a9eff", alpha=0.5, label="Skel tox1", hatch="//")
        ax1.axhline(target_t1_val, lw=1.2, alpha=0.85, color="#80c0ff", ls="--")
        ax1.set_ylabel("P(tox1)", fontsize=10, color=_DARK_FG)
        ax1.set_xticks(x); ax1.set_xticklabels([f"L{i}" for i in range(5)], fontsize=10)
        _y1 = max(max(true_t1), max(skel_t1), target_t1_val)
        ax1.set_ylim(0, min(1.0, _y1 * 1.3 + 0.02))
        ax1.legend(fontsize=10, frameon=True, loc="upper left", labelcolor=_DARK_FG,
                   facecolor=_DARK_AX, edgecolor=_DARK_GRD)
        compact_style(ax1)

        ax2.bar(x - bw/2, true_t2, bw, color="#ffaa44", label="True tox2")
        ax2.bar(x + bw/2, skel_t2, bw, color="#ffaa44", alpha=0.5, label="Skel tox2", hatch="//")
        ax2.axhline(target_t2_val, lw=1.2, alpha=0.85, color="#ffd080", ls="--")
        ax2.set_ylabel("P(tox2)", fontsize=10, color=_DARK_FG)
        ax2.set_xticks(x); ax2.set_xticklabels([f"L{i}" for i in range(5)], fontsize=10)
        _y2 = max(max(true_t2), max(skel_t2), target_t2_val)
        ax2.set_ylim(0, min(1.0, _y2 * 1.25 + 0.02))
        ax2.legend(fontsize=10, frameon=True, loc="upper left", labelcolor=_DARK_FG,
                   facecolor=_DARK_AX, edgecolor=_DARK_GRD)
        compact_style(ax2)

        fig.tight_layout(pad=0.5)
        st.image(fig_to_png_bytes(fig), width=int(_cfg("preview_w_px")))

    # ==============================================================================
    # Run simulations
    # ==============================================================================

    if run:
        rng_master = np.random.default_rng(int(_cfg("seed")))
        ns         = int(_cfg("n_sims"))
        start_0b   = int(np.clip(int(_cfg("start_level_1b")), 0, len(true_t1) - 1))

        sel_63  = np.zeros(5, dtype=int)
        sel_crm = np.zeros(5, dtype=int)

        nmat_63  = np.zeros((ns, 5), dtype=float)
        nmat_crm = np.zeros((ns, 5), dtype=float)
        nsurg_63  = np.zeros((ns, 5), dtype=float)
        nsurg_crm = np.zeros((ns, 5), dtype=float)

        ya63  = np.zeros(ns); ys63  = np.zeros(ns); ns63  = np.zeros(ns)
        yacrm = np.zeros(ns); yscrm = np.zeros(ns); nscrm = np.zeros(ns)

        dur_63  = np.zeros(ns)
        dur_crm = np.zeros(ns)
        nbridg  = np.zeros(ns, dtype=int)
        early_stop_crm  = np.zeros(ns, dtype=bool)
        n_at_stop_crm   = np.zeros(ns, dtype=int)
        sel_crm_per_trial = np.zeros(ns, dtype=int)
        mtd_support_arr   = np.zeros(ns, dtype=float)   # posterior support for selected MTD
        early_dlt1_6  = np.zeros(ns)
        early_dlt1_9  = np.zeros(ns)
        early_dlt1_12 = np.zeros(ns)

        # tox1_win is derived: extends from RT start all the way to surgery
        # = RT duration + (RT end → surgery) — no separate UI input needed
        _tox1_win_derived = int(_cfg("rt_dur")) + int(_cfg("rt_to_surg"))

        timing_kw = dict(
            accrual_per_month = float(_cfg("accrual_per_month")),
            incl_to_rt        = int(_cfg("incl_to_rt")),
            rt_dur            = int(_cfg("rt_dur")),
            rt_to_surg        = int(_cfg("rt_to_surg")),
            tox1_win          = _tox1_win_derived,
            tox2_win          = int(_cfg("tox2_win")),
        )

        for s in range(ns):
            rng_s = np.random.default_rng(rng_master.integers(0, 2**31))

            # ── TITE 6+3 ─────────────────────────────────────────────────────
            sel63, pts63, sd63, nb63 = run_tite_6plus3(
                true_t1=true_t1, p_surgery=p_surg_val, true_t2=true_t2,
                start_level=start_0b,
                max_n=int(_cfg("max_n_63")),
                a6_esc_max  = int(_cfg("a6_esc_max")),
                a6_stop_min = int(_cfg("a6_stop_min")),
                a9_esc_max  = int(_cfg("a9_esc_max")),
                s6_esc_max  = int(_cfg("s6_esc_max")),
                s6_stop_min = int(_cfg("s6_stop_min")),
                s9_esc_max  = int(_cfg("s9_esc_max")),
                s9_stop_min = int(_cfg("s9_stop_min")),
                rng=rng_s, **timing_kw,
            )
            sel_63[sel63] += 1
            for p in pts63:
                nmat_63[s, p["dose"]]  += 1
                nsurg_63[s, p["dose"]] += int(p["has_surgery"])
                ya63[s]  += int(p["has_tox1"])
                ys63[s]  += int(p["has_tox2"])
                ns63[s]  += int(p["has_surgery"])
            dur_63[s]  = sd63
            nbridg[s]  = nb63

            # ── TITE-CRM ─────────────────────────────────────────────────────
            rng_s2 = np.random.default_rng(rng_master.integers(0, 2**31))
            selc, ptsc, sdc, trace_s, _stopped_c = run_tite_crm(
                true_t1=true_t1, p_surgery=p_surg_val, true_t2=true_t2,
                target1=target_t1_val, target2=target_t2_val,
                skel1=skel_t1, skel2=skel_t2,
                sigma        = float(_cfg("sigma")),
                start_level  = start_0b,
                max_n        = int(_cfg("max_n_crm")),
                cohort_size  = int(_cfg("cohort_size")),
                max_step     = int(_cfg("max_step")),
                gh_n         = int(_cfg("gh_n")),
                enforce_guardrail      = bool(_cfg("enforce_guardrail")),
                restrict_final_to_tried= bool(_cfg("restrict_final_mtd")),
                ewoc_on      = bool(_cfg("ewoc_on")),
                ewoc_alpha   = float(_cfg("ewoc_alpha")),
                ewoc_application = str(_cfg("ewoc_application")),
                burn_in      = bool(_cfg("burn_in")),
                require_full_tox1_fu_before_escalation = bool(_cfg("require_full_tox1_fu")),
                n_safe_d1    = int(_cfg("n_safe_d1")),
                p_stop       = (float(_cfg("p_stop"))
                                if bool(_cfg("early_stop_on")) else 1.0),
                rng=rng_s2, **timing_kw,
                collect_trace=(s == 0),   # record full trace for first trial only
            )
            early_stop_crm[s] = _stopped_c
            n_at_stop_crm[s]  = len(ptsc)
            # Save first-trial trace for the decision walkthrough display
            if s == 0:
                _crm_trace_first = {
                    "patients":   ptsc,
                    "decisions":  trace_s,
                    "true_t1":    list(true_t1),
                    "true_t2":    list(true_t2),
                    "tox1_win":   _tox1_win_derived,
                    "tox2_win":   int(_cfg("tox2_win")),
                    # Final MTD for this trial and CRM parameters needed to
                    # re-compute posterior summaries in the UI explanation table.
                    "final_mtd":  selc,
                    "study_days": sdc,
                    "sigma":      float(_cfg("sigma")),
                    "skel_t1":    list(skel_t1),
                    "skel_t2":    list(skel_t2),
                    "target_t1":  float(_cfg("target_t1")),
                    "target_t2":  float(_cfg("target_t2")),
                    "gh_n":       int(_cfg("gh_n")),
                    "ewoc_on":    bool(_cfg("ewoc_on")),
                    "ewoc_alpha": float(_cfg("ewoc_alpha")),
                    "ewoc_application": str(_cfg("ewoc_application")),
                    "restrict_final_mtd": bool(_cfg("restrict_final_mtd")),
                }
            sel_crm[selc] += 1
            sel_crm_per_trial[s] = selc

            # Compute posterior support for the selected MTD using final follow-up weights.
            # Uses ewoc_final_eff — the same alpha crm_select_mtd() applied for this trial.
            _n1f_s, _y1f_s, _n2f_s, _y2f_s = tite_weights(
                ptsc, sdc, _tox1_win_derived, int(_cfg("tox2_win")), len(true_t1))
            _, _ewoc_final_eff_s = ewoc_effective_alphas(
                str(_cfg("ewoc_application")), float(_cfg("ewoc_alpha")),
                ewoc_on=bool(_cfg("ewoc_on")))
            _supp_probs = crm_mtd_posterior_probs(
                float(_cfg("sigma")), skel_t1, skel_t2,
                _n1f_s, _y1f_s, _n2f_s, _y2f_s,
                target_t1_val, target_t2_val,
                ewoc_alpha=_ewoc_final_eff_s,
                gh_n=int(_cfg("gh_n")),
                restrict_to_tried=bool(_cfg("restrict_final_mtd")),
            )
            mtd_support_arr[s] = float(_supp_probs[selc])

            for p in ptsc:
                nmat_crm[s, p["dose"]]  += 1
                nsurg_crm[s, p["dose"]] += int(p["has_surgery"])
                yacrm[s]  += int(p["has_tox1"])
                yscrm[s]  += int(p["has_tox2"])
                nscrm[s]  += int(p["has_surgery"])
            dur_crm[s] = sdc
            for _k, _earr in [(6, early_dlt1_6), (9, early_dlt1_9), (12, early_dlt1_12)]:
                _earr[s] = float(sum(int(p["has_tox1"]) for p in ptsc[:_k]))

        # Store results
        p63   = sel_63  / ns
        pcrm  = sel_crm / ns
        st.session_state["_tite_results"] = {
            "p63":  p63, "pcrm": pcrm,
            "avg_n63":      nmat_63.mean(axis=0),
            "avg_ncrm":     nmat_crm.mean(axis=0),
            "avg_nsurg63":  nsurg_63.mean(axis=0),
            "avg_nsurgcrm": nsurg_crm.mean(axis=0),
            "acute_rate_63":  ya63.sum()  / max(1, nmat_63.sum()),
            "acute_rate_crm": yacrm.sum() / max(1, nmat_crm.sum()),
            "sub_gs_rate_63":  ys63.sum()  / max(1, ns63.sum()),
            "sub_gs_rate_crm": yscrm.sum() / max(1, nscrm.sum()),
            "surg_rate_63":   ns63.mean()  / max(1, nmat_63.sum() / ns),
            "surg_rate_crm":  nscrm.mean() / max(1, nmat_crm.sum() / ns),
            "dur63_mean":  dur_63.mean()  / MONTH,
            "dur63_med":   float(np.median(dur_63))  / MONTH,
            "durcrm_mean": dur_crm.mean() / MONTH,
            "durcrm_med":  float(np.median(dur_crm)) / MONTH,
            "avg_bridging": float(nbridg.mean()),
            "true_safe": true_safe,
            "ns": ns,
            "seed": int(_cfg("seed")),
            "p_surgery": p_surg_val,
            "crm_trace": _crm_trace_first,   # first-trial trace for walkthrough
            "early_stop_on":    bool(_cfg("early_stop_on")),
            "p_stop":           float(_cfg("p_stop")),
            "crm_early_stop_pct": float(early_stop_crm.mean()),
            "crm_early_stop_n_mean": (
                float(n_at_stop_crm[early_stop_crm].mean())
                if early_stop_crm.any() else None
            ),
            "n_per_trial_crm":  n_at_stop_crm.tolist(),
            "early_stop_arr":   early_stop_crm.tolist(),
            "sel_crm_per_trial":  sel_crm_per_trial.tolist(),
            "mtd_support_arr":    mtd_support_arr.tolist(),
            "early_dlt1_6":     early_dlt1_6.tolist(),
            "early_dlt1_9":     early_dlt1_9.tolist(),
            "early_dlt1_12":    early_dlt1_12.tolist(),
            "nmat_crm_raw":     nmat_crm.tolist(),
            "yacrm_raw":        yacrm.tolist(),
            "yscrm_raw":        yscrm.tolist(),
            "dur_crm_raw":      dur_crm.tolist(),
        }

# ==============================================================================
# Results
# ==============================================================================

if view == "Playground" and "_tite_results" in st.session_state:
    res = st.session_state["_tite_results"]
    p63  = res["p63"]
    pcrm = res["pcrm"]
    ts   = res["true_safe"]

    st.write("")
    r1, r2, r3 = st.columns([1.05, 1.05, 0.90], gap="large")

    with r1:
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
        if ts is not None:
            ax.axvline(ts, lw=1, alpha=0.6, color="#80ff80", label=f"True safe=L{ts}")
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, labelcolor=_DARK_FG)
        fig.tight_layout(pad=0.4)
        st.image(fig_to_png_bytes(fig), use_container_width=True)

    with r2:
        fig, (ax_n, ax_s) = plt.subplots(2, 1,
                                          figsize=(RESULT_W_IN, RESULT_H_IN),
                                          dpi=RESULT_DPI)
        _apply_dark_fig(fig, ax_n, ax_s)
        xx = np.arange(5); w = 0.38
        ax_n.bar(xx - w/2, res["avg_n63"],    w, color="#4a9eff", label="6+3")
        ax_n.bar(xx + w/2, res["avg_ncrm"],   w, color="#ffaa44", label="CRM")
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
        st.image(fig_to_png_bytes(fig), use_container_width=True)

    with r3:
        mc1, mc2 = st.columns(2, gap="small")
        with mc1:
            st.metric("Tox1/pt (6+3)",
                      f"{res['acute_rate_63']:.3f}",
                      help="Acute DLT rate per treated patient — TITE 6+3")
            st.metric("Tox1/pt (CRM)",
                      f"{res['acute_rate_crm']:.3f}",
                      help="Acute DLT rate per treated patient — TITE-CRM")
            st.metric("Tox2/surg (6+3)",
                      f"{res['sub_gs_rate_63']:.3f}",
                      help="Subacute DLT rate per surgery-evaluable patient — TITE 6+3")
            st.metric("Tox2/surg (CRM)",
                      f"{res['sub_gs_rate_crm']:.3f}",
                      help="Subacute DLT rate per surgery-evaluable patient — TITE-CRM")
        with mc2:
            st.metric("Duration mean (6+3)",
                      f"{res['dur63_mean']:.1f} mo",
                      help="Mean trial duration from first patient in to last follow-up complete — TITE 6+3")
            st.metric("Duration mean (CRM)",
                      f"{res['durcrm_mean']:.1f} mo",
                      help="Mean trial duration — TITE-CRM")
            st.metric("Duration median (6+3)",
                      f"{res['dur63_med']:.1f} mo",
                      help="Median trial duration — TITE 6+3")
            st.metric("Duration median (CRM)",
                      f"{res['durcrm_med']:.1f} mo",
                      help="Median trial duration — TITE-CRM")
        st.metric("Avg bridging pts (6+3)",
                  f"{res['avg_bridging']:.1f}",
                  help="Average number of patients treated at a lower bridging dose "
                       "per trial while the 6+3 arm awaited full evaluability.")

        if res["early_stop_on"]:
            _es_pct = res["crm_early_stop_pct"]
            _es_n   = res["crm_early_stop_n_mean"]
            st.metric(
                "CRM early-stop rate",
                f"{_es_pct:.1%}",
                help=f"Proportion of CRM trials that stopped early "
                     f"(p_stop threshold = {res['p_stop']:.2f}).",
            )
            st.metric(
                "CRM avg N at early stop",
                f"{_es_n:.1f}" if _es_n is not None else "—",
                help="Mean number of patients enrolled in trials that triggered "
                     "early stopping.",
            )

        st.caption(
            f"Surgery rate: 6+3={res['surg_rate_63']:.3f}  "
            f"CRM={res['surg_rate_crm']:.3f}  (expected ≈ {res['p_surgery']:.2f})"
        )
        st.caption(
            f"n_sims={res['ns']} | seed={res['seed']}"
            + (f" | True safe=L{ts}" if ts is not None else " | No jointly safe dose")
        )

    # ── Posterior support for selected MTD ─────────────────────────────────────
    if "mtd_support_arr" in res:
        _msa = np.array(res["mtd_support_arr"], dtype=float)
        _msa_sel = np.array(res["sel_crm_per_trial"], dtype=int)

        st.markdown("---")
        st.subheader("Posterior support for selected MTD")
        st.caption(
            "Posterior support for selected MTD shows how strongly the final CRM model "
            "supported the dose it selected at the end of each simulated trial. "
            "A higher value means the selected dose was clearly favoured by the final "
            "posterior. A lower value means the model selected an MTD, but the posterior "
            "still spread meaningful probability across other dose levels."
        )

        _ps_c1, _ps_c2 = st.columns(2, gap="large")

        with _ps_c1:
            # Histogram of posterior support values
            fig_ps, ax_ps = plt.subplots(figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
            _apply_dark_fig(fig_ps, ax_ps)
            _bins = np.linspace(0, 1, 21)
            ax_ps.hist(_msa, bins=_bins, color="#ffaa44", edgecolor=_DARK_BG, linewidth=0.4)
            ax_ps.set_xlabel("Posterior support", fontsize=9)
            ax_ps.set_ylabel("Simulated trials", fontsize=9)
            ax_ps.set_xlim(0, 1)
            ax_ps.set_title("Posterior support for selected MTD", fontsize=10)
            compact_style(ax_ps)
            fig_ps.tight_layout(pad=0.4)
            st.image(fig_to_png_bytes(fig_ps), use_container_width=True)
            plt.close(fig_ps)

            # Summary metrics
            _med_supp  = float(np.median(_msa))
            _q25, _q75 = float(np.percentile(_msa, 25)), float(np.percentile(_msa, 75))
            _pct50 = float(np.mean(_msa >= 0.50)) * 100
            _pct70 = float(np.mean(_msa >= 0.70)) * 100
            _pct80 = float(np.mean(_msa >= 0.80)) * 100
            _ms1, _ms2 = st.columns(2, gap="small")
            with _ms1:
                st.metric("Median support",
                          f"{_med_supp:.2f}",
                          help="Median posterior support for selected MTD across all simulated trials.")
                st.metric("IQR",
                          f"{_q25:.2f} – {_q75:.2f}",
                          help="Interquartile range of posterior support (25th–75th percentile).")
            with _ms2:
                st.metric("≥ 0.50",
                          f"{_pct50:.1f}%",
                          help="Percentage of trials where posterior support for selected MTD ≥ 0.50.")
                st.metric("≥ 0.70",
                          f"{_pct70:.1f}%",
                          help="Percentage of trials where posterior support for selected MTD ≥ 0.70.")
                st.metric("≥ 0.80",
                          f"{_pct80:.1f}%",
                          help="Percentage of trials where posterior support for selected MTD ≥ 0.80.")

        with _ps_c2:
            # Box plot: MTD certainty grouped by selected MTD level
            _dose_lbls_ps = [f"L{i}" for i in range(5)]
            _groups = [_msa[_msa_sel == d] for d in range(5)]
            _groups_nonempty = [(lbl, g) for lbl, g in zip(_dose_lbls_ps, _groups) if len(g) > 0]

            fig_bp, ax_bp = plt.subplots(figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
            _apply_dark_fig(fig_bp, ax_bp)
            if _groups_nonempty:
                _bp_lbls   = [lbl for lbl, _ in _groups_nonempty]
                _bp_data   = [g   for _, g   in _groups_nonempty]
                _ps_dose_colors = ["#4a9eff", "#ffaa44", "#ff6677", "#55dd99", "#cc88ff"]
                _bp_colors = [_ps_dose_colors[i] for i, (lbl, g)
                              in enumerate(zip(_dose_lbls_ps, _groups))
                              if len(g) > 0]
                bp = ax_bp.boxplot(
                    _bp_data,
                    patch_artist=True,
                    medianprops=dict(color="#ffffff", linewidth=1.5),
                    whiskerprops=dict(color=_DARK_FG),
                    capprops=dict(color=_DARK_FG),
                    flierprops=dict(marker="o", markersize=2,
                                   markerfacecolor=_DARK_FG, alpha=0.4),
                )
                for patch, col in zip(bp["boxes"], _bp_colors):
                    patch.set_facecolor(col)
                    patch.set_alpha(0.75)
                ax_bp.set_xticks(range(1, len(_bp_lbls) + 1))
                ax_bp.set_xticklabels(_bp_lbls, fontsize=9)
            ax_bp.set_ylim(0, 1.05)
            ax_bp.set_xlabel("Selected MTD", fontsize=9)
            ax_bp.set_ylabel("Posterior support", fontsize=9)
            ax_bp.set_title("MTD certainty by selected MTD", fontsize=10)
            compact_style(ax_bp)
            fig_bp.tight_layout(pad=0.4)
            st.image(fig_to_png_bytes(fig_bp), use_container_width=True)
            plt.close(fig_bp)

# ==============================================================================
# CRM dichotomy diagnostic helpers
# ==============================================================================

def _crm_bimodality(pcrm):
    """Detect peaks and valleys in CRM selection-probability distribution.

    Returns
    -------
    dict with keys:
      peaks             : list[(dose_idx, prob)] sorted by prob desc
      valleys           : list[(dose_idx, prob)]
      second_peak_prob  : float  — probability of 2nd highest peak (0 if <2 peaks)
      valley_depth      : float  — (p2 − min_between) / p2, 0 if no valley between peaks
      peak_gap          : int    — |dose_idx_1 − dose_idx_2| for two highest peaks
    """
    n = len(pcrm)
    peaks, valleys = [], []
    for i in range(n):
        lv = pcrm[i - 1] if i > 0 else -1.0
        rv = pcrm[i + 1] if i < n - 1 else -1.0
        if pcrm[i] >= lv and pcrm[i] >= rv:
            peaks.append((i, float(pcrm[i])))
        if i > 0 and i < n - 1 and pcrm[i] <= lv and pcrm[i] <= rv:
            valleys.append((i, float(pcrm[i])))
    peaks.sort(key=lambda t: -t[1])

    second_peak_prob = 0.0
    valley_depth     = 0.0
    peak_gap         = 0
    if len(peaks) >= 2:
        p1_idx = peaks[0][0]
        p2_idx = peaks[1][0]
        p2_val = peaks[1][1]
        second_peak_prob = p2_val
        peak_gap = abs(p1_idx - p2_idx)
        lo, hi = sorted([p1_idx, p2_idx])
        min_between = float(min(pcrm[lo:hi + 1]))
        valley_depth = (p2_val - min_between) / p2_val if p2_val > 1e-9 else 0.0

    return {
        "peaks":            peaks,
        "valleys":          valleys,
        "second_peak_prob": second_peak_prob,
        "valley_depth":     valley_depth,
        "peak_gap":         peak_gap,
    }


def _plot_dichotomy_selection(pcrm, bim, dose_labels_list):
    """Annotated bar chart of CRM selection probabilities.

    Peaks coloured green, identified valleys red, others orange.
    """
    fig, ax = plt.subplots(figsize=(7, 3.0))
    _apply_dark_fig(fig, ax)
    x = np.arange(len(pcrm))
    peak_idxs   = {t[0] for t in bim["peaks"]}
    valley_idxs = {t[0] for t in bim["valleys"]}
    colors = [
        "#44dd88" if i in peak_idxs else "#ff6666" if i in valley_idxs else "#ffaa44"
        for i in range(len(pcrm))
    ]
    ax.bar(x, pcrm, color=colors, width=0.6)
    for i, p in enumerate(pcrm):
        if p > 0.005:
            ax.text(i, p + 0.012, f"{p:.0%}", ha="center", fontsize=8,
                    color=_DARK_FG, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dose_labels_list, fontsize=9)
    ax.set_ylabel("Selection probability", color=_DARK_FG, fontsize=8)
    ax.set_ylim(0, min(1.0, max(pcrm) * 1.25 + 0.08))
    ax.set_title("CRM dose selection", fontsize=9, color=_DARK_FG)
    ax.tick_params(colors=_DARK_FG, labelsize=8)
    compact_style(ax)
    fig.tight_layout(pad=1.0)
    return fig


def _plot_pathway_bars(groups, metric_label, values_by_group, colors_by_group):
    """Horizontal bar chart comparing a metric across Low/Middle/High pathways."""
    fig, ax = plt.subplots(figsize=(7, 2.4))
    _apply_dark_fig(fig, ax)
    y = np.arange(len(groups))
    bars = ax.barh(y, values_by_group, color=colors_by_group, height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_xlabel(metric_label, color=_DARK_FG, fontsize=8)
    ax.tick_params(colors=_DARK_FG, labelsize=8)
    for bar, val in zip(bars, values_by_group):
        if val > 0:
            ax.text(bar.get_width() + 0.02 * max(values_by_group),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8, color=_DARK_FG)
    compact_style(ax)
    fig.tight_layout(pad=1.0)
    return fig


def _quality_score(selected, true_t1, true_t2, target1, target2):
    """Asymmetric exponential loss: penalises overdose more than underdose."""
    d1 = float(true_t1[selected]) - float(target1)
    d2 = float(true_t2[selected]) - float(target2)
    bd = max(d1, d2)
    w  = 1.8 if bd > 0 else 1.0
    return float(np.exp(-6.0 * w * abs(bd)))


def _true_optimal(true_t1, true_t2, target1, target2):
    """Dose with highest quality score under the same asymmetric loss."""
    scores = [_quality_score(d, true_t1, true_t2, target1, target2)
              for d in range(len(true_t1))]
    return int(np.argmax(scores))


# ==============================================================================
# CRM Dichotomy Diagnostic
# ==============================================================================

if view == "Playground" and "_tite_results" in st.session_state and "sel_crm_per_trial" in st.session_state["_tite_results"]:
    _dd = st.session_state["_tite_results"]
    _dd_pcrm  = np.array(_dd["pcrm"])
    _dd_ns    = int(_dd["ns"])
    _dd_sel   = np.array(_dd["sel_crm_per_trial"], dtype=int)
    _dd_ya    = np.array(_dd["yacrm_raw"])
    _dd_ys    = np.array(_dd["yscrm_raw"])
    _dd_dur   = np.array(_dd["dur_crm_raw"])
    _dd_n     = np.array(_dd["n_per_trial_crm"], dtype=int)
    _dd_ed6   = np.array(_dd["early_dlt1_6"])
    _dd_ed9   = np.array(_dd["early_dlt1_9"])
    _dd_ed12  = np.array(_dd["early_dlt1_12"])
    _dd_nmat  = np.array(_dd["nmat_crm_raw"])          # shape (ns, 5)
    _dd_dl    = dose_labels                             # ["5×4 Gy", …]
    _dd_bim   = _crm_bimodality(_dd_pcrm)

    st.markdown("---")
    st.subheader("CRM Dichotomy Diagnostic")
    st.caption(
        "Two peaks can occur when simulated trials split into different paths. "
        "Some trials see early toxicity and the model selects a lower dose; "
        "others see little or late toxicity and keep higher doses admissible. "
        "EWOC may amplify this pattern when the final rule selects the highest "
        "admissible dose."
    )

    # ── 1. Bimodality metrics ─────────────────────────────────────────────────
    _bm_c1, _bm_c2, _bm_c3, _bm_c4 = st.columns(4)
    _bm_c1.metric(
        "Peaks detected",
        str(len(_dd_bim["peaks"])),
        help="Number of local maxima in the CRM selection-probability distribution.",
    )
    _bm_c2.metric(
        "2nd peak probability",
        f"{_dd_bim['second_peak_prob']:.1%}",
        help="Selection probability at the second-largest peak (0 % if only one peak).",
    )
    _bm_c3.metric(
        "Valley depth",
        f"{_dd_bim['valley_depth']:.1%}",
        help=(
            "How deep the dip is between the two highest peaks, relative to the "
            "smaller peak: (p2 − min_between) / p2.  "
            "0 % = no valley; 100 % = valley drops to zero."
        ),
    )
    _bm_c4.metric(
        "Peak gap (dose levels)",
        str(_dd_bim["peak_gap"]),
        help="Distance in dose levels between the two highest peaks.",
    )

    _bm_fig = _plot_dichotomy_selection(_dd_pcrm, _dd_bim, _dd_dl)
    st.image(fig_to_png_bytes(_bm_fig), use_container_width=False, width=560)
    plt.close(_bm_fig)

    # ── 2. Trial stratification by final selected MTD ─────────────────────────
    st.markdown("#### Trial stratification by final selected MTD")
    _strat_rows = []
    for _di in range(5):
        _mask = _dd_sel == _di
        _cnt  = int(_mask.sum())
        if _cnt == 0:
            continue
        _strat_rows.append({
            "MTD selected":        _dd_dl[_di],
            "Trials (n)":          _cnt,
            "Trials (%)":          f"{100 * _cnt / _dd_ns:.1f}",
            "Mean tox1 DLTs":      f"{_dd_ya[_mask].mean():.2f}",
            "Mean tox2 DLTs":      f"{_dd_ys[_mask].mean():.2f}",
            "Mean N enrolled":     f"{_dd_n[_mask].mean():.1f}",
            "Mean duration (mo)":  f"{(_dd_dur[_mask].mean() / MONTH):.1f}",
            "Early tox1 DLTs (6)": f"{_dd_ed6[_mask].mean():.2f}",
            "Early tox1 DLTs (9)": f"{_dd_ed9[_mask].mean():.2f}",
            "Early tox1 DLTs(12)": f"{_dd_ed12[_mask].mean():.2f}",
        })
    if _strat_rows:
        st.dataframe(pd.DataFrame(_strat_rows), hide_index=True,
                     use_container_width=True)

    # ── 3. Low / Middle / High pathway grouping ───────────────────────────────
    st.markdown("#### Low / Middle / High pathway grouping")
    st.caption("Low = MTD ≤ L2 (1-indexed) · Middle = L3 · High = MTD ≥ L4")
    _pw_groups  = ["Low (≤L2)", "Middle (L3)", "High (≥L4)"]
    _pw_masks   = [_dd_sel <= 1, _dd_sel == 2, _dd_sel >= 3]
    _pw_colors  = ["#4a9eff", "#ffaa44", "#ff6666"]
    _pw_rows = []
    for _pg, _pm in zip(_pw_groups, _pw_masks):
        _pc = int(_pm.sum())
        if _pc == 0:
            _pw_rows.append({"Pathway": _pg, "Trials (n)": 0,
                             "Trials (%)": "0.0",
                             "Mean tox1 DLTs": "—",
                             "Mean early tox1 (6)": "—",
                             "Mean N enrolled": "—",
                             "Mean duration (mo)": "—"})
        else:
            _pw_rows.append({
                "Pathway":            _pg,
                "Trials (n)":         _pc,
                "Trials (%)":         f"{100 * _pc / _dd_ns:.1f}",
                "Mean tox1 DLTs":     f"{_dd_ya[_pm].mean():.2f}",
                "Mean early tox1 (6)": f"{_dd_ed6[_pm].mean():.2f}",
                "Mean N enrolled":    f"{_dd_n[_pm].mean():.1f}",
                "Mean duration (mo)": f"{(_dd_dur[_pm].mean() / MONTH):.1f}",
            })
    st.dataframe(pd.DataFrame(_pw_rows), hide_index=True, use_container_width=True)

    _pw_counts = [int(m.sum()) for m in _pw_masks]
    _pw_fig = _plot_pathway_bars(
        _pw_groups, "Mean early tox1 DLTs (first 6 patients)",
        [(_dd_ed6[m].mean() if m.sum() > 0 else 0.0) for m in _pw_masks],
        _pw_colors,
    )
    st.image(fig_to_png_bytes(_pw_fig), use_container_width=False, width=560)
    plt.close(_pw_fig)

    # ── 4. EWOC sensitivity check ─────────────────────────────────────────────
    st.markdown("#### EWOC sensitivity check")
    _ewoc_btn = st.button(
        "Compare EWOC ON vs OFF for this scenario",
        key="pg_ewoc_compare_btn",
        help=(
            "Reruns the same CRM scenario with EWOC ON (current α) and EWOC OFF. "
            "All other settings stay fixed."
        ),
    )
    if _ewoc_btn or "_ewoc_compare" in st.session_state:
        if _ewoc_btn:
            _ew_ns    = min(int(_cfg("n_sims")), 500)   # cap at 500 for speed
            _ew_seed  = int(_cfg("seed")) + 9999
            _ew_t1    = list(true_t1)
            _ew_t2    = list(true_t2)
            _ew_base  = dict(
                p_surgery=float(_cfg("p_surgery")),
                target1=float(get_config_value("target_t1")),
                target2=float(get_config_value("target_t2")),
                sigma=float(_cfg("sigma")),
                start_level=int(_cfg("start_level_1b")),
                max_n=int(_cfg("max_n_crm")),
                cohort_size=int(_cfg("cohort_size")),
                accrual_per_month=float(_cfg("accrual_per_month")),
                incl_to_rt=int(_cfg("incl_to_rt")),
                rt_dur=int(_cfg("rt_dur")),
                rt_to_surg=int(_cfg("rt_to_surg")),
                tox1_win=int(_cfg("rt_dur")) + int(_cfg("rt_to_surg")),
                tox2_win=int(_cfg("tox2_win")),
                max_step=int(_cfg("max_step")),
                gh_n=int(_cfg("gh_n")),
                burn_in=bool(_cfg("burn_in")),
                enforce_guardrail=bool(_cfg("enforce_guardrail")),
                restrict_final_to_tried=bool(_cfg("restrict_final_mtd")),
                n_safe_d1=int(_cfg("n_safe_d1")),
                require_full_tox1_fu_before_escalation=bool(_cfg("require_full_tox1_fu")),
                p_stop=1.0,
            )
            _ew_skel1 = list(skel_t1)
            _ew_skel2 = list(skel_t2)
            _ew_target1 = float(get_config_value("target_t1"))
            _ew_target2 = float(get_config_value("target_t2"))
            _ew_optimal = _true_optimal(
                np.array(_ew_t1), np.array(_ew_t2), _ew_target1, _ew_target2
            )

            def _run_ewoc_arm(ewoc_on_flag, arm_seed):
                rng = np.random.default_rng(arm_seed)
                sel_counts = np.zeros(5, dtype=int)
                qs_list, od_list, th_list = [], [], []
                base_kw = dict(_ew_base)
                for _dup in ("true_t1", "true_t2", "skel1", "skel2", "rng",
                              "collect_trace", "ewoc_on", "ewoc_alpha"):
                    base_kw.pop(_dup, None)
                for _ in range(_ew_ns):
                    _s, *_ = run_tite_crm(
                        true_t1=np.asarray(_ew_t1, dtype=float),
                        true_t2=np.asarray(_ew_t2, dtype=float),
                        skel1=np.asarray(_ew_skel1, dtype=float),
                        skel2=np.asarray(_ew_skel2, dtype=float),
                        ewoc_on=ewoc_on_flag,
                        ewoc_alpha=float(_cfg("ewoc_alpha")),
                        rng=rng,
                        collect_trace=False,
                        **base_kw,
                    )
                    sel_counts[_s] += 1
                    qs_list.append(_quality_score(_s, np.array(_ew_t1), np.array(_ew_t2),
                                                  _ew_target1, _ew_target2))
                    od_list.append(int(max(float(_ew_t1[_s]) - _ew_target1,
                                          float(_ew_t2[_s]) - _ew_target2) > 0))
                    th_list.append(int(_s > _ew_optimal))
                psel = sel_counts / _ew_ns
                bim  = _crm_bimodality(psel)
                return {
                    "psel": psel,
                    "bim":  bim,
                    "quality_score":      float(np.mean(qs_list)),
                    "pct_correct":        100.0 * sel_counts[_ew_optimal] / _ew_ns,
                    "overdose_rate":      100.0 * float(np.mean(od_list)),
                    "too_high_rate":      100.0 * float(np.mean(th_list)),
                }

            with st.spinner(f"Running EWOC ON vs OFF ({_ew_ns} sims each)…"):
                _ew_on  = _run_ewoc_arm(True,  _ew_seed)
                _ew_off = _run_ewoc_arm(False, _ew_seed + 1)
            st.session_state["_ewoc_compare"] = {
                "on": _ew_on, "off": _ew_off,
                "alpha": float(_cfg("ewoc_alpha")),
                "ns": _ew_ns,
            }

        if "_ewoc_compare" in st.session_state:
            _ec   = st.session_state["_ewoc_compare"]
            _ec_on  = _ec["on"]
            _ec_off = _ec["off"]
            _ec_alpha = _ec["alpha"]
            _ec_ns    = _ec["ns"]

            st.caption(
                f"Each arm: {_ec_ns} simulations · EWOC ON uses α = {_ec_alpha:.2f}"
            )

            # Side-by-side metrics
            _ew_mc1, _ew_mc2 = st.columns(2)
            with _ew_mc1:
                st.markdown(f"**EWOC ON (α = {_ec_alpha:.2f})**")
                st.metric("Quality score",      f"{_ec_on['quality_score']:.3f}")
                st.metric("Correct selection",  f"{_ec_on['pct_correct']:.1f}%")
                st.metric("Overdose rate",      f"{_ec_on['overdose_rate']:.1f}%")
                st.metric("Too-high selection", f"{_ec_on['too_high_rate']:.1f}%")
                st.metric("2nd peak prob",      f"{_ec_on['bim']['second_peak_prob']:.1%}")
                st.metric("Valley depth",       f"{_ec_on['bim']['valley_depth']:.1%}")
            with _ew_mc2:
                st.markdown("**EWOC OFF**")
                st.metric("Quality score",      f"{_ec_off['quality_score']:.3f}")
                st.metric("Correct selection",  f"{_ec_off['pct_correct']:.1f}%")
                st.metric("Overdose rate",      f"{_ec_off['overdose_rate']:.1f}%")
                st.metric("Too-high selection", f"{_ec_off['too_high_rate']:.1f}%")
                st.metric("2nd peak prob",      f"{_ec_off['bim']['second_peak_prob']:.1%}")
                st.metric("Valley depth",       f"{_ec_off['bim']['valley_depth']:.1%}")

            # Side-by-side selection charts
            _ew_fc1, _ew_fc2 = st.columns(2)
            with _ew_fc1:
                _ew_fig_on = _plot_dichotomy_selection(
                    _ec_on["psel"], _ec_on["bim"], _dd_dl)
                st.image(fig_to_png_bytes(_ew_fig_on), use_container_width=True)
                plt.close(_ew_fig_on)
            with _ew_fc2:
                _ew_fig_off = _plot_dichotomy_selection(
                    _ec_off["psel"], _ec_off["bim"], _dd_dl)
                st.image(fig_to_png_bytes(_ew_fig_off), use_container_width=True)
                plt.close(_ew_fig_off)

# ==============================================================================
# CRM sample-size distribution histogram
# ==============================================================================

if (view == "Playground"
        and "_tite_results" in st.session_state
        and st.session_state["_tite_results"].get("early_stop_on", False)):
    _hres      = st.session_state["_tite_results"]
    _n_arr     = np.array(_hres["n_per_trial_crm"], dtype=int)
    _es_arr    = np.array(_hres["early_stop_arr"],  dtype=bool)
    _max_n_cfg = int(_n_arr.max()) if len(_n_arr) else 0   # equals max_n_crm

    st.markdown("---")
    st.subheader("CRM trial sample-size distribution")
    st.caption(
        "Distribution of patients enrolled in simulated CRM trials that "
        "triggered early stopping (orange). Trials that ran to the maximum "
        "sample size are noted separately. Dashed lines mark the mean "
        "(white) and median (green) of early-stopped trials."
    )

    _n_early   = _n_arr[_es_arr]
    _n_full_ct = int((~_es_arr).sum())

    _lo   = max(1, (_n_early.min() if _es_arr.any() else 1))
    _bins = np.arange(_lo, _max_n_cfg + 2, 1)   # integer edges: [lo, lo+1, …, max_n+1]

    _ns_h = len(_n_arr)

    fig_h, ax_h = plt.subplots(figsize=(9.0, 3.2), dpi=RESULT_DPI)
    _apply_dark_fig(fig_h, ax_h)

    if _es_arr.any():
        ax_h.hist(_n_early, bins=_bins, color="#ffaa44",
                  label=f"Early stopped ({_es_arr.sum()} trials)",
                  edgecolor="#2a2a4a", linewidth=0.6)
        _mean_e   = float(_n_early.mean())
        _median_e = float(np.median(_n_early))
        ax_h.axvline(_mean_e,   color="#ffffff", lw=1.4, ls="--",
                     label=f"Mean {_mean_e:.1f}")
        ax_h.axvline(_median_e, color="#80ff80", lw=1.4, ls="--",
                     label=f"Median {_median_e:.0f}")
        _stats_lines = [
            f"early-stopped trials",
            f"n       : {_es_arr.sum()} / {_ns_h} "
            f"({_hres['crm_early_stop_pct']:.1%})",
            f"mean N  : {_mean_e:.1f}",
            f"median N: {_median_e:.0f}",
            f"range   : {_n_early.min()}–{_n_early.max()}",
            f"p_stop  : {_hres['p_stop']:.2f}",
        ]
    else:
        _stats_lines = [f"No early stops in {_ns_h} trials",
                        f"p_stop={_hres['p_stop']:.2f}"]

    # Annotate full-length trials as a text note at the right edge
    if _n_full_ct > 0:
        _ymax = ax_h.get_ylim()[1] if ax_h.get_ylim()[1] > 0 else _ns_h
        ax_h.annotate(
            f"{_n_full_ct} trial{'s' if _n_full_ct != 1 else ''}\nran to max N",
            xy=(_max_n_cfg, 0), xycoords="data",
            xytext=(-6, 6), textcoords="offset points",
            ha="right", va="bottom", fontsize=7.5, color="#8888bb",
            arrowprops=None,
        )

    ax_h.text(0.015, 0.97, "\n".join(_stats_lines),
              transform=ax_h.transAxes, fontsize=7.5,
              verticalalignment="top", horizontalalignment="left",
              color=_DARK_FG,
              bbox=dict(facecolor=_DARK_AX, edgecolor="#444466",
                        boxstyle="round,pad=0.4", alpha=0.9))

    ax_h.set_xlim(left=_lo, right=_max_n_cfg + 1)   # fits all integer-edge bins exactly
    ax_h.set_xlabel("Patients enrolled at early stop", fontsize=9)
    ax_h.set_ylabel("Frequency (trials)", fontsize=9)
    ax_h.set_title("CRM early-stopping sample-size distribution", fontsize=10)
    compact_style(ax_h)
    ax_h.legend(fontsize=8, frameon=False, labelcolor=_DARK_FG)
    fig_h.tight_layout(pad=0.4)
    st.image(fig_to_png_bytes(fig_h), use_container_width=True)

# ==============================================================================
# Posterior Tracking — first simulated trial
# Shows how posterior mean toxicity estimates evolved across cohort decisions.
# Uses the pm1/pm2 values already stored in the CRM trace for the first trial.
# ==============================================================================

if view == "Playground" and "_tite_results" in st.session_state:
    _post_tr   = st.session_state["_tite_results"]["crm_trace"]
    _post_decs = _post_tr.get("decisions", [])

    if _post_decs:
        st.markdown("---")
        st.subheader("Posterior mean tracking — first CRM trial")
        st.caption(
            "How the TITE-CRM posterior mean toxicity estimates evolved across "
            "cohort decisions in the first simulated trial. "
            "Each line represents one dose level with a shaded 90% credible band; "
            "the dashed green line marks the target toxicity rate."
        )

        _pt_steps = [d["step"]  for d in _post_decs]
        _pt_pm1   = np.array([d["pm1"] for d in _post_decs])   # (n_steps, n_levels)
        _pt_pm2   = np.array([d["pm2"] for d in _post_decs])   # (n_steps, n_levels)
        _n_lvls   = _pt_pm1.shape[1]

        # ── Re-compute 5th/95th weighted percentiles of the marginal posterior ──
        _sigma_pt   = float(_post_tr.get("sigma",   _cfg("sigma")))
        _skel_t1_pt = np.asarray(_post_tr.get("skel_t1"), dtype=float)
        _skel_t2_pt = np.asarray(_post_tr.get("skel_t2"), dtype=float)
        _gh_n_pt    = int(_post_tr.get("gh_n", _cfg("gh_n")))

        _n_steps = len(_post_decs)
        _lo1 = np.zeros((_n_steps, _n_lvls))
        _hi1 = np.zeros((_n_steps, _n_lvls))
        _lo2 = np.zeros((_n_steps, _n_lvls))
        _hi2 = np.zeros((_n_steps, _n_lvls))

        def _weighted_pct(vals, weights, pcts):
            """Weighted percentile via sorted cumulative-weight interpolation."""
            idx = np.argsort(vals)
            sv  = vals[idx]
            sw  = weights[idx]
            cum = np.cumsum(sw)
            cum = cum / cum[-1]
            return np.interp(pcts, cum, sv)

        for _si, _d in enumerate(_post_decs):
            _n1a = np.asarray(_d["n1"], dtype=float)
            _y1a = np.asarray(_d["y1"], dtype=float)
            _n2a = np.asarray(_d["n2"], dtype=float)
            _y2a = np.asarray(_d["y2"], dtype=float)

            _pw1, _P1 = posterior_via_gh(
                _sigma_pt, _skel_t1_pt, _n1a, _y1a, gh_n=_gh_n_pt)
            _pw2, _P2 = posterior_via_gh(
                _sigma_pt, _skel_t2_pt, _n2a, _y2a, gh_n=_gh_n_pt)

            for _lvl in range(_n_lvls):
                _lo1[_si, _lvl], _hi1[_si, _lvl] = _weighted_pct(
                    _P1[:, _lvl], _pw1, [0.05, 0.95])
                _lo2[_si, _lvl], _hi2[_si, _lvl] = _weighted_pct(
                    _P2[:, _lvl], _pw2, [0.05, 0.95])

        _dose_colors = ["#4a9eff", "#ffaa44", "#ff6677", "#55dd99", "#cc88ff"]
        _tgt1 = _post_tr.get("target_t1", float(_cfg("target_t1")))
        _tgt2 = _post_tr.get("target_t2", float(_cfg("target_t2")))
        _ewoc_alpha_pt = float(_post_tr.get("ewoc_alpha", float(_cfg("ewoc_alpha"))))

        fig, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(11.0, 3.8), dpi=150)
        _apply_dark_fig(fig, ax_p1, ax_p2)

        for _lvl in range(_n_lvls):
            ax_p1.plot(_pt_steps, _pt_pm1[:, _lvl], "o-",
                       color=_dose_colors[_lvl], lw=1.8, ms=4, label=f"L{_lvl}")
            ax_p1.fill_between(_pt_steps, _lo1[:, _lvl], _hi1[:, _lvl],
                               alpha=0.15, color=_dose_colors[_lvl], linewidth=0)
            ax_p2.plot(_pt_steps, _pt_pm2[:, _lvl], "o-",
                       color=_dose_colors[_lvl], lw=1.8, ms=4, label=f"L{_lvl}")
            ax_p2.fill_between(_pt_steps, _lo2[:, _lvl], _hi2[:, _lvl],
                               alpha=0.15, color=_dose_colors[_lvl], linewidth=0)

        ax_p1.axhline(_tgt1, lw=1.5, ls="--", color="#80ff80",
                      alpha=0.8, label=f"Target ({_tgt1:.2f})")
        ax_p1.axhline(_ewoc_alpha_pt, lw=1.5, ls="--", color="#ff9944",
                      alpha=0.8, label=f"EWOC α={_ewoc_alpha_pt:.2f}")
        ax_p2.axhline(_tgt2, lw=1.5, ls="--", color="#80ff80",
                      alpha=0.8, label=f"Target ({_tgt2:.2f})")
        ax_p2.axhline(_ewoc_alpha_pt, lw=1.5, ls="--", color="#ff9944",
                      alpha=0.8, label=f"EWOC α={_ewoc_alpha_pt:.2f}")

        ax_p1.set_title("Tox1 posterior mean per dose level", fontsize=10)
        ax_p1.set_xlabel("Cohort decision step", fontsize=9)
        ax_p1.set_ylabel("Posterior mean P(tox1)", fontsize=9)
        ax_p1.set_ylim(0, min(1.0, max(float(_hi1.max()), _tgt1, _ewoc_alpha_pt) * 1.18 + 0.05))
        ax_p1.legend(fontsize=8, frameon=False, labelcolor=_DARK_FG, ncol=2)
        compact_style(ax_p1)

        ax_p2.set_title("Tox2 posterior mean per dose level", fontsize=10)
        ax_p2.set_xlabel("Cohort decision step", fontsize=9)
        ax_p2.set_ylabel("Posterior mean P(tox2)", fontsize=9)
        ax_p2.set_ylim(0, min(1.0, max(float(_hi2.max()), _tgt2, _ewoc_alpha_pt) * 1.18 + 0.05))
        ax_p2.legend(fontsize=8, frameon=False, labelcolor=_DARK_FG, ncol=2)
        compact_style(ax_p2)

        fig.tight_layout(pad=0.5)
        st.image(fig_to_png_bytes(fig), use_container_width=True)
        st.caption(
            "Posterior means are computed at each cohort decision using TITE fractional "
            "weights. Shaded bands = 90% credible intervals (5th–95th weighted percentiles "
            "of the marginal posterior over P(tox) at each dose). "
            "Dashed green line = target toxicity rate; dashed orange line = EWOC α "
            "overdose-exclusion threshold. Based on the first simulated trial only."
        )

        # ── Study-end posterior distribution (ridge plot) ──────────────────────
        st.subheader("Study-end posterior distributions — first CRM trial")

        # Recompute TITE weights at study end (all follow-up windows closed).
        # This matches the weights used by crm_select_mtd for the final MTD
        # decision, so OD probabilities here agree with the eligibility heatmap.
        _n1_end, _y1_end, _n2_end, _y2_end = tite_weights(
            _post_tr["patients"],
            float(_post_tr["study_days"]),
            int(_post_tr["tox1_win"]),
            int(_post_tr["tox2_win"]),
            _n_lvls,
        )

        _pw1f, _P1f = posterior_via_gh(
            _sigma_pt, _skel_t1_pt, _n1_end, _y1_end, gh_n=_gh_n_pt)
        _pw2f, _P2f = posterior_via_gh(
            _sigma_pt, _skel_t2_pt, _n2_end, _y2_end, gh_n=_gh_n_pt)

        # Overdose probability per dose: P(P[:,d] > target_tox).
        # This matches crm_posterior_summaries / crm_select_mtd exactly:
        #   OD_prob = sum(post_w where P[:,d] > target_tox)
        #   EWOC exclusion rule: OD_prob > ewoc_alpha.
        # Using ewoc_alpha as the threshold here would give P(P[:,d] > 0.25)
        # which is a different quantity and does NOT match the decision logic.
        _od1_final = np.array([
            float(np.sum(_pw1f[_P1f[:, _d] > _tgt1]))
            for _d in range(_n_lvls)
        ])
        _od2_final = np.array([
            float(np.sum(_pw2f[_P2f[:, _d] > _tgt2]))
            for _d in range(_n_lvls)
        ])

        _xgrid    = np.linspace(0.0, 1.0, 400)
        _od_mask1 = _xgrid >= _tgt1   # region that integrates to tox1 OD probability
        _od_mask2 = _xgrid >= _tgt2   # region that integrates to tox2 OD probability

        def _weighted_kde(p_col, weights, xgrid):
            """Weighted Gaussian KDE; bandwidth via Scott's rule on weighted std."""
            mu    = np.sum(weights * p_col)
            var   = np.sum(weights * (p_col - mu) ** 2)
            std   = np.sqrt(var) if var > 1e-12 else 0.01
            eff_n = 1.0 / max(np.sum(weights ** 2), 1e-30)
            bw    = max(1.06 * std * eff_n ** (-0.2), 0.005)
            diff  = xgrid[:, None] - p_col[None, :]       # (nx, gh_n)
            kde   = (weights[None, :] * np.exp(-0.5 * (diff / bw) ** 2)).sum(axis=1)
            return kde

        fig2, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)
        _apply_dark_fig(fig2, ax_d1, ax_d2)

        # x anchor for OD labels: just right of each panel's toxicity target
        _x_ann1 = min(0.94, _tgt1 + 0.05)
        _x_ann2 = min(0.94, _tgt2 + 0.05)

        for _lvl in range(_n_lvls):
            _kde1 = _weighted_kde(_P1f[:, _lvl], _pw1f, _xgrid)
            _kde2 = _weighted_kde(_P2f[:, _lvl], _pw2f, _xgrid)

            # Normalise each KDE so its peak reaches 0.85 within its slot
            _k1 = 0.85 / max(float(_kde1.max()), 1e-12)
            _k2 = 0.85 / max(float(_kde2.max()), 1e-12)

            _y1c = _lvl + _kde1 * _k1
            _y2c = _lvl + _kde2 * _k2

            # Main dose-coloured fill and outline
            ax_d1.fill_between(_xgrid, _lvl, _y1c,
                               color=_dose_colors[_lvl], alpha=0.38, linewidth=0)
            ax_d1.plot(_xgrid, _y1c, color=_dose_colors[_lvl], lw=1.3)

            ax_d2.fill_between(_xgrid, _lvl, _y2c,
                               color=_dose_colors[_lvl], alpha=0.38, linewidth=0)
            ax_d2.plot(_xgrid, _y2c, color=_dose_colors[_lvl], lw=1.3)

            # Red shading: x >= target_tox — this region integrates to the OD probability
            ax_d1.fill_between(_xgrid, _lvl, _y1c,
                               where=_od_mask1, color=(1.0, 0.39, 0.39),
                               alpha=0.45, linewidth=0)
            ax_d2.fill_between(_xgrid, _lvl, _y2c,
                               where=_od_mask2, color=(1.0, 0.39, 0.39),
                               alpha=0.45, linewidth=0)

            # OD probability label inside the red region
            ax_d1.text(_x_ann1, _lvl + 0.08,
                       f"OD: {_od1_final[_lvl] * 100:.1f}%",
                       color="#e0e0e0", fontsize=7, ha="left", va="bottom",
                       fontweight="bold")
            ax_d2.text(_x_ann2, _lvl + 0.08,
                       f"OD: {_od2_final[_lvl] * 100:.1f}%",
                       color="#e0e0e0", fontsize=7, ha="left", va="bottom",
                       fontweight="bold")

        for _ax, _tgt in ((ax_d1, _tgt1), (ax_d2, _tgt2)):
            _ax.axvline(_tgt, lw=1.5, ls="--", color="#80ff80",
                        alpha=0.85, label=f"Target ({_tgt:.2f})")
            _ax.set_xlim(0.0, 1.0)
            _ax.set_ylim(-0.2, _n_lvls - 0.1)
            _ax.set_yticks(range(_n_lvls))
            _ax.set_yticklabels([f"L{i}" for i in range(_n_lvls)], fontsize=8)
            _ax.set_xlabel("P(tox)", fontsize=9)
            _ax.set_ylabel("Dose level", fontsize=9)
            _ax.legend(fontsize=8, frameon=False, labelcolor=_DARK_FG)
            compact_style(_ax)

        ax_d1.set_title("Tox1 — study-end posterior per dose level", fontsize=10)
        ax_d2.set_title("Tox2 — study-end posterior per dose level", fontsize=10)
        fig2.tight_layout(pad=0.5)
        st.image(fig_to_png_bytes(fig2), use_container_width=True)
        st.caption(
            "Posteriors are computed using complete TITE weights at study end "
            "(all follow-up windows closed), matching the inputs to crm_select_mtd. "
            "Red shading marks the region x ≥ target toxicity rate; its area equals "
            "the OD probability P(tox > target) — the same quantity the EWOC filter "
            "compares against α to admit or exclude each dose. "
            "'OD: X.X%' labels match the decision walkthrough table exactly. "
            "Green dashed line = target toxicity rate (shading boundary). "
            "Orange dashed line = EWOC α — OD probabilities above this value "
            "cause dose exclusion."
        )

        # ── Final posterior support by dose (first trial) ─────────────────────
        st.subheader("Final posterior support by dose — first CRM trial")
        _final_mtd_for_supp = _post_tr.get("final_mtd")
        _ewoc_on_ps  = bool(_post_tr.get("ewoc_on",  _cfg("ewoc_on")))
        _ewoc_alp_ps = float(_post_tr.get("ewoc_alpha", _cfg("ewoc_alpha")))
        _ewoc_app_ps = str(_post_tr.get("ewoc_application", _cfg("ewoc_application")))
        _restrict_ps = bool(_post_tr.get("restrict_final_mtd", _cfg("restrict_final_mtd")))
        _, _ewoc_final_eff_ps = ewoc_effective_alphas(
            _ewoc_app_ps, _ewoc_alp_ps, ewoc_on=_ewoc_on_ps)
        _supp_probs_tr = crm_mtd_posterior_probs(
            _sigma_pt,
            _skel_t1_pt, _skel_t2_pt,
            _n1_end, _y1_end, _n2_end, _y2_end,
            float(_post_tr.get("target_t1", _cfg("target_t1"))),
            float(_post_tr.get("target_t2", _cfg("target_t2"))),
            ewoc_alpha=_ewoc_final_eff_ps,
            gh_n=_gh_n_pt,
            restrict_to_tried=_restrict_ps,
        )
        _dose_labels_sp = [f"L{i}" for i in range(_n_lvls)]
        fig_sp, ax_sp = plt.subplots(figsize=(6.0, 2.8), dpi=150)
        _apply_dark_fig(fig_sp, ax_sp)
        _sp_colors = ["#4a9eff", "#ffaa44", "#ff6677", "#55dd99", "#cc88ff"]
        _bars_sp = ax_sp.bar(
            range(_n_lvls), _supp_probs_tr,
            color=[_sp_colors[i] for i in range(_n_lvls)],
            edgecolor=_DARK_BG, linewidth=0.5,
        )
        # Annotate bars with probability values
        for _i, (bar, prob) in enumerate(zip(_bars_sp, _supp_probs_tr)):
            if prob > 0.01:
                ax_sp.text(bar.get_x() + bar.get_width() / 2, prob + 0.01,
                           f"{prob:.2f}", ha="center", va="bottom",
                           fontsize=8, color=_DARK_FG)
        # Highlight final selected MTD
        if _final_mtd_for_supp is not None:
            _bars_sp[_final_mtd_for_supp].set_edgecolor("#ffd700")
            _bars_sp[_final_mtd_for_supp].set_linewidth(2.5)
            ax_sp.text(
                _final_mtd_for_supp, _supp_probs_tr[_final_mtd_for_supp] / 2,
                "▶ selected",
                ha="center", va="center", fontsize=7.5, color="#ffd700",
                fontweight="bold",
            )
        ax_sp.set_xticks(range(_n_lvls))
        ax_sp.set_xticklabels(_dose_labels_sp, fontsize=9)
        ax_sp.set_ylim(0, min(1.05, max(float(_supp_probs_tr.max()) * 1.25, 0.12)))
        ax_sp.set_xlabel("Dose level", fontsize=9)
        ax_sp.set_ylabel("Posterior probability", fontsize=9)
        ax_sp.set_title(
            f"Final posterior support by dose  "
            f"(selected MTD = L{_final_mtd_for_supp}, "
            f"support = {float(_supp_probs_tr[_final_mtd_for_supp]):.2f})"
            if _final_mtd_for_supp is not None else
            "Final posterior support by dose",
            fontsize=10,
        )
        compact_style(ax_sp)
        fig_sp.tight_layout(pad=0.4)
        st.image(fig_to_png_bytes(fig_sp), use_container_width=False, width=520)
        plt.close(fig_sp)
        st.caption(
            "Posterior probability that each dose level would be selected as the final MTD "
            "by the CRM rule, integrating over the study-end posterior. "
            "Gold border = the dose actually selected in this trial. "
            "Dual-endpoint selection rule (EWOC + tox1 target matching) applied."
        )

        # ── Dose eligibility trajectory ────────────────────────────────────────
        st.subheader("Dose eligibility trajectory across trial")

        _n_steps_el  = len(_post_decs)
        _final_mtd_v = _post_tr.get("final_mtd")
        _elig_border = _ewoc_alpha_pt - 0.05    # lower bound of borderline zone
        _ewoc_decision_eff_el, _ = ewoc_effective_alphas(
            str(_post_tr.get("ewoc_application", _cfg("ewoc_application"))),
            _ewoc_alpha_pt,
            ewoc_on=bool(_post_tr.get("ewoc_on", _cfg("ewoc_on"))))

        fig3, ax3 = plt.subplots(
            figsize=(max(8.0, _n_steps_el * 0.90 + 1.5), 4.2), dpi=150)
        _apply_dark_fig(fig3, ax3)
        ax3.set_facecolor(_DARK_BG)   # darker canvas; cell colours carry meaning
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        for _si, _step_d in enumerate(_post_decs):
            _od1_row = _step_d["od1"]
            _od2_row = _step_d["od2"]
            _curr    = _step_d["current_dose"]

            for _lvl in range(_n_lvls):
                _o1 = float(_od1_row[_lvl])
                _o2 = float(_od2_row[_lvl])

                # Priority: red > amber > green
                if _o1 > _ewoc_alpha_pt or _o2 > _ewoc_alpha_pt:
                    _fc = "#7b241c"   # dark red   — excluded
                elif _o1 >= _elig_border or _o2 >= _elig_border:
                    _fc = "#7e5109"   # dark amber — borderline
                else:
                    _fc = "#1a6835"   # dark green — allowed

                ax3.add_patch(mpatches.Rectangle(
                    (_si, _lvl), 1, 1,
                    facecolor=_fc, edgecolor="#2c2c4c", linewidth=0.8, zorder=1))

                # Tox1 OD (white, top half) and Tox2 OD (light grey, bottom half)
                ax3.text(_si + 0.5, _lvl + 0.64, f"{_o1:.2f}",
                         color="white",   fontsize=6.5, ha="center", va="center",
                         fontweight="bold", zorder=3)
                ax3.text(_si + 0.5, _lvl + 0.34, f"{_o2:.2f}",
                         color="#cccccc", fontsize=6.5, ha="center", va="center",
                         fontweight="bold", zorder=3)

            # Silver outline: dose being treated this cohort
            ax3.add_patch(mpatches.Rectangle(
                (_si, _curr), 1, 1,
                fill=False, edgecolor="#b8b8b8", linewidth=2.0, zorder=4))

        # Gold outline: final MTD at the last decision step
        if _final_mtd_v is not None:
            ax3.add_patch(mpatches.Rectangle(
                (_n_steps_el - 1, _final_mtd_v), 1, 1,
                fill=False, edgecolor="#ffd700", linewidth=3.5, zorder=5))

        ax3.set_xlim(0, _n_steps_el)
        ax3.set_ylim(0, _n_lvls)
        ax3.set_xticks([_s + 0.5 for _s in range(_n_steps_el)])
        ax3.set_xticklabels([str(_sd["step"]) for _sd in _post_decs], fontsize=8)
        ax3.set_yticks([_i + 0.5 for _i in range(_n_lvls)])
        ax3.set_yticklabels([f"L{_i}" for _i in range(_n_lvls)], fontsize=9)
        ax3.set_xlabel("Cohort decision step", fontsize=9)
        ax3.set_ylabel("Dose level", fontsize=9)
        ax3.set_title("Dose eligibility trajectory across trial", fontsize=10)
        ax3.tick_params(length=0)

        _elig_handles = [
            mpatches.Patch(facecolor="#1a6835", edgecolor="#555", linewidth=0.8,
                           label="Allowed — both OD < α"),
            mpatches.Patch(facecolor="#7e5109", edgecolor="#555", linewidth=0.8,
                           label=f"Borderline — either OD ∈ [{_elig_border:.2f}, α]"),
            mpatches.Patch(facecolor="#7b241c", edgecolor="#555", linewidth=0.8,
                           label=f"Excluded — either OD > α ({_ewoc_alpha_pt:.2f})"),
            mpatches.Patch(fill=False, edgecolor="#b8b8b8", linewidth=2.0,
                           label="Active dose this cohort"),
            mpatches.Patch(fill=False, edgecolor="#ffd700", linewidth=3.0,
                           label="Final MTD selected"),
        ]
        ax3.legend(handles=_elig_handles, fontsize=7, frameon=True, ncol=3,
                   loc="upper left", bbox_to_anchor=(0.0, -0.20),
                   labelcolor=_DARK_FG, facecolor=_DARK_AX, edgecolor=_DARK_GRD)

        fig3.tight_layout(rect=[0, 0.17, 1, 1], pad=0.4)
        st.image(fig_to_png_bytes(fig3), use_container_width=True)
        st.caption(
            "Each cell shows posterior overdose probabilities at that dose and step: "
            "tox1 OD prob (white, top) and tox2 OD prob (grey, bottom). "
            "Green = both below EWOC α (dose is admissible); "
            f"amber = borderline (either OD ∈ [{_elig_border:.2f}, {_ewoc_alpha_pt:.2f}]); "
            f"red = excluded (either OD > α = {_ewoc_alpha_pt:.2f}). "
            "Silver border = dose being given to that cohort; "
            "gold border = final MTD selected after full follow-up."
            + ("" if _ewoc_decision_eff_el is not None else
               " **Note:** EWOC was NOT applied to dose assignment in this trial "
               "(mode = Final MTD only / Off) — the colouring above shows where each "
               "dose stands relative to α for reference only; it did not drive the "
               "cohort-by-cohort dose decisions.")
        )

# ==============================================================================
# CRM Decision Trace — first simulated trial walkthrough
# Shown only when "Explain first CRM trial" toggle is ON.
#
# Patient timeline table: one row per patient with true probs, event times,
#   surgery status, and their fractional TITE weight at the decision that
#   immediately followed their cohort enrollment.
#
# Decision walkthrough table: one row per cohort decision — posteriors,
#   overdose probabilities, allowed doses, and the reason for the choice.
#
# Plots:
#   A. Dose level over cohort steps (dose assigned to each successive cohort)
#   B. Overdose probabilities at the selected dose over steps (tox1 & tox2)
#   C. TITE follow-up accumulation (total effective n for tox1 and tox2)
# ==============================================================================

if (view == "Playground"
        and "_tite_results" in st.session_state
        and bool(st.session_state.get("show_crm_trace", False))):

    import pandas as pd  # local import — only needed for trace tables

    _tr = st.session_state["_tite_results"]["crm_trace"]
    _pts      = _tr["patients"]
    _decs     = _tr["decisions"]
    _true_t1  = _tr["true_t1"]
    _true_t2  = _tr["true_t2"]
    _tox1_win = _tr["tox1_win"]
    _tox2_win = _tr["tox2_win"]

    st.markdown("---")
    st.subheader("First CRM trial — decision walkthrough")
    _ewoc_alpha   = float(_cfg("ewoc_alpha"))
    _ewoc_decision_eff_wt, _ewoc_final_eff_wt = ewoc_effective_alphas(
        str(_cfg("ewoc_application")), _ewoc_alpha, ewoc_on=bool(_cfg("ewoc_on")))
    if _ewoc_decision_eff_wt is not None:
        st.caption(
            f"**EWOC ON for dose assignment (α = {_ewoc_alpha:.2f})** — At each decision "
            "the model filters doses to those where P(tox1 > target) < α **and** "
            "P(tox2 > target) < α (joint safety rule). The **highest** jointly admissible "
            "dose is then selected, subject to max-step and guardrail constraints."
        )
    else:
        _off_note = (
            " EWOC is still applied when selecting the **final MTD** at study end."
            if _ewoc_final_eff_wt is not None else ""
        )
        st.caption(
            "**EWOC OFF for dose assignment** — No overdose-probability filter is applied "
            "during the trial. Among all doses (subject to step and guardrail constraints), "
            "the model picks the dose whose posterior mean P(tox1) is **closest to target1** "
            "(standard CRM argmin rule). This is target-based and does not automatically "
            "escalate to the highest dose." + _off_note
        )

    # ── helper: compute per-patient TITE weight at a given decision day ───────
    def _w1(pt, day):
        t = float(day)
        if t < pt["rt_start"]:
            return 0.0
        if pt["has_tox1"] and pt["tox1_day"] is not None and pt["tox1_day"] <= t:
            return 1.0
        if t >= pt["tox1_win_end"]:
            return 1.0
        return (t - pt["rt_start"]) / float(_tox1_win)

    def _w2(pt, day):
        if not pt["has_surgery"] or pt["surgery_day"] is None:
            return None
        t  = float(day)
        sd = pt["surgery_day"]
        if t < sd:
            return 0.0
        if pt["has_tox2"] and pt["tox2_day"] is not None and pt["tox2_day"] <= t:
            return 1.0
        if pt["tox2_win_end"] is not None and t >= pt["tox2_win_end"]:
            return 1.0
        return (t - sd) / float(_tox2_win)

    # Map each patient index to the decision that followed their cohort
    _pt_to_dec = {}
    for _d in _decs:
        for _pid in _d["cohort_pts"]:
            _pt_to_dec[_pid] = _d

    # ── Patient timeline table ─────────────────────────────────────────────────
    with st.expander("Patient timeline", expanded=True):
        _rows = []
        for _i, _pt in enumerate(_pts):
            _dec      = _pt_to_dec.get(_i)
            _dec_day  = _dec["decision_day"] if _dec else None
            _w1v      = round(_w1(_pt, _dec_day), 2) if _dec_day is not None else "—"
            _w2v      = _w2(_pt, _dec_day) if _dec_day is not None else None
            _w2v_str  = "—" if _w2v is None else round(_w2v, 2)
            _rows.append({
                "Pt #":       _i + 1,
                "Incl (day)": round(_pt["arrival"], 1),
                "Dose":       f"L{_pt['dose']}",
                "True P(tox1)": round(_true_t1[_pt["dose"]], 3),
                "True P(tox2)": round(_true_t2[_pt["dose"]], 3),
                "Surgery":    "Yes" if _pt["has_surgery"] else "No",
                "Tox1 event": "Yes" if _pt["has_tox1"]   else "No",
                "Tox2 event": "Yes" if _pt["has_tox2"]   else "No",
                "Tox1 day":   (round(_pt["tox1_day"], 0)
                               if _pt["tox1_day"] is not None else "—"),
                "Surgery day":(round(_pt["surgery_day"], 0)
                               if _pt["surgery_day"] is not None else "—"),
                "Tox2 day":   (round(_pt["tox2_day"], 0)
                               if _pt["tox2_day"] is not None else "—"),
                "Tox1 wt @dec": _w1v,
                "Tox2 wt @dec": _w2v_str,
            })
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Tox1/Tox2 wt @dec = fractional TITE weight at the decision immediately "
            "following this patient's enrollment. "
            "Weight = 0 (window not yet open), between 0–1 (partial follow-up), "
            "or 1 (event observed or window complete)."
        )

    # ── Decision walkthrough table ─────────────────────────────────────────────
    with st.expander("Dose decision walkthrough", expanded=True):
        _drows = []
        for _d in _decs:
            _allowed_str = ", ".join(f"L{a}" for a in _d["allowed"]) or "none"
            _pm1_str = "[" + " ".join(f"{v:.2f}" for v in _d["pm1"]) + "]"
            _pm2_str = "[" + " ".join(f"{v:.2f}" for v in _d["pm2"]) + "]"
            _od1_str = "[" + " ".join(f"{v:.2f}" for v in _d["od1"]) + "]"
            _od2_str = "[" + " ".join(f"{v:.2f}" for v in _d["od2"]) + "]"
            _drows.append({
                "Step":          _d["step"],
                "Day":           round(_d["decision_day"], 0),
                "N enrolled":    _d["n_enrolled"],
                "Curr dose":     f"L{_d['current_dose']}",
                "Next dose":     f"L{_d['next_dose']}",
                "Highest tried": f"L{_d['highest_tried']}",
                "Burn-in":       "Yes" if _d["burn_in"] else "No",
                "EWOC mode":     _d.get("ewoc_mode", "?"),
                "Obs tox1":      _d["obs_t1"],
                "Obs tox2":      _d["obs_t2"],
                "N surgery":     _d["n_surgery"],
                "Allowed doses": _allowed_str,
                "Post mean tox1 [L0..L4]": _pm1_str,
                "Post mean tox2 [L0..L4]": _pm2_str,
                "OD prob tox1  [L0..L4]":  _od1_str,
                "OD prob tox2  [L0..L4]":  _od2_str,
                "Decision reason": _d["reason"],
            })
        st.dataframe(pd.DataFrame(_drows), use_container_width=True, hide_index=True)
        st.caption(
            "Post mean = posterior mean toxicity probability at each dose level. "
            "OD prob = posterior probability that the true toxicity exceeds the target. "
            "**EWOC ON**: a dose is allowed only if BOTH OD prob tox1 < α AND "
            "OD prob tox2 < α; the highest allowed dose is selected. "
            "**EWOC OFF**: no OD filter — the dose with post mean tox1 closest to "
            "target1 is selected (argmin rule)."
        )

    # ── Final MTD selection ────────────────────────────────────────────────────
    _final_mtd   = _tr.get("final_mtd")
    _sigma_tr    = _tr.get("sigma",    float(_cfg("sigma")))
    _skel_t1_tr  = _tr.get("skel_t1")
    _skel_t2_tr  = _tr.get("skel_t2")
    _tgt1_tr     = _tr.get("target_t1", float(_cfg("target_t1")))
    _tgt2_tr     = _tr.get("target_t2", float(_cfg("target_t2")))
    _gh_n_tr     = _tr.get("gh_n",     int(_cfg("gh_n")))
    _ewoc_on_raw_tr = _tr.get("ewoc_on",  bool(_cfg("ewoc_on")))
    _ewoc_a_tr      = _tr.get("ewoc_alpha", float(_cfg("ewoc_alpha")))
    _ewoc_app_tr    = _tr.get("ewoc_application", str(_cfg("ewoc_application")))
    _restr_tr    = _tr.get("restrict_final_mtd", bool(_cfg("restrict_final_mtd")))
    # Final MTD selection always uses the FINAL-selection effective alpha,
    # which is ON under "Dose assignment + final MTD" and "Final MTD only",
    # and OFF only under "Off".
    _, _ewoc_final_eff_tr = ewoc_effective_alphas(
        _ewoc_app_tr, _ewoc_a_tr, ewoc_on=_ewoc_on_raw_tr)
    _ewoc_on_tr = _ewoc_final_eff_tr is not None
    _sd_tr       = _tr.get("study_days", 0.0)

    if _final_mtd is not None and _skel_t1_tr is not None:
        import pandas as pd  # noqa: F811
        st.markdown("---")
        st.subheader("Final MTD selection — this trial")

        # Prominent MTD banner
        st.markdown(
            f"<div style='font-size:1.35em; font-weight:700; padding:0.5em 0.8em; "
            f"border-radius:6px; background:#1a3a5c; color:#e8f4ff; "
            f"border-left:5px solid #4da6ff; margin-bottom:0.6em;'>"
            f"&#128073; Final selected MTD (this trial): "
            f"<span style='color:#7dd3fc;'>L{_final_mtd}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Re-compute final-state TITE weights (full follow-up = study end)
        _n1f_arr = np.zeros(5)
        _y1f_arr = np.zeros(5)
        _n2f_arr = np.zeros(5)
        _y2f_arr = np.zeros(5)
        if _pts:
            _n1f_arr, _y1f_arr, _n2f_arr, _y2f_arr = tite_weights(
                _pts, _sd_tr, _tox1_win, _tox2_win, 5
            )

        # Final posterior summaries for tox1 and tox2
        _pm1f, _od1f = crm_posterior_summaries(
            _sigma_tr, _skel_t1_tr, _n1f_arr, _y1f_arr, _tgt1_tr, gh_n=_gh_n_tr
        )
        _pm2f, _od2f = crm_posterior_summaries(
            _sigma_tr, _skel_t2_tr, _n2f_arr, _y2f_arr, _tgt2_tr, gh_n=_gh_n_tr
        )

        # Determine per-dose status using the same logic as crm_select_mtd()
        _n_levels = 5
        _tried    = set(int(i) for i in range(_n_levels) if _n1f_arr[i] > 0)

        # Step 1: EWOC admissibility filter
        if _ewoc_on_tr:
            _ewoc_cands = set(
                i for i in range(_n_levels)
                if float(_od1f[i]) < _ewoc_a_tr and float(_od2f[i]) < _ewoc_a_tr
            )
        else:
            _ewoc_cands = set(range(_n_levels))  # all doses when EWOC OFF

        # Step 2: Intersect with tried (if restrict_final_mtd); fall back to
        #         tried-only when intersection is empty (mirrors crm_select_mtd)
        if _restr_tr and _tried:
            _cands_intersect = _ewoc_cands & _tried
            _final_cands = _cands_intersect if _cands_intersect else _tried
        else:
            _final_cands = _ewoc_cands

        # Step 3: Selection rule
        if _ewoc_on_tr:
            _sel_rule_str = (
                f"**EWOC ON for final MTD selection (α = {_ewoc_a_tr:.2f})**: doses "
                f"admitted only where P(tox1 OD) < α **and** P(tox2 OD) < α.  "
                f"Among admitted {'tried ' if _restr_tr else ''}doses, "
                f"the **highest** is selected."
            )
        else:
            _sel_rule_str = (
                "**EWOC OFF for final MTD selection**: no overdose-probability filter.  "
                f"Among {'tried ' if _restr_tr else ''}doses, the one with "
                f"posterior mean P(tox1) **closest to target ({_tgt1_tr:.2f})** "
                "is selected (standard CRM argmin rule)."
            )
        st.caption(_sel_rule_str)

        # Build the per-dose explanation table
        _exp_rows = []
        for _i in range(_n_levels):
            _tried_str   = "Yes" if _i in _tried   else "No"
            _ewoc_ok_1   = float(_od1f[_i]) < _ewoc_a_tr
            _ewoc_ok_2   = float(_od2f[_i]) < _ewoc_a_tr
            _in_cands    = _i in _final_cands

            # Build exclusion reason
            if _i == _final_mtd:
                _status  = "Selected ★"
                _reason  = "Final MTD"
            elif _i in _final_cands:
                _status  = "Allowed ✅"
                _reason  = "Admissible — not selected"
            else:
                _status  = "Excluded ❌"
                # Determine primary exclusion cause
                _excl_parts = []
                if _ewoc_on_tr:
                    if not _ewoc_ok_1 and not _ewoc_ok_2:
                        _excl_parts.append(
                            f"OD prob too high for both tox1 ({_od1f[_i]:.2f} ≥ {_ewoc_a_tr:.2f}) "
                            f"and tox2 ({_od2f[_i]:.2f} ≥ {_ewoc_a_tr:.2f})"
                        )
                    elif not _ewoc_ok_1:
                        _excl_parts.append(
                            f"tox1 OD prob {_od1f[_i]:.2f} ≥ α ({_ewoc_a_tr:.2f})"
                        )
                    elif not _ewoc_ok_2:
                        _excl_parts.append(
                            f"tox2 OD prob {_od2f[_i]:.2f} ≥ α ({_ewoc_a_tr:.2f})"
                        )
                if _restr_tr and _i not in _tried and not _ewoc_cands.isdisjoint(_tried):
                    _excl_parts.append("dose not tried")
                _reason = "Excluded: " + "; ".join(_excl_parts) if _excl_parts else "Excluded"

            _exp_rows.append({
                "Dose":              f"L{_i}",
                "Tried":             _tried_str,
                "Post mean tox1":    round(float(_pm1f[_i]), 3),
                "Post mean tox2":    round(float(_pm2f[_i]), 3),
                "OD prob tox1":      round(float(_od1f[_i]), 3),
                "OD prob tox2":      round(float(_od2f[_i]), 3),
                "Status":            _status,
                "Reason":            _reason,
            })

        _exp_df = pd.DataFrame(_exp_rows)

        # Colour rows: selected = gold, allowed = green tint, excluded = red tint
        def _colour_row(row):
            if "Selected" in row["Status"]:
                bg = "background-color:#1a3a1a; color:#7dffab; font-weight:700"
            elif "Allowed" in row["Status"]:
                bg = "background-color:#0d2a1a; color:#a3e4c0"
            else:
                bg = "background-color:#2a0d0d; color:#e4a3a3"
            return [bg] * len(row)

        st.dataframe(
            _exp_df.style.apply(_colour_row, axis=1),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Final posterior computed at study end (full TITE follow-up weights). "
            "OD prob = P(true tox > target). "
            + ("EWOC filter requires OD prob < α for **both** tox1 and tox2 simultaneously. "
               if _ewoc_on_tr else
               "EWOC OFF: no overdose filter applied; selection is argmin |post mean tox1 − target|. ")
            + ("Excluded doses that passed the safety filter but were never treated are "
               "marked 'dose not tried'." if _restr_tr else "")
        )

    # ── Trace plots ────────────────────────────────────────────────────────────
    if _decs:
        _steps    = [d["step"]          for d in _decs]
        _curr     = [d["current_dose"]  for d in _decs]
        _next     = [d["next_dose"]     for d in _decs]
        _n_enr    = [d["n_enrolled"]    for d in _decs]
        _n1_sum   = [d["n1_sum"]        for d in _decs]
        _n2_sum   = [d["n2_sum"]        for d in _decs]

        # OD prob at the currently assigned dose
        _od1_curr = [d["od1"][d["current_dose"]] for d in _decs]
        _od2_curr = [d["od2"][d["current_dose"]] for d in _decs]

        _tc1, _tc2, _tc3 = st.columns(3, gap="large")

        # ── Plot A: dose level assigned per cohort step ───────────────────────
        with _tc1:
            fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=130)
            _apply_dark_fig(fig, ax)
            ax.step(_steps, _curr, where="post", color="#4a9eff",
                    lw=2, label="Dose assigned")
            ax.step(_steps, _next, where="post", color="#4a9eff",
                    lw=1.2, ls="--", alpha=0.6, label="Next dose chosen")
            ax.set_title("Dose level over cohort steps", fontsize=9)
            ax.set_xlabel("Decision step", fontsize=8)
            ax.set_ylabel("Dose level (L0 – L4)", fontsize=8)
            ax.set_yticks(range(5))
            ax.set_yticklabels([f"L{i}" for i in range(5)], fontsize=7)
            ax.legend(fontsize=7, frameon=False, labelcolor=_DARK_FG)
            compact_style(ax)
            fig.tight_layout(pad=0.5)
            st.image(fig_to_png_bytes(fig), use_container_width=True)
            st.caption("Solid: dose given to current cohort.  "
                       "Dashed: dose selected for the next cohort.")

        # ── Plot B: overdose probabilities at the current dose over time ──────
        with _tc2:
            fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=130)
            _apply_dark_fig(fig, ax)
            ax.plot(_steps, _od1_curr, "o-", color="#4a9eff",
                    lw=1.8, ms=4, label="OD prob tox1")
            ax.plot(_steps, _od2_curr, "s-", color="#ffaa44",
                    lw=1.8, ms=4, label="OD prob tox2")
            ewoc_a = float(_cfg("ewoc_alpha"))
            _ewoc_decision_eff_plt, _ = ewoc_effective_alphas(
                str(_cfg("ewoc_application")), ewoc_a, ewoc_on=bool(_cfg("ewoc_on")))
            if _ewoc_decision_eff_plt is not None:
                ax.axhline(ewoc_a, lw=1, ls="--", color="#80ff80",
                           alpha=0.7, label=f"EWOC α={ewoc_a:.2f}")
            ax.set_title("Safety evolution at current dose", fontsize=9)
            ax.set_xlabel("Decision step", fontsize=8)
            ax.set_ylabel("P(overdose)", fontsize=8)
            ax.set_ylim(0, min(1.05, max(max(_od1_curr), max(_od2_curr)) * 1.3 + 0.05))
            ax.legend(fontsize=7, frameon=False, labelcolor=_DARK_FG)
            compact_style(ax)
            fig.tight_layout(pad=0.5)
            st.image(fig_to_png_bytes(fig), use_container_width=True)
            st.caption(
                "Posterior probability that tox1 (blue) or tox2 (orange) "
                "exceeds the target at the dose currently being given. "
                "Dashed line = EWOC safety threshold."
            )

        # ── Plot C: TITE follow-up accumulation ───────────────────────────────
        with _tc3:
            fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=130)
            _apply_dark_fig(fig, ax)
            ax.plot(_steps, _n1_sum, "o-", color="#4a9eff",
                    lw=1.8, ms=4, label="Effective n (tox1)")
            ax.plot(_steps, _n2_sum, "s-", color="#ffaa44",
                    lw=1.8, ms=4, label="Effective n (tox2)")
            ax.plot(_steps, _n_enr,  "^--", color="#c0c0c0",
                    lw=1.2, ms=4, label="Patients enrolled")
            ax.set_title("TITE follow-up accumulation", fontsize=9)
            ax.set_xlabel("Decision step", fontsize=8)
            ax.set_ylabel("Effective patient count", fontsize=8)
            ax.legend(fontsize=7, frameon=False, labelcolor=_DARK_FG)
            compact_style(ax)
            fig.tight_layout(pad=0.5)
            st.image(fig_to_png_bytes(fig), use_container_width=True)
            st.caption(
                "Sum of fractional TITE weights across all enrolled patients "
                "at each decision point. The gap between enrolled (grey) and "
                "effective n shows how much follow-up is still pending. "
                "Tox2 (orange) lags tox1 because surgery must occur first."
            )


# ==============================================================================
# --- Design Exploration merged code start ---
# ==============================================================================
# These functions are copied/adapted from design_exploration.py.
# They are not yet wired into the main navigation; that is a future step.
# The code here is meant to be complete and importable-clean, ready for
# integration into a third "Design Exploration" view when the time comes.
# ==============================================================================


# ------------------------------------------------------------------------------
# Quality score helpers (from design_exploration.py)
# ------------------------------------------------------------------------------

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


# ------------------------------------------------------------------------------
# True-probability stress-test helpers
# ------------------------------------------------------------------------------

def generate_symmetric_values(center, spread, n, min_value=None, max_value=None):
    """Generate *n* evenly-spaced values symmetric around *center*.

    Parameters
    ----------
    center    : float — midpoint (1.0 for scaling, 0.0 for shifts)
    spread    : float — half-range; produces [center-spread … center+spread]
    n         : int   — number of points (odd → centre is exactly *center*)
    min_value : float | None — optional lower clip bound
    max_value : float | None — optional upper clip bound

    Returns
    -------
    list[float]
    """
    values = np.linspace(float(center) - float(spread),
                         float(center) + float(spread), int(n))
    lo = -np.inf if min_value is None else float(min_value)
    hi =  np.inf if max_value is None else float(max_value)
    return [float(v) for v in np.clip(values, lo, hi)]


def make_scaled_truth(true_probs, factor):
    """Return true_probs element-wise multiplied by *factor*, clipped to [0, 1]."""
    return [float(np.clip(p * factor, 0.0, 1.0)) for p in true_probs]


def fit_logistic_curve(dose_indices, true_probs):
    """Fit logistic(a + b*x) to *true_probs*, returning (a, b).

    Enforces monotonicity with np.maximum.accumulate before fitting.
    Uses np.polyfit on the logit scale.  Clamps slope b to a minimum
    of 1e-3 so the fitted curve stays non-decreasing.
    """
    eps = 1e-5
    x = np.asarray(dose_indices, dtype=float)
    y = np.asarray(true_probs, dtype=float)
    y = np.maximum.accumulate(y)          # enforce monotone non-decreasing
    y = np.clip(y, eps, 1.0 - eps)
    logit_y = np.log(y / (1.0 - y))
    b, a = np.polyfit(x, logit_y, 1)     # returns [slope, intercept]
    b = max(float(b), 1e-3)              # guard against non-positive slope
    return float(a), float(b)


def make_shifted_truth(true_probs, shift):
    """Return shifted true-probability curve via logistic horizontal shift.

    Fits logistic(a + b*x) to the (monotone-enforced) baseline, then
    evaluates the fitted curve at x + shift for each dose level x.

    shift > 0 → more toxic (curve moves left; same toxicity at lower dose).
    shift < 0 → less toxic (curve moves right).
    shift == 0 → returns the original probabilities unchanged.

    Output clipped to [1e-6, 0.95].
    """
    if shift == 0.0:
        return list(true_probs)
    dose_idx = np.arange(len(true_probs), dtype=float)
    a, b = fit_logistic_curve(dose_idx, true_probs)
    shifted = [1.0 / (1.0 + np.exp(-(a + b * (x + float(shift))))) for x in dose_idx]
    return [float(np.clip(v, 1e-6, 0.95)) for v in shifted]


def build_truth_scenarios(true_t1, true_t2, method, mode, values):
    """Build a list of (label, t1_probs, t2_probs) truth scenarios.

    Parameters
    ----------
    true_t1, true_t2 : list[float] — baseline true probabilities (length 5)
    method : "Scale probabilities" | "Curve shift"
    mode   : "Both endpoints" | "Tox1 only" | "Tox2 only"
    values : list[float] — scale factors or shift amounts

    Returns
    -------
    list of (label: str, t1: list[float], t2: list[float])
    """
    scenarios = []
    for v in values:
        if method == "Scale probabilities":
            label = f"x{v:.3g}"
            t1 = make_scaled_truth(true_t1, v) if mode in ("Both endpoints", "Tox1 only") else list(true_t1)
            t2 = make_scaled_truth(true_t2, v) if mode in ("Both endpoints", "Tox2 only") else list(true_t2)
        else:
            sign = "+" if v >= 0 else ""
            label = f"shift {sign}{v:.3g}"
            t1 = make_shifted_truth(true_t1, v) if mode in ("Both endpoints", "Tox1 only") else list(true_t1)
            t2 = make_shifted_truth(true_t2, v) if mode in ("Both endpoints", "Tox2 only") else list(true_t2)
        scenarios.append((label, t1, t2))
    return scenarios


def run_truth_stress_test(scenarios, base_ss, skel_t1, skel_t2, n_sim, seed):
    """Run TITE-CRM simulations for each truth scenario in *scenarios*.

    Parameters
    ----------
    scenarios : list of (label, t1_list, t2_list) from build_truth_scenarios
    base_ss   : dict — same format as for run_parameter_sweep
    skel_t1/t2 : list[float] — CRM skeletons
    n_sim     : int — simulations per scenario
    seed      : int — base RNG seed

    Returns
    -------
    pd.DataFrame with columns:
        scenario, scenario_raw, true_t1_L0..4, true_t2_L0..4,
        true_optimal, quality_score, pct_correct_selection,
        overdose_rate, too_high_rate, mean_selected_dose, sel_L0..4
    """
    target1 = float(base_ss["target_tox1"])
    target2 = float(base_ss["target_tox2"])
    base_kw = dict(
        p_surgery=float(base_ss["p_surgery"]),
        target1=target1, target2=target2,
        skel1=list(skel_t1), skel2=list(skel_t2),
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
        ewoc_application=str(base_ss.get("ewoc_application", EWOC_APP_BOTH)),
        n_safe_d1=int(base_ss.get("n_safe_d1", 0)),
        require_full_tox1_fu_before_escalation=bool(base_ss.get("require_full_tox1_fu", False)),
        p_stop=float(base_ss.get("p_stop", 1.0)),
        collect_trace=False,
    )
    rows = []
    n_levels = len(scenarios[0][1]) if scenarios else 5
    for s_idx, (label, t1, t2) in enumerate(scenarios):
        rng = np.random.default_rng(int(seed) + s_idx * 1000)
        t1_arr = np.asarray(t1, dtype=float)
        t2_arr = np.asarray(t2, dtype=float)
        true_opt = _true_optimal(t1_arr, t2_arr, target1, target2)
        sel_counts = [0] * n_levels
        qual_scores, overdoses, too_high = [], 0, 0
        for _ in range(int(n_sim)):
            sel, *_ = run_tite_crm(true_t1=t1_arr, true_t2=t2_arr, **base_kw, rng=rng)
            qual_scores.append(_quality_score(sel, t1_arr, t2_arr, target1, target2))
            sel_counts[sel] += 1
            if max(float(t1_arr[sel]) - target1, float(t2_arr[sel]) - target2) > 0:
                overdoses += 1
            if sel > true_opt:
                too_high += 1
        n = n_sim
        row = dict(
            scenario=label,
            scenario_raw=s_idx,
            true_optimal=true_opt + 1,
            quality_score=float(np.mean(qual_scores)),
            pct_correct_selection=100.0 * sel_counts[true_opt] / n,
            overdose_rate=100.0 * overdoses / n,
            too_high_rate=100.0 * too_high / n,
            mean_selected_dose=float(
                np.mean([i for i, c in enumerate(sel_counts) for _ in range(c)])
            ) if n > 0 else 0.0,
        )
        for li in range(n_levels):
            row[f"sel_L{li}"] = 100.0 * sel_counts[li] / n
            row[f"true_t1_L{li}"] = float(t1_arr[li])
            row[f"true_t2_L{li}"] = float(t2_arr[li])
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Parameter sweep runner (adapted from design_exploration.py)
# Note: run_tite_crm returns (selected, patients, study_days, trace,
# stopped_early); the calls below use sel, *_ to discard unused values.
# ------------------------------------------------------------------------------

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
        ewoc_application=str(base_ss.get("ewoc_application", EWOC_APP_BOTH)),
        n_safe_d1=int(base_ss.get("n_safe_d1", 0)),
        require_full_tox1_fu_before_escalation=bool(base_ss.get("require_full_tox1_fu", False)),
        p_stop=float(base_ss.get("p_stop", 1.0)),
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
            # pv is the cohort size: number of patients treated before each
            # new dose decision.  max_n (total trial size) stays fixed.
            kw["cohort_size"] = int(pv)
            label = str(int(pv))
        elif param_name == "prior_nu_t1":
            # Recompute tox1 skeleton for this prior MTD level; tox2 skeleton fixed.
            kw["skel1"] = dfcrm_getprior(
                float(base_ss["prior_hw1"]), float(base_ss["prior_pt1"]),
                int(pv), len(true_t1),
                model=str(base_ss.get("prior_model_str", "empiric")),
                intcpt=float(base_ss.get("logistic_intcpt", 3.0)),
            )
            label = f"L{int(pv)}"
        elif param_name == "prior_nu_t2":
            # Recompute tox2 skeleton for this prior MTD level; tox1 skeleton fixed.
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
            # run_tite_crm returns (selected, patients, study_days, trace, stopped_early)
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
        ewoc_application=str(base_ss.get("ewoc_application", EWOC_APP_BOTH)),
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


# ------------------------------------------------------------------------------
# Batch-report helpers
# ------------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    """Encode a matplotlib figure as a base64 PNG data-URI for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


_DE_PARAM_DISPLAY_NAMES = {
    "sigma":       "Prior sigma (σ)",
    "ewoc_alpha":  "EWOC α — overdose threshold",
    "max_n":       "Maximum sample size (max N)",
    "cohort_size": "Cohort size — patients per dose decision",
    "prior_nu_t1": "Prior MTD level — tox1 (acute)",
    "prior_nu_t2": "Prior MTD level — tox2 (subacute / surgery)",
}


def _generate_de_html_report(param_name, param_label, result_df,
                              base_ss, n_sim, seed, pv_list,
                              fig_b64, ts_str, run_label=""):
    """Build a self-contained HTML Design Exploration batch report.

    Parameters
    ----------
    param_name  : internal sweep key (e.g. "ewoc_alpha")
    param_label : display label for x-axis (e.g. "EWOC α")
    result_df   : DataFrame returned by run_parameter_sweep()
    base_ss     : the _de_base_ss dict used for the run
    n_sim       : simulations per sweep point
    seed        : base RNG seed
    pv_list     : the actual param_values list passed to run_parameter_sweep()
    fig_b64     : base64 PNG data-URI from _fig_to_b64()
    ts_str      : human-readable timestamp string
    run_label   : optional user-supplied label
    """
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

    # ── Base configuration table ──────────────────────────────────────────
    cfg_rows = [
        ("Target tox1 (acute)",           f"{base_ss['target_tox1']:.3f}"),
        ("Target tox2 (surgery/subacute)", f"{base_ss['target_tox2']:.3f}"),
        ("Prior sigma (σ)",                f"{base_ss['sigma']:.3g}"),
        ("EWOC application",               base_ss.get("ewoc_application", EWOC_APP_BOTH)),
        ("EWOC alpha",  f"{base_ss['ewoc_alpha']:.3g}" if base_ss["ewoc_on"] else "N/A"),
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

    # ── Sweep definition table ────────────────────────────────────────────
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

    # ── Results table ─────────────────────────────────────────────────────
    df_disp = result_df[["param_label", "n_patients", "quality_score",
                          "pct_correct_selection", "overdose_rate"]].copy()
    df_disp.columns = [param_label, "N patients",
                       "Quality score", "% Correct selection", "Overdose rate (%)"]
    df_disp["Quality score"]        = df_disp["Quality score"].round(4)
    df_disp["% Correct selection"]  = df_disp["% Correct selection"].round(1)
    df_disp["Overdose rate (%)"]    = df_disp["Overdose rate (%)"].round(1)
    results_html = df_disp.to_html(index=False, border=0,
                                   classes="results-tbl", justify="left")

    # ── CSS ───────────────────────────────────────────────────────────────
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


def _de_pv_for_param(param_name, ss, speed=False):
    """Return (pv_list, display_label) for *param_name* using session-state values.

    Falls back to safe defaults for any key that has not yet been set by a
    widget interaction, so the function is safe to call for parameters that
    the user has never opened in the interactive sweep UI.
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
        # Build the combined list FIRST (matching single-sweep: _de_pv[:8]),
        # then apply the speed cap on the combined list so the count is
        # identical whether Run Sweep or Run Batch is used.
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


_DE_ALL_PARAMS = [
    "sigma", "ewoc_alpha", "max_n", "cohort_size", "prior_nu_t1", "prior_nu_t2",
    "enforce_guardrail", "restrict_final_mtd", "burn_in",
]


def _generate_de_all_html_report(results_list, base_ss, n_sim, seed,
                                  ts_str, run_label=""):
    """Build a single self-contained HTML report covering all swept parameters.

    Parameters
    ----------
    results_list : list of dicts, each with keys:
        param_name, param_label, pv_list, result_df, fig_b64
    base_ss      : the base_ss dict used for the runs
    n_sim        : simulations per sweep point
    seed         : base RNG seed
    ts_str       : human-readable timestamp string
    run_label    : optional user label
    """
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

    # ── Base configuration table ──────────────────────────────────────────
    cfg_rows = [
        ("Target tox1 (acute)",            f"{base_ss['target_tox1']:.3f}"),
        ("Target tox2 (surgery/subacute)",  f"{base_ss['target_tox2']:.3f}"),
        ("Prior sigma (σ)",                 f"{base_ss['sigma']:.3g}"),
        ("EWOC application",                base_ss.get("ewoc_application", EWOC_APP_BOTH)),
        ("EWOC alpha",
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

    # ── Summary table — best point per parameter ──────────────────────────
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

    # ── Per-parameter detail sections ─────────────────────────────────────
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

    # ── CSS (same as single-param report) ─────────────────────────────────
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


# ------------------------------------------------------------------------------
# Sweep results plot helper (from design_exploration.py)
# Call: fig = _plot_sweep_results(df, param_label)
# Returns a matplotlib Figure; caller is responsible for st.pyplot / plt.close.
# ------------------------------------------------------------------------------

def _plot_sweep_results(df, param_label, param_name="", param_info=None):
    """Render the three-panel sweep results chart.

    X-axis semantics:
    - max_n       : total patient count ("18", "21", …) — directly the quantity
    - cohort_size : cohort size values (1, 2, 3, …) — patients per dose decision
    - sigma/ewoc  : parameter value; fixed N shown in xlabel
    """
    param_info = param_info or {}
    # Sort bars low-to-high by the swept parameter value so the x-axis is
    # always in ascending order regardless of selection order.
    df = df.sort_values("param_raw", ascending=True).reset_index(drop=True)
    fig, axes  = plt.subplots(1, 3, figsize=(13, 3.6))
    x          = np.arange(len(df))

    if param_name == "max_n":
        xtick_labels = [str(v) for v in df["n_patients"].tolist()]
        xlabel = "Maximum total patients in trial (max N)"
    elif param_name == "cohort_size":
        # n_patients is constant (fixed max_n); x-axis shows cohort sizes.
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
    """Light-themed version of _plot_sweep_results for embedded HTML reports.

    Identical logic to _plot_sweep_results but uses a white background and
    dark text so the chart reads cleanly in a printed / exported document.
    The app UI still uses the dark version via _plot_sweep_results.
    """
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


def _plot_stress_metrics(df, method_label, mode_label):
    """4-panel line chart: quality score, correct selection %, overdose rate %, too-high %."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    _apply_dark_fig(fig, *axes.flat)
    scenarios  = df["scenario"].tolist()
    x          = list(range(len(scenarios)))
    metrics    = [
        ("quality_score",          "Quality score",         "#4a9eff"),
        ("pct_correct_selection",  "Correct selection (%)", "#44dd88"),
        ("overdose_rate",          "Overdose rate (%)",     "#ff6666"),
        ("too_high_rate",          "Too-high selection (%)", "#ffaa44"),
    ]
    for ax, (col, title, color) in zip(axes.flat, metrics):
        ax.plot(x, df[col].tolist(), marker="o", color=color, linewidth=1.8,
                markersize=5, markeredgewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontsize=7, rotation=30, ha="right")
        ax.set_title(title, fontsize=9, color=_DARK_FG)
        ax.tick_params(colors=_DARK_FG, labelsize=7)
        compact_style(ax)
    fig.suptitle(
        f"True-probability stress test — {method_label} | {mode_label}",
        fontsize=10, color=_DARK_FG, y=1.01,
    )
    fig.tight_layout(pad=1.5)
    return fig


def _plot_stress_selection(df, dose_labels_list):
    """Stacked bar chart of dose selection % by scenario."""
    n_levels = len(dose_labels_list)
    sel_cols  = [f"sel_L{i}" for i in range(n_levels)]
    scenarios = df["scenario"].tolist()
    x         = np.arange(len(scenarios))
    colors    = ["#4a9eff", "#44dd88", "#ffaa44", "#ff6666", "#cc66ff"][:n_levels]
    fig, ax   = plt.subplots(figsize=(12, 3.6))
    _apply_dark_fig(fig, ax)
    bottom = np.zeros(len(scenarios))
    for li, col in enumerate(sel_cols):
        vals = df[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=colors[li],
               label=dose_labels_list[li], width=0.6)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Selection (%)", color=_DARK_FG, fontsize=8)
    ax.set_title("Dose selection by scenario", fontsize=9, color=_DARK_FG)
    ax.tick_params(colors=_DARK_FG, labelsize=7)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.3)
    compact_style(ax)
    fig.tight_layout(pad=1.5)
    return fig


def _plot_stress_truth_curves(scenarios, baseline_t1, baseline_t2,
                              dose_labels_list, title="", light=False):
    """Two-panel line plot: tox1 (left) and tox2 (right) true-probability curves.

    Baseline is drawn as a dashed grey line; each scenario is a solid colored
    line.  Shows the actual five dose-level probabilities passed to the
    simulator — no smoothing.

    Parameters
    ----------
    scenarios        : list of (label, t1, t2) — may be a single-element list
    baseline_t1/t2   : list[float] — Playground baseline probabilities
    dose_labels_list : list[str]   — x-axis tick labels (e.g. ['L0'…'L4'])
    title            : str         — optional suptitle
    light            : bool        — True → white bg for HTML export
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    if light:
        fig.patch.set_facecolor("white")
        ax1.set_facecolor("white")
        ax2.set_facecolor("white")
        fg = "#444444"
    else:
        _apply_dark_fig(fig, ax1, ax2)
        fg = _DARK_FG

    x = np.arange(len(dose_labels_list))
    _SC_COLORS = ["#4a9eff", "#44dd88", "#ffaa44", "#ff6666", "#cc66ff",
                  "#ff99cc", "#aaddff", "#ccff99", "#ffdd88", "#88ffee"]

    # Baseline dashed — thicker so it reads clearly against the scenario lines
    ax1.plot(x, list(baseline_t1), color="#aaaaaa", linestyle="--",
             linewidth=2.5, marker="o", markersize=5, label="Baseline", zorder=5)
    ax2.plot(x, list(baseline_t2), color="#aaaaaa", linestyle="--",
             linewidth=2.5, marker="o", markersize=5, label="Baseline", zorder=5)

    for si, (label, t1, t2) in enumerate(scenarios):
        c = _SC_COLORS[si % len(_SC_COLORS)]
        ax1.plot(x, list(t1), color=c, linewidth=1.8, marker="o", markersize=4,
                 label=label)
        ax2.plot(x, list(t2), color=c, linewidth=1.8, marker="o", markersize=4,
                 label=label)

    for ax, ep_title in [(ax1, "Tox1 (acute)"), (ax2, "Tox2 (subacute)")]:
        ax.set_xticks(x)
        ax.set_xticklabels(dose_labels_list, fontsize=8)
        ax.set_ylabel("True tox probability", color=fg, fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(ep_title, fontsize=9, color=fg)
        ax.tick_params(colors=fg, labelsize=7)
        ax.legend(fontsize=7, framealpha=0.3)
        compact_style(ax)

    if title:
        fig.suptitle(title, fontsize=9, color=fg, y=1.02)
    fig.tight_layout(pad=1.5)
    return fig


def _plot_prior_mtd_context(true_tox, pv_list, tox_label, title,
                            prior_target, prior_halfwidth,
                            model="empiric", intcpt=3.0, light=False):
    """Compact chart: true toxicity bars + one CRM skeleton line per prior MTD
    level choice in *pv_list*.  Skeletons are computed with dfcrm_getprior()
    exactly as the simulation engine uses them.

    Parameters
    ----------
    true_tox       : array-like, length 5 — true tox probability per dose level
    pv_list        : list[int] — 1-indexed MTD level values in the sweep
    tox_label      : str — e.g. "tox1 (acute)"
    title          : str — chart title
    prior_target   : float — prior target tox rate (pt)
    prior_halfwidth: float — prior halfwidth (hw)
    model          : str — "empiric" or "logistic"
    intcpt         : float — logistic intercept (used only when model="logistic")
    light          : bool — True → white bg (HTML report); False → dark bg (app)
    """
    true_tox = list(true_tox)
    n_levels = len(true_tox)
    x        = np.arange(n_levels)
    x_labels = [f"L{i + 1}" for i in range(n_levels)]

    # Compact size — similar to the Dose-risk preview panel
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
                pass  # skip invalid skeleton silently

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


# ------------------------------------------------------------------------------
# Design Exploration view — placeholder (not yet wired into navigation)
# ------------------------------------------------------------------------------
# TODO: When ready to build the Design Exploration view:
#   1. Add "Design Exploration" to the nav_view selectbox options (line ~921)
#   2. Add `elif view == "Design Exploration":` block below the Playground block
#   3. Move or inline the UI code from the stub below into that elif block
#   4. Wire base_ss to use st.session_state values (Essentials parameters as
#      starting point, with per-sweep overrides in the Design Exploration view)
#
# def _render_design_exploration_view():
#     """
#     Stub for the Design Exploration view.
#     Implements the parameter sweep UI from design_exploration.py,
#     adapted to use session-state values from the Essentials view as defaults.
#     """
#     st.markdown("#### Parameter sweep")
#     _N_LEVELS = 5
#
#     # ── Sweep parameter selector ──────────────────────────────────────────
#     param_name = st.selectbox(
#         "Parameter to sweep",
#         ["sigma", "ewoc_alpha", "max_n", "cohort_size"],
#         format_func={
#             "sigma":       "Sigma (prior precision)",
#             "ewoc_alpha":  "EWOC α (overdose threshold)",
#             "max_n":       "Max N (sample size)",
#             "cohort_size": "Cohort size",
#         }.get,
#         key="de_param_name",
#     )
#
#     # ── Sweep range controls (per parameter) ──────────────────────────────
#     if param_name == "sigma":
#         c1, c2, c3 = st.columns(3)
#         sig_min = c1.number_input("Min σ", 0.1, 4.9, 0.3, step=0.1, key="de_sig_min")
#         sig_max = c2.number_input("Max σ", sig_min + 0.1, 5.0, 2.0, step=0.1, key="de_sig_max")
#         sig_pts = c3.slider("Points", 3, 20, 8, key="de_sig_pts")
#         _param_values = np.linspace(sig_min, sig_max, sig_pts).tolist()
#         param_label   = "Sigma (σ)"
#         param_type    = "continuous"
#     elif param_name == "ewoc_alpha":
#         c1, c2, c3 = st.columns(3)
#         ea_min = c1.number_input("Min α", 0.05, 0.55, 0.15, step=0.01, key="de_ea_min")
#         ea_max = c2.number_input("Max α", ea_min + 0.01, 0.60, 0.45, step=0.01, key="de_ea_max")
#         ea_pts = c3.slider("Points", 3, 20, 8, key="de_ea_pts")
#         inc_off = st.checkbox("Include EWOC OFF", value=True, key="de_inc_off")
#         _param_values = ([None] if inc_off else []) + \
#                         np.linspace(ea_min, ea_max, ea_pts).tolist()
#         param_label   = "EWOC α"
#         param_type    = "continuous"
#     elif param_name == "max_n":
#         _param_values = st.multiselect(
#             "Max N values", [12, 15, 18, 21, 24, 27, 30, 33, 36],
#             default=[18, 21, 24, 27, 30], key="de_max_n_vals")
#         param_label = "Max N"
#         param_type  = "discrete"
#     else:  # cohort_size
#         _param_values = st.multiselect(
#             "Cohort sizes", [1, 2, 3, 4, 5, 6], default=[1, 2, 3, 4],
#             key="de_cohort_vals")
#         param_label = "Cohort size"
#         param_type  = "discrete"
#
#     st.divider()
#     n_sim_input = st.slider("n_sim (per point)", 50, 2000, 200, step=50, key="de_n_sim")
#     seed_val    = st.number_input("Seed", 0, 99999, 42, step=1, key="de_seed")
#     speed_mode  = st.checkbox("Speed mode", key="de_speed_mode",
#                               help="Cuts n_sim to max(50, n_sim÷4); "
#                                    "caps grid to ≤8 continuous / ≤3 discrete.")
#     if speed_mode:
#         n_sim_eff = max(50, int(n_sim_input) // 4)
#         pv_eff    = (_param_values[:8] if param_type == "continuous"
#                      else _param_values[:3])
#         st.caption(f"Speed mode — {n_sim_eff} sims × {len(pv_eff)} points")
#     else:
#         n_sim_eff = int(n_sim_input)
#         pv_eff    = _param_values
#         st.caption(f"{n_sim_eff} sims × {len(pv_eff)} points = "
#                    f"{n_sim_eff * len(pv_eff):,} total trials")
#
#     # ── Skeleton computation (reuse Playground session-state priors) ───────
#     try:
#         skel_t1 = dfcrm_getprior(
#             st.session_state["halfwidth_t1"], st.session_state["prior_target_t1"],
#             st.session_state["prior_nu_t1"], _N_LEVELS)
#         skel_t2 = dfcrm_getprior(
#             st.session_state["halfwidth_t2"], st.session_state["prior_target_t2"],
#             st.session_state["prior_nu_t2"], _N_LEVELS)
#         _skel_ok = True
#     except Exception as e:
#         st.error(f"Skeleton error: {e}")
#         _skel_ok = False
#         skel_t1 = skel_t2 = None
#
#     true_t1_arr = np.array([st.session_state[k] for k in TRUE_T1_KEYS])
#     true_t2_arr = np.array([st.session_state[k] for k in TRUE_T2_KEYS])
#
#     if _skel_ok:
#         opt = _true_optimal(true_t1_arr, true_t2_arr,
#                             st.session_state["target_t1"],
#                             st.session_state["target_t2"])
#         st.info(f"True optimal dose: **L{opt}** — "
#                 f"Tox1 = {true_t1_arr[opt]:.3f}, Tox2 = {true_t2_arr[opt]:.3f}")
#
#     run_btn = st.button("▶ Run Sweep", type="primary",
#                         key="de_run_btn",
#                         disabled=not _skel_ok or len(pv_eff) == 0)
#
#     # ── Sweep execution ────────────────────────────────────────────────────
#     if run_btn and _skel_ok and len(pv_eff) > 0:
#         base_ss = dict(
#             target_tox1=st.session_state["target_t1"],
#             target_tox2=st.session_state["target_t2"],
#             p_surgery=st.session_state["p_surgery"],
#             sigma=st.session_state["sigma"],
#             ewoc_on=st.session_state["ewoc_on"],
#             ewoc_alpha=st.session_state["ewoc_alpha"],
#             max_n=st.session_state["max_n_crm"],
#             cohort_size=st.session_state["cohort_size"],
#             start_level=st.session_state["start_level_1b"],
#             accrual_per_month=st.session_state["accrual_per_month"],
#             incl_to_rt=st.session_state["incl_to_rt"],
#             rt_dur=st.session_state["rt_dur"],
#             rt_to_surg=st.session_state["rt_to_surg"],
#             tox2_win=st.session_state["tox2_win"],
#             max_step=st.session_state["max_step"],
#             gh_n=st.session_state["gh_n"],
#             burn_in=st.session_state["burn_in"],
#             enforce_guardrail=st.session_state["enforce_guardrail"],
#             restrict_final_to_tried=st.session_state["restrict_final_mtd"],
#         )
#         with st.spinner(f"Running {n_sim_eff * len(pv_eff):,} trials…"):
#             df = run_parameter_sweep(
#                 param_name, pv_eff, base_ss,
#                 true_t1_arr, true_t2_arr, skel_t1, skel_t2,
#                 n_sim_eff, int(seed_val))
#         st.session_state["_de_df"]    = df
#         st.session_state["_de_label"] = param_label
#
#     # ── Results display ────────────────────────────────────────────────────
#     if "_de_df" in st.session_state:
#         df  = st.session_state["_de_df"]
#         lbl = st.session_state["_de_label"]
#         fig = _plot_sweep_results(df, lbl)
#         st.pyplot(fig)
#         plt.close(fig)
#         disp = df[["param_label", "quality_score",
#                    "pct_correct_selection", "overdose_rate"]].copy()
#         disp.columns = [lbl, "Quality score", "% Correct selection",
#                         "Overdose rate (%)"]
#         disp["Quality score"]        = disp["Quality score"].round(4)
#         disp["% Correct selection"]  = disp["% Correct selection"].round(1)
#         disp["Overdose rate (%)"]    = disp["Overdose rate (%)"].round(1)
#         st.dataframe(disp, use_container_width=True, hide_index=True)

# ==============================================================================
# --- Design Exploration merged code end ---
# ==============================================================================

# ==============================================================================
# DESIGN EXPLORATION VIEW
# ==============================================================================
# Placed here (after helper function definitions) so _quality_score,
# _true_optimal, run_parameter_sweep and _plot_sweep_results are already
# defined when this block executes.
# ==============================================================================

if view == "Design Exploration":
    _N_LEVELS = 5
    _ss = st.session_state

    st.markdown("#### Design Exploration — TITE-CRM Parameter Sweep")
    st.caption(
        "Varies one design parameter while keeping all others fixed at the "
        "current Essentials / Playground settings.  Uses TITE-CRM only."
    )

    # ── Compute skeletons from Playground priors ──────────────────────────
    try:
        _de_pt1  = float(get_config_value("prior_target_t1"))
        _de_hw1  = float(get_config_value("halfwidth_t1"))
        _de_nu1  = int  (get_config_value("prior_nu_t1"))
        _de_pt2  = float(get_config_value("prior_target_t2"))
        _de_hw2  = float(get_config_value("halfwidth_t2"))
        _de_nu2  = int  (get_config_value("prior_nu_t2"))
        _de_sk1  = dfcrm_getprior(_de_hw1, _de_pt1, _de_nu1, _N_LEVELS)
        _de_sk2  = dfcrm_getprior(_de_hw2, _de_pt2, _de_nu2, _N_LEVELS)
        _de_skel_ok = True
    except Exception as _de_err:
        st.error(f"Skeleton error (check Playground priors): {_de_err}")
        _de_skel_ok = False
        _de_sk1 = _de_sk2 = None

    # ── True toxicity arrays from Playground session state ────────────────
    _de_t1 = np.array([float(get_config_value(k)) for k in TRUE_T1_KEYS])
    _de_t2 = np.array([float(get_config_value(k)) for k in TRUE_T2_KEYS])

    # ── True optimal dose + baseline summary ──────────────────────────────
    if _de_skel_ok:
        _de_tgt1 = float(get_config_value("target_t1"))
        _de_tgt2 = float(get_config_value("target_t2"))
        _de_opt  = _true_optimal(_de_t1, _de_t2, _de_tgt1, _de_tgt2)
        _ewoc_str = (f"ON α={float(get_config_value('ewoc_alpha')):.2f}"
                     if bool(get_config_value("ewoc_on")) else "OFF")
        st.markdown(
            f'<div style="background:#1e3a5f;border-left:4px solid #4a9eff;'
            f'padding:12px 16px;border-radius:0 4px 4px 0;margin:8px 0;">'
            f'<span style="color:#d0e8ff;font-size:0.95em;">'
            f'True optimal dose (highest quality score): '
            f'<strong>L{_de_opt}</strong> — '
            f'Tox1&nbsp;=&nbsp;{_de_t1[_de_opt]:.3f}, '
            f'&nbsp;Tox2&nbsp;=&nbsp;{_de_t2[_de_opt]:.3f}'
            f'</span><br>'
            f'<span style="color:#90b8d8;font-size:0.88em;">'
            f'Baseline — σ&nbsp;=&nbsp;{float(_cfg("sigma")):.2f}'
            f'&nbsp;·&nbsp;EWOC&nbsp;{_ewoc_str}'
            f'&nbsp;·&nbsp;max&nbsp;N&nbsp;=&nbsp;{int(_cfg("max_n_crm"))}'
            f'&nbsp;·&nbsp;cohort&nbsp;=&nbsp;{int(_cfg("cohort_size"))}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        # ── Debug: per-dose loss table ─────────────────────────────────────
        with st.expander("Debug — per-dose loss / quality scores", expanded=False):
            _de_debug_rows = []
            for _di in range(len(_de_t1)):
                _d1  = float(_de_t1[_di]) - _de_tgt1
                _d2  = float(_de_t2[_di]) - _de_tgt2
                _bd  = max(_d1, _d2)
                _w   = 1.8 if _bd > 0 else 1.0
                _loss = _w * abs(_bd)
                _qs   = float(np.exp(-6.0 * _loss))
                _de_debug_rows.append({
                    "Dose": f"L{_di}",
                    "True Tox1": round(float(_de_t1[_di]), 3),
                    "True Tox2": round(float(_de_t2[_di]), 3),
                    "diff1 (t1−target)": round(_d1, 3),
                    "diff2 (t2−target)": round(_d2, 3),
                    "binding diff": round(_bd, 3),
                    "weight": _w,
                    "loss": round(_loss, 4),
                    "quality score": round(_qs, 4),
                    "optimal ★": "★" if _di == _de_opt else "",
                })
            st.dataframe(pd.DataFrame(_de_debug_rows), hide_index=True,
                         use_container_width=True)
            st.caption(
                "loss = weight × |binding_diff|  ·  "
                "weight = 1.8 if overdose, 1.0 if underdose  ·  "
                "quality_score = exp(−6 × loss)"
            )

    # ── Exploration type selector ─────────────────────────────────────────
    st.session_state["wl_de_expl_type"] = str(get_config_value("de_expl_type"))
    _de_expl_type = st.radio(
        "Exploration type",
        options=["Design parameter sweep", "True probability stress test"],
        horizontal=True,
        key="wl_de_expl_type",
        on_change=_make_sync("de_expl_type", str, "wl_de_expl_type"),
    )
    st.session_state["de_expl_type"] = st.session_state["wl_de_expl_type"]

    # Sentinels — prevent NameError when the matching controls block doesn't render
    _de_run_btn    = False
    _de_batch_btn  = False
    _de_st_run_btn = False
    _de_pv_eff     = []

    if _de_expl_type == "Design parameter sweep":
        # ── Sweep controls ────────────────────────────────────────────────────
        # Short description shown under the selectbox for each parameter.
        _DE_PARAM_DESCRIPTIONS = {
            "sigma": (
                "Varying the **prior uncertainty** on the dose-toxicity slope "
                "(σ on θ).  A larger σ = wider prior = faster dose escalation.  "
                "Max N and cohort size stay fixed."
            ),
            "ewoc_alpha": (
                "Varying the **EWOC overdose threshold** α.  α is the maximum "
                "acceptable posterior probability that the selected dose exceeds "
                "the target toxicity rate.  Smaller α = more conservative.  "
                "Max N and cohort size stay fixed."
            ),
            "max_n": (
                "Varying the **maximum total number of patients** in the trial.  "
                "Cohort size and all other settings stay fixed."
            ),
            "cohort_size": (
                "Varying the **cohort size**: how many patients are enrolled and "
                "observed before the model makes the next dose-level decision.  "
                "The total max N stays fixed — only the decision frequency changes."
            ),
            "prior_nu_t1": (
                "Varying the **prior MTD level for tox1 (acute toxicity)**.  "
                "This is the dose level the CRM skeleton is centred on before "
                "any data are observed.  L1 = lowest dose, L5 = highest dose.  "
                "All other settings stay fixed."
            ),
            "prior_nu_t2": (
                "Varying the **prior MTD level for tox2 (subacute / surgery toxicity)**.  "
                "This is the dose level the CRM skeleton is centred on before "
                "any data are observed.  L1 = lowest dose, L5 = highest dose.  "
                "All other settings stay fixed."
            ),
            "enforce_guardrail": (
                "Comparing **guardrail OFF vs ON**.  "
                "When ON, the next recommended dose cannot skip untried levels "
                "(next dose ≤ highest tried + 1).  "
                "Turning OFF allows the CRM to escalate more aggressively.  "
                "All other settings stay fixed."
            ),
            "restrict_final_mtd": (
                "Comparing **final MTD restriction OFF vs ON**.  "
                "When ON, the trial's final MTD recommendation is constrained to "
                "dose levels that were actually tested on at least one patient.  "
                "All other settings stay fixed."
            ),
            "burn_in": (
                "Comparing **burn-in OFF vs ON**.  "
                "When ON, the CRM holds at the starting dose until the first "
                "tox1 DLT is observed before switching to model-guided escalation.  "
                "All other settings stay fixed."
            ),
        }

        _de_ctrl, _ = st.columns([2, 3])
        with _de_ctrl:
            _de_param = st.selectbox(
                "Parameter to sweep",
                ["sigma", "ewoc_alpha", "max_n", "cohort_size",
                 "prior_nu_t1", "prior_nu_t2",
                 "enforce_guardrail", "restrict_final_mtd", "burn_in"],
                format_func={
                    "sigma":              "Prior sigma (σ)",
                    "ewoc_alpha":         "EWOC α — overdose threshold",
                    "max_n":              "Maximum sample size (max N)",
                    "cohort_size":        "Cohort size — patients per dose decision",
                    "prior_nu_t1":        "Prior MTD level — tox1 (acute)",
                    "prior_nu_t2":        "Prior MTD level — tox2 (subacute / surgery)",
                    "enforce_guardrail":  "Safety: guardrail (≤ highest tried + 1)",
                    "restrict_final_mtd": "Safety: final MTD restricted to tried doses",
                    "burn_in":            "Behaviour: burn-in until first tox1 DLT",
                }.get,
                key="de_param_name",
                help=(
                    "Choose which single design parameter to vary.  "
                    "All other parameters are held at their current "
                    "Essentials / Playground values."
                ),
            )
            # Per-parameter explanatory caption
            st.caption(_DE_PARAM_DESCRIPTIONS[_de_param])

            # Initialise DE widget defaults once (avoids key+value conflict).
            # EWOC alpha keys (de_ea_*) are intentionally excluded here — they
            # are initialised via explicit value= parameters on their widgets,
            # which is the only pattern that survives Streamlit's widget-state
            # cleanup when switching between sweep parameters (≥1.28).
            for _dk, _dv in [("de_sig_min", 0.3), ("de_sig_max", 2.0),
                              ("de_sig_pts", 8),
                              ("de_n_sim",  200), ("de_seed",    42),
                              ("de_nu1_vals", [1, 2, 3, 4, 5]),
                              ("de_nu2_vals", [1, 2, 3, 4, 5])]:
                if _dk not in _ss:
                    _ss[_dk] = _dv

            # Clamp de_sig_max so its stored value stays above the dynamic
            # min_value when de_sig_min is raised.
            _sig_min_floor = round(float(_ss["de_sig_min"]) + 0.1, 1)
            if float(_ss["de_sig_max"]) < _sig_min_floor:
                _ss["de_sig_max"] = min(_sig_min_floor, 5.0)

            # Clamp de_ea_max only when the key already exists in session state
            # (i.e. the EWOC widgets have been rendered at least once).  On the
            # first render the widget initialises itself via its value= parameter.
            if "de_ea_min" in _ss and "de_ea_max" in _ss:
                _ea_min_floor = round(float(_ss["de_ea_min"]) + 0.01, 2)
                if float(_ss["de_ea_max"]) < _ea_min_floor:
                    _ss["de_ea_max"] = min(_ea_min_floor, 0.99)

            if _de_param == "sigma":
                _c1, _c2, _c3 = st.columns(3)
                _de_sig_min = _c1.number_input(
                    "Min σ", 0.1, 4.9, step=0.1, key="de_sig_min",
                    help="Lower bound of the σ sweep. σ controls how peaked the "
                         "CRM prior is — larger σ means a wider, more uncertain prior.")
                _de_sig_max = _c2.number_input(
                    "Max σ", float(max(_de_sig_min + 0.1, 0.2)), 5.0,
                    step=0.1, key="de_sig_max",
                    help="Upper bound of the σ sweep range.")
                _de_sig_pts = _c3.slider(
                    "Points", 3, 20, key="de_sig_pts",
                    help="Number of evenly-spaced σ values between Min and Max.")
                _de_pv      = np.linspace(_de_sig_min, _de_sig_max,
                                          _de_sig_pts).tolist()
                _de_label   = "σ (prior sigma)"
                _de_ptype   = "continuous"

            elif _de_param == "ewoc_alpha":
                _c1, _c2, _c3 = st.columns(3)
                # Use explicit value= for all four controls.  This is the only
                # pattern that reliably survives Streamlit's widget-state cleanup
                # (≥1.28): when switching away from the EWOC param and back,
                # Streamlit removes widget-backed keys from session state, so any
                # init-loop pre-write would be lost before the widget renders.
                # value=_ss.get(key, default) ensures the correct default on first
                # render while honouring any user-edited value on subsequent runs.
                _de_ea_min  = _c1.number_input(
                    "Min α", 0.05, 0.97,
                    value=float(_ss.get("de_ea_min", 0.05)),
                    step=0.01, key="de_ea_min",
                    help="Lower bound of the EWOC overdose-probability threshold "
                         "sweep. α is the maximum acceptable probability of "
                         "recommending a dose above the MTD.")
                _ea_max_min = float(max(_de_ea_min + 0.01, 0.06))
                _de_ea_max  = _c2.number_input(
                    "Max α", _ea_max_min, 0.99,
                    value=float(max(_ss.get("de_ea_max", 0.60), _ea_max_min)),
                    step=0.01, key="de_ea_max",
                    help="Upper bound of the EWOC α sweep range.")
                _de_ea_pts  = _c3.slider(
                    "Points", 3, 20,
                    value=int(_ss.get("de_ea_pts", 8)),
                    key="de_ea_pts",
                    help="Number of evenly-spaced α values between Min and Max.")
                _de_inc_off = st.checkbox(
                    "Include EWOC OFF as a point",
                    value=bool(_ss.get("de_inc_off", True)),
                    key="de_inc_off",
                    help="Also run the design with EWOC disabled. Useful to "
                         "quantify how much safety benefit EWOC provides.")
                _de_pv      = (([None] if _de_inc_off else []) +
                               np.linspace(_de_ea_min, _de_ea_max,
                                           _de_ea_pts).tolist())
                _de_label   = "EWOC α"
                _de_ptype   = "continuous"

            elif _de_param == "max_n":
                _de_mn_baseline = int(get_config_value("max_n_crm"))
                st.caption(
                    f"Baseline max N from Essentials = **{_de_mn_baseline}**.  "
                    "Select the total patient counts you want to compare."
                )
                _de_pv = st.multiselect(
                    "Total patient counts to sweep",
                    [12, 15, 18, 21, 24, 27, 30, 33, 36],
                    default=[12, 15, 18, 21, 24, 27, 30, 33, 36],
                    key="de_max_n_vals",
                    help="Select the maximum total patient counts to compare. "
                         "Larger N gives the CRM more data but lengthens the trial.",
                )
                _de_label = "Maximum total patients (max N)"
                _de_ptype = "discrete"

            elif _de_param == "prior_nu_t1":
                st.caption(
                    f"Current tox1 prior MTD level: **L{_de_nu1}** (from Playground).  "
                    "Select the levels you want to compare."
                )
                _de_pv = st.multiselect(
                    "Tox1 prior MTD levels to sweep (L1 = lowest, L5 = highest)",
                    [1, 2, 3, 4, 5],
                    default=_ss["de_nu1_vals"],
                    key="de_nu1_vals",
                    help="Select which prior MTD level assumptions to compare for "
                         "acute tox1. The assumed level shifts the CRM skeleton — "
                         "a lower assumed level makes the model more conservative.",
                )
                _de_label = "Prior MTD level — tox1 (acute)"
                _de_ptype = "discrete"

            elif _de_param == "prior_nu_t2":
                st.caption(
                    f"Current tox2 prior MTD level: **L{_de_nu2}** (from Playground).  "
                    "Select the levels you want to compare."
                )
                _de_pv = st.multiselect(
                    "Tox2 prior MTD levels to sweep (L1 = lowest, L5 = highest)",
                    [1, 2, 3, 4, 5],
                    default=_ss["de_nu2_vals"],
                    key="de_nu2_vals",
                    help="Select which prior MTD level assumptions to compare for "
                         "subacute tox2 (surgery endpoint). The assumed level "
                         "shapes how the CRM weights subacute toxicity risk.",
                )
                _de_label = "Prior MTD level — tox2 (subacute / surgery)"
                _de_ptype = "discrete"

            elif _de_param in ("enforce_guardrail", "restrict_final_mtd", "burn_in"):
                _bool_labels = {
                    "enforce_guardrail":  "Guardrail (next dose ≤ highest tried + 1)",
                    "restrict_final_mtd": "Final MTD restricted to tried doses",
                    "burn_in":            "Burn-in until first tox1 DLT",
                }
                _de_label = _bool_labels[_de_param]
                _de_pv    = [False, True]
                _de_ptype = "discrete"
                st.caption("Sweeps both **OFF** and **ON** (2 sweep points).")

            else:  # cohort_size
                _de_mn_baseline = int(get_config_value("max_n_crm"))
                st.caption(
                    f"Max N fixed at **{_de_mn_baseline}** (from Essentials).  "
                    "Select the cohort sizes you want to compare."
                )
                _de_pv = st.multiselect(
                    "Cohort sizes to sweep (patients per dose decision)",
                    [1, 2, 3, 4, 5, 6],
                    default=[1, 2, 3, 4],
                    key="de_cohort_vals",
                    help="Select the number of patients enrolled per dose "
                         "decision. Larger cohorts provide more information per "
                         "step but slow down dose escalation.",
                )
                _de_label = "Cohort size (patients per dose decision)"
                _de_ptype = "discrete"

            st.divider()
            _de_n_sim  = st.slider(
                "Simulations per point", 50, 2000, step=50, key="de_n_sim",
                help="Number of simulated trials run at each parameter value. "
                     "Higher values give more stable estimates but take longer.")
            _de_seed   = st.number_input(
                "Seed", 0, 99999, step=1, key="de_seed",
                help="Random seed for reproducibility. Changing the seed varies "
                     "the simulated patient outcomes across runs.")
            _de_speed  = st.checkbox(
                "Speed mode (faster, less precise)",
                key="de_speed_mode",
                help="Reduces simulations per point to max(50, n÷4) and caps "
                     "the grid to ≤8 continuous / ≤3 discrete points. "
                     "Results are approximate.",
            )

            if _de_speed:
                _de_n_eff = max(50, int(_de_n_sim) // 4)
                _de_pv_eff = (_de_pv[:8] if _de_ptype == "continuous"
                              else _de_pv[:3])
                st.caption(
                    f":orange[Speed mode — {_de_n_eff} sims × "
                    f"{len(_de_pv_eff)} points (approximate)]"
                )
            else:
                _de_n_eff  = int(_de_n_sim)
                _de_pv_eff = list(_de_pv)
                if _de_pv_eff:
                    st.caption(
                        f"{_de_n_eff} sims × {len(_de_pv_eff)} points = "
                        f"{_de_n_eff * len(_de_pv_eff):,} total trials"
                    )

            _de_run_btn = st.button(
                "▶ Run Sweep", type="primary", key="de_run_btn",
                disabled=(not _de_skel_ok or len(_de_pv_eff) == 0),
            )
            _de_batch_btn = st.button(
                "▶▶ Run Batch Exploration",
                type="primary",
                key="de_batch_all_btn",
                disabled=not _de_skel_ok,
                help=(
                    "Sweeps all six parameters in sequence using the values "
                    "configured above (or defaults) and saves a combined HTML report."
                ),
            )

    # end if _de_expl_type == "Design parameter sweep" sweep controls

    elif _de_expl_type == "True probability stress test":
        # ── Stress test controls ─────────────────────────────────────────────
        _st_c1, _st_c2 = st.columns(2)
        with _st_c1:
            st.session_state["wl_de_st_method"] = str(get_config_value("de_st_method"))
            _de_st_method = st.selectbox(
                "Stress method",
                options=["Scale probabilities", "Curve shift"],
                key="wl_de_st_method",
                on_change=_make_sync("de_st_method", str, "wl_de_st_method"),
                help=(
                    "**Scale probabilities**: multiply each true probability by a factor.  \n"
                    "**Curve shift**: fit a logistic curve and shift it horizontally — "
                    "positive = more toxic (toxicity appears at lower dose levels)."
                ),
            )
            st.session_state["de_st_method"] = st.session_state["wl_de_st_method"]
        with _st_c2:
            st.session_state["wl_de_st_mode"] = str(get_config_value("de_st_mode"))
            _de_st_mode = st.selectbox(
                "Endpoint mode",
                options=["Both endpoints", "Tox1 only", "Tox2 only"],
                key="wl_de_st_mode",
                on_change=_make_sync("de_st_mode", str, "wl_de_st_mode"),
                help="Which toxicity endpoint(s) to stress-test.",
            )
            st.session_state["de_st_mode"] = st.session_state["wl_de_st_mode"]

        # ── Slider-based scenario generation ─────────────────────────────────
        _st_sl_c1, _st_sl_c2 = st.columns([3, 1])
        _n_sc_stored = int(get_config_value("de_st_n_scenarios"))
        _n_sc_stored = _n_sc_stored if _n_sc_stored in [3, 5, 7] else 5
        if _de_st_method == "Scale probabilities":
            with _st_sl_c1:
                _spread_val = float(np.clip(get_config_value("de_st_scale_spread"), 0.0, 1.0))
                st.session_state["wl_de_st_scale_spread"] = _spread_val
                _de_st_spread = st.slider(
                    "Scaling spread",
                    min_value=0.0, max_value=1.0, step=0.05,
                    key="wl_de_st_scale_spread",
                    on_change=_make_sync("de_st_scale_spread", float, "wl_de_st_scale_spread"),
                    help=(
                        "Half-range around baseline (1.0).  "
                        "Spread 0.5 → scale factors from 0.5 to 1.5."
                    ),
                )
                st.session_state["de_st_scale_spread"] = float(
                    st.session_state.get("wl_de_st_scale_spread", _de_st_spread)
                )
            with _st_sl_c2:
                st.session_state["wl_de_st_n_scenarios"] = _n_sc_stored
                _de_st_n_sc = st.selectbox(
                    "Scenarios",
                    options=[3, 5, 7],
                    key="wl_de_st_n_scenarios",
                    on_change=_make_sync("de_st_n_scenarios", int, "wl_de_st_n_scenarios"),
                    help="Number of stress scenarios (always includes the baseline).",
                )
                st.session_state["de_st_n_scenarios"] = int(
                    st.session_state.get("wl_de_st_n_scenarios", _de_st_n_sc)
                )
            _de_st_values = generate_symmetric_values(
                center=1.0, spread=_de_st_spread, n=_de_st_n_sc,
                min_value=0.05, max_value=2.0,
            )
            st.caption(
                "Generated scale factors: "
                + ", ".join(f"{v:.2f}" for v in _de_st_values)
            )
        else:  # Curve shift
            with _st_sl_c1:
                _spread_val = float(np.clip(get_config_value("de_st_shift_spread"), 0.0, 2.0))
                st.session_state["wl_de_st_shift_spread"] = _spread_val
                _de_st_spread = st.slider(
                    "Curve shift spread",
                    min_value=0.0, max_value=2.0, step=0.1,
                    key="wl_de_st_shift_spread",
                    on_change=_make_sync("de_st_shift_spread", float, "wl_de_st_shift_spread"),
                    help=(
                        "Half-range around baseline (0.0) in dose levels.  "
                        "Spread 1.0 → shifts from −1.0 to +1.0.  "
                        "Positive shift = more toxic."
                    ),
                )
                st.session_state["de_st_shift_spread"] = float(
                    st.session_state.get("wl_de_st_shift_spread", _de_st_spread)
                )
            with _st_sl_c2:
                st.session_state["wl_de_st_n_scenarios"] = _n_sc_stored
                _de_st_n_sc = st.selectbox(
                    "Scenarios",
                    options=[3, 5, 7],
                    key="wl_de_st_n_scenarios",
                    on_change=_make_sync("de_st_n_scenarios", int, "wl_de_st_n_scenarios"),
                    help="Number of stress scenarios (always includes the baseline).",
                )
                st.session_state["de_st_n_scenarios"] = int(
                    st.session_state.get("wl_de_st_n_scenarios", _de_st_n_sc)
                )
            _de_st_values = generate_symmetric_values(
                center=0.0, spread=_de_st_spread, n=_de_st_n_sc,
                min_value=-2.0, max_value=2.0,
            )
            st.caption(
                "Generated curve shifts: "
                + ", ".join(f"{v:.2f}" for v in _de_st_values)
            )

        # Advanced: custom scenario values (hidden by default)
        with st.expander("Advanced: custom scenario values", expanded=False):
            if _de_st_method == "Scale probabilities":
                st.session_state["wl_de_st_scale_str"] = str(_cfg("de_st_scale_str"))
                _st_custom_raw = st.text_input(
                    "Custom scale factors (comma-separated)",
                    key="wl_de_st_scale_str",
                    help="If non-empty, overrides the slider-generated values.",
                )
                st.session_state["de_st_scale_str"] = str(
                    st.session_state.get("wl_de_st_scale_str", "")
                )
            else:
                st.session_state["wl_de_st_shift_str"] = str(_cfg("de_st_shift_str"))
                _st_custom_raw = st.text_input(
                    "Custom shift amounts (comma-separated)",
                    key="wl_de_st_shift_str",
                    help="If non-empty, overrides the slider-generated values.",
                )
                st.session_state["de_st_shift_str"] = str(
                    st.session_state.get("wl_de_st_shift_str", "")
                )
            if _st_custom_raw.strip():
                try:
                    _custom_vals = [
                        float(v.strip()) for v in _st_custom_raw.split(",") if v.strip()
                    ]
                    if _custom_vals:
                        _de_st_values = _custom_vals
                        st.caption(
                            "Using custom values: "
                            + ", ".join(f"{v:.3g}" for v in _de_st_values)
                        )
                except ValueError:
                    st.warning("Could not parse custom values — using slider-generated values.")

            # Debug: horizontal shift verification (curve shift only)
            if _de_st_method == "Curve shift":
                st.markdown("**Horizontal shift check** — shift = +1.0")
                st.caption(
                    "Shifted L_i should equal baseline fitted L_{i+1}.  "
                    "Small deviations are expected due to linear logit approximation."
                )
                _dbx = np.arange(5, dtype=float)
                _db_a1, _db_b1 = fit_logistic_curve(_dbx, _de_t1)
                _db_a2, _db_b2 = fit_logistic_curve(_dbx, _de_t2)
                _db_sh_t1 = make_shifted_truth(_de_t1, 1.0)
                _db_sh_t2 = make_shifted_truth(_de_t2, 1.0)
                _db_rows = []
                for _li in range(4):
                    _db_fit_t1 = float(1.0 / (1.0 + np.exp(-(_db_a1 + _db_b1 * (_li + 1)))))
                    _db_fit_t2 = float(1.0 / (1.0 + np.exp(-(_db_a2 + _db_b2 * (_li + 1)))))
                    _db_rows.append({
                        "Dose": f"L{_li}",
                        "Shifted T1": f"{_db_sh_t1[_li]:.4f}",
                        "Fitted base T1 (L+1)": f"{_db_fit_t1:.4f}",
                        "T1 ✓": "✓" if abs(_db_sh_t1[_li] - _db_fit_t1) < 0.02 else "≈",
                        "Shifted T2": f"{_db_sh_t2[_li]:.4f}",
                        "Fitted base T2 (L+1)": f"{_db_fit_t2:.4f}",
                        "T2 ✓": "✓" if abs(_db_sh_t2[_li] - _db_fit_t2) < 0.02 else "≈",
                    })
                st.dataframe(pd.DataFrame(_db_rows), hide_index=True,
                             use_container_width=True)

        # ── Simulations / seed ────────────────────────────────────────────────
        _st_sim_c1, _st_sim_c2 = st.columns(2)
        with _st_sim_c1:
            st.session_state["wl_de_st_n_sim"] = int(_cfg("de_st_n_sim"))
            _de_st_n_sim = st.number_input(
                "Simulations", min_value=10, max_value=5000,
                step=50, key="wl_de_st_n_sim",
                on_change=_make_sync("de_st_n_sim", int, "wl_de_st_n_sim"),
            )
            st.session_state["de_st_n_sim"] = int(
                st.session_state.get("wl_de_st_n_sim", _de_st_n_sim)
            )
        with _st_sim_c2:
            st.session_state["wl_de_st_seed"] = int(_cfg("de_st_seed"))
            _de_st_seed = st.number_input(
                "Seed", min_value=0, max_value=99999,
                step=1, key="wl_de_st_seed",
                on_change=_make_sync("de_st_seed", int, "wl_de_st_seed"),
            )
            st.session_state["de_st_seed"] = int(
                st.session_state.get("wl_de_st_seed", _de_st_seed)
            )

        _de_st_run_btn = st.button(
            "Run stress test",
            type="primary",
            disabled=not _de_skel_ok or len(_de_st_values) == 0,
            help="Run TITE-CRM simulations for each stress scenario.",
        )

        # ── Live truth-curve preview ──────────────────────────────────────────
        if _de_skel_ok and _de_st_values:
            _prev_sc = build_truth_scenarios(
                _de_t1, _de_t2,
                method=_de_st_method,
                mode=_de_st_mode,
                values=_de_st_values,
            )
            _PREV_DOSE_LBLS = [f"L{i}" for i in range(5)]
            with st.expander("📈 Truth curve preview", expanded=True):
                if len(_prev_sc) <= 8:
                    _prev_fig = _plot_stress_truth_curves(
                        _prev_sc, _de_t1, _de_t2, _PREV_DOSE_LBLS,
                    )
                    st.image(fig_to_png_bytes(_prev_fig), use_container_width=True)
                    plt.close(_prev_fig)
                else:
                    _prev_sc_labels = [sc[0] for sc in _prev_sc]
                    _prev_chosen = st.selectbox(
                        "Scenario to preview",
                        options=_prev_sc_labels,
                        key="wl_de_st_preview_sel",
                        help="Select a scenario to preview its truth curves.",
                    )
                    _prev_one = [sc for sc in _prev_sc if sc[0] == _prev_chosen]
                    _prev_fig = _plot_stress_truth_curves(
                        _prev_one, _de_t1, _de_t2, _PREV_DOSE_LBLS,
                        title=f"Scenario: {_prev_chosen}",
                    )
                    st.image(fig_to_png_bytes(_prev_fig), use_container_width=True)
                    plt.close(_prev_fig)
                if _de_st_method == "Curve shift":
                    st.caption(
                        "Positive shift = more toxic: the baseline curve moves left, "
                        "so the same toxicity probability appears at a lower dose level.  "
                        "Negative shift = less toxic: the curve moves right."
                    )

    # ── Execute sweep ─────────────────────────────────────────────────────
    if _de_run_btn and _de_skel_ok and len(_de_pv_eff) > 0:
        _de_base_ss = dict(
            target_tox1          = float(get_config_value("target_t1")),
            target_tox2          = float(get_config_value("target_t2")),
            p_surgery            = float(_cfg("p_surgery")),
            sigma                = float(_cfg("sigma")),
            ewoc_on              = bool(_cfg("ewoc_on")),
            ewoc_alpha           = float(get_config_value("ewoc_alpha")),
            ewoc_application     = str(_cfg("ewoc_application")),
            max_n                = int(_cfg("max_n_crm")),
            cohort_size          = int(_cfg("cohort_size")),
            start_level          = int(_cfg("start_level_1b")),
            accrual_per_month    = float(_cfg("accrual_per_month")),
            incl_to_rt           = int(_cfg("incl_to_rt")),
            rt_dur               = int(_cfg("rt_dur")),
            rt_to_surg           = int(_cfg("rt_to_surg")),
            tox2_win             = int(_cfg("tox2_win")),
            max_step             = int(_cfg("max_step")),
            gh_n                 = int(_cfg("gh_n")),
            burn_in              = bool(_cfg("burn_in")),
            require_full_tox1_fu = bool(_cfg("require_full_tox1_fu")),
            enforce_guardrail    = bool(_cfg("enforce_guardrail")),
            restrict_final_to_tried = bool(_cfg("restrict_final_mtd")),
            # Prior params (needed when sweeping prior_nu_t1 / prior_nu_t2)
            prior_pt1            = _de_pt1,
            prior_hw1            = _de_hw1,
            prior_pt2            = _de_pt2,
            prior_hw2            = _de_hw2,
            prior_model_str      = str(get_config_value("prior_model")),
            logistic_intcpt      = float(get_config_value("logistic_intcpt")),
        )
        _de_total = _de_n_eff * len(_de_pv_eff)
        with st.spinner(f"Running {_de_total:,} trials…"):
            _de_result_df = run_parameter_sweep(
                _de_param, _de_pv_eff, _de_base_ss,
                _de_t1, _de_t2, _de_sk1, _de_sk2,
                _de_n_eff, int(_de_seed),
            )
        _ss["_de_df"]        = _de_result_df
        _ss["_de_label"]     = _de_label
        _ss["_de_param_key"] = _de_param   # needed so results block can pass it
        # Store cohort_size used during this run for plot annotation
        _ss["_de_cs_used"]   = int(get_config_value("cohort_size"))

    # ── Metric help strings ───────────────────────────────────────────────
    _HELP_QS = (
        "**Quality score** (0–1, higher = better)\n\n"
        "Measures how close the model-selected dose is to the true optimum, "
        "using an asymmetric exponential loss:\n\n"
        "1. For each trial compute diff1 = true_tox1[selected] − target_tox1 "
        "and diff2 = true_tox2[selected] − target_tox2.\n"
        "2. binding_diff = max(diff1, diff2)  — the worst endpoint.\n"
        "3. weight = **1.8** if binding_diff > 0 (overdose), **1.0** if ≤ 0 (underdose).\n"
        "4. loss = weight × |binding_diff|\n"
        "5. score = exp(−6 × loss)\n\n"
        "A score of 1.0 means the selected dose is exactly at target; "
        "overdoses are penalised 1.8× more than underdoses. "
        "Average over all simulations."
    )
    _HELP_CS = (
        "**% Correct selection** (0–100 %, higher = better)\n\n"
        "Percentage of simulated trials in which the TITE-CRM selected the "
        "true optimal dose level.\n\n"
        "The true optimal is defined as the dose with the **highest quality score** "
        "(same asymmetric loss as above). This is consistent with the quality "
        "score metric — both agree on what 'best' means.\n\n"
        "A higher value means the algorithm reliably homes in on the right dose."
    )
    _HELP_OR = (
        "**Overdose rate** (0–100 %, lower = better)\n\n"
        "Percentage of simulated trials in which the selected dose exceeded "
        "at least one toxicity target:\n\n"
        "  overdose = max(true_tox1[selected] − target_tox1,\n"
        "                 true_tox2[selected] − target_tox2) > 0\n\n"
        "An overdose does **not** mean a catastrophic outcome — it means the "
        "selected dose is slightly above the target probability. "
        "The EWOC constraint is the primary guardrail against large overdoses. "
        "Lower is safer."
    )

    # ── Display results ───────────────────────────────────────────────────
    if "_de_df" in _ss:
        _df       = _ss["_de_df"]
        _lbl      = _ss["_de_label"]
        _pkey     = _ss.get("_de_param_key", "")
        _p_info   = {"cohort_size": _ss.get("_de_cs_used", int(get_config_value("cohort_size")))}

        _fig = _plot_sweep_results(_df, _lbl, _pkey, param_info=_p_info)
        st.pyplot(_fig)
        plt.close(_fig)

        # Extra explanatory chart for prior MTD level sweeps
        if _pkey in ("prior_nu_t1", "prior_nu_t2"):
            _ctx_swept  = sorted(int(v) for v in _df["param_raw"].unique())
            _ctx_model  = str(get_config_value("prior_model"))
            _ctx_intcpt = float(get_config_value("logistic_intcpt"))
            if _pkey == "prior_nu_t1":
                _ctx_true  = _de_t1
                _ctx_pt    = _de_pt1
                _ctx_hw    = _de_hw1
                _ctx_label = "tox1 (acute)"
                _ctx_title = "Tox1 acute: true toxicity vs prior MTD level choices"
            else:
                _ctx_true  = _de_t2
                _ctx_pt    = _de_pt2
                _ctx_hw    = _de_hw2
                _ctx_label = "tox2 (subacute)"
                _ctx_title = "Tox2 subacute: true toxicity vs prior MTD level choices"
            _ctx_fig = _plot_prior_mtd_context(
                _ctx_true, _ctx_swept, _ctx_label, _ctx_title,
                _ctx_pt, _ctx_hw,
                model=_ctx_model, intcpt=_ctx_intcpt, light=False)
            # Render at a fixed compact width (similar to dose-risk preview)
            st.image(fig_to_png_bytes(_ctx_fig), width=480)

        # Metric header row with help tooltips
        _mc1, _mc2, _mc3 = st.columns(3)
        _mc1.markdown("**Quality score**",
                      help=_HELP_QS)
        _mc2.markdown("**% Correct selection**",
                      help=_HELP_CS)
        _mc3.markdown("**Overdose rate (%)**",
                      help=_HELP_OR)

    # ── Batch execution ────────────────────────────────────────────────────
    if _de_batch_btn and _de_skel_ok:
        _bts       = datetime.datetime.now()
        _bts_str   = _bts.strftime("%Y-%m-%d %H:%M:%S")
        _auto_fname = f"design_exploration_{_bts.strftime('%Y%m%d_%H%M%S')}.html"
        _out_abs    = os.path.abspath(os.path.join("outputs", _auto_fname))
        os.makedirs(os.path.dirname(_out_abs), exist_ok=True)

        _all_base_ss = dict(
            target_tox1             = float(get_config_value("target_t1")),
            target_tox2             = float(get_config_value("target_t2")),
            p_surgery               = float(_cfg("p_surgery")),
            sigma                   = float(_cfg("sigma")),
            ewoc_on                 = bool(_cfg("ewoc_on")),
            ewoc_alpha              = float(get_config_value("ewoc_alpha")),
            ewoc_application        = str(_cfg("ewoc_application")),
            max_n                   = int(_cfg("max_n_crm")),
            cohort_size             = int(_cfg("cohort_size")),
            start_level             = int(_cfg("start_level_1b")),
            accrual_per_month       = float(_cfg("accrual_per_month")),
            incl_to_rt              = int(_cfg("incl_to_rt")),
            rt_dur                  = int(_cfg("rt_dur")),
            rt_to_surg              = int(_cfg("rt_to_surg")),
            tox2_win                = int(_cfg("tox2_win")),
            max_step                = int(_cfg("max_step")),
            gh_n                    = int(_cfg("gh_n")),
            burn_in                 = bool(_cfg("burn_in")),
            require_full_tox1_fu    = bool(_cfg("require_full_tox1_fu")),
            enforce_guardrail       = bool(_cfg("enforce_guardrail")),
            restrict_final_to_tried = bool(_cfg("restrict_final_mtd")),
            prior_pt1               = _de_pt1,
            prior_hw1               = _de_hw1,
            prior_pt2               = _de_pt2,
            prior_hw2               = _de_hw2,
            prior_model_str         = str(get_config_value("prior_model")),
            logistic_intcpt         = float(get_config_value("logistic_intcpt")),
        )

        # ── Debug: show exactly which sweep values will be used ──────────
        # Pre-compute all grids so the user can verify before simulations run.
        _batch_plan = {
            _pn: _de_pv_for_param(_pn, _ss, speed=_de_speed)
            for _pn in _DE_ALL_PARAMS
        }
        with st.expander("🔍 Batch sweep plan (click to verify values)", expanded=False):
            for _pn, (_ppv, _ppl) in _batch_plan.items():
                if _ppv:
                    _ppv_str = ", ".join(
                        "OFF" if v is None
                        else ("ON" if v is True else ("OFF" if v is False
                              else f"{v:.4g}"))
                        for v in _ppv
                    )
                    st.caption(f"**{_ppl}** ({len(_ppv)} pts): {_ppv_str}")
                else:
                    st.caption(f"**{_ppl}**: *(empty — skipped)*")
        # ─────────────────────────────────────────────────────────────────

        _all_prog    = st.progress(0, text="Starting…")
        _n_params    = len(_DE_ALL_PARAMS)
        _all_results = []

        for _pi, _pname in enumerate(_DE_ALL_PARAMS):
            _pv, _plabel = _batch_plan[_pname]   # already computed above
            if not _pv:
                continue
            _all_prog.progress(
                _pi / _n_params,
                text=f"Sweeping {_plabel} ({_pi + 1}/{_n_params})…",
            )
            with st.spinner(f"Sweeping {_plabel}…"):
                _pres_df = run_parameter_sweep(
                    _pname, _pv, _all_base_ss,
                    _de_t1, _de_t2, _de_sk1, _de_sk2,
                    _de_n_eff, int(_de_seed),
                )
            _pinfo    = {"cohort_size": int(get_config_value("cohort_size"))}
            _pfig     = _plot_sweep_results_light(_pres_df, _plabel, _pname,
                                                  param_info=_pinfo)
            _pfig_b64 = _fig_to_b64(_pfig)
            plt.close(_pfig)

            # Prior-MTD context chart (light theme, for the HTML report)
            _pctx_b64 = None
            if _pname in ("prior_nu_t1", "prior_nu_t2"):
                _ctx_true  = _de_t1 if _pname == "prior_nu_t1" else _de_t2
                _ctx_pt    = (_all_base_ss["prior_pt1"] if _pname == "prior_nu_t1"
                              else _all_base_ss["prior_pt2"])
                _ctx_hw    = (_all_base_ss["prior_hw1"] if _pname == "prior_nu_t1"
                              else _all_base_ss["prior_hw2"])
                _ctx_lbl   = ("tox1 (acute)"
                              if _pname == "prior_nu_t1"
                              else "tox2 (subacute)")
                _ctx_title = ("Tox1 acute: true toxicity vs prior MTD level choices"
                              if _pname == "prior_nu_t1"
                              else "Tox2 subacute: true toxicity vs prior MTD level choices")
                _ctx_fig   = _plot_prior_mtd_context(
                    _ctx_true, [int(v) for v in _pv],
                    _ctx_lbl, _ctx_title,
                    _ctx_pt, _ctx_hw,
                    model=_all_base_ss["prior_model_str"],
                    intcpt=float(_all_base_ss["logistic_intcpt"]),
                    light=True)
                _pctx_b64  = _fig_to_b64(_ctx_fig)
                plt.close(_ctx_fig)

            _all_results.append(dict(
                param_name      = _pname,
                param_label     = _plabel,
                pv_list         = _pv,
                result_df       = _pres_df,
                fig_b64         = _pfig_b64,
                context_fig_b64 = _pctx_b64,
            ))

        _all_prog.progress(0.90, text="Generating report…")
        _all_html = _generate_de_all_html_report(
            results_list = _all_results,
            base_ss      = _all_base_ss,
            n_sim        = _de_n_eff,
            seed         = int(_de_seed),
            ts_str       = _bts_str,
            run_label    = "",
        )

        _all_prog.progress(0.95, text="Saving…")
        with open(_out_abs, "w", encoding="utf-8") as _f:
            _f.write(_all_html)

        # Save CSV silently alongside the HTML
        _csv_path = _out_abs.replace(".html", ".csv")
        _csv_parts = []
        for _r in _all_results:
            _rdf = _r["result_df"].copy()
            _rdf.insert(0, "parameter",    _r["param_name"])
            _rdf.insert(1, "param_display", _r["param_label"])
            _csv_parts.append(_rdf)
        pd.concat(_csv_parts, ignore_index=True).to_csv(_csv_path, index=False)

        _all_prog.progress(1.0, text="Done!")
        st.success(f"✅ **Batch run complete.**")

        _ss["_de_batch_html"]     = _all_html
        _ss["_de_batch_filename"] = _auto_fname

        # Auto-download via JS data-URI
        _dl_b64   = base64.b64encode(_all_html.encode("utf-8")).decode("ascii")
        _dl_fname = _auto_fname.replace("'", "\\'")
        _stcv1.html(
            f"""<!DOCTYPE html><html><body style="margin:0;padding:0">
<script>(function(){{
  var a = document.createElement('a');
  a.href = 'data:text/html;base64,{_dl_b64}';
  a.download = '{_dl_fname}';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}})();</script>
</body></html>""",
            height=0,
        )

    # Fallback download button — persists after any successful batch run
    if "_de_batch_html" in _ss:
        st.download_button(
            "⬇ Download report",
            data=_ss["_de_batch_html"].encode("utf-8"),
            file_name=_ss.get("_de_batch_filename", "design_exploration.html"),
            mime="text/html",
            key="de_batch_dl_btn",
        )
        st.caption(
            "Your report was generated.  "
            "If the download did not start automatically, "
            "click the button above to download it."
        )

    # ── Execute stress test ───────────────────────────────────────────────
    if _de_st_run_btn and _de_skel_ok:
        _de_st_base_ss = dict(
            target_tox1             = float(get_config_value("target_t1")),
            target_tox2             = float(get_config_value("target_t2")),
            p_surgery               = float(_cfg("p_surgery")),
            sigma                   = float(_cfg("sigma")),
            ewoc_on                 = bool(_cfg("ewoc_on")),
            ewoc_alpha              = float(get_config_value("ewoc_alpha")),
            ewoc_application        = str(_cfg("ewoc_application")),
            max_n                   = int(_cfg("max_n_crm")),
            cohort_size             = int(_cfg("cohort_size")),
            start_level             = int(_cfg("start_level_1b")),
            accrual_per_month       = float(_cfg("accrual_per_month")),
            incl_to_rt              = int(_cfg("incl_to_rt")),
            rt_dur                  = int(_cfg("rt_dur")),
            rt_to_surg              = int(_cfg("rt_to_surg")),
            tox2_win                = int(_cfg("tox2_win")),
            max_step                = int(_cfg("max_step")),
            gh_n                    = int(_cfg("gh_n")),
            burn_in                 = bool(_cfg("burn_in")),
            require_full_tox1_fu    = bool(_cfg("require_full_tox1_fu")),
            enforce_guardrail       = bool(_cfg("enforce_guardrail")),
            restrict_final_to_tried = bool(_cfg("restrict_final_mtd")),
        )
        _de_st_scenarios = build_truth_scenarios(
            _de_t1, _de_t2,
            method=str(get_config_value("de_st_method")),
            mode=str(get_config_value("de_st_mode")),
            values=_de_st_values,
        )
        _de_st_total = len(_de_st_scenarios) * int(get_config_value("de_st_n_sim"))
        with st.spinner(f"Running {_de_st_total:,} stress-test trials…"):
            _de_st_result_df = run_truth_stress_test(
                _de_st_scenarios, _de_st_base_ss,
                _de_sk1, _de_sk2,
                int(get_config_value("de_st_n_sim")),
                int(get_config_value("de_st_seed")),
            )
        _ss["_de_st_df"]     = _de_st_result_df
        _ss["_de_st_method"] = str(get_config_value("de_st_method"))
        _ss["_de_st_mode"]   = str(get_config_value("de_st_mode"))

    # ── Display stress test results ───────────────────────────────────────
    if "_de_st_df" in _ss and _de_expl_type == "True probability stress test":
        _st_df   = _ss["_de_st_df"]
        _st_meth = _ss.get("_de_st_method", "")
        _st_mode = _ss.get("_de_st_mode", "")

        st.markdown("#### Scenario true probabilities")
        _t1_cols = [f"true_t1_L{i}" for i in range(5)]
        _t2_cols = [f"true_t2_L{i}" for i in range(5)]
        _disp_cols = ["scenario"] + _t1_cols + _t2_cols + ["true_optimal"]
        _st_disp = _st_df[_disp_cols].copy()
        _st_disp.columns = (
            ["Scenario"] +
            [f"T1 L{i+1}" for i in range(5)] +
            [f"T2 L{i+1}" for i in range(5)] +
            ["True optimal (1-idx)"]
        )
        st.dataframe(
            _st_disp.style.format({c: "{:.3f}" for c in _st_disp.columns if c.startswith("T")}),
            hide_index=True, use_container_width=True,
        )

        # Truth curve chart — reconstruct scenarios from stored DataFrame
        _st_scenarios_from_df = [
            (
                row["scenario"],
                [row[f"true_t1_L{i}"] for i in range(5)],
                [row[f"true_t2_L{i}"] for i in range(5)],
            )
            for _, row in _st_df.iterrows()
        ]
        _RES_DOSE_LBLS = [f"L{i}" for i in range(5)]
        _MAX_RES_COMPACT = 8
        if len(_st_scenarios_from_df) <= _MAX_RES_COMPACT:
            _res_curve_fig = _plot_stress_truth_curves(
                _st_scenarios_from_df, _de_t1, _de_t2, _RES_DOSE_LBLS,
            )
            st.image(fig_to_png_bytes(_res_curve_fig), use_container_width=True)
            plt.close(_res_curve_fig)
        else:
            _res_sc_labels = [sc[0] for sc in _st_scenarios_from_df]
            _res_chosen = st.selectbox(
                "Scenario curves to display",
                options=["All"] + _res_sc_labels,
                key="wl_de_st_res_curve_sel",
                help="Show all scenarios or zoom into one.",
            )
            if _res_chosen == "All":
                _res_subset = _st_scenarios_from_df
            else:
                _res_subset = [sc for sc in _st_scenarios_from_df if sc[0] == _res_chosen]
            _res_curve_fig = _plot_stress_truth_curves(
                _res_subset, _de_t1, _de_t2, _RES_DOSE_LBLS,
                title=("" if _res_chosen == "All" else f"Scenario: {_res_chosen}"),
            )
            st.image(fig_to_png_bytes(_res_curve_fig), use_container_width=True)
            plt.close(_res_curve_fig)

        st.markdown("#### Metrics by scenario")
        _metric_cols = ["scenario", "quality_score", "pct_correct_selection",
                        "overdose_rate", "too_high_rate", "mean_selected_dose"]
        _st_metric_disp = _st_df[_metric_cols].copy()
        _st_metric_disp.columns = [
            "Scenario", "Quality score", "Correct sel. (%)",
            "Overdose rate (%)", "Too-high sel. (%)", "Mean selected dose (0-idx)",
        ]
        st.dataframe(
            _st_metric_disp.style.format({c: "{:.2f}" for c in _st_metric_disp.columns if c != "Scenario"}),
            hide_index=True, use_container_width=True,
        )

        _st_metrics_fig = _plot_stress_metrics(_st_df, _st_meth, _st_mode)
        st.pyplot(_st_metrics_fig)
        plt.close(_st_metrics_fig)

        _st_sel_fig = _plot_stress_selection(_st_df, dose_labels)
        st.pyplot(_st_sel_fig)
        plt.close(_st_sel_fig)
