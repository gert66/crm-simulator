"""
sim_tite.py — TITE dual-endpoint dose-escalation simulator
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

import numpy as np
import pandas as pd
import streamlit as st
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
      cohort update (posteriors, weights, allowed doses, decision reason).
      Adds negligible runtime; used only for the first simulated trial.

    Returns (selected_level, patients_list, study_days, trace).
      trace is a list of dicts (one per cohort decision) when collect_trace=True,
      otherwise an empty list.
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

        # Enroll cohort: each patient arrives after an Exp(1/rate) inter-arrival
        for _ in range(n_add):
            current_day += rng.exponential(1.0 / rate_per_day)
            pt = make_patient(rng, level, current_day,
                              true_t1, p_surgery, true_t2,
                              incl_to_rt, rt_dur, rt_to_surg, tox1_win, tox2_win)
            patients.append(pt)

        # Decision time = calendar day when last patient in cohort arrived
        decision_day  = current_day
        highest_tried = max(highest_tried, level)

        # Compute fractional TITE weights for all enrolled patients
        n1, y1, n2, y2 = tite_weights(
            patients, decision_day, tox1_win, tox2_win, n_levels)

        burn_was_active = burn_active

        # Burn-in: check if any tox1 event has been observed by now
        if burn_active:
            obs_any_dlt = any(
                p["has_tox1"] and p["tox1_day"] is not None
                and p["tox1_day"] <= decision_day
                for p in patients
            )
            if obs_any_dlt:
                burn_active = False

        if burn_active:
            # No DLT observed yet — escalate one level
            next_level = min(level + 1, n_levels - 1)
            if next_level == n_levels - 1:
                burn_active = False   # reached top, switch to CRM next round
        else:
            next_level = crm_choose_next(
                sigma, skel1, skel2,
                n1, y1, n2, y2,
                level, target1, target2,
                ewoc_alpha=ewoc_eff, max_step=max_step, gh_n=gh_n,
                enforce_guardrail=enforce_guardrail,
                highest_tried=highest_tried, n_levels=n_levels,
            )

        # ── Collect trace for this decision (first trial only) ────────────────
        if collect_trace:
            pm1, od1 = crm_posterior_summaries(
                sigma, skel1, n1, y1, target1, gh_n=gh_n)
            pm2, od2 = crm_posterior_summaries(
                sigma, skel2, n2, y2, target2, gh_n=gh_n)

            # EWOC mode label for the trace
            ewoc_mode = "OFF" if ewoc_eff is None else f"ON (α={ewoc_eff:.2f})"

            # Which doses pass the joint EWOC safety filter?
            if ewoc_eff is None:
                # EWOC OFF: all doses are candidates
                allowed_arr = list(range(n_levels))
            else:
                allowed_arr = [int(d) for d in
                               np.where((od1 < ewoc_eff) & (od2 < ewoc_eff))[0]]

            # Human-readable reason for the dose selected
            if burn_was_active:
                reason = (f"Burn-in: escalate one level (no tox1 DLT "
                          f"observed yet → L{next_level})")
            elif not allowed_arr:
                reason = "No dose within joint safety bounds → fallback to L0"
            elif ewoc_eff is None:
                # EWOC OFF: closest-to-target1 rule
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
            })

        cohort_step += 1
        level = next_level

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
  div[data-baseweb="select"] { background-color: #0f3460 !important; color: #e0e0e0 !important; }
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
    # Playground prior-endpoint tab (must be initialised so the slider
    # conditional never sees an undefined key on first Playground load)
    "prior_ep_tab":       "Tox1 (acute)",
    # Playground prior scenario selector
    "prior_scenario":     "Neutral",
}

TRUE_T1_KEYS  = [f"true_t1_L{i}"  for i in range(5)]
TRUE_T2_KEYS  = [f"true_t2_L{i}"  for i in range(5)]

# Single merged defaults registry — the ONE source of all default values.
# true_t1/t2 are intentionally EXCLUDED: those number_input widgets supply
# value=DEFAULT_TRUE_T* so Streamlit seeds session_state on first render.
# Pre-seeding them here via init_state() causes Streamlit ≥1.31 to reset
# the displayed value to min_value (0.0) instead of the intended default.
_ALL_DEFAULTS: dict = {
    **R_DEFAULTS,
}

# Separate read-only fallback for true-tox keys, used only by get_config_value
# so Design Exploration works even before Playground has been visited once.
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

_STATE_VERSION = "2026-03-22b"

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
            st.session_state.get(wl_key, R_DEFAULTS[canonical_key])
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
# Essentials right column
_sync_gh_n              = _make_sync("gh_n",              int,   "wl_gh_n")
_sync_max_step          = _make_sync("max_step",          int,   "wl_max_step")
_sync_sigma             = _make_sync("sigma",             float, "sl_sigma")
_sync_enforce_guardrail = _make_sync("enforce_guardrail", bool,  "wl_enforce_guardrail")
_sync_restrict_final_mtd= _make_sync("restrict_final_mtd",bool,  "wl_restrict_final_mtd")
_sync_burn_in           = _make_sync("burn_in",           bool,  "wl_burn_in")
_sync_ewoc_on           = _make_sync("ewoc_on",           bool,  "wl_ewoc_on")
_sync_ewoc_alpha        = _make_sync("ewoc_alpha",        float, "wl_ewoc_alpha")
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
    scenario = str(st.session_state.get("wl_prior_scenario", "Neutral"))
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
    Single-row horizontal timeline with labelled coloured bands.
    Returns a matplotlib Figure.

    Tox1 window is derived as rt_dur + rt_to_surg, so it ends exactly at
    surgery — the same derivation used in the simulation.
    """
    _BG = _DARK_BG
    _FG = _DARK_FG

    fig, ax = plt.subplots(figsize=(9.0, 0.9), dpi=120)
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

    y0, h_bar = 0.25, 0.50
    # Each entry: (x_start, x_end, facecolor, alpha, legend_label)
    # Saturated palette chosen to contrast well on the light background.
    segments = [
        (0,        rt_start, "#64b5f6", 0.85, "Incl → RT start"),   # medium blue
        (rt_start, rt_end,   "#1565c0", 0.90, "RT"),                 # dark blue
        (rt_start, t1_end,   "#ff8f00", 0.55, "Tox1 window"),        # amber overlay
        (rt_end,   surg,     "#78909c", 0.80, "RT end → Surgery"),   # blue-grey
        (surg,     t2_end,   "#c62828", 0.85, "Tox2 window"),        # dark red
    ]
    plotted_labels = set()
    handles = []
    for x0, x1, col, alpha, lbl in segments:
        bar = mpatches.FancyBboxPatch(
            (x(x0), y0), x(x1) - x(x0), h_bar,
            boxstyle="square,pad=0",
            facecolor=col, alpha=alpha,
            edgecolor="#bdbdbd", linewidth=0.5,
        )
        ax.add_patch(bar)
        if lbl not in plotted_labels:
            handles.append(mpatches.Patch(facecolor=col, alpha=alpha, label=lbl))
            plotted_labels.add(lbl)

    markers = [(0, "Incl"), (rt_start, "RT\nstart"), (rt_end, "RT\nend"),
               (surg, "Surgery"), (t2_end, "Done")]
    for d, lbl in markers:
        xp = x(d)
        ax.axvline(xp, ymin=0.15, ymax=0.85,
                   color=_FG, lw=0.8, alpha=0.6)
        ax.text(xp, 0.04, lbl, ha="center", va="bottom",
                fontsize=7.0, color=_FG, fontweight="bold")

    legend = ax.legend(handles=handles, loc="upper right", fontsize=7,
                       frameon=False, ncol=len(handles),
                       bbox_to_anchor=(1, 1.15))
    plt.setp(legend.get_texts(), color=_FG)

    fig.tight_layout(pad=0.15)
    return fig

# ==============================================================================
# Navigation (sidebar)
# ==============================================================================

view = st.sidebar.radio(
    "View",
    options=["Essentials", "Playground", "Design Exploration"],
    key="nav_view",
    label_visibility="collapsed",
)

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

        st.session_state["wl_start_level_1b"] = int(_cfg("start_level_1b"))
        st.number_input(
            "Start dose level (1-based)",
            min_value=1, max_value=5, step=1, key="wl_start_level_1b",
            on_change=_sync_start_level_1b,
            help=h("start_level_1b", "Starting dose level (1 = lowest).")
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
                   "Days from end of radiotherapy to surgery. Default 84 days ≈ 12 weeks. "
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

        # ── ewoc_on ───────────────────────────────────────────────────────
        st.session_state["wl_ewoc_on"] = bool(_cfg("ewoc_on"))
        st.toggle(
            "Enable EWOC joint overdose control",
            key="wl_ewoc_on",
            on_change=_sync_ewoc_on,
            help=h("ewoc_on",
                   "Restrict doses where BOTH P(tox1 OD) and P(tox2 OD) < EWOC alpha.")
        )
        # Post-read immediately so ewoc_alpha disabled= sees the updated value.
        st.session_state["ewoc_on"] = st.session_state["wl_ewoc_on"]

        # ── ewoc_alpha ────────────────────────────────────────────────────
        st.session_state["wl_ewoc_alpha"] = float(_cfg("ewoc_alpha"))
        st.number_input(
            "EWOC alpha",
            min_value=0.01, max_value=0.99, step=0.01, key="wl_ewoc_alpha",
            on_change=_sync_ewoc_alpha,
            disabled=(not bool(_cfg("ewoc_on"))),
            help=h("ewoc_alpha",
                   "EWOC threshold applied to both endpoints independently.")
        )
        st.session_state["ewoc_alpha"] = st.session_state["wl_ewoc_alpha"]

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
    st.info(
        "**Modified 6+3 (TITE version) — full evaluability required.**\n\n"
        "Decisions are only made once ALL enrolled patients in the evaluation "
        "cohort have completed their relevant follow-up windows.\n\n"
        "**Bridging rule:** while waiting for evaluability at the current "
        "dose, new arrivals are assigned to the next lower dose (*safe dose*). "
        "These bridging patients count toward the trial total but not toward "
        "the formal evaluation cohort.\n\n"
        "**Rate-based acute thresholds:** if the HOLD rule causes more than 6 "
        "(or 9) patients to be enrolled at eval dose, the acute threshold "
        "is scaled proportionally to preserve the original protocol ratio.",
        icon="ℹ️",
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
    st.button("Reset to defaults", on_click=_do_reset)

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
                # value= seeds session_state on first render (safe because
                # TRUE_T1_KEYS are NOT pre-seeded by init_state()).
                v1 = st.number_input(
                    f"T1 L{i}",
                    min_value=0.0, max_value=1.0, step=0.01,
                    value=DEFAULT_TRUE_T1[i],
                    key=TRUE_T1_KEYS[i],
                    label_visibility="collapsed",
                    help=f"True probability of acute toxicity at dose L{i}.",
                )
                true_t1.append(float(v1))
            with rT2:
                v2 = st.number_input(
                    f"T2 L{i}",
                    min_value=0.0, max_value=1.0, step=0.01,
                    value=DEFAULT_TRUE_T2[i],
                    key=TRUE_T2_KEYS[i],
                    label_visibility="collapsed",
                    help=f"True probability of subacute toxicity given surgery at L{i}.",
                )
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
        run = st.button("Run simulations", use_container_width=True)

    # ── Mid: Priors ───────────────────────────────────────────────────────────
    with mid:
        st.markdown("#### Priors")

        # ── Skeleton model ────────────────────────────────────────────────
        st.radio(
            "Skeleton model",
            options=["empiric", "logistic"],
            horizontal=True, key="prior_model",
            help=h("prior_model", "Skeleton generation method, shared for both endpoints.")
        )
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
            ep_tab = st.radio(
                "Endpoint",
                options=["Tox1 (acute)", "Tox2 (subacute | surgery)"],
                horizontal=True, key="prior_ep_tab",
                help="Switch between tox1 and tox2 prior parameter sets.",
            )

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
                st.slider("Prior MTD level (tox1, 1-based)", 1, 5, step=1,
                          key="sl_prior_nu_t1",
                          on_change=_sync_prior_nu_t1,
                          help=h("prior_nu_t1",
                                 "Dose level that is a priori closest to the tox1 target. "
                                 "L1 = most cautious, L5 = most optimistic."))
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
                st.slider("Prior MTD level (tox2, 1-based)", 1, 5, step=1,
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
        start_0b   = int(np.clip(int(_cfg("start_level_1b")) - 1, 0, 4))

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
            selc, ptsc, sdc, trace_s = run_tite_crm(
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
                burn_in      = bool(_cfg("burn_in")),
                rng=rng_s2, **timing_kw,
                collect_trace=(s == 0),   # record full trace for first trial only
            )
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
                    "restrict_final_mtd": bool(_cfg("restrict_final_mtd")),
                }
            sel_crm[selc] += 1
            for p in ptsc:
                nmat_crm[s, p["dose"]]  += 1
                nsurg_crm[s, p["dose"]] += int(p["has_surgery"])
                yacrm[s]  += int(p["has_tox1"])
                yscrm[s]  += int(p["has_tox2"])
                nscrm[s]  += int(p["has_surgery"])
            dur_crm[s] = sdc

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

        st.caption(
            f"Surgery rate: 6+3={res['surg_rate_63']:.3f}  "
            f"CRM={res['surg_rate_crm']:.3f}  (expected ≈ {res['p_surgery']:.2f})"
        )
        st.caption(
            f"n_sims={res['ns']} | seed={res['seed']}"
            + (f" | True safe=L{ts}" if ts is not None else " | No jointly safe dose")
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
    _ewoc_on_flag = bool(_cfg("ewoc_on"))
    _ewoc_alpha   = float(_cfg("ewoc_alpha"))
    if _ewoc_on_flag:
        st.caption(
            f"**EWOC ON (α = {_ewoc_alpha:.2f})** — At each decision the model filters "
            "doses to those where P(tox1 > target) < α **and** P(tox2 > target) < α "
            "(joint safety rule). The **highest** jointly admissible dose is then selected, "
            "subject to max-step and guardrail constraints."
        )
    else:
        st.caption(
            "**EWOC OFF** — No overdose-probability filter is applied. "
            "Among all doses (subject to step and guardrail constraints), the model picks "
            "the dose whose posterior mean P(tox1) is **closest to target1** "
            "(standard CRM argmin rule). This is target-based and does not automatically "
            "escalate to the highest dose."
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
            _w2v      = _w2(_pt, _dec_day)
            _w2v_str  = ("—" if _w2v is None else
                         round(_w2v, 2) if _dec_day is not None else "—")
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
    _ewoc_on_tr  = _tr.get("ewoc_on",  bool(_cfg("ewoc_on")))
    _ewoc_a_tr   = _tr.get("ewoc_alpha", float(_cfg("ewoc_alpha")))
    _restr_tr    = _tr.get("restrict_final_mtd", bool(_cfg("restrict_final_mtd")))
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
                f"**EWOC ON (α = {_ewoc_a_tr:.2f})**: doses admitted only where "
                f"P(tox1 OD) < α **and** P(tox2 OD) < α.  "
                f"Among admitted {'tried ' if _restr_tr else ''}doses, "
                f"the **highest** is selected."
            )
        else:
            _sel_rule_str = (
                "**EWOC OFF**: no overdose-probability filter.  "
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
            if bool(_cfg("ewoc_on")):
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
# Parameter sweep runner (adapted from design_exploration.py)
# Note: run_tite_crm in this file returns (selected, patients, study_days,
# trace); the call below unpacks only the first element accordingly.
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
#             start_level=st.session_state["start_level_1b"] - 1,
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
        st.info(
            f"True optimal dose (highest quality score): **L{_de_opt}** — "
            f"Tox1 = {_de_t1[_de_opt]:.3f},  Tox2 = {_de_t2[_de_opt]:.3f}\n\n"
            f"Baseline — σ = {float(_cfg('sigma')):.2f} · EWOC {_ewoc_str} · "
            f"max N = {int(_cfg('max_n_crm'))} · cohort = {int(_cfg('cohort_size'))}"
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
    }

    _de_ctrl, _ = st.columns([2, 3])
    with _de_ctrl:
        _de_param = st.selectbox(
            "Parameter to sweep",
            ["sigma", "ewoc_alpha", "max_n", "cohort_size",
             "prior_nu_t1", "prior_nu_t2"],
            format_func={
                "sigma":        "Prior sigma (σ)",
                "ewoc_alpha":   "EWOC α — overdose threshold",
                "max_n":        "Maximum sample size (max N)",
                "cohort_size":  "Cohort size — patients per dose decision",
                "prior_nu_t1":  "Prior MTD level — tox1 (acute)",
                "prior_nu_t2":  "Prior MTD level — tox2 (subacute / surgery)",
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

        # Initialise DE widget defaults once (avoids key+value conflict)
        for _dk, _dv in [("de_sig_min", 0.3), ("de_sig_max", 2.0),
                          ("de_sig_pts", 8),  ("de_ea_min",  0.05),
                          ("de_ea_max",  0.60), ("de_ea_pts", 8),
                          ("de_inc_off", True), ("de_n_sim",  200),
                          ("de_seed",    42),
                          ("de_nu1_vals", [1, 2, 3, 4, 5]),
                          ("de_nu2_vals", [1, 2, 3, 4, 5])]:
            if _dk not in _ss:
                _ss[_dk] = _dv

        # Clamp de_sig_max / de_ea_max so their stored value is always
        # above their dynamic min_value.  This prevents Streamlit from
        # raising an exception or silently clipping the widget value when
        # de_sig_min / de_ea_min is raised above the stored max.
        _sig_min_floor = round(float(_ss["de_sig_min"]) + 0.1, 1)
        if float(_ss["de_sig_max"]) < _sig_min_floor:
            _ss["de_sig_max"] = min(_sig_min_floor, 5.0)

        _ea_min_floor = round(float(_ss["de_ea_min"]) + 0.01, 2)
        if float(_ss["de_ea_max"]) < _ea_min_floor:
            _ss["de_ea_max"] = min(_ea_min_floor, 0.99)

        if _de_param == "sigma":
            _c1, _c2, _c3 = st.columns(3)
            _de_sig_min = _c1.number_input("Min σ", 0.1, 4.9,
                                           step=0.1, key="de_sig_min")
            _de_sig_max = _c2.number_input("Max σ",
                                           float(max(_de_sig_min + 0.1, 0.2)),
                                           5.0,
                                           step=0.1, key="de_sig_max")
            _de_sig_pts = _c3.slider("Points", 3, 20, key="de_sig_pts")
            _de_pv      = np.linspace(_de_sig_min, _de_sig_max,
                                      _de_sig_pts).tolist()
            _de_label   = "σ (prior sigma)"
            _de_ptype   = "continuous"

        elif _de_param == "ewoc_alpha":
            _c1, _c2, _c3 = st.columns(3)
            _de_ea_min  = _c1.number_input("Min α", 0.05, 0.97,
                                           step=0.01, key="de_ea_min")
            _de_ea_max  = _c2.number_input("Max α",
                                           float(max(_de_ea_min + 0.01, 0.06)),
                                           0.99,
                                           step=0.01, key="de_ea_max")
            _de_ea_pts  = _c3.slider("Points", 3, 20, key="de_ea_pts")
            _de_inc_off = st.checkbox("Include current EWOC α as a point",
                                      key="de_inc_off")
            _de_pv      = (([float(get_config_value("ewoc_alpha"))]
                            if _de_inc_off else []) +
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
            )
            _de_label = "Prior MTD level — tox2 (subacute / surgery)"
            _de_ptype = "discrete"

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
            )
            _de_label = "Cohort size (patients per dose decision)"
            _de_ptype = "discrete"

        st.divider()
        _de_n_sim  = st.slider("Simulations per point", 50, 2000,
                               step=50, key="de_n_sim")
        _de_seed   = st.number_input("Seed", 0, 99999,
                                     step=1, key="de_seed")
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

    # ── Execute sweep ─────────────────────────────────────────────────────
    if _de_run_btn and _de_skel_ok and len(_de_pv_eff) > 0:
        _de_base_ss = dict(
            target_tox1          = float(get_config_value("target_t1")),
            target_tox2          = float(get_config_value("target_t2")),
            p_surgery            = float(_cfg("p_surgery")),
            sigma                = float(_cfg("sigma")),
            ewoc_on              = bool(_cfg("ewoc_on")),
            ewoc_alpha           = float(get_config_value("ewoc_alpha")),
            max_n                = int(_cfg("max_n_crm")),
            cohort_size          = int(_cfg("cohort_size")),
            start_level          = int(_cfg("start_level_1b")) - 1,
            accrual_per_month    = float(_cfg("accrual_per_month")),
            incl_to_rt           = int(_cfg("incl_to_rt")),
            rt_dur               = int(_cfg("rt_dur")),
            rt_to_surg           = int(_cfg("rt_to_surg")),
            tox2_win             = int(_cfg("tox2_win")),
            max_step             = int(_cfg("max_step")),
            gh_n                 = int(_cfg("gh_n")),
            burn_in              = bool(_cfg("burn_in")),
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

        # Metric header row with help tooltips
        _mc1, _mc2, _mc3 = st.columns(3)
        _mc1.markdown("**Quality score**",
                      help=_HELP_QS)
        _mc2.markdown("**% Correct selection**",
                      help=_HELP_CS)
        _mc3.markdown("**Overdose rate (%)**",
                      help=_HELP_OR)

        _disp = _df[["param_label", "n_patients", "quality_score",
                     "pct_correct_selection", "overdose_rate"]].copy()
        _disp.columns = [_lbl, "N patients", "Quality score",
                         "% Correct selection", "Overdose rate (%)"]
        _disp["Quality score"]       = _disp["Quality score"].round(4)
        _disp["% Correct selection"] = _disp["% Correct selection"].round(1)
        _disp["Overdose rate (%)"]   = _disp["Overdose rate (%)"].round(1)
        st.dataframe(_disp, use_container_width=True, hide_index=True)

