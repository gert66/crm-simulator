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
    "show_crm_trace":     False,
}

TRUE_T1_KEYS  = [f"true_t1_L{i}"  for i in range(5)]
TRUE_T2_KEYS  = [f"true_t2_L{i}"  for i in range(5)]

def h(key, desc, r_name=None):
    txt = desc
    if r_name:
        txt += f"\n\n*R equivalent: `{r_name}`*"
    return txt

# Initialise session state on first run
for k, v in R_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
for i, v in enumerate(DEFAULT_TRUE_T1):
    if TRUE_T1_KEYS[i] not in st.session_state:
        st.session_state[TRUE_T1_KEYS[i]] = v
for i, v in enumerate(DEFAULT_TRUE_T2):
    if TRUE_T2_KEYS[i] not in st.session_state:
        st.session_state[TRUE_T2_KEYS[i]] = v

# ── Reset-to-defaults button ──────────────────────────────────────────────────
def _do_reset():
    for k, v in R_DEFAULTS.items():
        st.session_state[k] = v
    for i, v in enumerate(DEFAULT_TRUE_T1):
        st.session_state[TRUE_T1_KEYS[i]] = v
    for i, v in enumerate(DEFAULT_TRUE_T2):
        st.session_state[TRUE_T2_KEYS[i]] = v

def _draw_timeline(incl_to_rt, rt_dur, rt_to_surg, tox2_win):
    """
    Single-row horizontal timeline with labelled coloured bands.
    Returns a matplotlib Figure.

    Tox1 window is derived as rt_dur + rt_to_surg, so it ends exactly at
    surgery — the same derivation used in the simulation.
    """
    fig, ax = plt.subplots(figsize=(9.0, 0.9), dpi=120)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rt_start = incl_to_rt
    rt_end   = rt_start + rt_dur
    surg     = rt_end   + rt_to_surg
    t1_end   = surg                    # tox1 window = rt_dur + rt_to_surg → ends at surgery
    t2_end   = surg     + tox2_win
    total    = t2_end * 1.05

    def x(d): return float(d) / total

    y0, h_bar = 0.25, 0.50
    segments = [
        (0,        rt_start, "#d0d0d0", "Incl→RT"),
        (rt_start, rt_end,   "#4a90d9", "RT"),
        (rt_start, t1_end,   "#ff9900", "Tox1 window"),
        (rt_end,   surg,     "#d0d0d0", "RT end→Surgery"),
        (surg,     t2_end,   "#e04040", "Tox2 window"),
    ]
    plotted_labels = set()
    handles = []
    for x0, x1, col, lbl in segments:
        alpha = 0.55 if "window" in lbl else 0.30
        bar = mpatches.FancyBboxPatch(
            (x(x0), y0), x(x1) - x(x0), h_bar,
            boxstyle="square,pad=0",
            facecolor=col, alpha=alpha, edgecolor="none",
        )
        ax.add_patch(bar)
        if lbl not in plotted_labels:
            handles.append(mpatches.Patch(facecolor=col, alpha=alpha, label=lbl))
            plotted_labels.add(lbl)

    markers = [(0, "Incl"), (rt_start, "RT\nstart"), (rt_end, "RT\nend"),
               (surg, "Surgery"), (t2_end, "Done")]
    for d, lbl in markers:
        xp = x(d)
        ax.axvline(xp, ymin=0.20, ymax=0.80, color="#555", lw=0.8)
        ax.text(xp, 0.05, lbl, ha="center", va="bottom", fontsize=6.5, color="#444")

    ax.legend(handles=handles, loc="upper right", fontsize=7,
              frameon=False, ncol=len(handles), bbox_to_anchor=(1, 1.1))
    fig.patch.set_facecolor("none")
    fig.tight_layout(pad=0)
    return fig

# ==============================================================================
# Navigation (sidebar)
# ==============================================================================

with st.sidebar:
    st.markdown("## Navigation")
    view = st.radio(
        "View",
        options=["Essentials", "Playground"],
        label_visibility="collapsed",
    )

# ==============================================================================
# ESSENTIALS VIEW
# ==============================================================================

if view == "Essentials":
    _ec1, _ec2, _ec3 = st.columns(3, gap="large")

    with _ec1:
        st.markdown("#### Study endpoints")
        st.number_input(
            "Target tox1 (acute) rate",
            min_value=0.05, max_value=0.50, step=0.01, key="target_t1",
            help=h("target_t1", "Target acute DLT probability for MTD definition.")
        )
        st.number_input(
            "Target tox2 (subacute | surgery) rate",
            min_value=0.05, max_value=0.50, step=0.01, key="target_t2",
            help=h("target_t2",
                   "Target subacute DLT probability conditional on surgery. "
                   "Only surgery patients contribute to the tox2 model.")
        )
        st.number_input(
            "Probability of surgery",
            min_value=0.0, max_value=1.0, step=0.01, key="p_surgery",
            help=h("p_surgery",
                   "Global probability that a patient proceeds to surgery. "
                   "Dose-independent. Subacute toxicity only observed in these patients.")
        )
        st.number_input(
            "Start dose level (1-based)",
            min_value=1, max_value=5, step=1, key="start_level_1b",
            help=h("start_level_1b", "Starting dose level (1 = lowest).")
        )

        st.markdown("#### Simulation")
        st.number_input(
            "Number of simulated trials",
            min_value=50, max_value=5000, step=50, key="n_sims",
            help=h("n_sims", "Replicates for the simulation study.")
        )
        st.number_input(
            "Random seed",
            min_value=1, max_value=10_000_000, step=1, key="seed",
            help=h("seed", "Random seed for reproducibility.")
        )
        st.number_input(
            "Avg patients per month",
            min_value=0.1, max_value=20.0, step=0.1, key="accrual_per_month",
            help=h("accrual_per_month",
                   "Average accrual rate. Arrivals simulated as a Poisson process "
                   "(exponential inter-arrival times at this rate).")
        )

    with _ec2:
        st.markdown("#### Timing (days)")
        st.number_input(
            "Inclusion to RT start",
            min_value=0, max_value=180, step=1, key="incl_to_rt",
            help=h("incl_to_rt",
                   "Days from enrolment to start of radiotherapy. "
                   "Tox1 window begins at RT start. Default ≈ 3 weeks.")
        )
        st.number_input(
            "Radiotherapy duration",
            min_value=1, max_value=60, step=1, key="rt_dur",
            help=h("rt_dur",
                   "Duration of radiotherapy in days. Default ≈ 2 weeks (10 fractions).")
        )
        st.number_input(
            "RT end to surgery",
            min_value=1, max_value=365, step=1, key="rt_to_surg",
            help=h("rt_to_surg",
                   "Days from end of radiotherapy to surgery. Default 84 days ≈ 12 weeks. "
                   "The tox1 (acute) follow-up window is derived as RT duration + this value, "
                   "so it always extends from RT start to the moment of surgery.")
        )
        st.number_input(
            "Tox2 follow-up window (days)",
            min_value=7, max_value=180, step=1, key="tox2_win",
            help=h("tox2_win",
                   "Post-surgery window for subacute toxicity assessment. Default 30 days.")
        )

        st.markdown("#### Sample size")
        st.number_input(
            "Max sample size (6+3)",
            min_value=6, max_value=200, step=3, key="max_n_63",
            help=h("max_n_63",
                   "Maximum total enrolled patients in the 6+3 arm, including "
                   "bridging patients treated at lower doses while awaiting evaluability.")
        )
        st.number_input(
            "Max sample size (CRM)",
            min_value=6, max_value=200, step=3, key="max_n_crm",
            help=h("max_n_crm", "Maximum total enrolled patients in the TITE-CRM arm.")
        )
        st.number_input(
            "Cohort size (CRM)",
            min_value=1, max_value=12, step=1, key="cohort_size",
            help=h("cohort_size",
                   "Number of patients per CRM cohort. CRM updates after each "
                   "cohort is fully enrolled, using TITE weights at that moment.")
        )

    with _ec3:
        st.markdown("#### CRM integration")
        st.selectbox(
            "Gauss–Hermite points",
            options=[31, 41, 61, 81], key="gh_n",
            help=h("gh_n",
                   "Quadrature points for CRM posterior. Higher = more accurate, slower.")
        )
        st.selectbox(
            "Max dose step per update",
            options=[1, 2], key="max_step",
            help=h("max_step",
                   "Max dose levels the CRM can move per cohort update.")
        )
        st.slider(
            "Prior sigma on theta",
            min_value=0.2, max_value=5.0, step=0.1, key="sigma",
            help=h("sigma",
                   "SD of theta in the CRM prior (shared for both endpoints). "
                   "Larger = more diffuse prior.",
                   r_name="prior.sigma / sigma")
        )

        st.markdown("#### CRM safety / selection")
        st.toggle(
            "Guardrail: next dose ≤ highest tried + 1",
            key="enforce_guardrail",
            help=h("enforce_guardrail", "Prevent skipping untried dose levels.")
        )
        st.toggle(
            "Final MTD must be among tried doses",
            key="restrict_final_mtd",
            help=h("restrict_final_mtd",
                   "Restrict final MTD selection to doses where n > 0.")
        )

        st.markdown("#### CRM behaviour")
        st.toggle(
            "Burn-in until first tox1 DLT",
            key="burn_in",
            help=h("burn_in",
                   "Escalate one level at a time until the first observed acute DLT, "
                   "then switch to CRM updates.")
        )
        st.toggle(
            "Enable EWOC joint overdose control",
            key="ewoc_on",
            help=h("ewoc_on",
                   "Restrict doses where BOTH P(tox1 OD) and P(tox2 OD) < EWOC alpha.")
        )
        st.number_input(
            "EWOC alpha",
            min_value=0.01, max_value=0.99, step=0.01, key="ewoc_alpha",
            disabled=(not bool(st.session_state["ewoc_on"])),
            help=h("ewoc_alpha",
                   "EWOC threshold applied to both endpoints independently.")
        )

        st.markdown("#### CRM decision trace")
        st.toggle(
            "Explain first CRM trial",
            key="show_crm_trace",
            help=h("show_crm_trace",
                   "When ON, shows a detailed walkthrough for the first simulated "
                   "CRM trial only: which dose each patient received, what follow-up "
                   "data were available at each decision point, how the model judged "
                   "safety for each dose level, and why the next dose was chosen. "
                   "Has no effect on the summary results across all simulated trials.")
        )

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
        "<div style='font-size:0.79rem;font-weight:600;color:#555;'>"
        "Acute thresholds</div>",
        unsafe_allow_html=True,
    )
    _ar1, _ar2, _ar3 = st.columns(3, gap="small")
    with _ar1:
        st.number_input("≥6 — esc if tox1 ≤", min_value=0, max_value=5,
                        step=1, key="a6_esc_max",
                        help=h("a6_esc_max", "Phase 1 acute escalation threshold."))
    with _ar2:
        st.number_input("≥6 — stop if tox1 ≥", min_value=1, max_value=6,
                        step=1, key="a6_stop_min",
                        help=h("a6_stop_min", "Phase 1 acute stopping threshold."))
    with _ar3:
        st.number_input("≥9 — esc if tox1 ≤", min_value=0, max_value=8,
                        step=1, key="a9_esc_max",
                        help=h("a9_esc_max", "Phase 2 acute escalation threshold."))

    st.markdown(
        "<div style='font-size:0.79rem;font-weight:600;color:#555;margin-top:0.3rem;'>"
        "Subacute thresholds</div>",
        unsafe_allow_html=True,
    )
    _sr1, _sr2, _sr3, _sr4 = st.columns(4, gap="small")
    with _sr1:
        st.number_input("≥6 surg — esc if tox2 ≤", min_value=0, max_value=6,
                        step=1, key="s6_esc_max",
                        help=h("s6_esc_max", "Phase 1 subacute escalation threshold."))
    with _sr2:
        st.number_input("≥6 surg — stop if tox2 ≥", min_value=1, max_value=6,
                        step=1, key="s6_stop_min",
                        help=h("s6_stop_min", "Phase 1 subacute stopping threshold."))
    with _sr3:
        st.number_input("≥9 surg — esc if tox2 ≤", min_value=0, max_value=9,
                        step=1, key="s9_esc_max",
                        help=h("s9_esc_max", "Phase 2 subacute escalation threshold."))
    with _sr4:
        st.number_input("≥9 surg — stop if tox2 ≥", min_value=1, max_value=9,
                        step=1, key="s9_stop_min",
                        help=h("s9_stop_min", "Phase 2 subacute stopping threshold."))

    st.write("")
    st.button("Reset to defaults", on_click=_do_reset)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.78rem;font-weight:600;color:#555;margin-top:0.3rem;'>"
        "Patient timeline (based on current timing settings)</div>",
        unsafe_allow_html=True,
    )
    _tl_fig = _draw_timeline(
        int(st.session_state["incl_to_rt"]),
        int(st.session_state["rt_dur"]),
        int(st.session_state["rt_to_surg"]),
        int(st.session_state["tox2_win"]),
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
                v1 = st.number_input(f"T1 L{i}", 0.0, 1.0,
                                     value=float(st.session_state.get(TRUE_T1_KEYS[i], DEFAULT_TRUE_T1[i])),
                                     step=0.01,
                                     key=TRUE_T1_KEYS[i],
                                     label_visibility="collapsed",
                                     help=f"True probability of acute toxicity at dose L{i}.")
                true_t1.append(float(v1))
            with rT2:
                v2 = st.number_input(f"T2 L{i}", 0.0, 1.0,
                                     value=float(st.session_state.get(TRUE_T2_KEYS[i], DEFAULT_TRUE_T2[i])),
                                     step=0.01,
                                     key=TRUE_T2_KEYS[i],
                                     label_visibility="collapsed",
                                     help=f"True probability of subacute toxicity given surgery at L{i}.")
                true_t2.append(float(v2))

        target_t1_val = float(st.session_state["target_t1"])
        target_t2_val = float(st.session_state["target_t2"])
        p_surg_val    = float(st.session_state["p_surgery"])
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
        st.radio(
            "Skeleton model",
            options=["empiric", "logistic"],
            horizontal=True, key="prior_model",
            help=h("prior_model", "Skeleton generation method, shared for both endpoints.")
        )
        prior_model_val = str(st.session_state["prior_model"])
        intcpt_val      = float(st.session_state["logistic_intcpt"])

        # Force-write halfwidth keys before any slider is created (Streamlit safety)
        _pt1     = float(st.session_state["prior_target_t1"])
        _max_hw1 = round(max(0.01, min(0.30, _pt1 - 0.01, 1.0 - _pt1 - 0.01)), 2)
        _pt2     = float(st.session_state["prior_target_t2"])
        _max_hw2 = round(max(0.01, min(0.30, _pt2 - 0.01, 1.0 - _pt2 - 0.01)), 2)

        _prior_init = [
            ("prior_target_t1", float, 0.05, 0.50,    R_DEFAULTS["prior_target_t1"]),
            ("halfwidth_t1",    float, 0.01, _max_hw1, R_DEFAULTS["halfwidth_t1"]),
            ("prior_nu_t1",     int,   1,    5,        R_DEFAULTS["prior_nu_t1"]),
            ("prior_target_t2", float, 0.05, 0.50,    R_DEFAULTS["prior_target_t2"]),
            ("halfwidth_t2",    float, 0.01, _max_hw2, R_DEFAULTS["halfwidth_t2"]),
            ("prior_nu_t2",     int,   1,    5,        R_DEFAULTS["prior_nu_t2"]),
        ]
        for _k, _typ, _lo, _hi, _def in _prior_init:
            _v = _typ(st.session_state.get(_k, _def))
            st.session_state[_k] = _typ(np.clip(_v, _lo, _hi))

        ep_tab = st.radio(
            "Endpoint",
            options=["Tox1 (acute)", "Tox2 (subacute | surgery)"],
            horizontal=True, key="prior_ep_tab",
            help="Switch between tox1 and tox2 prior parameter sets.",
        )

        if ep_tab == "Tox1 (acute)":
            st.slider("Prior target (tox1)", 0.05, 0.50, step=0.01,
                      key="prior_target_t1",
                      help=h("prior_target_t1", "Target probability for the tox1 skeleton."))
            st.slider("Halfwidth (tox1)", 0.01, float(_max_hw1), step=0.01,
                      key="halfwidth_t1",
                      help=h("halfwidth_t1", "Skeleton steepness. target ± halfwidth must stay in (0,1)."))
            st.slider("Prior MTD level (tox1, 1-based)", 1, 5, step=1,
                      key="prior_nu_t1",
                      help=h("prior_nu_t1", "Dose level a priori closest to the tox1 target."))
        else:
            st.slider("Prior target (tox2)", 0.05, 0.50, step=0.01,
                      key="prior_target_t2",
                      help=h("prior_target_t2", "Target probability for the tox2 skeleton."))
            st.slider("Halfwidth (tox2)", 0.01, float(_max_hw2), step=0.01,
                      key="halfwidth_t2",
                      help=h("halfwidth_t2", "Skeleton steepness. target ± halfwidth must stay in (0,1)."))
            st.slider("Prior MTD level (tox2, 1-based)", 1, 5, step=1,
                      key="prior_nu_t2",
                      help=h("prior_nu_t2", "Dose level a priori closest to the tox2 conditional target."))

        # Compute skeletons for preview and simulation
        hw1_eff = float(st.session_state["halfwidth_t1"])
        try:
            skel_t1 = dfcrm_getprior(
                halfwidth=hw1_eff, target=float(st.session_state["prior_target_t1"]),
                nu=int(st.session_state["prior_nu_t1"]),
                nlevel=5, model=prior_model_val, intcpt=intcpt_val,
            ).tolist()
        except ValueError as e:
            st.warning(f"Tox1 skeleton: {e}")
            hw1_eff = 0.10
            skel_t1 = dfcrm_getprior(
                halfwidth=hw1_eff, target=float(st.session_state["prior_target_t1"]),
                nu=int(st.session_state["prior_nu_t1"]),
                nlevel=5, model=prior_model_val, intcpt=intcpt_val,
            ).tolist()

        hw2_eff = float(st.session_state["halfwidth_t2"])
        try:
            skel_t2 = dfcrm_getprior(
                halfwidth=hw2_eff, target=float(st.session_state["prior_target_t2"]),
                nu=int(st.session_state["prior_nu_t2"]),
                nlevel=5, model=prior_model_val, intcpt=intcpt_val,
            ).tolist()
        except ValueError as e:
            st.warning(f"Tox2 skeleton: {e}")
            hw2_eff = 0.10
            skel_t2 = dfcrm_getprior(
                halfwidth=hw2_eff, target=float(st.session_state["prior_target_t2"]),
                nu=int(st.session_state["prior_nu_t2"]),
                nlevel=5, model=prior_model_val, intcpt=intcpt_val,
            ).tolist()

    # ── Right: dose-risk preview ──────────────────────────────────────────────
    with right:
        st.markdown("#### Dose-risk preview")

        fig, (ax1, ax2) = plt.subplots(2, 1,
                                       figsize=(PREVIEW_W_IN, PREVIEW_H_IN),
                                       dpi=PREVIEW_DPI)
        x = np.arange(5)

        ax1.plot(x, true_t1,  "o-",  color="tab:blue",   lw=1.5, label="True tox1")
        ax1.plot(x, skel_t1,  "o--", color="tab:blue",   lw=1.5, label="Skel tox1")
        ax1.axhline(target_t1_val, lw=1, alpha=0.55, color="tab:blue")
        ax1.set_ylabel("P(tox1)", fontsize=8)
        ax1.set_xticks(x); ax1.set_xticklabels([f"L{i}" for i in range(5)], fontsize=8)
        _y1 = max(max(true_t1), max(skel_t1), target_t1_val)
        ax1.set_ylim(0, min(1.0, _y1 * 1.3 + 0.02))
        ax1.legend(fontsize=7, frameon=False, loc="upper left")
        compact_style(ax1)

        ax2.plot(x, true_t2,  "s-",  color="tab:orange", lw=1.5, label="True tox2")
        ax2.plot(x, skel_t2,  "s--", color="tab:orange", lw=1.5, label="Skel tox2")
        ax2.axhline(target_t2_val, lw=1, alpha=0.55, color="tab:orange")
        ax2.set_ylabel("P(tox2)", fontsize=8)
        ax2.set_xticks(x); ax2.set_xticklabels([f"L{i}" for i in range(5)], fontsize=8)
        _y2 = max(max(true_t2), max(skel_t2), target_t2_val)
        ax2.set_ylim(0, min(1.0, _y2 * 1.25 + 0.02))
        ax2.legend(fontsize=7, frameon=False, loc="upper left")
        compact_style(ax2)

        fig.tight_layout(pad=0.5)
        st.image(fig_to_png_bytes(fig), width=int(st.session_state["preview_w_px"]))

    # ==============================================================================
    # Run simulations
    # ==============================================================================

    if run:
        rng_master = np.random.default_rng(int(st.session_state["seed"]))
        ns         = int(st.session_state["n_sims"])
        start_0b   = int(np.clip(int(st.session_state["start_level_1b"]) - 1, 0, 4))

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
        _tox1_win_derived = int(st.session_state["rt_dur"]) + int(st.session_state["rt_to_surg"])

        timing_kw = dict(
            accrual_per_month = float(st.session_state["accrual_per_month"]),
            incl_to_rt        = int(st.session_state["incl_to_rt"]),
            rt_dur            = int(st.session_state["rt_dur"]),
            rt_to_surg        = int(st.session_state["rt_to_surg"]),
            tox1_win          = _tox1_win_derived,
            tox2_win          = int(st.session_state["tox2_win"]),
        )

        for s in range(ns):
            rng_s = np.random.default_rng(rng_master.integers(0, 2**31))

            # ── TITE 6+3 ─────────────────────────────────────────────────────
            sel63, pts63, sd63, nb63 = run_tite_6plus3(
                true_t1=true_t1, p_surgery=p_surg_val, true_t2=true_t2,
                start_level=start_0b,
                max_n=int(st.session_state["max_n_63"]),
                a6_esc_max  = int(st.session_state["a6_esc_max"]),
                a6_stop_min = int(st.session_state["a6_stop_min"]),
                a9_esc_max  = int(st.session_state["a9_esc_max"]),
                s6_esc_max  = int(st.session_state["s6_esc_max"]),
                s6_stop_min = int(st.session_state["s6_stop_min"]),
                s9_esc_max  = int(st.session_state["s9_esc_max"]),
                s9_stop_min = int(st.session_state["s9_stop_min"]),
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
                sigma        = float(st.session_state["sigma"]),
                start_level  = start_0b,
                max_n        = int(st.session_state["max_n_crm"]),
                cohort_size  = int(st.session_state["cohort_size"]),
                max_step     = int(st.session_state["max_step"]),
                gh_n         = int(st.session_state["gh_n"]),
                enforce_guardrail      = bool(st.session_state["enforce_guardrail"]),
                restrict_final_to_tried= bool(st.session_state["restrict_final_mtd"]),
                ewoc_on      = bool(st.session_state["ewoc_on"]),
                ewoc_alpha   = float(st.session_state["ewoc_alpha"]),
                burn_in      = bool(st.session_state["burn_in"]),
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
                    "tox2_win":   int(st.session_state["tox2_win"]),
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
            "seed": int(st.session_state["seed"]),
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
        xx = np.arange(5); w = 0.38
        ax.bar(xx - w/2, p63,  w, label="TITE 6+3")
        ax.bar(xx + w/2, pcrm, w, label="TITE-CRM")
        ax.set_title("P(select dose as MTD)", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, max(p63.max(), pcrm.max()) * 1.15 + 1e-6)
        if ts is not None:
            ax.axvline(ts, lw=1, alpha=0.6, label=f"True safe=L{ts}")
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False)
        st.image(fig_to_png_bytes(fig), width=int(st.session_state["result_w_px"]))

    with r2:
        fig, (ax_n, ax_s) = plt.subplots(2, 1,
                                          figsize=(RESULT_W_IN, RESULT_H_IN),
                                          dpi=RESULT_DPI)
        xx = np.arange(5); w = 0.38
        ax_n.bar(xx - w/2, res["avg_n63"],    w, label="6+3")
        ax_n.bar(xx + w/2, res["avg_ncrm"],   w, label="CRM")
        ax_n.set_title("Avg patients treated", fontsize=9)
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
    _ewoc_on_flag = bool(st.session_state.get("ewoc_on", True))
    _ewoc_alpha   = float(st.session_state.get("ewoc_alpha", 0.25))
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
            ax.step(_steps, _curr, where="post", color="tab:blue",
                    lw=2, label="Dose assigned")
            ax.step(_steps, _next, where="post", color="tab:blue",
                    lw=1.2, ls="--", alpha=0.6, label="Next dose chosen")
            ax.set_title("Dose level over cohort steps", fontsize=9)
            ax.set_xlabel("Decision step", fontsize=8)
            ax.set_ylabel("Dose level (L0 – L4)", fontsize=8)
            ax.set_yticks(range(5))
            ax.set_yticklabels([f"L{i}" for i in range(5)], fontsize=7)
            ax.legend(fontsize=7, frameon=False)
            compact_style(ax)
            fig.tight_layout(pad=0.5)
            st.image(fig_to_png_bytes(fig), use_container_width=True)
            st.caption("Solid: dose given to current cohort.  "
                       "Dashed: dose selected for the next cohort.")

        # ── Plot B: overdose probabilities at the current dose over time ──────
        with _tc2:
            fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=130)
            ax.plot(_steps, _od1_curr, "o-", color="tab:blue",
                    lw=1.8, ms=4, label="OD prob tox1")
            ax.plot(_steps, _od2_curr, "s-", color="tab:orange",
                    lw=1.8, ms=4, label="OD prob tox2")
            ewoc_a = float(st.session_state.get("ewoc_alpha", 0.25))
            if bool(st.session_state.get("ewoc_on", True)):
                ax.axhline(ewoc_a, lw=1, ls="--", color="#888",
                           alpha=0.7, label=f"EWOC α={ewoc_a:.2f}")
            ax.set_title("Safety evolution at current dose", fontsize=9)
            ax.set_xlabel("Decision step", fontsize=8)
            ax.set_ylabel("P(overdose)", fontsize=8)
            ax.set_ylim(0, min(1.05, max(max(_od1_curr), max(_od2_curr)) * 1.3 + 0.05))
            ax.legend(fontsize=7, frameon=False)
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
            ax.plot(_steps, _n1_sum, "o-", color="tab:blue",
                    lw=1.8, ms=4, label="Effective n (tox1)")
            ax.plot(_steps, _n2_sum, "s-", color="tab:orange",
                    lw=1.8, ms=4, label="Effective n (tox2)")
            ax.plot(_steps, _n_enr,  "^--", color="#aaa",
                    lw=1.2, ms=4, label="Patients enrolled")
            ax.set_title("TITE follow-up accumulation", fontsize=9)
            ax.set_xlabel("Decision step", fontsize=8)
            ax.set_ylabel("Effective patient count", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
            compact_style(ax)
            fig.tight_layout(pad=0.5)
            st.image(fig_to_png_bytes(fig), use_container_width=True)
            st.caption(
                "Sum of fractional TITE weights across all enrolled patients "
                "at each decision point. The gap between enrolled (grey) and "
                "effective n shows how much follow-up is still pending. "
                "Tox2 (orange) lags tox1 because surgery must occur first."
            )
