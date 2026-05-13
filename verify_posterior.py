"""
Verification: do the ridge-plot OD probabilities match crm_select_mtd?

Three sources are compared for every dose level at study end:
  A) crm_select_mtd path  – od computed via crm_posterior_summaries(target=target_tox)
                            then filtered against ewoc_alpha
  B) ridge-plot path      – od = sum(post_w[ P[:,d] > ewoc_alpha ])
  C) last trace step      – od from _post_decs[-1]["od1/od2"]
                            (fractional weights, crm_posterior_summaries(target=target_tox))

Expected results:
  A == C in terms of formula (both use target_tox threshold), but A uses study-end
       weights while C uses last-cohort-decision fractional weights → values differ.
  B   uses a different threshold (ewoc_alpha, not target_tox) → values differ from A.
"""

import numpy as np

MONTH = 30.0

# ── copied verbatim from sim.py ────────────────────────────────────────────────

def safe_probs(x):
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)

def make_patient(rng, dose, arrival_day,
                 true_t1, p_surgery, true_t2,
                 incl_to_rt, rt_dur, rt_to_surg, tox1_win, tox2_win,
                 is_bridging=False):
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
        "dose": int(dose), "arrival": float(arrival_day),
        "rt_start": rt_start, "tox1_win_end": tox1_win_end,
        "has_tox1": has_tox1, "tox1_day": tox1_day,
        "has_surgery": has_surgery, "surgery_day": surgery_day,
        "tox2_win_end": tox2_win_end,
        "has_tox2": has_tox2, "tox2_day": tox2_day,
        "is_bridging": bool(is_bridging),
    }

def patient_follow_up_end(pt):
    last = pt["tox1_win_end"]
    if pt["has_surgery"] and pt["tox2_win_end"] is not None:
        last = max(last, pt["tox2_win_end"])
    return float(last)

def tite_weights(patients, current_day, tox1_win, tox2_win, n_levels):
    n1 = np.zeros(n_levels, dtype=float)
    y1 = np.zeros(n_levels, dtype=float)
    n2 = np.zeros(n_levels, dtype=float)
    y2 = np.zeros(n_levels, dtype=float)
    t  = float(current_day)
    for p in patients:
        d = p["dose"]
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

def crm_choose_next(sigma, skel1, skel2, n1, y1, n2, y2,
                    current_level, target1, target2,
                    ewoc_alpha=None, max_step=1, gh_n=61,
                    enforce_guardrail=True, highest_tried=-1, n_levels=5):
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

def crm_select_mtd(sigma, skel1, skel2, n1, y1, n2, y2,
                   target1, target2, ewoc_alpha=None, gh_n=61,
                   restrict_to_tried=True):
    pm1, od1 = crm_posterior_summaries(sigma, skel1, n1, y1, target1, gh_n=gh_n)
    _,   od2 = crm_posterior_summaries(sigma, skel2, n2, y2, target2, gh_n=gh_n)
    n_levels = len(skel1)
    if ewoc_alpha is None:
        candidates = np.arange(n_levels)
    else:
        candidates = np.where((od1 < float(ewoc_alpha)) & (od2 < float(ewoc_alpha)))[0]
    if candidates.size == 0:
        return 0, od1, od2
    if restrict_to_tried:
        tried = np.where(np.asarray(n1) > 0)[0]
        if tried.size > 0:
            candidates2 = np.intersect1d(candidates, tried)
            candidates = candidates2 if candidates2.size > 0 else tried
    if ewoc_alpha is not None:
        sel = int(candidates.max())
    else:
        dist = np.abs(pm1[candidates] - float(target1))
        sel = int(candidates[int(np.argmin(dist))])
    return sel, od1, od2

def run_tite_crm_instrumented(
        true_t1, p_surgery, true_t2, target1, target2, skel1, skel2,
        sigma=1.0, start_level=0, max_n=27, cohort_size=3,
        accrual_per_month=1.5, incl_to_rt=21, rt_dur=14, rt_to_surg=84,
        tox1_win=84, tox2_win=30, max_step=1, gh_n=61,
        ewoc_on=True, ewoc_alpha=0.25, burn_in=True, rng=None):
    """Run sim, return (selected, patients, study_days, trace, n1f,y1f,n2f,y2f)."""
    if rng is None:
        rng = np.random.default_rng()
    true_t1  = np.asarray(true_t1, dtype=float)
    true_t2  = np.asarray(true_t2, dtype=float)
    n_levels = len(true_t1)
    rate_per_day = float(accrual_per_month) / MONTH
    level = int(start_level)
    patients = []
    highest_tried = -1
    current_day = 0.0
    burn_active = bool(burn_in)
    ewoc_eff = float(ewoc_alpha) if ewoc_on else None
    trace = []
    cohort_step = 0

    while len(patients) < int(max_n):
        n_add = min(int(cohort_size), int(max_n) - len(patients))
        cohort_start = len(patients)
        for _ in range(n_add):
            current_day += rng.exponential(1.0 / rate_per_day)
            pt = make_patient(rng, level, current_day, true_t1, p_surgery, true_t2,
                              incl_to_rt, rt_dur, rt_to_surg, tox1_win, tox2_win)
            patients.append(pt)
        decision_day = current_day
        highest_tried = max(highest_tried, level)
        n1, y1, n2, y2 = tite_weights(patients, decision_day, tox1_win, tox2_win, n_levels)
        burn_was_active = burn_active
        if burn_active:
            obs = any(p["has_tox1"] and p["tox1_day"] is not None
                      and p["tox1_day"] <= decision_day for p in patients)
            if obs:
                burn_active = False
        if burn_active:
            next_level = min(level + 1, n_levels - 1)
            if next_level == n_levels - 1:
                burn_active = False
        else:
            next_level = crm_choose_next(
                sigma, skel1, skel2, n1, y1, n2, y2,
                level, target1, target2,
                ewoc_alpha=ewoc_eff, max_step=max_step, gh_n=gh_n,
                highest_tried=highest_tried, n_levels=n_levels)
        pm1, od1_step = crm_posterior_summaries(sigma, skel1, n1, y1, target1, gh_n=gh_n)
        pm2, od2_step = crm_posterior_summaries(sigma, skel2, n2, y2, target2, gh_n=gh_n)
        trace.append({
            "step": cohort_step + 1,
            "n1": n1.copy(), "y1": y1.copy(),
            "n2": n2.copy(), "y2": y2.copy(),
            "od1": od1_step.copy(),
            "od2": od2_step.copy(),
            "decision_day": decision_day,
        })
        cohort_step += 1
        level = next_level

    study_days = max(patient_follow_up_end(p) for p in patients)
    n1f, y1f, n2f, y2f = tite_weights(patients, study_days, tox1_win, tox2_win, n_levels)
    sel, od1_final, od2_final = crm_select_mtd(
        sigma, skel1, skel2, n1f, y1f, n2f, y2f,
        target1, target2, ewoc_alpha=ewoc_eff, gh_n=gh_n)
    return sel, patients, study_days, trace, n1f, y1f, n2f, y2f, od1_final, od2_final

# ── simulation parameters (match UI defaults) ─────────────────────────────────

from numpy.polynomial.hermite import hermgauss  # noqa: F401 (confirms numpy works)

rng_master = np.random.default_rng(42)
rng_s      = np.random.default_rng(rng_master.integers(0, 2**31))

TRUE_T1   = [0.01, 0.02, 0.12, 0.20, 0.35]
TRUE_T2   = [0.02, 0.05, 0.15, 0.25, 0.40]
P_SURG    = 0.80
TARGET_T1 = 0.15
TARGET_T2 = 0.33
EWOC_A    = 0.25
SIGMA     = 1.0
GH_N      = 61
TOX1_WIN  = 98   # rt_dur(14) + rt_to_surg(84)
TOX2_WIN  = 30
SKEL_T1   = [0.05, 0.10, 0.15, 0.22, 0.30]   # placeholder skeletons
SKEL_T2   = [0.06, 0.12, 0.22, 0.33, 0.45]

sel, patients, study_days, trace, n1f, y1f, n2f, y2f, od1_mtd, od2_mtd = \
    run_tite_crm_instrumented(
        TRUE_T1, P_SURG, TRUE_T2, TARGET_T1, TARGET_T2, SKEL_T1, SKEL_T2,
        sigma=SIGMA, ewoc_alpha=EWOC_A, gh_n=GH_N,
        tox1_win=TOX1_WIN, tox2_win=TOX2_WIN, rng=rng_s)

last = trace[-1]

# ── ridge-plot path: posterior_via_gh on study-end weights ────────────────────
pw1_end, P1_end = posterior_via_gh(SIGMA, SKEL_T1, n1f, y1f, gh_n=GH_N)
pw2_end, P2_end = posterior_via_gh(SIGMA, SKEL_T2, n2f, y2f, gh_n=GH_N)

# Ridge-plot OD: P(P[:,d] > ewoc_alpha)   ← current implementation
od1_ridge = np.array([float(np.sum(pw1_end[P1_end[:, d] > EWOC_A])) for d in range(5)])
od2_ridge = np.array([float(np.sum(pw2_end[P2_end[:, d] > EWOC_A])) for d in range(5)])

# Correct OD: P(P[:,d] > target_tox)      ← what crm_select_mtd computes
od1_correct = np.array([float(np.sum(pw1_end[P1_end[:, d] > TARGET_T1])) for d in range(5)])
od2_correct = np.array([float(np.sum(pw2_end[P2_end[:, d] > TARGET_T2])) for d in range(5)])

# ── last-trace-step OD (fractional weights, target_tox threshold) ─────────────
pw1_last, P1_last = posterior_via_gh(SIGMA, SKEL_T1, last["n1"], last["y1"], gh_n=GH_N)
pw2_last, P2_last = posterior_via_gh(SIGMA, SKEL_T2, last["n2"], last["y2"], gh_n=GH_N)
od1_last_recomputed = np.array([float(np.sum(pw1_last[P1_last[:, d] > TARGET_T1])) for d in range(5)])
od2_last_recomputed = np.array([float(np.sum(pw2_last[P2_last[:, d] > TARGET_T2])) for d in range(5)])

# ── print results ─────────────────────────────────────────────────────────────
SEP = "─" * 68

print(SEP)
print(f"Final MTD selected: L{sel}   |   study_days = {study_days:.1f}")
print(f"Cohort decisions: {len(trace)}   |   patients: {len(patients)}")
print()
print("TITE weights at study end vs last cohort step (n1 / y1):")
print(f"  Study-end  n1={n1f.round(3).tolist()}  y1={y1f.tolist()}")
print(f"  Last-step  n1={last['n1'].round(3).tolist()}  y1={last['y1'].tolist()}")
print(f"  Weights identical: {np.allclose(n1f, last['n1']) and np.allclose(y1f, last['y1'])}")
print()

print("─── L2 deep-dive ────────────────────────────────────────────────────")
d = 2
print(f"\n[A] crm_select_mtd path  (threshold = target_tox = {TARGET_T1})")
print(f"    od1_mtd[L{d}]   = {od1_mtd[d]:.6f}   (from crm_posterior_summaries)")
print(f"    threshold check: {od1_mtd[d]:.4f} < {EWOC_A} → {'PASS' if od1_mtd[d] < EWOC_A else 'FAIL'}")

print(f"\n[B] Ridge-plot path – CURRENT  (threshold = ewoc_alpha = {EWOC_A})")
print(f"    P1_end[:, {d}] range  = [{P1_end[:, d].min():.4f}, {P1_end[:, d].max():.4f}]")
print(f"    pw1_end sum           = {pw1_end.sum():.6f}  (must be 1.0)")
print(f"    mass above ewoc_alpha = {od1_ridge[d]:.6f}")
print(f"    od1_ridge[L{d}]  = {od1_ridge[d]:.6f}")

print(f"\n[C] Ridge-plot path – CORRECTED  (threshold = target_tox = {TARGET_T1})")
print(f"    mass above target_tox = {od1_correct[d]:.6f}")
print(f"    od1_correct[L{d}] = {od1_correct[d]:.6f}")
print(f"    Matches crm_select_mtd: {np.isclose(od1_correct[d], od1_mtd[d])}")

print(f"\n[D] Last trace step  (fractional weights, threshold = target_tox = {TARGET_T1})")
print(f"    n1_last[L{d}] = {last['n1'][d]:.3f}   n1_end[L{d}] = {n1f[d]:.3f}")
print(f"    y1_last[L{d}] = {last['y1'][d]:.3f}   y1_end[L{d}] = {y1f[d]:.3f}")
print(f"    od1_trace[L{d}]  (stored)     = {last['od1'][d]:.6f}")
print(f"    od1_last_recomputed[L{d}]     = {od1_last_recomputed[d]:.6f}")
print(f"    Matches crm_select_mtd:        {np.isclose(od1_last_recomputed[d], od1_mtd[d])}")

print()
print(SEP)
print("Full comparison across all dose levels (tox1):")
print(f"  {'Dose':<6} {'[A] mtd':>10} {'[B] ridge':>12} {'[C] corrected':>15} {'[D] last-step':>15} {'B==A':>6} {'C==A':>6}")
for d in range(5):
    ba = np.isclose(od1_ridge[d],   od1_mtd[d])
    ca = np.isclose(od1_correct[d], od1_mtd[d])
    print(f"  L{d}    {od1_mtd[d]:10.4f} {od1_ridge[d]:12.4f} {od1_correct[d]:15.4f} {od1_last_recomputed[d]:15.4f}  {str(ba):>5}  {str(ca):>5}")

print()
print("Full comparison across all dose levels (tox2):")
od2_last_rc = np.array([float(np.sum(pw2_last[P2_last[:, d] > TARGET_T2])) for d in range(5)])
print(f"  {'Dose':<6} {'[A] mtd':>10} {'[B] ridge':>12} {'[C] corrected':>15} {'[D] last-step':>15} {'B==A':>6} {'C==A':>6}")
for d in range(5):
    ba = np.isclose(od2_ridge[d],   od2_mtd[d])
    ca = np.isclose(od2_correct[d], od2_mtd[d])
    print(f"  L{d}    {od2_mtd[d]:10.4f} {od2_ridge[d]:12.4f} {od2_correct[d]:15.4f} {od2_last_rc[d]:15.4f}  {str(ba):>5}  {str(ca):>5}")

print()
print(SEP)
print("SUMMARY")
print(SEP)
print()
print("[A] crm_select_mtd OD = P(P[:,d] > target_tox)  vs  ewoc_alpha threshold")
print("[B] Ridge-plot OD      = P(P[:,d] > ewoc_alpha)   ← DIFFERENT threshold")
print("[C] Corrected ridge OD = P(P[:,d] > target_tox)   ← matches A ✓")
print("[D] Last trace step    = fractional weights → different n1/y1 → different values")
print()
print("ISSUE: [B] uses ewoc_alpha as the threshold for OD probability, but [A]")
print("       (crm_select_mtd / decision walkthrough table) uses target_tox.")
print(f"       target_tox1={TARGET_T1}, ewoc_alpha={EWOC_A} — they are NOT the same.")
print()
print("FIX NEEDED in ridge plot: use _tgt1/_tgt2 as threshold for od1_final/od2_final,")
print("  not _ewoc_alpha_pt.  The vertical EWOC alpha dashed line marks where the")
print("  OD probability (y-axis of the OD panel) is filtered, not the P(tox) threshold.")
