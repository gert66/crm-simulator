"""design_exploration.py — self-contained TITE-CRM simulation core
Copied/adapted from sim_tite.py (cannot be imported: top-level st calls).
No Streamlit UI here — pure simulation logic only.
"""
import numpy as np

MONTH = 30.0

def safe_probs(x):
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)

def dfcrm_getprior(halfwidth, target, nu, nlevel):
    hw, tgt, nl = float(halfwidth), float(target), int(nlevel)
    nu = int(nu)
    if not (0 < tgt < 1) or hw <= 0 or (tgt - hw) <= 0 or (tgt + hw) >= 1:
        raise ValueError("target or halfwidth out of range.")
    if not (1 <= nu <= nl):
        raise ValueError("nu must be in [1, nlevel].")
    ds = np.full(nl, np.nan, dtype=float)
    ds[nu - 1] = tgt
    for k in range(nu, 1, -1):
        bk = np.log(np.log(tgt + hw) / np.log(ds[k - 1]))
        ds[k - 2] = np.exp(np.log(tgt - hw) / np.exp(bk))
    for k in range(nu, nl):
        bk1 = np.log(np.log(tgt - hw) / np.log(ds[k - 1]))
        ds[k] = np.exp(np.log(tgt + hw) / np.exp(bk1))
    return ds


# ── Patient timeline ──────────────────────────────────────────────────────────

def make_patient(rng, dose, arrival_day,
                 true_t1, p_surgery, true_t2,
                 incl_to_rt, rt_dur, rt_to_surg, tox1_win, tox2_win):
    rt_start     = arrival_day + float(incl_to_rt)
    rt_end       = rt_start    + float(rt_dur)
    tox1_win_end = rt_start    + float(tox1_win)

    has_tox1 = rng.random() < float(true_t1[dose])
    tox1_day = float(rt_start + rng.uniform(0.0, float(tox1_win))) if has_tox1 else None

    hs  = rng.random() < float(p_surgery)
    sd  = float(rt_end + float(rt_to_surg)) if hs else None
    tw2 = float(sd + float(tox2_win)) if hs else None
    ht2 = bool(hs and rng.random() < float(true_t2[dose]))
    td2 = float(sd + rng.uniform(0.0, float(tox2_win))) if ht2 else None
    return dict(dose=int(dose), arrival=float(arrival_day),
                rt_start=rt_start, tox1_win_end=tox1_win_end,
                has_tox1=bool(has_tox1), tox1_day=tox1_day,
                has_surgery=bool(hs), surgery_day=sd,
                tox2_win_end=tw2, has_tox2=ht2, tox2_day=td2)


def patient_follow_up_end(pt):
    last = pt["tox1_win_end"]
    if pt["has_surgery"] and pt["tox2_win_end"] is not None:
        last = max(last, pt["tox2_win_end"])
    return float(last)


# ── TITE weights ──────────────────────────────────────────────────────────────

def tite_weights(patients, current_day, tox1_win, tox2_win, n_levels):
    n1=np.zeros(n_levels); y1=np.zeros(n_levels); n2=np.zeros(n_levels); y2=np.zeros(n_levels)
    t = float(current_day)
    for p in patients:
        d = p["dose"]
        obs1 = p["has_tox1"] and p["tox1_day"] is not None and p["tox1_day"] <= t
        w1   = 0.0 if t < p["rt_start"] else (1.0 if obs1 or t >= p["tox1_win_end"]
               else (t - p["rt_start"]) / float(tox1_win))
        n1[d] += np.clip(w1, 0.0, 1.0)
        if obs1: y1[d] += 1.0
        if p["has_surgery"] and p["surgery_day"] is not None:
            sd   = p["surgery_day"]
            obs2 = p["has_tox2"] and p["tox2_day"] is not None and p["tox2_day"] <= t
            w2   = 0.0 if t < sd else (1.0 if obs2 or (p["tox2_win_end"] and t >= p["tox2_win_end"])
                   else (t - sd) / float(tox2_win))
            n2[d] += np.clip(w2, 0.0, 1.0)
            if obs2: y2[d] += 1.0
    return n1, y1, n2, y2


def posterior_via_gh(sigma, skeleton, n_per, dlt_per, gh_n=61):
    sk = safe_probs(skeleton); n = np.asarray(n_per, float); y = np.asarray(dlt_per, float)
    x, w = np.polynomial.hermite.hermgauss(int(gh_n))
    P    = safe_probs(sk[None, :] ** np.exp(float(sigma) * np.sqrt(2.0) * x)[:, None])
    ll   = (y*np.log(P) + (n-y)*np.log(1-P)).sum(axis=1)
    lun  = np.log(w) + ll; pw = np.exp(lun - lun.max()); pw /= pw.sum()
    return pw, P


def crm_posterior_summaries(sigma, skeleton, n_per, dlt_per, target, gh_n=61):
    pw, P = posterior_via_gh(sigma, skeleton, n_per, dlt_per, gh_n=gh_n)
    return (pw[:, None]*P).sum(0), (pw[:, None]*(P > float(target))).sum(0)


# ── CRM dose selection ────────────────────────────────────────────────────────

def crm_choose_next(sigma, skel1, skel2, n1, y1, n2, y2,
                    current_level, target1, target2,
                    ewoc_alpha=None, max_step=1, gh_n=61,
                    enforce_guardrail=True, highest_tried=-1, n_levels=5):
    pm1, od1 = crm_posterior_summaries(sigma, skel1, n1, y1, target1, gh_n=gh_n)
    _,   od2 = crm_posterior_summaries(sigma, skel2, n2, y2, target2, gh_n=gh_n)
    if ewoc_alpha is None:
        cands = np.arange(n_levels)
    else:
        cands = np.where((od1 < float(ewoc_alpha)) & (od2 < float(ewoc_alpha)))[0]
    if cands.size == 0:
        cands = np.array([0], dtype=int)
    k = int(cands.max()) if ewoc_alpha is not None else int(cands[np.argmin(np.abs(pm1[cands] - float(target1)))])
    k = int(np.clip(k, current_level - max_step, current_level + max_step))
    if enforce_guardrail and highest_tried >= 0:
        k = min(k, highest_tried + 1)
    return int(np.clip(k, 0, n_levels - 1))


def crm_select_mtd(sigma, skel1, skel2, n1, y1, n2, y2, target1, target2,
                   ewoc_alpha=None, gh_n=61, restrict_to_tried=True):
    pm1, od1 = crm_posterior_summaries(sigma, skel1, n1, y1, target1, gh_n=gh_n)
    _,   od2 = crm_posterior_summaries(sigma, skel2, n2, y2, target2, gh_n=gh_n)
    n_levels = len(skel1)
    cands = np.arange(n_levels) if ewoc_alpha is None else \
            np.where((od1 < float(ewoc_alpha)) & (od2 < float(ewoc_alpha)))[0]
    if cands.size == 0:
        return 0
    if restrict_to_tried:
        tried = np.where(np.asarray(n1) > 0)[0]
        if tried.size > 0:
            c2 = np.intersect1d(cands, tried)
            cands = c2 if c2.size > 0 else tried
    return int(cands.max()) if ewoc_alpha is not None else \
           int(cands[np.argmin(np.abs(pm1[cands] - float(target1)))])


# ── TITE-CRM trial runner (no trace collection — sweep optimised) ─────────────

def run_tite_crm(
    true_t1, p_surgery, true_t2, target1, target2, skel1, skel2,
    sigma=1.0, start_level=0, max_n=27, cohort_size=3,
    accrual_per_month=1.5,
    incl_to_rt=21, rt_dur=14, rt_to_surg=84, tox1_win=98, tox2_win=30,
    max_step=1, gh_n=61,
    enforce_guardrail=True, restrict_final_to_tried=True,
    ewoc_on=True, ewoc_alpha=0.25, burn_in=True, rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    true_t1 = np.asarray(true_t1, dtype=float)
    true_t2 = np.asarray(true_t2, dtype=float)
    n_levels     = len(true_t1)
    rate_per_day = float(accrual_per_month) / MONTH
    ewoc_eff     = float(ewoc_alpha) if ewoc_on else None
    level        = int(start_level)
    patients, current_day, highest_tried = [], 0.0, -1
    burn_active  = bool(burn_in)

    while len(patients) < int(max_n):
        n_add = min(int(cohort_size), int(max_n) - len(patients))
        for _ in range(n_add):
            current_day += rng.exponential(1.0 / rate_per_day)
            patients.append(make_patient(
                rng, level, current_day, true_t1, p_surgery, true_t2,
                incl_to_rt, rt_dur, rt_to_surg, tox1_win, tox2_win))
        decision_day  = current_day
        highest_tried = max(highest_tried, level)
        n1, y1, n2, y2 = tite_weights(patients, decision_day, tox1_win, tox2_win, n_levels)

        if burn_active:
            if any(p["has_tox1"] and p["tox1_day"] is not None
                   and p["tox1_day"] <= decision_day for p in patients):
                burn_active = False
        if burn_active:
            level = min(level + 1, n_levels - 1)
            if level == n_levels - 1:
                burn_active = False
        else:
            level = crm_choose_next(
                sigma, skel1, skel2, n1, y1, n2, y2,
                level, target1, target2,
                ewoc_alpha=ewoc_eff, max_step=max_step, gh_n=gh_n,
                enforce_guardrail=enforce_guardrail,
                highest_tried=highest_tried, n_levels=n_levels)

    study_days = max(patient_follow_up_end(p) for p in patients)
    n1f, y1f, n2f, y2f = tite_weights(patients, study_days, tox1_win, tox2_win, n_levels)
    return crm_select_mtd(
        sigma, skel1, skel2, n1f, y1f, n2f, y2f, target1, target2,
        ewoc_alpha=ewoc_eff, gh_n=gh_n, restrict_to_tried=restrict_final_to_tried)


# ==============================================================================
# Quality score helpers
# ==============================================================================

def _quality_score(selected, true_t1, true_t2, target1, target2):
    """Asymmetric exponential loss: penalises overdose more than underdose."""
    d1 = float(true_t1[selected]) - float(target1)
    d2 = float(true_t2[selected]) - float(target2)
    bd = max(d1, d2)                          # binding (worst) endpoint
    w  = 1.8 if bd > 0 else 1.0              # w_over=1.8, w_under=1.0
    return float(np.exp(-6.0 * w * abs(bd)))


def _true_optimal(true_t1, true_t2, target1, target2):
    """Dose minimising max(true_t1[d]-target1, true_t2[d]-target2)."""
    excess = [max(float(true_t1[d]) - float(target1),
                  float(true_t2[d]) - float(target2))
              for d in range(len(true_t1))]
    return int(np.argmin(excess))


# ==============================================================================
# Parameter sweep
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
        param_label, quality_score, pct_correct_selection, overdose_rate
    """
    import pandas as pd

    true_t1 = np.asarray(true_t1, dtype=float)
    true_t2 = np.asarray(true_t2, dtype=float)
    t1      = float(base_ss["target_tox1"])
    t2      = float(base_ss["target_tox2"])
    optimal = _true_optimal(true_t1, true_t2, t1, t2)

    # Fixed kwargs that never change across the sweep
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
        else:
            raise ValueError(f"Unknown param_name: {param_name!r}")

        rng = np.random.default_rng(int(seed) + idx * 1000)
        scores, correct, overdosed = [], [], []
        for _ in range(int(n_sim)):
            sel = run_tite_crm(**kw, rng=rng)
            scores.append(_quality_score(sel, true_t1, true_t2, t1, t2))
            correct.append(int(sel == optimal))
            overdosed.append(int(max(float(true_t1[sel]) - t1,
                                     float(true_t2[sel]) - t2) > 0))

        rows.append(dict(
            param_label=label,
            param_raw=pv,
            quality_score=float(np.mean(scores)),
            pct_correct_selection=float(np.mean(correct)) * 100.0,
            overdose_rate=float(np.mean(overdosed)) * 100.0,
        ))

    return pd.DataFrame(rows)
