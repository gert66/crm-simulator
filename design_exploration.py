"""design_exploration.py — TITE-CRM Design Exploration / Parameter Sweep
Simulation functions copied/adapted from sim_tite.py (cannot be imported:
top-level st calls).  Run with: streamlit run design_exploration.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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


# ==============================================================================
# Streamlit UI
# ==============================================================================

st.set_page_config(page_title="Design Exploration – TITE-CRM", layout="wide")
st.title("TITE-CRM Design Exploration")
st.caption("Sweep one design parameter while holding all others fixed. "
           "Metrics are averaged over repeated simulated trials.")

_N_LEVELS = 5
_DEFAULT_T1 = [0.05, 0.10, 0.15, 0.22, 0.30]
_DEFAULT_T2 = [0.10, 0.20, 0.33, 0.45, 0.55]

# ── Sidebar: base scenario ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Base Scenario")

    st.subheader("True toxicity rates")
    cols = st.columns(2)
    true_t1, true_t2 = [], []
    for i in range(_N_LEVELS):
        c = cols[i % 2]
        true_t1.append(c.number_input(f"Tox1 L{i}", 0.0, 1.0, _DEFAULT_T1[i],
                                       step=0.01, key=f"t1_{i}"))
        true_t2.append(c.number_input(f"Tox2 L{i}", 0.0, 1.0, _DEFAULT_T2[i],
                                       step=0.01, key=f"t2_{i}"))

    st.subheader("Targets & accrual")
    target_tox1 = st.number_input("Target Tox1", 0.01, 0.99, 0.15, step=0.01)
    target_tox2 = st.number_input("Target Tox2", 0.01, 0.99, 0.33, step=0.01)
    p_surgery   = st.number_input("P(surgery)",  0.0,  1.0,  0.80, step=0.01)
    accrual     = st.number_input("Accrual/month", 0.1, 10.0, 1.5, step=0.1)

    st.subheader("CRM settings")
    sigma_base  = st.number_input("Sigma",       0.1, 5.0, 1.0, step=0.1)
    ewoc_on     = st.checkbox("EWOC ON", value=True)
    ewoc_alpha  = st.number_input("EWOC α", 0.05, 0.60, 0.25, step=0.01,
                                   disabled=not ewoc_on)
    max_n       = st.number_input("Max N",        6, 120, 27, step=1)
    cohort_sz   = st.number_input("Cohort size",  1,  12,  3, step=1)
    start_lvl   = st.number_input("Start level",  0, _N_LEVELS - 1, 0, step=1)

    st.subheader("Timing (days)")
    incl_to_rt  = st.number_input("Incl → RT",       0, 180,  21, step=1)
    rt_dur      = st.number_input("RT duration",      1,  90,  14, step=1)
    rt_to_surg  = st.number_input("RT end → Surgery", 0, 365,  84, step=1)
    tox2_win    = st.number_input("Tox2 window",      7, 180,  30, step=1)

    st.subheader("Prior / skeleton")
    hw_t1 = st.number_input("Halfwidth Tox1", 0.01, 0.40, 0.10, step=0.01)
    nu_t1 = st.number_input("Prior nu Tox1",  1, _N_LEVELS, 3, step=1)
    hw_t2 = st.number_input("Halfwidth Tox2", 0.01, 0.40, 0.10, step=0.01)
    nu_t2 = st.number_input("Prior nu Tox2",  1, _N_LEVELS, 3, step=1)
    burn_in           = st.checkbox("Burn-in",              value=True)
    enforce_guardrail = st.checkbox("Guardrail",            value=True)
    restrict_final    = st.checkbox("Restrict MTD to tried",value=True)

# Compute skeletons
try:
    skel_t1 = dfcrm_getprior(hw_t1, target_tox1, nu_t1, _N_LEVELS)
    skel_t2 = dfcrm_getprior(hw_t2, target_tox2, nu_t2, _N_LEVELS)
    _skel_ok = True
except Exception as e:
    st.error(f"Skeleton error: {e}")
    _skel_ok = False
    skel_t1 = skel_t2 = None

true_t1_arr = np.asarray(true_t1)
true_t2_arr = np.asarray(true_t2)

# True optimal dose banner
if _skel_ok:
    opt = _true_optimal(true_t1_arr, true_t2_arr, target_tox1, target_tox2)
    st.info(f"True optimal dose (argmin max-excess): **L{opt}** — "
            f"Tox1 = {true_t1[opt]:.3f}, Tox2 = {true_t2[opt]:.3f}")

# ── Sweep controls ────────────────────────────────────────────────────────────
st.subheader("Parameter Sweep")

ctrl_col, _ = st.columns([2, 3])
with ctrl_col:
    param_name = st.selectbox(
        "Parameter to sweep",
        ["sigma", "ewoc_alpha", "max_n", "cohort_size"],
        format_func={"sigma": "Sigma (prior precision)",
                     "ewoc_alpha": "EWOC α (overdose threshold)",
                     "max_n": "Max N (sample size)",
                     "cohort_size": "Cohort size"}.get,
    )

    if param_name == "sigma":
        c1, c2, c3 = st.columns(3)
        sig_min = c1.number_input("Min σ", 0.1, 4.9, 0.3, step=0.1)
        sig_max = c2.number_input("Max σ", sig_min + 0.1, 5.0, 2.0, step=0.1)
        sig_pts = c3.slider("Points", 3, 20, 8)
        _param_values = np.linspace(sig_min, sig_max, sig_pts).tolist()
        param_label   = "Sigma (σ)"
        param_type    = "continuous"

    elif param_name == "ewoc_alpha":
        c1, c2, c3 = st.columns(3)
        ea_min = c1.number_input("Min α", 0.05, 0.55, 0.15, step=0.01)
        ea_max = c2.number_input("Max α", ea_min + 0.01, 0.60, 0.45, step=0.01)
        ea_pts = c3.slider("Points", 3, 20, 8)
        inc_off = st.checkbox("Include EWOC OFF", value=True)
        _param_values = ([None] if inc_off else []) + \
                        np.linspace(ea_min, ea_max, ea_pts).tolist()
        param_label   = "EWOC α"
        param_type    = "continuous"

    elif param_name == "max_n":
        _param_values = st.multiselect(
            "Max N values", [12, 15, 18, 21, 24, 27, 30, 33, 36],
            default=[18, 21, 24, 27, 30])
        param_label = "Max N"
        param_type  = "discrete"

    else:  # cohort_size
        _param_values = st.multiselect(
            "Cohort sizes", [1, 2, 3, 4, 5, 6], default=[1, 2, 3, 4])
        param_label = "Cohort size"
        param_type  = "discrete"

    st.divider()
    n_sim_input = st.slider("n_sim (per point)", 50, 2000, 200, step=50)
    seed_val    = st.number_input("Seed", 0, 99999, 42, step=1)
    speed_mode  = st.checkbox(
        "Speed mode",
        help="Cuts n_sim to max(50, n_sim÷4); caps grid to ≤8 continuous / ≤3 discrete.")

    # Apply speed mode
    if speed_mode:
        n_sim_eff = max(50, int(n_sim_input) // 4)
        pv_eff = (_param_values[:8] if param_type == "continuous"
                  else _param_values[:3])
        st.caption(f"Speed mode — {n_sim_eff} sims × {len(pv_eff)} points")
    else:
        n_sim_eff = int(n_sim_input)
        pv_eff    = _param_values
        st.caption(f"{n_sim_eff} sims × {len(pv_eff)} points = "
                   f"{n_sim_eff * len(pv_eff):,} total trials")

    run_btn = st.button("▶ Run Sweep", type="primary",
                        disabled=not _skel_ok or len(pv_eff) == 0)

# ── Run & display results ─────────────────────────────────────────────────────
if run_btn and _skel_ok and len(pv_eff) > 0:
    base_ss = dict(
        target_tox1=target_tox1, target_tox2=target_tox2,
        p_surgery=p_surgery, sigma=sigma_base,
        ewoc_on=ewoc_on, ewoc_alpha=ewoc_alpha,
        max_n=int(max_n), cohort_size=int(cohort_sz),
        start_level=int(start_lvl), accrual_per_month=accrual,
        incl_to_rt=int(incl_to_rt), rt_dur=int(rt_dur),
        rt_to_surg=int(rt_to_surg), tox2_win=int(tox2_win),
        max_step=1, gh_n=61,
        burn_in=burn_in, enforce_guardrail=enforce_guardrail,
        restrict_final_to_tried=restrict_final,
    )
    with st.spinner(f"Running {n_sim_eff * len(pv_eff):,} trials…"):
        df = run_parameter_sweep(
            param_name, pv_eff, base_ss,
            true_t1_arr, true_t2_arr, skel_t1, skel_t2,
            n_sim_eff, int(seed_val))
    st.session_state["_de_df"]    = df
    st.session_state["_de_label"] = param_label

if "_de_df" in st.session_state:
    df  = st.session_state["_de_df"]
    lbl = st.session_state["_de_label"]

    # ── Three line charts ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(len(df))
    specs = [
        ("quality_score",        "Quality score",      "#2563eb"),
        ("pct_correct_selection","% Correct selection","#16a34a"),
        ("overdose_rate",        "Overdose rate (%)",  "#dc2626"),
    ]
    for ax, (col, ylabel, color) in zip(axes, specs):
        ax.plot(x, df[col].values, marker="o", color=color, lw=2, ms=5)
        ax.set_xticks(x)
        ax.set_xticklabels(df["param_label"].tolist(),
                           rotation=35 if len(x) > 6 else 0,
                           ha="right", fontsize=8)
        ax.set_xlabel(lbl, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", lw=0.5, alpha=0.3)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig)

    # ── Summary table ──────────────────────────────────────────────────────
    disp = df[["param_label", "quality_score",
               "pct_correct_selection", "overdose_rate"]].copy()
    disp.columns = [lbl, "Quality score", "% Correct selection", "Overdose rate (%)"]
    disp["Quality score"]         = disp["Quality score"].round(4)
    disp["% Correct selection"]   = disp["% Correct selection"].round(1)
    disp["Overdose rate (%)"]     = disp["Overdose rate (%)"].round(1)
    st.dataframe(disp, use_container_width=True, hide_index=True)
