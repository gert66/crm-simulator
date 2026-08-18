#!/usr/bin/env python3
"""Reproduce the published Acute low dose distribution using the R decision rule.

Runs our engine twice on the Acute low scenario -- once with our own rule, once
with the R implementation's rule -- and records the full selection distribution
so it can be placed alongside the distribution reported on the R side.
"""
import contextlib, io, json
with contextlib.redirect_stderr(io.StringIO()):
    import sim, metc_simulation_report as metc
import numpy as np, pandas as pd

# Reported on the R side for Acute low, TITE-CRM, no EWOC, no burn-in.
PUBLISHED = {"L0": 0.0, "L1": 5.1, "L2": 54.5, "L3": 40.3, "L4": 0.0}

skel1 = np.asarray(metc.ACUTE_PRIOR); skel2 = np.asarray(metc.SUBACUTE_PRIOR)
t1 = metc.ACUTE_SCENARIOS["Acute low"]

def dist(rule, restrict, n=1000, seed=4242):
    kw = dict(p_surgery=0.80, target1=0.20, target2=0.33, skel1=skel1, skel2=skel2,
        sigma=1.0, start_level=2, max_n=30, cohort_size=3, accrual_per_month=0.75,
        incl_to_rt=21, rt_dur=14, rt_to_surg=42, tox1_win=56, tox2_win=30,
        max_step=1, gh_n=61, enforce_guardrail=True, restrict_final_to_tried=restrict,
        ewoc_on=False, ewoc_application=sim.EWOC_APP_OFF, ewoc_alpha=None,
        n_safe_d1=6, n_safe_d1_dlt=1, p_stop=1.0, dose_rule=rule,
        require_full_tox1_fu_before_escalation=True, collect_trace=False)
    rm = np.random.default_rng(seed); sels = []
    for _ in range(n):
        r = np.random.default_rng(int(rm.integers(0, 2**32 - 1)))
        s, *_ = sim.run_tite_crm(true_t1=t1, true_t2=metc.TRUE_SUBACUTE,
                                  burn_in=False, rng=r, **kw)
        sels.append(s)
    s = np.asarray(sels)
    return {f"L{d}": round(100 * float(np.mean(s == d)), 1) for d in range(5)}

out = {
    "published_R": PUBLISHED,
    "ours_own_rule": dist("argmin", True),
    "ours_R_rule": dist("highest_below", False),
}
with open("dlt_attribution_sensitivity/reproduction_check.json", "w") as f:
    json.dump(out, f, indent=2)
print(pd.DataFrame(out).T.to_string())
