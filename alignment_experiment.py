#!/usr/bin/env python3
"""Measure how much of the gap to the R implementation each difference explains.

Starts from our configuration and switches one thing at a time towards Sama's:
the restriction to tried doses, the dose-selection rule, and the prior scale.
Writes dlt_attribution_sensitivity/rule_alignment.csv.
"""
import contextlib, io
with contextlib.redirect_stderr(io.StringIO()):
    import sim, metc_simulation_report as metc
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
skel1 = np.asarray(metc.ACUTE_PRIOR); skel2 = np.asarray(metc.SUBACUTE_PRIOR)

def cell(job):
    scen, t1, label, rule, restrict, sigma, seed = job
    tm = max([i for i,p in enumerate(t1) if p<=0.20] or [0])
    kw = dict(p_surgery=0.80, target1=0.20, target2=0.33, skel1=skel1, skel2=skel2,
        sigma=sigma, start_level=2, max_n=30, cohort_size=3, accrual_per_month=0.75,
        incl_to_rt=21, rt_dur=14, rt_to_surg=42, tox1_win=56, tox2_win=30,
        max_step=1, gh_n=61, enforce_guardrail=True, restrict_final_to_tried=restrict,
        ewoc_on=False, ewoc_application=sim.EWOC_APP_OFF, ewoc_alpha=None,
        n_safe_d1=6, n_safe_d1_dlt=1, p_stop=1.0, dose_rule=rule,
        require_full_tox1_fu_before_escalation=True, collect_trace=False)
    rm=np.random.default_rng(seed); sels=[]
    for _ in range(1000):
        r=np.random.default_rng(int(rm.integers(0,2**32-1)))
        s,*_ = sim.run_tite_crm(true_t1=t1, true_t2=metc.TRUE_SUBACUTE, burn_in=False, rng=r, **kw)
        sels.append(s)
    s=np.array(sels)
    return {"scenario":scen,"variant":label,"correct":100*np.mean(s==tm),"too_high":100*np.mean(s>tm)}

VARIANTS = [
    ("A onze code (argmin, restrict, sigma 1.0)", "argmin",         True,  1.0),
    ("B + geen restrict",                          "argmin",         False, 1.0),
    ("C + Sama's dosisregel",                      "highest_below",  False, 1.0),
    ("D + sigma sqrt(1.34)",                       "highest_below",  False, 1.34**0.5),
]
jobs=[]
for scen,t1 in metc.ACUTE_SCENARIOS.items():
    for label,rule,restrict,sigma in VARIANTS:
        jobs.append((scen,list(map(float,t1)),label,rule,restrict,sigma,
                     abs(hash((scen,label,99)))%(2**31)))
rows=[]
with ProcessPoolExecutor(max_workers=4) as ex:
    for r in ex.map(cell,jobs): rows.append(r); print(".",end="",flush=True)
print()
df=pd.DataFrame(rows)
order=[v[0] for v in VARIANTS]
print("\nCorrecte MTD-selectie (%):")
print(df.pivot(index="scenario",columns="variant",values="correct")[order].round(1).to_string())
print("\nGemiddeld correct:")
print(df.groupby("variant")["correct"].mean()[order].round(1).to_string())
print("\nGemiddeld te hoog:")
print(df.groupby("variant")["too_high"].mean()[order].round(1).to_string())
df.to_csv("dlt_attribution_sensitivity/rule_alignment.csv",index=False)
