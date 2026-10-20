#!/usr/bin/env python3
"""Re-run the switch analysis under the revised subacute DLT definition.

The subacute DLT no longer has to be attributable to radiotherapy, so it now
carries the background complication rate.  Koen proposes prior AND true subacute
probabilities of 0.25, 0.26, 0.27, 0.28, 0.29 against an unchanged 0.33 target.

This script runs the current and the revised subacute specification side by side,
for both dose-assignment rules, so we can see exactly what the change does and
what it leaves untouched.

Usage:
    python subacute_revision_analysis.py [--n-sim 3000]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

with contextlib.redirect_stderr(io.StringIO()):
    import sim
    import metc_simulation_report as metc

DOSE_LABELS = metc.DOSE_LABELS
N_LEVELS = len(DOSE_LABELS)
TARGET1, TARGET2 = 0.20, 0.33

SCEN = ["Acute low", "Acute middle", "Acute high", "Acute shallow", "Acute steep"]

SUBACUTE_SPECS = {
    "current": dict(
        prior=list(metc.SUBACUTE_PRIOR),
        true=list(metc.TRUE_SUBACUTE),
        label="Current: RT-attributable subacute DLT",
    ),
    "revised": dict(
        prior=[0.25, 0.26, 0.27, 0.28, 0.29],
        true=[0.25, 0.26, 0.27, 0.28, 0.29],
        label="Revised: subacute DLT incl. background rate",
    ),
}

# Current trial state, as in switch_decision_analysis.py
N_HIST, N_HIST_DLT, N_FORWARD, N_TOTAL, NEXT_LEVEL = 9, 1, 21, 30, 2

COMMON = dict(
    p_surgery=0.80, accrual_per_month=0.75,
    incl_to_rt=21, rt_dur=14, rt_to_surg=42, tox1_win=56, tox2_win=30,
)


def _kwargs(dose_rule, skel2, start_level):
    return dict(
        target1=TARGET1, target2=TARGET2,
        skel1=np.asarray(metc.ACUTE_PRIOR, dtype=float),
        skel2=np.asarray(skel2, dtype=float),
        sigma=1.0, start_level=start_level, max_n=N_TOTAL, cohort_size=3,
        max_step=1, gh_n=61,
        enforce_guardrail=True, restrict_final_to_tried=False,
        ewoc_on=False, ewoc_application=sim.EWOC_APP_OFF, ewoc_alpha=None,
        burn_in=False,
        n_safe_d1=N_HIST, n_safe_d1_dlt=N_HIST_DLT,
        hist_weight=1.0, hist_dlt_attribution=1.0,
        dose_rule=dose_rule, escalation_override_n=0, p_stop=1.0,
        require_full_tox1_fu_before_escalation=True, collect_trace=False,
        **COMMON,
    )


def _job(args):
    scenario, spec_key, dose_rule, n_sim, seed = args
    spec = SUBACUTE_SPECS[spec_key]
    true_t1 = np.asarray(metc.ACUTE_SCENARIOS[scenario], dtype=float)
    true_t2 = np.asarray(spec["true"], dtype=float)
    start = NEXT_LEVEL if dose_rule == "argmin" else 1
    kw = _kwargs(dose_rule, spec["prior"], start)

    master = np.random.default_rng(seed)
    sels, n_ac, n_sa, n_surg = [], [], [], []
    for _ in range(n_sim):
        r = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
        s, pts, _d, _t, _st = sim.run_tite_crm(true_t1=true_t1, true_t2=true_t2,
                                               rng=r, **kw)
        new = [p for p in pts if p["arrival"] >= 0]
        sels.append(s)
        n_ac.append(sum(bool(p["has_tox1"]) for p in new))
        n_sa.append(sum(bool(p["has_tox2"]) for p in new))
        n_surg.append(sum(bool(p["has_surgery"]) for p in new))
    sels = np.asarray(sels)

    safe = [i for i, p in enumerate(true_t1) if p <= TARGET1]
    tm = max(safe) if safe else 0
    tc = int(np.argmin(np.abs(true_t1 - TARGET1)))
    return {
        "scenario": scenario, "subacute": spec_key, "dose_rule": dose_rule,
        "true_mtd": tm, "true_closest": tc,
        "correct_pct": 100.0 * float(np.mean(sels == tm)),
        "correct_closest_pct": 100.0 * float(np.mean(sels == tc)),
        "mean_acute_dlt": float(np.mean(n_ac)),
        "mean_subacute_dlt": float(np.mean(n_sa)),
        "mean_surgeries": float(np.mean(n_surg)),
        **{f"sel_L{d}_pct": 100.0 * float(np.mean(sels == d)) for d in range(N_LEVELS)},
    }


def subacute_binding_table(skel2, target2=TARGET2) -> pd.DataFrame:
    """At how many observed subacute events does each dose cross the target?

    With a skeleton this compressed the power model can barely separate the dose
    levels, so the endpoint tends to clear every dose or veto every dose at once.
    This tabulates where that switch happens, for a plausible number of patients
    with completed subacute follow-up at one level.
    """
    skel2 = np.asarray(skel2, dtype=float)
    rows = []
    for n in (6, 9, 12, 15, 18, 21):
        for y in range(0, n + 1):
            n2 = np.zeros(N_LEVELS); y2 = np.zeros(N_LEVELS)
            n2[2], y2[2] = n, y          # events observed at L2
            pm, _ = sim.crm_posterior_summaries(1.0, skel2, n2, y2, target2, gh_n=61)
            over = [d for d in range(N_LEVELS) if pm[d] > target2]
            if over:
                rows.append({"n_subacute_evaluable": n, "first_y_over_target": y,
                             "observed_rate": y / n,
                             "lowest_dose_over_target": f"L{min(over)}",
                             "n_doses_over_target": len(over),
                             **{f"pm_L{d}": float(pm[d]) for d in range(N_LEVELS)}})
                break
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sim", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--outdir", type=Path, default=Path("subacute_revision"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Subacute specifications")
    for k, s in SUBACUTE_SPECS.items():
        print(f"  {k:8s} prior={s['prior']}  true={s['true']}")
    print(f"  target subacute = {TARGET2}\n")

    print("How many subacute events before the endpoint starts vetoing doses")
    print("(revised skeleton, events observed at L2)")
    bt = subacute_binding_table(SUBACUTE_SPECS["revised"]["prior"])
    print(bt[["n_subacute_evaluable", "first_y_over_target", "observed_rate",
              "lowest_dose_over_target", "n_doses_over_target"]].round(3).to_string(index=False))
    bt.to_csv(args.outdir / "subacute_binding.csv", index=False)
    print()

    jobs = []
    for scenario in SCEN:
        for spec_key in SUBACUTE_SPECS:
            for rule in ("argmin", "highest_below"):
                seed = abs(hash((scenario, spec_key, rule, args.seed))) % (2**31)
                jobs.append((scenario, spec_key, rule, args.n_sim, seed))

    with ProcessPoolExecutor(max_workers=10) as ex:
        rows = list(ex.map(_job, jobs))

    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "subacute_revision_summary.csv", index=False)

    for rule, rname in (("argmin", "closest-to-target"),
                        ("highest_below", "highest-dose-below-target")):
        print(f"\n=== Dose rule: {rname} ===")
        sub = df[df.dose_rule == rule]
        for metric, title in (("correct_closest_pct", "Selects the dose closest to 0.20 (%)"),
                              ("mean_subacute_dlt", "Mean subacute DLTs in the 21 new patients"),
                              ("mean_acute_dlt", "Mean acute DLTs in the 21 new patients")):
            piv = sub.pivot(index="subacute", columns="scenario", values=metric)
            piv = piv.reindex(["current", "revised"])[SCEN]
            print(f"\n{title}")
            print(piv.round(2).to_string())

    print("\n\nFinal dose distribution, closest-to-target rule (%)")
    for scenario in SCEN:
        print(f"\n{scenario}")
        s = df[(df.scenario == scenario) & (df.dose_rule == "argmin")]
        s = s.set_index("subacute")[[f"sel_L{d}_pct" for d in range(N_LEVELS)]]
        s = s.reindex(["current", "revised"])
        s.columns = [f"L{d}" for d in range(N_LEVELS)]
        print(s.round(1).to_string())

    meta = {"n_sim": args.n_sim, "seed": args.seed,
            "subacute_specs": SUBACUTE_SPECS, "target_acute": TARGET1,
            "target_subacute": TARGET2, "common": COMMON,
            "n_hist": N_HIST, "n_hist_dlt": N_HIST_DLT, "n_total": N_TOTAL}
    (args.outdir / "subacute_revision_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {args.outdir / 'subacute_revision_summary.csv'}")


if __name__ == "__main__":
    main()
