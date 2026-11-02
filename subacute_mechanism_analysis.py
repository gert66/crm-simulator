#!/usr/bin/env python3
"""Why does the revised subacute specification hit Sama's results and not ours?

Appendix A section 1.3 states the rule actually used: fit both endpoints, take
the dose closest to target for each, then assign the LOWER of the two.  Our runs
used the acute endpoint alone, so the subacute specification could not affect
dose assignment at all.  This script tests that explanation directly, and also
checks the 6+3 design, whose subacute rules are absolute counts and were never
recalibrated for an endpoint carrying a 25% background rate.

Scenario names follow the appendix (optimistic / realistic / pessimistic /
steep / shallow) and the dose levels are the appendix's -1..3, i.e. our L0..L4.

Usage:
    python subacute_mechanism_analysis.py [--n-sim 3000]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

with contextlib.redirect_stderr(io.StringIO()):
    import sim
    import metc_simulation_report as metc

TARGET1, TARGET2 = 0.20, 0.33
DOSE_LABELS = ["5x4 Gy", "5x5 Gy", "5x6 Gy", "5x7 Gy", "5x8 Gy"]
APPENDIX_LEVEL = {0: "-1", 1: "0", 2: "1", 3: "2", 4: "3"}

# Acute true probabilities, Appendix A Table A2.  Same numbers as our own
# scenario set, renamed to the appendix's labels.
ACUTE = {
    "optimistic":  [0.01, 0.02, 0.06, 0.10, 0.15],
    "realistic":   [0.01, 0.04, 0.12, 0.17, 0.27],
    "pessimistic": [0.01, 0.05, 0.15, 0.22, 0.30],
    "steep":       [0.01, 0.02, 0.08, 0.24, 0.34],
    "shallow":     [0.10, 0.13, 0.15, 0.18, 0.24],
}
SCEN = list(ACUTE)

SUBACUTE = {
    "old": dict(prior=[0.010, 0.036, 0.084, 0.157, 0.250],
                true=[0.02, 0.05, 0.10, 0.15, 0.25]),
    "new": dict(prior=[0.25, 0.26, 0.27, 0.28, 0.29],
                true=[0.25, 0.26, 0.27, 0.28, 0.29]),
}

# Appendix A Table A1 timing, so the comparison is on their parameters.
COMMON = dict(p_surgery=0.96, accrual_per_month=30.0 / 21.0,
              incl_to_rt=14, rt_dur=0, rt_to_surg=56, tox1_win=56, tox2_win=28)


def true_closest(p):
    """MTD as the appendix now defines it: the dose closest to target."""
    return int(np.argmin(np.abs(np.asarray(p, float) - TARGET1)))


def _kwargs(dose_rule, skel2, n_hist):
    return dict(
        target1=TARGET1, target2=TARGET2,
        skel1=np.asarray(metc.ACUTE_PRIOR, dtype=float),
        skel2=np.asarray(skel2, dtype=float),
        sigma=1.0, start_level=1, max_n=30 + n_hist, cohort_size=3,
        max_step=1, gh_n=61,
        enforce_guardrail=True, restrict_final_to_tried=False,
        ewoc_on=False, ewoc_application=sim.EWOC_APP_OFF, ewoc_alpha=None,
        burn_in=False,
        n_safe_d1=n_hist, n_safe_d1_dlt=(1 if n_hist else 0),
        hist_weight=1.0, hist_dlt_attribution=1.0,
        dose_rule=dose_rule, escalation_override_n=0, p_stop=1.0,
        require_full_tox1_fu_before_escalation=True, collect_trace=False,
        **COMMON,
    )


def _job(args):
    scenario, sub_key, arm, n_hist, n_sim, seed = args
    t1 = np.asarray(ACUTE[scenario], dtype=float)
    t2 = np.asarray(SUBACUTE[sub_key]["true"], dtype=float)
    master = np.random.default_rng(seed)
    sels, n_ac, n_sa, used = [], [], [], []

    for _ in range(n_sim):
        r = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
        if arm == "6+3":
            s, pts, _d, _nb = sim.run_tite_6plus3(
                true_t1=t1, true_t2=t2, start_level=1, max_n=39, rng=r,
                **COMMON)
            new = pts
        else:
            kw = _kwargs(arm, SUBACUTE[sub_key]["prior"], n_hist)
            s, pts, _d, _t, _st = sim.run_tite_crm(true_t1=t1, true_t2=t2,
                                                   rng=r, **kw)
            new = [p for p in pts if p["arrival"] >= 0]
        sels.append(s)
        n_ac.append(sum(bool(p["has_tox1"]) for p in new))
        n_sa.append(sum(bool(p["has_tox2"]) for p in new))
        used.append(len(new))

    sels = np.asarray(sels)
    tc = true_closest(t1)
    return {
        "scenario": scenario, "subacute": sub_key, "arm": arm,
        "conditioned_on": n_hist, "true_mtd_closest": tc,
        "pcs_pct": 100.0 * float(np.mean(sels == tc)),
        "above_pct": 100.0 * float(np.mean(sels > tc)),
        "mean_acute_dlt": float(np.mean(n_ac)),
        "mean_subacute_dlt": float(np.mean(n_sa)),
        "mean_n": float(np.mean(used)),
        **{f"sel_{APPENDIX_LEVEL[d]}_pct": 100.0 * float(np.mean(sels == d))
           for d in range(5)},
    }


def six_plus_three_subacute_maths() -> pd.DataFrame:
    """The 6+3 subacute rules are counts, so their behaviour is exact arithmetic.

    Original MERGE rules as implemented in run_tite_6plus3: escalate from a
    6-patient cohort only if subacute events <= 1, stop if >= 3.
    """
    def pmf_at_least(n, p, k):
        return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))

    def pmf_at_most(n, p, k):
        return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(0, k + 1))

    rows = []
    for spec, probs in (("old", SUBACUTE["old"]["true"]),
                        ("new", SUBACUTE["new"]["true"])):
        for d, p in enumerate(probs):
            rows.append({
                "subacute": spec, "dose_level": APPENDIX_LEVEL[d],
                "true_p_subacute": p,
                "P(escalate: <=1 of 6)": pmf_at_most(6, p, 1),
                "P(stop: >=3 of 6)": pmf_at_least(6, p, 3),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sim", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--outdir", type=Path, default=Path("subacute_mechanism"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("True MTD per scenario, appendix definition (dose closest to 0.20):")
    for s in SCEN:
        print(f"  {s:12s} acute={ACUTE[s]}  -> dose level {APPENDIX_LEVEL[true_closest(ACUTE[s])]}")
    print()

    maths = six_plus_three_subacute_maths()
    maths.to_csv(args.outdir / "six_plus_three_subacute_maths.csv", index=False)
    print("6+3 subacute rules are absolute counts. Exact probabilities per cohort of 6:")
    print(maths.round(3).to_string(index=False))
    print()

    jobs = []
    for scenario in SCEN:
        for sub_key in SUBACUTE:
            for arm, n_hist in (("argmin", 0), ("min_closest", 0),
                                ("min_closest", 6), ("6+3", 0)):
                seed = abs(hash((scenario, sub_key, arm, n_hist, args.seed))) % (2**31)
                jobs.append((scenario, sub_key, arm, n_hist, args.n_sim, seed))

    with ProcessPoolExecutor(max_workers=10) as ex:
        rows = list(ex.map(_job, jobs))
    df = pd.DataFrame(rows)
    df["arm_label"] = df.apply(
        lambda r: {"argmin": "CRM, acute only (ours)",
                   "min_closest": ("CRM, min(acute, subacute) (appendix)"
                                   + (" + first 6 fixed" if r["conditioned_on"] else "")),
                   "6+3": "6+3"}[r["arm"]], axis=1)
    df.to_csv(args.outdir / "subacute_mechanism_summary.csv", index=False)

    order = ["CRM, acute only (ours)", "CRM, min(acute, subacute) (appendix)",
             "CRM, min(acute, subacute) (appendix) + first 6 fixed", "6+3"]

    print("\nProbability of correct selection (%), old vs new subacute specification")
    for lab in order:
        sub = df[df.arm_label == lab]
        piv = sub.pivot(index="subacute", columns="scenario", values="pcs_pct")
        piv = piv.reindex(["old", "new"])[SCEN]
        delta = piv.loc["new"] - piv.loc["old"]
        print(f"\n{lab}")
        print(piv.round(1).to_string())
        print("  change  " + "  ".join(f"{v:+.1f}" for v in delta))

    print("\n\nMean subacute DLTs per trial")
    piv = df.pivot_table(index="arm_label", columns="subacute",
                         values="mean_subacute_dlt").reindex(order)
    print(piv.round(2).to_string())

    (args.outdir / "subacute_mechanism_meta.json").write_text(json.dumps({
        "n_sim": args.n_sim, "seed": args.seed, "acute_scenarios": ACUTE,
        "subacute_specs": SUBACUTE, "common": COMMON,
        "mtd_definition": "dose closest to acute target 0.20 (Appendix A Table A2)",
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.outdir / 'subacute_mechanism_summary.csv'}")


if __name__ == "__main__":
    main()
