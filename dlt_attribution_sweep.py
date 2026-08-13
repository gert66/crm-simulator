#!/usr/bin/env python3
"""Sweep the attribution probability of the contested acute DLT.

Two paired series, over all five acute toxicity scenarios:

  attribution  n stays at full weight, y is scaled by p
               -- the non-binary DLT: the patient was fully observed, the event
               is only partly attributable to treatment.
  weight       n and y are both scaled by w
               -- power-prior discounting of the whole historical block, which
               also discounts the five patients who had no event.

At p = 1 both series coincide with the conventional binary DLT. At p = 0 the
attribution series is equivalent to never recording the event, while the weight
series discards the historical block entirely -- a genuinely different thing,
and the clearest illustration of why the two are not interchangeable.

Usage:
    python dlt_attribution_sweep.py [--n-sim 1000] [--seed 20260813]
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
TARGET1, TARGET2 = 0.20, 0.33
P_GRID = [0.00, 0.05, 0.10, 0.25, 0.50, 1.00]


def true_mtd_of(true_t1) -> int:
    safe = [i for i, p in enumerate(true_t1) if p <= TARGET1]
    return max(safe) if safe else 0


def run(true_t1, *, attribution=1.0, weight=1.0, override=0, n_sim=1000, seed=0):
    skel1 = np.asarray(metc.ACUTE_PRIOR, dtype=float)
    skel2 = np.asarray(metc.SUBACUTE_PRIOR, dtype=float)
    kw = dict(
        p_surgery=0.80, target1=TARGET1, target2=TARGET2, skel1=skel1, skel2=skel2,
        sigma=1.0, start_level=2, max_n=30, cohort_size=3, accrual_per_month=0.75,
        incl_to_rt=21, rt_dur=14, rt_to_surg=42, tox1_win=56, tox2_win=30,
        max_step=1, gh_n=61, enforce_guardrail=True, restrict_final_to_tried=True,
        ewoc_on=False, ewoc_application=sim.EWOC_APP_OFF, ewoc_alpha=None,
        n_safe_d1=6, n_safe_d1_dlt=1,
        hist_weight=weight, hist_dlt_attribution=attribution,
        escalation_override_n=override, p_stop=1.0,
        require_full_tox1_fu_before_escalation=True, collect_trace=False,
    )
    rm = np.random.default_rng(seed)
    sels, acute, subac, top = [], [], [], []
    for _ in range(n_sim):
        r = np.random.default_rng(int(rm.integers(0, 2**32 - 1)))
        s, pts, _d, _t, _st = sim.run_tite_crm(
            true_t1=true_t1, true_t2=metc.TRUE_SUBACUTE, burn_in=False, rng=r, **kw)
        new = [p for p in pts if p["arrival"] >= 0]
        sels.append(s)
        acute.append(sum(bool(p["has_tox1"]) for p in new))
        subac.append(sum(bool(p["has_tox2"]) for p in new))
        top.append(max(p["dose"] for p in new) == len(DOSE_LABELS) - 1)
    return (np.asarray(sels), float(np.mean(acute)), float(np.mean(subac)),
            100.0 * float(np.mean(top)))


def _cell(job):
    """One (scenario, series, p) cell — top level so it can be sent to a worker."""
    (scenario, true_t1, skey, slabel, kind, override, p, n_sim, seed) = job
    tm = true_mtd_of(true_t1)
    kwargs = dict(override=override, n_sim=n_sim, seed=seed)
    if kind == "attribution":
        kwargs["attribution"] = p
    else:
        kwargs["weight"] = p
    sels, ac, sa, top = run(true_t1, **kwargs)
    return {
        "scenario": scenario, "series": skey, "series_label": slabel,
        "p": p, "true_mtd": tm,
        "true_mtd_label": f"L{tm} ({DOSE_LABELS[tm]})",
        "correct_pct": 100.0 * float(np.mean(sels == tm)),
        "too_high_pct": 100.0 * float(np.mean(sels > tm)),
        "too_low_pct": 100.0 * float(np.mean(sels < tm)),
        "mean_acute_tox": ac, "mean_subacute_tox": sa,
        "ever_reached_top_pct": top,
        **{f"sel_L{d}_pct": 100.0 * float(np.mean(sels == d))
           for d in range(len(DOSE_LABELS))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sim", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--outdir", type=Path, default=Path("dlt_attribution_sensitivity"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)

    series = [
        ("attribution", "Non-binary DLT (n full, y scaled)", dict(kind="attribution", override=0)),
        ("weight",      "Block discount (n and y scaled)",   dict(kind="weight", override=0)),
        ("attribution_override", "Non-binary DLT + escalation override",
         dict(kind="attribution", override=3)),
    ]

    jobs = []
    for scenario, true_t1 in metc.ACUTE_SCENARIOS.items():
        for skey, slabel, cfg in series:
            for p in P_GRID:
                seed = abs(hash((scenario, skey, p, args.seed))) % (2**31)
                jobs.append((scenario, list(map(float, true_t1)), skey, slabel,
                             cfg["kind"], cfg["override"], p, args.n_sim, seed))

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(_cell, jobs), start=1):
            rows.append(row)
            print(f"[{i:>2}/{len(jobs)}] {row['scenario']:15s} {row['series']:22s} "
                  f"p={row['p']:<5} correct={row['correct_pct']:5.1f} "
                  f"too_high={row['too_high_pct']:5.1f}", flush=True)

    df = pd.DataFrame(rows)
    csv = args.outdir / "dlt_attribution_sweep.csv"
    df.to_csv(csv, index=False)
    meta = {
        "n_sim": args.n_sim, "seed": args.seed, "p_grid": P_GRID,
        "target_acute": TARGET1, "target_subacute": TARGET2,
        "dose_labels": DOSE_LABELS,
        "scenarios": {k: list(map(float, v)) for k, v in metc.ACUTE_SCENARIOS.items()},
        "true_subacute": list(map(float, metc.TRUE_SUBACUTE)),
        "acute_prior": list(map(float, metc.ACUTE_PRIOR)),
        "subacute_prior": list(map(float, metc.SUBACUTE_PRIOR)),
    }
    (args.outdir / "dlt_attribution_sweep_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {csv}")

    piv = df[df["series"] == "attribution"].pivot(
        index="p", columns="scenario", values="correct_pct")
    print("\nCorrect MTD % by attribution probability:")
    print(piv.round(1).to_string())


if __name__ == "__main__":
    main()
