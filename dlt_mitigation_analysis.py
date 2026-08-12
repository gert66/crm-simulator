#!/usr/bin/env python3
"""Full grid: how to handle the contested acute DLT in the L1 initialization cohort.

Compares five ways of letting the pre-amendment L1 data (6 patients, 1 contested
acute DLT -- SAE MERGE-011, causality assessed "possible") enter the TITE-CRM,
across all five acute toxicity scenarios:

  A  DLT counted at full weight              (current default)
  B  DLT counted, historical block at 50%    (power-prior discount)
  C  DLT counted, historical block at 25%
  E  DLT at full weight + escalation override after 3 clean fully-followed pts
  H  DLT not counted                         (counterfactual: adjudicated unrelated)

Plus the 6+3 design as a reference arm.

Outputs a tidy CSV consumed by the summary report.

Usage:
    python dlt_mitigation_analysis.py [--n-sim 400] [--seed 20260812]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

with contextlib.redirect_stderr(io.StringIO()):
    import sim
    import metc_simulation_report as metc

DOSE_LABELS = metc.DOSE_LABELS
TARGET1, TARGET2 = 0.20, 0.33

# label, n_safe_d1_dlt, hist_weight, escalation_override_n
OPTIONS = [
    ("A",   "DLT telt volledig mee (huidige situatie)",  1, 1.00, 0),
    ("B",   "Downweging historie naar 50%",              1, 0.50, 0),
    ("C",   "Downweging historie naar 25%",              1, 0.25, 0),
    ("E",   "Escalatie-override na 3 schone patiënten",  1, 1.00, 3),
    ("E+B", "Override 3 + downweging 50%",               1, 0.50, 3),
    ("E+C", "Override 3 + downweging 25%",               1, 0.25, 3),
    ("H",   "DLT telt niet mee (geadjudiceerd unrelated)", 0, 1.00, 0),
]


def true_mtd_of(true_t1) -> int:
    safe = [i for i, p in enumerate(true_t1) if p <= TARGET1]
    return max(safe) if safe else 0


def run_crm(true_t1, dlt, hist_weight, override_n, n_sim, seed):
    skel1 = np.asarray(metc.ACUTE_PRIOR, dtype=float)
    skel2 = np.asarray(metc.SUBACUTE_PRIOR, dtype=float)
    kw = dict(
        p_surgery=0.80, target1=TARGET1, target2=TARGET2, skel1=skel1, skel2=skel2,
        sigma=1.0, start_level=2, max_n=30, cohort_size=3, accrual_per_month=0.75,
        incl_to_rt=21, rt_dur=14, rt_to_surg=42, tox1_win=56, tox2_win=30,
        max_step=1, gh_n=61, enforce_guardrail=True, restrict_final_to_tried=True,
        ewoc_on=False, ewoc_application=sim.EWOC_APP_OFF, ewoc_alpha=None,
        n_safe_d1=6, n_safe_d1_dlt=dlt, hist_weight=hist_weight,
        escalation_override_n=override_n, p_stop=1.0,
        require_full_tox1_fu_before_escalation=True, collect_trace=False,
    )
    rm = np.random.default_rng(seed)
    sels, acute, subac, ever_top = [], [], [], []
    for _ in range(n_sim):
        r = np.random.default_rng(int(rm.integers(0, 2**32 - 1)))
        s, pts, _days, _tr, _st = sim.run_tite_crm(
            true_t1=true_t1, true_t2=metc.TRUE_SUBACUTE, burn_in=False, rng=r, **kw)
        new = [p for p in pts if p["arrival"] >= 0]
        sels.append(s)
        acute.append(sum(bool(p["has_tox1"]) for p in new))
        subac.append(sum(bool(p["has_tox2"]) for p in new))
        ever_top.append(max(p["dose"] for p in new) == len(DOSE_LABELS) - 1)
    return np.asarray(sels), float(np.mean(acute)), float(np.mean(subac)), 100.0 * float(np.mean(ever_top))


def run_63(true_t1, n_sim, seed):
    rm = np.random.default_rng(seed)
    sels, acute, subac = [], [], []
    for _ in range(n_sim):
        r = np.random.default_rng(int(rm.integers(0, 2**32 - 1)))
        s, pts, _days, _nb = sim.run_tite_6plus3(
            true_t1=true_t1, p_surgery=0.80, true_t2=metc.TRUE_SUBACUTE,
            start_level=2, max_n=30, accrual_per_month=0.75,
            incl_to_rt=21, rt_dur=14, rt_to_surg=42, tox1_win=56, tox2_win=30, rng=r)
        sels.append(s)
        acute.append(sum(bool(p["has_tox1"]) for p in pts))
        subac.append(sum(bool(p["has_tox2"]) for p in pts))
    return np.asarray(sels), float(np.mean(acute)), float(np.mean(subac))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--outdir", type=Path, default=Path("dlt_attribution_sensitivity"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scenario, true_t1 in metc.ACUTE_SCENARIOS.items():
        tm = true_mtd_of(true_t1)
        for code, label, dlt, hw, ov in OPTIONS:
            seed = abs(hash((scenario, code, args.seed))) % (2**31)
            sels, ac, sa, top = run_crm(true_t1, dlt, hw, ov, args.n_sim, seed)
            rows.append({
                "scenario": scenario, "option": code, "option_label": label,
                "design": "TITE-CRM", "true_mtd": tm,
                "true_mtd_label": f"L{tm} ({DOSE_LABELS[tm]})",
                "correct_pct": 100.0 * float(np.mean(sels == tm)),
                "too_high_pct": 100.0 * float(np.mean(sels > tm)),
                "too_low_pct": 100.0 * float(np.mean(sels < tm)),
                "mean_acute_tox": ac, "mean_subacute_tox": sa,
                "ever_reached_top_pct": top,
                **{f"sel_L{d}_pct": 100.0 * float(np.mean(sels == d)) for d in range(len(DOSE_LABELS))},
            })
        seed = abs(hash((scenario, "63", args.seed))) % (2**31)
        sels, ac, sa = run_63(true_t1, args.n_sim, seed)
        rows.append({
            "scenario": scenario, "option": "6+3", "option_label": "Huidig 6+3 design",
            "design": "6+3", "true_mtd": tm,
            "true_mtd_label": f"L{tm} ({DOSE_LABELS[tm]})",
            "correct_pct": 100.0 * float(np.mean(sels == tm)),
            "too_high_pct": 100.0 * float(np.mean(sels > tm)),
            "too_low_pct": 100.0 * float(np.mean(sels < tm)),
            "mean_acute_tox": ac, "mean_subacute_tox": sa,
            "ever_reached_top_pct": np.nan,
            **{f"sel_L{d}_pct": 100.0 * float(np.mean(sels == d)) for d in range(len(DOSE_LABELS))},
        })

    df = pd.DataFrame(rows)
    csv_path = args.outdir / "dlt_mitigation_summary.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "n_sim": args.n_sim, "seed": args.seed,
        "target_acute": TARGET1, "target_subacute": TARGET2,
        "dose_labels": DOSE_LABELS,
        "scenarios": {k: list(map(float, v)) for k, v in metc.ACUTE_SCENARIOS.items()},
        "true_subacute": list(map(float, metc.TRUE_SUBACUTE)),
        "options": [{"code": c, "label": l, "dlt": d, "hist_weight": w, "override_n": o}
                    for c, l, d, w, o in OPTIONS],
    }
    (args.outdir / "dlt_mitigation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")

    print("\nCorrect MTD % per optie:")
    piv = df[df["design"] == "TITE-CRM"].pivot(index="option", columns="scenario", values="correct_pct")
    print(piv.round(1).to_string())
    print("\nTe hoge (onveilige) MTD % per optie:")
    piv2 = df[df["design"] == "TITE-CRM"].pivot(index="option", columns="scenario", values="too_high_pct")
    print(piv2.round(1).to_string())


if __name__ == "__main__":
    main()
