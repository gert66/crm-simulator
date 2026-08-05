#!/usr/bin/env python3
"""Export one simulated trial as a patient-level "generating file" plus the
matching TITE-CRM decision output, for cross-checking against Sama's R model.

The generating file contains, for every simulated patient, which dose they
were treated at, whether/when they had an acute DLT, whether/when surgery
("OK") took place, and whether/when a subacute DLT occurred -- i.e. exactly
the ground-truth event data Sama currently authors by hand before feeding it
into her TITE-CRM fit. Running the same generated patients through our own
run_tite_crm() and exporting the cohort-by-cohort decisions lets her compare
her model's output against ours on identical input data, isolating whether
any discrepancy comes from data generation or from the model fit itself.

Usage:
    python generate_trial_export.py [--seed SEED] [--scenario NAME] [--outdir DIR]

Scenario names come from metc_simulation_report.ACUTE_SCENARIOS
(default: "Acute middle").
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


def build_generating_file(patients: list[dict]) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(patients, start=1):
        rows.append({
            "patient_id":     i,
            "is_pretreated":  p["arrival"] < 0,
            "dose_level":     p["dose"],
            "dose_label":     DOSE_LABELS[p["dose"]],
            "arrival_day":    round(p["arrival"], 2),
            "rt_start_day":   round(p["rt_start"], 2),
            "dlt1_acute":     p["has_tox1"],
            "dlt1_day":       (round(p["tox1_day"], 2) if p["tox1_day"] is not None else ""),
            "tox1_fu_end_day": round(p["tox1_win_end"], 2),
            "ok_performed":   p["has_surgery"],
            "ok_day":         (round(p["surgery_day"], 2) if p["surgery_day"] is not None else ""),
            "dlt2_subacute":  p["has_tox2"],
            "dlt2_day":       (round(p["tox2_day"], 2) if p["tox2_day"] is not None else ""),
            "tox2_fu_end_day": (round(p["tox2_win_end"], 2) if p["tox2_win_end"] is not None else ""),
        })
    return pd.DataFrame(rows)


def build_crm_output(trace: list[dict], selected: int, study_days: float,
                      stopped_early: bool) -> pd.DataFrame:
    rows = []
    for t in trace:
        rows.append({
            "cohort_step":    t["step"],
            "decision_day":   round(t["decision_day"], 2),
            "n_enrolled":     t["n_enrolled"],
            "burn_in":        t["burn_in"],
            "ewoc_mode":      t["ewoc_mode"],
            "dose_before":    t["current_dose"],
            "dose_assigned_next": t["next_dose"],
            "highest_tried":  t["highest_tried"],
            **{f"n1_L{d}": t["n1"][d] for d in range(len(DOSE_LABELS))},
            **{f"y1_L{d}": t["y1"][d] for d in range(len(DOSE_LABELS))},
            **{f"n2_L{d}": t["n2"][d] for d in range(len(DOSE_LABELS))},
            **{f"y2_L{d}": t["y2"][d] for d in range(len(DOSE_LABELS))},
            **{f"post_mean_t1_L{d}": t["pm1"][d] for d in range(len(DOSE_LABELS))},
            **{f"overdose_prob_t1_L{d}": t["od1"][d] for d in range(len(DOSE_LABELS))},
            **{f"post_mean_t2_L{d}": t["pm2"][d] for d in range(len(DOSE_LABELS))},
            **{f"overdose_prob_t2_L{d}": t["od2"][d] for d in range(len(DOSE_LABELS))},
            "ewoc_admissible_doses": ",".join(f"L{d}" for d in t["allowed"]),
            "reason":         t["reason"],
            "p_stop_prob":    t["p_stop_prob"],
        })
    df = pd.DataFrame(rows)
    df.attrs["selected_mtd"] = selected
    df.attrs["selected_mtd_label"] = DOSE_LABELS[selected]
    df.attrs["study_days"] = study_days
    df.attrs["stopped_early"] = stopped_early
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--scenario", default="Acute middle", choices=list(metc.ACUTE_SCENARIOS))
    parser.add_argument("--outdir", type=Path, default=Path("trial_export"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    true_t1 = metc.ACUTE_SCENARIOS[args.scenario]
    true_t2 = metc.TRUE_SUBACUTE
    skel1 = np.asarray(metc.ACUTE_PRIOR, dtype=float)
    skel2 = np.asarray(metc.SUBACUTE_PRIOR, dtype=float)

    base_kw = dict(
        p_surgery=0.80,
        target1=0.20,
        target2=0.33,
        skel1=skel1,
        skel2=skel2,
        sigma=1.0,
        start_level=2,
        max_n=30,
        cohort_size=3,
        accrual_per_month=0.75,
        incl_to_rt=21,
        rt_dur=14,
        rt_to_surg=42,
        tox1_win=56,
        tox2_win=30,
        max_step=1,
        gh_n=61,
        enforce_guardrail=True,
        restrict_final_to_tried=True,
        ewoc_on=True,
        ewoc_alpha=0.35,
        ewoc_application=sim.EWOC_APP_BOTH,
        n_safe_d1=6,
        p_stop=1.0,
        require_full_tox1_fu_before_escalation=True,
        collect_trace=True,
    )

    rng = np.random.default_rng(args.seed)
    selected, patients, study_days, trace, stopped_early = sim.run_tite_crm(
        true_t1=true_t1, true_t2=true_t2, burn_in=True, rng=rng, **base_kw,
    )

    gen_df = build_generating_file(patients)
    crm_df = build_crm_output(trace, selected, study_days, stopped_early)

    gen_path = args.outdir / "generating_file_trial1.csv"
    crm_path = args.outdir / "tite_crm_output_trial1.csv"
    gen_df.to_csv(gen_path, index=False)
    crm_df.to_csv(crm_path, index=False)

    summary = {
        "seed": args.seed,
        "scenario": args.scenario,
        "true_t1_acute": true_t1,
        "true_t2_subacute": true_t2,
        "prior_skeleton_t1": list(map(float, skel1)),
        "prior_skeleton_t2": list(map(float, skel2)),
        "dose_labels": DOSE_LABELS,
        "settings": {k: (v if not isinstance(v, np.ndarray) else list(map(float, v)))
                     for k, v in base_kw.items() if k not in ("skel1", "skel2")},
        "n_patients_total": len(patients),
        "n_patients_new": sum(1 for p in patients if p["arrival"] >= 0),
        "study_days": round(study_days, 2),
        "stopped_early": stopped_early,
        "selected_mtd_level": selected,
        "selected_mtd_label": DOSE_LABELS[selected],
    }
    summary_path = args.outdir / "trial1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {gen_path} ({len(gen_df)} patients)")
    print(f"Wrote {crm_path} ({len(crm_df)} cohort decisions)")
    print(f"Wrote {summary_path}")
    print(f"Selected MTD: L{selected} ({DOSE_LABELS[selected]}), "
          f"study_days={study_days:.1f}, stopped_early={stopped_early}")


if __name__ == "__main__":
    main()
