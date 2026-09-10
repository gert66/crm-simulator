#!/usr/bin/env python3
"""What would the TITE-CRM recommend at the next decision points, exactly?

No simulation: this enumerates the possible observed outcomes of the next
cohorts and reports the dose the model would assign, given the fixed history of
nine patients at L1 with one recorded acute DLT.  It is the "what does the model
do to us next" table the trial team needs in order to judge the amendment.

All follow-up is assumed complete at each decision point (TITE weight 1), which
is the conservative reading: partial weights would make each new observation
count for less, not more.

Usage:
    python switch_decision_tree.py [--attribution 1.0]
"""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

with contextlib.redirect_stderr(io.StringIO()):
    import sim
    import metc_simulation_report as metc

DOSE_LABELS = metc.DOSE_LABELS
N_LEVELS = len(DOSE_LABELS)
TARGET1 = 0.20
TARGET2 = 0.33
SKEL1 = np.asarray(metc.ACUTE_PRIOR, dtype=float)
SKEL2 = np.asarray(metc.SUBACUTE_PRIOR, dtype=float)

N_HIST, Y_HIST = 9, 1


def decide(n1, y1, current_level, highest_tried, dose_rule="argmin"):
    """Dose the model assigns next, with the max_step=1 and guardrail limits."""
    return sim.crm_choose_next(
        1.0, SKEL1, SKEL2, np.asarray(n1, float), np.asarray(y1, float),
        np.zeros(N_LEVELS), np.zeros(N_LEVELS),
        current_level=current_level, target1=TARGET1, target2=TARGET2,
        ewoc_alpha=None, max_step=1, gh_n=61,
        enforce_guardrail=True, highest_tried=highest_tried,
        n_levels=N_LEVELS, dose_rule=dose_rule)


def posterior(n1, y1):
    return sim.crm_posterior_summaries(
        1.0, SKEL1, np.asarray(n1, float), np.asarray(y1, float), TARGET1, gh_n=61)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attribution", type=float, default=1.0,
                    help="attributable fraction of the contested L1 DLT (1.0 = counts fully)")
    ap.add_argument("--outdir", type=Path, default=Path("switch_decision"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    base_n = np.zeros(N_LEVELS); base_y = np.zeros(N_LEVELS)
    base_n[1] = N_HIST
    base_y[1] = Y_HIST * float(args.attribution)

    rows = []

    # Step 0 — the decision on the table today.
    pm, _od = posterior(base_n, base_y)
    rows.append({
        "step": 0, "history": f"{N_HIST} pts at L1, {Y_HIST} DLT",
        "cohort_dose": "", "cohort_dlts": "",
        **{f"pm_L{d}": pm[d] for d in range(N_LEVELS)},
        "next_argmin": decide(base_n, base_y, 1, 1),
        "next_highest_below": decide(base_n, base_y, 1, 1, "highest_below"),
    })

    # Step 1 — a cohort of three at L2, every possible number of acute DLTs.
    for y2 in range(4):
        n1 = base_n.copy(); y1 = base_y.copy()
        n1[2] += 3; y1[2] += y2
        pm, _od = posterior(n1, y1)
        nxt = decide(n1, y1, 2, 2)
        rows.append({
            "step": 1, "history": f"{N_HIST}@L1 (1 DLT) + 3@L2",
            "cohort_dose": "L2", "cohort_dlts": y2,
            **{f"pm_L{d}": pm[d] for d in range(N_LEVELS)},
            "next_argmin": nxt,
            "next_highest_below": decide(n1, y1, 2, 2, "highest_below"),
        })

        # Step 2 — a second cohort of three, at whatever step 1 recommended.
        for y3 in range(4):
            n2 = n1.copy(); y2v = y1.copy()
            n2[nxt] += 3; y2v[nxt] += y3
            pm2, _ = posterior(n2, y2v)
            rows.append({
                "step": 2,
                "history": f"{N_HIST}@L1 (1 DLT) + 3@L2 ({y2} DLT) + 3@L{nxt}",
                "cohort_dose": f"L{nxt}", "cohort_dlts": y3,
                **{f"pm_L{d}": pm2[d] for d in range(N_LEVELS)},
                "next_argmin": decide(n2, y2v, nxt, max(2, nxt)),
                "next_highest_below": decide(n2, y2v, nxt, max(2, nxt), "highest_below"),
            })

    df = pd.DataFrame(rows)
    df["next_argmin_label"] = df["next_argmin"].map(lambda k: f"L{k} ({DOSE_LABELS[k]})")
    df["next_highest_below_label"] = df["next_highest_below"].map(
        lambda k: f"L{k} ({DOSE_LABELS[k]})")
    out = args.outdir / f"decision_tree_attr{args.attribution:g}.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(f"Contested L1 DLT counted at attribution = {args.attribution:g}\n")
    for step in (0, 1, 2):
        sub = df[df["step"] == step]
        cols = (["history", "cohort_dose", "cohort_dlts"]
                + [f"pm_L{d}" for d in range(N_LEVELS)]
                + ["next_argmin_label", "next_highest_below_label"])
        print(f"--- Step {step} ---")
        print(sub[cols].round(3).to_string(index=False))
        print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
