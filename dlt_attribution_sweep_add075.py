#!/usr/bin/env python3
"""Extend the attribution sweep with p = 0.75 ("probable") for all three series.

The original grid (0, 0.05, 0.10, 0.25, 0.50, 1.00) omitted the fifth point
needed to map cleanly onto the SAE causality scale (unrelated / unlikely /
possible / probable / definite -> 0 / 0.05 / 0.25 / 0.75 / 1.00). This computes
the 15 missing (scenario, series) cells at p = 0.75 and merges them into the
existing sweep CSV and meta, so every figure and table in the report is
regenerated from one consistent, complete grid.

Usage:
    python dlt_attribution_sweep_add075.py
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from dlt_attribution_sweep import DOSE_LABELS, _cell, metc, sim

NEW_P = 0.75
OUTDIR = Path("dlt_attribution_sensitivity")


def main() -> None:
    csv_path = OUTDIR / "dlt_attribution_sweep.csv"
    meta_path = OUTDIR / "dlt_attribution_sweep_meta.json"
    df = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    seed0 = int(meta["seed"])

    series = [
        ("attribution", "Non-binary DLT (n full, y scaled)", "attribution", 0),
        ("weight",      "Block discount (n and y scaled)",   "weight", 0),
        ("attribution_override", "Non-binary DLT + escalation override",
         "attribution", 3),
    ]

    jobs = []
    for scenario, true_t1 in metc.ACUTE_SCENARIOS.items():
        for skey, slabel, kind, override in series:
            seed = abs(hash((scenario, skey, NEW_P, seed0))) % (2**31)
            jobs.append((scenario, list(map(float, true_t1)), skey, slabel,
                         kind, override, NEW_P, int(meta["n_sim"]), seed))

    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i, row in enumerate(ex.map(_cell, jobs), start=1):
            rows.append(row)
            print(f"[{i:>2}/{len(jobs)}] {row['scenario']:15s} {row['series']:22s} "
                  f"p={row['p']:<5} correct={row['correct_pct']:5.1f} "
                  f"too_high={row['too_high_pct']:5.1f}", flush=True)

    new_df = pd.DataFrame(rows)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["scenario", "series", "p"], keep="last")
    combined.to_csv(csv_path, index=False)

    meta["p_grid"] = sorted(set(meta["p_grid"]) | {NEW_P})
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nWrote {csv_path} ({len(combined)} rows)")
    print(f"Wrote {meta_path} (p_grid = {meta['p_grid']})")


if __name__ == "__main__":
    main()
