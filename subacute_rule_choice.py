#!/usr/bin/env python3
"""Which rule should combine the acute and subacute endpoints?

Appendix A section 1.3 assigns the lower of the two closest-to-target
recommendations. This script measures what that costs under the revised
subacute specification, and shows why the cost is not a tuning problem but a
property of the model.

Two results, both reported by main():

1. Expressive range. The empiric model is p(d) = skeleton(d)^exp(beta), a single
   parameter that can only scale the whole curve. With a skeleton spanning
   0.25 to 0.29 the largest difference it can ever place between the lowest and
   highest dose is about 4 percentage points, for any beta and any data. The
   model therefore cannot represent a dose-dependent subacute risk at all; it
   can only move all five doses together.

2. Cost of including it in dose assignment. Run both rules on identical seeds
   and compare. Because the revised true subacute probabilities (0.25 to 0.29)
   lie below the 0.33 target at every dose level in every scenario, any dose
   reduction the subacute endpoint causes is a false alarm by construction.

Usage:
    python subacute_rule_choice.py [--n-sim 3000]
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

import subacute_mechanism_analysis as M

NEW = M.SUBACUTE["new"]
OLD = M.SUBACUTE["old"]


def expressive_range(skeleton, n_grid: int = 20001) -> dict:
    """Largest p(top) - p(bottom) the power model can produce, over all beta."""
    sk = np.asarray(skeleton, dtype=float)
    t = np.exp(np.linspace(-4.0, 4.0, n_grid))
    P = sk[None, :] ** t[:, None]
    spread = P[:, -1] - P[:, 0]
    i = int(np.argmax(spread))
    return {"skeleton": [float(x) for x in sk],
            "max_spread": float(spread[i]),
            "at_exp_beta": float(t[i]),
            "curve_at_max": [float(x) for x in P[i]]}


def _job(args):
    scenario, n_sim, seed = args
    t1 = np.asarray(M.ACUTE[scenario], dtype=float)
    t2 = np.asarray(NEW["true"], dtype=float)
    tc = M.true_closest(t1)
    seeds = np.random.default_rng(seed).integers(0, 2**32 - 1, n_sim)

    sels = {}
    for rule in ("argmin", "min_closest"):
        kw = M._kwargs(rule, NEW["prior"], 0)
        out = []
        for sd in seeds:                      # identical seeds across the two rules
            s, _p, _d, _t, _st = sim.run_tite_crm(
                true_t1=t1, true_t2=t2, rng=np.random.default_rng(int(sd)), **kw)
            out.append(s)
        sels[rule] = np.asarray(out)

    a, m = sels["argmin"], sels["min_closest"]
    return {
        "scenario": scenario,
        "true_mtd_level": M.APPENDIX_LEVEL[tc],
        "pcs_acute_only_pct": 100.0 * float(np.mean(a == tc)),
        "pcs_min_rule_pct": 100.0 * float(np.mean(m == tc)),
        "pcs_cost_pts": 100.0 * float(np.mean(a == tc) - np.mean(m == tc)),
        "min_rule_lower_pct": 100.0 * float(np.mean(m < a)),
        "mean_levels_lost": float(np.mean(a - m)),
        "acute_only_ends_at_or_below_lvl0_pct": 100.0 * float(np.mean(a <= 1)),
        "min_rule_ends_at_or_below_lvl0_pct": 100.0 * float(np.mean(m <= 1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sim", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--outdir", type=Path, default=Path("subacute_mechanism"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("1. What the subacute model can express\n")
    ranges = {}
    for name, sk in (("revised (0.25-0.29)", NEW["prior"]),
                     ("previous (0.010-0.250)", OLD["prior"])):
        r = expressive_range(sk)
        ranges[name] = r
        print(f"  {name}")
        print(f"    largest achievable p(top) - p(bottom) = {r['max_spread']:.3f}"
              f"  (at exp(beta) = {r['at_exp_beta']:.3f})")
        print(f"    curve there: {np.round(r['curve_at_max'], 3)}")
    print("\n  The revised skeleton cannot place more than about 4 percentage points")
    print("  between 5x4 Gy and 5x8 Gy, for any beta and any data. A genuine")
    print("  dose-dependent subacute risk is outside the model's range entirely.\n")

    jobs = [(s, args.n_sim, args.seed) for s in M.SCEN]
    with ProcessPoolExecutor(max_workers=5) as ex:
        rows = list(ex.map(_job, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "subacute_rule_choice.csv", index=False)

    print("2. Cost of letting the subacute endpoint assign the dose")
    print("   (identical seeds; true subacute probability is below target at every")
    print("   dose level, so every reduction it causes is a false alarm)\n")
    show = df[["scenario", "true_mtd_level", "pcs_acute_only_pct", "pcs_min_rule_pct",
               "pcs_cost_pts", "min_rule_lower_pct", "mean_levels_lost",
               "acute_only_ends_at_or_below_lvl0_pct",
               "min_rule_ends_at_or_below_lvl0_pct"]]
    print(show.round(1).to_string(index=False))

    (args.outdir / "subacute_rule_choice_meta.json").write_text(json.dumps({
        "n_sim": args.n_sim, "seed": args.seed,
        "expressive_range": ranges,
        "subacute_true": NEW["true"], "subacute_prior": NEW["prior"],
        "target_subacute": M.TARGET2, "target_acute": M.TARGET1,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.outdir / 'subacute_rule_choice.csv'}")


if __name__ == "__main__":
    main()
