#!/usr/bin/env python3
"""Switch-now analysis: continue 6+3, or amend to TITE-CRM, from the current state.

Current trial state (September 2026)
------------------------------------
Nine patients have been treated at L1 (5x5 Gy).  One of them had a toxicity that
could not be excluded as treatment-related and was therefore recorded as an acute
DLT.  Acute follow-up is complete for all nine; no subacute DLTs were observed.
Under the running 6+3 rules the L1 evaluation is passed (1/9 acute DLTs is within
the phase-2 escalation threshold), so the trial may proceed to L2 (5x6 Gy).

Question
--------
Is it wise to amend to TITE-CRM at this point, given the DLT at L1, and does the
design still reach a sensible dose?

Both arms are given the same forward budget of 21 further patients (30 in total,
including the nine already treated), start their next cohort at L2, and are
evaluated against the same five acute-toxicity scenarios.

Usage:
    python switch_decision_analysis.py [--n-sim 2000] [--seed 20260910]
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

# The state as it stands today.
N_HIST = 9          # patients already treated at L1
N_HIST_DLT = 1      # of which recorded an acute DLT
N_FORWARD = 21      # further patients allowed by the protocol
N_TOTAL = N_HIST + N_FORWARD
NEXT_LEVEL = 2      # 6+3 has cleared L1, so the next cohort goes to L2

# Shared trial-conduct parameters (identical to the METC report configuration).
COMMON = dict(
    p_surgery=0.80,
    accrual_per_month=0.75,
    incl_to_rt=21, rt_dur=14, rt_to_surg=42,
    tox1_win=56, tox2_win=30,
)


def true_mtd_of(true_t1) -> int:
    """Highest dose whose true acute toxicity is at or below target.

    This is the definition the running protocol and the 6+3 rules use.
    """
    safe = [i for i, p in enumerate(true_t1) if p <= TARGET1]
    return max(safe) if safe else 0


def true_closest_of(true_t1) -> int:
    """Dose whose true acute toxicity is closest to target.

    This is the estimand a CRM with the closest-to-target assignment rule is
    actually built to find, and it is NOT always the same dose as true_mtd_of().
    In 'Acute high' (L2=0.15, L3=0.22) and 'Acute steep' (L2=0.08, L3=0.24) the
    two definitions disagree, so scoring the closest-to-target rule against
    true_mtd_of() alone would count the design as wrong for doing exactly what
    it was asked to do.  Both are reported.
    """
    return int(np.argmin(np.abs(np.asarray(true_t1, dtype=float) - TARGET1)))


# ── the two arms ──────────────────────────────────────────────────────────────

def _crm_kwargs(dose_rule: str, n_hist: int, max_n: int, start_level: int) -> dict:
    return dict(
        target1=TARGET1, target2=TARGET2,
        skel1=np.asarray(metc.ACUTE_PRIOR, dtype=float),
        skel2=np.asarray(metc.SUBACUTE_PRIOR, dtype=float),
        sigma=1.0, start_level=start_level, max_n=max_n, cohort_size=3,
        max_step=1, gh_n=61,
        enforce_guardrail=True, restrict_final_to_tried=True,
        ewoc_on=False, ewoc_application=sim.EWOC_APP_OFF, ewoc_alpha=None,
        burn_in=False,
        n_safe_d1=n_hist, n_safe_d1_dlt=N_HIST_DLT,
        hist_weight=1.0, hist_dlt_attribution=1.0,
        dose_rule=dose_rule, escalation_override_n=0, p_stop=1.0,
        require_full_tox1_fu_before_escalation=True, collect_trace=False,
        **COMMON,
    )


def _one_crm(true_t1, kw, seed):
    rng = np.random.default_rng(seed)
    selected, pts, days, _tr, stopped = sim.run_tite_crm(
        true_t1=true_t1, true_t2=metc.TRUE_SUBACUTE, rng=rng, **kw)
    new = [p for p in pts if p["arrival"] >= 0]
    return selected, new, days, stopped


def _one_63(true_t1, max_n, start_level, seed):
    rng = np.random.default_rng(seed)
    selected, pts, days, _nb = sim.run_tite_6plus3(
        true_t1=true_t1, true_t2=metc.TRUE_SUBACUTE,
        start_level=start_level, max_n=max_n, rng=rng, **COMMON)
    return selected, pts, days, False


def _run_arm(args):
    """Worker: one (scenario, arm) cell.  Returns a summary dict."""
    scenario, true_t1, arm, n_sim, seed = args
    true_t1 = np.asarray(true_t1, dtype=float)
    master = np.random.default_rng(seed)

    sels, n_acute, n_subac, days, top, exposure = [], [], [], [], [], []
    for _ in range(n_sim):
        s = int(master.integers(0, 2**32 - 1))
        if arm["kind"] == "crm":
            kw = _crm_kwargs(arm["dose_rule"], arm["n_hist"], arm["max_n"],
                             arm["start_level"])
            sel, new, d, _stop = _one_crm(true_t1, kw, s)
        else:
            sel, new, d, _stop = _one_63(true_t1, arm["max_n"], arm["start_level"], s)
        sels.append(sel)
        n_acute.append(sum(bool(p["has_tox1"]) for p in new))
        n_subac.append(sum(bool(p["has_tox2"]) for p in new))
        days.append(d)
        top.append(max((p["dose"] for p in new), default=-1) == N_LEVELS - 1)
        counts = np.zeros(N_LEVELS)
        for p in new:
            counts[p["dose"]] += 1
        exposure.append(counts)

    sels = np.asarray(sels)
    exposure = np.asarray(exposure)
    tm = true_mtd_of(true_t1)
    tc = true_closest_of(true_t1)
    p_sel = true_t1[sels]                     # true acute toxicity of the chosen dose
    out = {
        "scenario": scenario, "arm": arm["code"], "arm_label": arm["label"],
        "design": arm["design"], "n_sim": n_sim,
        "true_mtd": tm, "true_mtd_label": f"L{tm} ({DOSE_LABELS[tm]})",
        "true_closest": tc, "true_closest_label": f"L{tc} ({DOSE_LABELS[tc]})",
        "correct_pct": 100.0 * float(np.mean(sels == tm)),
        "too_high_pct": 100.0 * float(np.mean(sels > tm)),
        "too_low_pct": 100.0 * float(np.mean(sels < tm)),
        "correct_closest_pct": 100.0 * float(np.mean(sels == tc)),
        # Clinically framed: how toxic is the dose we end up recommending?
        "mean_true_tox_selected": float(np.mean(p_sel)),
        "unsafe_pct": 100.0 * float(np.mean(p_sel > 0.30)),
        "stuck_at_or_below_L1_pct": 100.0 * float(np.mean(sels <= 1)),
        "mean_acute_dlt_new": float(np.mean(n_acute)),
        "mean_subacute_dlt_new": float(np.mean(n_subac)),
        "mean_new_patients": float(np.mean(exposure.sum(axis=1))),
        "ever_reached_top_pct": 100.0 * float(np.mean(top)),
        "mean_study_days": float(np.mean(days)),
    }
    out.update({f"sel_L{d}_pct": 100.0 * float(np.mean(sels == d)) for d in range(N_LEVELS)})
    out.update({f"exposure_L{d}": float(np.mean(exposure[:, d])) for d in range(N_LEVELS)})
    return out


ARMS = [
    dict(code="63",        design="6+3",      kind="63",
         label="Continue with 6+3 (next cohort at L2)",
         max_n=N_FORWARD, start_level=NEXT_LEVEL),
    dict(code="CRM",       design="TITE-CRM", kind="crm", dose_rule="argmin",
         label="Switch to TITE-CRM, closest-to-target rule",
         n_hist=N_HIST, max_n=N_TOTAL, start_level=NEXT_LEVEL),
    dict(code="CRM-HB",    design="TITE-CRM", kind="crm", dose_rule="highest_below",
         label="Switch to TITE-CRM, highest-dose-below-target rule",
         n_hist=N_HIST, max_n=N_TOTAL, start_level=1),
    dict(code="CRM-OLD6",  design="TITE-CRM", kind="crm", dose_rule="argmin",
         label="Reference: switching after 6 patients (1 DLT), as analysed in July",
         n_hist=6, max_n=6 + N_FORWARD, start_level=1),
]


def posterior_state_table() -> pd.DataFrame:
    """Deterministic posterior given the observed L1 data — no simulation involved."""
    skel1 = np.asarray(metc.ACUTE_PRIOR, dtype=float)
    rows = []
    for label, n, y in [("6 patients, 1 DLT (July state)", 6, 1),
                        ("9 patients, 1 DLT (current state)", 9, 1),
                        ("9 patients, 0 DLT (counterfactual)", 9, 0),
                        ("9 patients, 2 DLT (counterfactual)", 9, 2),
                        ("12 patients, 1 DLT (if L1 grew further)", 12, 1)]:
        n1 = np.zeros(N_LEVELS); y1 = np.zeros(N_LEVELS)
        n1[1], y1[1] = n, y
        pm, od = sim.crm_posterior_summaries(1.0, skel1, n1, y1, TARGET1, gh_n=61)
        cand = np.where(pm <= TARGET1)[0]
        rows.append({
            "state": label, "n_L1": n, "y_L1": y,
            **{f"post_mean_L{d}": float(pm[d]) for d in range(N_LEVELS)},
            **{f"p_over_target_L{d}": float(od[d]) for d in range(N_LEVELS)},
            "rule_argmin": int(np.argmin(np.abs(pm - TARGET1))),
            "rule_highest_below": int(cand.max()) if cand.size else 0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--outdir", type=Path, default=Path("switch_decision"))
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    state = posterior_state_table()
    state.to_csv(args.outdir / "posterior_state.csv", index=False)
    print("Posterior on the acute endpoint given the L1 data (no simulation):")
    show = state[["state"] + [f"post_mean_L{d}" for d in range(N_LEVELS)]
                 + ["rule_argmin", "rule_highest_below"]]
    print(show.round(3).to_string(index=False))
    print()

    jobs = []
    for scenario, true_t1 in metc.ACUTE_SCENARIOS.items():
        for arm in ARMS:
            seed = abs(hash((scenario, arm["code"], args.seed))) % (2**31)
            jobs.append((scenario, list(map(float, true_t1)), arm, args.n_sim, seed))

    workers = args.workers or min(len(jobs), 8)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_run_arm, jobs))

    df = pd.DataFrame(rows)
    order = {a["code"]: i for i, a in enumerate(ARMS)}
    df["_o"] = df["arm"].map(order)
    df = df.sort_values(["scenario", "_o"]).drop(columns="_o")
    csv_path = args.outdir / "switch_decision_summary.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "n_sim": args.n_sim, "seed": args.seed,
        "n_hist": N_HIST, "n_hist_dlt": N_HIST_DLT,
        "n_forward": N_FORWARD, "n_total": N_TOTAL, "next_level": NEXT_LEVEL,
        "target_acute": TARGET1, "target_subacute": TARGET2,
        "dose_labels": DOSE_LABELS,
        "acute_prior": list(map(float, metc.ACUTE_PRIOR)),
        "subacute_prior": list(map(float, metc.SUBACUTE_PRIOR)),
        "scenarios": {k: list(map(float, v)) for k, v in metc.ACUTE_SCENARIOS.items()},
        "true_subacute": list(map(float, metc.TRUE_SUBACUTE)),
        "common": COMMON,
        "arms": [{k: v for k, v in a.items()} for a in ARMS],
    }
    (args.outdir / "switch_decision_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")

    print("\nTrue MTD per definition")
    dfn = df.drop_duplicates("scenario")[["scenario", "true_mtd_label", "true_closest_label"]]
    print(dfn.to_string(index=False))

    for metric, title in [
            ("correct_pct", "Selects the highest dose with true tox <= 0.20 (%)"),
            ("correct_closest_pct", "Selects the dose with true tox closest to 0.20 (%)"),
            ("mean_true_tox_selected", "Mean true acute toxicity of the selected dose"),
            ("unsafe_pct", "Selects a dose with true acute toxicity > 0.30 (%)"),
            ("stuck_at_or_below_L1_pct", "Ends at L1 or lower (%)"),
            ("ever_reached_top_pct", "Ever treated at L4 (%)"),
            ("mean_acute_dlt_new", "Mean acute DLTs in the new patients")]:
        print(f"\n{title}")
        piv = df.pivot(index="arm", columns="scenario", values=metric)
        piv = piv.reindex([a["code"] for a in ARMS])
        print(piv.round(1).to_string())


if __name__ == "__main__":
    main()
