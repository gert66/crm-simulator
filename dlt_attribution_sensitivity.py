#!/usr/bin/env python3
"""Sensitivity analysis: impact of counting the contested MERGE-011 event as an
acute DLT in the L1 initialization cohort (6 patients, 0 vs 1 DLT).

Runs TITE-CRM (EWOC off, burn-in off -- the configuration under discussion,
matching Sama's slide comparison) for each of the 5 acute toxicity scenarios,
once with the contested patient counted as a DLT and once without, and reports
the impact on final MTD selection and on-trial dose allocation. The 6+3 design
is included as a static reference (it has no equivalent "fixed pre-history"
mechanism, so it always starts from a freshly simulated 6-patient cohort).

Usage:
    python dlt_attribution_sensitivity.py [--n-sim 500] [--seed 20260812] [--outdir dlt_attribution_sensitivity]
"""
from __future__ import annotations

import argparse
import contextlib
import html
import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with contextlib.redirect_stderr(io.StringIO()):
    import sim
    import metc_simulation_report as metc

DOSE_LABELS = metc.DOSE_LABELS
TRUE_SUBACUTE = metc.TRUE_SUBACUTE
ACUTE_PRIOR = metc.ACUTE_PRIOR
SUBACUTE_PRIOR = metc.SUBACUTE_PRIOR
ACUTE_SCENARIOS = metc.ACUTE_SCENARIOS


def safe_true_mtd(true_t1: list[float], target: float) -> int:
    safe = [i for i, p in enumerate(true_t1) if p <= target]
    return max(safe) if safe else 0


def run_crm(true_t1, true_t2, skel1, skel2, target1, target2, dlt_flag, n_sim, seed):
    base_kw = dict(
        p_surgery=0.80, target1=target1, target2=target2, skel1=skel1, skel2=skel2,
        sigma=1.0, start_level=2, max_n=30, cohort_size=3, accrual_per_month=0.75,
        incl_to_rt=21, rt_dur=14, rt_to_surg=42, tox1_win=56, tox2_win=30,
        max_step=1, gh_n=61, enforce_guardrail=True, restrict_final_to_tried=True,
        ewoc_on=False, ewoc_application=sim.EWOC_APP_OFF, ewoc_alpha=None,
        n_safe_d1=6, n_safe_d1_dlt=(1 if dlt_flag else 0), p_stop=1.0,
        require_full_tox1_fu_before_escalation=True, collect_trace=False,
    )
    rng_master = np.random.default_rng(seed)
    selected, n_new_list, dose_time = [], [], np.zeros(len(DOSE_LABELS))
    acute_tox, subacute_tox, duration = [], [], []
    for _ in range(n_sim):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        sel, patients, study_days, _trace, _stopped = sim.run_tite_crm(
            true_t1=true_t1, true_t2=true_t2, burn_in=False, rng=rng, **base_kw)
        selected.append(sel)
        new_patients = [p for p in patients if p["arrival"] >= 0]
        n_new_list.append(len(new_patients))
        for p in new_patients:
            dose_time[p["dose"]] += 1
        acute_tox.append(sum(bool(p["has_tox1"]) for p in new_patients))
        subacute_tox.append(sum(bool(p["has_tox2"]) for p in new_patients))
        duration.append(len(new_patients) * 4.0)  # weeks, one patient / 4 weeks
    return {
        "selected": np.asarray(selected),
        "dose_time_pct": 100.0 * dose_time / dose_time.sum(),
        "mean_acute_tox": float(np.mean(acute_tox)),
        "mean_subacute_tox": float(np.mean(subacute_tox)),
        "mean_duration_weeks": float(np.mean(duration)),
    }


def run_63(true_t1, true_t2, n_sim, seed):
    rng_master = np.random.default_rng(seed)
    selected, acute_tox, subacute_tox, duration = [], [], [], []
    for _ in range(n_sim):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        sel, patients, study_days, _n_bridge = sim.run_tite_6plus3(
            true_t1=true_t1, p_surgery=0.80, true_t2=true_t2,
            start_level=2, max_n=30, accrual_per_month=0.75,
            incl_to_rt=21, rt_dur=14, rt_to_surg=42, tox1_win=56, tox2_win=30,
            rng=rng,
        )
        selected.append(sel)
        acute_tox.append(sum(bool(p["has_tox1"]) for p in patients))
        subacute_tox.append(sum(bool(p["has_tox2"]) for p in patients))
        duration.append(len(patients) * 4.0)
    return {
        "selected": np.asarray(selected),
        "mean_acute_tox": float(np.mean(acute_tox)),
        "mean_subacute_tox": float(np.mean(subacute_tox)),
        "mean_duration_weeks": float(np.mean(duration)),
    }


def run_all(n_sim: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    target1, target2 = 0.20, 0.33
    skel1 = np.asarray(ACUTE_PRIOR, dtype=float)
    skel2 = np.asarray(SUBACUTE_PRIOR, dtype=float)
    rows, dose_rows = [], []
    for scenario_name, true_t1 in ACUTE_SCENARIOS.items():
        true_mtd = safe_true_mtd(true_t1, target1)
        for dlt_flag in [False, True]:
            res = run_crm(true_t1, TRUE_SUBACUTE, skel1, skel2, target1, target2,
                           dlt_flag, n_sim, seed=hash((scenario_name, dlt_flag, seed)) % (2**31))
            sel = res["selected"]
            rows.append({
                "scenario": scenario_name,
                "design": "TITE-CRM (EWOC off, burn-in off)",
                "dlt_at_init": dlt_flag,
                "n_sim": n_sim,
                "true_mtd": true_mtd,
                "true_mtd_label": f"L{true_mtd} ({DOSE_LABELS[true_mtd]})",
                "correct_mtd_pct": 100.0 * np.mean(sel == true_mtd),
                "too_high_pct": 100.0 * np.mean(sel > true_mtd),
                "too_low_pct": 100.0 * np.mean(sel < true_mtd),
                "mean_acute_tox": res["mean_acute_tox"],
                "mean_subacute_tox": res["mean_subacute_tox"],
                "mean_duration_weeks": res["mean_duration_weeks"],
                **{f"sel_L{d}_pct": 100.0 * np.mean(sel == d) for d in range(len(DOSE_LABELS))},
                **{f"alloc_L{d}_pct": res["dose_time_pct"][d] for d in range(len(DOSE_LABELS))},
            })
        res63 = run_63(true_t1, TRUE_SUBACUTE, n_sim, seed=hash((scenario_name, "63", seed)) % (2**31))
        sel63 = res63["selected"]
        rows.append({
            "scenario": scenario_name,
            "design": "6+3 (reference, no fixed pre-history)",
            "dlt_at_init": None,
            "n_sim": n_sim,
            "true_mtd": true_mtd,
            "true_mtd_label": f"L{true_mtd} ({DOSE_LABELS[true_mtd]})",
            "correct_mtd_pct": 100.0 * np.mean(sel63 == true_mtd),
            "too_high_pct": 100.0 * np.mean(sel63 > true_mtd),
            "too_low_pct": 100.0 * np.mean(sel63 < true_mtd),
            "mean_acute_tox": res63["mean_acute_tox"],
            "mean_subacute_tox": res63["mean_subacute_tox"],
            "mean_duration_weeks": res63["mean_duration_weeks"],
            **{f"sel_L{d}_pct": 100.0 * np.mean(sel63 == d) for d in range(len(DOSE_LABELS))},
            **{f"alloc_L{d}_pct": np.nan for d in range(len(DOSE_LABELS))},
        })
    summary = pd.DataFrame(rows)
    return summary, summary  # details == summary here (aggregate-only analysis)


def plot_comparison(summary: pd.DataFrame, outdir: Path) -> list[str]:
    paths = []
    crm = summary[summary["design"] == "TITE-CRM (EWOC off, burn-in off)"]
    for scenario, sdf in crm.groupby("scenario", sort=False):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        x = np.arange(len(DOSE_LABELS))
        width = 0.35
        row_no  = sdf[sdf["dlt_at_init"] == False].iloc[0]
        row_yes = sdf[sdf["dlt_at_init"] == True].iloc[0]
        vals_no  = [row_no[f"sel_L{d}_pct"] for d in range(len(DOSE_LABELS))]
        vals_yes = [row_yes[f"sel_L{d}_pct"] for d in range(len(DOSE_LABELS))]
        true_mtd = int(row_no["true_mtd"])
        ax.bar(x - width/2, vals_no, width, label="Zonder DLT (0/6)", color="#3ca370")
        ax.bar(x + width/2, vals_yes, width, label="Met DLT (1/6, MERGE-011)", color="#e76f51")
        ax.axvline(true_mtd, color="black", linestyle="--", linewidth=1.2, label="True MTD")
        ax.set_xticks(x, [f"L{d}\n{DOSE_LABELS[d]}" for d in range(len(DOSE_LABELS))], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Finale MTD-selectie (%)")
        ax.set_title(f"Impact van 1 DLT bij initialisatie — {scenario}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fname = f"dlt_impact_{scenario.lower().replace(' ', '_')}.svg"
        fig.savefig(outdir / fname, format="svg", bbox_inches="tight")
        plt.close(fig)
        paths.append(fname)
    return paths


def render_report(summary: pd.DataFrame, plot_paths: list[str], outdir: Path, n_sim: int) -> Path:
    crm = summary[summary["design"] == "TITE-CRM (EWOC off, burn-in off)"]
    cols = ["scenario", "dlt_at_init", "true_mtd_label", "correct_mtd_pct", "too_high_pct",
            "too_low_pct", "mean_acute_tox", "mean_subacute_tox", "mean_duration_weeks"]
    tbl = crm[cols].copy()
    for c in tbl.columns:
        if pd.api.types.is_float_dtype(tbl[c]):
            tbl[c] = tbl[c].map(lambda x: f"{x:.1f}")
    tbl_html = tbl.to_html(index=False, escape=False, classes="table")

    ref63 = summary[summary["design"].str.startswith("6+3")][
        ["scenario", "true_mtd_label", "correct_mtd_pct", "too_high_pct", "too_low_pct",
         "mean_acute_tox", "mean_subacute_tox", "mean_duration_weeks"]].copy()
    for c in ref63.columns:
        if pd.api.types.is_float_dtype(ref63[c]):
            ref63[c] = ref63[c].map(lambda x: f"{x:.1f}")
    ref63_html = ref63.to_html(index=False, escape=False, classes="table")

    images = "\n".join(f'<section><img src="{html.escape(p)}" alt="{html.escape(p)}"></section>' for p in plot_paths)
    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>DLT attribution sensitivity analysis</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
h1, h2 {{ color: #0b5ea8; }}
.table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-bottom: 24px; }}
.table th, .table td {{ border: 1px solid #d6dde5; padding: 6px 8px; text-align: right; }}
.table th:first-child, .table td:first-child, .table td:nth-child(2) {{ text-align: left; }}
.table th {{ background: #e8f1fb; }}
.note {{ background: #f6f8fa; border-left: 4px solid #0b5ea8; padding: 12px 16px; }}
img {{ max-width: 100%; border: 1px solid #d6dde5; margin: 12px 0 28px; }}
</style></head><body>
<h1>Sensitivity analysis: contested acute DLT at trial initialization</h1>
<p class='note'>Compares TITE-CRM (EWOC off, burn-in off) simulated with the L1 initialization
cohort (6 patients) having 0 vs. 1 observed acute DLT — the latter matching the currently reported
SAE for patient MERGE-011, whose causality was assessed as "possible" (not "unrelated"). All other
settings match the design compared against 6+3 in Sama's slides: target acute 0.20, target subacute
0.33, EWOC off, burn-in off, {n_sim} simulations per scenario/arm. The 6+3 design is included purely
as a reference from the same true-toxicity scenarios; it has no equivalent mechanism for a fixed
pre-history, so it always starts from a freshly simulated 6-patient cohort (not necessarily 1 DLT).</p>
<h2>TITE-CRM: with vs. without the contested DLT</h2>
{tbl_html}
<h2>Final MTD selection distribution per scenario</h2>
{images}
<h2>6+3 reference (dynamic initialization, for context only)</h2>
{ref63_html}
</body></html>"""
    report_path = outdir / "dlt_sensitivity_report.html"
    report_path.write_text(html_doc, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sim", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--outdir", type=Path, default=Path("dlt_attribution_sensitivity"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary, details = run_all(args.n_sim, args.seed)
    summary.to_csv(args.outdir / "dlt_sensitivity_summary.csv", index=False)
    plots = plot_comparison(summary, args.outdir)
    report = render_report(summary, plots, args.outdir, args.n_sim)

    print(f"Wrote {report}")
    print(f"Wrote {args.outdir / 'dlt_sensitivity_summary.csv'}")

    crm = summary[summary["design"] == "TITE-CRM (EWOC off, burn-in off)"]
    print("\nQuick view (correct MTD %, with vs without contested DLT):")
    for scenario, sdf in crm.groupby("scenario", sort=False):
        r_no  = sdf[sdf["dlt_at_init"] == False].iloc[0]
        r_yes = sdf[sdf["dlt_at_init"] == True].iloc[0]
        print(f"  {scenario:15s} zonder: {r_no['correct_mtd_pct']:5.1f}%   met: {r_yes['correct_mtd_pct']:5.1f}%")


if __name__ == "__main__":
    main()
