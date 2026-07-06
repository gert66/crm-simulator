#!/usr/bin/env python3
"""Run the METC amendment simulation grid and render an HTML report.

The script uses the existing TITE-CRM engine in ``sim.py`` and applies the
study state discussed for the METC amendment: six fully followed patients at
L1 (5x5 Gy) with no acute/subacute toxicity, then the first new cohort starts
at L2 (5x6 Gy).  Outputs are written to ``metc_outputs/``.
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

# ``sim.py`` is a Streamlit app as well as a function library.  Importing it in
# bare Python emits Streamlit context warnings; suppress them for batch reports.
with contextlib.redirect_stderr(io.StringIO()):
    import sim

DOSE_LABELS = ["5x4 Gy", "5x5 Gy", "5x6 Gy", "5x7 Gy", "5x8 Gy"]
TRUE_SUBACUTE = [0.02, 0.05, 0.10, 0.15, 0.25]
ACUTE_SCENARIOS = {
    "Subacute reference": [0.01, 0.04, 0.12, 0.17, 0.27],
    "Acute low": [0.01, 0.02, 0.06, 0.10, 0.15],
    "Acute middle": [0.01, 0.04, 0.12, 0.17, 0.27],
    "Acute high": [0.01, 0.05, 0.15, 0.22, 0.30],
    "Acute steep": [0.01, 0.02, 0.08, 0.24, 0.34],
    "Acute shallow": [0.10, 0.13, 0.15, 0.18, 0.24],
}
EWOC_SCENARIOS = {
    "EWOC during + final": sim.EWOC_APP_BOTH,
    "EWOC final only": sim.EWOC_APP_FINAL,
    "EWOC off": sim.EWOC_APP_OFF,
}


def safe_true_mtd(true_t1: list[float], target: float) -> int:
    safe = [i for i, p in enumerate(true_t1) if p <= target]
    return max(safe) if safe else 0


def closest_target_dose(true_t1: list[float], target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(true_t1) - float(target))))


def patient_metrics(patients: list[dict[str, Any]], true_mtd: int) -> dict[str, Any]:
    new_patients = [p for p in patients if p["arrival"] >= 0]
    dose_counts = [sum(1 for p in patients if p["dose"] == d) for d in range(len(DOSE_LABELS))]
    new_counts = [sum(1 for p in new_patients if p["dose"] == d) for d in range(len(DOSE_LABELS))]
    return {
        "n_total": len(patients),
        "n_new": len(new_patients),
        "n_acute_tox": sum(bool(p["has_tox1"]) for p in patients),
        "n_subacute_tox": sum(bool(p["has_tox2"]) for p in patients),
        "any_acute_tox": any(bool(p["has_tox1"]) for p in patients),
        "any_subacute_tox": any(bool(p["has_tox2"]) for p in patients),
        "n_above_true_mtd": sum(1 for p in patients if p["dose"] > true_mtd),
        "pct_above_true_mtd": 100.0 * sum(1 for p in patients if p["dose"] > true_mtd) / max(1, len(patients)),
        **{f"n_L{d}": dose_counts[d] for d in range(len(DOSE_LABELS))},
        **{f"n_new_L{d}": new_counts[d] for d in range(len(DOSE_LABELS))},
    }


def run_grid(n_sim: int, seed: int, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target1 = 0.20
    target2 = 0.33
    skel1 = sim.dfcrm_getprior(0.05, 0.15, 3, len(DOSE_LABELS), model="empiric")
    skel2 = sim.dfcrm_getprior(0.05, 0.25, 4, len(DOSE_LABELS), model="empiric")
    base_kw = dict(
        p_surgery=0.80,
        target1=target1,
        target2=target2,
        skel1=skel1,
        skel2=skel2,
        sigma=1.0,
        start_level=2,
        max_n=30,
        cohort_size=3,
        accrual_per_month=0.75,  # one patient per four weeks
        incl_to_rt=21,
        rt_dur=14,
        rt_to_surg=42,
        tox1_win=56,
        tox2_win=30,
        max_step=1,
        gh_n=41,
        enforce_guardrail=True,
        restrict_final_to_tried=True,
        ewoc_on=True,
        ewoc_alpha=0.35,
        n_safe_d1=6,
        p_stop=1.0,
        require_full_tox1_fu_before_escalation=True,
        collect_trace=False,
    )
    rows = []
    detail_rows = []
    rng_master = np.random.default_rng(seed)
    for scenario_name, true_t1 in ACUTE_SCENARIOS.items():
        true_mtd = safe_true_mtd(true_t1, target1)
        closest = closest_target_dose(true_t1, target1)
        for ewoc_name, ewoc_app in EWOC_SCENARIOS.items():
            for burn_in in [True, False]:
                selected = []
                final_without_ewoc = []
                for i in range(n_sim):
                    rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
                    sel, patients, study_days, _trace, stopped = sim.run_tite_crm(
                        true_t1=true_t1,
                        true_t2=TRUE_SUBACUTE,
                        ewoc_application=ewoc_app,
                        burn_in=burn_in,
                        rng=rng,
                        **base_kw,
                    )
                    selected.append(sel)
                    # Counterfactual final selector without EWOC, for final-EWOC downshift metric.
                    n1f, y1f, n2f, y2f = sim.tite_weights(
                        patients, max(sim.patient_follow_up_end(p) for p in patients),
                        base_kw["tox1_win"], base_kw["tox2_win"], len(DOSE_LABELS)
                    )
                    no_ewoc_sel = sim.crm_select_mtd(
                        base_kw["sigma"], skel1, skel2, n1f, y1f, n2f, y2f,
                        target1, target2, ewoc_alpha=None, gh_n=base_kw["gh_n"],
                        restrict_to_tried=True,
                    )
                    final_without_ewoc.append(no_ewoc_sel)
                    pm = patient_metrics(patients, true_mtd)
                    detail_rows.append({
                        "scenario": scenario_name,
                        "ewoc": ewoc_name,
                        "burn_in": burn_in,
                        "simulation": i + 1,
                        "true_mtd": true_mtd,
                        "closest_target_dose": closest,
                        "selected_mtd": sel,
                        "selected_without_final_ewoc": no_ewoc_sel,
                        "ewoc_final_downshift": int(sel < no_ewoc_sel and ewoc_app != sim.EWOC_APP_OFF),
                        "additional_duration_weeks": pm["n_new"] * 4.0,
                        "total_duration_weeks": pm["n_total"] * 4.0,
                        "stopped_early": stopped,
                        **pm,
                    })
                arr = np.asarray(selected)
                counts = np.bincount(arr, minlength=len(DOSE_LABELS))
                setting = [r for r in detail_rows if r["scenario"] == scenario_name and r["ewoc"] == ewoc_name and r["burn_in"] == burn_in]
                sdf = pd.DataFrame(setting)
                row = {
                    "scenario": scenario_name,
                    "ewoc": ewoc_name,
                    "burn_in": burn_in,
                    "n_sim": n_sim,
                    "true_mtd": true_mtd,
                    "true_mtd_label": f"L{true_mtd} ({DOSE_LABELS[true_mtd]})",
                    "closest_target_dose": closest,
                    "closest_target_label": f"L{closest} ({DOSE_LABELS[closest]})",
                    "correct_mtd_pct": 100.0 * np.mean(arr == true_mtd),
                    "too_high_mtd_pct": 100.0 * np.mean(arr > true_mtd),
                    "too_low_mtd_pct": 100.0 * np.mean(arr < true_mtd),
                    "mean_selected_mtd": float(arr.mean()),
                    "mean_acute_tox": float(sdf["n_acute_tox"].mean()),
                    "mean_subacute_tox": float(sdf["n_subacute_tox"].mean()),
                    "pct_any_acute_tox": 100.0 * float(sdf["any_acute_tox"].mean()),
                    "pct_any_subacute_tox": 100.0 * float(sdf["any_subacute_tox"].mean()),
                    "mean_n_total": float(sdf["n_total"].mean()),
                    "mean_n_new": float(sdf["n_new"].mean()),
                    "mean_patients_above_true_mtd": float(sdf["n_above_true_mtd"].mean()),
                    "mean_pct_patients_above_true_mtd": float(sdf["pct_above_true_mtd"].mean()),
                    "mean_additional_duration_weeks": float(sdf["additional_duration_weeks"].mean()),
                    "mean_total_duration_weeks": float(sdf["total_duration_weeks"].mean()),
                    "ewoc_final_downshift_pct": 100.0 * float(sdf["ewoc_final_downshift"].mean()),
                }
                for d in range(len(DOSE_LABELS)):
                    row[f"sel_L{d}_pct"] = 100.0 * counts[d] / n_sim
                    row[f"true_t1_L{d}"] = true_t1[d]
                    row[f"true_t2_L{d}"] = TRUE_SUBACUTE[d]
                rows.append(row)
    summary = pd.DataFrame(rows)
    details = pd.DataFrame(detail_rows)
    summary.to_csv(outdir / "metc_summary.csv", index=False)
    details.to_csv(outdir / "metc_simulation_details.csv", index=False)
    settings = {"target1": target1, "target2": target2, "skel1": list(map(float, skel1)), "skel2": list(map(float, skel2)), **base_kw}
    return summary, details, settings


def plot_distributions(summary: pd.DataFrame, outdir: Path) -> list[str]:
    paths = []
    for scenario, sdf in summary.groupby("scenario", sort=False):
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharey=True)
        axes = axes.ravel()
        for ax, (_, row) in zip(axes, sdf.iterrows()):
            vals = [row[f"sel_L{d}_pct"] for d in range(len(DOSE_LABELS))]
            true_mtd = int(row["true_mtd"])
            colors = ["#9aa8b2"] * len(DOSE_LABELS)
            for d in range(len(DOSE_LABELS)):
                if d < true_mtd:
                    colors[d] = "#8ab6d6"
                elif d == true_mtd:
                    colors[d] = "#3ca370"
                else:
                    colors[d] = "#e76f51"
            ax.bar(range(len(DOSE_LABELS)), vals, color=colors, edgecolor="white")
            ax.axvline(true_mtd, color="black", linestyle="--", linewidth=1.2, label="True MTD")
            ax.set_xticks(range(len(DOSE_LABELS)), [f"L{d}\n{DOSE_LABELS[d].split()[0]}" for d in range(len(DOSE_LABELS))])
            ax.set_ylim(0, 100)
            ax.set_title(f"{row['ewoc']} | burn-in {'on' if row['burn_in'] else 'off'}", fontsize=10)
            ax.grid(axis="y", alpha=0.25)
            ax.text(0.02, 0.92, f"True MTD: L{true_mtd}", transform=ax.transAxes, fontsize=9)
        axes[0].set_ylabel("Selected in simulations (%)")
        axes[2].set_ylabel("Selected in simulations (%)")
        axes[4].set_ylabel("Selected in simulations (%)")
        fig.suptitle(f"Final selected dose distribution — {scenario}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fname = f"selected_mtd_{scenario.lower().replace(' ', '_')}.svg"
        fig.savefig(outdir / fname, format="svg", bbox_inches="tight")
        plt.close(fig)
        paths.append(fname)
    return paths


def fmt_table(df: pd.DataFrame, cols: list[str]) -> str:
    out = df[cols].copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: f"{x:.1f}")
    return out.to_html(index=False, escape=False, classes="table")


def render_report(summary: pd.DataFrame, settings: dict[str, Any], plot_paths: list[str], outdir: Path) -> Path:
    key_cols = ["scenario", "ewoc", "burn_in", "true_mtd_label", "correct_mtd_pct", "too_high_mtd_pct", "too_low_mtd_pct", "mean_acute_tox", "mean_subacute_tox", "mean_n_new", "mean_additional_duration_weeks", "ewoc_final_downshift_pct"]
    dist_cols = ["scenario", "ewoc", "burn_in", "true_mtd_label"] + [f"sel_L{d}_pct" for d in range(len(DOSE_LABELS))]
    best = summary.sort_values(["too_high_mtd_pct", "correct_mtd_pct"], ascending=[True, False]).head(6)
    images = "\n".join(f'<section><img src="{html.escape(p)}" alt="{html.escape(p)}"></section>' for p in plot_paths)
    html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>METC amendment CRM simulations</title>
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
<h1>METC amendment CRM simulation report</h1>
<p class='note'>Each simulated trial is initialized with six fully followed patients at L1 (5x5 Gy), zero acute toxicity and zero subacute toxicity. The first new cohort of three patients starts at L2 (5x6 Gy). Maximum sample size is 30 including the six existing patients.</p>
<h2>Design settings</h2>
<ul>
<li>Simulations per combination: {int(summary['n_sim'].iloc[0])}</li>
<li>Scenario grid: 6 toxicity scenarios × 3 EWOC modes × 2 burn-in settings = {len(summary)} combinations.</li>
<li>Acute target: {settings['target1']:.2f}; subacute target: {settings['target2']:.2f}; EWOC alpha: {settings['ewoc_alpha']:.2f}.</li>
<li>Prior skeleton acute: {', '.join(f'{v:.3f}' for v in settings['skel1'])}; subacute: {', '.join(f'{v:.3f}' for v in settings['skel2'])}.</li>
<li>Trial duration is reported as one patient per four weeks: additional duration = new patients × 4 weeks.</li>
<li>True MTD is defined as the highest dose level with true acute toxicity ≤ target acute toxicity.</li>
</ul>
<h2>Top settings by safety first, then MTD accuracy</h2>
{fmt_table(best, key_cols)}
<h2>Main output metrics</h2>
{fmt_table(summary, key_cols)}
<h2>Final selected dose distributions</h2>
<p>Bars show the percentage of simulations selecting each final dose level. Green indicates the scenario-specific true MTD, blue lower selections, and red higher selections. The dashed vertical line marks the true MTD.</p>
{images}
<h2>Selection distribution table</h2>
{fmt_table(summary, dist_cols)}
<h2>Output files</h2>
<ul><li>metc_summary.csv</li><li>metc_simulation_details.csv</li><li>SVG distribution figures in this folder</li></ul>
</body></html>"""
    report_path = outdir / "metc_simulation_report.html"
    report_path.write_text(html_doc, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sim", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--outdir", type=Path, default=Path("metc_outputs"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary, _details, settings = run_grid(args.n_sim, args.seed, args.outdir)
    plots = plot_distributions(summary, args.outdir)
    report = render_report(summary, settings, plots, args.outdir)
    print(f"Wrote {report}")
    print(f"Wrote {args.outdir / 'metc_summary.csv'}")
    print(f"Wrote {args.outdir / 'metc_simulation_details.csv'}")


if __name__ == "__main__":
    main()
