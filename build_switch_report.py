#!/usr/bin/env python3
"""Render the amendment-decision report: switch to TITE-CRM now, or continue 6+3?

Reads switch_decision/ (written by switch_decision_analysis.py and
switch_decision_tree.py) and emits a self-contained English HTML page for the
trial team and the statistician: what the model recommends given the nine
patients already treated at L1, what it would do at the next decision points,
and how the two designs compare over the 21 patients that remain.

Usage:
    python build_switch_report.py [--indir DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_dlt_report import CSS  # shared visual identity across the report set

SCEN = ["Acute low", "Acute middle", "Acute high", "Acute shallow", "Acute steep"]
LEVELS = [f"L{d}" for d in range(5)]

# Sequential ramp for the ordered dose levels L0..L4 (one hue, light to dark).
# Dose level is a magnitude, not an identity, so it gets a ramp rather than
# categorical hues.  Dark-mode steps are chosen against the dark surface rather
# than flipped automatically.
RAMP_CSS = """
/* Text on each step is picked for >=4.5:1 against that step, not by eye:
   on the light ramp only the darkest step takes white; on the dark ramp only
   the two brightest steps take dark ink. */
:root{ --d0:#D6E9EA; --d1:#A3CFD3; --d2:#68B0B7; --d3:#2C8B95; --d4:#0D626D;
       --d0t:var(--ink); --d1t:var(--ink); --d2t:#06282C; --d3t:#04191C; --d4t:#FFFFFF; }
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --d0:#26383F; --d1:#2F5A63; --d2:#3B8089; --d3:#4BA6B0; --d4:#6CCBD3;
  --d0t:var(--ink); --d1t:var(--ink); --d2t:#EAF3F5; --d3t:#04191C; --d4t:#04191C; } }
:root[data-theme="dark"]{
  --d0:#26383F; --d1:#2F5A63; --d2:#3B8089; --d3:#4BA6B0; --d4:#6CCBD3;
  --d0t:var(--ink); --d1t:var(--ink); --d2t:#EAF3F5; --d3t:#04191C; --d4t:#04191C; }
.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;
  vertical-align:-1px;margin-right:5px}
.keyrow{display:flex;flex-wrap:wrap;gap:6px 20px;margin:14px 0 0;
  font-family:var(--mono);font-size:11.5px;color:var(--muted)}
"""

DESIGN_COLOR = {"63": "var(--slate)", "CRM": "var(--accent)"}
DESIGN_NAME = {"63": "Continue 6+3", "CRM": "Switch to TITE-CRM"}


def fmt(x, d=1):
    return f"{x:.{d}f}"


# ── figure 1: the dose-toxicity posterior, before and after the three extra patients ──

def build_posterior_svg(state: pd.DataFrame) -> str:
    W, H = 720, 400
    ml, mr, mt, mb = 60, 150, 24, 74
    pw, ph = W - ml - mr, H - mt - mb
    xs = [ml + pw * i / 4 for i in range(5)]
    vmax = 0.55

    def py(v):
        return mt + ph - (float(v) / vmax) * ph

    rows = [("6 patients, 1 DLT (July state)", "var(--slate)", "After 6 patients"),
            ("9 patients, 1 DLT (current state)", "var(--accent)", "After 9 patients")]

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
           f'aria-label="Posterior acute toxicity by dose level, before and after '
           f'the three additional patients at L1">']
    for g in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        y = py(g)
        cls = "axis" if g == 0.2 else "grid"
        dash = ' stroke-dasharray="4 4"' if g == 0.2 else ""
        out.append(f'<line class="{cls}" x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" '
                   f'y2="{y:.1f}"{dash}/>')
        out.append(f'<text class="tick" x="{ml-10}" y="{y+4:.1f}" '
                   f'text-anchor="end">{g:.1f}</text>')
    out.append(f'<text class="hint" x="{ml+pw+8}" y="{py(0.20)+4:.1f}">TARGET 0.20</text>')

    for label, col, short in rows:
        r = state[state["state"] == label].iloc[0]
        pts = [(xs[d], py(r[f"post_mean_L{d}"])) for d in range(5)]
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                        for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2" '
                   f'stroke-linejoin="round"/>')
        for d, (x, y) in enumerate(pts):
            rec = (col == "var(--accent)" and d == 2)
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5 if rec else 4}" '
                       f'fill="{col}" stroke="var(--surface)" stroke-width="2"/>')
        vx, vy = pts[2]
        # inline style, not a fill attribute: the shared .val/.lbl rules outrank
        # presentation attributes and would repaint these muted grey / ink
        out.append(f'<text class="val" x="{vx:.1f}" y="{vy-12:.1f}" text-anchor="middle" '
                   f'style="fill:{col}">{r["post_mean_L2"]:.3f}</text>')
        out.append(f'<text class="lbl" x="{pts[-1][0]+12:.1f}" y="{pts[-1][1]+4:.1f}" '
                   f'style="fill:{col}">{short}</text>')

    labs = ["5x4 Gy", "5x5 Gy", "5x6 Gy", "5x7 Gy", "5x8 Gy"]
    for d, x in enumerate(xs):
        hi = " rec" if d == 2 else ""
        out.append(f'<text class="cat{hi}" x="{x:.1f}" y="{mt+ph+20}" '
                   f'text-anchor="middle">L{d}</text>')
        out.append(f'<text class="cat{hi}" x="{x:.1f}" y="{mt+ph+34}" '
                   f'text-anchor="middle">{labs[d]}</text>')
    out.append(f'<text class="axlbl" transform="rotate(-90 14 {mt+ph/2:.1f})" x="14" '
               f'y="{mt+ph/2:.1f}" text-anchor="middle">Posterior mean P(acute DLT)</text>')
    out.append(f'<text class="hint" x="{ml}" y="{H-10}">L2 CROSSES FROM 0.255 TO 0.205 '
               f'&#8212; FROM ABOVE TARGET TO ON IT</text>')
    out.append("</svg>")
    return "\n".join(out)


# ── figure 2: the headline risk — ending at or below the dose already given ──

def build_stuck_svg(df: pd.DataFrame) -> str:
    W, H = 720, 372
    ml, mr, mt, mb = 56, 20, 24, 86
    pw, ph = W - ml - mr, H - mt - mb
    slot = pw / len(SCEN)
    bw = min(46.0, slot * 0.30)
    vmax = 60.0

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
           f'aria-label="Probability of ending at L1 or lower, by design and scenario">']
    for g in range(0, int(vmax) + 1, 20):
        y = mt + ph - (g / vmax) * ph
        out.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{ml-10}" y="{y+4:.1f}" text-anchor="end">{g}%</text>')
    out.append(f'<line class="axis" x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}"/>')

    for i, sc in enumerate(SCEN):
        cx = ml + slot * (i + 0.5)
        for j, arm in enumerate(("63", "CRM")):
            v = float(df[(df.scenario == sc) & (df.arm == arm)]["stuck_at_or_below_L1_pct"].iloc[0])
            h = (v / vmax) * ph
            # 2px surface gap between the two adjacent bars
            x = cx - bw - 1 + j * (bw + 2)
            y = mt + ph - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                       f'height="{max(h,1.5):.1f}" rx="4" fill="{DESIGN_COLOR[arm]}"/>')
            out.append(f'<text class="val" x="{x+bw/2:.1f}" y="{y-6:.1f}" '
                       f'text-anchor="middle">{fmt(v)}</text>')
        out.append(f'<text class="cat" x="{cx:.1f}" y="{mt+ph+20}" '
                   f'text-anchor="middle">{sc.replace("Acute ", "")}</text>')

    ly = H - 40
    for j, arm in enumerate(("63", "CRM")):
        x0 = ml + j * 250
        out.append(f'<rect x="{x0}" y="{ly-9}" width="11" height="11" rx="2" '
                   f'fill="{DESIGN_COLOR[arm]}"/>')
        out.append(f'<text class="leg" x="{x0+16}" y="{ly}">{DESIGN_NAME[arm]}</text>')
    out.append(f'<text class="axlbl" x="{ml+pw/2:.1f}" y="{H-10}" text-anchor="middle">'
               f'Final MTD at L1 or lower &#8212; the escalation gained nothing</text>')
    out.append("</svg>")
    return "\n".join(out)


# ── figure 3: where each design actually lands ────────────────────────────────

def build_dist_svg(df: pd.DataFrame) -> str:
    """Stacked share of the final selected dose, two rows per scenario.

    The scenario caption sits on its own line above each pair rather than in a
    left gutter, which is too narrow to hold it without colliding with the bars.
    Segment values are set with an inline style, not a fill attribute: the shared
    stylesheet's `.val{fill:var(--muted)}` rule outranks a presentation
    attribute and would otherwise repaint every label muted grey.
    """
    rowh, gap, caph, grouped = 24, 3, 21, 20
    ml, mr, mt = 52, 16, 4
    W = 720
    pw = W - ml - mr
    H = mt + len(SCEN) * (caph + 2 * rowh + gap + grouped) - grouped + 6

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
           f'aria-label="Distribution of the final selected dose, by design and scenario">']
    y = mt
    for sc in SCEN:
        tm = int(df[df.scenario == sc]["true_mtd"].iloc[0])
        tc = int(df[df.scenario == sc]["true_closest"].iloc[0])
        note = f"MTD L{tm}" + (f", closest L{tc}" if tc != tm else "")
        out.append(f'<text class="lbl" x="0" y="{y+13}">{sc.replace("Acute ", "").title()}</text>')
        out.append(f'<text class="hint" x="{ml + 62}" y="{y+13}">{note.upper()}</text>')
        y += caph
        for arm in ("63", "CRM"):
            r = df[(df.scenario == sc) & (df.arm == arm)].iloc[0]
            x = ml
            for d in range(5):
                v = float(r[f"sel_L{d}_pct"])
                w = pw * v / 100.0
                if w > 0.4:
                    out.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w-2,0.8):.1f}" '
                               f'height="{rowh-5}" rx="3" fill="var(--d{d})"/>')
                if v >= 9:
                    out.append(f'<text class="val" x="{x+(w-2)/2:.1f}" y="{y+13}" '
                               f'text-anchor="middle" style="fill:var(--d{d}t)">'
                               f'{fmt(v,0)}</text>')
                x += w
            out.append(f'<text class="hint" x="{ml-9}" y="{y+13}" text-anchor="end">'
                       f'{"6+3" if arm == "63" else "CRM"}</text>')
            y += rowh + (gap if arm == "63" else 0)
        y += grouped
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path("switch_decision"))
    ap.add_argument("--out", type=Path,
                    default=Path("switch_decision/switch_to_titecrm.html"))
    args = ap.parse_args()

    df = pd.read_csv(args.indir / "switch_decision_summary.csv")
    state = pd.read_csv(args.indir / "posterior_state.csv")
    tree = pd.read_csv(args.indir / "decision_tree_attr1.csv")
    meta = json.loads((args.indir / "switch_decision_meta.json").read_text(encoding="utf-8"))

    def cell(sc, arm, col, d=1):
        return fmt(float(df[(df.scenario == sc) & (df.arm == arm)][col].iloc[0]), d)

    # ── accuracy table, both definitions ─────────────────────────────────────
    acc_rows = []
    for arm in ("63", "CRM", "CRM-HB"):
        for defn, col in (("Highest dose &le; 0.20", "correct_pct"),
                          ("Closest to 0.20", "correct_closest_pct")):
            cells = "".join(f"<td>{cell(s, arm, col)}</td>" for s in SCEN)
            cls = ' class="rec"' if (arm == "CRM" and col == "correct_closest_pct") else ""
            acc_rows.append(f'<tr{cls}><td>{DESIGN_NAME.get(arm, "TITE-CRM, R dose rule")}'
                            f' &middot; <span style="color:var(--muted)">{defn}</span></td>'
                            f'{cells}</tr>')
    head = "".join(f"<th>{s.replace('Acute ', '')}</th>" for s in SCEN)
    acc_table = (f'<table><caption>Correct dose selected (%), under both definitions'
                 f'</caption><thead><tr><th>Design &middot; definition of correct</th>'
                 f'{head}</tr></thead><tbody>{"".join(acc_rows)}</tbody></table>')

    # ── safety table ─────────────────────────────────────────────────────────
    saf_rows = []
    for arm in ("63", "CRM"):
        for lab, col, dd in (("Mean acute DLTs in the 21 new patients", "mean_acute_dlt_new", 1),
                             ("True acute toxicity of the selected dose", "mean_true_tox_selected", 3),
                             ("Selected dose has true toxicity &gt; 0.30 (%)", "unsafe_pct", 1)):
            cells = "".join(f"<td>{cell(s, arm, col, dd)}</td>" for s in SCEN)
            saf_rows.append(f'<tr><td>{DESIGN_NAME[arm]} &middot; '
                            f'<span style="color:var(--muted)">{lab}</span></td>{cells}</tr>')
    saf_table = (f'<table><caption>What each design costs in toxicity</caption>'
                 f'<thead><tr><th>Design &middot; measure</th>{head}</tr></thead>'
                 f'<tbody>{"".join(saf_rows)}</tbody></table>')

    # ── decision-tree table ──────────────────────────────────────────────────
    t1 = tree[tree["step"] == 1]
    tree_rows = []
    for _, r in t1.iterrows():
        nxt = int(r["next_argmin"])
        move = ("stay at L2" if nxt == 2 else
                f"down to L{nxt}" if nxt < 2 else f"up to L{nxt}")
        cls = ' class="rec"' if nxt < 2 else ""
        tree_rows.append(
            f'<tr{cls}><td>{int(r["cohort_dlts"])} of 3</td>'
            f'<td>{r["pm_L1"]:.3f}</td><td>{r["pm_L2"]:.3f}</td><td>{r["pm_L3"]:.3f}</td>'
            f'<td>L{nxt}</td><td style="text-align:left">{move}</td></tr>')
    tree_table = (f'<table><caption>What the model does after the first cohort of three at L2'
                  f'</caption><thead><tr><th>Acute DLTs at L2</th><th>P(DLT) L1</th>'
                  f'<th>P(DLT) L2</th><th>P(DLT) L3</th><th>Next dose</th>'
                  f'<th style="text-align:left">Move</th></tr></thead>'
                  f'<tbody>{"".join(tree_rows)}</tbody></table>')

    # ── state table ──────────────────────────────────────────────────────────
    st_rows = []
    for _, r in state.iterrows():
        cls = ' class="rec"' if "current state" in r["state"] else ""
        st_rows.append(
            f'<tr{cls}><td>{r["state"]}</td>'
            + "".join(f'<td>{r[f"post_mean_L{d}"]:.3f}</td>' for d in range(5))
            + f'<td>L{int(r["rule_argmin"])}</td><td>L{int(r["rule_highest_below"])}</td></tr>')
    st_table = (f'<table><caption>Posterior acute toxicity and the resulting dose, '
                f'by state of the L1 data</caption><thead><tr><th>Data at L1</th>'
                + "".join(f"<th>L{d}</th>" for d in range(5))
                + f'<th>Closest rule</th><th>Highest-below rule</th></tr></thead>'
                f'<tbody>{"".join(st_rows)}</tbody></table>')

    fig_post = build_posterior_svg(state)
    fig_stuck = build_stuck_svg(df)
    fig_dist = build_dist_svg(df)

    cur = state[state["state"].str.contains("current")].iloc[0]
    jul = state[state["state"].str.contains("July")].iloc[0]
    ramp_key = "".join(
        f'<span><span class="swatch" style="background:var(--d{d})"></span>L{d} '
        f'{meta["dose_labels"][d]}</span>' for d in range(5))

    html = f"""<title>Amending to TITE-CRM</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}{RAMP_CSS}</style>
<div class="wrap">
<header class="masthead">
  <p class="kicker">Amendment decision &middot; MERGE dose escalation</p>
  <h1>Switching to TITE-CRM after nine patients at L1</h1>
  <p class="standfirst">Nine patients have been treated at 5&times;5&nbsp;Gy and one of them
  recorded an acute DLT that could not be excluded as treatment-related. This note asks
  whether it is sound to amend to a TITE-CRM at this point, and whether the design still
  arrives at a defensible dose given that DLT.</p>
  <div class="meta">
    <span>Simulations <b>{meta['n_sim']}</b> per cell</span>
    <span>Patients remaining <b>{meta['n_forward']}</b></span>
    <span>Acute target <b>{meta['target_acute']:.2f}</b></span>
    <span>Configuration <b>no EWOC, no burn-in</b></span>
  </div>
</header>

<section class="step"><div class="col">
  <div class="steph"><span class="num">1</span><h2>The answer, before the detail</h2></div>
  <p class="lede">Amending is sound, and the DLT at L1 is no longer the obstacle it was in
  July. The reason is arithmetic rather than statistical: one event in nine patients is a
  very different observation from one in six.</p>
  <p>The observed rate falls from 0.167 to 0.111, comfortably below the 0.20 target, and
  eight DLT-free patients now pull the whole fitted curve down with it. That is enough to
  move the model's recommendation off L1.</p>
</div>

<div class="callout rec">
  <p class="ct">Recommendation</p>
  <p><strong>Amend, and use the closest-to-target assignment rule.</strong> The model's
  first recommendation is L2 &mdash; the same dose the 6+3 rules already permit &mdash; so
  the amendment does not itself force a change of dose. Over the remaining 21 patients the
  model-based design ends at L1 or lower in 0&ndash;11% of trials against 16&ndash;50% for
  6+3, at a cost of roughly 0.5&ndash;0.9 additional acute DLTs.</p>
</div>
</section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">2</span><h2>What the model says today</h2></div>
  <p>This is not a simulation. It is the posterior fitted to the nine patients actually
  treated, with the prior skeleton from the July protocol and complete acute follow-up.</p>
</div>

<figure>
  <div class="figbox">{fig_post}</div>
  <figcaption><strong>The three additional patients move L2 onto the target.</strong>
  In July the posterior put L2 at {jul['post_mean_L2']:.3f}, above the 0.20 target, and the
  design recommended staying at L1. With nine patients L2 sits at {cur['post_mean_L2']:.3f}
  &mdash; effectively on target &mdash; and the recommendation becomes L2.</figcaption>
</figure>

<div class="tablewrap">{st_table}</div>

<div class="col">
  <p class="pull">The recommendation is L2 whether the contested DLT is counted at full
  weight, at half, or not at all.</p>
  <p>That last point is worth stating plainly to the team. In July the attribution of the
  single contested event decided the dose. It no longer does: with nine patients on the
  level, counting the DLT at attribution 1.0, 0.5 or 0.0 all yield L2 under the
  closest-to-target rule. The adjudication discussion remains relevant to the trial record,
  but it has stopped being load-bearing for the next dose decision.</p>
</div>
</section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">3</span><h2>What the model does next</h2></div>
  <p>A model-based design is easier to accept if the committee can see its behaviour rather
  than be asked to trust it. The table below is exhaustive for the next cohort: every
  possible outcome of three patients at L2, and the dose that follows. It is computed
  directly from the posterior, not sampled.</p>
</div>

<div class="tablewrap">{tree_table}</div>

<div class="col">
  <p>The design moves one level at a time and never skips. One DLT in three at L2 is not
  enough to send it back down; two is. Escalation to L3 requires six DLT-free patients at
  L2. Compared with the 6+3 rules currently in force, which stop and de-escalate on two
  DLTs in six, this is the more conservative response to an isolated event and the less
  conservative response to an accumulation of clean data &mdash; which is the trade the
  amendment is really making.</p>
</div>
</section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">4</span><h2>Two definitions of the right answer</h2></div>
  <p>Before the simulation results, one caveat that changes how they read.</p>
  <p>The running protocol defines the MTD as the highest dose with acute toxicity at or
  below 0.20. A CRM with the closest-to-target rule is built to find the dose
  <em>nearest</em> 0.20. In three of the five scenarios these are the same dose. In
  <strong>Acute high</strong> (L2 = 0.15, L3 = 0.22) and <strong>Acute steep</strong>
  (L2 = 0.08, L3 = 0.24) they are not: the nearest dose is L3, the highest dose under
  target is L2.</p>
  <p>Scoring the design against the protocol definition alone would mark it wrong for
  meeting its own objective, so both are reported throughout. This is not a presentational
  choice &mdash; it is a decision the amendment has to make explicitly, because the
  assignment rule and the MTD definition must be the same rule.</p>
</div>

<div class="tablewrap">{acc_table}</div>

<div class="col">
  <p>Compare the two designs within a definition, not across them. Under the protocol
  definition the model-based design is ahead of 6+3 in four of five scenarios and behind in
  Acute steep, where it is deliberately choosing L3. Under its own definition it is ahead
  everywhere, by wide margins.</p>
  <p>The bottom pair is the R implementation's rule, included here because it is the live
  alternative. It should be read fairly: under the protocol definition it is the
  <em>better</em> estimator in exactly the two scenarios where the definitions diverge
  (Acute high {cell('Acute high','CRM-HB','correct_pct')}% against
  {cell('Acute high','CRM','correct_pct')}%, Acute steep
  {cell('Acute steep','CRM-HB','correct_pct')}% against
  {cell('Acute steep','CRM','correct_pct')}%), which is what a rule that never crosses the
  target should do. What it gives up is everything else: Acute low
  {cell('Acute low','CRM-HB','correct_pct')}%, Acute middle
  {cell('Acute middle','CRM-HB','correct_pct')}%, Acute shallow
  {cell('Acute shallow','CRM-HB','correct_pct')}%.</p>
</div>
</section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">5</span><h2>The risk that has changed hands</h2></div>
  <p>In July the concern was that a model-based design would lock onto L1 and never escalate.
  With 21 patients left and nine required per level, that risk now belongs to the design
  currently in the protocol.</p>
</div>

<figure>
  <div class="figbox">{fig_stuck}</div>
  <figcaption><strong>Probability the trial ends at L1 or lower.</strong> 6+3 falls back to
  the dose already given in {cell('Acute low','63','stuck_at_or_below_L1_pct')}&ndash;{cell('Acute high','63','stuck_at_or_below_L1_pct')}%
  of trials; the model-based design in {cell('Acute low','CRM','stuck_at_or_below_L1_pct')}&ndash;{cell('Acute shallow','CRM','stuck_at_or_below_L1_pct')}%.
  In Acute high, where the true toxicity at L2 is only 0.15, the chance of seeing two or
  more DLTs among nine patients is about 40% &mdash; enough to stop 6+3 and send it back to
  L1 half the time.</figcaption>
</figure>

<figure>
  <div class="figbox">{fig_dist}<div class="keyrow">{ramp_key}</div></div>
  <figcaption><strong>Where each design finishes.</strong> Each bar is the distribution of
  the final selected dose over {meta['n_sim']} simulated continuations. The 6+3 rows are
  concentrated on L1 and L2; the model-based rows shift a substantial share to L3, and in
  Acute low reach L4 at all.</figcaption>
</figure>
</section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">6</span><h2>What it costs</h2></div>
  <p>The model-based design treats patients at higher doses, so it produces more toxicity.
  The question is how much, and whether it ever recommends something unsafe.</p>
</div>

<div class="tablewrap">{saf_table}</div>

<div class="col">
  <p>The additional burden is roughly half to one acute DLT across the 21 remaining
  patients. The dose finally recommended carries a true acute toxicity of
  0.10&ndash;0.17 on average, against a 0.20 target. A dose with true toxicity above 0.30 is
  selected in at most {cell('Acute steep','CRM','unsafe_pct')}% of trials, and only in
  Acute steep, where L3 sits at 0.24 and the fourth level at 0.34. There is no scenario in
  which the amended design runs away from the data.</p>
  <p>One asymmetry to note in the comparison: 6+3 stops early when it stops for toxicity, so
  it uses {cell('Acute high','63','mean_new_patients')}&ndash;{cell('Acute low','63','mean_new_patients')}
  of the 21 remaining patients on average, while the model-based design always uses all 21.
  A difference of one to two patients &mdash; not material to the comparison, but it is
  there.</p>
</div>
</section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">7</span><h2>The one decision still open</h2></div>
  <p>The assignment rule matters more than anything else in this note, and the two candidate
  rules disagree about what to do today.</p>
  <p>Under the highest-dose-below-target rule used in the R implementation, the
  recommendation given the current data is <strong>L1, not L2</strong>. Amending under that
  rule would mean stepping back from the dose the 6+3 rules already permit. That rule also
  never reaches L4 in any of the five scenarios, and in Acute shallow it ends at L1 or lower
  in {cell('Acute shallow','CRM-HB','stuck_at_or_below_L1_pct')}% of trials.</p>
  <p>This is the same divergence documented in the earlier simulator-comparison note, now
  with a concrete consequence attached. Whichever rule is adopted, the amendment must state
  it together with the matching definition of the MTD.</p>
</div>

<div class="callout">
  <p class="ct">To be specified in the amendment</p>
  <p>The assignment rule (closest to target, or highest below target); the matching MTD
  definition; the one-level step limit; the requirement that no dose above the highest yet
  tried may be assigned; how the contested L1 DLT enters the likelihood; and whether the
  subacute endpoint constrains dose assignment. In the configuration analysed here the
  subacute endpoint is modelled and reported but does not drive assignment, which reflects
  the current code and should be confirmed rather than assumed.</p>
</div>
</section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">8</span><h2>Limits of this analysis</h2></div>
  <ul>
    <li>The nine treated patients are represented as nine observations at L1 with one acute
    DLT and complete follow-up. Their surgery status is drawn from the 0.80 rate used
    elsewhere in the simulator rather than taken from the record.</li>
    <li>No subacute DLTs are assumed among them, per the trial team.</li>
    <li>The 6+3 comparator restarts its level evaluation at L2 with 21 patients. It carries
    no memory of the L1 cohort beyond the fact that L1 was passed, which is what the
    protocol rules actually use.</li>
    <li>Scenario probabilities are the five acute scenarios from the July report, unchanged.
    They are assumptions, not estimates.</li>
    <li>The differences between this simulator and the R implementation described in the
    accompanying comparison note are unresolved. The numbers here are internally consistent;
    they are not yet reconciled with Sama's.</li>
  </ul>
</div>
</section>

<footer><div class="col">
  <p>Simulations: {meta['n_sim']} per design and scenario, seed {meta['seed']}.
  Acute target {meta['target_acute']:.2f}, subacute target {meta['target_subacute']:.2f},
  prior scale 1.0, one-level step limit, no EWOC, no burn-in, final MTD restricted to
  tried doses.</p>
  <p>Reproduced by <span style="font-family:var(--mono);font-size:12px">switch_decision_analysis.py</span>
  and <span style="font-family:var(--mono);font-size:12px">switch_decision_tree.py</span>;
  outputs in <span style="font-family:var(--mono);font-size:12px">switch_decision/</span>.</p>
  <p>This document contains aggregated simulation output only. It contains no patient data.</p>
</div></footer>
</div>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
