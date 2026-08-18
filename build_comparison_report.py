#!/usr/bin/env python3
"""Render the simulator-comparison report for the trial team and statistician.

Reads dlt_attribution_sensitivity/rule_alignment.csv (produced by
alignment_experiment.py) and reproduction_check.json (produced by
reproduction_check.py) and emits a self-contained English HTML page describing
where the two implementations diverge, which difference accounts for the
observed gap, and what has to be settled before the numbers can be compared.

Usage:
    python build_comparison_report.py [--indir DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_dlt_report import CSS  # shared visual identity across the report set

VARIANTS = [
    ("A onze code (argmin, restrict, sigma 1.0)", "A", "Our implementation",
     "argmin rule, final MTD restricted to tried doses, sigma 1.0"),
    ("B + geen restrict", "B", "+ final MTD unrestricted",
     "the R side searches all dose levels, capped at last treated dose + 1"),
    ("C + Sama's dosisregel", "C", "+ R dose-selection rule",
     "highest dose with posterior mean at or below target, on both endpoints"),
    ("D + sigma sqrt(1.34)", "D", "+ dfcrm default prior scale",
     "sigma sqrt(1.34) instead of 1.0"),
]


def fmt(x, d=1):
    return f"{x:.{d}f}"


def build_ladder_svg(agg: pd.DataFrame) -> str:
    """Average accuracy and the Acute low scenario across the four variants."""
    W, H = 720, 400
    ml, mr, mt, mb = 58, 128, 26, 74
    pw, ph = W - ml - mr, H - mt - mb
    n = len(VARIANTS)
    xs = [ml + pw * (i + 0.5) / n for i in range(n)]
    vmax = 45.0

    def py(v):
        return mt + ph - (float(v) / vmax) * ph

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
           f'aria-label="Accuracy across the four alignment variants">']
    for g in range(0, int(vmax) + 1, 10):
        y = py(g)
        out.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{ml-10}" y="{y+4:.1f}" text-anchor="end">{g}%</text>')
    out.append(f'<line class="axis" x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}"/>')

    series = [("avg", "var(--slate)", "none", "Average, five scenarios"),
              ("low", "var(--accent)", "none", "Acute low")]
    for key, col, dash, lab in series:
        pts = [(xs[i], py(agg[key][i])) for i in range(n)]
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                        for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2.4" '
                   f'stroke-dasharray="{dash}" stroke-linejoin="round"/>')
        for i, (x, y) in enumerate(pts):
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{col}"/>')
            out.append(f'<text class="val" x="{x:.1f}" y="{y-11:.1f}" text-anchor="middle" '
                       f'fill="{col}">{fmt(agg[key][i])}</text>')
        out.append(f'<text class="lbl" x="{pts[-1][0]+12:.1f}" y="{pts[-1][1]+4:.1f}" '
                   f'fill="{col}">{lab}</text>')

    for i, (_, code, short, _d) in enumerate(VARIANTS):
        hi = ' rec' if code == "C" else ''
        out.append(f'<text class="cat{hi}" x="{xs[i]:.1f}" y="{mt+ph+20}" '
                   f'text-anchor="middle">{code}</text>')
        words = short.split(" ")
        mid = (len(words) + 1) // 2
        out.append(f'<text class="cat{hi}" x="{xs[i]:.1f}" y="{mt+ph+34}" '
                   f'text-anchor="middle">{" ".join(words[:mid])}</text>')
        if words[mid:]:
            out.append(f'<text class="cat{hi}" x="{xs[i]:.1f}" y="{mt+ph+46}" '
                       f'text-anchor="middle">{" ".join(words[mid:])}</text>')

    out.append(f'<text class="axlbl" transform="rotate(-90 14 {mt+ph/2:.1f})" x="14" '
               f'y="{mt+ph/2:.1f}" text-anchor="middle">Correct MTD selected</text>')
    out.append("</svg>")
    return "\n".join(out)


def build_repro_svg(rep: dict) -> str:
    """Published distribution against our two rules, Acute low."""
    W, H = 720, 360
    ml, mr, mt, mb = 56, 20, 24, 80
    pw, ph = W - ml - mr, H - mt - mb
    levels = [f"L{d}" for d in range(5)]
    slot = pw / len(levels)
    bw = min(26.0, slot * 0.24)

    series = [("published_R", "var(--slate)", "Reported, R implementation"),
              ("ours_R_rule", "var(--accent)", "Ours, R dose rule"),
              ("ours_own_rule", "var(--line)", "Ours, own rule")]

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
           f'aria-label="Acute low selection distribution, three series">']
    for g in range(0, 61, 20):
        y = mt + ph - (g / 60) * ph
        out.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{ml-10}" y="{y+4:.1f}" text-anchor="end">{g}%</text>')
    out.append(f'<line class="axis" x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}"/>')

    for i, lev in enumerate(levels):
        cx = ml + slot * (i + 0.5)
        for j, (key, col, _lab) in enumerate(series):
            v = float(rep[key][lev])
            h = (v / 60) * ph
            x = cx - 1.5 * bw + j * (bw + 2)
            y = mt + ph - h
            stroke = ' stroke="var(--muted)" stroke-width="0.8"' if key == "ours_own_rule" else ''
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                       f'height="{max(h,1):.1f}" rx="2" fill="{col}"{stroke}/>')
            if v >= 1:
                out.append(f'<text class="val" x="{x+bw/2:.1f}" y="{y-5:.1f}" '
                           f'text-anchor="middle">{fmt(v,0)}</text>')
        hi = ' rec' if lev == "L4" else ''
        out.append(f'<text class="cat{hi}" x="{cx:.1f}" y="{mt+ph+20}" '
                   f'text-anchor="middle">{lev}</text>')

    ly = H - 34
    for j, (_key, col, lab) in enumerate(series):
        x0 = ml + j * 228
        stroke = ' stroke="var(--muted)" stroke-width="0.8"' if j == 2 else ''
        out.append(f'<rect x="{x0}" y="{ly-9}" width="11" height="11" rx="2" fill="{col}"{stroke}/>')
        out.append(f'<text class="leg" x="{x0+16}" y="{ly}">{lab}</text>')
    out.append(f'<text class="axlbl" x="{ml+pw/2:.1f}" y="{H-8}" text-anchor="middle">'
               f'True MTD is L4</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path("dlt_attribution_sensitivity"))
    ap.add_argument("--out", type=Path,
                    default=Path("dlt_attribution_sensitivity/simulator_comparison.html"))
    args = ap.parse_args()

    df = pd.read_csv(args.indir / "rule_alignment.csv")
    rep = json.loads((args.indir / "reproduction_check.json").read_text(encoding="utf-8"))

    agg = {"avg": [], "low": []}
    for label, _code, _s, _d in VARIANTS:
        d = df[df["variant"] == label]
        agg["avg"].append(float(d["correct"].mean()))
        agg["low"].append(float(d[d["scenario"] == "Acute low"]["correct"].iloc[0]))
    risk = [float(df[df["variant"] == v[0]]["too_high"].mean()) for v in VARIANTS]

    scen = list(dict.fromkeys(df["scenario"]))
    rows = []
    for i, (label, code, short, detail) in enumerate(VARIANTS):
        d = df[df["variant"] == label].set_index("scenario")
        cells = "".join(f"<td>{fmt(float(d.loc[s, 'correct']))}</td>" for s in scen)
        cls = ' class="rec"' if code == "C" else ""
        rows.append(f'<tr{cls}><td><b>{code}</b> &middot; {short}</td>{cells}'
                    f'<td><b>{fmt(agg["avg"][i])}</b></td><td>{fmt(risk[i])}</td></tr>')
    head = "".join(f"<th>{s.replace('Acute ', '')}</th>" for s in scen)
    ladder_table = (f'<table><caption>Correct MTD selection, switching one difference at a '
                    f'time (%)</caption><thead><tr><th>Variant</th>{head}'
                    f'<th>Average</th><th>Too high</th></tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table>')

    rrows = []
    for key, lab in [("published_R", "Reported, R implementation"),
                     ("ours_own_rule", "Ours, own rule"),
                     ("ours_R_rule", "Ours, R dose rule")]:
        cells = "".join(f"<td>{fmt(float(rep[key][f'L{d}']))}</td>" for d in range(5))
        cls = ' class="rec"' if key == "ours_R_rule" else ""
        rrows.append(f'<tr{cls}><td>{lab}</td>{cells}</tr>')
    repro_table = (f'<table><caption>Acute low: final dose selected (%)</caption>'
                   f'<thead><tr><th>Series</th>'
                   + "".join(f"<th>L{d}</th>" for d in range(5))
                   + f'</tr></thead><tbody>{"".join(rrows)}</tbody></table>')

    ladder = build_ladder_svg(agg)
    repro = build_repro_svg(rep)

    d_restrict = agg["low"][1] - agg["low"][0]
    d_rule = agg["low"][2] - agg["low"][1]

    html = f"""<title>Two Designs, Not One</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style>
<div class="wrap">
<header class="masthead">
  <p class="kicker">Simulator comparison &middot; MERGE dose escalation</p>
  <h1>Why the two TITE-CRM simulators disagree, and which difference accounts for it</h1>
  <p class="standfirst">The R and Python implementations produce substantially different
  MTD selection rates. This note traces that gap to a single design choice, reproduces the
  published result by adopting it, and lists what still has to be agreed before any
  comparison of accuracy is meaningful.</p>
  <div class="meta">
    <span>Simulations <b>1000</b> per cell</span>
    <span>Acute target <b>0.20</b></span>
    <span>Subacute target <b>0.33</b></span>
    <span>Configuration <b>no EWOC, no burn-in</b></span>
  </div>
</header>

<section class="step"><div class="col">
  <div class="steph"><span class="num">1</span><h2>What this note is</h2></div>
  <p class="lede">Both groups have simulated what was meant to be the same design, and
  reached different conclusions about how often the correct MTD is chosen. Before either
  set of numbers can inform the amendment, we need to know whether the difference comes
  from the code or from the design.</p>
  <p>It comes from the design. The two programs implement genuinely different rules, and one
  of those rules accounts for almost all of the observed gap. Nothing below is a claim that
  either implementation is wrong; the point is that they are not yet answering the same
  question.</p>
  <p>The method throughout is the same: start from our configuration, switch one thing at a
  time towards the R implementation, and measure what changes. Code and outputs are in the
  repository (<span style="font-family:var(--mono);font-size:12px">alignment_experiment.py</span>,
  <span style="font-family:var(--mono);font-size:12px">reproduction_check.py</span>).</p>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">2</span><h2>The dose-selection rule accounts for the gap</h2></div>
  <p>Four variants, each adding one difference to the previous one.</p>
</div>

<figure>
  <div class="figbox">{ladder}</div>
  <figcaption><strong>Accuracy as each difference is adopted.</strong> Removing the
  restriction to tried doses (A&nbsp;&rarr;&nbsp;B) moves the Acute low scenario by
  {fmt(d_restrict)} points and the average barely at all. Adopting the R dose-selection rule
  (B&nbsp;&rarr;&nbsp;C) drops Acute low by {fmt(abs(d_rule))} points, to zero, and takes the
  average down with it. The prior scale (C&nbsp;&rarr;&nbsp;D) changes almost nothing.</figcaption>
</figure>

<div class="tablewrap">{ladder_table}</div>

<div class="col">
  <h3>Why the rule excludes the top dose</h3>
  <p>The acute skeleton is 0.004, 0.021, 0.066, 0.150, 0.266 against a target of 0.20.
  <strong>The prior at L4 already exceeds the target.</strong> Under the rule &ldquo;highest
  dose whose posterior mean is at or below target&rdquo;, L4 therefore sits outside the
  admissible set from the very first decision, and the data would have to pull that estimate
  below 0.20 before it can ever be chosen. With one DLT at L1 raising the whole curve, that
  does not happen within 30 patients.</p>
  <p>We checked whether the subacute endpoint contributes by relaxing its target to 0.99.
  The result is unchanged at 0.0%, so this is the acute rule alone.</p>
  <div class="callout">
    <p class="ct">Note on the interaction with the contested DLT</p>
    <p>This rule and the single contested toxicity at L1 compound each other. The rule makes
    the top dose inadmissible until the estimate falls; the DLT keeps the estimate up. Either
    on its own would be recoverable. Together they close off the top dose for the whole
    trial.</p>
  </div>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">3</span><h2>Reproduction check</h2></div>
  <p>If the rule really is the explanation, then adopting it should reproduce the published
  distribution, not merely move in its direction. It does.</p>
</div>

<figure>
  <div class="figbox">{repro}</div>
  <figcaption><strong>Acute low, where the true MTD is the top dose.</strong> Our own rule
  selects L4 in {fmt(float(rep['ours_own_rule']['L4']))}% of trials. With the R rule that
  becomes {fmt(float(rep['ours_R_rule']['L4']))}%, matching the reported
  {fmt(float(rep['published_R']['L4']))}%, and the rest of the distribution moves close to
  the reported shape as well. The remaining differences are consistent with the other
  design differences listed in section 5.</figcaption>
</figure>

<div class="tablewrap">{repro_table}</div>
</section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">4</span><h2>Errors on our side</h2></div>
  <p class="lede">Three things we got wrong, listed before anything about the R code,
  because two of them affect results we have already circulated.</p>
  <div class="callout">
    <p class="ct">1 &middot; Our subacute prior was wrong</p>
    <p>The R code uses the exact skeletons from the 8 July message, including a subacute
    prior of <strong>0.010</strong> at L0. Ours read 0.012. Ours was the deviating value and
    has been corrected. The effect is negligible &mdash; trials start at L2 and rarely visit
    L0 &mdash; but earlier outputs were generated with the wrong figure.</p>
  </div>
  <div class="callout">
    <p class="ct">2 &middot; With EWOC off, our subacute endpoint does nothing</p>
    <p>In our dose-selection function the subacute overdose probability is computed and then
    never used when EWOC is disabled. The subacute endpoint therefore has no influence on
    dose selection at all in that configuration &mdash; which is the configuration every
    recent analysis of ours has run under. In the R code both endpoints always bind. This is
    a design question, not only a coding one: should the subacute endpoint constrain dose
    selection when EWOC is off?</p>
  </div>
  <div class="callout">
    <p class="ct">3 &middot; Our 6+3 comparison was not like for like</p>
    <p>The R example runs the rule-based design with a sample size of 50 and an expansion
    concept (<span style="font-family:var(--mono);font-size:12px">target.patients.at.mtd = 12</span>)
    that our implementation does not have; we gave ours 30 patients. Our figure of 2.8%
    correct selection against the reported 32.3% should therefore <strong>not</strong> be
    read as contradicting the R result.</p>
  </div>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">5</span><h2>Differences we found but have not quantified</h2></div>
  <p>Each of these is a real difference in what is being simulated. None has been isolated
  the way the dose rule was, so their contributions are unknown.</p>
  <h3>Surgery is certain, and a delay can become a DLT</h3>
  <p>In the R model every patient is operated: the surgery time matrix is initialised at
  <span style="font-family:var(--mono);font-size:12px">operation.Time</span> for everyone.
  Our model draws surgery from a Bernoulli with probability 0.80, and patients without
  surgery contribute no subacute information at all.</p>
  <p>More consequentially, a surgery delayed beyond
  <span style="font-family:var(--mono);font-size:12px">Max.Surgery.Time</span> is
  <strong>recorded as an acute DLT</strong>, reached through a non-DLT toxicity occurring
  with probability 0.15 that we do not model at all. This means the realised acute toxicity
  in the R simulation is higher than the specified
  <span style="font-family:var(--mono);font-size:12px">P.early</span>, and so the true MTD
  derived from <span style="font-family:var(--mono);font-size:12px">P.early</span> is
  strictly not the true MTD of the data-generating process. That affects the interpretation
  of every accuracy figure on both sides, and is the first thing we would want to confirm.</p>
  <h3>The run-in phase does not escalate</h3>
  <p>In the R code the dose is fixed at the starting level throughout the first phase and the
  loop always runs to six patients, because the stopping condition cannot trigger before
  patient six. Ours escalates one level per cohort until the first observed acute DLT, as
  described in July. The two &ldquo;burn-in on/off&rdquo; comparisons therefore do not
  measure the same thing and should not be placed side by side.</p>
  <h3>Accrual</h3>
  <p>The R code enrols a whole cohort at one shared entry time, with the next cohort a fixed
  interval later; ours uses a Poisson process with individual arrivals. Within-cohort TITE
  weights are therefore identical on the R side and not on ours. The time unit of
  <span style="font-family:var(--mono);font-size:12px">time_para</span> is not stated
  anywhere we could find, so we have not tried to compare the rates.</p>
  <h3>Smaller items</h3>
  <ul>
    <li>De-escalation is unrestricted in the R code; ours is capped at one level per cohort.</li>
    <li>The final MTD may be one level above the <em>last treated</em> dose &mdash; not the
    highest ever tried &mdash; so it can be a dose nobody received, and can also exclude a
    higher dose that was tried earlier.</li>
    <li>A Gaussian copula links the two endpoints on the R side (set to independent in the
    example); we do not model any dependence.</li>
    <li>The prior scale is the dfcrm default, sqrt(1.34), against our 1.0. Same empiric
    model; measured effect here is negligible.</li>
    <li>Toxicity is reported as a proportion on the R side and as a count on ours. These
    agree once converted.</li>
  </ul>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">6</span><h2>Two things that look like defects</h2></div>
  <p>Offered for checking rather than asserted &mdash; both are the kind of thing that
  survives testing precisely because it is invisible in the configurations run so far.</p>
  <div class="callout">
    <p class="ct">Undefined argument names</p>
    <p><span style="font-family:var(--mono);font-size:12px">real.data.early</span> and
    <span style="font-family:var(--mono);font-size:12px">real.data.late</span> are passed to
    the trial function, but the parameters are named
    <span style="font-family:var(--mono);font-size:12px">real.dlt.early</span> and
    <span style="font-family:var(--mono);font-size:12px">real.dlt.late</span>, and the names
    passed do not exist anywhere. This does not error today because R evaluates arguments
    lazily and the trial function never uses them.</p>
  </div>
  <div class="callout">
    <p class="ct">The run-in overwrites the fixed history's dose</p>
    <p>The first phase assigns the starting dose to patients one through six regardless of
    what was supplied in
    <span style="font-family:var(--mono);font-size:12px">real.dose</span>. For the MERGE
    configuration the two coincide, so nothing goes wrong in practice. The recorded outcomes
    of those patients are used correctly.</p>
  </div>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">7</span><h2>Questions we would like to settle</h2></div>
  <p>In order of measured or expected effect.</p>
  <ol>
    <li><strong>The dose-selection rule.</strong> With the current skeleton, &ldquo;highest
    dose at or below target&rdquo; excludes L4 from the outset because the prior there is
    already 0.266. Is that intended? This drives most of the difference and affects the whole
    allocation, not only the final choice.</li>
    <li><strong>Does a surgery delay beyond the maximum count as an acute DLT?</strong> If so,
    the realised acute toxicity exceeds <span style="font-family:var(--mono);font-size:12px">P.early</span>
    and the reference MTD needs rethinking.</li>
    <li><strong>Should the subacute endpoint constrain dose selection</strong> when EWOC is
    off? It does in the R code and does not in ours.</li>
    <li><strong>Should every patient be operated,</strong> or should surgery be probabilistic
    as in our model?</li>
    <li><strong>May the final MTD be a dose nobody received?</strong></li>
    <li><strong>Is the prior scale a choice or the package default?</strong></li>
    <li><strong>Should the first phase escalate</strong> until the first DLT, or treat six
    patients at the starting dose?</li>
    <li><strong>Which accrual structure, time unit and 6+3 settings</strong> should the
    amendment use?</li>
  </ol>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">8</span><h2>Conclusion</h2></div>
  <div class="callout rec">
    <p class="ct">Where this leaves the comparison</p>
    <p>The two simulators are not two implementations of one design. They differ on the dose
    rule, the role of the subacute endpoint, whether surgery is certain, what counts as an
    acute DLT, the accrual structure, the run-in, the prior scale and the 6+3 settings.</p>
    <p>Comparing accuracy percentages between them is therefore not yet meaningful, and we
    would not use either set of numbers to argue the other is wrong.</p>
  </div>
  <p>What has been established is where the difference comes from, and that it can be
  reproduced on demand. The natural next step is to agree which rules represent the intended
  MERGE design, align both implementations on those, and re-run. Most of the alignment is
  small: the dose rule, the prior scale and the subacute constraint are each a few lines.</p>
  <p>A patient-level generating file for a single trial is already prepared on our side, and
  now that the data structures on both sides are understood it can be mapped across directly.
  Running one identical trial through both programs would confirm the alignment before any
  larger comparison is attempted.</p>
</div></section>

<footer><div class="col">
  <p>Generated from <span style="font-family:var(--mono);font-size:12px">alignment_experiment.py</span>
  and <span style="font-family:var(--mono);font-size:12px">reproduction_check.py</span>,
  1000 simulations per cell. TITE-CRM without EWOC and without burn-in, starting at L2,
  30 patients including six with fixed history, cohort size 3, one acute DLT in the sixth
  of those. Reported R figures are taken from the circulated slides for the corresponding
  configuration.</p>
  <p>This document contains aggregated simulation output only. No patient data is included.</p>
</div></footer>
</div>
"""
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
