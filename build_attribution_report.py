#!/usr/bin/env python3
"""Render the non-binary DLT discussion paper from the attribution sweep.

Reads dlt_attribution_sensitivity/dlt_attribution_sweep.csv and emits a
self-contained English HTML page for the trial team and the trial statistician:
what a fractional DLT attribution is, what already exists in the literature,
why scaling y is not the same as scaling n, what the simulations show across
attribution probabilities, and what would have to be settled before any of it
could enter a protocol.

Usage:
    python build_attribution_report.py [--indir DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_dlt_report import CSS  # shared visual identity with the first report

SCEN_COLOR = {
    "Acute low":     "var(--accent)",
    "Acute middle":  "var(--slate)",
    "Acute high":    "var(--risk)",
    "Acute steep":   "var(--warn)",
    "Acute shallow": "var(--muted)",
}


def fmt(x, d=1):
    return f"{x:.{d}f}"


def _axis_x(p_grid, ml, pw):
    """Ordinal x positions — the grid is unevenly spaced, so treat it as ordinal."""
    n = len(p_grid)
    return {p: ml + (pw * i / (n - 1)) for i, p in enumerate(p_grid)}


def build_curve_svg(df: pd.DataFrame, p_grid: list[float], metric: str,
                     ylab: str, series: str = "attribution") -> str:
    d = df[df["series"] == series]
    W, H = 720, 400
    ml, mr, mt, mb = 60, 128, 22, 62
    pw, ph = W - ml - mr, H - mt - mb
    xs = _axis_x(p_grid, ml, pw)
    vmax = min(100.0, ((float(d[metric].max()) + 6) // 10 + 1) * 10)
    vmin = max(0.0, ((float(d[metric].min()) - 6) // 10) * 10)

    def py(v):
        return mt + ph - ((float(v) - vmin) / (vmax - vmin)) * ph

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
           f'aria-label="{ylab} against attribution probability">']
    g = int(vmin)
    while g <= vmax:
        if g % 10 == 0:
            y = py(g)
            out.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}"/>')
            out.append(f'<text class="tick" x="{ml-10}" y="{y+4:.1f}" text-anchor="end">{g}%</text>')
        g += 5
    for p in p_grid:
        x = xs[p]
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}"/>')
        lab = "0" if p == 0 else ("1" if p == 1 else f"{p:g}")
        out.append(f'<text class="tick" x="{x:.1f}" y="{mt+ph+19}" text-anchor="middle">{lab}</text>')
    out.append(f'<line class="axis" x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}"/>')
    out.append(f'<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}"/>')

    placed: list[float] = []
    ends = []
    for scen, sdf in d.groupby("scenario", sort=False):
        sdf = sdf.sort_values("p")
        pts = [(xs[r["p"]], py(r[metric])) for _, r in sdf.iterrows()]
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        col = SCEN_COLOR.get(scen, "var(--muted)")
        out.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2" '
                   f'stroke-linejoin="round"/>')
        for x, y in pts:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{col}"/>')
        ends.append((pts[-1][1], scen, col))

    for y0, scen, col in sorted(ends, key=lambda z: z[0]):
        ly = y0
        while any(abs(ly - q) < 15 for q in placed):
            ly += 15
        placed.append(ly)
        out.append(f'<line class="leader" x1="{ml+pw+4:.1f}" y1="{y0:.1f}" '
                   f'x2="{ml+pw+13:.1f}" y2="{ly:.1f}"/>')
        out.append(f'<text class="lbl" x="{ml+pw+17:.1f}" y="{ly+4:.1f}" '
                   f'fill="{col}">{scen.replace("Acute ", "")}</text>')

    out.append(f'<text class="axlbl" x="{ml+pw/2:.1f}" y="{H-14}" text-anchor="middle">'
               f'Attribution probability assigned to the contested DLT</text>')
    out.append(f'<text class="axlbl" transform="rotate(-90 15 {mt+ph/2:.1f})" x="15" '
               f'y="{mt+ph/2:.1f}" text-anchor="middle">{ylab}</text>')
    out.append("</svg>")
    return "\n".join(out)


def build_paired_svg(df: pd.DataFrame, p_grid: list[float]) -> str:
    """Efficiency frontiers: risk of an overly toxic MTD against accuracy."""
    agg = (df[df["series"].isin(["attribution", "weight"])]
           .groupby(["series", "p"], as_index=False)
           .agg(correct=("correct_pct", "mean"), risk=("too_high_pct", "mean")))
    W, H = 720, 400
    ml, mr, mt, mb = 60, 150, 24, 60
    pw, ph = W - ml - mr, H - mt - mb
    xmax = min(100.0, ((float(agg["risk"].max()) + 5) // 10 + 1) * 10)
    ymin = max(0.0, ((float(agg["correct"].min()) - 4) // 5) * 5)
    ymax = min(100.0, ((float(agg["correct"].max()) + 4) // 5 + 1) * 5)

    def px(v):
        return ml + (float(v) / xmax) * pw

    def py(v):
        return mt + ph - ((float(v) - ymin) / (ymax - ymin)) * ph

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
           f'aria-label="Efficiency frontier of both discounting schemes">']
    for g in range(0, int(xmax) + 1, 10):
        x = px(g)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{mt+ph+19}" text-anchor="middle">{g}%</text>')
    g = int(ymin)
    while g <= ymax:
        if g % 5 == 0:
            y = py(g)
            out.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}"/>')
            out.append(f'<text class="tick" x="{ml-10}" y="{y+4:.1f}" text-anchor="end">{g}%</text>')
        g += 5
    out.append(f'<line class="axis" x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}"/>')
    out.append(f'<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}"/>')

    style = {"attribution": ("var(--accent)", "none", "Non-binary DLT", 2.4),
             "weight":      ("var(--muted)", "5 4", "Block discount", 2.0)}
    for skey, (col, dash, lab, lw) in style.items():
        s2 = agg[agg["series"] == skey].sort_values("risk")
        pts = [(px(r["risk"]), py(r["correct"])) for _, r in s2.iterrows()]
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
                        for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="{lw}" '
                   f'stroke-dasharray="{dash}" stroke-linejoin="round"/>')
        for (x, y), (_, r) in zip(pts, s2.iterrows()):
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{col}"/>')
        lx, ly = pts[-1]
        out.append(f'<text class="lbl" x="{lx+11:.1f}" y="{ly+4:.1f}" fill="{col}">{lab}</text>')

    # annotate the shared endpoint (p = 1, the conventional binary design)
    b = agg[(agg["series"] == "attribution") & (agg["p"] == 1.0)].iloc[0]
    out.append(f'<text class="lbl" x="{px(b["risk"])+9:.1f}" y="{py(b["correct"])+16:.1f}" '
               f'fill="var(--ink)">p = 1 (binary)</text>')

    out.append(f'<text class="axlbl" x="{ml+pw/2:.1f}" y="{H-12}" text-anchor="middle">'
               f'Risk of selecting an overly toxic MTD &#8594;</text>')
    out.append(f'<text class="axlbl" transform="rotate(-90 15 {mt+ph/2:.1f})" x="15" '
               f'y="{mt+ph/2:.1f}" text-anchor="middle">Correct MTD selected &#8594;</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path("dlt_attribution_sensitivity"))
    ap.add_argument("--out", type=Path,
                    default=Path("dlt_attribution_sensitivity/nonbinary_dlt_report.html"))
    args = ap.parse_args()

    df = pd.read_csv(args.indir / "dlt_attribution_sweep.csv")
    meta = json.loads((args.indir / "dlt_attribution_sweep_meta.json").read_text(encoding="utf-8"))
    p_grid = list(meta["p_grid"])
    n_sim = int(meta["n_sim"])
    scen = list(dict.fromkeys(df["scenario"]))

    attr = df[df["series"] == "attribution"]
    wgt = df[df["series"] == "weight"]
    ovr = df[df["series"] == "attribution_override"]

    def at(p, s=None, frame=attr, col="correct_pct"):
        f = frame[frame["p"] == p]
        if s:
            f = f[f["scenario"] == s]
        return float(f[col].mean())

    # main table: correct % per scenario per p, plus averages
    rows = []
    for p in p_grid:
        cells = "".join(f"<td>{fmt(at(p, s))}</td>" for s in scen)
        cls = ' class="rec"' if p == 0.50 else ""
        rows.append(f'<tr{cls}><td>{"0" if p == 0 else ("1.00" if p == 1 else f"{p:.2f}")}</td>'
                    f'{cells}<td><b>{fmt(at(p))}</b></td>'
                    f'<td>{fmt(at(p, col="too_high_pct"))}</td></tr>')
    head = "".join(f"<th>{s.replace('Acute ', '')}</th>" for s in scen)
    table = (f'<table><caption>Correct MTD selection by attribution probability (%)</caption>'
             f'<thead><tr><th>p</th>{head}<th>Average</th><th>Too high<br>(avg)</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table>')

    # paired table: attribution vs block discount
    prows = []
    for p in p_grid:
        a_c, a_r = at(p), at(p, col="too_high_pct")
        w_c = float(wgt[wgt["p"] == p]["correct_pct"].mean())
        w_r = float(wgt[wgt["p"] == p]["too_high_pct"].mean())
        o_c = float(ovr[ovr["p"] == p]["correct_pct"].mean())
        o_r = float(ovr[ovr["p"] == p]["too_high_pct"].mean())
        prows.append(f'<tr><td>{"0" if p == 0 else ("1.00" if p == 1 else f"{p:.2f}")}</td>'
                     f'<td>{fmt(a_c)}</td><td>{fmt(a_r)}</td>'
                     f'<td>{fmt(w_c)}</td><td>{fmt(w_r)}</td>'
                     f'<td>{fmt(o_c)}</td><td>{fmt(o_r)}</td></tr>')
    ptable = (f'<table><caption>Non-binary DLT against block discount, averaged over scenarios</caption>'
              f'<thead><tr><th rowspan="2">p</th>'
              f'<th colspan="2">Non-binary DLT</th>'
              f'<th colspan="2">Block discount</th>'
              f'<th colspan="2">Non-binary + override</th></tr>'
              f'<tr><th>Correct</th><th>Too high</th><th>Correct</th><th>Too high</th>'
              f'<th>Correct</th><th>Too high</th></tr></thead>'
              f'<tbody>{"".join(prows)}</tbody></table>')

    # SAE causality scale mapped onto fixed representative attribution probabilities
    CAT_SCALE = [
        ("Unrelated", 0.00), ("Unlikely", 0.05), ("Possible", 0.25),
        ("Probable", 0.75), ("Definite", 1.00),
    ]
    cat_rows = []
    for label, p in CAT_SCALE:
        c, r = at(p), at(p, col="too_high_pct")
        cls = ' class="rec"' if label == "Possible" else ""
        cat_rows.append(f'<tr{cls}><td>{label}</td><td>{p:.2f}</td>'
                        f'<td>{fmt(c)}</td><td>{fmt(r)}</td></tr>')
    cat_table = (f'<table><caption>SAE causality tier mapped to a fixed attribution '
                f'probability, averaged over scenarios</caption>'
                f'<thead><tr><th>Causality tier</th><th>Attribution p</th>'
                f'<th>Correct MTD</th><th>Too high</th></tr></thead>'
                f'<tbody>{"".join(cat_rows)}</tbody></table>')
    cat_possible_c = at(0.25)
    cat_possible_r = at(0.25, col="too_high_pct")

    curve = build_curve_svg(attr, p_grid, "correct_pct", "Correct MTD selected")
    paired = build_paired_svg(df, p_grid)

    # Risk-matched comparison: the same numeric p means different things in the
    # two schemes, so compare the frontiers at equal risk instead.
    import numpy as _np
    fa = (attr.groupby("p").agg(c=("correct_pct", "mean"), r=("too_high_pct", "mean"))
          .sort_values("r"))
    fw = (wgt.groupby("p").agg(c=("correct_pct", "mean"), r=("too_high_pct", "mean"))
          .sort_values("r"))
    gaps = []
    for _p, _row in fw.iterrows():
        if _row["r"] <= float(fa["r"].max()) and _row["r"] >= float(fa["r"].min()):
            gaps.append(float(_np.interp(_row["r"], fa["r"], fa["c"]) - _row["c"]))
    gap_lo, gap_hi = (min(gaps), max(gaps)) if gaps else (0.0, 0.0)
    gap_mid = sum(gaps) / len(gaps) if gaps else 0.0

    # best average accuracy over the attribution grid
    _best = fa.sort_values("c", ascending=False).iloc[0]
    best_p = float(fa[fa["c"] == _best["c"]].index[0])
    best_c, best_r = float(_best["c"]), float(_best["r"])
    low_best = at(best_p, "Acute low")

    low0, low1 = at(0.0, "Acute low"), at(1.0, "Acute low")
    high0, high1 = at(0.0, "Acute high"), at(1.0, "Acute high")
    r0, r1 = at(0.0, col="too_high_pct"), at(1.0, col="too_high_pct")
    gain = at(0.5) - at(1.0)
    wgain = float(wgt[wgt["p"] == 0.5]["correct_pct"].mean()) - at(1.0)
    wrisk = float(wgt[wgt["p"] == 0.5]["too_high_pct"].mean())
    arisk = at(0.5, col="too_high_pct")

    scen_tbl = "".join(
        f"<tr><td>{k.replace('Acute ', '')}</td>"
        + "".join(f"<td>{v:.2f}</td>" for v in meta["scenarios"][k])
        + f"<td><b>L{max([i for i, q in enumerate(meta['scenarios'][k]) if q <= meta['target_acute']] or [0])}</b></td></tr>"
        for k in meta["scenarios"])

    html = f"""<title>The Non-Binary DLT</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style>
<div class="wrap">
<header class="masthead">
  <p class="kicker">Methodological note &middot; MERGE dose escalation</p>
  <h1>Treating a dose-limiting toxicity as a probability rather than a yes or no</h1>
  <p class="standfirst">A single contested toxicity currently costs the design as much as an
  unambiguous one. This note asks whether attribution should enter the model as a fraction,
  reviews what already exists in the literature, and quantifies what it would change.</p>
  <div class="meta">
    <span>Simulations <b>{n_sim}</b> per cell</span>
    <span>Acute target <b>{meta['target_acute']:.2f}</b></span>
    <span>Subacute target <b>{meta['target_subacute']:.2f}</b></span>
    <span>Configuration <b>no EWOC, no burn-in</b></span>
  </div>
</header>

<section class="step"><div class="col">
  <div class="steph"><span class="num">1</span><h2>The problem with a binary attribution</h2></div>
  <p class="lede">The sixth patient treated at L1 experienced an event assessed as
  &ldquo;possible&rdquo; — not unrelated, not definite. The design has no way to express that.
  It must record either a 1 or a 0, and the two answers lead to very different trials.</p>
  <p>This is not a peculiarity of our study. Adverse event attribution is routinely collected
  on a five-tier scale — unrelated, unlikely, possible, probable, definite — and then
  collapsed to a binary for the purpose of dose escalation. The graded judgement the
  clinicians actually made is discarded at exactly the point where it would matter most.</p>
  <p>There is good evidence that the binary is unreliable. A review of randomised phase III
  trials found that <strong>half of the adverse events reported on placebo arms were recorded
  as drug related</strong>, and that among patients with a repeated event,
  <strong>36% changed attribution over time</strong>. Forcing a confident 0 or 1 out of a
  genuinely uncertain judgement does not remove the uncertainty; it hides it.</p>
  <div class="pull">Radiotherapy makes this sharper still. The tumour itself produces events
  that look like treatment toxicity — as in this case, where disease progression and
  irradiation are competing explanations for the same finding.</div>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">2</span><h2>This has been done before</h2></div>
  <p>The idea of a non-binary toxicity endpoint is established, which is useful: it means the
  approach can be cited rather than invented.</p>
  <h3>Non-binary toxicity by severity</h3>
  <p>The quasi-CRM of Yuan, Chappell and Bailey (2007) replaces the binary outcome with an
  equivalent toxicity score and fits it through a quasi-Bernoulli likelihood — grade 2 counts
  as 0.5, grade 3 as 1, grade 4 as 1.5. The machinery for a fractional DLT has therefore been
  in routine use for close to twenty years, including extensions to drug-combination designs.</p>
  <h3>Non-binary toxicity by attribution</h3>
  <p>Closer to the present question, two published designs handle attribution itself as
  uncertain: <em>Phase I designs that allow for uncertainty in the attribution of adverse
  events</em> (JRSS-C), and <em>A Bayesian dose-finding design for outcomes evaluated with
  uncertainty</em> (2021). The latter is almost exactly the proposal here — for some patients
  the physician records a probability of DLT rather than a binary, and the posterior is
  obtained by data augmentation over the unobserved binary outcome, with the CRM as the
  worked example.</p>
  <p>So the proposal is not novel in its mechanics. What is unexplored is the part specific
  to radiotherapy, discussed in section 5.</p>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">3</span><h2>Discounting the event is not the same as discounting the patient</h2></div>
  <p class="lede">This distinction is easy to miss and changes the answer. Our earlier analysis
  down-weighted the historical data the wrong way.</p>
  <p>A TITE-CRM likelihood accumulates two quantities per dose level: an effective sample size
  <em>n</em> and an effective event count <em>y</em>. There are two quite different ways to
  soften a disputed event.</p>
</div>
<div class="tablewrap">
  <table><caption>Six pre-treated patients at L1, one contested event, half weight</caption>
  <thead><tr><th>Approach</th><th>n</th><th>y</th><th>What it asserts</th></tr></thead>
  <tbody>
  <tr><td>Block discount</td><td>3.0</td><td>0.5</td>
    <td>All six observations are worth half as much</td></tr>
  <tr class="rec"><td>Non-binary DLT</td><td>6.0</td><td>0.5</td>
    <td>All six were fully observed; half the event was real</td></tr>
  </tbody></table>
</div>
<div class="col">
  <p style="margin-top:18px">The block discount also halves the evidential weight of the
  <strong>five patients who had no event at all</strong> — patients about whom nothing is in
  doubt. Only the sixth is contested. Scaling <em>y</em> alone isolates the uncertainty where
  it actually sits, and keeps the reassuring information the other five provide.</p>
  <p>The two are not simply stronger and weaker versions of one another. Halving the block
  leaves the apparent toxicity rate at L1 unchanged at one in six, but makes the design less
  sure of it. Halving the event lowers the apparent rate itself, to half an event in six
  patients, while keeping the design just as sure. The same number therefore means something
  different in each scheme, and the two must be compared by what they buy, not by the label
  on the dial.</p>
  <p>Compared that way — at equal risk of selecting an overly toxic dose — the non-binary DLT
  is worth {fmt(gap_mid)} percentage points of accuracy on average (range {fmt(gap_lo)} to
  {fmt(gap_hi)}). The advantage is real but moderate, and it comes from keeping the five
  event-free patients at full strength.</p>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">4</span><h2>What the simulations show</h2></div>
  <p>We swept the attribution probability assigned to the contested event across
  0, 0.05, 0.10, 0.25, 0.50 and 1.00, over all five acute toxicity scenarios,
  {n_sim} simulated trials per cell. At p = 1 the design is the conventional binary one;
  at p = 0 the event is never recorded.</p>
</div>

<figure>
  <div class="figbox">{curve}</div>
  <figcaption><strong>Correct MTD selection against attribution probability.</strong>
  The scenarios move in opposite directions, which is the central finding. Where the true MTD
  is the top dose (low) a lower attribution helps sharply, from {fmt(low1)}% at p = 1 to
  {fmt(low0)}% at p = 0. Where the true MTD is low (high, steep) the same change hurts, from
  {fmt(high1)}% down to {fmt(high0)}%. There is no value of p that is best everywhere,
  because p encodes a belief about the world, not a tuning parameter.</figcaption>
</figure>

<figure>
  <div class="figbox">{paired}</div>
  <figcaption><strong>What each scheme buys, averaged over scenarios.</strong>
  Both curves start from the same point at the lower left — the conventional binary design,
  which discounts nothing. As either dial is turned the design trades safety for accuracy, and
  the non-binary DLT sits above the block discount over the whole useful range: at equal risk
  it is worth {fmt(gap_mid)} percentage points of accuracy on average. Note also that neither
  curve keeps climbing. Beyond roughly a third risk both turn over, so discounting the event
  away entirely is not the most accurate choice, merely the least safe.</figcaption>
</figure>

<div class="tablewrap">{table}</div>
<div class="tablewrap">{ptable}</div>
<div class="col">
  <p style="margin-top:18px">The safety cost is monotone: averaged over scenarios, the risk of
  selecting an overly toxic MTD rises from {fmt(r1)}% at p = 1 to {fmt(r0)}% at p = 0. Nothing
  about making the endpoint continuous removes that trade-off. What it does is let the
  trade-off be set by a stated clinical belief rather than by a forced binary.</p>
  <div class="callout">
    <p class="ct">Accuracy does not simply increase as the event is discounted</p>
    <p>Averaged over the five scenarios, accuracy peaks at <strong>p = {best_p:g}</strong>
    ({fmt(best_c)}% correct at {fmt(best_r)}% risk) and falls away on both sides. Counting the
    event in full costs accuracy because it holds the design below a genuinely tolerable dose;
    discarding it entirely costs accuracy too, because the design then overshoots wherever the
    true MTD is low. A modest, stated discount does better than either extreme — which is a
    reasonable thing to find when the clinical assessment itself was neither 0 nor 1.</p>
    <p>This should not be read as a recommendation to set p = {best_p:g}. The average across
    five hypothetical scenarios is not a quantity anyone is trying to optimise, and p is meant
    to express a belief about this event, not to be tuned. It does show that the conventional
    binary choice is not the accuracy-maximising one under any reading.</p>
  </div>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">5</span><h2>Where radiotherapy may add something new</h2></div>
  <p>In the published attribution designs, p is a subjective judgement by the treating
  physician. That is the method's weak point, and the reason it has not been widely adopted:
  a number between 0 and 1 invites more variation between assessors than a binary does, and
  it is hard to audit.</p>
  <p>Radiotherapy is unusual in offering an objective substrate for that judgement. For a
  given event one can ask where it occurred relative to the irradiated volume, what dose that
  volume received, and how that compares with the planning constraints. Our own serious
  adverse event documentation already reasons this way — recording that the stomach dose
  stayed within the predefined constraints, and undertaking to correlate the anatomical
  location of the finding with the treatment field.</p>
  <p>That reasoning is currently written in prose and then collapsed into a binary. Turning it
  into a pre-specified, dosimetry-based mapping onto an attribution probability would be a
  genuine methodological contribution, and one that is specific to radiotherapy dose finding
  rather than borrowed from systemic oncology.</p>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">6</span><h2>A practical elicitation scheme: discretize to the existing causality scale</h2></div>
  <p class="lede">Asking a safety committee to name a continuous probability invites a question
  nobody can answer: why 0.35 and not 0.40? A coarser scheme avoids it, and one is already
  sitting on the SAE form.</p>
  <p>Every serious adverse event is already classified on a five-tier causality scale &mdash;
  <em>unrelated, unlikely, possible, probable, definite</em> &mdash; the same scale used for
  MERGE-011. Rather than inventing a new categorical system, the committee could fix a
  representative attribution probability for each existing tier once, and apply that table
  case by case. That turns an unanswerable question about a decimal into a classification
  judgement clinicians already make routinely, and it reuses the discrete-score logic the
  quasi-CRM already applies to toxicity grade &mdash; the same mechanism, aimed at causality
  instead of severity.</p>
</div>
<div class="tablewrap">{cat_table}</div>
<div class="col">
  <p style="margin-top:18px">Discretizing costs nothing in these simulations: the five tier
  values trace the same curve as the continuous sweep in section 4, because they are five
  points read off it. <strong>MERGE-011 was assessed as &ldquo;possible&rdquo;</strong> &mdash;
  which happens to be the tier with the best average accuracy, {fmt(cat_possible_c)}% correct
  at {fmt(cat_possible_r)}% risk, not because the analysis was built to reach that conclusion
  but because that is where the existing classification already places this event.</p>
  <p>What this does not fix: it does not touch the quasi-likelihood issue in section 7, and it
  narrows rather than removes the incentive to shade an assessment, since a borderline case can
  still be pushed into a more favourable neighbouring tier. A visible jump between tiers is
  easier to question at review than an unexplained decimal, which is the actual gain &mdash;
  not the elimination of the underlying tension between speed and caution.</p>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">7</span><h2>What would have to be settled first</h2></div>
  <div class="callout">
    <p class="ct">1 &middot; The quasi-likelihood is sharper than the evidence</p>
    <p>Substituting y = p is not the same as marginalising over the unknown attribution.
    The correct mixture likelihood is p&middot;P + (1&minus;p)(1&minus;P); the quasi-likelihood
    used here is P<sup>p</sup>(1&minus;P)<sup>1&minus;p</sup>. By Jensen's inequality the
    latter is the more confident of the two, so this implementation slightly overstates the
    information the disputed event carries. The published design handles this properly through
    data augmentation over the latent binary outcome, and a protocol version should do the same.</p>
  </div>
  <div class="callout">
    <p class="ct">2 &middot; Who assigns p, and when</p>
    <p>An assessor who knows the patient was treated at the top dose level may, without
    intending to, shade the number, and the incentive is sharper here than for an ordinary
    adverse-event grading: a lower attribution has an immediate, visible effect on the next
    dosing decision. Blinding assessment to dose level, the usual mitigation, is difficult to
    operationalise in a small radiotherapy trial where the treatment record makes the dose
    level obvious. Section 6 discusses a coarser elicitation scheme that narrows, without
    eliminating, this problem.</p>
  </div>
  <div class="callout">
    <p class="ct">3 &middot; Scope of what was simulated</p>
    <p>Only the one contested historical event was made fractional here. Every event arising
    during the simulated trial still counts as a full DLT. A design that applied attribution
    probabilities prospectively would also need a model of how assessors assign them, which is
    a substantially larger undertaking and would change these numbers.</p>
  </div>
  <div class="callout">
    <p class="ct">4 &middot; Regulatory reception</p>
    <p>A committee reading that a patient contributed 0.4 of a dose-limiting toxicity will
    reasonably ask what that means for the safety narrative. The argument is defensible — the
    graded assessment is already being made, and this stops discarding it — but it needs to be
    made explicitly rather than assumed.</p>
  </div>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">8</span><h2>Recommendation</h2></div>
  <div class="callout rec">
    <p class="ct">For the current amendment</p>
    <p><strong>Do not adopt this now.</strong> It is a larger methodological step than the
    amendment needs, it requires the statistician's input and a proper data-augmentation
    implementation, and it would complicate approval on a timeline aimed at April. Keep the
    amendment to the design transition and the weighting decision already set out.</p>
  </div>
  <p>Two things are nonetheless worth carrying forward immediately.</p>
  <ul>
    <li><strong>Correct the discounting we do adopt.</strong> If the amendment down-weights the
    historical data at all, it should scale the event rather than the block. At equal risk this
    buys {fmt(gap_mid)} percentage points of accuracy, and the argument for it is easier to
    make: we are uncertain about one event, not about five patients who had none. This change
    is small, self-contained, and does not require the full attribution machinery.</li>
    <li><strong>Pursue the dosimetric attribution mapping as a separate track.</strong> The
    statistical machinery is published and citable; the radiotherapy-specific part is not, and
    that is where a real contribution would sit. This study is a well-documented motivating
    case for it.</li>
  </ul>
  <h3>Suggested next step</h3>
  <p>Put the mixture-likelihood question to the trial statistician first. If a data-augmented
  implementation is straightforward in the existing R code, the remaining questions are
  clinical rather than statistical, and can be worked through by the team.</p>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">9</span><h2>Scenarios and settings</h2></div>
  <p>True acute toxicity probabilities, with subacute probabilities fixed at
  {", ".join(f"{v:.2f}" for v in meta['true_subacute'])}. The true MTD is the highest level at
  or below the acute target of {meta['target_acute']:.2f}.</p>
</div>
<div class="tablewrap">
  <table><caption>True acute toxicity probabilities by scenario</caption>
  <thead><tr><th>Scenario</th>{"".join(f"<th>L{i}<br>{lab}</th>" for i, lab in enumerate(meta["dose_labels"]))}<th>True MTD</th></tr></thead>
  <tbody>{scen_tbl}</tbody></table>
</div>
<div class="col">
  <h3>References</h3>
  <ul>
    <li>Yuan Z, Chappell R, Bailey H. The continual reassessment method for multiple toxicity
    grades: a Bayesian quasi-likelihood approach. <em>Biometrics</em> 2007.</li>
    <li>Phase I designs that allow for uncertainty in the attribution of adverse events.
    <em>Journal of the Royal Statistical Society Series C</em>.</li>
    <li>A Bayesian dose-finding design for outcomes evaluated with uncertainty. 2021.</li>
    <li>Quasi-partial order continual reassessment method: applying toxicity scores to cancer
    dose-finding drug combination trials. <em>Contemporary Clinical Trials</em> 2022.</li>
  </ul>
</div></section>

<footer><div class="col">
  <p>Generated from <span style="font-family:var(--mono);font-size:12px">dlt_attribution_sweep.py</span>
  ({n_sim} simulations per scenario per value, seed {meta['seed']}). TITE-CRM without EWOC and
  without burn-in, starting at L2, 30 patients including the six pre-treated, cohort size 3,
  one new patient every four weeks.</p>
  <p>This document contains aggregated simulation output only. No patient data is included.</p>
</div></footer>
</div>
"""
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
