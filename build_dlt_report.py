#!/usr/bin/env python3
"""Render the DLT-handling decision report from the simulation output.

Reads dlt_attribution_sensitivity/dlt_mitigation_summary.csv (written by
dlt_mitigation_analysis.py) and emits a self-contained HTML page aimed at the
whole trial team, including non-statisticians: what the contested acute DLT in
the L1 initialization cohort does to the TITE-CRM design, which options exist
for handling it, and what the simulations say about each.

Usage:
    python build_dlt_report.py [--indir DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# ── palette (kept in sync with the CSS tokens below) ──────────────────────────
OPTION_ORDER = ["A", "B", "C", "E", "E+B", "E+C", "H"]
RECOMMENDED = "E"

SHORT = {
    "A":   "DLT volledig",
    "B":   "Downweging 50%",
    "C":   "Downweging 25%",
    "E":   "Override",
    "E+B": "Override + 50%",
    "E+C": "Override + 25%",
    "H":   "DLT niet geteld",
}

WHAT = {
    "A":   "De zes patiënten op L1 gaan mee als volledige observaties, inclusief de DLT. Dit is wat de code nu doet.",
    "B":   "Zelfde data, maar het hele historische blok telt voor de helft mee — een gangbare Bayesiaanse korting op data van vóór het amendement.",
    "C":   "Zelfde idee, maar het blok telt nog maar voor een kwart mee.",
    "E":   "De DLT blijft volledig meetellen. Daarnaast mag er één niveau omhoog zodra drie patiënten op het huidige niveau hun volledige acute follow-up zonder DLT hebben afgerond — ook als het model dat nog niet aanraadt.",
    "E+B": "De override, gecombineerd met een korting van 50% op het historische blok.",
    "E+C": "De override, gecombineerd met een korting van 75% op het historische blok.",
    "H":   "De DLT telt niet mee, bijvoorbeeld omdat het event formeel als niet-behandelgerelateerd wordt beoordeeld. Opgenomen als referentiepunt.",
}


def fmt(x, d=1):
    return f"{x:.{d}f}"


def build_tradeoff_svg(agg: pd.DataFrame) -> str:
    """Accuracy vs overshoot risk — one marker per option."""
    W, H = 720, 430
    ml, mr, mt, mb = 66, 132, 26, 56
    pw, ph = W - ml - mr, H - mt - mb

    xs = agg["too_high_pct"]
    ys = agg["correct_pct"]
    x_max = max(50.0, (float(xs.max()) // 10 + 1) * 10)
    # tight vertical framing: the options differ by a few points, so a 0-100
    # axis would flatten the whole comparison into one band
    y_min = max(0.0, ((float(ys.min()) - 4) // 5) * 5)
    y_max = min(100.0, ((float(ys.max()) + 4) // 5 + 1) * 5)

    def px(v):
        return ml + (float(v) / x_max) * pw

    def py(v):
        return mt + ph - ((float(v) - y_min) / (y_max - y_min)) * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Nauwkeurigheid tegen overschattingsrisico per optie" '
             f'class="chart">']

    # grid + axes
    for gv in range(0, int(x_max) + 1, 10):
        x = px(gv)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{mt+ph+20}" text-anchor="middle">{gv}%</text>')
    gy = int(y_min)
    while gy <= y_max:
        if gy % 10 == 0:
            y = py(gy)
            parts.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}"/>')
            parts.append(f'<text class="tick" x="{ml-10}" y="{y+4:.1f}" text-anchor="end">{gy}%</text>')
        gy += 5

    parts.append(f'<line class="axis" x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}"/>')
    parts.append(f'<line class="axis" x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}"/>')

    # markers
    rows = list(agg.itertuples())
    for r in rows:
        x, y = px(r.too_high_pct), py(r.correct_pct)
        rec = (r.option == RECOMMENDED)
        cls = "pt rec" if rec else "pt"
        rad = 9 if rec else 6.5
        if rec:
            parts.append(f'<circle class="halo" cx="{x:.1f}" cy="{y:.1f}" r="{rad+7:.1f}"/>')
        parts.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}"/>')

    # labels placed to the right, de-collided vertically
    placed: list[float] = []
    for r in sorted(rows, key=lambda z: -z.correct_pct):
        x, y = px(r.too_high_pct), py(r.correct_pct)
        ly = y
        while any(abs(ly - p) < 17 for p in placed):
            ly += 17
        placed.append(ly)
        lx = min(x + 16, ml + pw + 12)
        weight = ' rec' if r.option == RECOMMENDED else ''
        parts.append(f'<line class="leader" x1="{x+7:.1f}" y1="{y:.1f}" x2="{lx-4:.1f}" y2="{ly:.1f}"/>')
        parts.append(f'<text class="lbl{weight}" x="{lx:.1f}" y="{ly+4:.1f}">{SHORT[r.option]}</text>')

    parts.append(f'<text class="axlbl" x="{ml+pw/2:.1f}" y="{H-12}" text-anchor="middle">'
                 f'Kans op een te hoge (onveilige) MTD →</text>')
    parts.append(f'<text class="axlbl" transform="rotate(-90 16 {mt+ph/2:.1f})" '
                 f'x="16" y="{mt+ph/2:.1f}" text-anchor="middle">Correcte MTD gekozen →</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def build_low_svg(low: pd.DataFrame) -> str:
    """Acute low: correct MTD selection + whether L4 was ever reached."""
    opts = [o for o in OPTION_ORDER if o in set(low["option"])]
    W, H = 720, 330
    ml, mr, mt, mb = 56, 20, 30, 74
    pw, ph = W - ml - mr, H - mt - mb
    n = len(opts)
    slot = pw / n
    bw = min(38.0, slot * 0.32)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Acute low scenario per optie" class="chart">']
    for gv in range(0, 101, 20):
        y = mt + ph - (gv / 100) * ph
        parts.append(f'<line class="grid" x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{ml-10}" y="{y+4:.1f}" text-anchor="end">{gv}%</text>')
    parts.append(f'<line class="axis" x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}"/>')

    for i, o in enumerate(opts):
        r = low[low["option"] == o].iloc[0]
        cx = ml + slot * (i + 0.5)
        for j, (val, cls) in enumerate([(float(r["ever_reached_top_pct"]), "bar reach"),
                                         (float(r["correct_pct"]), "bar sel")]):
            h = (val / 100) * ph
            x = cx - bw + j * (bw + 3)
            y = mt + ph - h
            extra = " rec" if o == RECOMMENDED else ""
            parts.append(f'<rect class="{cls}{extra}" x="{x:.1f}" y="{y:.1f}" '
                         f'width="{bw:.1f}" height="{max(h,1):.1f}" rx="2"/>')
            parts.append(f'<text class="val" x="{x+bw/2:.1f}" y="{y-6:.1f}" '
                         f'text-anchor="middle">{fmt(val,0)}</text>')
        lab = SHORT[o]
        words = lab.split(" ")
        mid = (len(words) + 1) // 2
        l1, l2 = " ".join(words[:mid]), " ".join(words[mid:])
        weight = " rec" if o == RECOMMENDED else ""
        parts.append(f'<text class="cat{weight}" x="{cx:.1f}" y="{mt+ph+20}" text-anchor="middle">{l1}</text>')
        if l2:
            parts.append(f'<text class="cat{weight}" x="{cx:.1f}" y="{mt+ph+34}" text-anchor="middle">{l2}</text>')

    ly = H - 16
    parts.append(f'<rect class="bar reach" x="{ml}" y="{ly-9}" width="11" height="11" rx="2"/>')
    parts.append(f'<text class="leg" x="{ml+17}" y="{ly}">L4 tijdens de trial bereikt</text>')
    parts.append(f'<rect class="bar sel" x="{ml+215}" y="{ly-9}" width="11" height="11" rx="2"/>')
    parts.append(f'<text class="leg" x="{ml+232}" y="{ly}">L4 als finale MTD gekozen (= correct)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


CSS = """
:root{
  --paper:#FAFCFD; --surface:#FFFFFF; --ink:#101B26; --slate:#1E3A52;
  --muted:#5C6B7A; --line:#DCE4EA; --line-soft:#EAEFF3;
  --accent:#0E7C86; --accent-soft:#E3F1F2;
  --safe:#2F7D5B; --risk:#B4472F; --warn:#8A6A1F;
  --serif:Georgia,"Iowan Old Style","Palatino Linotype",Palatino,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0D141C; --surface:#151F29; --ink:#DCE4EB; --slate:#AFC4D4;
    --muted:#8A9AA9; --line:#25333F; --line-soft:#1C2833;
    --accent:#3AAEB6; --accent-soft:#14262C;
    --safe:#5BAE84; --risk:#D97A5E; --warn:#C8A24A;
  }
}
:root[data-theme="dark"]{
  --paper:#0D141C; --surface:#151F29; --ink:#DCE4EB; --slate:#AFC4D4;
  --muted:#8A9AA9; --line:#25333F; --line-soft:#1C2833;
  --accent:#3AAEB6; --accent-soft:#14262C;
  --safe:#5BAE84; --risk:#D97A5E; --warn:#C8A24A;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--sans); font-size:16.5px; line-height:1.65;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 96px}
.col{max-width:68ch}
header.masthead{padding:64px 0 34px;border-bottom:1px solid var(--line)}
.kicker{
  font-family:var(--mono); font-size:11.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); margin:0 0 16px;
}
h1{
  font-family:var(--serif); font-weight:400; font-size:clamp(31px,4.4vw,46px);
  line-height:1.14; letter-spacing:-.012em; margin:0 0 18px;
  text-wrap:balance; color:var(--ink);
}
.standfirst{font-size:19px;line-height:1.6;color:var(--muted);margin:0;max-width:60ch}
.meta{
  display:flex;flex-wrap:wrap;gap:10px 26px;margin:26px 0 0;
  font-family:var(--mono);font-size:11.5px;letter-spacing:.05em;
  text-transform:uppercase;color:var(--muted);
}
.meta b{color:var(--ink);font-weight:600}
section.step{padding:52px 0 0}
.steph{display:flex;gap:16px;align-items:baseline;margin:0 0 20px}
.num{
  font-family:var(--mono);font-size:12px;font-weight:600;color:var(--accent);
  border:1px solid var(--accent);border-radius:3px;padding:3px 7px;
  flex:none;letter-spacing:.06em;
}
h2{
  font-family:var(--serif);font-weight:400;font-size:clamp(23px,2.7vw,30px);
  line-height:1.22;margin:0;letter-spacing:-.008em;text-wrap:balance;
}
h3{
  font-family:var(--sans);font-size:14px;font-weight:650;letter-spacing:.02em;
  margin:34px 0 10px;color:var(--slate);
}
p{margin:0 0 15px}
a{color:var(--accent)}
strong{font-weight:650}
.lede{font-size:17.5px;color:var(--ink)}
ul,ol{margin:0 0 15px;padding-left:22px}
li{margin:0 0 8px}
.pull{
  border-left:2px solid var(--accent);padding:2px 0 2px 20px;margin:26px 0;
  font-family:var(--serif);font-size:19.5px;line-height:1.5;color:var(--slate);
}
.callout{
  background:var(--surface);border:1px solid var(--line);border-radius:5px;
  padding:20px 22px;margin:26px 0;
}
.callout.rec{border-color:var(--accent);background:var(--accent-soft)}
.callout .ct{
  font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--accent);margin:0 0 9px;
}
.callout p:last-child{margin-bottom:0}
figure{margin:32px 0 0}
.figbox{
  background:var(--surface);border:1px solid var(--line);border-radius:5px;
  padding:22px 20px 14px;overflow-x:auto;
}
figcaption{
  font-size:13.5px;color:var(--muted);margin:12px 0 0;max-width:70ch;line-height:1.55;
}
svg.chart{display:block;width:100%;min-width:600px;height:auto;overflow:visible}
.grid{stroke:var(--line-soft);stroke-width:1}
.axis{stroke:var(--line);stroke-width:1}
.tick,.leg,.cat,.val,.hint{font-family:var(--mono);fill:var(--muted)}
.tick{font-size:10.5px}
.cat{font-size:10.5px}
.cat.rec{fill:var(--accent);font-weight:600}
.val{font-size:10px;fill:var(--muted)}
.leg{font-size:10.5px}
.hint{font-size:10px;fill:var(--muted);letter-spacing:.08em}
.axlbl{font-family:var(--sans);font-size:12px;fill:var(--muted)}
.lbl{font-family:var(--sans);font-size:12.5px;fill:var(--ink)}
.lbl.rec{fill:var(--accent);font-weight:650}
.pt{fill:var(--surface);stroke:var(--muted);stroke-width:1.8}
.pt.rec{fill:var(--accent);stroke:var(--accent)}
.halo{fill:none;stroke:var(--accent);stroke-width:1;opacity:.32}
.leader{stroke:var(--line);stroke-width:1}
.bar.reach{fill:var(--line);stroke:var(--muted);stroke-width:.8}
.bar.sel{fill:var(--muted)}
.bar.sel.rec{fill:var(--accent)}
.bar.reach.rec{fill:var(--accent-soft);stroke:var(--accent)}
.tablewrap{overflow-x:auto;margin:30px 0 0;border:1px solid var(--line);border-radius:5px;background:var(--surface)}
table{border-collapse:collapse;width:100%;min-width:660px;font-size:13.5px}
caption{
  text-align:left;padding:16px 18px 0;font-family:var(--mono);font-size:11px;
  letter-spacing:.12em;text-transform:uppercase;color:var(--accent);
}
th,td{padding:9px 14px;text-align:right;border-bottom:1px solid var(--line-soft)}
th:first-child,td:first-child{text-align:left}
thead th{
  font-family:var(--mono);font-size:10.5px;letter-spacing:.07em;text-transform:uppercase;
  color:var(--muted);font-weight:500;border-bottom:1px solid var(--line);vertical-align:bottom;
}
tbody td{font-family:var(--mono);font-variant-numeric:tabular-nums}
tbody td:first-child{font-family:var(--sans)}
tbody tr:last-child td{border-bottom:none}
tr.rec td{background:var(--accent-soft)}
tr.rec td:first-child{font-weight:650;color:var(--accent)}
.good{color:var(--safe)}
.bad{color:var(--risk)}
.optlist{display:grid;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:5px;margin:30px 0 0}
.opt{background:var(--surface);padding:18px 20px;display:grid;grid-template-columns:auto 1fr;gap:4px 16px}
.opt.rec{background:var(--accent-soft)}
.opt .tag{
  font-family:var(--mono);font-size:11.5px;font-weight:600;color:var(--accent);
  border:1px solid var(--accent);border-radius:3px;padding:2px 7px;height:fit-content;
}
.opt .on{font-weight:650;font-size:15px}
.opt .od{grid-column:2;color:var(--muted);font-size:14.5px;line-height:1.55}
footer{margin-top:64px;padding-top:26px;border-top:1px solid var(--line);
  font-size:13px;color:var(--muted)}
footer p{margin:0 0 8px}
@media (max-width:640px){
  body{font-size:16px}
  header.masthead{padding-top:44px}
  .steph{gap:12px}
}
@media (prefers-reduced-motion:no-preference){
  .pt,.bar{transition:none}
}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path("dlt_attribution_sensitivity"))
    ap.add_argument("--out", type=Path, default=Path("dlt_attribution_sensitivity/dlt_beslisrapport.html"))
    args = ap.parse_args()

    df = pd.read_csv(args.indir / "dlt_mitigation_summary.csv")
    meta = json.loads((args.indir / "dlt_mitigation_meta.json").read_text(encoding="utf-8"))
    n_sim = int(meta["n_sim"])

    crm = df[df["design"] == "TITE-CRM"].copy()
    ref63 = df[df["design"] == "6+3"].copy()
    scen = list(dict.fromkeys(df["scenario"]))

    agg = (crm.groupby("option", as_index=False)
              .agg(correct_pct=("correct_pct", "mean"),
                   too_high_pct=("too_high_pct", "mean"),
                   reach=("ever_reached_top_pct", "mean"),
                   acute=("mean_acute_tox", "mean")))
    agg["ord"] = agg["option"].map({o: i for i, o in enumerate(OPTION_ORDER)})
    agg = agg.sort_values("ord")

    a = agg[agg["option"] == "A"].iloc[0]
    e = agg[agg["option"] == "E"].iloc[0]
    h = agg[agg["option"] == "H"].iloc[0]
    c = agg[agg["option"] == "C"].iloc[0]
    low = crm[crm["scenario"] == "Acute low"]
    low_a = low[low["option"] == "A"].iloc[0]
    low_e = low[low["option"] == "E"].iloc[0]
    low_h = low[low["option"] == "H"].iloc[0]
    ref_low = ref63[ref63["scenario"] == "Acute low"].iloc[0]

    # ── per-scenario table (correct %) ───────────────────────────────────────
    rows = []
    for _, r in agg.iterrows():
        o = r["option"]
        cells = []
        for s in scen:
            v = crm[(crm["option"] == o) & (crm["scenario"] == s)]["correct_pct"].iloc[0]
            cells.append(f"<td>{fmt(v)}</td>")
        cls = ' class="rec"' if o == RECOMMENDED else ""
        rows.append(
            f'<tr{cls}><td>{SHORT[o]}</td>{"".join(cells)}'
            f'<td><b>{fmt(r["correct_pct"])}</b></td>'
            f'<td class="{"good" if r["too_high_pct"] <= a["too_high_pct"] + 4 else "bad"}">'
            f'{fmt(r["too_high_pct"])}</td></tr>')
    ref_cells = "".join(
        f'<td>{fmt(ref63[ref63["scenario"] == s]["correct_pct"].iloc[0])}</td>' for s in scen)
    rows.append(
        f'<tr><td>Huidig 6+3</td>{ref_cells}'
        f'<td><b>{fmt(ref63["correct_pct"].mean())}</b></td>'
        f'<td>{fmt(ref63["too_high_pct"].mean())}</td></tr>')

    head = "".join(f"<th>{s.replace('Acute ', '')}</th>" for s in scen)
    table = (f'<table><caption>Correcte MTD-selectie per scenario (%)</caption>'
             f'<thead><tr><th>Optie</th>{head}<th>Gemiddeld</th>'
             f'<th>Te hoog<br>(gem.)</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table>')

    optblocks = "".join(
        f'<div class="opt{" rec" if o == RECOMMENDED else ""}">'
        f'<span class="tag">{o}</span><span class="on">{SHORT[o]}'
        f'{" — advies" if o == RECOMMENDED else ""}</span>'
        f'<span class="od">{WHAT[o]}</span></div>'
        for o in OPTION_ORDER if o in set(crm["option"]))

    tradeoff = build_tradeoff_svg(agg)
    lowsvg = build_low_svg(low)

    # ── comparisons stated in prose, derived so they cannot drift from data ──
    def cmp_phrase(x, y, xl, yl):
        """Describe option x relative to option y on both axes."""
        d_acc = float(x["correct_pct"]) - float(y["correct_pct"])
        d_risk = float(x["too_high_pct"]) - float(y["too_high_pct"])
        acc = ("nauwkeuriger" if d_acc > 1.5 else
               "minder nauwkeurig" if d_acc < -1.5 else "even nauwkeurig")
        risk = ("veiliger" if d_risk < -1.5 else
                "riskanter" if d_risk > 1.5 else "even veilig")
        return (f"{xl} is {acc} dan {yl} "
                f"({fmt(x['correct_pct'])}% tegen {fmt(y['correct_pct'])}%) en {risk} "
                f"({fmt(x['too_high_pct'])}% tegen {fmt(y['too_high_pct'])}% kans op een te hoge MTD)")

    b = agg[agg["option"] == "B"].iloc[0]
    low_c = low[low["option"] == "C"].iloc[0]
    para5 = (
        f"Twee dingen vallen op. Ten eerste is de override vrijwel gratis: "
        f"{fmt(e['correct_pct'])}% correcte selectie tegen {fmt(a['correct_pct'])}% nu, bij "
        f"nauwelijks meer risico ({fmt(e['too_high_pct'])}% tegen {fmt(a['too_high_pct'])}%). "
        f"Ten tweede lost hij het Acute low probleem niet op — daar komt hij van "
        f"{fmt(low_a['correct_pct'])}% naar {fmt(low_e['correct_pct'])}%, en meer niet. "
        f"Alleen downweging beweegt dat scenario echt: 25% gewicht tilt Acute low naar "
        f"{fmt(low_c['correct_pct'])}%, maar bijna verdubbelt het risico op een te hoge MTD "
        f"({fmt(c['too_high_pct'])}% tegen {fmt(a['too_high_pct'])}%)."
    )

    scen_tbl = "".join(
        f"<tr><td>{k.replace('Acute ', '')}</td>"
        + "".join(f"<td>{v:.2f}</td>" for v in meta["scenarios"][k])
        + f"<td><b>L{max([i for i, p in enumerate(meta['scenarios'][k]) if p <= meta['target_acute']] or [0])}</b></td></tr>"
        for k in meta["scenarios"])

    html = f"""<title>Dosisescalatie na de eerste DLT</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style>
<div class="wrap">
<header class="masthead">
  <p class="kicker">Simulatieanalyse · dosisescalatie MERGE</p>
  <h1>Wat één betwiste DLT doet met het TITE-CRM ontwerp</h1>
  <p class="standfirst">De zesde patiënt op dosisniveau L1 kreeg een toxiciteit die als
  &ldquo;possible&rdquo; is beoordeeld. Dat ene gegeven verandert het gedrag van het
  voorgestelde model ingrijpend. Deze analyse laat zien hoe, wat de opties zijn, en welke
  we aanraden.</p>
  <div class="meta">
    <span>Simulaties <b>{n_sim}</b> per arm</span>
    <span>Doel acuut <b>{meta['target_acute']:.2f}</b></span>
    <span>Doel subacuut <b>{meta['target_subacute']:.2f}</b></span>
    <span>Configuratie <b>EWOC uit, burn-in uit</b></span>
  </div>
</header>

<section class="step"><div class="col">
  <div class="steph"><span class="num">1</span><h2>Het uitgangspunt is veranderd</h2></div>
  <p class="lede">Het amendement was opgezet als een overgang van het huidige 6+3 design
  naar TITE-CRM. Die keuze rustte op simulaties waarin de zes patiënten op L1 gestart waren
  zonder enige DLT. Dat uitgangspunt geldt niet meer.</p>
  <p>De toxiciteit bij de zesde patiënt is op het SAE-formulier beoordeeld als
  <strong>&ldquo;possible&rdquo;</strong> gerelateerd aan de behandeling — niet als
  &ldquo;unrelated&rdquo;, niet als &ldquo;definite&rdquo;. De behandelaars noemen
  ziekteprogressie de primaire differentiaaldiagnose en de dosis op de maag bleef ruim
  binnen de planningsconstraints, maar een verband met de bestraling is niet volledig
  uit te sluiten.</p>
  <p>Onder de gangbare conventie in fase 1-oncologie telt &ldquo;possible&rdquo; of hoger
  mee als DLT. Het model krijgt dus één DLT op zes patiënten op het laagste niveau te zien,
  voordat de eerste nieuwe patiënt is ingesloten.</p>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">2</span><h2>Waarom dat het model zo raakt</h2></div>
  <p>De CRM schat de hele dosis-toxiciteitscurve met <strong>één gedeelde parameter</strong>.
  Er is geen aparte schatting per dosisniveau. Eén DLT op L1 verhoogt daardoor ook de geschatte
  toxiciteit van L3 en L4 — niveaus waar nog geen enkele patiënt is behandeld. En die verhoging
  blijft de hele trial staan.</p>
  <p>Het 6+3 design werkt fundamenteel anders. Bij één DLT op zes patiënten escaleert het niet,
  maar het stopt ook niet: het breidt uit naar negen patiënten, en escaleert alsnog als er geen
  tweede DLT bij komt. Daarna wordt het volgende niveau beoordeeld op <em>zijn eigen</em> cohort.
  De DLT op L1 speelt geen rol meer.</p>
  <div class="pull">Het 6+3 design heeft een kort geheugen per dosisniveau. De CRM heeft
  één lang geheugen over alle niveaus heen. Dat lange geheugen is normaal juist de kracht —
  hier verspreidt het een waarschijnlijk niet-behandelgerelateerd event over de hele curve.</div>
  <p>Dit is geen fout in de code. Het model doet precies wat het hoort te doen met de data die
  we het geven. De vraag is dus niet hoe we het model repareren, maar hoe dit ene gegeven
  het model in hoort te gaan.</p>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">3</span><h2>Het effect is niet overal negatief</h2></div>
  <p>De DLT maakt het model <strong>conservatiever</strong>. Of dat goed of slecht uitpakt,
  hangt volledig af van waar de werkelijke MTD ligt — en dat weten we niet. Vandaar de vijf
  scenario&rsquo;s.</p>
  <p>Waar de werkelijke MTD op het hoogste niveau ligt (Acute low) is de schade groot: de
  correcte MTD-selectie zakt van {fmt(low_h['correct_pct'])}% naar
  <strong>{fmt(low_a['correct_pct'])}%</strong>. Maar waar de werkelijke MTD laag ligt,
  verbetert het model juist, omdat het voorheen te hoog uitkwam. En in álle scenario&rsquo;s
  daalt de kans op een te hoge, onveilige MTD.</p>
  <p>Dat is precies het gedrag dat je van een veiligheidsontwerp wilt zien na een waargenomen
  toxiciteit. Het probleem is dus specifiek, niet algemeen: het zit in het scenario waarin de
  hoogste dosis werkelijk de juiste is.</p>
</div>

<figure>
  <div class="figbox">{lowsvg}</div>
  <figcaption><strong>Het Acute low scenario, waar de werkelijke MTD L4 (5&times;8 Gy) is.</strong>
  De lichte balk laat zien in hoeveel simulaties L4 überhaupt bereikt wordt tijdens de trial;
  de volle balk in hoeveel simulaties L4 ook als finale MTD gekozen wordt. Bij de huidige
  instelling wordt L4 in {fmt(low_a['ever_reached_top_pct'],0)}% van de trials nog bereikt,
  maar slechts in {fmt(low_a['correct_pct'],0)}% ook gekozen. De escalatie-override tilt het
  bereiken naar {fmt(low_e['ever_reached_top_pct'],0)}% — de finale keuze blijft echter
  modelgestuurd en volgt maar deels.</figcaption>
</figure>
</section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">4</span><h2>Zeven manieren om ermee om te gaan</h2></div>
  <p>We hebben zeven varianten doorgerekend, over alle vijf scenario&rsquo;s. Ze vallen in
  twee families: de data anders <em>wegen</em>, of een <em>ontwerpregel</em> toevoegen.</p>
</div>
<div class="optlist">{optblocks}</div>
<div class="col">
  <h3>De escalatie-override, in gewone taal</h3>
  <p>Als drie patiënten op het huidige dosisniveau hun volledige acute follow-up hebben
  afgerond zonder DLT, mag er één niveau omhoog — ook als het model dat nog niet aanraadt.
  Dat is exact de logica die het 6+3 design al heeft, en die de METC dus al heeft goedgekeurd.
  De DLT blijft volledig meetellen; nieuwe schone data op een niveau kan zich alleen omhoog
  verdienen.</p>
  <p>Twee grendels zorgen dat de regel de veiligheid nooit kan ondermijnen. Hij vuurt
  <strong>alleen als het model wil blijven staan</strong> — een de-escalatie wordt nooit
  overruled. En hij vuurt <strong>alleen als op dat niveau geen enkele acute DLT is
  waargenomen</strong>; één toxiciteit blokkeert hem volledig. Escaleren gaat altijd met
  precies één niveau tegelijk, net als de-escaleren.</p>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">5</span><h2>Er is geen gratis oplossing</h2></div>
  <p>Elke variant die het Acute low scenario herstelt, kost nauwkeurigheid in de scenario&rsquo;s
  waar de werkelijke MTD laag ligt, en verhoogt de kans op een te hoge dosis. Die uitruil is
  onvermijdelijk: de DLT is echte informatie. Hoe minder gewicht je hem geeft, hoe vaker je te
  hoog uitkomt als de toxiciteit werkelijk hoog is.</p>
</div>

<figure>
  <div class="figbox">{tradeoff}</div>
  <figcaption><strong>De uitruil, gemiddeld over alle vijf scenario&rsquo;s.</strong>
  Naar boven is nauwkeuriger, naar links is veiliger — linksboven is dus het beste.
  De punten liggen vrijwel op één oplopende lijn: elke stap nauwkeuriger kost veiligheid.
  De override (links) is de enige uitzondering — hij schuift omhoog zonder noemenswaardig
  naar rechts te gaan, en is daarmee praktisch gratis. Alle andere winst wordt gekocht:
  van {fmt(a['too_high_pct'])}% risico bij vol gewicht tot {fmt(h['too_high_pct'])}% wanneer
  de DLT helemaal niet meetelt.</figcaption>
</figure>

<div class="tablewrap">{table}</div>
<div class="col">
  <p style="margin-top:18px">{para5}</p>
  <div class="callout">
    <p class="ct">Waarom de override Acute low niet oplost</p>
    <p>De override bepaalt welke doses je <em>onderweg</em> uitprobeert, niet welke je aan het
    eind kiest. In Acute low bereikt hij L4 in {fmt(low_e['ever_reached_top_pct'],0)}% van de
    trials tegen {fmt(low_a['ever_reached_top_pct'],0)}% nu — een forse verbetering in
    exploratie. Maar de finale MTD-keuze is puur modelgestuurd, en dat model blijft door de
    DLT omlaag getrokken. Vandaar dat de correcte selectie maar van
    {fmt(low_a['correct_pct'])}% naar {fmt(low_e['correct_pct'])}% gaat.</p>
    <p>Wie de finale keuze wil verschuiven, moet het gewicht van de DLT in het model
    aanpassen. Dat is geen ontwerpvraag meer maar een inhoudelijke: hoe zwaar weegt dit
    ene event?</p>
  </div>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">6</span><h2>Advies: twee losse beslissingen</h2></div>
  <p class="lede">Het is verleidelijk hier één knop te zoeken, maar er liggen twee
  onafhankelijke vragen. De eerste is technisch en heeft een duidelijk antwoord. De tweede
  is inhoudelijk en hoort bij het team, niet bij de statistiek.</p>

  <div class="callout rec">
    <p class="ct">Beslissing 1 — neem de escalatie-override op</p>
    <p>Vrijwel gratis: {fmt(e['correct_pct'])}% correcte selectie tegen
    {fmt(a['correct_pct'])}% nu, bij een risicotoename van {fmt(a['too_high_pct'])}% naar
    slechts {fmt(e['too_high_pct'])}%. De regel overrulet nooit een de-escalatie en vuurt
    nooit op een niveau waar toxiciteit is gezien, dus hij kan de veiligheid niet
    ondermijnen. En hij is goed uit te leggen: inhoudelijk dezelfde waarborg die het
    6+3 design al heeft.</p>
  </div>

  <div class="callout">
    <p class="ct">Beslissing 2 — hoe zwaar weegt deze ene DLT?</p>
    <p>Hier zit de echte uitruil, en die is niet statistisch op te lossen. Het SAE-formulier
    zegt &ldquo;possible&rdquo;, met ziekteprogressie als primaire differentiaaldiagnose.
    Hoe sterk dat oordeel doorwerkt in het model is een klinische keuze:</p>
    <ul>
      <li><strong>Volledig gewicht</strong> — Acute low blijft op
      {fmt(low_e['correct_pct'])}%, risico op een te hoge MTD {fmt(e['too_high_pct'])}%.
      De veiligste optie.</li>
      <li><strong>Half gewicht</strong> — Acute low naar
      {fmt(low[low['option'] == 'E+B'].iloc[0]['correct_pct'])}%, risico naar
      {fmt(agg[agg['option'] == 'E+B'].iloc[0]['too_high_pct'])}%. Sluit aan bij een
      causaliteitsoordeel dat het ongeveer fifty-fifty houdt.</li>
      <li><strong>Kwart gewicht</strong> — Acute low naar
      {fmt(low[low['option'] == 'E+C'].iloc[0]['correct_pct'])}%, risico naar
      {fmt(agg[agg['option'] == 'E+C'].iloc[0]['too_high_pct'])}%. Alleen verdedigbaar als
      het team het event overwegend aan progressie toeschrijft.</li>
    </ul>
    <p>Onze rol stopt bij het zichtbaar maken van deze uitruil. Wat een acceptabel risico op
    een te hoge MTD is, is een oordeel van het studieteam en uiteindelijk van de METC.</p>
  </div>

  <h3>Voor het amendement betekent dit</h3>
  <p>Het amendement beschrijft dan niet alleen de overgang van 6+3 naar TITE-CRM, maar ook
  expliciet hoe de bestaande L1-data meegaat en welke escalatiewaarborg daarbij hoort — met
  de gekozen weging onderbouwd vanuit de causaliteitsbeoordeling. Dat is een sterker stuk dan
  het oorspronkelijke voorstel, omdat het het scenario behandelt dat zich daadwerkelijk heeft
  voorgedaan.</p>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">7</span><h2>Wat nog open staat</h2></div>
  <p>Drie punten die eerlijk op tafel moeten, voordat hier een protocoltekst van wordt gemaakt.</p>
  <div class="callout">
    <p class="ct">1 · Onze code en die van Sama geven andere uitkomsten</p>
    <p>Alle cijfers hier komen uit onze eigen simulator. Voor het 6+3 design vinden wij in
    Acute low {fmt(ref_low['correct_pct'])}% correcte selectie, waar Sama&rsquo;s slides
    32,3% laten zien. Ook de trialduur verschilt sterk. De vergelijking <em>binnen</em> onze
    code is betrouwbaar — daar verandert per variant maar één ding — maar de vergelijking
    tussen beide implementaties is dat nog niet. Zodra Sama&rsquo;s R-code beschikbaar is,
    zouden we die regel voor regel naast de onze moeten leggen.</p>
  </div>
  <div class="callout">
    <p class="ct">2 · De override is onze eigen constructie</p>
    <p>De regel is conceptueel dezelfde waarborg die het 6+3 design heeft, maar het is geen
    gepubliceerde methode die we uit de literatuur overnemen. Sama zou hier statistisch naar
    moeten kijken, en het is verstandig te controleren of vergelijkbare lokale
    escalatieregels in de CRM-literatuur beschreven zijn voordat dit een protocoltekst wordt.</p>
  </div>
  <div class="callout">
    <p class="ct">3 · De causaliteitsbeoordeling verdient een formele plek</p>
    <p>Of een SAE een protocol-gedefinieerde DLT is voor escalatiebeslissingen, is formeel een
    andere vraag dan de causaliteitsclassificatie op het SAE-formulier. Het is verstandig die
    beoordeling expliciet te beleggen en vast te leggen — welke kant het ook op valt. Zoals
    hierboven blijkt verandert het het advies niet, maar het maakt het dossier wel navolgbaar.</p>
  </div>
</div></section>

<section class="step"><div class="col">
  <div class="steph"><span class="num">8</span><h2>De scenario&rsquo;s</h2></div>
  <p>De vijf scenario&rsquo;s voor de werkelijke acute toxiciteit, met de subacute kansen
  vast op {", ".join(f"{v:.2f}" for v in meta['true_subacute'])}. De werkelijke MTD is het
  hoogste niveau met een acute toxiciteit op of onder het doel van {meta['target_acute']:.2f}.</p>
</div>
<div class="tablewrap">
  <table><caption>Werkelijke acute toxiciteitskansen per scenario</caption>
  <thead><tr><th>Scenario</th>{"".join(f"<th>L{i}<br>{lab}</th>" for i, lab in enumerate(meta["dose_labels"]))}<th>Werkelijke<br>MTD</th></tr></thead>
  <tbody>{scen_tbl}</tbody></table>
</div></section>

<footer><div class="col">
  <p>Gegenereerd uit <span style="font-family:var(--mono);font-size:12px">dlt_mitigation_analysis.py</span>
  ({n_sim} simulaties per scenario per variant, seed {meta['seed']}). Configuratie: TITE-CRM
  zonder EWOC en zonder burn-in, start op L2, maximaal 30 patiënten inclusief de zes bestaande,
  cohortgrootte 3, één nieuwe patiënt per vier weken.</p>
  <p>Deze pagina bevat uitsluitend geaggregeerde simulatie-uitkomsten. Er zijn geen
  patiëntgegevens in opgenomen.</p>
</div></footer>
</div>
"""
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
