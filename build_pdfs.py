#!/usr/bin/env python3
"""Render the DLT decision report and the code-comparison checklist to PDF.

Both documents go to the same meeting, so they share one print identity: the
clinical teal palette of the HTML report, set for paper (A4, light only,
tabular figures, tables that do not split across pages mid-row).

Usage:
    python build_pdfs.py [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

CHROMIUM = "/opt/pw-browsers/chromium"

PRINT_CSS = """
@page { size: A4; margin: 17mm 15mm 16mm; }
html, body { background:#fff !important; }
body {
  color:#101B26; font-size:10.2pt; line-height:1.5;
  font-family:-apple-system,"Segoe UI",system-ui,Arial,sans-serif;
  -webkit-print-color-adjust:exact; print-color-adjust:exact;
}
h1,h2,h3,h4 { break-after:avoid; page-break-after:avoid; }
table, figure, .callout, .opt, .figbox { break-inside:avoid; page-break-inside:avoid; }
tr, li, p { break-inside:avoid; page-break-inside:avoid; }
a { color:#0E7C86; text-decoration:none; }
"""

# Print overrides for the generated report page: it is designed for screen with a
# 1080px measure, so tighten the frame and drop the card chrome for paper.
REPORT_PRINT_CSS = PRINT_CSS + """
.wrap { max-width:100% !important; padding:0 !important; }
.col { max-width:100% !important; }
header.masthead { padding-top:0 !important; }
section.step { padding-top:26px !important; }
h1 { font-size:25pt !important; }
h2 { font-size:15pt !important; }
.standfirst { font-size:11.5pt !important; }
body { font-size:10.2pt !important; }
svg.chart { min-width:0 !important; }
.figbox { padding:10px 6px 4px !important; }
table { min-width:0 !important; font-size:8.6pt !important; }
th, td { padding:5px 8px !important; }
figcaption { font-size:8.8pt !important; }
.tablewrap, .figbox, .callout, .optlist { box-shadow:none !important; }
"""

MD_CSS = """
:root{
  --ink:#101B26; --slate:#1E3A52; --muted:#5C6B7A;
  --line:#DCE4EA; --line-soft:#EDF1F4; --accent:#0E7C86; --accent-soft:#F0F7F8;
  --risk:#B4472F; --safe:#2F7D5B;
}
body{ max-width:none; margin:0; }
h1{
  font-family:Georgia,"Palatino Linotype",serif; font-weight:400;
  font-size:21pt; line-height:1.18; letter-spacing:-.01em;
  margin:0 0 10px; color:var(--ink);
  text-wrap:balance; word-break:keep-all;
}
h1 + p { color:var(--muted); font-size:11pt; margin:0 0 4px; }
h2{
  font-family:Georgia,"Palatino Linotype",serif; font-weight:400; font-size:15pt;
  margin:26px 0 10px; padding-top:12px; border-top:1px solid var(--line);
  color:var(--ink); text-wrap:balance;
}
h3{
  font-size:10.8pt; font-weight:650; margin:18px 0 7px; color:var(--slate);
  letter-spacing:.01em;
}
p{ margin:0 0 9px; }
ul,ol{ margin:0 0 10px; padding-left:19px; }
li{ margin:0 0 5px; }
strong{ font-weight:650; }
em{ color:var(--slate); }
hr{ border:none; border-top:1px solid var(--line); margin:22px 0; }
code{
  font-family:ui-monospace,"SF Mono",Consolas,monospace; font-size:9pt;
  background:var(--accent-soft); padding:1px 4px; border-radius:2px; color:var(--slate);
}
table{
  border-collapse:collapse; width:100%; font-size:8.8pt; margin:12px 0 16px;
  font-variant-numeric:tabular-nums;
}
th,td{ border-bottom:1px solid var(--line-soft); padding:5px 9px; text-align:left;
  vertical-align:top; }
thead th{
  font-family:ui-monospace,"SF Mono",Consolas,monospace; font-size:7.8pt;
  letter-spacing:.06em; text-transform:uppercase; color:var(--muted);
  font-weight:500; border-bottom:1px solid var(--line); background:var(--accent-soft);
}
tbody tr:last-child td{ border-bottom:1px solid var(--line); }
blockquote{
  margin:14px 0; padding:2px 0 2px 15px; border-left:2px solid var(--accent);
  color:var(--slate);
}
"""


def md_to_html(md_path: Path, title: str) -> str:
    body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "sane_lists", "attr_list"],
    )
    return (f"<!doctype html><html lang='nl'><head><meta charset='utf-8'>"
            f"<title>{title}</title><style>{PRINT_CSS}{MD_CSS}</style></head>"
            f"<body>{body}</body></html>")


def render(page, url: str, out: Path, extra_css: str) -> None:
    page.goto(url)
    page.emulate_media(media="print", color_scheme="light")
    if extra_css:
        page.add_style_tag(content=extra_css)
    page.wait_for_timeout(500)
    page.pdf(path=str(out), format="A4", print_background=True,
             margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("dlt_attribution_sensitivity"))
    args = ap.parse_args()
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parent
    checklist_md = root / "CODE_COMPARISON_CHECKLIST.md"
    report_html = root / "dlt_attribution_sensitivity" / "dlt_beslisrapport.html"
    nonbinary_html = root / "dlt_attribution_sensitivity" / "nonbinary_dlt_report.html"
    comparison_html = root / "dlt_attribution_sensitivity" / "simulator_comparison.html"
    switch_html = root / "switch_decision" / "switch_to_titecrm.html"

    tmp = args.outdir / "_checklist_print.html"
    tmp.write_text(md_to_html(checklist_md, "Vergelijkingschecklist"), encoding="utf-8")

    out_report = args.outdir / "DLT_beslisrapport.pdf"
    out_check = args.outdir / "Vergelijkingschecklist_Sama.pdf"
    out_nb = args.outdir / "Non-binary_DLT_note.pdf"
    out_cmp = args.outdir / "Simulator_comparison.pdf"
    out_switch = root / "switch_decision" / "Switch_to_TITE-CRM.pdf"

    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROMIUM)
        pg = b.new_page()
        render(pg, report_html.as_uri(), out_report, REPORT_PRINT_CSS)
        render(pg, nonbinary_html.as_uri(), out_nb, REPORT_PRINT_CSS)
        render(pg, comparison_html.as_uri(), out_cmp, REPORT_PRINT_CSS)
        render(pg, switch_html.as_uri(), out_switch, REPORT_PRINT_CSS)
        render(pg, tmp.as_uri(), out_check, "")
        b.close()

    tmp.unlink(missing_ok=True)
    for f in (out_report, out_nb, out_cmp, out_switch, out_check):
        print(f"Wrote {f} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
