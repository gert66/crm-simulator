"""
trial_conduct_app.py — Real-patient TITE dual-endpoint dose-escalation trial conduct tool
==========================================================================================
Operational tool for entering real patients and computing CRM-based dose recommendations.
NOT a simulation tool — enter actual patient data, get actual recommendations.

Tox1 = acute toxicity   (window starts at RT start = inclusion_date + incl_to_rt)
Tox2 = subacute toxicity (window starts at surgery_date; surgery patients only)

TITE weights are computed from real calendar dates using the reference date
(default: today).  Joint EWOC safety: P(tox1 > target) < alpha AND
P(tox2 > target) < alpha.
"""

from __future__ import annotations

import json
import csv
import io
import datetime
import os
from typing import Any, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trial Conduct — TITE CRM",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  [data-testid="stSidebar"]        { display: none; }
  [data-testid="stSidebarNav"]     { display: none; }
  [data-testid="collapsedControl"] { display: none; }
  .block-container { padding-top: 2.2rem; padding-bottom: 0.5rem; }
  .element-container { margin-bottom: 0.10rem; }
  [data-testid="stMetric"]           { padding: 0.15rem 0 0.05rem 0 !important; }
  [data-testid="stMetricLabel"]      { font-size: 0.78rem !important; }
  [data-testid="stMetricValue"]      { font-size: 1.05rem !important; line-height:1.2 !important; }
</style>
""", unsafe_allow_html=True)

DEFAULT_SAVE_FILE = "trial_conduct_data.json"

# ==============================================================================
# Helpers
# ==============================================================================

def safe_probs(x: Any) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)


def fig_to_png_bytes(fig: plt.Figure) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def compact_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)


# ==============================================================================
# Prior skeleton (ported from sim_tite.py)
# ==============================================================================

def dfcrm_getprior(
    halfwidth: float,
    target: float,
    nu: int,
    nlevel: int,
    model: str = "empiric",
    intcpt: float = 3.0,
) -> np.ndarray:
    halfwidth = float(halfwidth); target = float(target)
    nu = int(nu); nlevel = int(nlevel); intcpt = float(intcpt)
    if not (0 < target < 1):
        raise ValueError("target must be in (0,1).")
    if halfwidth <= 0:
        raise ValueError("halfwidth must be > 0.")
    if (target - halfwidth) <= 0 or (target + halfwidth) >= 1:
        raise ValueError("halfwidth too large: target ± halfwidth must stay in (0,1).")
    if not (1 <= nu <= nlevel):
        raise ValueError("nu must be between 1 and nlevel.")
    dosescaled = np.full(nlevel, np.nan, dtype=float)
    if model == "empiric":
        dosescaled[nu - 1] = target
        for k in range(nu, 1, -1):
            b_k = np.log(np.log(target + halfwidth) / np.log(dosescaled[k - 1]))
            dosescaled[k - 2] = np.exp(np.log(target - halfwidth) / np.exp(b_k))
        for k in range(nu, nlevel):
            b_k1 = np.log(np.log(target - halfwidth) / np.log(dosescaled[k - 1]))
            dosescaled[k] = np.exp(np.log(target + halfwidth) / np.exp(b_k1))
        return dosescaled
    if model == "logistic":
        dosescaled[nu - 1] = np.log(target / (1 - target)) - intcpt
        for k in range(nu, 1, -1):
            b_k = np.log(
                (np.log((target + halfwidth) / (1 - target - halfwidth)) - intcpt)
                / dosescaled[k - 1]
            )
            dosescaled[k - 2] = (
                np.log((target - halfwidth) / (1 - target + halfwidth)) - intcpt
            ) / np.exp(b_k)
        for k in range(nu, nlevel):
            b_k1 = np.log(
                (np.log((target - halfwidth) / (1 - target + halfwidth)) - intcpt)
                / dosescaled[k - 1]
            )
            dosescaled[k] = (
                np.log((target + halfwidth) / (1 - target - halfwidth)) - intcpt
            ) / np.exp(b_k1)
        return (1 + np.exp(-intcpt - dosescaled)) ** (-1)
    raise ValueError('model must be "empiric" or "logistic".')


# ==============================================================================
# CRM posterior (GH quadrature, fractional n accepted)
# ==============================================================================

def posterior_via_gh(
    sigma: float,
    skeleton: np.ndarray,
    n_per: np.ndarray,
    dlt_per: np.ndarray,
    gh_n: int = 61,
) -> tuple[np.ndarray, np.ndarray]:
    sk = safe_probs(skeleton)
    n  = np.asarray(n_per,   dtype=float)
    y  = np.asarray(dlt_per, dtype=float)
    x, w = np.polynomial.hermite.hermgauss(int(gh_n))
    theta = float(sigma) * np.sqrt(2.0) * x
    P  = sk[None, :] ** np.exp(theta)[:, None]
    P  = safe_probs(P)
    ll = (y[None, :] * np.log(P) + (n[None, :] - y[None, :]) * np.log(1 - P)).sum(axis=1)
    log_unnorm = np.log(w) + ll
    m          = np.max(log_unnorm)
    unnorm     = np.exp(log_unnorm - m)
    post_w     = unnorm / np.sum(unnorm)
    return post_w, P


def crm_posterior_summaries(
    sigma: float,
    skeleton: np.ndarray,
    n_per: np.ndarray,
    dlt_per: np.ndarray,
    target: float,
    gh_n: int = 61,
) -> tuple[np.ndarray, np.ndarray]:
    post_w, P     = posterior_via_gh(sigma, skeleton, n_per, dlt_per, gh_n=gh_n)
    post_mean     = (post_w[:, None] * P).sum(axis=0)
    overdose_prob = (post_w[:, None] * (P > float(target))).sum(axis=0)
    return post_mean, overdose_prob


# ==============================================================================
# Date helpers
# ==============================================================================

def _to_date(val: Any) -> Optional[datetime.date]:
    """Convert string / date / None to datetime.date or None."""
    if val is None or val == "" or val != val:  # None, empty str, NaN
        return None
    if isinstance(val, datetime.date):
        return val
    if isinstance(val, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.datetime.strptime(val, fmt).date()
            except ValueError:
                pass
    return None


def _date_str(d: Optional[datetime.date]) -> str:
    return d.isoformat() if d is not None else ""


# ==============================================================================
# TITE weight computation from real dates
# ==============================================================================

def compute_tox1_weight(
    row: dict,
    reference_date: datetime.date,
    setup: dict,
) -> float:
    """
    Tox1 window starts at inclusion_date + incl_to_rt days.
    weight = 1 if event observed, 1 if window complete, fraction if partial, 0 if not started.
    """
    inc = _to_date(row.get("inclusion_date"))
    if inc is None:
        return 0.0
    tox1_win: int = int(setup.get("tox1_win", 84))
    incl_to_rt: int = int(setup.get("incl_to_rt", 21))

    rt_start = inc + datetime.timedelta(days=incl_to_rt)
    win_end  = rt_start + datetime.timedelta(days=tox1_win)

    tox1_event = str(row.get("tox1_event", "Unknown")).strip()
    tox1_date  = _to_date(row.get("tox1_date"))

    # Event observed before reference date → weight 1
    if tox1_event == "Yes" and tox1_date is not None and tox1_date <= reference_date:
        return 1.0

    if reference_date < rt_start:
        return 0.0
    if reference_date >= win_end:
        return 1.0

    elapsed = (reference_date - rt_start).days
    return float(np.clip(elapsed / tox1_win, 0.0, 1.0))


def compute_tox2_weight(
    row: dict,
    reference_date: datetime.date,
    setup: dict,
) -> float:
    """
    Tox2 window starts at surgery_date.
    Returns 0 if surgery_done is False or surgery_date is missing.
    """
    if not row.get("surgery_done", False):
        return 0.0
    surg_date = _to_date(row.get("surgery_date"))
    if surg_date is None:
        return 0.0

    tox2_win: int = int(setup.get("tox2_win", 30))
    win_end = surg_date + datetime.timedelta(days=tox2_win)

    tox2_event = str(row.get("tox2_event", "Unknown")).strip()
    tox2_date  = _to_date(row.get("tox2_date"))

    if tox2_event == "Yes" and tox2_date is not None and tox2_date <= reference_date:
        return 1.0

    if reference_date < surg_date:
        return 0.0
    if reference_date >= win_end:
        return 1.0

    elapsed = (reference_date - surg_date).days
    return float(np.clip(elapsed / tox2_win, 0.0, 1.0))


def compute_followup_weights(
    rows: list[dict],
    ref_date: datetime.date,
    setup: dict,
) -> list[dict]:
    """
    Return augmented rows with tox1_weight and tox2_weight added.
    """
    augmented = []
    for row in rows:
        r = dict(row)
        r["tox1_weight"] = compute_tox1_weight(row, ref_date, setup)
        r["tox2_weight"] = compute_tox2_weight(row, ref_date, setup)
        augmented.append(r)
    return augmented


# ==============================================================================
# Data summarisation: aggregate TITE weights per dose level
# ==============================================================================

def summarize_current_data(
    aug_rows: list[dict],
    dose_labels: list[str],
    setup: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate TITE-weighted n and event counts per dose level.

    Returns (n1, y1, n2, y2) each of shape (n_levels,).
    n1/y1 — tox1 (all patients)
    n2/y2 — tox2 (surgery patients only)
    """
    n_levels = len(dose_labels)
    n1 = np.zeros(n_levels, dtype=float)
    y1 = np.zeros(n_levels, dtype=float)
    n2 = np.zeros(n_levels, dtype=float)
    y2 = np.zeros(n_levels, dtype=float)

    label_to_idx = {lbl: i for i, lbl in enumerate(dose_labels)}

    for row in aug_rows:
        lbl = str(row.get("dose_level", "")).strip()
        if lbl not in label_to_idx:
            continue
        d = label_to_idx[lbl]

        w1 = float(row.get("tox1_weight", 0.0))
        n1[d] += w1
        if str(row.get("tox1_event", "")).strip() == "Yes":
            y1[d] += 1.0

        if row.get("surgery_done", False):
            w2 = float(row.get("tox2_weight", 0.0))
            n2[d] += w2
            if str(row.get("tox2_event", "")).strip() == "Yes":
                y2[d] += 1.0

    return n1, y1, n2, y2


# ==============================================================================
# CRM recommendation engine
# ==============================================================================

def compute_crm_recommendation(
    n1: np.ndarray,
    y1: np.ndarray,
    n2: np.ndarray,
    y2: np.ndarray,
    current_dose_idx: int,
    setup: dict,
) -> dict:
    """
    Compute CRM posteriors and recommended next dose.

    Returns a dict with keys:
      pm1, od1, pm2, od2  — posterior means and OD probs per dose
      allowed_doses        — list of admissible dose indices
      recommended_dose     — index of recommended next dose
      recommended_label    — dose label string
      hold_recommended     — bool: recommend holding (bridge only) if True
      hold_reason          — str explaining why hold is recommended
      n1, y1, n2, y2       — input arrays (for reporting)
    """
    n_levels   = len(setup["dose_labels"])
    target1    = float(setup.get("target_t1",  0.15))
    target2    = float(setup.get("target_t2",  0.33))
    sigma      = float(setup.get("sigma",      1.0))
    ewoc_on    = bool(setup.get("ewoc_on",     True))
    ewoc_alpha = float(setup.get("ewoc_alpha", 0.25))
    max_step   = int(setup.get("max_step",     1))
    gh_n       = int(setup.get("gh_n",         61))
    guardrail  = bool(setup.get("enforce_guardrail", True))
    skel1      = np.asarray(setup["skeleton_t1"], dtype=float)
    skel2      = np.asarray(setup["skeleton_t2"], dtype=float)

    pm1, od1 = crm_posterior_summaries(sigma, skel1, n1, y1, target1, gh_n=gh_n)
    pm2, od2 = crm_posterior_summaries(sigma, skel2, n2, y2, target2, gh_n=gh_n)

    # Admissible doses: both OD probs < ewoc_alpha
    if ewoc_on:
        allowed = [d for d in range(n_levels)
                   if od1[d] < ewoc_alpha and od2[d] < ewoc_alpha]
    else:
        allowed = list(range(n_levels))

    if not allowed:
        allowed = [0]

    # Guardrail: cannot skip untried doses
    if guardrail:
        tried = [d for d in range(n_levels) if n1[d] > 0]
        highest_tried = max(tried) if tried else -1
        allowed = [d for d in allowed if d <= highest_tried + 1]
        if not allowed:
            allowed = [0]

    # Max step constraint
    lo = current_dose_idx - max_step
    hi = current_dose_idx + max_step
    allowed_stepped = [d for d in allowed if lo <= d <= hi]
    if not allowed_stepped:
        # Relax to any admissible dose ≤ current
        allowed_stepped = [d for d in allowed if d <= current_dose_idx]
        if not allowed_stepped:
            allowed_stepped = [allowed[0]]

    # Closest to target1 posterior mean among allowed
    pm1_target_dist = [abs(pm1[d] - target1) for d in allowed_stepped]
    rec_idx = allowed_stepped[int(np.argmin(pm1_target_dist))]

    # Hold logic: recommend hold if we WANT to escalate but tox2 data is sparse
    # at the target dose (n2 < 3 effective patients at doses >= rec_idx)
    hold_recommended = False
    hold_reason = ""
    n2_at_rec = float(n2[rec_idx]) if rec_idx < len(n2) else 0.0
    if rec_idx > current_dose_idx and n2_at_rec < 1.0:
        hold_recommended = True
        hold_reason = (
            f"Escalation to {setup['dose_labels'][rec_idx]} desired by CRM, but "
            f"effective subacute follow-up at that level is only {n2_at_rec:.1f} "
            f"(< 1.0). Recommend holding at current dose until more subacute data "
            f"accumulate."
        )

    return {
        "pm1":               pm1.tolist(),
        "od1":               od1.tolist(),
        "pm2":               pm2.tolist(),
        "od2":               od2.tolist(),
        "allowed_doses":     allowed_stepped,
        "recommended_dose":  rec_idx,
        "recommended_label": setup["dose_labels"][rec_idx],
        "hold_recommended":  hold_recommended,
        "hold_reason":       hold_reason,
        "n1":                n1.tolist(),
        "y1":                y1.tolist(),
        "n2":                n2.tolist(),
        "y2":                y2.tolist(),
    }


# ==============================================================================
# Human-readable rationale
# ==============================================================================

def build_decision_rationale(rec: dict, setup: dict) -> str:
    """Return a multi-line human-readable explanation of the recommendation."""
    lines: list[str] = []
    dose_labels: list[str] = setup["dose_labels"]
    n_levels = len(dose_labels)
    target1    = float(setup.get("target_t1",  0.15))
    target2    = float(setup.get("target_t2",  0.33))
    ewoc_alpha = float(setup.get("ewoc_alpha", 0.25))
    ewoc_on    = bool(setup.get("ewoc_on",     True))

    lines.append("### CRM Decision Rationale\n")

    # Per-dose table
    lines.append("**Posterior summary per dose level:**\n")
    header = "| Dose | n1 (eff) | y1 | P(tox1) | OD1 | n2 (eff) | y2 | P(tox2) | OD2 | Status |"
    sep    = "|------|----------|-----|---------|-----|----------|-----|---------|-----|--------|"
    lines.append(header)
    lines.append(sep)

    pm1 = rec["pm1"]; od1 = rec["od1"]
    pm2 = rec["pm2"]; od2 = rec["od2"]
    n1  = rec["n1"];  y1  = rec["y1"]
    n2  = rec["n2"];  y2  = rec["y2"]
    allowed = rec["allowed_doses"]

    for d in range(n_levels):
        if ewoc_on:
            admissible = (od1[d] < ewoc_alpha) and (od2[d] < ewoc_alpha)
        else:
            admissible = True
        status = "**ADMISSIBLE**" if admissible else "EXCLUDED"
        if d == rec["recommended_dose"]:
            status += " ← **RECOMMENDED**"
        lines.append(
            f"| {dose_labels[d]} "
            f"| {n1[d]:.2f} | {y1[d]:.0f} "
            f"| {pm1[d]:.3f} | {od1[d]:.3f} "
            f"| {n2[d]:.2f} | {y2[d]:.0f} "
            f"| {pm2[d]:.3f} | {od2[d]:.3f} "
            f"| {status} |"
        )

    lines.append("")
    lines.append(
        f"**Targets:** tox1 = {target1:.2f}, tox2 = {target2:.2f} "
        + (f"| **EWOC α = {ewoc_alpha:.2f}**" if ewoc_on else "| EWOC OFF")
    )
    lines.append("")

    rec_lbl = rec["recommended_label"]
    if rec["hold_recommended"]:
        lines.append(f"**Recommendation: HOLD at current dose**")
        lines.append(f"> {rec['hold_reason']}")
    else:
        lines.append(f"**Recommendation: proceed to {rec_lbl}**")

    return "\n".join(lines)


# ==============================================================================
# Plots
# ==============================================================================

def make_plots(
    rec: dict,
    setup: dict,
    dose_history: list[str],
) -> plt.Figure:
    """
    Three-panel figure:
      A — posterior mean P(tox) per dose vs targets (both endpoints)
      B — OD probabilities per dose vs EWOC threshold
      C — cumulative effective follow-up (n1, n2) per dose
    """
    dose_labels = setup["dose_labels"]
    n_levels    = len(dose_labels)
    x           = np.arange(n_levels)
    target1     = float(setup.get("target_t1",  0.15))
    target2     = float(setup.get("target_t2",  0.33))
    ewoc_alpha  = float(setup.get("ewoc_alpha", 0.25))
    ewoc_on     = bool(setup.get("ewoc_on",     True))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    # ── Panel A: posterior mean P(tox) ────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, rec["pm1"], "o-", color="#2196F3", label="P(tox1)")
    ax.plot(x, rec["pm2"], "s--", color="#FF5722", label="P(tox2)")
    ax.axhline(target1, color="#2196F3", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.axhline(target2, color="#FF5722", linewidth=0.8, linestyle=":", alpha=0.7)
    # Mark recommended dose
    rd = rec["recommended_dose"]
    ax.axvline(rd, color="green", linewidth=1.2, linestyle="--", alpha=0.6, label="Recommended")
    ax.set_xticks(x); ax.set_xticklabels(dose_labels, fontsize=7, rotation=20)
    ax.set_ylabel("Posterior mean P(tox)", fontsize=8)
    ax.set_title("A: Posterior toxicity", fontsize=9)
    ax.legend(fontsize=7)
    compact_style(ax)

    # ── Panel B: OD probabilities ──────────────────────────────────────────────
    ax = axes[1]
    ax.bar(x - 0.18, rec["od1"], 0.35, color="#2196F3", alpha=0.75, label="OD prob tox1")
    ax.bar(x + 0.18, rec["od2"], 0.35, color="#FF5722", alpha=0.75, label="OD prob tox2")
    if ewoc_on:
        ax.axhline(ewoc_alpha, color="red", linewidth=1.2, linestyle="--", label=f"EWOC α={ewoc_alpha}")
    ax.axvline(rd, color="green", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(dose_labels, fontsize=7, rotation=20)
    ax.set_ylabel("P(overdose)", fontsize=8)
    ax.set_title("B: EWOC safety", fontsize=9)
    ax.legend(fontsize=7)
    compact_style(ax)

    # ── Panel C: effective follow-up ──────────────────────────────────────────
    ax = axes[2]
    ax.bar(x - 0.18, rec["n1"], 0.35, color="#2196F3", alpha=0.75, label="n1 eff (tox1)")
    ax.bar(x + 0.18, rec["n2"], 0.35, color="#FF5722", alpha=0.75, label="n2 eff (tox2)")
    ax.set_xticks(x); ax.set_xticklabels(dose_labels, fontsize=7, rotation=20)
    ax.set_ylabel("Effective patients (TITE)", fontsize=8)
    ax.set_title("C: Follow-up accumulation", fontsize=9)
    ax.legend(fontsize=7)
    compact_style(ax)

    fig.tight_layout()
    return fig


# ==============================================================================
# Persistence helpers
# ==============================================================================

def _serialize_row(row: dict) -> dict:
    """Convert date objects to ISO strings for JSON serialisation."""
    out = {}
    for k, v in row.items():
        if isinstance(v, datetime.date):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def _deserialize_row(row: dict) -> dict:
    """Convert ISO date strings back to datetime.date objects."""
    date_keys = {"inclusion_date", "surgery_date", "tox1_date",
                 "tox2_date", "last_followup_date"}
    out = {}
    for k, v in row.items():
        if k in date_keys and isinstance(v, str) and v:
            try:
                out[k] = datetime.date.fromisoformat(v)
            except ValueError:
                out[k] = None
        else:
            out[k] = v
    return out


def initialize_default_state() -> dict:
    """Return a fresh app state dict with empty patient list and default setup."""
    today = datetime.date.today()
    return {
        "setup": {
            "trial_name":        "New Trial",
            "dose_labels":       ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"],
            "current_dose_label":"5×5 Gy",
            "reference_date":    today.isoformat(),
            # Timing
            "incl_to_rt":        21,
            "rt_dur":            14,
            "rt_to_surg":        84,
            "tox1_win":          84,
            "tox2_win":          30,
            # CRM targets
            "target_t1":         0.15,
            "target_t2":         0.33,
            # Priors tox1
            "prior_model":       "empiric",
            "prior_target_t1":   0.15,
            "halfwidth_t1":      0.10,
            "prior_nu_t1":       3,
            # Priors tox2
            "prior_target_t2":   0.33,
            "halfwidth_t2":      0.10,
            "prior_nu_t2":       3,
            # CRM knobs
            "sigma":             1.0,
            "ewoc_on":           True,
            "ewoc_alpha":        0.25,
            "max_step":          1,
            "gh_n":              61,
            "enforce_guardrail": True,
            # Skeletons (computed at setup time)
            "skeleton_t1":       None,
            "skeleton_t2":       None,
        },
        "patient_rows": [],
        "dose_history":  [],
    }


def load_trial_state(path: str) -> tuple[dict, Optional[str]]:
    """Load JSON state from path. Returns (state, error_string)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Deserialize rows
        raw["patient_rows"] = [_deserialize_row(r) for r in raw.get("patient_rows", [])]
        # Deserialize reference_date in setup
        setup = raw.get("setup", {})
        if isinstance(setup.get("reference_date"), str) and setup["reference_date"]:
            try:
                setup["reference_date"] = datetime.date.fromisoformat(setup["reference_date"])
            except ValueError:
                setup["reference_date"] = datetime.date.today()
        return raw, None
    except FileNotFoundError:
        return initialize_default_state(), None
    except Exception as exc:
        return initialize_default_state(), str(exc)


def save_trial_state(state: dict, path: str) -> Optional[str]:
    """Save state dict to JSON. Returns error string or None on success."""
    try:
        out = dict(state)
        out["patient_rows"] = [_serialize_row(r) for r in state.get("patient_rows", [])]
        setup_out = dict(state.get("setup", {}))
        ref = setup_out.get("reference_date")
        if isinstance(ref, datetime.date):
            setup_out["reference_date"] = ref.isoformat()
        out["setup"] = setup_out
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        return None
    except Exception as exc:
        return str(exc)


def export_patient_table_csv(rows: list[dict]) -> str:
    """Return patient table as CSV string."""
    if not rows:
        return ""
    fieldnames = [
        "patient_id", "inclusion_date", "dose_level",
        "surgery_done", "surgery_date",
        "tox1_event", "tox1_date",
        "tox2_event", "tox2_date",
        "last_followup_date", "notes",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        ser = _serialize_row(row)
        writer.writerow({k: ser.get(k, "") for k in fieldnames})
    return buf.getvalue()


def export_decision_report(
    rec: dict,
    rows: list[dict],
    setup: dict,
    ref_date: datetime.date,
) -> str:
    """Return plain-text decision report."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"TRIAL CONDUCT DECISION REPORT")
    lines.append(f"Trial: {setup.get('trial_name', 'Unnamed')}")
    lines.append(f"Reference date: {ref_date.isoformat()}")
    lines.append(f"Generated: {datetime.date.today().isoformat()}")
    lines.append("=" * 60)
    lines.append(f"\nTotal patients entered: {len(rows)}")
    lines.append(f"Current dose level: {setup.get('current_dose_label', '?')}")
    lines.append(f"Recommended next dose: {rec.get('recommended_label', '?')}")
    if rec.get("hold_recommended"):
        lines.append(f"\n*** HOLD RECOMMENDED ***")
        lines.append(rec.get("hold_reason", ""))
    lines.append("\nPer-dose summary:")
    dose_labels = setup.get("dose_labels", [])
    for d, lbl in enumerate(dose_labels):
        lines.append(
            f"  {lbl}: n1={rec['n1'][d]:.2f} y1={rec['y1'][d]:.0f} "
            f"P(tox1)={rec['pm1'][d]:.3f} OD1={rec['od1'][d]:.3f} | "
            f"n2={rec['n2'][d]:.2f} y2={rec['y2'][d]:.0f} "
            f"P(tox2)={rec['pm2'][d]:.3f} OD2={rec['od2'][d]:.3f}"
        )
    lines.append("")
    return "\n".join(lines)


# ==============================================================================
# Validate patient table rows
# ==============================================================================

def validate_patient_table(rows: list[dict]) -> list[str]:
    """Return list of warning strings (empty = no problems)."""
    warnings: list[str] = []
    for i, row in enumerate(rows):
        pid = row.get("patient_id", f"row {i+1}")
        if _to_date(row.get("inclusion_date")) is None:
            warnings.append(f"{pid}: missing inclusion_date")
        if not str(row.get("dose_level", "")).strip():
            warnings.append(f"{pid}: missing dose_level")
        if row.get("surgery_done") and _to_date(row.get("surgery_date")) is None:
            warnings.append(f"{pid}: surgery_done=True but surgery_date missing")
        if str(row.get("tox1_event", "")).strip() == "Yes" and _to_date(row.get("tox1_date")) is None:
            warnings.append(f"{pid}: tox1_event=Yes but tox1_date missing")
        if str(row.get("tox2_event", "")).strip() == "Yes" and _to_date(row.get("tox2_date")) is None:
            warnings.append(f"{pid}: tox2_event=Yes but tox2_date missing")
    return warnings


# ==============================================================================
# Session-state bootstrap
# ==============================================================================

def _bootstrap() -> None:
    """Load state from default save file if not yet loaded this session."""
    if "_state_loaded" in st.session_state:
        return

    state, err = load_trial_state(DEFAULT_SAVE_FILE)
    if err:
        st.warning(f"Error loading {DEFAULT_SAVE_FILE}: {err}. Starting fresh.")
        state = initialize_default_state()

    st.session_state["_trial_state"]    = state
    st.session_state["_save_path"]      = DEFAULT_SAVE_FILE
    st.session_state["_unsaved"]        = False
    st.session_state["_last_saved"]     = (
        datetime.datetime.now().strftime("%H:%M:%S")
        if os.path.exists(DEFAULT_SAVE_FILE) else None
    )
    st.session_state["_last_rec"]       = None
    st.session_state["_state_loaded"]   = True


_bootstrap()

# Shorthand accessors
def _state() -> dict:
    return st.session_state["_trial_state"]

def _setup() -> dict:
    return _state()["setup"]

def _rows() -> list[dict]:
    return _state()["patient_rows"]


# ==============================================================================
# Auto-compute skeletons from current setup
# ==============================================================================

def _recompute_skeletons() -> None:
    """Rebuild skeleton arrays from current prior parameters. Store in setup."""
    s = _setup()
    n = len(s.get("dose_labels", []))
    if n == 0:
        return
    try:
        s["skeleton_t1"] = dfcrm_getprior(
            halfwidth=s["halfwidth_t1"],
            target=s["prior_target_t1"],
            nu=int(s["prior_nu_t1"]),
            nlevel=n,
            model=s.get("prior_model", "empiric"),
            intcpt=float(s.get("logistic_intcpt", 3.0)),
        ).tolist()
    except Exception:
        s["skeleton_t1"] = safe_probs(np.linspace(0.05, 0.6, n)).tolist()
    try:
        s["skeleton_t2"] = dfcrm_getprior(
            halfwidth=s["halfwidth_t2"],
            target=s["prior_target_t2"],
            nu=int(s["prior_nu_t2"]),
            nlevel=n,
            model=s.get("prior_model", "empiric"),
            intcpt=float(s.get("logistic_intcpt", 3.0)),
        ).tolist()
    except Exception:
        s["skeleton_t2"] = safe_probs(np.linspace(0.05, 0.6, n)).tolist()


if _setup().get("skeleton_t1") is None:
    _recompute_skeletons()


# ==============================================================================
# Status bar
# ==============================================================================

def _render_status_bar() -> None:
    path      = st.session_state.get("_save_path", DEFAULT_SAVE_FILE)
    unsaved   = st.session_state.get("_unsaved", False)
    last_save = st.session_state.get("_last_saved")
    indicator = "🔴 Unsaved changes" if unsaved else "✅ Saved"
    saved_str = f"Last saved: {last_save}" if last_save else "Not yet saved"
    st.caption(f"📁 **{path}** &nbsp;|&nbsp; {indicator} &nbsp;|&nbsp; {saved_str}")


def _do_autosave() -> None:
    path = st.session_state.get("_save_path", DEFAULT_SAVE_FILE)
    err  = save_trial_state(_state(), path)
    if err:
        st.warning(f"Autosave failed: {err}")
    else:
        st.session_state["_unsaved"]    = False
        st.session_state["_last_saved"] = datetime.datetime.now().strftime("%H:%M:%S")


# ==============================================================================
# Page header
# ==============================================================================

st.title("Trial Conduct — TITE Dual-Endpoint CRM")
st.caption(
    "Operational dose-escalation tool. Enter real patients, compute CRM recommendations. "
    "Data is auto-saved after each update."
)
_render_status_bar()

# ==============================================================================
# Section 1: Study Setup
# ==============================================================================

with st.expander("Study Setup", expanded=(_setup().get("skeleton_t1") is None)):
    s = _setup()

    st.markdown("#### Identification")
    new_name = st.text_input("Trial name", value=s.get("trial_name", "New Trial"), key="_ui_trial_name")
    if new_name != s.get("trial_name"):
        s["trial_name"] = new_name
        st.session_state["_unsaved"] = True

    st.markdown("#### Dose levels")
    raw_labels = st.text_input(
        "Dose labels (comma-separated)",
        value=", ".join(s.get("dose_labels", [])),
        help="Enter dose level labels in escalating order, e.g. '5×4 Gy, 5×5 Gy, 5×6 Gy'",
        key="_ui_dose_labels",
    )
    parsed_labels = [x.strip() for x in raw_labels.split(",") if x.strip()]
    if parsed_labels and parsed_labels != s.get("dose_labels"):
        s["dose_labels"] = parsed_labels
        if s.get("current_dose_label") not in parsed_labels:
            s["current_dose_label"] = parsed_labels[0]
        st.session_state["_unsaved"] = True

    dose_labels_now = s.get("dose_labels", ["L1"])
    cur_idx = dose_labels_now.index(s["current_dose_label"]) if s.get("current_dose_label") in dose_labels_now else 0
    cur_sel = st.selectbox(
        "Current dose level (most recently treated)",
        options=dose_labels_now,
        index=cur_idx,
        key="_ui_cur_dose",
    )
    if cur_sel != s.get("current_dose_label"):
        s["current_dose_label"] = cur_sel
        st.session_state["_unsaved"] = True

    st.markdown("#### Timing parameters (days)")
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    def _int_input(col, label, key_s, min_val=1, max_val=365, step=1):
        val = col.number_input(label, min_value=min_val, max_value=max_val,
                               value=int(s.get(key_s, 21)), step=step, key=f"_ui_{key_s}")
        if val != s.get(key_s):
            s[key_s] = int(val)
            st.session_state["_unsaved"] = True

    _int_input(tc1, "Inclusion → RT start", "incl_to_rt")
    _int_input(tc2, "RT duration",          "rt_dur")
    _int_input(tc3, "RT end → surgery",     "rt_to_surg")
    _int_input(tc4, "Tox1 window",          "tox1_win")
    _int_input(tc5, "Tox2 window",          "tox2_win")

    st.markdown("#### CRM targets & priors")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**Tox1 (acute)**")
        v = st.number_input("Target P(tox1)", 0.01, 0.99, float(s.get("target_t1", 0.15)), 0.01,
                            key="_ui_target_t1")
        if v != s.get("target_t1"):
            s["target_t1"] = float(v); st.session_state["_unsaved"] = True
        v = st.number_input("Prior target", 0.01, 0.99, float(s.get("prior_target_t1", 0.15)), 0.01,
                            key="_ui_prior_target_t1")
        if v != s.get("prior_target_t1"):
            s["prior_target_t1"] = float(v); st.session_state["_unsaved"] = True
        v = st.number_input("Halfwidth", 0.01, 0.49, float(s.get("halfwidth_t1", 0.10)), 0.01,
                            key="_ui_halfwidth_t1")
        if v != s.get("halfwidth_t1"):
            s["halfwidth_t1"] = float(v); st.session_state["_unsaved"] = True
        v = st.number_input("Prior ν (reference dose, 1-indexed)", 1, len(dose_labels_now),
                            int(np.clip(s.get("prior_nu_t1", 3), 1, len(dose_labels_now))), 1,
                            key="_ui_prior_nu_t1")
        if v != s.get("prior_nu_t1"):
            s["prior_nu_t1"] = int(v); st.session_state["_unsaved"] = True
    with pc2:
        st.markdown("**Tox2 (subacute)**")
        v = st.number_input("Target P(tox2)", 0.01, 0.99, float(s.get("target_t2", 0.33)), 0.01,
                            key="_ui_target_t2")
        if v != s.get("target_t2"):
            s["target_t2"] = float(v); st.session_state["_unsaved"] = True
        v = st.number_input("Prior target", 0.01, 0.99, float(s.get("prior_target_t2", 0.33)), 0.01,
                            key="_ui_prior_target_t2")
        if v != s.get("prior_target_t2"):
            s["prior_target_t2"] = float(v); st.session_state["_unsaved"] = True
        v = st.number_input("Halfwidth", 0.01, 0.49, float(s.get("halfwidth_t2", 0.10)), 0.01,
                            key="_ui_halfwidth_t2")
        if v != s.get("halfwidth_t2"):
            s["halfwidth_t2"] = float(v); st.session_state["_unsaved"] = True
        v = st.number_input("Prior ν (reference dose, 1-indexed)", 1, len(dose_labels_now),
                            int(np.clip(s.get("prior_nu_t2", 3), 1, len(dose_labels_now))), 1,
                            key="_ui_prior_nu_t2")
        if v != s.get("prior_nu_t2"):
            s["prior_nu_t2"] = int(v); st.session_state["_unsaved"] = True

    st.markdown("#### CRM algorithm settings")
    ck1, ck2, ck3, ck4 = st.columns(4)
    with ck1:
        v = st.number_input("σ (model variance)", 0.1, 5.0, float(s.get("sigma", 1.0)), 0.1,
                            key="_ui_sigma")
        if v != s.get("sigma"):
            s["sigma"] = float(v); st.session_state["_unsaved"] = True
    with ck2:
        v = st.checkbox("EWOC", value=bool(s.get("ewoc_on", True)), key="_ui_ewoc_on")
        if v != s.get("ewoc_on"):
            s["ewoc_on"] = bool(v); st.session_state["_unsaved"] = True
        v2 = st.number_input("EWOC α", 0.01, 0.99, float(s.get("ewoc_alpha", 0.25)), 0.01,
                             key="_ui_ewoc_alpha")
        if v2 != s.get("ewoc_alpha"):
            s["ewoc_alpha"] = float(v2); st.session_state["_unsaved"] = True
    with ck3:
        v = st.number_input("Max step", 1, 4, int(s.get("max_step", 1)), 1,
                            key="_ui_max_step")
        if v != s.get("max_step"):
            s["max_step"] = int(v); st.session_state["_unsaved"] = True
    with ck4:
        v = st.checkbox("Guardrail (no skipping)", value=bool(s.get("enforce_guardrail", True)),
                        key="_ui_guardrail")
        if v != s.get("enforce_guardrail"):
            s["enforce_guardrail"] = bool(v); st.session_state["_unsaved"] = True

    if st.button("Apply setup & rebuild skeletons"):
        _recompute_skeletons()
        st.session_state["_unsaved"] = True
        st.success("Skeletons rebuilt.")

    # Show computed skeletons
    if s.get("skeleton_t1") is not None:
        sk1 = [f"{v:.4f}" for v in s["skeleton_t1"]]
        sk2 = [f"{v:.4f}" for v in s["skeleton_t2"]]
        st.caption(f"Skeleton tox1: {sk1}")
        st.caption(f"Skeleton tox2: {sk2}")

# ==============================================================================
# Section 2: Patient Data
# ==============================================================================

with st.expander("Patient Data", expanded=True):
    st.markdown(
        "Enter one row per patient. Dates in YYYY-MM-DD format. "
        "**tox1_event / tox2_event**: `Yes`, `No`, or `Unknown`."
    )

    _empty_row: dict = {
        "patient_id":         "",
        "inclusion_date":     None,
        "dose_level":         (_setup().get("dose_labels") or [""])[0],
        "surgery_done":       False,
        "surgery_date":       None,
        "tox1_event":         "Unknown",
        "tox1_date":          None,
        "tox2_event":         "Unknown",
        "tox2_date":          None,
        "last_followup_date": None,
        "notes":              "",
    }

    import pandas as pd

    existing_rows = _rows()
    if not existing_rows:
        existing_rows = [dict(_empty_row)]

    df_edit = pd.DataFrame(existing_rows)

    # Ensure all columns present
    for col, default in _empty_row.items():
        if col not in df_edit.columns:
            df_edit[col] = default

    dose_labels_now = _setup().get("dose_labels", ["L1"])

    edited_df = st.data_editor(
        df_edit,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "patient_id":   st.column_config.TextColumn("Patient ID"),
            "inclusion_date": st.column_config.DateColumn("Inclusion date", format="YYYY-MM-DD"),
            "dose_level":   st.column_config.SelectboxColumn(
                "Dose level", options=dose_labels_now, required=True
            ),
            "surgery_done":  st.column_config.CheckboxColumn("Surgery?"),
            "surgery_date":  st.column_config.DateColumn("Surgery date", format="YYYY-MM-DD"),
            "tox1_event":   st.column_config.SelectboxColumn(
                "Tox1 event", options=["Yes", "No", "Unknown"], required=True
            ),
            "tox1_date":    st.column_config.DateColumn("Tox1 date", format="YYYY-MM-DD"),
            "tox2_event":   st.column_config.SelectboxColumn(
                "Tox2 event", options=["Yes", "No", "Unknown"], required=True
            ),
            "tox2_date":    st.column_config.DateColumn("Tox2 date", format="YYYY-MM-DD"),
            "last_followup_date": st.column_config.DateColumn("Last f/u", format="YYYY-MM-DD"),
            "notes":        st.column_config.TextColumn("Notes"),
        },
        key="_patient_editor",
    )

    # Sync edited dataframe back to state
    new_rows = edited_df.to_dict(orient="records")
    # Convert pandas NaT / NaN to None and dates
    cleaned_rows: list[dict] = []
    for row in new_rows:
        r: dict = {}
        for k, v in row.items():
            if hasattr(v, "isnull") and v.isnull():
                r[k] = None
            elif hasattr(v, "item"):          # numpy scalar
                r[k] = v.item()
            elif v is pd.NaT:
                r[k] = None
            elif isinstance(v, float) and np.isnan(v):
                r[k] = None
            else:
                r[k] = v
        cleaned_rows.append(r)

    if cleaned_rows != _rows():
        _state()["patient_rows"] = cleaned_rows
        st.session_state["_unsaved"] = True

    # Validation warnings
    warns = validate_patient_table(cleaned_rows)
    if warns:
        with st.expander(f"⚠️ {len(warns)} data warning(s)", expanded=False):
            for w in warns:
                st.warning(w)

# ==============================================================================
# Section 3: Decision Engine
# ==============================================================================

with st.expander("Decision Engine", expanded=True):
    col_ref, col_btn = st.columns([2, 1])
    with col_ref:
        ref_raw = _setup().get("reference_date", datetime.date.today())
        if isinstance(ref_raw, str):
            try:
                ref_raw = datetime.date.fromisoformat(ref_raw)
            except ValueError:
                ref_raw = datetime.date.today()
        ref_date_ui = st.date_input(
            "Reference date (today or data cut-off date)",
            value=ref_raw,
            key="_ui_ref_date",
        )
        if ref_date_ui != ref_raw:
            _setup()["reference_date"] = ref_date_ui
            st.session_state["_unsaved"] = True

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Update recommendation", type="primary", use_container_width=True)

    if run_btn:
        s = _setup()
        rows = _rows()

        # Rebuild skeletons if needed
        if s.get("skeleton_t1") is None:
            _recompute_skeletons()

        ref_date: datetime.date = s["reference_date"] if isinstance(s["reference_date"], datetime.date) \
            else datetime.date.fromisoformat(str(s["reference_date"]))

        aug_rows = compute_followup_weights(rows, ref_date, s)
        n1, y1, n2, y2 = summarize_current_data(aug_rows, s["dose_labels"], s)

        cur_lbl = s.get("current_dose_label", s["dose_labels"][0])
        cur_idx = s["dose_labels"].index(cur_lbl) if cur_lbl in s["dose_labels"] else 0

        rec = compute_crm_recommendation(n1, y1, n2, y2, cur_idx, s)

        # Append to dose history
        _state().setdefault("dose_history", [])
        entry = {
            "date": ref_date.isoformat(),
            "recommended": rec["recommended_label"],
            "current": cur_lbl,
        }
        _state()["dose_history"].append(entry)

        st.session_state["_last_rec"] = {
            "rec":      rec,
            "aug_rows": aug_rows,
            "ref_date": ref_date,
        }
        _do_autosave()
        st.rerun()

# ==============================================================================
# Section 4: Output Summary
# ==============================================================================

last_rec_state = st.session_state.get("_last_rec")

if last_rec_state is not None:
    rec      = last_rec_state["rec"]
    aug_rows = last_rec_state["aug_rows"]
    ref_date = last_rec_state["ref_date"]
    s        = _setup()

    with st.expander("Recommendation & Rationale", expanded=True):
        # Top-line summary metrics
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Current dose",     s.get("current_dose_label", "?"))
        mc2.metric("Recommended dose", rec["recommended_label"])
        n_pts = len(_rows())
        n_tox1 = int(sum(1 for r in _rows() if str(r.get("tox1_event","")).strip() == "Yes"))
        n_tox2 = int(sum(1 for r in _rows() if str(r.get("tox2_event","")).strip() == "Yes"))
        mc3.metric("Total patients", n_pts)
        mc4.metric("Tox1 / Tox2 events", f"{n_tox1} / {n_tox2}")

        if rec.get("hold_recommended"):
            st.error(f"**HOLD RECOMMENDED**: {rec['hold_reason']}")
        else:
            st.success(f"Recommended next dose: **{rec['recommended_label']}**")

        st.markdown(build_decision_rationale(rec, s))

    # ── Plots ─────────────────────────────────────────────────────────────────
    with st.expander("Safety & Follow-up Plots", expanded=True):
        dose_hist_labels = [e["recommended"] for e in _state().get("dose_history", [])]
        fig = make_plots(rec, s, dose_hist_labels)
        st.image(fig_to_png_bytes(fig), use_container_width=True)

    # ── Follow-up weight table ─────────────────────────────────────────────────
    with st.expander("Per-patient follow-up weights", expanded=False):
        wt_rows = []
        for r in aug_rows:
            wt_rows.append({
                "patient_id":    r.get("patient_id", ""),
                "inclusion_date": _date_str(_to_date(r.get("inclusion_date"))),
                "dose_level":    r.get("dose_level", ""),
                "tox1_event":    r.get("tox1_event", ""),
                "tox1_weight":   round(float(r.get("tox1_weight", 0)), 4),
                "surgery_done":  r.get("surgery_done", False),
                "tox2_event":    r.get("tox2_event", ""),
                "tox2_weight":   round(float(r.get("tox2_weight", 0)), 4),
            })
        st.dataframe(pd.DataFrame(wt_rows), use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    with st.expander("Export", expanded=False):
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            csv_str = export_patient_table_csv(_rows())
            st.download_button(
                "Export patient table to CSV",
                data=csv_str,
                file_name=f"patients_{datetime.date.today().isoformat()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with ecol2:
            report_str = export_decision_report(rec, _rows(), s, ref_date)
            st.download_button(
                "Export decision report (TXT)",
                data=report_str,
                file_name=f"decision_report_{datetime.date.today().isoformat()}.txt",
                mime="text/plain",
                use_container_width=True,
            )

# ==============================================================================
# Section 5: File management
# ==============================================================================

with st.expander("Save / Load", expanded=False):
    fm1, fm2, fm3, fm4 = st.columns(4)

    with fm1:
        if st.button("Save current trial data", use_container_width=True):
            path = st.session_state.get("_save_path", DEFAULT_SAVE_FILE)
            err  = save_trial_state(_state(), path)
            if err:
                st.error(f"Save failed: {err}")
            else:
                st.session_state["_unsaved"]    = False
                st.session_state["_last_saved"] = datetime.datetime.now().strftime("%H:%M:%S")
                st.success(f"Saved to {path}")

    with fm2:
        new_path = st.text_input("New file path", value=DEFAULT_SAVE_FILE, key="_ui_saveas_path",
                                 label_visibility="collapsed",
                                 placeholder="Path for save-as…")
        if st.button("Save as new file", use_container_width=True):
            if new_path.strip():
                err = save_trial_state(_state(), new_path.strip())
                if err:
                    st.error(f"Save-as failed: {err}")
                else:
                    st.session_state["_save_path"]  = new_path.strip()
                    st.session_state["_unsaved"]    = False
                    st.session_state["_last_saved"] = datetime.datetime.now().strftime("%H:%M:%S")
                    st.success(f"Saved as {new_path.strip()}")
            else:
                st.warning("Enter a file path first.")

    with fm3:
        load_path = st.text_input("Load from path", value=DEFAULT_SAVE_FILE, key="_ui_load_path",
                                  label_visibility="collapsed",
                                  placeholder="Path to load…")
        if st.button("Load saved trial data", use_container_width=True):
            if load_path.strip():
                new_state, err = load_trial_state(load_path.strip())
                if err:
                    st.error(f"Load failed: {err}")
                else:
                    st.session_state["_trial_state"] = new_state
                    st.session_state["_save_path"]   = load_path.strip()
                    st.session_state["_unsaved"]     = False
                    st.session_state["_last_saved"]  = datetime.datetime.now().strftime("%H:%M:%S")
                    st.session_state["_last_rec"]    = None
                    if new_state.get("setup", {}).get("skeleton_t1") is None:
                        _recompute_skeletons()
                    st.success(f"Loaded from {load_path.strip()}")
                    st.rerun()
            else:
                st.warning("Enter a file path first.")

    with fm4:
        if st.button("Reset to defaults", use_container_width=True):
            fresh = initialize_default_state()
            st.session_state["_trial_state"] = fresh
            st.session_state["_unsaved"]     = True
            st.session_state["_last_rec"]    = None
            _recompute_skeletons()
            st.rerun()

# Update status bar at the bottom
_render_status_bar()
