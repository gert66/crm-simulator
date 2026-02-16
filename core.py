# core.py
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


@dataclass
class Defaults:
    # Essentials
    target: float = 0.15
    start_dose_level: int = 1  # 1-based
    n_sims: int = 500
    seed: int = 123

    # Sample size / cohorts
    max_n_6p3: int = 27
    max_n_crm: int = 27
    cohort_size: int = 3

    # Prior info already observed at start dose (dose = start_dose_level), with 0 DLT
    n_prior_start_no_dlt: int = 0

    # Prior playground
    skeleton_model: str = "empiric"  # "empiric" or "logistic"
    prior_target: float = 0.15
    delta: float = 0.10
    prior_mtd: int = 3  # 1-based
    logistic_intercept: float = 0.0

    # CRM knobs
    prior_sigma_theta: float = 1.0
    burnin_until_first_dlt: bool = True
    ewoc_enable: bool = False
    ewoc_alpha: float = 0.25

    # True curve
    true_curve: Tuple[float, ...] = (0.01, 0.02, 0.12, 0.20, 0.35)


DEFAULTS = Defaults()


# -------------------------
# Session state management
# -------------------------
def _apply_defaults() -> None:
    s = st.session_state
    for k, v in asdict(DEFAULTS).items():
        if k == "true_curve":
            s["true_curve"] = list(v)
        else:
            s[k] = v

    # widget keys for true curve inputs
    _sync_true_curve_widget_keys()

    # housekeeping
    s["results"] = None
    s["results_meta"] = None
    s["last_error"] = None
    s["is_running"] = False


def reset_to_defaults() -> None:
    # do NOT delete all keys (that tends to cause weird cross-page behavior)
    st.session_state["_do_reset"] = True
    st.rerun()


def init_state() -> None:
    """
    Single source of truth.
    Called at top of every page. Never overwrites user values unless a reset was requested.
    """
    s = st.session_state

    # first ever init
    if not s.get("_initialized", False):
        _apply_defaults()
        s["_initialized"] = True

    # reset requested
    if s.get("_do_reset", False):
        s["_do_reset"] = False
        _apply_defaults()

    # setdefault for any missing keys (safe across page nav)
    for k, v in asdict(DEFAULTS).items():
        if k == "true_curve":
            s.setdefault("true_curve", list(v))
        else:
            s.setdefault(k, v)

    s.setdefault("results", None)
    s.setdefault("results_meta", None)
    s.setdefault("last_error", None)
    s.setdefault("is_running", False)
    s.setdefault("cohort_size_6p3", DEFAULTS.cohort_size_6p3)


    # keep true curve widget keys alive if user edited curve length/values
    _sync_true_curve_widget_keys()


def _sync_true_curve_widget_keys() -> None:
    s = st.session_state
    tc = list(s.get("true_curve", []))
    # ensure widget keys exist and match current true_curve values
    for i, v in enumerate(tc):
        key = f"true_p_{i}"
        if key not in s:
            s[key] = float(v)


def sync_true_curve_from_widgets() -> None:
    s = st.session_state
    tc = list(s.get("true_curve", []))
    if not tc:
        return
    new_vals: List[float] = []
    for i in range(len(tc)):
        key = f"true_p_{i}"
        v = s.get(key, tc[i])
        try:
            new_vals.append(float(v))
        except Exception:
            new_vals.append(float(tc[i]))
    s["true_curve"] = new_vals


# -------------------------
# Helpers: labels + skeleton
# -------------------------
def dose_labels(n: int) -> List[str]:
    base = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]
    if n <= len(base):
        return [f"L{i} {base[i]}" for i in range(n)]
    return [f"L{i}" for i in range(n)]


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def build_empiric_skeleton(n_dose: int, prior_target: float, delta: float, prior_mtd_1based: int) -> np.ndarray:
    m = int(prior_mtd_1based) - 1
    m = max(0, min(n_dose - 1, m))
    vals = [prior_target + (i - m) * delta for i in range(n_dose)]
    sk = np.array([_clamp(v, 0.0001, 0.9999) for v in vals], dtype=float)
    return np.maximum.accumulate(sk)


def build_logistic_skeleton(n_dose: int, prior_target: float, prior_mtd_1based: int, intercept: float) -> np.ndarray:
    m = int(prior_mtd_1based) - 1
    m = max(0, min(n_dose - 1, m))
    logit = np.log(prior_target / (1.0 - prior_target))
    b = 0.25 if m == 0 else (logit - intercept) / m
    xs = np.arange(n_dose, dtype=float)
    z = intercept + b * xs
    sk = 1.0 / (1.0 + np.exp(-z))
    return np.maximum.accumulate(sk)


def get_skeleton_from_state(n_dose: int) -> np.ndarray:
    s = st.session_state
    model = s.get("skeleton_model", DEFAULTS.skeleton_model)
    prior_target = float(s.get("prior_target", DEFAULTS.prior_target))
    prior_mtd = int(s.get("prior_mtd", DEFAULTS.prior_mtd))

    if model == "logistic":
        intercept = float(s.get("logistic_intercept", DEFAULTS.logistic_intercept))
        return build_logistic_skeleton(n_dose, prior_target, prior_mtd, intercept)

    delta = float(s.get("delta", DEFAULTS.delta))
    return build_empiric_skeleton(n_dose, prior_target, delta, prior_mtd)


def plot_true_vs_prior(true_p: List[float], prior_p: np.ndarray, target: float, true_mtd_idx0: int) -> plt.Figure:
    n = len(true_p)
    xs = np.arange(n)
    fig = plt.figure(figsize=(4.6, 2.6), dpi=140)
    ax = fig.add_subplot(111)
    ax.plot(xs, true_p, marker="o", linewidth=1.5, label="True P(DLT)")
    ax.plot(xs, prior_p, marker="o", linewidth=1.5, label="Prior (skeleton)")
    ax.axhline(target, linewidth=1.0)
    ax.axvline(true_mtd_idx0, linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{i}" for i in xs], fontsize=8)
    ax.set_ylabel("Probability", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


# -------------------------
# Sim runner
# -------------------------
def _import_simplesim():
    if "" not in sys.path:
        sys.path.insert(0, "")
    return importlib.import_module("simplesim")


def build_payload() -> Dict[str, Any]:
    s = st.session_state

    payload = {
        "target": float(s.get("target", DEFAULTS.target)),
        "start_dose_level": int(s.get("start_dose_level", DEFAULTS.start_dose_level)) - 1,
        "n_sims": int(s.get("n_sims", DEFAULTS.n_sims)),
        "seed": int(s.get("seed", DEFAULTS.seed)),

        # sample sizes
        "max_n_6p3": int(s.get("max_n_6p3", DEFAULTS.max_n_6p3)),
        "cohort_size_6p3": int(s.get("cohort_size_6p3", DEFAULTS.cohort_size_6p3)),
        "max_n_crm": int(s.get("max_n_crm", getattr(DEFAULTS, "max_n_crm", DEFAULTS.max_n_6p3))),

        # prior info
        "n_prior_start_no_dlt": int(s.get("n_prior_start_no_dlt", DEFAULTS.n_prior_start_no_dlt)),

        # curves / prior playground
        "true_curve": list(map(float, s.get("true_curve", list(DEFAULTS.true_curve)))),
        "skeleton_model": s.get("skeleton_model", DEFAULTS.skeleton_model),
        "prior_target": float(s.get("prior_target", DEFAULTS.prior_target)),
        "delta": float(s.get("delta", DEFAULTS.delta)),
        "prior_mtd_1based": int(s.get("prior_mtd", DEFAULTS.prior_mtd)),
        "logistic_intercept": float(s.get("logistic_intercept", DEFAULTS.logistic_intercept)),

        # crm knobs
        "prior_sigma_theta": float(s.get("prior_sigma_theta", DEFAULTS.prior_sigma_theta)),
        "burnin_until_first_dlt": bool(s.get("burnin_until_first_dlt", DEFAULTS.burnin_until_first_dlt)),
        "ewoc_enable": bool(s.get("ewoc_enable", DEFAULTS.ewoc_enable)),
        "ewoc_alpha": float(s.get("ewoc_alpha", DEFAULTS.ewoc_alpha)),
    }
    return payload



def run_simulations() -> None:
    s = st.session_state
    if s.get("is_running", False):
        return

    s["is_running"] = True
    s["last_error"] = None

    try:
        sim = _import_simplesim()
        payload = build_payload()
        out = sim.run_simulations(payload)
        s["results"] = out
        s["results_meta"] = {"n_sims": payload["n_sims"], "seed": payload["seed"]}
    except Exception as e:
        s["last_error"] = f"{type(e).__name__}: {e}"
    finally:
        s["is_running"] = False
