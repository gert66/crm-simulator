from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import streamlit as st


# ----------------------------
# Shared constants
# ----------------------------

DOSE_LABELS = ["L0\n5×4 Gy", "L1\n5×5 Gy", "L2\n5×6 Gy", "L3\n5×7 Gy", "L4\n5×8 Gy"]
N_DOSES = len(DOSE_LABELS)

# Session-state keys (keep consistent)
K_TRUE_P = "true_p_dlt"               # list[float] length N_DOSES
K_EDIT_TRUE = "edit_true_curve"       # bool

# Prior playground keys
K_SKEL_MODEL = "skel_model"           # "empiric" | "logistic"
K_PRIOR_TARGET = "prior_target"       # float
K_PRIOR_DELTA = "prior_delta"         # float (halfwidth)
K_PRIOR_MTD_NU = "prior_mtd_nu"        # int, 1-based index (1..N_DOSES)
K_LOGISTIC_INTERCEPT = "logistic_intercept"  # float

# CRM knobs
K_PRIOR_SIGMA_THETA = "prior_sigma_theta"     # float
K_BURNIN_FIRST_DLT = "burnin_until_first_dlt" # bool
K_ENABLE_EWOC = "enable_ewoc"                 # bool
K_EWOC_ALPHA = "ewoc_alpha"                   # float

# Essentials keys (example; adjust names to what you already use)
K_TARGET = "target_toxicity"
K_START_DOSE = "start_dose_level"            # 0-based index
K_MAX_N_SERUM = "max_n_serum"                # 6+3 max
K_COHORT = "cohort_size"                     # cohort size
K_N_SIMS = "n_sims"
K_SEED = "seed"
K_GAUSS_POINTS = "gauss_points"

K_RESULTS = "results"                         # dict stored after running


# ----------------------------
# Defaults
# ----------------------------

@dataclass(frozen=True)
class Defaults:
    target_toxicity: float = 0.15
    start_dose_level: int = 0
    max_n_serum: int = 27
    cohort_size: int = 3
    n_sims: int = 500
    seed: int = 123
    gauss_points: int = 15

    # True curve defaults
    true_p_dlt: Tuple[float, float, float, float, float] = (0.01, 0.02, 0.12, 0.20, 0.35)

    # Prior playground defaults
    skel_model: str = "empiric"
    prior_target: float = 0.15
    prior_delta: float = 0.10
    prior_mtd_nu: int = 3            # 1-based
    logistic_intercept: float = -2.5

    # CRM knobs defaults
    prior_sigma_theta: float = 1.0
    burnin_until_first_dlt: bool = False
    enable_ewoc: bool = False
    ewoc_alpha: float = 0.25


DEFAULTS = Defaults()


def _set_if_missing(key: str, value: Any) -> None:
    """Critical: only set defaults if key is missing, never overwrite."""
    if key not in st.session_state:
        st.session_state[key] = value


def init_state() -> None:
    """Call this at the top of every page. It is safe on reruns."""
    # Essentials
    _set_if_missing(K_TARGET, DEFAULTS.target_toxicity)
    _set_if_missing(K_START_DOSE, DEFAULTS.start_dose_level)
    _set_if_missing(K_MAX_N_SERUM, DEFAULTS.max_n_serum)
    _set_if_missing(K_COHORT, DEFAULTS.cohort_size)
    _set_if_missing(K_N_SIMS, DEFAULTS.n_sims)
    _set_if_missing(K_SEED, DEFAULTS.seed)
    _set_if_missing(K_GAUSS_POINTS, DEFAULTS.gauss_points)

    # True curve
    _set_if_missing(K_TRUE_P, list(DEFAULTS.true_p_dlt))
    _set_if_missing(K_EDIT_TRUE, False)

    # Prior playground
    _set_if_missing(K_SKEL_MODEL, DEFAULTS.skel_model)
    _set_if_missing(K_PRIOR_TARGET, DEFAULTS.prior_target)
    _set_if_missing(K_PRIOR_DELTA, DEFAULTS.prior_delta)
    _set_if_missing(K_PRIOR_MTD_NU, DEFAULTS.prior_mtd_nu)
    _set_if_missing(K_LOGISTIC_INTERCEPT, DEFAULTS.logistic_intercept)

    # CRM knobs
    _set_if_missing(K_PRIOR_SIGMA_THETA, DEFAULTS.prior_sigma_theta)
    _set_if_missing(K_BURNIN_FIRST_DLT, DEFAULTS.burnin_until_first_dlt)
    _set_if_missing(K_ENABLE_EWOC, DEFAULTS.enable_ewoc)
    _set_if_missing(K_EWOC_ALPHA, DEFAULTS.ewoc_alpha)

    # Results container
    _set_if_missing(K_RESULTS, None)


def reset_to_defaults(scope: str = "all") -> None:
    """
    scope: "all" | "essentials" | "playground"
    Resets relevant keys AND clears results.
    """
    if scope in ("all", "essentials"):
        st.session_state[K_TARGET] = DEFAULTS.target_toxicity
        st.session_state[K_START_DOSE] = DEFAULTS.start_dose_level
        st.session_state[K_MAX_N_SERUM] = DEFAULTS.max_n_serum
        st.session_state[K_COHORT] = DEFAULTS.cohort_size
        st.session_state[K_N_SIMS] = DEFAULTS.n_sims
        st.session_state[K_SEED] = DEFAULTS.seed
        st.session_state[K_GAUSS_POINTS] = DEFAULTS.gauss_points

    if scope in ("all", "playground"):
        st.session_state[K_TRUE_P] = list(DEFAULTS.true_p_dlt)
        st.session_state[K_EDIT_TRUE] = DEFAULTS.burnin_until_first_dlt  # harmless; toggles are per-widget anyway
        st.session_state[K_SKEL_MODEL] = DEFAULTS.skel_model
        st.session_state[K_PRIOR_TARGET] = DEFAULTS.prior_target
        st.session_state[K_PRIOR_DELTA] = DEFAULTS.prior_delta
        st.session_state[K_PRIOR_MTD_NU] = DEFAULTS.prior_mtd_nu
        st.session_state[K_LOGISTIC_INTERCEPT] = DEFAULTS.logistic_intercept
        st.session_state[K_PRIOR_SIGMA_THETA] = DEFAULTS.prior_sigma_theta
        st.session_state[K_BURNIN_FIRST_DLT] = DEFAULTS.burnin_until_first_dlt
        st.session_state[K_ENABLE_EWOC] = DEFAULTS.enable_ewoc
        st.session_state[K_EWOC_ALPHA] = DEFAULTS.ewoc_alpha

    st.session_state[K_RESULTS] = None


# ----------------------------
# Prior skeleton utilities
# ----------------------------

def get_true_curve() -> np.ndarray:
    p = np.array(st.session_state[K_TRUE_P], dtype=float)
    if p.shape[0] != N_DOSES:
        p = np.resize(p, N_DOSES)
    return np.clip(p, 0.0, 1.0)


def compute_empiric_skeleton(prior_target: float, delta: float, nu_1based: int) -> np.ndarray:
    """
    Simple empiric skeleton around a target at dose nu.
    Produces monotone increasing probabilities.
    """
    nu = int(np.clip(nu_1based, 1, N_DOSES)) - 1
    base = np.linspace(prior_target - 2 * delta, prior_target + 2 * delta, N_DOSES)
    shift = prior_target - base[nu]
    sk = base + shift
    sk = np.maximum.accumulate(sk)  # enforce monotone
    return np.clip(sk, 0.001, 0.999)


def compute_logistic_skeleton(prior_target: float, nu_1based: int, intercept: float) -> np.ndarray:
    """
    Logistic skeleton with a free intercept; slope is chosen so that P(nu) ~ prior_target.
    This is a lightweight preview skeleton, not a full CRM model.
    """
    nu = int(np.clip(nu_1based, 1, N_DOSES)) - 1
    x = np.arange(N_DOSES, dtype=float)

    # Solve slope so that sigmoid(intercept + slope*nu) = prior_target
    pt = float(np.clip(prior_target, 0.001, 0.999))
    logit_pt = np.log(pt / (1.0 - pt))
    slope = (logit_pt - intercept) / max(nu, 1e-9)

    z = intercept + slope * x
    sk = 1.0 / (1.0 + np.exp(-z))
    sk = np.maximum.accumulate(sk)
    return np.clip(sk, 0.001, 0.999)


def get_skeleton_preview_from_state() -> np.ndarray:
    model = st.session_state[K_SKEL_MODEL]
    pt = float(st.session_state[K_PRIOR_TARGET])
    nu = int(st.session_state[K_PRIOR_MTD_NU])
    if model == "logistic":
        icpt = float(st.session_state[K_LOGISTIC_INTERCEPT])
        return compute_logistic_skeleton(pt, nu, icpt)
    delta = float(st.session_state[K_PRIOR_DELTA])
    return compute_empiric_skeleton(pt, delta, nu)


def true_mtd_index_from_curve(p: np.ndarray, target: float) -> int:
    """Closest dose to target."""
    target = float(target)
    return int(np.argmin(np.abs(p - target)))


# ----------------------------
# Simulation hook (kept flexible)
# ----------------------------

def _call_user_simulator(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls your existing simulator in simplesim.py without forcing a specific API.
    Tries common function names. If none exist, raises a clear error.

    Expected return dict keys (minimal):
      - "p_select_6p3": np.ndarray shape (N_DOSES,)
      - "p_select_crm": np.ndarray shape (N_DOSES,)
      - "avg_n_per_dose_6p3": np.ndarray shape (N_DOSES,)
      - "avg_n_per_dose_crm": np.ndarray shape (N_DOSES,)
      - "p_dlt_per_patient_6p3": float
      - "p_dlt_per_patient_crm": float
    """
    import importlib

    sim = importlib.import_module("simplesim")

    candidates = [
        "run_simulations",
        "run_sims",
        "simulate",
        "simulate_trials",
        "simulate_many",
    ]

    fn = None
    for name in candidates:
        if hasattr(sim, name) and callable(getattr(sim, name)):
            fn = getattr(sim, name)
            break

    if fn is None:
        raise RuntimeError(
            "core.py could not find a callable simulator in simplesim.py.\n"
            "Add one of these functions to simplesim.py: "
            + ", ".join(candidates)
            + "\nIt must accept a single dict payload and return a results dict."
        )

    out = fn(payload)

    if not isinstance(out, dict):
        raise TypeError("simplesim simulator must return a dict.")

    return out


def run_simulations() -> None:
    """
    NO-ARG function.
    Safe to call from a Streamlit button.
    Stores results in st.session_state[K_RESULTS].
    """
    init_state()

    true_curve = get_true_curve()
    target = float(st.session_state[K_TARGET])
    skel = get_skeleton_preview_from_state()
    true_mtd = true_mtd_index_from_curve(true_curve, target)

    payload = {
        # essentials
        "target": target,
        "start_dose": int(st.session_state[K_START_DOSE]),
        "max_n_serum": int(st.session_state[K_MAX_N_SERUM]),
        "cohort_size": int(st.session_state[K_COHORT]),
        "n_sims": int(st.session_state[K_N_SIMS]),
        "seed": int(st.session_state[K_SEED]),
        "gauss_points": int(st.session_state[K_GAUSS_POINTS]),
        # playground
        "true_curve": true_curve,
        "skeleton_model": st.session_state[K_SKEL_MODEL],
        "skeleton": skel,
        "prior_target": float(st.session_state[K_PRIOR_TARGET]),
        "prior_delta": float(st.session_state[K_PRIOR_DELTA]),
        "prior_mtd_nu": int(st.session_state[K_PRIOR_MTD_NU]),
        "logistic_intercept": float(st.session_state[K_LOGISTIC_INTERCEPT]),
        "prior_sigma_theta": float(st.session_state[K_PRIOR_SIGMA_THETA]),
        "burnin_until_first_dlt": bool(st.session_state[K_BURNIN_FIRST_DLT]),
        "enable_ewoc": bool(st.session_state[K_ENABLE_EWOC]),
        "ewoc_alpha": float(st.session_state[K_EWOC_ALPHA]),
        "true_mtd_index": true_mtd,
    }

    out = _call_user_simulator(payload)

    # Basic sanity / coercion
    def _arr(key: str) -> np.ndarray:
        a = np.asarray(out.get(key, np.zeros(N_DOSES)), dtype=float)
        if a.shape[0] != N_DOSES:
            a = np.resize(a, N_DOSES)
        return a

    results = {
        "p_select_6p3": _arr("p_select_6p3"),
        "p_select_crm": _arr("p_select_crm"),
        "avg_n_per_dose_6p3": _arr("avg_n_per_dose_6p3"),
        "avg_n_per_dose_crm": _arr("avg_n_per_dose_crm"),
        "p_dlt_per_patient_6p3": float(out.get("p_dlt_per_patient_6p3", np.nan)),
        "p_dlt_per_patient_crm": float(out.get("p_dlt_per_patient_crm", np.nan)),
        "meta": {
            "n_sims": int(st.session_state[K_N_SIMS]),
            "seed": int(st.session_state[K_SEED]),
            "true_mtd_index": true_mtd,
        },
    }

    st.session_state[K_RESULTS] = results
