# core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

DEFAULTS: Dict[str, Any] = {
    # essentials
    "target": 0.15,
    "start_dose": 1,          # 0-based index
    "crm_max_n": 27,
    "crm_cohort": 3,
    "n_sims": 500,
    "seed": 123,

    # true curve (example defaults, pas aan aan jouw eerdere defaults)
    "true_p": [0.01, 0.02, 0.12, 0.20, 0.35],
    "edit_true_curve": False,

    # prior playground
    "skeleton_model": "empiric",   # "empiric" or "logistic"
    "prior_target": 0.15,
    "delta": 0.10,
    "prior_mtd_nu": 3,             # 1-based in UI
    "logistic_intercept": 3.0,

    # CRM knobs
    "sigma_theta": 1.0,
    "burn_in_first_dlt": True,
    "ewoc": False,
    "ewoc_alpha": 0.25,
}

DOSE_LABELS = ["L0\n5×4 Gy", "L1\n5×5 Gy", "L2\n5×6 Gy", "L3\n5×7 Gy", "L4\n5×8 Gy"]


def init_state(st) -> None:
    """Initialize session_state once without overwriting user input."""
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def true_mtd_index(true_p: List[float], target: float) -> int:
    arr = np.array(true_p, dtype=float)
    return int(np.argmin(np.abs(arr - target)))


def build_skeleton_from_controls(st) -> List[float]:
    """
    Produce a monotone skeleton vector length 5.
    Keep it simple; you can replace with your original R-like logic.
    """
    model = st.session_state["skeleton_model"]
    tgt = float(st.session_state["prior_target"])
    delta = float(st.session_state["delta"])
    nu = int(st.session_state["prior_mtd_nu"])  # 1..5
    nu0 = nu - 1

    if model == "logistic":
        a = float(st.session_state["logistic_intercept"])
        # Create increasing dose scores centered at nu0
        x = np.arange(5) - nu0
        p = 1 / (1 + np.exp(-(a + x)))
        # rescale to roughly hit target at nu0
        if p[nu0] > 0:
            p = p * (tgt / p[nu0])
        p = np.clip(p, 1e-6, 0.999)
        return p.tolist()

    # empiric: simple ladder around target with delta
    p = np.array([tgt - 2*delta, tgt - delta, tgt, tgt + delta, tgt + 2*delta], dtype=float)
    # shift so the nu index maps to target
    shift = tgt - p[nu0]
    p = p + shift
    p = np.clip(p, 1e-6, 0.999)
    # enforce monotone
    p = np.maximum.accumulate(p)
    return p.tolist()


# ---------- Plug your real simulation engine here ----------

@dataclass
class SimResults:
    p_select_6p3: List[float]
    p_select_crm: List[float]
    mean_n_6p3: List[float]
    mean_n_crm: List[float]
    dlt_prob_6p3: float
    dlt_prob_crm: float


def run_simulations_stub(st) -> SimResults:
    """
    Replace with your real CRM + 6+3 sim.
    This stub just makes deterministic-ish output so UI works.
    """
    rng = get_rng(st.session_state["seed"])
    k = 5

    # fake selection probabilities
    base = rng.random(k)
    p6 = base / base.sum()
    base2 = rng.random(k)
    pc = base2 / base2.sum()

    # fake mean treated
    n6 = (p6 * 27).tolist()
    nc = (pc * 27).tolist()

    # fake DLT probability per patient: weighted by true curve
    truep = np.array(st.session_state["true_p"], dtype=float)
    dlt6 = float(np.dot(p6, truep))
    dltc = float(np.dot(pc, truep))

    return SimResults(
        p_select_6p3=p6.tolist(),
        p_select_crm=pc.tolist(),
        mean_n_6p3=n6,
        mean_n_crm=nc,
        dlt_prob_6p3=dlt6,
        dlt_prob_crm=dltc,
    )
