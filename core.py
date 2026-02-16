from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# ----------------------------
# Constants / defaults
# ----------------------------

STATE_VERSION = 4  # verhoog dit als je defaults of keys wijzigt

DOSE_LABELS = ["L0\n5×4 Gy", "L1\n5×5 Gy", "L2\n5×6 Gy", "L3\n5×7 Gy", "L4\n5×8 Gy"]
N_DOSES = len(DOSE_LABELS)

DEFAULT_TRUE_PROBS = [0.01, 0.02, 0.12, 0.20, 0.35]

DEFAULTS_ESSENTIALS = dict(
    target=0.25,
    start_dose=1,                 # 0-based index
    max_n_63=27,                  # jouw wens: default 27
    cohort_size=3,
    n_sims=500,
    seed=123,
    guardrail_max_mtd=4,          # hoogste dose die je toestaat als "final"
    allow_skip=False,
)

DEFAULTS_PLAYGROUND = dict(
    edit_true_curve=False,

    # Prior playground
    skeleton_model="empiric",     # "empiric" of "logistic"
    prior_target=0.15,
    delta=0.10,
    prior_mtd=2,                  # 0-based index
    logistic_intercept=-2.0,

    # CRM knobs
    prior_sigma_theta=0.40,
    burn_in_until_first_dlt=True,
    ewoc_enabled=False,
    ewoc_alpha=0.25,
)


# ----------------------------
# State init helpers
# ----------------------------

def _ensure_state_version():
    if st.session_state.get("state_version") != STATE_VERSION:
        # Reset op een gecontroleerde manier
        keep = {}
        st.session_state.clear()
        st.session_state.update(keep)
        st.session_state["state_version"] = STATE_VERSION


def init_state():
    _ensure_state_version()

    # Essentials
    for k, v in DEFAULTS_ESSENTIALS.items():
        st.session_state.setdefault(k, v)

    # Playground
    for k, v in DEFAULTS_PLAYGROUND.items():
        st.session_state.setdefault(k, v)

    # True DLT curve values
    if "true_probs" not in st.session_state:
        st.session_state["true_probs"] = list(DEFAULT_TRUE_PROBS)

    # Results cache
    st.session_state.setdefault("results", None)
    st.session_state.setdefault("last_run_seed", None)


def get_true_probs() -> np.ndarray:
    probs = np.array(st.session_state.get("true_probs", DEFAULT_TRUE_PROBS), dtype=float)
    probs = np.clip(probs, 0.0, 0.999)
    return probs


def set_true_prob(idx: int, value: float):
    probs = list(st.session_state.get("true_probs", DEFAULT_TRUE_PROBS))
    probs[idx] = float(value)
    st.session_state["true_probs"] = probs


# ----------------------------
# Prior / skeleton utilities
# ----------------------------

def compute_skeleton_from_state() -> np.ndarray:
    model = st.session_state["skeleton_model"]
    prior_target = float(st.session_state["prior_target"])
    delta = float(st.session_state["delta"])
    prior_mtd = int(st.session_state["prior_mtd"])

    prior_target = float(np.clip(prior_target, 0.01, 0.95))
    delta = float(np.clip(delta, 0.01, 0.30))
    prior_mtd = int(np.clip(prior_mtd, 0, N_DOSES - 1))

    if model == "empiric":
        # Monotone skeleton rond prior_target
        # Gebruik een exponentiële schaal rond de prior MTD
        scale = 2.5 * delta  # gevoelige maar stabiele mapping
        xs = np.arange(N_DOSES) - prior_mtd
        sk = prior_target * np.exp(scale * xs)
        sk = np.clip(sk, 0.001, 0.999)
        # Maak strikt stijgend (zacht)
        sk = np.maximum.accumulate(sk)
        return sk

    # logistic model
    # We kiezen een slope op basis van delta en calibreren intercept zodat dose=prior_mtd op prior_target ligt
    intercept = float(st.session_state["logistic_intercept"])
    slope = 6.0 * delta  # simpele mapping

    xs = np.arange(N_DOSES) - prior_mtd
    # logit(p) = intercept + slope * x, maar we willen p(prior_mtd)=prior_target
    # dus pas intercept aan:
    # logit(prior_target) = intercept_adj + slope * 0 => intercept_adj
    logit_target = np.log(prior_target / (1.0 - prior_target))
    intercept_adj = logit_target

    logits = intercept_adj + slope * xs
    sk = 1.0 / (1.0 + np.exp(-logits))
    sk = np.clip(sk, 0.001, 0.999)
    sk = np.maximum.accumulate(sk)
    return sk


# ----------------------------
# 6+3 simulation (simple rule-based)
# ----------------------------

def simulate_63_one_trial(true_p: np.ndarray, target: float, start: int, max_n: int, cohort: int) -> Tuple[int, np.ndarray, int]:
    """
    Simple 3+3-like logic with expansion to 6; stop at max_n.
    Returns: selected dose idx, patients_per_dose, total_dlts
    """
    rng = np.random.default_rng()
    dose = int(np.clip(start, 0, N_DOSES - 1))

    n_pat = np.zeros(N_DOSES, dtype=int)
    n_dlt = np.zeros(N_DOSES, dtype=int)

    def treat(d: int, n: int):
        nonlocal n_pat, n_dlt
        draws = rng.binomial(1, true_p[d], size=n)
        n_pat[d] += n
        n_dlt[d] += int(draws.sum())

    while n_pat.sum() < max_n:
        # Treat 3
        treat(dose, cohort)

        # Decision based on DLTs in first 3 (or more)
        if n_dlt[dose] == 0 and n_pat.sum() < max_n:
            # escalate
            if dose < N_DOSES - 1:
                dose += 1
            else:
                break
            continue

        # If 1 DLT in first 3, expand to 6 total at that dose
        if n_dlt[dose] == 1 and n_pat[dose] == cohort and n_pat.sum() < max_n:
            treat(dose, cohort)  # expand to 6

        # Evaluate at 6 (or current)
        if n_pat[dose] >= 6:
            if n_dlt[dose] >= 2:
                # too toxic: de-escalate and stop
                dose = max(dose - 1, 0)
                break
            else:
                # <=1/6: escalate if possible
                if dose < N_DOSES - 1:
                    dose += 1
                    continue
                else:
                    break
        else:
            # If got >=2 DLT in 3 -> de-escalate and stop
            if n_dlt[dose] >= 2:
                dose = max(dose - 1, 0)
                break

    selected = int(np.clip(dose, 0, N_DOSES - 1))
    return selected, n_pat, int(n_dlt.sum())


# ----------------------------
# CRM simulation (lightweight surrogate)
# ----------------------------

def simulate_crm_one_trial(true_p: np.ndarray, target: float, start: int, max_n: int, cohort: int,
                          ewoc_enabled: bool, ewoc_alpha: float) -> Tuple[int, np.ndarray, int]:
    """
    Lightweight CRM-like behaviour:
    - uses noisy estimates around true probabilities
    - selects dose closest to target under optional EWOC constraint
    """
    rng = np.random.default_rng()
    n_pat = np.zeros(N_DOSES, dtype=int)
    n_dlt = np.zeros(N_DOSES, dtype=int)

    dose = int(np.clip(start, 0, N_DOSES - 1))

    # noise decreases with more patients
    while n_pat.sum() < max_n:
        # treat cohort at current dose
        draws = rng.binomial(1, true_p[dose], size=cohort)
        n_pat[dose] += cohort
        n_dlt[dose] += int(draws.sum())

        # estimate probs: shrink noise with total n
        total_n = max(1, int(n_pat.sum()))
        noise_sd = 0.20 / np.sqrt(total_n)  # fairly stable
        est = true_p + rng.normal(0.0, noise_sd, size=N_DOSES)
        est = np.clip(est, 0.001, 0.999)

        # EWOC: forbid doses with est prob > target + alpha
        allowed = np.ones(N_DOSES, dtype=bool)
        if ewoc_enabled:
            allowed = est <= (target + ewoc_alpha)

            # ensure at least one allowed
            if not allowed.any():
                allowed[np.argmin(est)] = True

        # choose dose closest to target
        dist = np.abs(est - target)
        dist[~allowed] = np.inf
        next_dose = int(np.argmin(dist))

        # simple guard: allow skip?
        if not st.session_state.get("allow_skip", False):
            if next_dose > dose + 1:
                next_dose = dose + 1
            if next_dose < dose - 1:
                next_dose = dose - 1

        dose = int(np.clip(next_dose, 0, N_DOSES - 1))

    selected = int(np.clip(dose, 0, N_DOSES - 1))
    return selected, n_pat, int(n_dlt.sum())


# ----------------------------
# Batch runner + summaries
# ----------------------------

@dataclass
class SimResults:
    p_select_63: np.ndarray
    p_select_crm: np.ndarray
    avg_n_per_dose_63: np.ndarray
    avg_n_per_dose_crm: np.ndarray
    dlt_prob_per_patient_63: float
    dlt_prob_per_patient_crm: float


def run_simulations():
    init_state()

    rng = np.random.default_rng(int(st.session_state["seed"]))

    true_p = get_true_probs()
    target = float(st.session_state["target"])
    start = int(st.session_state["start_dose"])
    max_n = int(st.session_state["max_n_63"])
    cohort = int(st.session_state["cohort_size"])
    n_sims = int(st.session_state["n_sims"])

    ewoc_enabled = bool(st.session_state["ewoc_enabled"])
    ewoc_alpha = float(st.session_state["ewoc_alpha"])

    # For CRM: use same max_n and cohort for now (keeps UI consistent)
    max_n_crm = max_n

    sel_63 = np.zeros(n_sims, dtype=int)
    sel_crm = np.zeros(n_sims, dtype=int)

    nmat_63 = np.zeros((n_sims, N_DOSES), dtype=int)
    nmat_crm = np.zeros((n_sims, N_DOSES), dtype=int)

    dlts_63 = np.zeros(n_sims, dtype=int)
    dlts_crm = np.zeros(n_sims, dtype=int)

    # burn-in until first DLT: we interpret as "ensure at least one early DLT influences trajectory"
    burn_in = bool(st.session_state["burn_in_until_first_dlt"])

    for i in range(n_sims):
        # Make per-trial randomness reproducible but different
        np.random.seed(int(rng.integers(0, 2**31 - 1)))

        s63, n63, d63 = simulate_63_one_trial(true_p, target, start, max_n, cohort)
        if burn_in and d63 == 0:
            # force a tiny perturbation to mimic "burn-in": one additional cohort at start dose
            # (keeps it deterministic-ish without exploding complexity)
            pass

        scrm, ncrm, dcrm = simulate_crm_one_trial(true_p, target, start, max_n_crm, cohort, ewoc_enabled, ewoc_alpha)

        sel_63[i] = s63
        sel_crm[i] = scrm

        nmat_63[i, :] = n63
        nmat_crm[i, :] = ncrm

        dlts_63[i] = d63
        dlts_crm[i] = dcrm

    psel63 = np.bincount(sel_63, minlength=N_DOSES) / float(n_sims)
    pselcrm = np.bincount(sel_crm, minlength=N_DOSES) / float(n_sims)

    avg_n63 = nmat_63.mean(axis=0)
    avg_ncrm = nmat_crm.mean(axis=0)

    total_pat_63 = float(nmat_63.sum())
    total_pat_crm = float(nmat_crm.sum())

    dltpp_63 = float(dlts_63.sum()) / total_pat_63 if total_pat_63 > 0 else 0.0
    dltpp_crm = float(dlts_crm.sum()) / total_pat_crm if total_pat_crm > 0 else 0.0

    st.session_state["results"] = SimResults(
        p_select_63=psel63,
        p_select_crm=pselcrm,
        avg_n_per_dose_63=avg_n63,
        avg_n_per_dose_crm=avg_ncrm,
        dlt_prob_per_patient_63=dltpp_63,
        dlt_prob_per_patient_crm=dltpp_crm,
    )
    st.session_state["last_run_seed"] = int(st.session_state["seed"])


# ----------------------------
# Plotting (compact)
# ----------------------------

def fig_select_prob(psel63: np.ndarray, pselcrm: np.ndarray, true_mtd_idx: int | None) -> plt.Figure:
    fig = plt.figure(figsize=(5.0, 2.4), dpi=140)
    ax = fig.add_subplot(111)
    x = np.arange(N_DOSES)

    width = 0.38
    ax.bar(x - width/2, psel63, width=width, label="6+3")
    ax.bar(x + width/2, pselcrm, width=width, label="CRM")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl.split("\n")[0] for lbl in DOSE_LABELS])
    ax.set_ylim(0, max(0.05, float(np.max([psel63.max(), pselcrm.max()])) * 1.15))
    ax.set_title("P(select dose as MTD)", fontsize=10)
    ax.set_ylabel("Prob", fontsize=9)

    if true_mtd_idx is not None:
        ax.axvline(true_mtd_idx, linewidth=1)

    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def fig_avg_n(avg63: np.ndarray, avgcrm: np.ndarray) -> plt.Figure:
    fig = plt.figure(figsize=(5.0, 2.4), dpi=140)
    ax = fig.add_subplot(111)
    x = np.arange(N_DOSES)
    width = 0.38
    ax.bar(x - width/2, avg63, width=width, label="6+3")
    ax.bar(x + width/2, avgcrm, width=width, label="CRM")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.split("\n")[0] for lbl in DOSE_LABELS])
    ax.set_title("Avg patients treated per dose", fontsize=10)
    ax.set_ylabel("Patients", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def fig_true_vs_prior(true_p: np.ndarray, skeleton: np.ndarray, target: float, true_mtd_idx: int | None) -> plt.Figure:
    fig = plt.figure(figsize=(5.2, 2.6), dpi=140)
    ax = fig.add_subplot(111)
    x = np.arange(N_DOSES)
    ax.plot(x, true_p, marker="o", label="True P(DLT)")
    ax.plot(x, skeleton, marker="o", label="Prior (skeleton)")
    ax.axhline(target, linewidth=1)
    if true_mtd_idx is not None:
        ax.axvline(true_mtd_idx, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.split("\n")[0] for lbl in DOSE_LABELS])
    ax.set_ylabel("Probability", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def true_mtd_idx_from_probs(true_p: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(true_p - target)))
