# core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import numpy as np


# -------------------------
# Defaults (single source)
# -------------------------
DEFAULTS: Dict[str, object] = {
    # Essentials
    "target": 0.15,
    "start_dose_idx": 1,  # 0-based (L0..L4). Default L1.
    "dose_labels": ["L0\n5×4 Gy", "L1\n5×5 Gy", "L2\n5×6 Gy", "L3\n5×7 Gy", "L4\n5×8 Gy"],
    "crm_max_n": 27,
    "crm_cohort": 3,
    "sixplus3_max_n": 27,
    "n_sims": 500,
    "seed": 123,

    # CRM knobs
    "prior_sigma_theta": 1.0,
    "burn_in_until_first_dlt": True,
    "enable_ewoc": False,
    "ewoc_alpha": 0.25,

    # Prior playground (skeleton)
    "skeleton_model": "empiric",  # "empiric" or "logistic"
    "prior_target": 0.15,
    "prior_halfwidth": 0.10,
    "prior_mtd_nu": 3,  # 1-based index shown in UI
    "logistic_intercept": 3.0,

    # True curve (editable)
    "edit_true_curve": False,
    "true_p0": 0.01,
    "true_p1": 0.02,
    "true_p2": 0.12,
    "true_p3": 0.20,
    "true_p4": 0.35,

    # Storage for last results
    "last_results": None,
}


def init_state(st, defaults: Dict[str, object] = DEFAULTS) -> None:
    """
    Initialize session_state BEFORE widgets are created.
    """
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_to_defaults(st, defaults: Dict[str, object] = DEFAULTS) -> None:
    """
    Reset relevant keys and hard-rerun.
    Put the reset button BEFORE any widgets on a page.
    """
    for k, v in defaults.items():
        st.session_state[k] = v
    st.rerun()


# -------------------------
# Helpers: curves & skeleton
# -------------------------
def get_true_curve_from_state(st) -> np.ndarray:
    return np.array(
        [
            float(st.session_state["true_p0"]),
            float(st.session_state["true_p1"]),
            float(st.session_state["true_p2"]),
            float(st.session_state["true_p3"]),
            float(st.session_state["true_p4"]),
        ],
        dtype=float,
    )


def clamp_probs(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def skeleton_empiric(prior_target: float, halfwidth: float, nu_1based: int, n_doses: int = 5) -> np.ndarray:
    """
    Simple monotone skeleton anchored so that skeleton[nu-1] ~= prior_target,
    with a linear step in logit-space controlled by 'halfwidth'.
    """
    nu = int(nu_1based) - 1
    nu = max(0, min(n_doses - 1, nu))

    t = float(prior_target)
    t = min(max(t, 1e-4), 1 - 1e-4)
    hw = float(halfwidth)
    hw = max(hw, 1e-3)

    # logit(t) at nu, then move up/down in equal steps
    base = math.log(t / (1 - t))
    step = math.log((t + hw) / (1 - (t + hw))) - base
    if not np.isfinite(step) or abs(step) < 1e-6:
        step = 0.35  # fallback

    logits = np.array([base + (i - nu) * step for i in range(n_doses)], dtype=float)
    p = 1 / (1 + np.exp(-logits))
    return clamp_probs(p)


def skeleton_logistic(logistic_intercept: float, prior_mtd_nu_1based: int, prior_target: float, n_doses: int = 5) -> np.ndarray:
    """
    Another monotone skeleton where intercept shifts the curve.
    We still anchor the selected MTD index near prior_target.
    """
    nu = int(prior_mtd_nu_1based) - 1
    nu = max(0, min(n_doses - 1, nu))
    t = min(max(float(prior_target), 1e-4), 1 - 1e-4)

    # choose slope so that moving 1 dose changes logit modestly
    slope = 0.9
    # pick center so that dose nu hits target
    center = (math.log(t / (1 - t)) - float(logistic_intercept)) / slope - nu

    x = np.arange(n_doses, dtype=float)
    logits = float(logistic_intercept) + slope * (x - (-center))
    p = 1 / (1 + np.exp(-logits))
    return clamp_probs(p)


def get_skeleton_from_state(st) -> np.ndarray:
    model = st.session_state["skeleton_model"]
    if model == "logistic":
        return skeleton_logistic(
            logistic_intercept=float(st.session_state["logistic_intercept"]),
            prior_mtd_nu_1based=int(st.session_state["prior_mtd_nu"]),
            prior_target=float(st.session_state["prior_target"]),
            n_doses=5,
        )
    return skeleton_empiric(
        prior_target=float(st.session_state["prior_target"]),
        halfwidth=float(st.session_state["prior_halfwidth"]),
        nu_1based=int(st.session_state["prior_mtd_nu"]),
        n_doses=5,
    )


# -------------------------
# Simulation engines
# -------------------------
def simulate_bernoulli(rng: np.random.Generator, p: float, n: int) -> int:
    return int(rng.binomial(n=n, p=p))


def six_plus_three_trial(
    rng: np.random.Generator,
    true_p: np.ndarray,
    start_idx: int,
    max_n: int,
) -> Tuple[int, np.ndarray, int]:
    """
    A practical "6+3" rule set:

    At a dose:
    - Treat 6.
      - 0 DLT -> escalate
      - 1 DLT -> treat +3 (total 9)
          - <=1/9 -> escalate
          - >=2/9 -> de-escalate and stop (select previous)
      - >=2/6 -> de-escalate and stop (select previous)
    Stops if max_n reached or boundaries hit.

    Returns:
    - selected_mtd_idx
    - n_treated_per_dose
    - total_dlts
    """
    n_doses = len(true_p)
    n_treated = np.zeros(n_doses, dtype=int)
    total_dlts = 0

    dose = int(start_idx)
    dose = max(0, min(n_doses - 1, dose))

    while True:
        remaining = max_n - int(n_treated.sum())
        if remaining <= 0:
            break

        # treat 6 (or fewer if running out)
        n1 = min(6, remaining)
        d1 = simulate_bernoulli(rng, float(true_p[dose]), n1)
        n_treated[dose] += n1
        total_dlts += d1

        if n1 < 6:
            # out of patients, pick current dose as "final"
            break

        if d1 == 0:
            # escalate if possible
            if dose == n_doses - 1:
                break
            dose += 1
            continue

        if d1 == 1:
            remaining = max_n - int(n_treated.sum())
            if remaining <= 0:
                break
            n2 = min(3, remaining)
            d2 = simulate_bernoulli(rng, float(true_p[dose]), n2)
            n_treated[dose] += n2
            total_dlts += d2

            if n2 < 3:
                break

            if (d1 + d2) <= 1:
                if dose == n_doses - 1:
                    break
                dose += 1
                continue
            else:
                # too toxic: select previous if exists
                dose = max(0, dose - 1)
                break

        # d1 >= 2
        dose = max(0, dose - 1)
        break

    selected = int(dose)
    return selected, n_treated, total_dlts


def crm_posterior_weights(
    theta_grid: np.ndarray,
    sigma: float,
    y: np.ndarray,
    n: np.ndarray,
    skeleton: np.ndarray,
) -> np.ndarray:
    """
    Posterior over theta for the 1-parameter power model:
      p_i(theta) = skeleton_i ** exp(theta)

    Prior: theta ~ Normal(0, sigma^2)
    Likelihood: Binomial(n_i, p_i(theta))
    """
    sigma = float(max(sigma, 1e-6))
    log_prior = -0.5 * (theta_grid / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)

    s = clamp_probs(skeleton)
    # shape: (G, D)
    p = np.power(s[None, :], np.exp(theta_grid)[:, None])
    p = clamp_probs(p)

    # binomial loglik for each theta
    # sum_i [ y_i*log(p_i) + (n_i-y_i)*log(1-p_i) ]
    loglik = (y[None, :] * np.log(p) + (n[None, :] - y[None, :]) * np.log(1 - p)).sum(axis=1)

    log_post = log_prior + loglik
    log_post -= np.max(log_post)
    w = np.exp(log_post)
    w /= np.sum(w)
    return w


def crm_choose_dose(
    theta_grid: np.ndarray,
    w: np.ndarray,
    skeleton: np.ndarray,
    target: float,
    current_dose: int,
    enable_ewoc: bool,
    ewoc_alpha: float,
) -> int:
    """
    Choose next dose based on posterior mean p_i(theta) closest to target.
    If EWOC enabled: require P(p_i > target) <= alpha.
    """
    s = clamp_probs(skeleton)
    p = np.power(s[None, :], np.exp(theta_grid)[:, None])
    p = clamp_probs(p)

    p_mean = (w[:, None] * p).sum(axis=0)

    # EWOC constraint
    if enable_ewoc:
        alpha = float(ewoc_alpha)
        alpha = min(max(alpha, 0.01), 0.5)
        prob_over = (w[:, None] * (p > target)).sum(axis=0)
        allowed = prob_over <= alpha
    else:
        allowed = np.ones_like(p_mean, dtype=bool)

    # among allowed, pick closest to target; if none allowed, de-escalate one step
    candidates = np.where(allowed)[0]
    if len(candidates) == 0:
        return max(0, current_dose - 1)

    best = candidates[np.argmin(np.abs(p_mean[candidates] - target))]
    # no skipping: move at most 1 level
    if best > current_dose + 1:
        best = current_dose + 1
    if best < current_dose - 1:
        best = current_dose - 1
    return int(best)


def crm_trial(
    rng: np.random.Generator,
    true_p: np.ndarray,
    skeleton: np.ndarray,
    target: float,
    start_idx: int,
    cohort: int,
    max_n: int,
    sigma_theta: float,
    burn_in_until_first_dlt: bool,
    enable_ewoc: bool,
    ewoc_alpha: float,
) -> Tuple[int, np.ndarray, int]:
    """
    CRM with 1-parameter power model and grid posterior.
    Returns selected MTD (closest to target in posterior mean at end),
    treated counts, total DLTs.
    """
    n_doses = len(true_p)
    y = np.zeros(n_doses, dtype=int)
    n = np.zeros(n_doses, dtype=int)
    total_dlts = 0

    dose = int(start_idx)
    dose = max(0, min(n_doses - 1, dose))

    # theta grid for posterior (fast + stable)
    theta_grid = np.linspace(-4.0, 4.0, 801)

    first_dlt_seen = False

    while int(n.sum()) < max_n:
        # burn-in: keep treating at start dose until first DLT appears
        if burn_in_until_first_dlt and (not first_dlt_seen):
            dose = int(start_idx)

        remaining = max_n - int(n.sum())
        m = min(int(cohort), remaining)
        d = simulate_bernoulli(rng, float(true_p[dose]), m)

        n[dose] += m
        y[dose] += d
        total_dlts += d

        if d > 0:
            first_dlt_seen = True

        if int(n.sum()) >= max_n:
            break

        # update posterior and choose next dose
        w = crm_posterior_weights(theta_grid, sigma_theta, y, n, skeleton)
        dose = crm_choose_dose(
            theta_grid=theta_grid,
            w=w,
            skeleton=skeleton,
            target=float(target),
            current_dose=dose,
            enable_ewoc=bool(enable_ewoc),
            ewoc_alpha=float(ewoc_alpha),
        )

        dose = max(0, min(n_doses - 1, dose))

    # final selection = closest to target based on posterior mean
    w = crm_posterior_weights(theta_grid, sigma_theta, y, n, skeleton)
    s = clamp_probs(skeleton)
    p = np.power(s[None, :], np.exp(theta_grid)[:, None])
    p = clamp_probs(p)
    p_mean = (w[:, None] * p).sum(axis=0)
    selected = int(np.argmin(np.abs(p_mean - float(target))))

    return selected, n, total_dlts


@dataclass
class SimResults:
    dose_labels: List[str]
    true_curve: List[float]
    skeleton: List[float]
    target: float
    true_mtd_idx: int
    n_sims: int
    seed: int

    # outputs
    prob_mtd_six: List[float]
    prob_mtd_crm: List[float]
    avg_n_six: List[float]
    avg_n_crm: List[float]
    dlt_prob_per_patient_six: float
    dlt_prob_per_patient_crm: float


def run_simulations(
    true_curve: np.ndarray,
    skeleton: np.ndarray,
    dose_labels: List[str],
    target: float,
    start_idx: int,
    n_sims: int,
    seed: int,
    six_max_n: int,
    crm_max_n: int,
    crm_cohort: int,
    prior_sigma_theta: float,
    burn_in_until_first_dlt: bool,
    enable_ewoc: bool,
    ewoc_alpha: float,
) -> SimResults:
    rng = np.random.default_rng(int(seed))

    n_doses = len(true_curve)
    true_curve = clamp_probs(true_curve)
    skeleton = clamp_probs(skeleton)

    # "True MTD" = dose with true p closest to target
    true_mtd_idx = int(np.argmin(np.abs(true_curve - float(target))))

    mtd_counts_six = np.zeros(n_doses, dtype=int)
    mtd_counts_crm = np.zeros(n_doses, dtype=int)
    n_sum_six = np.zeros(n_doses, dtype=float)
    n_sum_crm = np.zeros(n_doses, dtype=float)

    total_dlts_six = 0
    total_pts_six = 0
    total_dlts_crm = 0
    total_pts_crm = 0

    for _ in range(int(n_sims)):
        sel6, n6, d6 = six_plus_three_trial(rng, true_curve, start_idx, six_max_n)
        mtd_counts_six[sel6] += 1
        n_sum_six += n6
        total_dlts_six += d6
        total_pts_six += int(n6.sum())

        selc, nc, dc = crm_trial(
            rng=rng,
            true_p=true_curve,
            skeleton=skeleton,
            target=float(target),
            start_idx=start_idx,
            cohort=int(crm_cohort),
            max_n=int(crm_max_n),
            sigma_theta=float(prior_sigma_theta),
            burn_in_until_first_dlt=bool(burn_in_until_first_dlt),
            enable_ewoc=bool(enable_ewoc),
            ewoc_alpha=float(ewoc_alpha),
        )
        mtd_counts_crm[selc] += 1
        n_sum_crm += nc
        total_dlts_crm += dc
        total_pts_crm += int(nc.sum())

    prob_mtd_six = (mtd_counts_six / float(n_sims)).tolist()
    prob_mtd_crm = (mtd_counts_crm / float(n_sims)).tolist()
    avg_n_six = (n_sum_six / float(n_sims)).tolist()
    avg_n_crm = (n_sum_crm / float(n_sims)).tolist()

    dlt_prob_six = (total_dlts_six / total_pts_six) if total_pts_six > 0 else 0.0
    dlt_prob_crm = (total_dlts_crm / total_pts_crm) if total_pts_crm > 0 else 0.0

    return SimResults(
        dose_labels=dose_labels,
        true_curve=true_curve.tolist(),
        skeleton=skeleton.tolist(),
        target=float(target),
        true_mtd_idx=true_mtd_idx,
        n_sims=int(n_sims),
        seed=int(seed),
        prob_mtd_six=prob_mtd_six,
        prob_mtd_crm=prob_mtd_crm,
        avg_n_six=avg_n_six,
        avg_n_crm=avg_n_crm,
        dlt_prob_per_patient_six=float(dlt_prob_six),
        dlt_prob_per_patient_crm=float(dlt_prob_crm),
    )
