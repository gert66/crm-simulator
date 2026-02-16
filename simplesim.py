# simplesim.py
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _build_empiric_skeleton(n_dose: int, prior_target: float, delta: float, prior_mtd_1based: int) -> np.ndarray:
    m = int(prior_mtd_1based) - 1
    m = max(0, min(n_dose - 1, m))
    vals = [prior_target + (i - m) * delta for i in range(n_dose)]
    sk = np.array([_clamp(v, 0.0001, 0.9999) for v in vals], dtype=float)
    sk = np.maximum.accumulate(sk)
    return sk


def _build_logistic_skeleton(n_dose: int, prior_target: float, prior_mtd_1based: int, intercept: float) -> np.ndarray:
    m = int(prior_mtd_1based) - 1
    m = max(0, min(n_dose - 1, m))
    logit = np.log(prior_target / (1.0 - prior_target))
    b = 0.25 if m == 0 else (logit - intercept) / m
    xs = np.arange(n_dose, dtype=float)
    z = intercept + b * xs
    sk = 1.0 / (1.0 + np.exp(-z))
    sk = np.maximum.accumulate(sk)
    return sk


def _get_skeleton(payload: Dict[str, Any], n_dose: int) -> np.ndarray:
    model = payload.get("skeleton_model", "empiric")
    prior_target = float(payload.get("prior_target", 0.15))
    prior_mtd = int(payload.get("prior_mtd_1based", 3))
    if model == "logistic":
        intercept = float(payload.get("logistic_intercept", 0.0))
        return _build_logistic_skeleton(n_dose, prior_target, prior_mtd, intercept)
    delta = float(payload.get("delta", 0.10))
    return _build_empiric_skeleton(n_dose, prior_target, delta, prior_mtd)


def _simulate_6p3_trial(
    rng: np.random.Generator,
    true_curve: np.ndarray,
    start_idx: int,
    cohort_size: int,
    max_n: int,
) -> Tuple[int, np.ndarray, int, int]:
    n_dose = len(true_curve)
    treated = np.zeros(n_dose, dtype=int)
    dlts = np.zeros(n_dose, dtype=int)

    dose = int(np.clip(start_idx, 0, n_dose - 1))
    total_treated = 0

    prev_dose = dose
    stopped = False

    while not stopped and total_treated < max_n:
        n = min(cohort_size, max_n - total_treated)
        x = rng.binomial(1, true_curve[dose], size=n)
        treated[dose] += n
        dlts[dose] += int(x.sum())
        total_treated += n

        if n < cohort_size:
            break

        if treated[dose] == cohort_size:
            d = dlts[dose]
            if d == 0:
                prev_dose = dose
                dose = min(dose + 1, n_dose - 1)
                if dose == prev_dose and dose == n_dose - 1:
                    stopped = True
            elif d == 1:
                continue
            else:
                dose = max(dose - 1, 0)
                stopped = True
        else:
            d = dlts[dose]
            if d <= 1:
                prev_dose = dose
                dose = min(dose + 1, n_dose - 1)
                if dose == prev_dose and dose == n_dose - 1:
                    stopped = True
            else:
                dose = max(dose - 1, 0)
                stopped = True

    mtd = int(np.clip(dose, 0, n_dose - 1))
    tot_dlts = int(dlts.sum())
    return mtd, treated, total_treated, tot_dlts


def _ewoc_allowed(p_mean: np.ndarray, a: np.ndarray, b: np.ndarray, target: float, alpha: float) -> np.ndarray:
    # Pr(p > target) = 1 - CDF_beta(target; a, b)
    # Use a fast Monte Carlo approx to avoid SciPy dependency.
    # Small draws, adequate for filtering.
    draws = 200
    rng = np.random.default_rng(12345)
    samp = rng.beta(a[:, None], b[:, None], size=(len(a), draws))
    pr_over = (samp > target).mean(axis=1)
    return pr_over <= (1.0 - alpha)


def _simulate_crm_trial(
    rng: np.random.Generator,
    true_curve: np.ndarray,
    skeleton: np.ndarray,
    target: float,
    start_idx: int,
    cohort_size: int,
    max_n: int,
    burnin_until_first_dlt: bool,
    ewoc_enable: bool,
    ewoc_alpha: float,
    prior_sigma_theta: float,
    n_prior_start_no_dlt: int,
) -> Tuple[int, np.ndarray, int, int]:
    n_dose = len(true_curve)
    treated = np.zeros(n_dose, dtype=int)
    dlts = np.zeros(n_dose, dtype=int)

    dose = int(np.clip(start_idx, 0, n_dose - 1))
    total_treated = 0
    first_dlt_seen = False

    # --- incorporate "already treated at start dose with 0 DLT" ---
    if n_prior_start_no_dlt > 0:
        treated[dose] += int(n_prior_start_no_dlt)
        total_treated += int(n_prior_start_no_dlt)

    # --- map prior_sigma_theta -> prior strength (kappa) ---
    sigma = float(max(0.10, prior_sigma_theta))
    base_kappa = 6.0
    kappa = base_kappa / (sigma * sigma)
    kappa = float(np.clip(kappa, 0.5, 50.0))

    a0 = np.clip(skeleton * kappa, 0.5, None)
    b0 = np.clip((1.0 - skeleton) * kappa, 0.5, None)

    while total_treated < max_n:
        n = min(cohort_size, max_n - total_treated)
        x = rng.binomial(1, true_curve[dose], size=n)
        treated[dose] += n
        dlts[dose] += int(x.sum())
        total_treated += n

        if int(x.sum()) > 0:
            first_dlt_seen = True

        if burnin_until_first_dlt and not first_dlt_seen:
            dose = min(dose + 1, n_dose - 1)
            if dose == n_dose - 1 and total_treated >= max_n:
                break
            continue

        a = a0 + dlts
        b = b0 + treated - dlts
        p_mean = a / (a + b)

        cand = int(np.argmin(np.abs(p_mean - target)))

        if ewoc_enable:
            allowed = _ewoc_allowed(p_mean, a, b, target, float(ewoc_alpha))
            if not allowed[cand]:
                safe_idxs = np.where(allowed)[0]
                if len(safe_idxs) > 0:
                    cand = int(safe_idxs[np.argmin(np.abs(p_mean[safe_idxs] - target))])
                else:
                    cand = int(np.argmin(p_mean))

        if cand > dose + 1:
            cand = dose + 1
        if cand < dose - 1:
            cand = dose - 1
        dose = int(np.clip(cand, 0, n_dose - 1))

    a = a0 + dlts
    b = b0 + treated - dlts
    p_mean = a / (a + b)
    mtd = int(np.argmin(np.abs(p_mean - target)))

    tot_dlts = int(dlts.sum())
    return mtd, treated, total_treated, tot_dlts


def run_simulations(payload: Dict[str, Any]) -> Dict[str, Any]:
    true_curve = np.array(payload["true_curve"], dtype=float)
    n_dose = len(true_curve)

    target = float(payload["target"])
    start_idx = int(payload["start_dose_level"])
    n_sims = int(payload["n_sims"])
    seed = int(payload["seed"])

    prior_sigma_theta = float(payload.get("prior_sigma_theta", 1.0))
    n_prior_start_no_dlt = int(payload.get("n_prior_start_no_dlt", 0))


    max_n_6p3 = int(payload["max_n_6p3"])
    cohort_size_6p3 = int(payload["cohort_size_6p3"])

    skeleton = _get_skeleton(payload, n_dose)

    burnin_until_first_dlt = bool(payload.get("burnin_until_first_dlt", False))
    ewoc_enable = bool(payload.get("ewoc_enable", False))
    ewoc_alpha = float(payload.get("ewoc_alpha", 0.25))

    rng = np.random.default_rng(seed)

    mtd_counts_6 = np.zeros(n_dose, dtype=int)
    mtd_counts_c = np.zeros(n_dose, dtype=int)

    treated_sum_6 = np.zeros(n_dose, dtype=float)
    treated_sum_c = np.zeros(n_dose, dtype=float)

    total_treated_6 = 0
    total_treated_c = 0
    total_dlts_6 = 0
    total_dlts_c = 0

    for _ in range(n_sims):
        mtd6, tr6, n6, d6 = _simulate_6p3_trial(
            rng=rng,
            true_curve=true_curve,
            start_idx=start_idx,
            cohort_size=cohort_size_6p3,
            max_n=max_n_6p3,
        )
        mtd_counts_6[mtd6] += 1
        treated_sum_6 += tr6
        total_treated_6 += n6
        total_dlts_6 += d6

    mtdc, trc, nc, dc = _simulate_crm_trial(
        rng=rng,
        true_curve=true_curve,
        skeleton=skeleton,
        target=target,
        start_idx=start_idx,
        cohort_size=cohort_size_6p3,
        max_n=max_n_6p3,
        burnin_until_first_dlt=burnin_until_first_dlt,
        ewoc_enable=ewoc_enable,
        ewoc_alpha=ewoc_alpha,
        prior_sigma_theta=prior_sigma_theta,
        n_prior_start_no_dlt=n_prior_start_no_dlt,
    )

        mtd_counts_c[mtdc] += 1
        treated_sum_c += trc
        total_treated_c += nc
        total_dlts_c += dc

    mtd_probs_6p3 = (mtd_counts_6 / float(n_sims)).tolist()
    mtd_probs_crm = (mtd_counts_c / float(n_sims)).tolist()
    avg_n_per_dose_6p3 = (treated_sum_6 / float(n_sims)).tolist()
    avg_n_per_dose_crm = (treated_sum_c / float(n_sims)).tolist()

    p_dlt_per_patient_6p3 = float(total_dlts_6) / float(total_treated_6) if total_treated_6 > 0 else float("nan")
    p_dlt_per_patient_crm = float(total_dlts_c) / float(total_treated_c) if total_treated_c > 0 else float("nan")

    return {
        "mtd_probs_6p3": mtd_probs_6p3,
        "mtd_probs_crm": mtd_probs_crm,
        "avg_n_per_dose_6p3": avg_n_per_dose_6p3,
        "avg_n_per_dose_crm": avg_n_per_dose_crm,
        "p_dlt_per_patient_6p3": p_dlt_per_patient_6p3,
        "p_dlt_per_patient_crm": p_dlt_per_patient_crm,
    }
