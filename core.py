from __future__ import annotations
import numpy as np


def pick_true_mtd(true_p, target):
    true_p = np.asarray(true_p, dtype=float)
    return int(np.argmin(np.abs(true_p - target)))


def run_six_plus_three(true_p, start=0, max_n=27, cohort=3, rng=None):
    """
    Simple 3+3-style escalation with max_n cap.
    Returns selected dose index and n treated per dose and DLTs.
    """
    if rng is None:
        rng = np.random.default_rng()

    K = len(true_p)
    n = np.zeros(K, dtype=int)
    dlt = np.zeros(K, dtype=int)

    d = start
    while True:
        # treat one cohort
        c = min(cohort, max_n - n.sum())
        if c <= 0:
            break
        tox = rng.binomial(1, true_p[d], size=c)
        n[d] += c
        dlt[d] += tox.sum()

        # decision rules (classic 3+3 flavor)
        if c < cohort:
            break

        if dlt[d] == 0 and n[d] >= cohort:
            if d < K - 1:
                d += 1
                continue
            break

        if dlt[d] == 1 and n[d] == cohort:
            # expand to 6
            continue

        if n[d] >= 6:
            # if <=1/6 DLT escalate else de-escalate
            if dlt[d] <= 1 and d < K - 1:
                d += 1
                continue
            break

        if dlt[d] >= 2:
            # stop, select previous dose if possible
            if d > 0:
                d -= 1
            break

    selected = int(d)
    return selected, n, dlt


def skeleton_empiric(K, target, nu_1based, delta):
    # Simple monotone skeleton around target with halfwidth delta
    nu = int(np.clip(nu_1based - 1, 0, K - 1))
    p = np.zeros(K, dtype=float)
    for i in range(K):
        p[i] = target + (i - nu) * delta
    return np.clip(p, 0.001, 0.999)


def skeleton_logistic(K, target, nu_1based, intercept):
    # Basic logistic curve passing near target at nu
    nu = int(np.clip(nu_1based - 1, 0, K - 1))
    x = np.arange(K) - nu
    # slope chosen to keep modest increase
    slope = 1.0
    logits = intercept + slope * x
    p = 1 / (1 + np.exp(-logits))
    # rescale so p[nu] ~= target
    p_nu = p[nu]
    if p_nu > 0:
        p = np.clip(p * (target / p_nu), 0.001, 0.999)
    return p


def run_crm_trial(true_p, skeleton_p, target, start=0, max_n=27, cohort=3,
                  burn_in_first_dlt=False, rng=None):
    """
    Lightweight CRM-like rule:
    - treat cohorts
    - update empirical tox rate at visited doses
    - choose next dose minimizing |posterior-ish estimate - target|
    This is not a full Bayesian CRM, but it is stable and matches your UI needs.
    """
    if rng is None:
        rng = np.random.default_rng()

    K = len(true_p)
    n = np.zeros(K, dtype=int)
    dlt = np.zeros(K, dtype=int)

    d = start
    seen_dlt = False

    while n.sum() < max_n:
        c = min(cohort, max_n - n.sum())
        tox = rng.binomial(1, true_p[d], size=c)
        n[d] += c
        dlt[d] += tox.sum()
        if tox.sum() > 0:
            seen_dlt = True

        if burn_in_first_dlt and not seen_dlt:
            # during burn-in, only escalate by 1 each cohort
            d = min(d + 1, K - 1)
            continue

        # "posterior-ish" estimate: blend skeleton with observed
        obs = np.where(n > 0, dlt / np.maximum(n, 1), np.nan)
        est = skeleton_p.copy()
        w = np.clip(n / 6.0, 0.0, 1.0)  # more weight with more data
        for i in range(K):
            if np.isfinite(obs[i]):
                est[i] = (1 - w[i]) * skeleton_p[i] + w[i] * obs[i]

        # choose dose closest to target, with no skipping more than 1 level
        best = int(np.argmin(np.abs(est - target)))
        if best > d + 1:
            best = d + 1
        if best < d - 1:
            best = d - 1
        d = int(np.clip(best, 0, K - 1))

    selected = int(d)
    return selected, n, dlt


def simulate(params):
    rng = np.random.default_rng(int(params["seed"]))

    true_p = np.asarray(params["true_p"], dtype=float)
    K = len(true_p)
    target = float(params["target"])

    true_mtd = pick_true_mtd(true_p, target)

    # skeleton
    if params["skeleton_model"] == "logistic":
        skel = skeleton_logistic(K, params["prior_target"], params["prior_mtd_nu"], params["logit_intercept"])
    else:
        skel = skeleton_empiric(K, params["prior_target"], params["prior_mtd_nu"], params["delta"])

    n_sims = int(params["n_sims"])

    sel_633 = np.zeros(K, dtype=int)
    sel_crm = np.zeros(K, dtype=int)
    n_633 = np.zeros(K, dtype=float)
    n_crm = np.zeros(K, dtype=float)

    dlt_total_633 = 0
    dlt_total_crm = 0
    pt_total_633 = 0
    pt_total_crm = 0

    for _ in range(n_sims):
        s1, n1, d1 = run_six_plus_three(
            true_p=true_p,
            start=int(params["start_dose"]),
            max_n=int(params["sixplus3_max_n"]),
            cohort=3,
            rng=rng,
        )
        sel_633[s1] += 1
        n_633 += n1
        dlt_total_633 += int(d1.sum())
        pt_total_633 += int(n1.sum())

        s2, n2, d2 = run_crm_trial(
            true_p=true_p,
            skeleton_p=skel,
            target=target,
            start=int(params["start_dose"]),
            max_n=int(params["crm_max_n"]),
            cohort=int(params["crm_cohort"]),
            burn_in_first_dlt=bool(params["burn_in_first_dlt"]),
            rng=rng,
        )
        sel_crm[s2] += 1
        n_crm += n2
        dlt_total_crm += int(d2.sum())
        pt_total_crm += int(n2.sum())

    out = {
        "true_mtd": true_mtd,
        "skeleton": skel,
        "p_select_633": sel_633 / n_sims,
        "p_select_crm": sel_crm / n_sims,
        "avg_n_633": n_633 / n_sims,
        "avg_n_crm": n_crm / n_sims,
        "p_dlt_per_patient_633": (dlt_total_633 / pt_total_633) if pt_total_633 > 0 else 0.0,
        "p_dlt_per_patient_crm": (dlt_total_crm / pt_total_crm) if pt_total_crm > 0 else 0.0,
    }
    return out
