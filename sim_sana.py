import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List

# ============================================================
# Helpers
# ============================================================

def safe_probs(x):
    x = np.asarray(x, dtype=float)
    return np.clip(x, 1e-12, 1 - 1e-12)

def find_true_mtd(true_p, target):
    true_p = np.asarray(true_p, dtype=float)
    return int(np.argmin(np.abs(true_p - target)))

# ============================================================
# dfcrm-like getprior (Lee & Cheung calibration)
# This matches dfcrm::getprior for empiric skeleton generation.
# ============================================================

def dfcrm_getprior(halfwidth: float, target: float, nu: int, nlevel: int) -> np.ndarray:
    """
    Empiric (power) skeleton calibration like dfcrm::getprior.

    nu is 1-based.
    """
    halfwidth = float(halfwidth)
    target = float(target)
    nu = int(nu)
    nlevel = int(nlevel)

    if not (0 < target < 1):
        raise ValueError("target must be in (0,1)")
    if halfwidth <= 0:
        raise ValueError("halfwidth must be > 0")
    if (target - halfwidth) <= 0 or (target + halfwidth) >= 1:
        raise ValueError("halfwidth too large for target")
    if not (1 <= nu <= nlevel):
        raise ValueError("nu must be between 1 and nlevel")

    dosescaled = np.full(nlevel, np.nan, dtype=float)
    dosescaled[nu - 1] = target

    # Downward
    for k in range(nu, 1, -1):
        b_k = np.log(np.log(target + halfwidth) / np.log(dosescaled[k - 1]))
        dosescaled[k - 2] = np.exp(np.log(target - halfwidth) / np.exp(b_k))

    # Upward
    if nu < nlevel:
        for k in range(nu, nlevel):
            b_k1 = np.log(np.log(target - halfwidth) / np.log(dosescaled[k - 1]))
            dosescaled[k] = np.exp(np.log(target + halfwidth) / np.exp(b_k1))

    return dosescaled

# ============================================================
# Gaussian copula for correlated Bernoulli (acute/subacute)
# ============================================================

def norm_cdf(z):
    # stable normal CDF via erf
    return 0.5 * (1.0 + np.math.erf(z / np.sqrt(2.0)))

def bvn_cdf_gauss_hermite(z1, z2, rho, gh_n=40):
    """
    Approximate BVN CDF Phi_2(z1,z2;rho) using 1D Gauss-Hermite.
    Works fine for our simulation needs.

    Uses conditional form:
    Phi2(z1,z2;rho) = ∫ Phi((z2 - rho*x)/sqrt(1-rho^2)) phi(x) dx, x from -inf..inf
    Transform integral with Hermite under exp(-t^2).
    """
    rho = float(rho)
    if abs(rho) >= 0.999:
        rho = 0.999 * np.sign(rho)

    x, w = np.polynomial.hermite.hermgauss(int(gh_n))
    # x here integrates exp(-x^2) f(x)
    # For standard normal integral: ∫ f(u) phi(u) du
    # map u = sqrt(2) x, du = sqrt(2) dx, phi(u) = (1/sqrt(2pi)) exp(-u^2/2) = (1/sqrt(pi)) exp(-x^2)
    # So ∫ f(u) phi(u) du = (1/sqrt(pi)) ∫ f(sqrt(2)x) exp(-x^2) dx ≈ (1/sqrt(pi)) Σ w_i f(sqrt(2)x_i)

    u = np.sqrt(2.0) * x
    denom = np.sqrt(1.0 - rho * rho)
    inner = (z2 - rho * u) / denom

    # Phi(inner) pointwise
    Phi_inner = np.array([norm_cdf(v) for v in inner], dtype=float)
    # integrate where u <= z1 by multiplying indicator I(u <= z1)
    ind = (u <= z1).astype(float)

    val = (1.0 / np.sqrt(np.pi)) * np.sum(w * Phi_inner * ind)
    return float(val)

def C1_gaussian(p1, p2, gamma, gh_n=40):
    """
    C(p1,p2;gamma) = P(U<=p1,V<=p2) under Gaussian copula correlation gamma.
    Equivalent to BVN CDF at z1=Phi^{-1}(p1), z2=Phi^{-1}(p2).
    """
    p1 = float(p1)
    p2 = float(p2)
    gamma = float(gamma)

    z1 = float(np.quantile(np.random.default_rng(0).standard_normal(300000), p1)) if False else None
    # We do not want Monte Carlo quantiles. Use scipy would be easiest, but keeping pure numpy.
    # Use an approximation: inverse error function via np.erfinv is available.
    z1 = np.sqrt(2.0) * float(np.erfinv(2.0 * p1 - 1.0))
    z2 = np.sqrt(2.0) * float(np.erfinv(2.0 * p2 - 1.0))

    return bvn_cdf_gauss_hermite(z1, z2, gamma, gh_n=gh_n)

def feasible_rho_bounds(Pa, Ps):
    Pa = np.asarray(Pa, dtype=float)
    Ps = np.asarray(Ps, dtype=float)
    out = np.zeros((len(Pa), 2), dtype=float)
    for i in range(len(Pa)):
        qa = 1 - Pa[i]
        qs = 1 - Ps[i]
        rho_min = -np.sqrt((Pa[i] * Ps[i]) / (qa * qs))
        rho_max =  np.sqrt((Pa[i] * qs) / (qa * Ps[i]))
        out[i, 0] = rho_min
        out[i, 1] = rho_max
    return out

def binary_rho_to_gamma(p_a, p_s, rho_target, gh_n=40):
    """
    Solve for Gaussian copula gamma such that resulting Bernoulli corr is rho_target.
    Uses bisection on gamma in (-0.999, 0.999).
    """
    p_a = float(p_a)
    p_s = float(p_s)
    rho_target = float(rho_target)

    q_a = 1 - p_a
    q_s = 1 - p_s
    denom = np.sqrt(p_a * q_a * p_s * q_s)

    def f(gamma):
        P11 = C1_gaussian(p_a, p_s, gamma, gh_n=gh_n)
        rho = (P11 - p_a * p_s) / denom
        return rho - rho_target

    lo, hi = -0.999, 0.999
    flo, fhi = f(lo), f(hi)

    # If numerical issues, fall back to 0
    if not np.isfinite(flo) or not np.isfinite(fhi):
        return 0.0

    # If target outside achievable due to numeric approx, clamp
    if flo > 0 and fhi > 0:
        return lo
    if flo < 0 and fhi < 0:
        return hi

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid == 0 or (hi - lo) < 1e-6:
            return float(mid)
        if np.sign(fmid) == np.sign(flo):
            lo, flo = mid, fmid
        else:
            hi, fhi = mid, fmid
    return float(0.5 * (lo + hi))

def pmf_from_copula(p_a, p_s, rho_target, gh_n=40):
    gamma = binary_rho_to_gamma(p_a, p_s, rho_target, gh_n=gh_n)
    P11 = C1_gaussian(p_a, p_s, gamma, gh_n=gh_n)
    P10 = p_a - P11
    P01 = p_s - P11
    P00 = 1 - p_a - p_s + P11
    probs = np.array([P00, P01, P10, P11], dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / probs.sum()
    return probs, gamma

def simulate_joint_acute_subacute(n, Pa, Ps, rho_target=0.0, gh_n=40, rng=None):
    """
    Returns (acute[n, K], subacute[n, K]) simulated per dose level.
    """
    if rng is None:
        rng = np.random.default_rng()

    Pa = np.asarray(Pa, dtype=float)
    Ps = np.asarray(Ps, dtype=float)
    K = len(Pa)

    acute = np.zeros((n, K), dtype=int)
    subac = np.zeros((n, K), dtype=int)

    bounds = feasible_rho_bounds(Pa, Ps)

    for k in range(K):
        # match R default r=2: rho=0, but keep safe within feasible bounds
        rho = float(rho_target)
        rho = max(bounds[k, 0] + 1e-6, min(bounds[k, 1] - 1e-6, rho))

        probs, _ = pmf_from_copula(Pa[k], Ps[k], rho, gh_n=gh_n)
        # outcomes order: 00, 01, 10, 11
        idx = rng.choice(4, size=n, replace=True, p=probs)
        acute[:, k] = (idx == 2) | (idx == 3)
        subac[:, k] = (idx == 1) | (idx == 3)

    return acute, subac

# ============================================================
# Acute-only CRM using Gauss–Hermite (same as your Streamlit one)
# Model: p_k(theta) = skeleton_k ^ exp(theta), theta ~ N(0, sigma^2)
# ============================================================

def posterior_via_gh(sigma, skeleton, n_per_level, y_per_level, gh_n=61):
    sk = safe_probs(skeleton)
    n = np.asarray(n_per_level, dtype=float)
    y = np.asarray(y_per_level, dtype=float)

    x, w = np.polynomial.hermite.hermgauss(int(gh_n))
    theta = float(sigma) * np.sqrt(2.0) * x

    P = sk[None, :] ** np.exp(theta)[:, None]
    P = np.clip(P, 1e-12, 1 - 1e-12)

    mask = n > 0
    if np.any(mask):
        nm = n[mask][None, :]
        ym = y[mask][None, :]
        Pm = P[:, mask]
        ll = (ym * np.log(Pm) + (nm - ym) * np.log1p(-Pm)).sum(axis=1)
    else:
        ll = np.zeros(P.shape[0], dtype=float)

    log_unnorm = np.log(w) + ll
    m = np.max(log_unnorm)
    unnorm = np.exp(log_unnorm - m)
    post_w = unnorm / np.sum(unnorm)

    return post_w, P

def crm_posterior(sigma, skeleton, n_per_level, y_per_level, target, gh_n=61):
    post_w, P = posterior_via_gh(sigma, skeleton, n_per_level, y_per_level, gh_n=gh_n)
    post_mean = (post_w[:, None] * P).sum(axis=0)
    overdose_prob = (post_w[:, None] * (P > target)).sum(axis=0)
    return post_mean, overdose_prob

def crm_pick_mtd_tried_only(
    sigma, skeleton, n_per_level, y_per_level, target, alpha_overdose, gh_n=61
):
    post_mean, overdose_prob = crm_posterior(
        sigma, skeleton, n_per_level, y_per_level, target, gh_n=gh_n
    )
    tried = np.where(np.asarray(n_per_level) > 0)[0]
    if tried.size == 0:
        return 0

    allowed = tried[overdose_prob[tried] < alpha_overdose]
    if allowed.size == 0:
        return int(tried.min())

    k = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])
    return k

# ============================================================
# Simulation that mirrors the R "Final.Fun" structure (no time-to-event)
# ============================================================

@dataclass
class Params:
    NREP: int = 1000
    N_patient: int = 27
    CO: int = 3
    start_dose_1based: int = 2       # R used p=2 initially
    burn_in_n: int = 6               # R does while(j<6)
    max_step_up: int = 1
    halfwidth: float = 0.1
    sigma: float = 1.0               # you can tune this
    alpha_overdose: float = 0.25     # you can tune this
    gh_n: int = 61
    copula_gh_n: int = 40
    rho_target: float = 0.0
    Wait_Time: float = 1.0           # only used for surrogate "trial time"

def run_one_trial_acute_crm(
    Pa_true: List[float],
    Ps_true: List[float],
    skeleton: List[float],
    target_acute: float,
    par: Params,
    rng: np.random.Generator
) -> Tuple[int, np.ndarray]:
    """
    Simulates one trial:
      - generates acute/subacute DLT tables (like the R data generation)
      - uses only acute outcomes for CRM decisions
      - burn-in: first burn_in_n patients at start dose
      - then cohorts of size CO with CRM updates
    Returns:
      selected_mtd_1based, n_treated_per_dose (length K)
    """
    K = len(Pa_true)

    acute_mat, subac_mat = simulate_joint_acute_subacute(
        n=par.N_patient,
        Pa=Pa_true,
        Ps=Ps_true,
        rho_target=par.rho_target,
        gh_n=par.copula_gh_n,
        rng=rng
    )

    # patient-level data as we enroll:
    dose_levels = np.zeros(par.N_patient, dtype=int)  # 0-based
    tox_acute = np.zeros(par.N_patient, dtype=int)

    start0 = par.start_dose_1based - 1
    start0 = int(np.clip(start0, 0, K - 1))

    # burn-in at fixed start dose
    j = 0
    while j < min(par.burn_in_n, par.N_patient):
        dose_levels[j] = start0
        tox_acute[j] = int(acute_mat[j, start0])
        j += 1

    # CRM state counts
    n_per = np.zeros(K, dtype=int)
    y_per = np.zeros(K, dtype=int)
    for i in range(j):
        d = dose_levels[i]
        n_per[d] += 1
        y_per[d] += tox_acute[i]

    current = int(dose_levels[j - 1]) if j > 0 else start0

    # now enroll remaining in cohorts
    while j < par.N_patient:
        # pick CRM mtd among tried doses only (conservative)
        mtd0 = crm_pick_mtd_tried_only(
            sigma=par.sigma,
            skeleton=skeleton,
            n_per_level=n_per,
            y_per_level=y_per,
            target=target_acute,
            alpha_overdose=par.alpha_overdose,
            gh_n=par.gh_n
        )

        # apply the R-style "no skipping upwards"
        proposed = int(mtd0)
        if current < proposed:
            proposed = current + par.max_step_up

        proposed = int(np.clip(proposed, 0, K - 1))

        # treat next cohort at proposed
        cohort_n = min(par.CO, par.N_patient - j)
        for _ in range(cohort_n):
            dose_levels[j] = proposed
            tox_acute[j] = int(acute_mat[j, proposed])

            n_per[proposed] += 1
            y_per[proposed] += tox_acute[j]
            j += 1

        current = proposed

    # final selected MTD (tried-only)
    final_mtd0 = crm_pick_mtd_tried_only(
        sigma=par.sigma,
        skeleton=skeleton,
        n_per_level=n_per,
        y_per_level=y_per,
        target=target_acute,
        alpha_overdose=par.alpha_overdose,
        gh_n=par.gh_n
    )
    final_mtd_1based = int(final_mtd0) + 1

    return final_mtd_1based, n_per

def final_fun_python(
    Pa_true: List[float],
    Ps_true: List[float],
    prior_target_acute: float,
    prior_mtd_acute_1based: int,
    par: Params,
    seed: int = 123
) -> Dict[str, np.ndarray]:
    """
    Mimics the R Final.Fun outputs (without time-to-event mechanics).

    Returns dict with:
      - p_select (K,)
      - mean_trial_time (scalar surrogate)
      - mean_treated_frac (K,)
    """
    rng = np.random.default_rng(int(seed))
    K = len(Pa_true)

    # skeleton like dfcrm::getprior
    skeleton = dfcrm_getprior(
        halfwidth=par.halfwidth,
        target=prior_target_acute,
        nu=prior_mtd_acute_1based,
        nlevel=K
    ).tolist()

    mtd_counts = np.zeros(K, dtype=int)
    treated_sum = np.zeros(K, dtype=float)

    for _ in range(par.NREP):
        mtd_1based, n_per = run_one_trial_acute_crm(
            Pa_true=Pa_true,
            Ps_true=Ps_true,
            skeleton=skeleton,
            target_acute=prior_target_acute,  # if you want R-like, use target_acute here
            par=par,
            rng=rng
        )
        mtd_counts[mtd_1based - 1] += 1
        treated_sum += n_per / float(par.N_patient)

    p_select = mtd_counts / float(par.NREP)
    mean_treated_frac = treated_sum / float(par.NREP)

    # surrogate trial time (since you said: no time-to-event)
    mean_trial_time = (par.N_patient - 1) * par.Wait_Time

    return {
        "p_select": p_select,
        "mean_trial_time": np.array([mean_trial_time], dtype=float),
        "mean_treated_frac": mean_treated_frac,
        "skeleton": np.array(skeleton, dtype=float)
    }

# ============================================================
# Example: your scenario (same as your R snippet)
# ============================================================

if __name__ == "__main__":
    Pa_true = [0.01, 0.02, 0.12, 0.20, 0.35]
    Ps_true = [0.15, 0.22, 0.28, 0.35, 0.40]

    par = Params(
        NREP=1000,
        N_patient=27,
        CO=3,
        start_dose_1based=2,
        burn_in_n=6,
        max_step_up=1,
        halfwidth=0.1,
        sigma=1.0,
        alpha_overdose=0.25,
        gh_n=61,
        copula_gh_n=40,
        rho_target=0.0,
        Wait_Time=1.0
    )

    out = final_fun_python(
        Pa_true=Pa_true,
        Ps_true=Ps_true,
        prior_target_acute=0.15,
        prior_mtd_acute_1based=3,
        par=par,
        seed=123
    )

    print("Skeleton:", np.round(out["skeleton"], 4))
    print("P(select each dose as MTD):", np.round(out["p_select"], 4))
    print("Mean trial time (surrogate):", out["mean_trial_time"][0])
    print("Mean % treated per dose:", np.round(out["mean_treated_frac"], 4))
