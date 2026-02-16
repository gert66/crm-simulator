import numpy as np

# ============================================================
# Defaults (R-aligned where possible)
# ============================================================

DOSE_LABELS = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]
DEFAULT_TRUE_P = [0.01, 0.02, 0.12, 0.20, 0.35]

DEFAULTS = {
    # Essentials
    "target": 0.15,
    "start_level": 1,          # 0-based: Level 1 = 5×5
    "max_n_crm": 27,
    "cohort_size": 3,

    # Sim settings
    "n_sims": 500,
    "seed": 123,

    # CRM integration + movement
    "gh_n": 61,
    "max_step": 1,

    # CRM policy
    "sigma": 1.0,
    "burn_in": True,
    "ewoc_on": False,
    "ewoc_alpha": 0.25,
    "guardrail": True,         # next <= highest tried + 1
    "final_tried_only": True,  # final MTD restricted to tried

    # 6+3 settings
    "max_n_63": 27,            # you requested default = 27
    "accept_rule_63": 1,
    "show_debug": False,
}

# ============================================================
# Helpers
# ============================================================

def safe_probs(x):
    x = np.asarray(x, dtype=float)
    return np.clip(x, 1e-6, 1 - 1e-6)

def simulate_bernoulli(n, p, rng):
    return rng.binomial(1, p, size=int(n))

def find_true_mtd(true_p, target):
    true_p = np.asarray(true_p, dtype=float)
    return int(np.argmin(np.abs(true_p - float(target))))

# ============================================================
# dfcrm getprior port (empiric / logistic)
# ============================================================

def dfcrm_getprior(halfwidth, target, nu, nlevel, model="empiric", intcpt=3.0):
    halfwidth = float(halfwidth)
    target = float(target)
    nu = int(nu)
    nlevel = int(nlevel)
    intcpt = float(intcpt)

    if not (0 < target < 1):
        raise ValueError("target must be in (0, 1).")
    if halfwidth <= 0:
        raise ValueError("halfwidth must be > 0.")
    if (target - halfwidth) <= 0 or (target + halfwidth) >= 1:
        raise ValueError("halfwidth too large: target±halfwidth must stay within (0,1).")
    if not (1 <= nu <= nlevel):
        raise ValueError("nu must be between 1 and nlevel (inclusive).")

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
            b_k = np.log((np.log((target + halfwidth) / (1 - target - halfwidth)) - intcpt) / dosescaled[k - 1])
            dosescaled[k - 2] = (np.log((target - halfwidth) / (1 - target + halfwidth)) - intcpt) / np.exp(b_k)
        for k in range(nu, nlevel):
            b_k1 = np.log((np.log((target - halfwidth) / (1 - target + halfwidth)) - intcpt) / dosescaled[k - 1])
            dosescaled[k] = (np.log((target + halfwidth) / (1 - target - halfwidth)) - intcpt) / np.exp(b_k1)
        prior = (1 + np.exp(-intcpt - dosescaled)) ** (-1)
        return prior

    raise ValueError('model must be "empiric" or "logistic".')

# ============================================================
# 6+3 design (simple)
# ============================================================

def run_6plus3(true_p, start_level=1, max_n=27, accept_max_dlt=1, rng=None):
    """
    Simple 6+3:
      - Treat 6 at current level.
      - 0/6: escalate
      - 1/6: add 3 (to 9); accept if total DLTs <= accept_max_dlt
      - >=2/6 or expansion fails: de-escalate and stop
    Returns: selected_level (0-based), n_per_level, total_dlts
    """
    if rng is None:
        rng = np.random.default_rng()

    true_p = np.asarray(true_p, dtype=float)
    n_levels = len(true_p)

    level = int(start_level)
    n_per = np.zeros(n_levels, dtype=int)
    y_per = np.zeros(n_levels, dtype=int)

    total_n = 0
    last_acceptable = None

    while total_n < int(max_n):
        n_add = min(6, int(max_n) - total_n)
        out6 = simulate_bernoulli(n_add, true_p[level], rng)

        n_per[level] += n_add
        y_per[level] += int(out6.sum())
        total_n += n_add

        if n_add < 6:
            break

        d6 = int(out6.sum())

        if d6 == 0:
            last_acceptable = level
            if level < n_levels - 1:
                level += 1
                continue
            break

        if d6 == 1:
            n_add2 = min(3, int(max_n) - total_n)
            out3 = simulate_bernoulli(n_add2, true_p[level], rng)

            n_per[level] += n_add2
            y_per[level] += int(out3.sum())
            total_n += n_add2

            if n_add2 < 3:
                break

            d9 = d6 + int(out3.sum())
            if d9 <= int(accept_max_dlt):
                last_acceptable = level
                if level < n_levels - 1:
                    level += 1
                    continue
                break
            else:
                if level > 0:
                    level -= 1
                break

        if level > 0:
            level -= 1
        break

    selected = 0 if last_acceptable is None else int(last_acceptable)
    return selected, n_per, int(y_per.sum())

# ============================================================
# CRM posterior via Gauss–Hermite quadrature (acute-only)
# p_k(theta) = skeleton_k ^ exp(theta), theta ~ N(0, sigma^2)
# ============================================================

def posterior_via_gh(sigma, skeleton, n_per_level, dlt_per_level, gh_n=61):
    sk = safe_probs(skeleton)
    n = np.asarray(n_per_level, dtype=float)
    y = np.asarray(dlt_per_level, dtype=float)

    x, w = np.polynomial.hermite.hermgauss(int(gh_n))
    theta = float(sigma) * np.sqrt(2.0) * x

    P = sk[None, :] ** np.exp(theta)[:, None]
    P = safe_probs(P)

    ll = (y[None, :] * np.log(P) + (n[None, :] - y[None, :]) * np.log(1 - P)).sum(axis=1)

    log_unnorm = np.log(w) + ll
    m = np.max(log_unnorm)
    unnorm = np.exp(log_unnorm - m)
    post_w = unnorm / np.sum(unnorm)
    return post_w, P

def crm_posterior_summaries(sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=61):
    post_w, P = posterior_via_gh(sigma, skeleton, n_per_level, dlt_per_level, gh_n=gh_n)
    post_mean = (post_w[:, None] * P).sum(axis=0)
    overdose_prob = (post_w[:, None] * (P > float(target))).sum(axis=0)
    return post_mean, overdose_prob

def crm_choose_next(
    sigma, skeleton, n_per_level, dlt_per_level,
    current_level, target,
    ewoc_alpha=None,
    max_step=1, gh_n=61,
    guardrail=True,
    highest_tried=None,
):
    post_mean, overdose_prob = crm_posterior_summaries(
        sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=gh_n
    )

    if ewoc_alpha is None:
        allowed = np.arange(len(skeleton))
    else:
        allowed = np.where(overdose_prob < float(ewoc_alpha))[0]

    if allowed.size == 0:
        return 0, post_mean, overdose_prob, allowed

    k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - float(target)))])

    # step limiting
    k_star = int(np.clip(k_star, int(current_level) - int(max_step), int(current_level) + int(max_step)))

    # guardrail: cannot jump beyond highest tried + 1
    if guardrail and highest_tried is not None:
        k_star = int(min(k_star, int(highest_tried) + 1))

    k_star = int(np.clip(k_star, 0, len(skeleton) - 1))
    return k_star, post_mean, overdose_prob, allowed

def crm_select_mtd(
    sigma, skeleton, n_per_level, dlt_per_level,
    target, ewoc_alpha=None, gh_n=61,
    tried_only=True
):
    post_mean, overdose_prob = crm_posterior_summaries(
        sigma, skeleton, n_per_level, dlt_per_level, target, gh_n=gh_n
    )

    if ewoc_alpha is None:
        allowed = np.arange(len(skeleton))
    else:
        allowed = np.where(overdose_prob < float(ewoc_alpha))[0]

    if allowed.size == 0:
        return 0

    if tried_only:
        tried = np.where(np.asarray(n_per_level) > 0)[0]
        if tried.size > 0:
            allowed = np.intersect1d(allowed, tried)
            if allowed.size == 0:
                return int(tried[0])

    return int(allowed[np.argmin(np.abs(post_mean[allowed] - float(target)))])

def run_crm_trial(
    true_p, target, skeleton,
    sigma=1.0,
    start_level=1,
    max_n=27,
    cohort_size=3,
    max_step=1,
    gh_n=61,
    guardrail=True,
    final_tried_only=True,
    ewoc_on=False,
    ewoc_alpha=0.25,
    burn_in_until_first_dlt=True,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    true_p = np.asarray(true_p, dtype=float)
    n_levels = len(true_p)

    level = int(start_level)
    n_per = np.zeros(n_levels, dtype=int)
    y_per = np.zeros(n_levels, dtype=int)

    highest_tried = -1
    any_dlt_seen = False

    while int(n_per.sum()) < int(max_n):
        n_add = min(int(cohort_size), int(max_n) - int(n_per.sum()))
        out = simulate_bernoulli(n_add, true_p[level], rng)

        n_per[level] += n_add
        y_per[level] += int(out.sum())
        highest_tried = max(highest_tried, level)

        if int(out.sum()) > 0:
            any_dlt_seen = True

        if n_add < int(cohort_size):
            break

        if burn_in_until_first_dlt and (not any_dlt_seen):
            if level < n_levels - 1:
                level += 1
            continue

        ewoc_alpha_eff = float(ewoc_alpha) if ewoc_on else None
        next_level, _, _, _ = crm_choose_next(
            sigma=sigma,
            skeleton=skeleton,
            n_per_level=n_per,
            dlt_per_level=y_per,
            current_level=level,
            target=target,
            ewoc_alpha=ewoc_alpha_eff,
            max_step=max_step,
            gh_n=gh_n,
            guardrail=guardrail,
            highest_tried=highest_tried
        )
        level = int(next_level)

    ewoc_alpha_eff = float(ewoc_alpha) if ewoc_on else None
    selected = crm_select_mtd(
        sigma=sigma,
        skeleton=skeleton,
        n_per_level=n_per,
        dlt_per_level=y_per,
        target=target,
        ewoc_alpha=ewoc_alpha_eff,
        gh_n=gh_n,
        tried_only=final_tried_only
    )

    return int(selected), n_per, int(y_per.sum())

# ============================================================
# Batch simulation and summary (for results panel)
# ============================================================

def simulate_many(true_p, skeleton, settings):
    rng = np.random.default_rng(int(settings["seed"]))
    ns = int(settings["n_sims"])
    n_levels = len(true_p)

    sel_63 = np.zeros(n_levels, dtype=int)
    sel_crm = np.zeros(n_levels, dtype=int)

    nmat_63 = np.zeros((ns, n_levels), dtype=int)
    nmat_crm = np.zeros((ns, n_levels), dtype=int)

    dlts_63 = np.zeros(ns, dtype=int)
    dlts_crm = np.zeros(ns, dtype=int)

    for s in range(ns):
        chosen63, n63, y63 = run_6plus3(
            true_p=true_p,
            start_level=int(settings["start_level"]),
            max_n=int(settings["max_n_63"]),
            accept_max_dlt=int(settings["accept_rule_63"]),
            rng=rng
        )

        chosenc, nc, yc = run_crm_trial(
            true_p=true_p,
            target=float(settings["target"]),
            skeleton=skeleton,
            sigma=float(settings["sigma"]),
            start_level=int(settings["start_level"]),
            max_n=int(settings["max_n_crm"]),
            cohort_size=int(settings["cohort_size"]),
            max_step=int(settings["max_step"]),
            gh_n=int(settings["gh_n"]),
            guardrail=bool(settings["guardrail"]),
            final_tried_only=bool(settings["final_tried_only"]),
            ewoc_on=bool(settings["ewoc_on"]),
            ewoc_alpha=float(settings["ewoc_alpha"]),
            burn_in_until_first_dlt=bool(settings["burn_in"]),
            rng=rng,
        )

        sel_63[chosen63] += 1
        sel_crm[chosenc] += 1

        nmat_63[s, :] = n63
        nmat_crm[s, :] = nc

        dlts_63[s] = y63
        dlts_crm[s] = yc

    p_mtd_63 = sel_63 / float(ns)
    p_mtd_crm = sel_crm / float(ns)

    avg_n_63 = np.mean(nmat_63, axis=0)
    avg_n_crm = np.mean(nmat_crm, axis=0)

    # "DLT probability per patient" (what you asked): total DLTs / total patients
    dlt_prob_63 = float(dlts_63.sum()) / float(nmat_63.sum())
    dlt_prob_crm = float(dlts_crm.sum()) / float(nmat_crm.sum())

    return {
        "p_mtd_63": p_mtd_63,
        "p_mtd_crm": p_mtd_crm,
        "avg_n_63": avg_n_63,
        "avg_n_crm": avg_n_crm,
        "dlt_prob_63": dlt_prob_63,
        "dlt_prob_crm": dlt_prob_crm,
    }
