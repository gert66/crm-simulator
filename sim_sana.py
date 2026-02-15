import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

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
    return int(np.argmin(np.abs(true_p - target)))

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)

# ============================================================
# dfcrm getprior port
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

def run_6plus3(true_p, start_level=1, max_n=36, accept_max_dlt=1, rng=None):
    """
    Simple 6+3:
      - Treat 6 at current level.
      - If 0/6 DLT: escalate.
      - If 1/6 DLT: expand by 3 (total 9); accept if <= accept_max_dlt.
      - If >=2/6 DLT OR expansion fails: de-escalate and stop.
    Returns:
      selected_level (0-based), n_per_level, total_dlts
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

        # d6 >= 2
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
    overdose_prob = (post_w[:, None] * (P > target)).sum(axis=0)
    return post_mean, overdose_prob

def crm_choose_next(
    sigma, skeleton, n_per_level, dlt_per_level,
    current_level, target,
    ewoc_alpha=None,
    max_step=1, gh_n=61,
    enforce_highest_tried_plus_one=True,
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

    k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

    # step limiting
    k_star = int(np.clip(k_star, current_level - int(max_step), current_level + int(max_step)))

    # guardrail
    if enforce_highest_tried_plus_one and highest_tried is not None:
        k_star = int(min(k_star, int(highest_tried) + 1))

    k_star = int(np.clip(k_star, 0, len(skeleton) - 1))
    return k_star, post_mean, overdose_prob, allowed

def crm_select_mtd(
    sigma, skeleton, n_per_level, dlt_per_level,
    target, ewoc_alpha=None, gh_n=61,
    restrict_to_tried=True
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

    if restrict_to_tried:
        tried = np.where(np.asarray(n_per_level) > 0)[0]
        if tried.size > 0:
            allowed = np.intersect1d(allowed, tried)
            if allowed.size == 0:
                return int(tried[0])

    return int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

def run_crm_trial(
    true_p, target, skeleton,
    sigma=1.0,
    start_level=1,
    max_n=27,
    cohort_size=3,
    max_step=1,
    gh_n=61,
    enforce_guardrail=True,
    restrict_final_mtd_to_tried=True,
    ewoc_on=False,
    ewoc_alpha=0.25,
    burn_in_until_first_dlt=True,
    rng=None,
    debug=False
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
    debug_rows = []

    while int(n_per.sum()) < int(max_n):
        n_add = min(int(cohort_size), int(max_n) - int(n_per.sum()))
        out = simulate_bernoulli(n_add, true_p[level], rng)

        n_per[level] += n_add
        y_per[level] += int(out.sum())
        highest_tried = max(highest_tried, level)

        if int(out.sum()) > 0:
            any_dlt_seen = True

        if debug:
            debug_rows.append({
                "treated_level": level,
                "cohort_n": int(n_add),
                "cohort_dlts": int(out.sum()),
                "any_dlt_seen": bool(any_dlt_seen),
            })

        if n_add < int(cohort_size):
            break

        # simplified R-like burning phase
        if burn_in_until_first_dlt and (not any_dlt_seen):
            if level < n_levels - 1:
                level = level + 1
            continue

        ewoc_alpha_eff = float(ewoc_alpha) if ewoc_on else None
        next_level, post_mean, od_prob, allowed = crm_choose_next(
            sigma=sigma,
            skeleton=skeleton,
            n_per_level=n_per,
            dlt_per_level=y_per,
            current_level=level,
            target=target,
            ewoc_alpha=ewoc_alpha_eff,
            max_step=max_step,
            gh_n=gh_n,
            enforce_highest_tried_plus_one=enforce_guardrail,
            highest_tried=highest_tried
        )

        if debug:
            debug_rows[-1].update({
                "next_level": int(next_level),
                "allowed_levels": ",".join([str(int(a)) for a in allowed]),
                "post_mean": [float(x) for x in post_mean],
                "od_prob": [float(x) for x in od_prob],
                "highest_tried": int(highest_tried),
            })

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
        restrict_to_tried=restrict_final_mtd_to_tried
    )

    return int(selected), n_per, int(y_per.sum()), debug_rows

# ============================================================
# Defaults (R-aligned + fixed 6+3 defaults)
# ============================================================

st.set_page_config(page_title="6+3 vs CRM (prior playground)", layout="wide")

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]
DEFAULT_TRUE_P = [0.01, 0.02, 0.12, 0.20, 0.35]

R_DEFAULTS = {
    # Essentials
    "target": 0.15,                 # R example target.acute
    "start_level": 1,               # default start 5×5 (R p<-2 1-based)
    "max_n": 27,                    # R N.patient
    "cohort_size": 3,               # R CO
    # Prior playground
    "prior_model": "empiric",
    "prior_target": 0.15,           # R prior.target.acute
    "halfwidth": 0.10,              # R getprior halfwidth
    "prior_nu": 3,                  # R prior.MTD.acute (1-based)
    "logistic_intcpt": 3.0,
    # Key CRM knobs
    "sigma": 1.0,                   # not explicit in your R snippet
    "burn_in": True,
    "ewoc_on": False,
    "ewoc_alpha": 0.25,
    # Advanced (CRM)
    "gh_n": 61,
    "max_step": 1,
    "enforce_guardrail": True,
    "restrict_final_mtd": True,
    "show_debug": False,
    # Advanced (sim)
    "n_sims": 500,
    "seed": 123,
    # 6+3 (advanced)
    "max_n_63": 36,
    "accept_rule_63": 1,
}

def init_state():
    for k, v in R_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    for i in range(5):
        st.session_state.setdefault(f"true_{i}", float(DEFAULT_TRUE_P[i]))

def reset_all_defaults():
    for k, v in R_DEFAULTS.items():
        st.session_state[k] = v
    # True curve: keep as-is by default (usually you want to keep it while tweaking priors)
    # If you want reset to also reset true curve, uncomment:
    # for i in range(5):
    #     st.session_state[f"true_{i}"] = float(DEFAULT_TRUE_P[i])

init_state()

# ============================================================
# UI
# ============================================================

st.title("Dose Escalation Simulator: 6+3 vs CRM (acute-only)")
st.caption("Prior playground is central. Advanced settings are tucked away.")

# Top bar: essentials + reset
st.subheader("Essentials")
top1, top2, top3, top4, top5 = st.columns([1.0, 1.0, 1.0, 1.0, 0.9], gap="large")

with top1:
    st.number_input(
        "Study target (acute)",
        min_value=0.05, max_value=0.50, step=0.01,
        key="target",
        help="R reference: target.acute = 0.15 in your example."
    )

with top2:
    st.selectbox(
        "Start dose",
        options=list(range(0, 5)),
        format_func=lambda i: f"Level {i} ({dose_labels[i]})",
        key="start_level",
        help="Default is Level 1 = 5×5 (R p <- 2 in 1-based indexing)."
    )

with top3:
    st.number_input(
        "Max sample size (CRM)",
        min_value=12, max_value=200, step=3,
        key="max_n",
        help="R reference: N.patient = 27."
    )

with top4:
    st.number_input(
        "CRM cohort size",
        min_value=1, max_value=12, step=1,
        key="cohort_size",
        help="R reference: CO = 3."
    )

with top5:
    st.write("")
    st.write("")
    if st.button("Reset to R defaults"):
        reset_all_defaults()
        st.rerun()

st.divider()

# Workbench: left true + key knobs, right playground + plot
left, right = st.columns([1.05, 1.35], gap="large")

with left:
    st.subheader("True acute DLT curve")
    manual_true = st.toggle("Edit true curve", value=True)

    true_p = []
    for i, lab in enumerate(dose_labels):
        true_p.append(float(st.number_input(
            f"True P(DLT) at {lab}",
            min_value=0.0, max_value=1.0, step=0.01,
            key=f"true_{i}",
            disabled=(not manual_true),
            help="Ground truth used to simulate Bernoulli DLT outcomes."
        )))

    true_mtd = find_true_mtd(true_p, float(st.session_state["target"]))
    st.info(f"True MTD (closest to target {float(st.session_state['target']):.2f}) = Level {true_mtd} ({dose_labels[true_mtd]})")

    st.subheader("Key CRM knobs")
    st.slider(
        "Prior sigma on theta",
        min_value=0.2, max_value=5.0, step=0.1,
        key="sigma",
        help="Controls prior looseness: larger sigma means a wider prior on theta."
    )

    st.toggle(
        "Burn-in until first DLT (R-like)",
        key="burn_in",
        help="Simplified mimic of the R burning phase: escalate cohort-wise until first DLT, then run CRM."
    )

    st.toggle(
        "Enable EWOC overdose control",
        key="ewoc_on",
        help="If enabled: allowed dose k must satisfy P(p_k > target | data) < alpha."
    )

    st.slider(
        "EWOC alpha",
        min_value=0.05, max_value=0.99, step=0.01,
        key="ewoc_alpha",
        disabled=(not st.session_state["ewoc_on"]),
        help="Only used if EWOC is enabled."
    )

with right:
    st.subheader("Prior playground")
    p1, p2 = st.columns([1.0, 1.0], gap="medium")

    with p1:
        st.radio(
            "Skeleton model",
            options=["empiric", "logistic"],
            horizontal=True,
            key="prior_model",
            help="How the skeleton is generated (dfcrm-style)."
        )
        st.slider(
            "Prior target (skeleton calibration)",
            min_value=0.05, max_value=0.50, step=0.01,
            key="prior_target",
            help="R reference: prior.target.acute = 0.15."
        )
        st.slider(
            "Halfwidth (delta)",
            min_value=0.01, max_value=0.30, step=0.01,
            key="halfwidth",
            help="R reference: halfwidth = 0.10 in getprior."
        )
        st.slider(
            "Prior MTD (nu, 1-based)",
            min_value=1, max_value=5, step=1,
            key="prior_nu",
            help="R reference: prior.MTD.acute = 3."
        )
        st.slider(
            "Logistic intercept (only if logistic)",
            min_value=0.0, max_value=10.0, step=0.1,
            key="logistic_intcpt",
            help="Only relevant for logistic skeleton generation."
        )

        skeleton = dfcrm_getprior(
            halfwidth=float(st.session_state["halfwidth"]),
            target=float(st.session_state["prior_target"]),
            nu=int(st.session_state["prior_nu"]),
            nlevel=5,
            model=str(st.session_state["prior_model"]),
            intcpt=float(st.session_state["logistic_intcpt"]),
        ).tolist()

        st.caption("Skeleton: " + ", ".join([f"{v:.3f}" for v in skeleton]))

    with p2:
        st.markdown("**Preview (True vs Prior)**")
        fig, ax = plt.subplots(figsize=(5.8, 2.8), dpi=160)
        x = np.arange(5)
        target_val = float(st.session_state["target"])
        ax.plot(x, true_p, marker="o", linewidth=1.6, label="True P(DLT)")
        ax.plot(x, skeleton, marker="o", linewidth=1.6, label="Prior (skeleton)")
        ax.axhline(target_val, linewidth=1, alpha=0.6)
        ax.text(0.05, target_val + 0.01, f"Target = {target_val:.2f}", fontsize=8)
        ax.axvline(true_mtd, linewidth=1, alpha=0.35)
        ax.text(true_mtd + 0.05, 0.92, "True MTD", fontsize=8, transform=ax.get_xaxis_transform())
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, min(1.0, max(max(true_p), max(skeleton), target_val) * 1.25 + 0.02))
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        st.pyplot(fig, clear_figure=True)

st.divider()

# Advanced settings (CRM + sim + 6+3)
with st.expander("Advanced settings", expanded=False):
    a1, a2, a3 = st.columns([1.0, 1.0, 1.0], gap="large")

    with a1:
        st.number_input(
            "Number of simulated trials",
            min_value=50, max_value=5000, step=50,
            key="n_sims",
            help="More trials = smoother estimates, slower runtime."
        )
        st.number_input(
            "Random seed",
            min_value=1, max_value=10_000_000, step=1,
            key="seed",
            help="Controls reproducibility."
        )

    with a2:
        st.selectbox(
            "Gauss–Hermite points",
            options=[31, 41, 61, 81],
            index=[31, 41, 61, 81].index(int(st.session_state["gh_n"])),
            key="gh_n",
            help="Posterior integration accuracy vs speed."
        )
        st.selectbox(
            "Max dose step per CRM update",
            options=[1, 2],
            index=[1, 2].index(int(st.session_state["max_step"])),
            key="max_step",
            help="Dose movement limit per CRM update."
        )

    with a3:
        st.toggle(
            "Guardrail: next dose ≤ highest tried + 1",
            key="enforce_guardrail",
            help="Prevents skipping over untried dose levels."
        )
        st.toggle(
            "Final MTD must be among tried doses",
            key="restrict_final_mtd",
            help="Final selection restricted to doses that were treated."
        )

    st.toggle(
        "Show CRM decision debug (first simulated trial)",
        key="show_debug",
        help="Prints admissible set and posterior summaries per update for the first trial."
    )

    st.markdown("---")
    st.markdown("**6+3 design settings (usually kept fixed)**")

    b1, b2 = st.columns([1.0, 1.0], gap="large")
    with b1:
        st.number_input(
            "Max sample size (6+3)",
            min_value=12, max_value=200, step=3,
            key="max_n_63",
            help="Total cap for 6+3."
        )
    with b2:
        st.selectbox(
            "Acceptance rule after expansion to 9",
            options=[1, 2],
            index=[1, 2].index(int(st.session_state["accept_rule_63"])),
            key="accept_rule_63",
            help="If 1/6 DLT, expand to 9; accept if total DLTs <= this threshold."
        )

# Run
run = st.button("Run simulations")

if run:
    rng = np.random.default_rng(int(st.session_state["seed"]))
    ns = int(st.session_state["n_sims"])

    sel_63 = np.zeros(5, dtype=int)
    sel_crm = np.zeros(5, dtype=int)

    nmat_63 = np.zeros((ns, 5), dtype=int)
    nmat_crm = np.zeros((ns, 5), dtype=int)

    dlt_63 = np.zeros(ns, dtype=int)
    dlt_crm = np.zeros(ns, dtype=int)

    debug_dump = None

    for s in range(ns):
        # 6+3
        chosen63, n63, y63 = run_6plus3(
            true_p=true_p,
            start_level=int(st.session_state["start_level"]),
            max_n=int(st.session_state["max_n_63"]),
            accept_max_dlt=int(st.session_state["accept_rule_63"]),
            rng=rng
        )

        # CRM
        debug_flag = bool(st.session_state["show_debug"] and s == 0)
        chosenc, nc, yc, dbg = run_crm_trial(
            true_p=true_p,
            target=float(st.session_state["target"]),
            skeleton=skeleton,
            sigma=float(st.session_state["sigma"]),
            start_level=int(st.session_state["start_level"]),
            max_n=int(st.session_state["max_n"]),
            cohort_size=int(st.session_state["cohort_size"]),
            max_step=int(st.session_state["max_step"]),
            gh_n=int(st.session_state["gh_n"]),
            enforce_guardrail=bool(st.session_state["enforce_guardrail"]),
            restrict_final_mtd_to_tried=bool(st.session_state["restrict_final_mtd"]),
            ewoc_on=bool(st.session_state["ewoc_on"]),
            ewoc_alpha=float(st.session_state["ewoc_alpha"]),
            burn_in_until_first_dlt=bool(st.session_state["burn_in"]),
            rng=rng,
            debug=debug_flag
        )

        sel_63[chosen63] += 1
        sel_crm[chosenc] += 1

        nmat_63[s, :] = n63
        nmat_crm[s, :] = nc

        dlt_63[s] = y63
        dlt_crm[s] = yc

        if debug_flag:
            debug_dump = dbg

    p63 = sel_63 / float(ns)
    pcrm = sel_crm / float(ns)

    avg63 = np.mean(nmat_63, axis=0)
    avgcrm = np.mean(nmat_crm, axis=0)

    mean_n63 = float(np.mean(nmat_63.sum(axis=1)))
    mean_ncrm = float(np.mean(nmat_crm.sum(axis=1)))

    mean_dlt63 = float(np.mean(dlt_63))
    mean_dltcrm = float(np.mean(dlt_crm))

    st.subheader("Results (6+3 vs CRM)")

    r1, r2 = st.columns([1.0, 1.0], gap="large")

    with r1:
        fig, ax = plt.subplots(figsize=(6.2, 2.8), dpi=160)
        xx = np.arange(5)
        w = 0.38
        ax.bar(xx - w/2, p63, w, label="6+3")
        ax.bar(xx + w/2, pcrm, w, label="CRM")
        ax.set_title("Probability of selecting each dose as MTD", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, max(p63.max(), pcrm.max()) * 1.15 + 1e-6)
        ax.axvline(true_mtd, linewidth=1, alpha=0.6)
        ax.text(true_mtd + 0.05, ax.get_ylim()[1] * 0.92, "True MTD", fontsize=8)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with r2:
        fig, ax = plt.subplots(figsize=(6.2, 2.8), dpi=160)
        xx = np.arange(5)
        w = 0.38
        ax.bar(xx - w/2, avg63, w, label="6+3")
        ax.bar(xx + w/2, avgcrm, w, label="CRM")
        ax.set_title("Average number treated per dose level", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Patients", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    s1, s2, s3, s4 = st.columns([1.0, 1.0, 1.0, 1.0], gap="large")
    with s1:
        st.metric("Mean sample size (6+3)", f"{mean_n63:.1f}")
    with s2:
        st.metric("Mean sample size (CRM)", f"{mean_ncrm:.1f}")
    with s3:
        st.metric("Mean total DLTs (6+3)", f"{mean_dlt63:.2f}")
    with s4:
        st.metric("Mean total DLTs (CRM)", f"{mean_dltcrm:.2f}")

    if st.session_state["show_debug"] and debug_dump:
        st.subheader("CRM debug (first simulated trial)")
        for i, row in enumerate(debug_dump, start=1):
            st.write(
                f"Update {i}: treated L{row['treated_level']} | n={row['cohort_n']} "
                f"| dlts={row['cohort_dlts']} | any_dlt_seen={row['any_dlt_seen']}"
            )
            if "next_level" in row:
                st.write(f"  allowed: {row['allowed_levels']} | next: L{row['next_level']} | highest_tried={row['highest_tried']}")
                st.write(f"  post_mean: {[round(v,3) for v in row['post_mean']]}")
                st.write(f"  od_prob:   {[round(v,3) for v in row['od_prob']]}")
