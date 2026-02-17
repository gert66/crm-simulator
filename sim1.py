import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# ============================================================
# Plot sizing (ONE place to tune)
# ============================================================
# Fixed pixel widths used by st.image(). Height is controlled by figsize.
PREVIEW_W_PX = 520        # small plot in CRM knobs panel
RESULT_W_PX  = 720        # results plots (bigger)

# Matplotlib figure sizes (in inches) + dpi -> pixel geometry is fixed.
# If you want “more square / taller” results, increase RESULT_H_IN.
PREVIEW_W_IN, PREVIEW_H_IN, PREVIEW_DPI = 5.2, 2.7, 160

# Results: make height ~ width (or a bit taller). With dpi=160 and 7.0" width -> ~1120 px wide,
# but we still display at RESULT_W_PX; the aspect ratio stays correct and stable.
RESULT_W_IN, RESULT_H_IN, RESULT_DPI = 7.0, 7.8, 160

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

def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

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

def run_6plus3(true_p, start_level=1, max_n=27, accept_max_dlt=1, rng=None):
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
        allowed = np.array([0], dtype=int)

    k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

    k_star = int(np.clip(k_star, current_level - int(max_step), current_level + int(max_step)))

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
            allowed2 = np.intersect1d(allowed, tried)
            if allowed2.size > 0:
                allowed = allowed2
            else:
                return int(tried.min())

    return int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

def run_crm_trial(
    true_p, target, skeleton,
    sigma=1.0,
    start_level=1,
    already_treated_start=0,
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

    already_treated_start = int(max(0, already_treated_start))
    if already_treated_start > 0:
        n_per[level] += already_treated_start

    highest_tried = level if already_treated_start > 0 else -1
    any_dlt_seen = False
    debug_rows = []

    burn_in_active = bool(burn_in_until_first_dlt and already_treated_start == 0)

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

        if burn_in_active and (not any_dlt_seen):
            if level < n_levels - 1:
                level += 1
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
# Defaults (R-aligned)
# ============================================================

st.set_page_config(
    page_title="6+3 vs CRM",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none; }
      [data-testid="stSidebarNav"] { display: none; }
      [data-testid="collapsedControl"] { display: none; }
      .block-container { padding-top: 2.6rem; padding-bottom: 1.2rem; }
      .element-container { margin-bottom: 0.35rem; }

      /* Keep images from stretching with container width */
      [data-testid="stImage"] img {
        max-width: none !important;
        width: auto !important;
        height: auto !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]
DEFAULT_TRUE_P = [0.01, 0.02, 0.12, 0.20, 0.35]

R_DEFAULTS = {
    "target": 0.15,
    "start_level_1b": 2,           # Sama: p <- 2 (1-based)
    "already_treated_start": 0,
    "n_sims": 1000,
    "seed": 123,
    "max_n_63": 27,
    "max_n_crm": 27,
    "cohort_size": 3,

    "prior_model": "empiric",
    "prior_target": 0.15,
    "halfwidth": 0.10,
    "prior_nu": 3,
    "logistic_intcpt": 3.0,

    "sigma": 1.0,
    "burn_in": True,
    "ewoc_on": False,
    "ewoc_alpha": 0.25,

    "gh_n": 61,
    "max_step": 1,
    "enforce_guardrail": True,
    "restrict_final_mtd": True,
    "show_debug": False,

    "accept_rule_63": 1,
}

def init_state():
    for k, v in R_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    for i in range(5):
        st.session_state.setdefault(f"true_{i}", float(DEFAULT_TRUE_P[i]))

def reset_defaults():
    for k, v in R_DEFAULTS.items():
        st.session_state[k] = v

init_state()

# ============================================================
# Essentials
# ============================================================

with st.expander("Essentials", expanded=False):
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("#### Study")
        st.number_input(
            "Target DLT rate",
            min_value=0.05, max_value=0.50, step=0.01,
            key="target",
            help="R mapping: target.acute / prior.target.acute. Sama example: 0.15."
        )
        st.number_input(
            "Start dose level (1-based)",
            min_value=1, max_value=5, step=1,
            key="start_level_1b",
            help="R mapping: burning phase start is p <- 2 (1-based)."
        )
        st.number_input(
            "Already treated at start dose (0 DLT)",
            min_value=0, max_value=500, step=1,
            key="already_treated_start",
            help="Adds N patients treated at start dose with 0 acute DLT before CRM starts."
        )

    with c2:
        st.markdown("#### Simulation")
        st.number_input(
            "Number of simulated trials",
            min_value=50, max_value=5000, step=50,
            key="n_sims",
            help="R mapping: NREP <- 1000."
        )
        st.number_input(
            "Random seed",
            min_value=1, max_value=10_000_000, step=1,
            key="seed",
            help="R mapping: set.seed(123)."
        )
        st.markdown("#### CRM integration")
        st.selectbox(
            "Gauss–Hermite points",
            options=[31, 41, 61, 81],
            index=[31, 41, 61, 81].index(int(st.session_state["gh_n"])),
            key="gh_n",
            help="Posterior integration accuracy vs speed."
        )
        st.selectbox(
            "Max dose step per update",
            options=[1, 2],
            index=[1, 2].index(int(st.session_state["max_step"])),
            key="max_step",
            help="Dose movement limit per CRM update."
        )

    with c3:
        st.markdown("#### Sample size")
        st.number_input(
            "Maximum sample size (6+3)",
            min_value=6, max_value=200, step=3,
            key="max_n_63",
            help="R mapping: N.patient = 27."
        )
        st.number_input(
            "Maximum sample size (CRM)",
            min_value=6, max_value=200, step=3,
            key="max_n_crm",
            help="R mapping: N.patient = 27."
        )
        st.number_input(
            "Cohort size",
            min_value=1, max_value=12, step=1,
            key="cohort_size",
            help="R mapping: CO = 3."
        )
        st.markdown("#### CRM safety / selection")
        st.toggle(
            "Guardrail: next dose ≤ highest tried + 1",
            key="enforce_guardrail",
            help="Prevents skipping untried dose levels."
        )
        st.toggle(
            "Final MTD must be among tried doses",
            key="restrict_final_mtd",
            help="Restricts final MTD to doses with n>0."
        )
        st.toggle(
            "Show CRM debug (first simulated trial)",
            key="show_debug",
            help="Print admissible set and posterior summaries for the first trial."
        )

    st.write("")
    if st.button("Reset to defaults"):
        reset_defaults()
        st.rerun()

# ============================================================
# Playground
# ============================================================

with st.expander("Playground", expanded=True):
    left, mid, right = st.columns([1.0, 1.0, 1.15], gap="large")

    with left:
        st.markdown("#### True acute DLT")
        true_p = []
        for i, lab in enumerate(dose_labels):
            true_p.append(float(st.number_input(
                f"L{i} {lab}",
                min_value=0.0, max_value=1.0, step=0.01,
                key=f"true_{i}",
            )))
        target_val = float(st.session_state["target"])
        true_mtd = find_true_mtd(true_p, target_val)
        st.caption(f"True MTD (closest to target) = L{true_mtd}")

    with mid:
        st.markdown("#### Priors")
        st.radio(
            "Skeleton model",
            options=["empiric", "logistic"],
            horizontal=True,
            key="prior_model",
            help="R mapping: dfcrm::getprior() skeleton generation."
        )
        st.slider(
            "Prior target",
            min_value=0.05, max_value=0.50, step=0.01,
            key="prior_target",
            help="R mapping: prior.target.acute (0.15)."
        )

        # Dynamic halfwidth bounds so we never crash
        prior_target = float(st.session_state["prior_target"])
        max_hw = min(0.30, prior_target - 0.001, 1.0 - prior_target - 0.001)
        max_hw = max(0.01, max_hw)

        if float(st.session_state["halfwidth"]) > max_hw:
            st.session_state["halfwidth"] = float(max_hw)

        st.slider(
            "Halfwidth (delta)",
            min_value=0.01,
            max_value=float(max_hw),
            step=0.01,
            key="halfwidth",
            help="R mapping: getprior(halfwidth = 0.1). Must satisfy target±halfwidth within (0,1)."
        )

        st.slider(
            "Prior MTD (1-based)",
            min_value=1, max_value=5, step=1,
            key="prior_nu",
            help="R mapping: prior.MTD.acute = 3."
        )
        st.slider(
            "Logistic intercept",
            min_value=-10.0, max_value=10.0, step=0.1,
            key="logistic_intcpt",
            help="Only used if skeleton model is logistic."
        )

        try:
            skeleton = dfcrm_getprior(
                halfwidth=float(st.session_state["halfwidth"]),
                target=float(st.session_state["prior_target"]),
                nu=int(st.session_state["prior_nu"]),
                nlevel=5,
                model=str(st.session_state["prior_model"]),
                intcpt=float(st.session_state["logistic_intcpt"]),
            ).tolist()
        except ValueError as e:
            st.warning(str(e))
            st.session_state["halfwidth"] = min(0.10, max_hw)
            skeleton = dfcrm_getprior(
                halfwidth=float(st.session_state["halfwidth"]),
                target=float(st.session_state["prior_target"]),
                nu=int(st.session_state["prior_nu"]),
                nlevel=5,
                model=str(st.session_state["prior_model"]),
                intcpt=float(st.session_state["logistic_intcpt"]),
            ).tolist()

    with right:
        st.markdown("#### CRM knobs")
        st.slider(
            "Prior sigma on theta",
            min_value=0.2, max_value=5.0, step=0.1,
            key="sigma",
            help="Theta prior: theta ~ N(0, sigma^2). Larger sigma => weaker prior around the skeleton."
        )
        st.toggle(
            "Burn-in until first DLT",
            key="burn_in",
            help="Sama’s R code includes a burning phase before CRM."
        )
        st.toggle(
            "Enable EWOC overdose control",
            key="ewoc_on",
            help="EWOC rule: allow doses with P(p_k > target | data) < alpha."
        )
        st.slider(
            "EWOC alpha",
            min_value=0.01, max_value=0.99, step=0.01,
            key="ewoc_alpha",
            disabled=(not st.session_state["ewoc_on"]),
            help="Only active when EWOC is enabled."
        )

        fig, ax = plt.subplots(figsize=(PREVIEW_W_IN, PREVIEW_H_IN), dpi=PREVIEW_DPI)
        x = np.arange(5)
        ax.plot(x, true_p, marker="o", linewidth=1.6, label="True P(DLT)")
        ax.plot(x, skeleton, marker="o", linewidth=1.6, label="Prior (skeleton)")
        ax.axhline(target_val, linewidth=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, min(1.0, max(max(true_p), max(skeleton), target_val) * 1.25 + 0.02))
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        st.image(fig_to_png_bytes(fig), width=PREVIEW_W_PX)

        st.write("")
        run = st.button("Run simulations", use_container_width=True)

# ============================================================
# Results
# ============================================================

if "run" in locals() and run:
    rng = np.random.default_rng(int(st.session_state["seed"]))
    ns = int(st.session_state["n_sims"])

    start_0b = int(st.session_state["start_level_1b"]) - 1
    start_0b = int(np.clip(start_0b, 0, 4))

    sel_63 = np.zeros(5, dtype=int)
    sel_crm = np.zeros(5, dtype=int)

    nmat_63 = np.zeros((ns, 5), dtype=int)
    nmat_crm = np.zeros((ns, 5), dtype=int)

    dlt_63 = np.zeros(ns, dtype=int)
    dlt_crm = np.zeros(ns, dtype=int)

    debug_dump = None

    for s in range(ns):
        chosen63, n63, y63 = run_6plus3(
            true_p=true_p,
            start_level=start_0b,
            max_n=int(st.session_state["max_n_63"]),
            accept_max_dlt=int(st.session_state["accept_rule_63"]),
            rng=rng
        )

        debug_flag = bool(st.session_state["show_debug"] and s == 0)

        chosenc, nc, yc, dbg = run_crm_trial(
            true_p=true_p,
            target=float(st.session_state["target"]),
            skeleton=skeleton,
            sigma=float(st.session_state["sigma"]),
            start_level=start_0b,
            already_treated_start=int(st.session_state["already_treated_start"]),
            max_n=int(st.session_state["max_n_crm"]),
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

    dlt_prob_63 = float(np.mean(dlt_63) / max(1e-9, mean_n63))
    dlt_prob_crm = float(np.mean(dlt_crm) / max(1e-9, mean_ncrm))

    st.markdown("---")

    r1, r2, r3 = st.columns([1.2, 1.2, 0.8], gap="large")

    with r1:
        fig, ax = plt.subplots(figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
        xx = np.arange(5)
        w = 0.38
        ax.bar(xx - w/2, p63, w, label="6+3")
        ax.bar(xx + w/2, pcrm, w, label="CRM")
        ax.set_title("P(select dose as MTD)", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, max(p63.max(), pcrm.max()) * 1.15 + 1e-6)
        ax.axvline(true_mtd, linewidth=1, alpha=0.6)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.image(fig_to_png_bytes(fig), width=RESULT_W_PX)

    with r2:
        fig, ax = plt.subplots(figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
        xx = np.arange(5)
        w = 0.38
        ax.bar(xx - w/2, avg63, w, label="6+3")
        ax.bar(xx + w/2, avgcrm, w, label="CRM")
        ax.set_title("Avg patients treated per dose", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=9)
        ax.set_ylabel("Patients", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.image(fig_to_png_bytes(fig), width=RESULT_W_PX)

    with r3:
        st.metric("DLT prob per patient (6+3)", f"{dlt_prob_63:.3f}")
        st.metric("DLT prob per patient (CRM)", f"{dlt_prob_crm:.3f}")
        st.caption(f"n_sims={ns} | seed={int(st.session_state['seed'])} | True MTD marker=L{true_mtd}")

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
