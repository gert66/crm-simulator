import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# Plot sizing (ONE place to tune)
# ============================================================
# Fixed pixel widths used by st.image(). Height is controlled by figsize.
PREVIEW_W_PX = 200        # small plot in CRM knobs panel
RESULT_W_PX  = 380        # results plots (bigger)

# Matplotlib figure sizes (in inches) + dpi -> pixel geometry is fixed.
# If you want “more square / taller” results, increase RESULT_H_IN.
PREVIEW_W_IN, PREVIEW_H_IN, PREVIEW_DPI = 3.5, 2.7, 160

# Results: make height ~ width (or a bit taller). With dpi=160 and 7.0" width -> ~1120 px wide,
# but we still display at RESULT_W_PX; the aspect ratio stays correct and stable.
RESULT_W_IN, RESULT_H_IN, RESULT_DPI = 6.0, 5.0, 160

# ============================================================
# Page config + CSS (hide sidebar + tighten top padding)
# ============================================================
st.set_page_config(
    page_title="6+3 vs CRM",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
      /* Hide Streamlit sidebar entirely (also hides multi-page list) */
      [data-testid="stSidebar"] { display: none; }

      /* Reduce top padding so the Essentials expander is fully visible */
      .block-container { padding-top: 1.1rem; padding-bottom: 1.0rem; }

      /* Slightly tighter expander header spacing */
      div[data-testid="stExpander"] details summary { padding-top: 0.35rem; padding-bottom: 0.35rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Helpers
# ============================================================
def safe_probs(x):
    x = np.asarray(x, dtype=float)
    return np.clip(x, 1e-6, 1 - 1e-6)

def simulate_bernoulli(n, p, rng):
    return rng.binomial(1, float(p), size=int(n))

def find_true_mtd(true_p, target):
    true_p = np.asarray(true_p, dtype=float)
    return int(np.argmin(np.abs(true_p - float(target))))

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)

def clamp_halfwidth(target, halfwidth, eps=1e-3):
    """
    Ensure target±halfwidth stays inside (0,1). If not, clamp halfwidth.
    Returns (halfwidth_clamped, was_clamped, max_allowed).
    """
    target = float(target)
    halfwidth = float(halfwidth)
    max_allowed = max(eps, min(target - eps, 1.0 - target - eps))
    if halfwidth > max_allowed:
        return max_allowed, True, max_allowed
    if halfwidth <= 0:
        return max(eps, min(max_allowed, 0.01)), True, max_allowed
    return halfwidth, False, max_allowed

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
# 6+3 design (simple) + optional "already treated at start"
# ============================================================
def run_6plus3(true_p, start_level=1, max_n=27, accept_max_dlt=1, already_n0=0, rng=None):
    """
    Simple 6+3:
      - Work in blocks aiming for 6 patients at current level; if 1/6 expand to 9.
      - already_n0: number already treated at start_level with 0 DLT (counts toward sample size).
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

    # preload already-treated
    already_n0 = int(max(0, already_n0))
    if already_n0 > 0:
        n_add0 = min(already_n0, int(max_n))
        n_per[level] += n_add0
        # y_per += 0
    total_n = int(n_per.sum())

    last_acceptable = None

    while total_n < int(max_n):
        # fill up to 6 at this level (or treat 6 if starting fresh)
        need_for_6 = max(0, 6 - int(n_per[level]))
        if need_for_6 == 0:
            # if already >=6, treat in chunks of 6 for simplicity
            need_for_6 = 6

        n_add = min(need_for_6, int(max_n) - total_n)
        out = simulate_bernoulli(n_add, true_p[level], rng)

        n_per[level] += n_add
        y_per[level] += int(out.sum())
        total_n += n_add

        if n_add < need_for_6:
            break

        # evaluate first 6 at this level (approx by looking at first 6 treated count)
        # Since we don't track order, use current cumulative when it just reached >=6.
        # This keeps behavior stable for simulation purposes.
        if int(n_per[level]) < 6:
            break

        # approximate "d6" using binomial draw from current batch if it completed the 6 target
        # For stable behavior: compute d6 based on the most recent n_add when it exactly completes 6.
        d6 = int(out.sum()) if need_for_6 <= n_add else int(out.sum())

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
    overdose_prob = (post_w[:, None] * (P > float(target))).sum(axis=0)
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

    k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - float(target)))])

    # step limiting
    k_star = int(np.clip(k_star, int(current_level) - int(max_step), int(current_level) + int(max_step)))

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

    return int(allowed[np.argmin(np.abs(post_mean[allowed] - float(target)))])

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
    already_n0=0,
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

    # preload already-treated at start with 0 DLT
    already_n0 = int(max(0, already_n0))
    if already_n0 > 0:
        n_add0 = min(already_n0, int(max_n))
        n_per[level] += n_add0

    highest_tried = level if int(n_per.sum()) > 0 else -1
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

        # simplified R-like burning phase: escalate cohort-wise until first DLT
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
# Defaults aligned to your R context
# ============================================================
dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]
DEFAULT_TRUE_P = [0.01, 0.02, 0.12, 0.20, 0.35]

DEFAULTS = {
    # Essentials (R example)
    "target": 0.15,
    "start_level_1based": 2,        # R burn-in p <- 2 (1-based)
    "already_n0": 0,                # additional "already treated at start dose (0 DLT)"
    "n_sims": 200,
    "seed": 123,
    "max_n_63": 27,
    "max_n_crm": 27,
    "cohort_size": 3,

    # Prior playground
    "prior_model": "empiric",
    "prior_target": 0.15,
    "halfwidth": 0.10,
    "prior_nu": 3,
    "logistic_intcpt": 3.0,

    # CRM knobs
    "sigma": 1.0,
    "burn_in": True,
    "ewoc_on": False,
    "ewoc_alpha": 0.25,

    # CRM integration + selection
    "gh_n": 61,
    "max_step": 1,
    "enforce_guardrail": True,
    "restrict_final_mtd": True,
    "show_debug": False,

    # 6+3 acceptance rule
    "accept_rule_63": 1,
}

TRUE_KEYS = [f"true_{i}" for i in range(5)]
ALL_KEYS = list(DEFAULTS.keys()) + TRUE_KEYS

def apply_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    for i in range(5):
        st.session_state[f"true_{i}"] = float(DEFAULT_TRUE_P[i])

# Reset flag logic: ensure reset happens BEFORE widgets are created
if st.session_state.get("_do_reset", False):
    apply_defaults()
    st.session_state["_do_reset"] = False
    st.rerun()

# Initialize if missing
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
for i in range(5):
    st.session_state.setdefault(f"true_{i}", float(DEFAULT_TRUE_P[i]))

# ============================================================
# Essentials expander (top)
# ============================================================
with st.expander("Essentials", expanded=True):
    c1, c2, c3 = st.columns([1.25, 1.25, 1.25], gap="large")

    with c1:
        st.subheader("Study")
        st.number_input(
            "Target DLT rate",
            min_value=0.05, max_value=0.50, step=0.01,
            key="target",
            help="R: target.acute is the target used by titecrm(prior, target=...). In your R example it is 0.15."
        )
        st.number_input(
            "Start dose level (1-based)",
            min_value=1, max_value=5, step=1,
            key="start_level_1based",
            help="R: burning phase uses p <- 2 (1-based). Here you enter 1..5. Internally we convert to 0-based."
        )
        st.number_input(
            "Already treated at start dose (0 DLT)",
            min_value=0, max_value=200, step=1,
            key="already_n0",
            help="Preloads x patients at the start dose with 0 DLT before simulation starts. Useful if you already treated patients safely at the starting dose."
        )

    with c2:
        st.subheader("Simulation")
        st.number_input(
            "Number of simulated trials",
            min_value=50, max_value=5000, step=50,
            key="n_sims",
            help="R: NREP is the number of replications. You used NREP=1000; here default is 200 for speed."
        )
        st.number_input(
            "Random seed",
            min_value=1, max_value=10_000_000, step=1,
            key="seed",
            help="R: set.seed(...). Controls reproducibility."
        )
        st.subheader("CRM integration")
        st.selectbox(
            "Gauss–Hermite points",
            options=[31, 41, 61, 81],
            index=[31, 41, 61, 81].index(int(st.session_state["gh_n"])),
            key="gh_n",
            help="Numerical integration accuracy for the CRM posterior. More points = slower but more accurate."
        )
        st.selectbox(
            "Max dose step per update",
            options=[1, 2],
            index=[1, 2].index(int(st.session_state["max_step"])),
            key="max_step",
            help="Dose movement limit per CRM update (step limiting)."
        )

    with c3:
        st.subheader("Sample size")
        st.number_input(
            "Maximum sample size (6+3)",
            min_value=6, max_value=200, step=3,
            key="max_n_63",
            help="Cap on total patients for 6+3 simulations."
        )
        st.number_input(
            "Maximum sample size (CRM)",
            min_value=6, max_value=200, step=3,
            key="max_n_crm",
            help="R: N.patient = 27 in your example."
        )
        st.number_input(
            "Cohort size",
            min_value=1, max_value=12, step=1,
            key="cohort_size",
            help="R: CO = 3 in your example."
        )

        st.subheader("CRM safety / selection")
        st.toggle(
            "Guardrail: next dose ≤ highest tried + 1",
            key="enforce_guardrail",
            help="Prevents skipping over untried dose levels."
        )
        st.toggle(
            "Final MTD must be among tried doses",
            key="restrict_final_mtd",
            help="Restrict final MTD selection to dose levels that were actually treated."
        )
        st.toggle(
            "Show CRM debug (first simulated trial)",
            key="show_debug",
            help="Shows admissible set and posterior summaries for the first simulated trial."
        )

    st.write("")
    if st.button("Reset to defaults", use_container_width=False):
        st.session_state["_do_reset"] = True
        st.rerun()

# ============================================================
# Playground expander (main)
# ============================================================
with st.expander("Playground", expanded=True):
    col_true, col_prior, col_knobs = st.columns([1.05, 1.10, 1.25], gap="large")

    # ---- True curve
    with col_true:
        st.subheader("True acute DLT")
        true_p = []
        for i, lab in enumerate(dose_labels):
            true_p.append(float(st.number_input(
                f"L{i} {lab}",
                min_value=0.0, max_value=1.0, step=0.01,
                key=f"true_{i}",
                help="Ground truth P(DLT) used to simulate Bernoulli outcomes."
            )))
        true_mtd = find_true_mtd(true_p, float(st.session_state["target"]))
        st.caption(f"True MTD (closest to target) = L{true_mtd}")

    # ---- Priors
    with col_prior:
        st.subheader("Priors")
        st.radio(
            "Skeleton model",
            options=["empiric", "logistic"],
            horizontal=True,
            key="prior_model",
            help="R: dfcrm::getprior(...) returns a skeleton (power model). Here we support empiric and logistic options."
        )
        st.slider(
            "Prior target",
            min_value=0.05, max_value=0.50, step=0.01,
            key="prior_target",
            help="R: getprior(target=prior.target.acute). In your R example: 0.15."
        )

        # clamp halfwidth to avoid hard crash
        hw_raw = float(st.session_state["halfwidth"])
        tgt_raw = float(st.session_state["prior_target"])
        hw_clamped, was_clamped, hw_max = clamp_halfwidth(tgt_raw, hw_raw, eps=1e-3)
        if was_clamped and abs(hw_clamped - hw_raw) > 1e-12:
            st.session_state["halfwidth"] = float(hw_clamped)
            st.warning(f"Halfwidth was adjusted to {hw_clamped:.3f} so that target±halfwidth stays inside (0,1). Max allowed here is {hw_max:.3f}.")

        st.slider(
            "Halfwidth (delta)",
            min_value=0.01, max_value=0.30, step=0.01,
            key="halfwidth",
            help="R: getprior(halfwidth=0.1). Must satisfy target±halfwidth within (0,1)."
        )
        st.slider(
            "Prior MTD (1-based)",
            min_value=1, max_value=5, step=1,
            key="prior_nu",
            help="R: getprior(nu=prior.MTD.acute). In your R example: 3."
        )
        st.slider(
            "Logistic intercept",
            min_value=0.0, max_value=10.0, step=0.1,
            key="logistic_intcpt",
            help="Only used for logistic skeleton generation."
        )

        # build skeleton safely (with clamped halfwidth)
        try:
            skeleton = dfcrm_getprior(
                halfwidth=float(st.session_state["halfwidth"]),
                target=float(st.session_state["prior_target"]),
                nu=int(st.session_state["prior_nu"]),
                nlevel=5,
                model=str(st.session_state["prior_model"]),
                intcpt=float(st.session_state["logistic_intcpt"]),
            ).tolist()
        except Exception as e:
            # fallback safe skeleton (very conservative)
            skeleton = [0.05, 0.08, 0.12, 0.18, 0.25]
            st.error(f"Could not build skeleton from current settings. Using fallback. Details: {e}")

    # ---- CRM knobs + preview plot
    with col_knobs:
        st.subheader("CRM knobs")
        st.slider(
            "Prior sigma on theta",
            min_value=0.2, max_value=5.0, step=0.1,
            key="sigma",
            help="R: dfcrm::titecrm uses a prior skeleton; sigma is the prior SD on theta in the power model integration here. Default 1.0."
        )
        st.toggle(
            "Burn-in until first DLT",
            key="burn_in",
            help="Mimics the idea of a burning phase: escalate cohort-wise until the first DLT is observed, then use CRM rules."
        )
        st.toggle(
            "Enable EWOC overdose control",
            key="ewoc_on",
            help="If enabled, admissible doses satisfy P(p_k > target | data) < alpha."
        )
        st.slider(
            "EWOC alpha",
            min_value=0.05, max_value=0.99, step=0.01,
            key="ewoc_alpha",
            disabled=(not st.session_state["ewoc_on"]),
            help="Only used when EWOC is enabled."
        )

        # fixed-size preview plot -> rendered as fixed-width image
        fig, ax = plt.subplots(figsize=(PREVIEW_W_IN, PREVIEW_H_IN), dpi=PREVIEW_DPI)
        x = np.arange(5)
        target_val = float(st.session_state["target"])
        ax.plot(x, true_p, marker="o", linewidth=1.6, label="True P(DLT)")
        ax.plot(x, skeleton, marker="o", linewidth=1.6, label="Prior (skeleton)")
        ax.axhline(target_val, linewidth=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, min(1.0, max(max(true_p), max(skeleton), target_val) * 1.25 + 0.02))
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper left")

        fig.tight_layout()
        st.image(fig, width=PREVIEW_W_PX)
        plt.close(fig)

        run = st.button("Run simulations", use_container_width=True)

# ============================================================
# Run simulations + Results (no "Results" header)
# ============================================================
if "run" not in locals():
    run = False

if run:
    rng = np.random.default_rng(int(st.session_state["seed"]))
    ns = int(st.session_state["n_sims"])

    start_level0 = int(st.session_state["start_level_1based"]) - 1
    start_level0 = int(np.clip(start_level0, 0, 4))

    already_n0 = int(st.session_state["already_n0"])

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
            start_level=start_level0,
            max_n=int(st.session_state["max_n_63"]),
            accept_max_dlt=int(st.session_state["accept_rule_63"]),
            already_n0=already_n0,
            rng=rng
        )

        debug_flag = bool(st.session_state["show_debug"] and s == 0)
        chosenc, nc, yc, dbg = run_crm_trial(
            true_p=true_p,
            target=float(st.session_state["target"]),
            skeleton=skeleton,
            sigma=float(st.session_state["sigma"]),
            start_level=start_level0,
            max_n=int(st.session_state["max_n_crm"]),
            cohort_size=int(st.session_state["cohort_size"]),
            max_step=int(st.session_state["max_step"]),
            gh_n=int(st.session_state["gh_n"]),
            enforce_guardrail=bool(st.session_state["enforce_guardrail"]),
            restrict_final_mtd_to_tried=bool(st.session_state["restrict_final_mtd"]),
            ewoc_on=bool(st.session_state["ewoc_on"]),
            ewoc_alpha=float(st.session_state["ewoc_alpha"]),
            burn_in_until_first_dlt=bool(st.session_state["burn_in"]),
            already_n0=already_n0,
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

    # DLT prob per patient (mean total DLTs / mean total patients)
    mean_n63 = float(np.mean(nmat_63.sum(axis=1)))
    mean_ncrm = float(np.mean(nmat_crm.sum(axis=1)))
    mean_dlt63 = float(np.mean(dlt_63))
    mean_dltcrm = float(np.mean(dlt_crm))
    dltpp_63 = (mean_dlt63 / mean_n63) if mean_n63 > 0 else 0.0
    dltpp_crm = (mean_dltcrm / mean_ncrm) if mean_ncrm > 0 else 0.0

    # Layout: two plots + metrics
    r1, r2, r3 = st.columns([1.10, 1.10, 0.80], gap="large")

    with r1:
        fig, ax = plt.subplots(figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
        xx = np.arange(5)
        w = 0.38
        ax.bar(xx - w/2, p63, w, label="6+3")
        ax.bar(xx + w/2, pcrm, w, label="CRM")
        ax.set_title("P(select dose as MTD)", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, max(p63.max(), pcrm.max()) * 1.15 + 1e-6)
        ax.axvline(true_mtd, linewidth=1, alpha=0.6)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        fig.tight_layout()
        st.image(fig, width=RESULT_W_PX)
        plt.close(fig)

    with r2:
        fig, ax = plt.subplots(figsize=(RESULT_W_IN, RESULT_H_IN), dpi=RESULT_DPI)
        xx = np.arange(5)
        w = 0.38
        ax.bar(xx - w/2, avg63, w, label="6+3")
        ax.bar(xx + w/2, avgcrm, w, label="CRM")
        ax.set_title("Avg patients treated per dose", fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"L{i}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Patients", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        fig.tight_layout()
        st.image(fig, width=RESULT_W_PX)
        plt.close(fig)

    with r3:
        st.metric("DLT prob per patient (6+3)", f"{dltpp_63:.3f}")
        st.metric("DLT prob per patient (CRM)", f"{dltpp_crm:.3f}")
        st.caption(f"n_sims={ns} | seed={int(st.session_state['seed'])} | True MTD marker=L{true_mtd}")

    if st.session_state["show_debug"] and debug_dump:
        st.write("")
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
