import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# Utilities
# ============================================================

def safe_probs(x):
    x = np.array(x, dtype=float)
    return np.clip(x, 1e-6, 1 - 1e-6)

def simulate_bernoulli(n, p, rng):
    return rng.binomial(1, p, size=n)

def find_true_mtd(true_p, target):
    true_p = np.array(true_p, dtype=float)
    return int(np.argmin(np.abs(true_p - target)))

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)

def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

# ============================================================
# Ping-pong / oscillation metrics
# ============================================================

def switch_rate(path):
    """Fraction of adjacent transitions that change dose."""
    if path is None or len(path) < 2:
        return 0.0
    path = np.asarray(path, dtype=int)
    return float(np.mean(path[1:] != path[:-1]))

def mean_step_size(path):
    """Average absolute step size between consecutive patients."""
    if path is None or len(path) < 2:
        return 0.0
    path = np.asarray(path, dtype=int)
    return float(np.mean(np.abs(path[1:] - path[:-1])))

def oscillation_index(path):
    """
    Fraction of triplets that show A->B->A with A!=B.
    This is the cleanest 'ping-pong' measure.
    """
    if path is None or len(path) < 3:
        return 0.0
    path = np.asarray(path, dtype=int)
    a = path[:-2]
    b = path[1:-1]
    c = path[2:]
    osc = (a == c) & (a != b)
    return float(np.mean(osc))

# ============================================================
# 6+3 Design
# ============================================================

def run_6plus3(true_p, start_level=0, max_n=36, accept_max_dlt=1, rng=None):
    """
    Cohorts of 6.
    - 0/6 DLT -> escalate
    - 1/6 DLT -> expand by 3 at same level; escalate if DLTs among those 9 <= accept_max_dlt
    - >=2/6 DLT -> stop/de-escalate (simple stop rule here)
    MTD selected as last acceptable dose visited.
    Also returns patient-level dose path (for ping-pong metrics).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_levels = len(true_p)
    level = int(start_level)

    n_per_level = np.zeros(n_levels, dtype=int)
    dlt_per_level = np.zeros(n_levels, dtype=int)

    total_n = 0
    last_acceptable = None
    dose_path = []

    while total_n < max_n:
        # Treat 6 at current level
        n_add = min(6, max_n - total_n)
        dose_path.extend([level] * n_add)
        out6 = simulate_bernoulli(n_add, true_p[level], rng)

        n_per_level[level] += n_add
        dlt_per_level[level] += int(out6.sum())
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
            # Expand by 3 at same dose
            n_add2 = min(3, max_n - total_n)
            dose_path.extend([level] * n_add2)
            out3 = simulate_bernoulli(n_add2, true_p[level], rng)

            n_per_level[level] += n_add2
            dlt_per_level[level] += int(out3.sum())
            total_n += n_add2

            if n_add2 < 3:
                break

            d9_cycle = d6 + int(out3.sum())
            if d9_cycle <= accept_max_dlt:
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
    total_dlts = int(dlt_per_level.sum())
    return selected, n_per_level, total_dlts, dose_path

# ============================================================
# CRM: 1-parameter power model
# p_k(theta) = skeleton_k ^ exp(theta)
# theta ~ Normal(0, sigma^2)
# Posterior computed on grid
# ============================================================

def crm_power_probs(theta_grid, skeleton):
    sk = safe_probs(skeleton)
    a = np.exp(theta_grid)[:, None]     # (G,1)
    return sk[None, :] ** a             # (G,K)

def normal_prior_logpdf(theta_grid, sigma):
    return -0.5 * (theta_grid / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

def posterior_on_grid(theta_grid, sigma, skeleton, n_per_level, dlt_per_level):
    n = np.array(n_per_level, dtype=float)
    y = np.array(dlt_per_level, dtype=float)

    P = crm_power_probs(theta_grid, skeleton)  # (G,K)
    ll = (y[None, :] * np.log(P) + (n[None, :] - y[None, :]) * np.log(1 - P)).sum(axis=1)

    lp = normal_prior_logpdf(theta_grid, sigma)
    log_post = lp + ll

    lse = logsumexp(log_post)
    w = np.exp(log_post - lse)
    return w, P

def crm_choose_next(theta_grid, sigma, skeleton, n_per_level, dlt_per_level,
                    current_level, target, alpha_overdose, max_step=1):
    w, P = posterior_on_grid(theta_grid, sigma, skeleton, n_per_level, dlt_per_level)

    post_mean = (w[:, None] * P).sum(axis=0)
    overdose_prob = (w[:, None] * (P > target)).sum(axis=0)

    allowed = np.where(overdose_prob < alpha_overdose)[0]
    if allowed.size == 0:
        return 0, post_mean, overdose_prob

    k_star = int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

    # Restrict step size
    if k_star > current_level + max_step:
        k_star = current_level + max_step
    if k_star < current_level - max_step:
        k_star = current_level - max_step

    k_star = int(np.clip(k_star, 0, len(skeleton) - 1))
    return k_star, post_mean, overdose_prob

def crm_select_mtd(theta_grid, sigma, skeleton, n_per_level, dlt_per_level, target, alpha_overdose):
    w, P = posterior_on_grid(theta_grid, sigma, skeleton, n_per_level, dlt_per_level)
    post_mean = (w[:, None] * P).sum(axis=0)
    overdose_prob = (w[:, None] * (P > target)).sum(axis=0)

    allowed = np.where(overdose_prob < alpha_overdose)[0]
    if allowed.size == 0:
        return 0
    return int(allowed[np.argmin(np.abs(post_mean[allowed] - target))])

def run_crm(true_p, target, skeleton, sigma=1.0, start_level=0, max_n=36,
            cohort_size=6, alpha_overdose=0.25, theta_min=-4.0, theta_max=4.0,
            theta_grid_n=401, max_step=1, rng=None):
    """
    Simulate CRM trial with patient-level dose path.
    Dose is held constant within a cohort, then updated.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_levels = len(true_p)
    level = int(start_level)

    n_per_level = np.zeros(n_levels, dtype=int)
    dlt_per_level = np.zeros(n_levels, dtype=int)
    total_n = 0
    dose_path = []

    theta_grid = np.linspace(theta_min, theta_max, int(theta_grid_n))

    while total_n < max_n:
        n_add = min(int(cohort_size), max_n - total_n)

        # record patient-level path
        dose_path.extend([level] * n_add)

        out = simulate_bernoulli(n_add, true_p[level], rng)
        n_per_level[level] += n_add
        dlt_per_level[level] += int(out.sum())
        total_n += n_add

        # If this was an incomplete cohort, stop
        if n_add < cohort_size:
            break

        next_level, _, _ = crm_choose_next(
            theta_grid, sigma, skeleton,
            n_per_level, dlt_per_level,
            current_level=level,
            target=target,
            alpha_overdose=alpha_overdose,
            max_step=max_step
        )
        level = next_level

    selected = crm_select_mtd(
        theta_grid, sigma, skeleton,
        n_per_level, dlt_per_level,
        target=target,
        alpha_overdose=alpha_overdose
    )

    total_dlts = int(dlt_per_level.sum())
    return selected, n_per_level, total_dlts, dose_path

# ============================================================
# App UI
# ============================================================

st.set_page_config(page_title="Dose Escalation Simulator: 6+3 vs CRM", layout="wide")
st.title("Dose Escalation Simulator: 6+3 vs CRM")

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]
level_labels = [f"L{i}" for i in range(5)]

scenario_library = {
    "Default (MTD around level 2)": [0.05, 0.10, 0.20, 0.35, 0.55],
    "Safer overall (pushes higher)": [0.03, 0.06, 0.12, 0.20, 0.30],
    "More toxic overall (lands lower)": [0.08, 0.16, 0.28, 0.42, 0.60],
    "MTD around level 1": [0.05, 0.22, 0.35, 0.50, 0.65],
    "MTD around level 3": [0.03, 0.06, 0.12, 0.24, 0.40],
    "Sharp jump after level 2": [0.05, 0.08, 0.18, 0.40, 0.65],
}

skeleton_presets = {
    "Match default scenario": [0.05, 0.10, 0.20, 0.35, 0.55],
    "More conservative skeleton": [0.07, 0.12, 0.22, 0.38, 0.58],
    "More optimistic skeleton": [0.03, 0.07, 0.16, 0.30, 0.48],
}

# -----------------------------
# Study setup
# -----------------------------
colA, colB = st.columns([1.05, 1.0])

with colA:
    st.subheader("Study setup")

    target = st.number_input(
        "Target DLT probability",
        min_value=0.05, max_value=0.50, value=0.25, step=0.01,
        help="Target toxicity level for the MTD. CRM aims for a dose with estimated DLT probability close to this value."
    )

    start_level = st.selectbox(
        "Start dose level",
        options=list(range(0, 5)),
        index=0,
        format_func=lambda i: f"Level {i} ({dose_labels[i]})",
        help="Starting dose for both designs."
    )

    n_sims = st.number_input(
        "Number of simulated trials",
        min_value=200, max_value=20000, value=2000, step=200,
        help="How many virtual trials to run."
    )

    seed = st.number_input(
        "Random seed",
        min_value=1, max_value=10_000_000, value=12345, step=1,
        help="Fix randomness so results are reproducible."
    )

with colB:
    st.subheader("True scenario (ground truth)")

    scenario_name = st.selectbox(
        "Scenario library",
        options=list(scenario_library.keys()),
        index=0,
        help="Pick a predefined true dose-toxicity scenario."
    )
    scenario_vals = scenario_library[scenario_name]

    manual_true = st.toggle(
        "Manually edit the true DLT probabilities",
        value=True,
        help="If on, you can tweak the true probabilities below."
    )

    true_p = []
    for i, lab in enumerate(dose_labels):
        default_val = float(scenario_vals[i])
        if manual_true:
            val = st.number_input(
                f"True P(DLT) at {lab}",
                min_value=0.0, max_value=1.0,
                value=default_val, step=0.01,
                key=f"true_{i}",
                help="Assumed true DLT probability at this dose level."
            )
        else:
            st.write(f"{lab}: {default_val:.2f}")
            val = default_val
        true_p.append(val)

    true_mtd = find_true_mtd(true_p, target)
    st.write(f"True MTD (closest to target {target:.2f}) = Level {true_mtd} ({dose_labels[true_mtd]})")

st.divider()

# -----------------------------
# Design settings
# -----------------------------
c1, c2 = st.columns([1.0, 1.0])

with c1:
    st.subheader("6+3 settings")

    max_n_6 = st.number_input(
        "Max sample size (6+3)",
        min_value=12, max_value=120, value=36, step=3,
        help="Maximum number of patients in each simulated 6+3 trial."
    )

    accept_rule = st.selectbox(
        "Acceptance rule after expansion to 9",
        options=[1, 2],
        index=0,
        help=(
            "If 1/6 DLT occurs, treat 3 more at the same dose. "
            "Escalate only if DLTs among those 9 are <= this threshold."
        )
    )

    st.info(
        f"""**6+3 decision rules**  
- Treat 6 patients at the current dose.  
- If 0/6 DLT, go up one level.  
- If 1/6 DLT, treat 3 more at the same dose. Escalate only if DLTs among those 9 <= {accept_rule}/9.  
- If 2+ DLT among the first 6, stop escalation (simple rule)."""
    )

with c2:
    st.subheader("CRM settings")

    max_n_crm = st.number_input(
        "Max sample size (CRM)",
        min_value=12, max_value=120, value=36, step=3,
        help="Maximum number of patients in each simulated CRM trial."
    )

    cohort_size = st.number_input(
        "Cohort size (CRM)",
        min_value=1, max_value=12, value=6, step=1,
        help=(
            "Number of patients treated before the CRM model updates and selects the next dose. "
            "This is not a minimum per dose level. It is the update frequency."
        )
    )

    skeleton_preset = st.selectbox(
        "Skeleton preset",
        options=list(skeleton_presets.keys()),
        index=0,
        help="Prior dose-toxicity shape used by CRM."
    )

    manual_skeleton = st.toggle(
        "Manually edit the skeleton",
        value=True,
        help="If on, you can tweak the skeleton values below."
    )

    sk_vals = skeleton_presets[skeleton_preset]
    skeleton = []
    for i, lab in enumerate(dose_labels):
        default_val = float(sk_vals[i])
        if manual_skeleton:
            val = st.number_input(
                f"Skeleton prior mean at {lab}",
                min_value=0.01, max_value=0.99,
                value=default_val, step=0.01,
                key=f"sk_{i}",
                help="Prior mean DLT probability at this dose level. Typically increasing with dose."
            )
        else:
            st.write(f"{lab}: {default_val:.2f}")
            val = default_val
        skeleton.append(val)

    sigma = st.number_input(
        "Prior sigma on theta",
        min_value=0.2, max_value=5.0, value=1.0, step=0.1,
        help="Higher sigma means more prior uncertainty and more learning from data."
    )

    alpha_overdose = st.number_input(
        "Overdose control alpha",
        min_value=0.05, max_value=0.50, value=0.25, step=0.01,
        help=(
            "Safety filter: a dose is allowed only if the posterior probability that its DLT rate exceeds the target "
            "is less than this alpha."
        )
    )

    max_step = st.selectbox(
        "Max dose step per update",
        options=[1, 2],
        index=0,
        help="Limits how far CRM can move between updates (up or down)."
    )

    st.info(
        f"""**CRM decision rules**  
- Treat {cohort_size} patients at the current dose, then update the model.  
- Among doses that pass overdose control, pick the dose whose estimated DLT probability is closest to {target:.2f}.  
- Overdose control: allow dose k only if P(DLT > {target:.2f}) < {alpha_overdose:.2f}.  
- Next dose can move at most ±{max_step} level(s) per update."""
    )

show_ping_plot = st.toggle(
    "Show ping-pong distribution plot",
    value=True,
    help="Shows a compact histogram of the oscillation index across simulated trials."
)

run = st.button("Run simulations")

# ============================================================
# Run simulations
# ============================================================

if run:
    rng = np.random.default_rng(int(seed))
    ns = int(n_sims)

    sel_6 = np.zeros(5, dtype=int)
    sel_c = np.zeros(5, dtype=int)

    tot_dlt_6 = np.zeros(ns, dtype=int)
    tot_dlt_c = np.zeros(ns, dtype=int)

    nmat_6 = np.zeros((ns, 5), dtype=int)
    nmat_c = np.zeros((ns, 5), dtype=int)

    ping_sw_6 = np.zeros(ns, dtype=float)
    ping_osc_6 = np.zeros(ns, dtype=float)
    ping_step_6 = np.zeros(ns, dtype=float)

    ping_sw_c = np.zeros(ns, dtype=float)
    ping_osc_c = np.zeros(ns, dtype=float)
    ping_step_c = np.zeros(ns, dtype=float)

    for s in range(ns):
        chosen6, n6, d6, path6 = run_6plus3(
            true_p=true_p,
            start_level=int(start_level),
            max_n=int(max_n_6),
            accept_max_dlt=int(accept_rule),
            rng=rng
        )

        chosenc, nc, dc, pathc = run_crm(
            true_p=true_p,
            target=float(target),
            skeleton=skeleton,
            sigma=float(sigma),
            start_level=int(start_level),
            max_n=int(max_n_crm),
            cohort_size=int(cohort_size),
            alpha_overdose=float(alpha_overdose),
            max_step=int(max_step),
            rng=rng
        )

        sel_6[chosen6] += 1
        sel_c[chosenc] += 1

        tot_dlt_6[s] = d6
        tot_dlt_c[s] = dc

        nmat_6[s, :] = n6
        nmat_c[s, :] = nc

        ping_sw_6[s] = switch_rate(path6)
        ping_osc_6[s] = oscillation_index(path6)
        ping_step_6[s] = mean_step_size(path6)

        ping_sw_c[s] = switch_rate(pathc)
        ping_osc_c[s] = oscillation_index(pathc)
        ping_step_c[s] = mean_step_size(pathc)

    # Results
    p_sel_6 = sel_6 / float(ns)
    p_sel_c = sel_c / float(ns)

    avg_n6 = np.mean(nmat_6, axis=0)
    avg_nc = np.mean(nmat_c, axis=0)

    overshoot6 = float(np.mean((nmat_6[:, true_mtd + 1:].sum(axis=1)) > 0)) if true_mtd < 4 else 0.0
    overshootc = float(np.mean((nmat_c[:, true_mtd + 1:].sum(axis=1)) > 0)) if true_mtd < 4 else 0.0

    st.subheader("Results")

    x = np.arange(5)
    width = 0.38

    r1, r2 = st.columns(2)

    with r1:
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=140)
        ax.bar(x - width/2, p_sel_6, width, label="6+3")
        ax.bar(x + width/2, p_sel_c, width, label="CRM")
        ax.set_title("Probability of selecting each dose as MTD", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, max(p_sel_6.max(), p_sel_c.max()) * 1.15 + 1e-6)
        compact_style(ax)
        ax.axvline(true_mtd, linewidth=1, alpha=0.6)
        ax.text(true_mtd + 0.05, ax.get_ylim()[1] * 0.92, "True MTD", fontsize=8)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with r2:
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=140)
        ax.bar(x - width/2, avg_n6, width, label="6+3")
        ax.bar(x + width/2, avg_nc, width, label="CRM")
        ax.set_title("Average number treated per dose level", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Patients", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    r3, r4 = st.columns(2)

    with r3:
        st.markdown("**Trial summary (averages)**")
        st.write(f"Mean total DLTs: 6+3 = {np.mean(tot_dlt_6):.2f} | CRM = {np.mean(tot_dlt_c):.2f}")
        st.write(f"Mean sample size: 6+3 = {np.mean(nmat_6.sum(axis=1)):.1f} | CRM = {np.mean(nmat_c.sum(axis=1)):.1f}")
        st.write(f"True MTD: Level {true_mtd} ({dose_labels[true_mtd]})")

    with r4:
        st.markdown("**Safety and ping-pong**")
        st.write(f"P(overshoot above true MTD): 6+3 = {overshoot6:.3f} | CRM = {overshootc:.3f}")
        st.write("")
        st.write(f"Switch rate: 6+3 = {np.mean(ping_sw_6):.3f} | CRM = {np.mean(ping_sw_c):.3f}")
        st.write(f"Oscillation index (A→B→A): 6+3 = {np.mean(ping_osc_6):.3f} | CRM = {np.mean(ping_osc_c):.3f}")
        st.write(f"Mean step size: 6+3 = {np.mean(ping_step_6):.3f} | CRM = {np.mean(ping_step_c):.3f}")

    if show_ping_plot:
        st.subheader("Ping-pong distribution (oscillation index)")

        cA, cB = st.columns(2)
        bins = np.linspace(0, 1, 21)

        with cA:
            fig, ax = plt.subplots(figsize=(5.6, 2.4), dpi=140)
            ax.hist(ping_osc_6, bins=bins, alpha=0.9, label="6+3")
            ax.set_title("6+3 oscillation index", fontsize=10)
            ax.set_xlabel("Oscillation index", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            compact_style(ax)
            st.pyplot(fig, clear_figure=True)

        with cB:
            fig, ax = plt.subplots(figsize=(5.6, 2.4), dpi=140)
            ax.hist(ping_osc_c, bins=bins, alpha=0.9, label="CRM")
            ax.set_title("CRM oscillation index", fontsize=10)
            ax.set_xlabel("Oscillation index", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            compact_style(ax)
            st.pyplot(fig, clear_figure=True)
