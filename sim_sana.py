import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# Utilities
# ============================================================

def clamp_probs(p):
    p = np.asarray(p, dtype=float)
    return np.clip(p, 1e-6, 1 - 1e-6)

def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

def compact_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.25)

def find_true_mtd(true_p, target):
    true_p = np.asarray(true_p, dtype=float)
    return int(np.argmin(np.abs(true_p - target)))

def switch_rate(path):
    if path is None or len(path) < 2:
        return 0.0
    p = np.asarray(path, dtype=int)
    return float(np.mean(p[1:] != p[:-1]))

def oscillation_index(path):
    if path is None or len(path) < 3:
        return 0.0
    p = np.asarray(path, dtype=int)
    a, b, c = p[:-2], p[1:-1], p[2:]
    return float(np.mean((a == c) & (a != b)))

# ============================================================
# 6+3 design
# ============================================================

def run_6plus3(true_p, start_level=0, max_n=36, accept_max_dlt=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    true_p = clamp_probs(true_p)
    K = len(true_p)

    level = int(start_level)
    n_per = np.zeros(K, dtype=int)
    dlt_per = np.zeros(K, dtype=int)
    dose_path = []
    total_n = 0
    last_acceptable = None

    while total_n < max_n:
        n_add = min(6, max_n - total_n)
        out = rng.binomial(1, true_p[level], size=n_add)
        n_per[level] += n_add
        dlt_per[level] += int(out.sum())
        dose_path.extend([level] * n_add)
        total_n += n_add

        if n_add < 6:
            break

        d6 = int(out.sum())

        if d6 == 0:
            last_acceptable = level
            if level < K - 1:
                level += 1
                continue
            break

        if d6 == 1:
            n_add2 = min(3, max_n - total_n)
            out2 = rng.binomial(1, true_p[level], size=n_add2)
            n_per[level] += n_add2
            dlt_per[level] += int(out2.sum())
            dose_path.extend([level] * n_add2)
            total_n += n_add2

            if n_add2 < 3:
                break

            d9 = d6 + int(out2.sum())
            if d9 <= accept_max_dlt:
                last_acceptable = level
                if level < K - 1:
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
    total_dlts = int(dlt_per.sum())
    observed_dlts = total_dlts  # 6+3 assumes outcomes observed within decision window
    exp_risk = float(np.sum(n_per * true_p) / max(1, np.sum(n_per)))

    return selected, n_per, total_dlts, observed_dlts, exp_risk, dose_path

# ============================================================
# R-style skeleton (getprior-like)
# ============================================================

def getprior_like(target, nu_0based, nlevel, halfwidth=0.1):
    sk = np.zeros(nlevel, dtype=float)
    nu = int(np.clip(nu_0based, 0, nlevel - 1))
    sk[nu] = float(target)

    for k in range(nu - 1, -1, -1):
        sk[k] = max(1e-6, sk[k + 1] - halfwidth)

    for k in range(nu + 1, nlevel):
        sk[k] = min(1 - 1e-6, sk[k - 1] + halfwidth)

    sk = np.maximum.accumulate(sk)
    return clamp_probs(sk)

# ============================================================
# CRM core (power model) with Normal prior
# p_k(theta) = skeleton_k ^ exp(theta)
# ============================================================

def crm_probs(theta_grid, skeleton):
    sk = clamp_probs(skeleton)
    a = np.exp(theta_grid)[:, None]
    return sk[None, :] ** a

def prior_logpdf(theta_grid, sigma):
    return -0.5 * (theta_grid / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

def posterior_weights(theta_grid, sigma, skeleton, n_eff, y):
    P = crm_probs(theta_grid, skeleton)
    n_eff = np.asarray(n_eff, dtype=float)
    y = np.asarray(y, dtype=float)

    ll = (y[None, :] * np.log(P) + (n_eff[None, :] - y[None, :]) * np.log(1 - P)).sum(axis=1)
    lp = prior_logpdf(theta_grid, sigma)
    log_post = lp + ll
    w = np.exp(log_post - logsumexp(log_post))
    return w, P

def crm_choose_mtd(theta_grid, sigma, skeleton, n_eff, y, target):
    w, P = posterior_weights(theta_grid, sigma, skeleton, n_eff, y)
    post_mean = (w[:, None] * P).sum(axis=0)
    mtd = int(np.argmin(np.abs(post_mean - target)))
    return mtd, post_mean

# ============================================================
# Classic CRM trial (no time-to-event)
# - Cohorts of size CO
# - tox observed immediately
# - no skipping: max +1 up
# ============================================================

def run_classic_crm_trial(
    true_p,
    target,
    skeleton,
    sigma=1.158,
    start_level=0,
    max_n=27,
    cohort_size=3,
    theta_min=-4.0,
    theta_max=4.0,
    theta_grid_n=401,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    true_p = clamp_probs(true_p)
    K = len(true_p)
    level = int(start_level)

    assigned = np.full(max_n, -1, dtype=int)
    tox = np.zeros(max_n, dtype=int)

    theta_grid = np.linspace(theta_min, theta_max, int(theta_grid_n))
    n_enrolled = 0

    while n_enrolled < max_n:
        n_add = min(int(cohort_size), max_n - n_enrolled)

        for i in range(n_enrolled, n_enrolled + n_add):
            assigned[i] = level
            tox[i] = int(rng.binomial(1, true_p[level]))

        n_enrolled += n_add

        n_eff = np.zeros(K, dtype=float)
        y = np.zeros(K, dtype=float)
        for i in range(n_enrolled):
            k = assigned[i]
            n_eff[k] += 1.0
            y[k] += tox[i]

        mtd, _ = crm_choose_mtd(theta_grid, sigma, skeleton, n_eff, y, target)

        if mtd > level:
            level = min(level + 1, K - 1)
        else:
            level = mtd

    n_eff = np.zeros(K, dtype=float)
    y = np.zeros(K, dtype=float)
    for i in range(n_enrolled):
        k = assigned[i]
        n_eff[k] += 1.0
        y[k] += tox[i]

    selected, _ = crm_choose_mtd(theta_grid, sigma, skeleton, n_eff, y, target)

    n_per_level = np.bincount(assigned, minlength=K)
    total_dlts = int(tox.sum())
    observed_dlts = total_dlts
    exp_risk = float(np.sum(n_per_level * true_p) / max(1, np.sum(n_per_level)))
    dose_path = assigned.tolist()

    return selected, n_per_level, total_dlts, observed_dlts, exp_risk, dose_path

# ============================================================
# TITE-CRM trial (time-to-event)
# - DLT time lognormal if latent DLT occurs
# - follow-up accrues by Wait.Time
# - weight = followup/obswin for non-events
# - no skipping: max +1 up
# ============================================================

def sample_event_time(obswin, rng, meanlog=np.log(2.0), sdlog=0.5):
    t = rng.lognormal(mean=meanlog, sigma=sdlog)
    return float(min(t, obswin))

def run_titecrm_trial(
    true_p,
    target,
    skeleton,
    sigma=1.158,
    start_level=0,
    max_n=27,
    cohort_size=3,
    wait_time=1.0,
    obswin=16.0,
    meanlog_event=np.log(2.0),
    sdlog_event=0.5,
    theta_min=-4.0,
    theta_max=4.0,
    theta_grid_n=401,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    true_p = clamp_probs(true_p)
    K = len(true_p)
    level = int(start_level)

    assigned = np.full(max_n, -1, dtype=int)
    latent_dlt = np.zeros(max_n, dtype=int)
    event_time = np.full(max_n, 1e9, dtype=float)
    followup = np.zeros(max_n, dtype=float)

    theta_grid = np.linspace(theta_min, theta_max, int(theta_grid_n))
    n_enrolled = 0

    while n_enrolled < max_n:
        n_add = min(int(cohort_size), max_n - n_enrolled)

        for i in range(n_enrolled, n_enrolled + n_add):
            assigned[i] = level
            latent_dlt[i] = int(rng.binomial(1, true_p[level]))
            if latent_dlt[i] == 1:
                event_time[i] = sample_event_time(obswin, rng, meanlog=meanlog_event, sdlog=sdlog_event)
            else:
                event_time[i] = 1e9
            followup[i] = min(wait_time, obswin)

        for i in range(0, n_enrolled + n_add):
            followup[i] = min(obswin, followup[i] + wait_time)

        n_enrolled += n_add

        n_eff = np.zeros(K, dtype=float)
        y = np.zeros(K, dtype=float)

        for i in range(n_enrolled):
            k = assigned[i]
            fu = min(followup[i], obswin)
            w = fu / obswin
            if latent_dlt[i] == 1 and event_time[i] <= fu:
                n_eff[k] += 1.0
                y[k] += 1.0
            else:
                n_eff[k] += w

        mtd, _ = crm_choose_mtd(theta_grid, sigma, skeleton, n_eff, y, target)

        if mtd > level:
            level = min(level + 1, K - 1)
        else:
            level = mtd

    n_eff = np.zeros(K, dtype=float)
    y = np.zeros(K, dtype=float)
    for i in range(n_enrolled):
        k = assigned[i]
        fu = min(followup[i], obswin)
        w = fu / obswin
        if latent_dlt[i] == 1 and event_time[i] <= fu:
            n_eff[k] += 1.0
            y[k] += 1.0
        else:
            n_eff[k] += w

    selected, _ = crm_choose_mtd(theta_grid, sigma, skeleton, n_eff, y, target)

    n_per_level = np.bincount(assigned, minlength=K)
    total_dlts = int(latent_dlt.sum())
    observed_dlts = int(np.sum((latent_dlt == 1) & (event_time <= followup)))
    exp_risk = float(np.sum(n_per_level * true_p) / max(1, np.sum(n_per_level)))
    dose_path = assigned.tolist()

    return selected, n_per_level, total_dlts, observed_dlts, exp_risk, dose_path

# ============================================================
# App UI
# ============================================================

st.set_page_config(page_title="6+3 vs CRM (Classic/TITE)", layout="wide")
st.title("6+3 vs CRM (Classic or TITE)")

dose_labels = ["5×4 Gy", "5×5 Gy", "5×6 Gy", "5×7 Gy", "5×8 Gy"]

scenario_library = {
    "R snippet acute example": [0.01, 0.02, 0.12, 0.20, 0.35],
    "MTD around level 2 for target ~0.25": [0.05, 0.10, 0.20, 0.35, 0.55],
    "Safer overall": [0.03, 0.06, 0.12, 0.20, 0.30],
    "More toxic overall": [0.08, 0.16, 0.28, 0.42, 0.60],
}

topL, topR = st.columns([1.05, 1.0])

with topL:
    st.subheader("Global setup")
    start_level = st.selectbox(
        "Start dose level",
        options=list(range(0, 5)),
        index=0,
        format_func=lambda i: f"Level {i} ({dose_labels[i]})",
        help="Level 0 = 5×4, Level 1 = 5×5, ..."
    )
    crm_mode = st.radio(
        "CRM mode",
        options=["Classic CRM", "TITE-CRM"],
        index=1,
        help="Classic: immediate binary outcomes. TITE: partial follow-up contributes fractional weight."
    )
    n_sims = st.number_input("Number of simulated trials", 200, 20000, 2000, 200)
    seed = st.number_input("Random seed", 1, 10_000_000, 12345, 1)

with topR:
    st.subheader("True acute scenario")
    scenario_name = st.selectbox("Scenario library", options=list(scenario_library.keys()), index=0)
    scenario_vals = scenario_library[scenario_name]
    manual_true = st.toggle("Manually edit true DLT probabilities", value=True)

    true_p = []
    for i, lab in enumerate(dose_labels):
        v0 = float(scenario_vals[i])
        if manual_true:
            v = st.number_input(
                f"True P(DLT) at {lab}",
                0.0, 1.0, v0, 0.01, key=f"true_{i}"
            )
        else:
            st.write(f"{lab}: {v0:.2f}")
            v = v0
        true_p.append(v)

st.divider()

tL, tR = st.columns([1.0, 1.0])

with tL:
    st.subheader("Target")
    use_r_target_rule = st.toggle(
        "Use R-style rule: target = P(true MTD level) + 0.03",
        value=True,
        help="Matches your R snippet pattern: target based on the true curve at a selected MTD-like level."
    )
    if use_r_target_rule:
        rule_level = st.selectbox(
            "Level used in the rule",
            options=list(range(0, 5)),
            index=2,
            format_func=lambda i: f"Level {i} ({dose_labels[i]})"
        )
        target = float(true_p[rule_level] + 0.03)
        st.write(f"Target = {target:.3f}")
    else:
        target = st.number_input("Target DLT probability", 0.05, 0.50, 0.25, 0.01)

true_mtd = find_true_mtd(true_p, target)

with tR:
    st.subheader("Derived truth")
    st.write(f"True MTD (closest to target {target:.2f}) = Level {true_mtd} ({dose_labels[true_mtd]})")

st.divider()

c1, c2 = st.columns([1.0, 1.0])

with c1:
    st.subheader("6+3 settings")
    max_n_6 = st.number_input("Max sample size (6+3)", 12, 120, 36, 3)
    accept_rule = st.selectbox(
        "Acceptance rule after expansion to 9",
        options=[1, 2],
        index=0,
        help="If 1/6 DLT, treat 3 more. Escalate if DLTs among 9 ≤ this number."
    )

with c2:
    st.subheader("CRM settings (R-aligned defaults)")
    max_n_crm = st.number_input("Max sample size (CRM)", 12, 120, 27, 3)
    cohort_size = st.number_input(
        "Cohort size (CO)",
        1, 12, 3, 1,
        help="Default 3 is taken from your R snippet (CO=3)."
    )
    sigma = st.number_input(
        "Prior sigma on theta",
        0.2, 5.0, 1.158, 0.01,
        help="dfcrm-style default when scale/sigma is not specified explicitly."
    )
    halfwidth = st.number_input(
        "getprior halfwidth",
        0.01, 0.30, 0.10, 0.01,
        help="Your R snippet uses halfwidth=0.1."
    )
    auto_prior = st.toggle(
        "Auto prior from true scenario",
        value=True,
        help="If on: prior nu = true MTD level, prior target = analysis target."
    )
    if auto_prior:
        prior_nu = int(true_mtd)
        prior_target = float(target)
        st.write(f"Prior nu = Level {prior_nu} | Prior target = {prior_target:.3f}")
    else:
        prior_nu = st.selectbox(
            "Prior nu (dose you believe is near MTD)",
            options=list(range(0, 5)),
            index=true_mtd,
            format_func=lambda i: f"Level {i} ({dose_labels[i]})"
        )
        prior_target = st.number_input("Prior target (for getprior)", 0.05, 0.50, float(target), 0.01)

    skeleton = getprior_like(target=prior_target, nu_0based=prior_nu, nlevel=5, halfwidth=float(halfwidth))
    sk_cols = st.columns(5)
    for i in range(5):
        sk_cols[i].metric(f"Skeleton L{i}", f"{skeleton[i]:.2f}")

# TITE-only controls
if crm_mode == "TITE-CRM":
    st.divider()
    st.subheader("TITE parameters")
    p1, p2, p3 = st.columns([1.0, 1.0, 1.1])
    with p1:
        obswin = st.number_input(
            "DLT window (weeks)",
            4.0, 52.0, 16.0, 1.0,
            help="Used in TITE weighting as obswin."
        )
    with p2:
        wait_time = st.number_input(
            "Wait.Time (weeks)",
            0.1, 8.0, 1.0, 0.1,
            help="Follow-up accrual increment between CRM updates."
        )
    with p3:
        advanced_tite = st.toggle(
            "Advanced time-to-event settings",
            value=False,
            help="Expose the event-time distribution parameters."
        )

    if advanced_tite:
        a1, a2 = st.columns(2)
        with a1:
            meanlog_event = st.number_input(
                "Event time meanlog",
                -1.0, 4.0, float(np.log(2.0)), 0.05,
                help="DLT event time ~ LogNormal(meanlog, sdlog). R severe uses meanlog=log(2)."
            )
        with a2:
            sdlog_event = st.number_input(
                "Event time sdlog",
                0.05, 2.0, 0.50, 0.05,
                help="R severe uses sdlog ~ 0.5 for acute severe time."
            )
    else:
        meanlog_event = float(np.log(2.0))
        sdlog_event = 0.50
else:
    # placeholders
    obswin = 16.0
    wait_time = 1.0
    meanlog_event = float(np.log(2.0))
    sdlog_event = 0.50

run = st.button("Run simulations")

if run:
    rng = np.random.default_rng(int(seed))
    ns = int(n_sims)
    true_p_arr = clamp_probs(true_p)

    # Storage
    sel_6 = np.zeros(5, dtype=int)
    sel_c = np.zeros(5, dtype=int)

    nmat_6 = np.zeros((ns, 5), dtype=int)
    nmat_c = np.zeros((ns, 5), dtype=int)

    dlts_6 = np.zeros(ns, dtype=int)
    dlts_c = np.zeros(ns, dtype=int)

    obs_dlts_6 = np.zeros(ns, dtype=int)
    obs_dlts_c = np.zeros(ns, dtype=int)

    exp_risk_6 = np.zeros(ns, dtype=float)
    exp_risk_c = np.zeros(ns, dtype=float)

    sw_6 = np.zeros(ns, dtype=float)
    sw_c = np.zeros(ns, dtype=float)
    osc_6 = np.zeros(ns, dtype=float)
    osc_c = np.zeros(ns, dtype=float)

    for s in range(ns):
        chosen6, n6, d6, od6, er6, path6 = run_6plus3(
            true_p=true_p_arr,
            start_level=int(start_level),
            max_n=int(max_n_6),
            accept_max_dlt=int(accept_rule),
            rng=rng
        )

        if crm_mode == "TITE-CRM":
            chosenc, nc, dc, odc, erc, pathc = run_titecrm_trial(
                true_p=true_p_arr,
                target=float(target),
                skeleton=skeleton,
                sigma=float(sigma),
                start_level=int(start_level),
                max_n=int(max_n_crm),
                cohort_size=int(cohort_size),
                wait_time=float(wait_time),
                obswin=float(obswin),
                meanlog_event=float(meanlog_event),
                sdlog_event=float(sdlog_event),
                rng=rng
            )
        else:
            chosenc, nc, dc, odc, erc, pathc = run_classic_crm_trial(
                true_p=true_p_arr,
                target=float(target),
                skeleton=skeleton,
                sigma=float(sigma),
                start_level=int(start_level),
                max_n=int(max_n_crm),
                cohort_size=int(cohort_size),
                rng=rng
            )

        sel_6[chosen6] += 1
        sel_c[chosenc] += 1

        nmat_6[s, :] = n6
        nmat_c[s, :] = nc

        dlts_6[s] = d6
        dlts_c[s] = dc

        obs_dlts_6[s] = od6
        obs_dlts_c[s] = odc

        exp_risk_6[s] = er6
        exp_risk_c[s] = erc

        sw_6[s] = switch_rate(path6)
        sw_c[s] = switch_rate(pathc)
        osc_6[s] = oscillation_index(path6)
        osc_c[s] = oscillation_index(pathc)

    # Summaries
    p_sel_6 = sel_6 / float(ns)
    p_sel_c = sel_c / float(ns)

    avg_n6 = np.mean(nmat_6, axis=0)
    avg_nc = np.mean(nmat_c, axis=0)

    mean_n6 = float(np.mean(nmat_6.sum(axis=1)))
    mean_nc = float(np.mean(nmat_c.sum(axis=1)))

    mean_latent_rate6 = float(np.mean(dlts_6 / np.maximum(nmat_6.sum(axis=1), 1)))
    mean_latent_ratec = float(np.mean(dlts_c / np.maximum(nmat_c.sum(axis=1), 1)))

    mean_observed_rate6 = float(np.mean(obs_dlts_6 / np.maximum(nmat_6.sum(axis=1), 1)))
    mean_observed_ratec = float(np.mean(obs_dlts_c / np.maximum(nmat_c.sum(axis=1), 1)))

    mean_exp6 = float(np.mean(exp_risk_6))
    mean_expc = float(np.mean(exp_risk_c))

    # ========================================================
    # Report header metrics
    # ========================================================

    st.subheader("Quality metrics (averages over simulated trials)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean expected DLT risk", f"{mean_expc:.3f}", help="Average per-patient DLT probability implied by allocations.")
    m2.metric("Mean observed DLT rate", f"{mean_observed_ratec:.3f}", help="Observed DLTs / N at end of each simulated trial.")
    m3.metric("Switch rate", f"{float(np.mean(sw_c)):.3f}", help="Fraction of adjacent patients assigned to different dose levels.")
    m4.metric("Oscillation index", f"{float(np.mean(osc_c)):.3f}", help="Fraction of A→B→A triples among patient dose assignments.")

    # ========================================================
    # Plots
    # ========================================================

    st.subheader("Results")

    x = np.arange(5)
    width = 0.38

    r1, r2 = st.columns(2)

    with r1:
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=140)
        ax.bar(x - width / 2, p_sel_6, width, label="6+3")
        ax.bar(x + width / 2, p_sel_c, width, label=crm_mode)
        ax.set_title("Probability of selecting each dose as MTD", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Probability", fontsize=9)
        compact_style(ax)
        ax.axvline(true_mtd, linewidth=1, alpha=0.6)
        ax.text(true_mtd + 0.05, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.9, "True MTD", fontsize=8)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with r2:
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=140)
        ax.bar(x - width / 2, avg_n6, width, label="6+3")
        ax.bar(x + width / 2, avg_nc, width, label=crm_mode)
        ax.set_title("Average number treated per dose level", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}\n{dose_labels[i]}" for i in range(5)], fontsize=8)
        ax.set_ylabel("Patients", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    r3, r4 = st.columns(2)

    with r3:
        fig, ax = plt.subplots(figsize=(5.6, 2.6), dpi=140)
        labels = ["6+3", crm_mode]
        obs = [mean_observed_rate6, mean_observed_ratec]
        exp = [mean_exp6, mean_expc]
        xi = np.arange(2)
        w = 0.38
        ax.bar(xi - w / 2, obs, w, label="Observed DLT rate")
        ax.bar(xi + w / 2, exp, w, label="Expected DLT risk")
        ax.set_title("Average DLT burden per patient", fontsize=10)
        ax.set_xticks(xi)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Rate", fontsize=9)
        compact_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        st.pyplot(fig, clear_figure=True)

    with r4:
        st.markdown("**Summary**")
        st.write(f"Mean sample size: 6+3 = {mean_n6:.1f} | {crm_mode} = {mean_nc:.1f}")
        st.write(f"Mean total DLTs (latent): 6+3 = {float(np.mean(dlts_6)):.2f} | {crm_mode} = {float(np.mean(dlts_c)):.2f}")
        st.write(f"Mean latent DLT rate: 6+3 = {mean_latent_rate6:.3f} | {crm_mode} = {mean_latent_ratec:.3f}")
        st.write(f"Mean observed DLT rate: 6+3 = {mean_observed_rate6:.3f} | {crm_mode} = {mean_observed_ratec:.3f}")
        st.write(f"Mean expected DLT risk: 6+3 = {mean_exp6:.3f} | {crm_mode} = {mean_expc:.3f}")

    # Optional: ping-pong distribution histogram (oscillation index)
    st.subheader("Ping-pong distribution (oscillation index)")
    fig, ax = plt.subplots(figsize=(5.6, 2.4), dpi=140)
    ax.hist(osc_c, bins=np.linspace(0, 1, 21), alpha=0.9)
    ax.set_xlabel("Oscillation index", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    compact_style(ax)
    st.pyplot(fig, clear_figure=True)
