import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from math import erf, sqrt, log

# ============================================================
# Small math helpers (no SciPy required)
# ============================================================

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def norm_ppf(p):
    # Acklam approximation (good enough for our use)
    # Source: Peter J. Acklam (public domain style implementation)
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = sqrt(-2*log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = sqrt(-2*log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

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

# ============================================================
# Gaussian copula for correlated Bernoulli (r=2 => rho=0 here)
# ============================================================

def gaussian_copula_joint_probs(p1, p2, gamma):
    # P11 = Phi2(z1,z2;gamma) approx via Monte Carlo (fast enough for small K)
    # We do a low-noise approximation using 40k draws.
    z1 = norm_ppf(p1)
    z2 = norm_ppf(p2)
    n = 40000
    rng = np.random.default_rng(12345)
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    x = u
    y = gamma*u + np.sqrt(max(1e-12, 1-gamma**2))*v
    P11 = np.mean((x <= z1) & (y <= z2))
    P10 = p1 - P11
    P01 = p2 - P11
    P00 = 1 - p1 - p2 + P11
    probs = np.array([P00, P01, P10, P11], dtype=float)
    probs = np.clip(probs, 0, 1)
    probs = probs / probs.sum()
    return probs  # order: 00,01,10,11

def simulate_acute_subacute(n_patients, P_acute, P_subacute, rho=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    K = len(P_acute)
    acute = np.zeros((n_patients, K), dtype=int)
    sub = np.zeros((n_patients, K), dtype=int)

    # r=2 in their script => rho_target = 0 (independence) per dose
    gamma = 0.0 if abs(rho) < 1e-12 else float(np.clip(rho, -0.95, 0.95))

    for k in range(K):
        p1 = float(P_acute[k])
        p2 = float(P_subacute[k])
        probs = gaussian_copula_joint_probs(p1, p2, gamma)
        idx = rng.choice(4, size=n_patients, replace=True, p=probs)
        # outcomes: 00,01,10,11 => (acute, subacute) = (0,0),(0,1),(1,0),(1,1)
        acute[:, k] = (idx == 2) | (idx == 3)
        sub[:, k] = (idx == 1) | (idx == 3)

    return acute, sub

# ============================================================
# dfcrm getprior-like skeleton
# We mimic the idea: nu is the "prior MTD" level, skeleton[nu] near target,
# and adjacent doses separated by roughly halfwidth.
# ============================================================

def getprior_like(target, nu, nlevel, halfwidth=0.1):
    # nu is 1-based in R; convert to 0-based
    nu0 = int(nu) - 1
    sk = np.zeros(nlevel, dtype=float)
    sk[nu0] = target

    # move down: target - halfwidth steps, clip > 0
    for k in range(nu0 - 1, -1, -1):
        sk[k] = max(1e-6, sk[k+1] - halfwidth)

    # move up: target + halfwidth steps, clip < 1
    for k in range(nu0 + 1, nlevel):
        sk[k] = min(1 - 1e-6, sk[k-1] + halfwidth)

    # enforce monotone and clamp
    sk = np.maximum.accumulate(sk)
    return clamp_probs(sk)

# ============================================================
# TITE-CRM core (power model) with weighted likelihood
# p_k(theta) = skeleton_k ^ exp(theta)
# theta ~ Normal(0, sigma^2)
# ============================================================

def crm_probs(theta_grid, skeleton):
    sk = clamp_probs(skeleton)
    a = np.exp(theta_grid)[:, None]
    return sk[None, :] ** a

def prior_logpdf(theta_grid, sigma):
    return -0.5 * (theta_grid / sigma) ** 2 - np.log(sigma * np.sqrt(2*np.pi))

def posterior_weights(theta_grid, sigma, skeleton, n_eff, y):
    P = crm_probs(theta_grid, skeleton)  # (G,K)
    n_eff = np.asarray(n_eff, dtype=float)
    y = np.asarray(y, dtype=float)
    ll = (y[None, :] * np.log(P) + (n_eff[None, :] - y[None, :]) * np.log(1 - P)).sum(axis=1)
    lp = prior_logpdf(theta_grid, sigma)
    log_post = lp + ll
    w = np.exp(log_post - logsumexp(log_post))
    return w, P

def titecrm_mtd(theta_grid, sigma, skeleton, n_eff, y, target):
    w, P = posterior_weights(theta_grid, sigma, skeleton, n_eff, y)
    post_mean = (w[:, None] * P).sum(axis=0)
    mtd = int(np.argmin(np.abs(post_mean - target)))
    return mtd, post_mean

# ============================================================
# Time-to-event simulation (simplified but consistent)
# We keep the spirit: event times lognormal; partial follow-up weights.
# ============================================================

def lognormal_time(meanlog, sdlog, rng):
    return float(rng.lognormal(mean=meanlog, sigma=sdlog))

def run_trial(
    P_acute, P_subacute,
    N_patient=27,
    CO=3,
    Wait_Time=1.0,
    operation_time=6.0,
    sigma=1.158,
    halfwidth=0.1,
    prior_target_acute=0.15,
    prior_nu_acute=3,
    target_acute=None,
    start_dose=2,
    rho=0.0,
    mu_tox=2.0,
    sd_tox=0.9,
    seed=None
):
    rng = np.random.default_rng(seed)

    K = len(P_acute)
    P_acute = clamp_probs(P_acute)
    P_subacute = clamp_probs(P_subacute)

    # Simulate correlated acute/subacute binary DLT indicators per dose for all patients
    acute_mat, sub_mat = simulate_acute_subacute(N_patient, P_acute, P_subacute, rho=rho, rng=rng)

    # Build prior skeleton like getprior
    prior_skeleton = getprior_like(prior_target_acute, prior_nu_acute, K, halfwidth=halfwidth)

    # Choose analysis target (their script uses scenario at True.MTD + 0.03)
    if target_acute is None:
        target_acute = 0.25

    # Patient ledger for CRM input (like tox, level, followup, obswin)
    # We mimic their idea:
    # - obswin is a patient-specific max time (surgery-ish), at least operation_time
    # - followup advances in Wait_Time increments
    # - DLT is observed if event_time < followup
    level = np.zeros(N_patient, dtype=int)
    tox_obs = np.zeros(N_patient, dtype=int)
    followup = np.zeros(N_patient, dtype=float)
    obswin = np.zeros(N_patient, dtype=float)

    # Simulate acute event times for severe events (if acute=1) as lognormal(log(2), 0.5)
    # If acute=0 => event time = very large
    acute_time = np.full((N_patient, K), 1e9, dtype=float)
    for i in range(N_patient):
        for k in range(K):
            if acute_mat[i, k] == 1:
                acute_time[i, k] = min(lognormal_time(meanlog=np.log(2.0), sdlog=0.5, rng=rng), 1e6)

    # Simple obswin: if acute DLT occurs before operation, obswin stretches a bit; else = operation_time
    # This is a simplification of their Surgery_Time logic but preserves: some patients have longer windows.
    def patient_obswin(i, k):
        t = acute_time[i, k]
        if t < operation_time:
            return max(operation_time, t + lognormal_time(meanlog=np.log(6.25), sdlog=0.5, rng=rng))
        return operation_time

    # Burning phase (their code starts at p=2 and runs j=1..6 before CRM triggered)
    p = int(start_dose)  # 1-based like R uses 2; we store 1..K
    j = 0
    CRM_run = False

    while not CRM_run and j < N_patient:
        level[j] = p
        ow = patient_obswin(j, p-1)
        obswin[j] = ow

        # follow-up at first decision: treat as ow (they set Wait to Surgery_Time early)
        followup[j] = ow

        # composite_DLT triggers only if j>=5 (since j was 1.. in R; their condition j>=6)
        composite = (acute_mat[j, p-1] == 1) and (j >= 5)
        tox_obs[j] = 1 if composite else 0

        if composite:
            CRM_run = True
        j += 1
        if j >= 6:
            break

    # Main TITE-CRM allocation in cohorts of CO
    theta_grid = np.linspace(-4.0, 4.0, 401)

    current_index = j
    current_dose = p

    while current_index < N_patient:
        # Build weighted counts from data observed so far (0..current_index-1)
        n_eff = np.zeros(K, dtype=float)
        y = np.zeros(K, dtype=float)
        for i in range(current_index):
            k = level[i] - 1
            fu = min(followup[i], obswin[i])
            w = fu / max(obswin[i], 1e-9)
            if tox_obs[i] == 1:
                n_eff[k] += 1.0
                y[k] += 1.0
            else:
                n_eff[k] += w

        mtd0, _ = titecrm_mtd(theta_grid, sigma, prior_skeleton, n_eff, y, target_acute)
        mtd = mtd0 + 1  # back to 1-based

        # step restriction like their code: if last dose < mtd => +1
        last_dose = level[current_index-1] if current_index > 0 else current_dose
        if last_dose < mtd:
            next_dose = last_dose + 1
        else:
            next_dose = mtd
        next_dose = int(np.clip(next_dose, 1, K))

        # Assign next cohort
        n_add = min(CO, N_patient - current_index)
        for t in range(n_add):
            i = current_index + t
            level[i] = next_dose

            ow = patient_obswin(i, next_dose-1)
            obswin[i] = ow

            # follow-up starts at Wait_Time and accrues (simplified)
            followup[i] = min(Wait_Time, ow)

            # observed tox if event occurs within follow-up
            tox_obs[i] = 1 if (acute_mat[i, next_dose-1] == 1 and acute_time[i, next_dose-1] < followup[i]) else 0

        # Advance "time": everyone gets additional follow-up Wait_Time (like they keep updating)
        for i in range(current_index + n_add):
            followup[i] = min(obswin[i], followup[i] + Wait_Time)

        current_index += n_add

    # Final MTD call on full data
    n_eff = np.zeros(K, dtype=float)
    y = np.zeros(K, dtype=float)
    for i in range(N_patient):
        k = level[i] - 1
        fu = min(followup[i], obswin[i])
        w = fu / max(obswin[i], 1e-9)
        if tox_obs[i] == 1:
            n_eff[k] += 1.0
            y[k] += 1.0
        else:
            n_eff[k] += w

    mtd0, _ = titecrm_mtd(theta_grid, sigma, prior_skeleton, n_eff, y, target_acute)
    final_mtd = mtd0 + 1

    # outputs similar to their Result:
    # - selection is final_mtd
    # - percentage treated at each dose
    alloc = np.bincount(level, minlength=K+1)[1:] / float(N_patient)

    return final_mtd, alloc, prior_skeleton

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="TITE-CRM (R-script style)", layout="wide")
st.title("TITE-CRM simulator (aligned with your Julius/dfcrm script)")

dose_labels = ["Dose 1", "Dose 2", "Dose 3", "Dose 4", "Dose 5"]

left, right = st.columns([1.0, 1.0])

with left:
    st.subheader("Trial setup")
    N_patient = st.number_input("N.patient", 6, 200, 27, 1)
    CO = st.number_input("CO (cohort size)", 1, 12, 3, 1)
    Wait_Time = st.number_input("Wait.Time (weeks)", 0.1, 8.0, 1.0, 0.1)
    start_dose = st.number_input("Start dose (1..5)", 1, 5, 2, 1)
    n_sims = st.number_input("NREP (simulated trials)", 200, 20000, 1000, 100)
    seed = st.number_input("Seed", 1, 10_000_000, 123, 1)

with right:
    st.subheader("True scenario")
    P_acute = []
    P_sub = []
    cols = st.columns(5)
    default_acute = [0.01, 0.02, 0.12, 0.20, 0.35]
    default_sub = [0.15, 0.22, 0.28, 0.35, 0.40]
    for i in range(5):
        with cols[i]:
            P_acute.append(st.number_input(f"Acute P @ L{i+1}", 0.0, 1.0, float(default_acute[i]), 0.01, key=f"a{i}"))
            P_sub.append(st.number_input(f"Sub P @ L{i+1}", 0.0, 1.0, float(default_sub[i]), 0.01, key=f"s{i}"))

st.divider()

c1, c2 = st.columns([1.0, 1.0])

with c1:
    st.subheader("Prior / dfcrm-like defaults")
    sigma = st.number_input("Prior sigma on theta", 0.2, 5.0, 1.158, 0.01,
                            help="dfcrm default scale is often ~sqrt(1.34)=1.158.")
    halfwidth = st.number_input("halfwidth", 0.01, 0.30, 0.10, 0.01)
    prior_target_acute = st.number_input("prior.target.acute", 0.01, 0.99, 0.15, 0.01)
    prior_nu_acute = st.number_input("prior.MTD.acute (nu)", 1, 5, 3, 1)

with c2:
    st.subheader("Analysis target")
    use_rule_target = st.toggle("Use R-rule: target = P(true MTD) + 0.03", value=True)
    true_mtd_level = st.number_input("True.MTD.acute (for the rule)", 1, 5, 3, 1)
    if use_rule_target:
        target_acute = float(P_acute[true_mtd_level-1] + 0.03)
        st.write(f"target.acute = {target_acute:.3f}")
    else:
        target_acute = st.number_input("target.acute", 0.01, 0.99, 0.25, 0.01)

run = st.button("Run simulations")

if run:
    rng = np.random.default_rng(int(seed))
    K = 5
    sel = np.zeros(K, dtype=int)
    alloc_mat = np.zeros((int(n_sims), K), dtype=float)

    # show skeleton once
    prior_skeleton = getprior_like(target=prior_target_acute, nu=prior_nu_acute, nlevel=K, halfwidth=halfwidth)

    for r in range(int(n_sims)):
        mtd, alloc, _ = run_trial(
            P_acute=P_acute, P_subacute=P_sub,
            N_patient=int(N_patient),
            CO=int(CO),
            Wait_Time=float(Wait_Time),
            sigma=float(sigma),
            halfwidth=float(halfwidth),
            prior_target_acute=float(prior_target_acute),
            prior_nu_acute=int(prior_nu_acute),
            target_acute=float(target_acute),
            start_dose=int(start_dose),
            rho=0.0,
            seed=int(rng.integers(1, 2_000_000_000))
        )
        sel[mtd-1] += 1
        alloc_mat[r, :] = alloc

    p_sel = sel / float(n_sims)
    avg_alloc = alloc_mat.mean(axis=0)

    st.subheader("Prior skeleton (getprior-like)")
    sk_cols = st.columns(5)
    for i in range(5):
        sk_cols[i].metric(f"L{i+1}", f"{prior_skeleton[i]:.2f}")

    st.subheader("Results")
    cA, cB = st.columns(2)

    with cA:
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=140)
        x = np.arange(1, 6)
        ax.bar(x, p_sel)
        ax.set_title("Probability of selecting each dose as MTD", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in x], fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        compact_style(ax)
        st.pyplot(fig, clear_figure=True)

    with cB:
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=140)
        x = np.arange(1, 6)
        ax.bar(x, avg_alloc)
        ax.set_title("Mean proportion treated at each dose", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in x], fontsize=9)
        ax.set_ylabel("Proportion", fontsize=9)
        compact_style(ax)
        st.pyplot(fig, clear_figure=True)
