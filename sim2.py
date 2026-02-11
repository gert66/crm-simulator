import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pprint import pformat

# -----------------------------
# CRM core
# -----------------------------
def normal_logpdf(x, mu=0.0, sigma=1.5):
    return -0.5*np.log(2*np.pi*sigma**2) - 0.5*((x-mu)/sigma)**2

def crm_p(skeleton_value, alpha_grid):
    # power model: p = s^(exp(alpha))
    return skeleton_value ** np.exp(alpha_grid)

def posterior_on_grid(alpha_grid, y, d, skeleton, prior_sigma=1.5):
    logprior = normal_logpdf(alpha_grid, 0.0, prior_sigma)
    loglik = np.zeros_like(alpha_grid, dtype=float)

    for yi, di in zip(y, d):
        p = crm_p(skeleton[di], alpha_grid)
        p = np.clip(p, 1e-12, 1-1e-12)
        loglik += yi*np.log(p) + (1-yi)*np.log(1-p)

    logpost = logprior + loglik
    m = np.max(logpost)
    w = np.exp(logpost - m)
    w /= np.sum(w)
    return w

def recommend_dose(alpha_grid, post_w, skeleton, target):
    K = len(skeleton)
    mean_p = np.zeros(K)
    for k in range(K):
        pk = crm_p(skeleton[k], alpha_grid)
        mean_p[k] = np.sum(post_w * pk)
    rec = int(np.argmin(np.abs(mean_p - target)))
    return rec, mean_p

def overdose_ok(alpha_grid, post_w, skeleton, dose, target, cutoff):
    pd = crm_p(skeleton[dose], alpha_grid)
    prob_over = float(np.sum(post_w * (pd > target)))
    return prob_over <= cutoff, prob_over

def simulate_one_trial(true_p, skeleton, N, target,
                       start_dose=0, no_skip=True,
                       overdose_control=False, od_cutoff=0.25,
                       alpha_grid=None, prior_sigma=1.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    K = len(true_p)
    if alpha_grid is None:
        alpha_grid = np.linspace(-4, 4, 2001)

    y = []
    d = []
    current = start_dose

    # prior only
    post_w = np.exp(normal_logpdf(alpha_grid, 0.0, prior_sigma))
    post_w /= post_w.sum()

    for _ in range(N):
        yi = int(rng.binomial(1, true_p[current]))
        y.append(yi)
        d.append(current)

        post_w = posterior_on_grid(alpha_grid, np.array(y), np.array(d), skeleton, prior_sigma)

        rec, _ = recommend_dose(alpha_grid, post_w, skeleton, target)

        if no_skip:
            rec = min(rec, current + 1)
            rec = max(rec, current - 1)

        if overdose_control:
            cand = rec
            # step down until safe, but don't go below 0
            while cand > 0:
                ok, _ = overdose_ok(alpha_grid, post_w, skeleton, cand, target, od_cutoff)
                if ok:
                    break
                cand -= 1
            current = cand
        else:
            current = rec

    rec_final, _ = recommend_dose(alpha_grid, post_w, skeleton, target)
    return rec_final, np.array(d), np.array(y)

def run_sims(n_sims, true_p, skeleton, N, target,
             overdose_control=False, od_cutoff=0.25,
             prior_sigma=1.5, seed=1):
    rng = np.random.default_rng(seed)
    true_p = np.array(true_p, float)
    skeleton = np.array(skeleton, float)
    K = len(true_p)

    finals = np.zeros(n_sims, dtype=int)
    dlts = np.zeros(n_sims, dtype=int)
    alloc = np.zeros((n_sims, K), dtype=int)

    alpha_grid = np.linspace(-4, 4, 2001)

    for i in range(n_sims):
        rec, treated, y = simulate_one_trial(
            true_p=true_p,
            skeleton=skeleton,
            N=N,
            target=target,
            overdose_control=overdose_control,
            od_cutoff=od_cutoff,
            alpha_grid=alpha_grid,
            prior_sigma=prior_sigma,
            rng=rng
        )
        finals[i] = rec
        dlts[i] = int(y.sum())
        for k in range(K):
            alloc[i, k] = int(np.sum(treated == k))

    true_mtd = int(np.argmin(np.abs(true_p - target)))

    res = {
        "true_mtd_index_0based": true_mtd,
        "true_mtd_level_1based": true_mtd + 1,
        "select_rate_true_mtd": float(np.mean(finals == true_mtd)),
        "final_recommendation_dist": (np.bincount(finals, minlength=K) / n_sims).tolist(),
        "mean_dlts": float(np.mean(dlts)),
        "mean_alloc_per_dose": alloc.mean(axis=0).tolist(),
        "finals_0based": finals,   # for plots
        "alloc_matrix": alloc      # for plots
    }
    return res

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="CRM Simulator", layout="wide")
st.title("CRM dose-escalation simulator (no time component)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Inputs")
    target = st.number_input("Target DLT probability (theta)", min_value=0.01, max_value=0.50, value=0.11, step=0.01)
    N = st.number_input("Total N (patients)", min_value=6, max_value=200, value=24, step=1)
    n_sims = st.number_input("Number of simulated trials", min_value=10, max_value=20000, value=200, step=10)
    prior_sigma = st.number_input("Prior sigma for alpha", min_value=0.1, max_value=5.0, value=1.5, step=0.1)

    overdose_control = st.checkbox("Use overdose control", value=True)
    od_cutoff = st.number_input("Overdose cutoff: P(p_d > target) must be ≤", min_value=0.01, max_value=0.99, value=0.25, step=0.05)

    st.caption("Dose levels are 1..5 in the UI. Internally Python uses 0..4.")

with col2:
    st.subheader("True scenario and skeleton")
    true_text = st.text_input("True DLT probabilities (comma-separated)", value="0.01,0.02,0.05,0.08,0.12")
    skel_text = st.text_input("Skeleton (comma-separated)", value="0.02,0.04,0.07,0.10,0.14")
    seed = st.number_input("Random seed", min_value=1, max_value=10_000_000, value=1, step=1)

def parse_list(txt):
    vals = [float(x.strip()) for x in txt.split(",") if x.strip() != ""]
    return vals

run = st.button("Run simulation")

if run:
    true_p = parse_list(true_text)
    skeleton = parse_list(skel_text)

    if len(true_p) != len(skeleton):
        st.error("True probabilities and skeleton must have the same length.")
        st.stop()

    if any(p <= 0 or p >= 1 for p in true_p + skeleton):
        st.error("All probabilities must be between 0 and 1 (exclusive).")
        st.stop()

    if any(np.diff(true_p) < 0) or any(np.diff(skeleton) < 0):
        st.warning("Note: true_p or skeleton is not increasing. That is allowed, but unusual in dose escalation.")

    with st.spinner("Running simulations..."):
        res = run_sims(
            n_sims=int(n_sims),
            true_p=true_p,
            skeleton=skeleton,
            N=int(N),
            target=float(target),
            overdose_control=bool(overdose_control),
            od_cutoff=float(od_cutoff),
            prior_sigma=float(prior_sigma),
            seed=int(seed)
        )

    st.subheader("Key results")
    st.write(f"True MTD (closest to target): dose level **{res['true_mtd_level_1based']}**")
    st.write(f"Selection rate of true MTD: **{res['select_rate_true_mtd']:.3f}**")
    st.write(f"Mean number of DLTs per trial: **{res['mean_dlts']:.3f}**")

    # Pretty print dictionary (excluding big arrays)
    small_res = {k: v for k, v in res.items() if k not in ["finals_0based", "alloc_matrix"]}
    st.code(pformat(small_res), language="python")

    # Plots
    st.subheader("Plots")
    K = len(parse_list(true_text))

    # 1) Final recommendation distribution
    fig1, ax1 = plt.subplots()
    x = np.arange(1, K+1)
    ax1.bar(x, res["final_recommendation_dist"])
    ax1.set_xlabel("Final recommended dose level")
    ax1.set_ylabel("Proportion of trials")
    ax1.set_title("Final recommendation distribution")
    st.pyplot(fig1)

    # 2) Mean allocation per dose
    fig2, ax2 = plt.subplots()
    ax2.bar(x, res["mean_alloc_per_dose"])
    ax2.set_xlabel("Dose level")
    ax2.set_ylabel("Mean # patients per trial")
    ax2.set_title("Mean allocation per dose")
    st.pyplot(fig2)

    st.caption("Tip: for quick tests, set trials to 200. For final reporting, 2000–10000 is typical.")
