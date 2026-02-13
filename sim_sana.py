# sim_sana.py
# Streamlit-ready Classic CRM simulator (no SciPy needed)
# Dependencies: streamlit, numpy

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
import numpy as np
import streamlit as st


# -----------------------------
# Helpers
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_floats_csv(s: str) -> List[float]:
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip() != ""]
    return [float(p) for p in parts]


# -----------------------------
# Gauss-Hermite quadrature (no SciPy)
# -----------------------------
# We need nodes/weights for ∫ f(x) e^{-x^2} dx.
# We implement a stable-ish GH generator using the Golub-Welsch method.
# Reference: GH via eigen-decomposition of Jacobi matrix.

def hermgauss(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("n must be positive")
    i = np.arange(1, n, dtype=float)
    a = np.zeros(n, dtype=float)
    b = np.sqrt(i / 2.0)

    # Jacobi matrix for Hermite polynomials
    J = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)
    vals, vecs = np.linalg.eigh(J)

    x = vals
    # weights: w_k = sqrt(pi) * (v_0k)^2
    w = math.sqrt(math.pi) * (vecs[0, :] ** 2)
    return x, w


# -----------------------------
# CRM config
# -----------------------------

@dataclass
class CRMConfig:
    skeleton: List[float]
    target: float = 0.25
    prior_mu: float = 0.0
    prior_sigma: float = 1.0

    max_step: int = 1

    use_overdose_control: bool = True
    overdose_cutoff: float = 0.25  # require P(p > target) < cutoff

    gh_n: int = 61  # 41/61/81 typical

    # Startup escape: avoids "stuck at level 0"
    startup_escalate_if_zero: bool = True
    startup_margin: float = 0.90
    cohort_dlt_threshold: int = 1  # allow escalate if last cohort DLTs <= threshold


@dataclass
class TrialConfig:
    cohort_size: int = 3
    n_cohorts: int = 10
    start_level: int = 0


# -----------------------------
# CRM posterior (power model)
# p_i(alpha) = skeleton_i ^ exp(alpha)
# alpha ~ Normal(mu, sigma^2)
# Posterior via GH nodes/weights
# -----------------------------

class CRMPosterior:
    def __init__(self, cfg: CRMConfig):
        self.cfg = cfg
        self.skeleton = np.array(cfg.skeleton, dtype=float)
        if np.any(self.skeleton <= 0) or np.any(self.skeleton >= 1):
            raise ValueError("All skeleton values must be strictly between 0 and 1.")

        x, w = hermgauss(cfg.gh_n)
        # Transform for normal prior using:
        # ∫ f(alpha) φ((alpha-mu)/sigma) d alpha
        # with alpha = mu + sqrt(2)*sigma*x, and weight factor w / sqrt(pi)
        self.alpha_nodes = cfg.prior_mu + math.sqrt(2.0) * cfg.prior_sigma * x
        self.base_weights = w / math.sqrt(math.pi)  # sums to 1 for f=1 under normal

    def p_tox(self, alpha: np.ndarray) -> np.ndarray:
        a = np.exp(alpha).reshape(-1, 1)  # exp(alpha) positive
        return np.power(self.skeleton.reshape(1, -1), a)

    def log_likelihood(self, alpha: np.ndarray, n: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = self.p_tox(alpha)  # (A, D)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        ll = (y.reshape(1, -1) * np.log(p) + (n.reshape(1, -1) - y.reshape(1, -1)) * np.log(1 - p)).sum(axis=1)
        return ll

    def posterior_weights(self, log_like: np.ndarray) -> np.ndarray:
        m = float(np.max(log_like))
        unnorm = self.base_weights * np.exp(log_like - m)
        s = float(np.sum(unnorm))
        if s <= 0 or not np.isfinite(s):
            return self.base_weights / float(np.sum(self.base_weights))
        return unnorm / s

    def posterior_mean_p(self, post_w: np.ndarray) -> np.ndarray:
        p = self.p_tox(self.alpha_nodes)  # (A, D)
        return (post_w.reshape(-1, 1) * p).sum(axis=0)

    def posterior_prob_over_target(self, post_w: np.ndarray, target: float) -> np.ndarray:
        p = self.p_tox(self.alpha_nodes)  # (A, D)
        ind = (p > target).astype(float)
        return (post_w.reshape(-1, 1) * ind).sum(axis=0)


# -----------------------------
# Dose selection
# -----------------------------

def select_next_level(
    cfg: CRMConfig,
    current_level: int,
    post_mean_p: np.ndarray,
    post_prob_over_target: np.ndarray,
) -> int:
    n_levels = len(post_mean_p)
    candidates = list(range(n_levels))

    if cfg.use_overdose_control:
        candidates = [i for i in candidates if post_prob_over_target[i] < cfg.overdose_cutoff]
        if not candidates:
            candidates = [0]

    # closest posterior mean to target
    best = min(candidates, key=lambda i: abs(float(post_mean_p[i]) - cfg.target))

    # enforce max step
    lo = max(0, current_level - cfg.max_step)
    hi = min(n_levels - 1, current_level + cfg.max_step)
    return int(clamp(best, lo, hi))


def apply_startup_escape(
    cfg: CRMConfig,
    current_level: int,
    proposed_level: int,
    post_mean_p: np.ndarray,
    last_cohort_dlts: int,
) -> int:
    if not cfg.startup_escalate_if_zero:
        return proposed_level
    if current_level != 0 or proposed_level != 0:
        return proposed_level
    if len(post_mean_p) < 2:
        return proposed_level

    safe_enough = float(post_mean_p[0]) < (cfg.target * cfg.startup_margin)
    ok_recent = last_cohort_dlts <= cfg.cohort_dlt_threshold
    if safe_enough and ok_recent:
        return 1
    return proposed_level


# -----------------------------
# Simulation
# -----------------------------

def simulate_one_trial(
    crm_cfg: CRMConfig,
    trial_cfg: TrialConfig,
    true_tox: List[float],
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    post = CRMPosterior(crm_cfg)

    n_levels = len(crm_cfg.skeleton)
    if len(true_tox) != n_levels:
        raise ValueError("true_tox must have same length as skeleton")

    level = trial_cfg.start_level
    cohorts_level: List[int] = []
    cohorts_dlts: List[int] = []

    # aggregated counts per level
    n = np.zeros(n_levels, dtype=int)
    y = np.zeros(n_levels, dtype=int)

    for _ in range(trial_cfg.n_cohorts):
        p_true = float(true_tox[level])
        dlts = int(rng.binomial(trial_cfg.cohort_size, p_true))

        cohorts_level.append(level)
        cohorts_dlts.append(dlts)

        n[level] += trial_cfg.cohort_size
        y[level] += dlts

        ll = post.log_likelihood(post.alpha_nodes, n.astype(float), y.astype(float))
        w = post.posterior_weights(ll)
        mean_p = post.posterior_mean_p(w)
        prob_over = post.posterior_prob_over_target(w, crm_cfg.target)

        proposed = select_next_level(crm_cfg, level, mean_p, prob_over)
        proposed = apply_startup_escape(crm_cfg, level, proposed, mean_p, dlts)

        level = proposed

    # final recommendation based on final posterior
    ll = post.log_likelihood(post.alpha_nodes, n.astype(float), y.astype(float))
    w = post.posterior_weights(ll)
    mean_p = post.posterior_mean_p(w)
    prob_over = post.posterior_prob_over_target(w, crm_cfg.target)
    final = select_next_level(crm_cfg, level, mean_p, prob_over)

    return {
        "final_level": int(final),
        "trajectory_levels": cohorts_level,
        "trajectory_dlts": cohorts_dlts,
        "final_post_mean_p": mean_p.tolist(),
        "final_post_prob_over_target": prob_over.tolist(),
        "n": n.tolist(),
        "y": y.tolist(),
    }


def simulate_many(
    crm_cfg: CRMConfig,
    trial_cfg: TrialConfig,
    true_tox: List[float],
    n_trials: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    finals = np.zeros(len(crm_cfg.skeleton), dtype=int)
    stuck_all_0 = 0

    for _ in range(n_trials):
        s = int(rng.integers(1, 2_000_000_000))
        out = simulate_one_trial(crm_cfg, trial_cfg, true_tox, seed=s)
        finals[out["final_level"]] += 1
        if all(lvl == 0 for lvl in out["trajectory_levels"]):
            stuck_all_0 += 1

    probs = (finals / max(1, n_trials)).tolist()
    return {
        "final_counts": finals.tolist(),
        "final_probs": probs,
        "stuck_all_cohorts_at_0": int(stuck_all_0),
    }


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Classic CRM Simulator", layout="wide")
    st.title("Classic CRM Simulator")
    st.caption("No SciPy required. Power model with Gauss-Hermite posterior integration.")

    with st.sidebar:
        st.header("Inputs")

        skeleton_str = st.text_input(
            "Skeleton (comma-separated, strictly between 0 and 1)",
            value="0.05, 0.08, 0.12, 0.18, 0.25, 0.33",
        )
        true_tox_str = st.text_input(
            "True toxicity curve for simulation (same length as skeleton)",
            value="0.03, 0.07, 0.12, 0.20, 0.28, 0.40",
        )

        target = st.number_input("Target DLT rate", min_value=0.01, max_value=0.80, value=0.25, step=0.01)
        prior_mu = st.number_input("Prior mu (alpha)", value=0.0, step=0.1)
        prior_sigma = st.number_input("Prior sigma (alpha)", min_value=0.01, value=1.0, step=0.1)

        st.subheader("Design")
        cohort_size = st.number_input("Cohort size", min_value=1, max_value=30, value=3, step=1)
        n_cohorts = st.number_input("Number of cohorts", min_value=1, max_value=50, value=10, step=1)
        start_level = st.number_input("Start dose level (0-indexed)", min_value=0, value=0, step=1)
        max_step = st.number_input("Max step per cohort", min_value=1, max_value=10, value=1, step=1)

        st.subheader("Overdose control")
        use_oc = st.checkbox("Use overdose control", value=True)
        oc_cutoff = st.number_input("Overdose cutoff P(p > target) <", min_value=0.01, max_value=0.99, value=0.25, step=0.01)

        st.subheader("Anti-stuck startup escape")
        use_escape = st.checkbox("Startup escape rule (avoid stuck at level 0)", value=True)
        startup_margin = st.number_input("Startup margin (target * margin)", min_value=0.50, max_value=1.20, value=0.90, step=0.01)
        cohort_dlt_threshold = st.number_input("Escalate if last cohort DLTs ≤", min_value=0, max_value=10, value=1, step=1)

        st.subheader("Posterior integration")
        gh_n = st.selectbox("Gauss-Hermite points", options=[41, 61, 81, 101], index=1)

        st.subheader("Simulation")
        n_trials = st.number_input("Number of simulated trials", min_value=10, max_value=5000, value=300, step=10)
        seed = st.number_input("Simulation seed", min_value=1, value=2026, step=1)

        run_btn = st.button("Run simulation", type="primary")

    if not run_btn:
        st.info("Set inputs in the sidebar and click 'Run simulation'.")
        return

    try:
        skeleton = parse_floats_csv(skeleton_str)
        true_tox = parse_floats_csv(true_tox_str)
    except Exception:
        st.error("Could not parse skeleton/true toxicity. Use comma-separated numbers.")
        return

    if len(skeleton) < 2:
        st.error("Skeleton must have at least 2 dose levels.")
        return

    if len(true_tox) != len(skeleton):
        st.error("True toxicity curve must have the same length as the skeleton.")
        return

    if start_level < 0 or start_level >= len(skeleton):
        st.error("Start level is out of range.")
        return

    crm_cfg = CRMConfig(
        skeleton=skeleton,
        target=float(target),
        prior_mu=float(prior_mu),
        prior_sigma=float(prior_sigma),
        max_step=int(max_step),
        use_overdose_control=bool(use_oc),
        overdose_cutoff=float(oc_cutoff),
        gh_n=int(gh_n),
        startup_escalate_if_zero=bool(use_escape),
        startup_margin=float(startup_margin),
        cohort_dlt_threshold=int(cohort_dlt_threshold),
    )

    trial_cfg = TrialConfig(
        cohort_size=int(cohort_size),
        n_cohorts=int(n_cohorts),
        start_level=int(start_level),
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Single trial example")
        one = simulate_one_trial(crm_cfg, trial_cfg, true_tox, seed=int(seed))
        st.write("Final recommended level:", one["final_level"])
        st.write("Trajectory levels:", one["trajectory_levels"])
        st.write("Trajectory DLTs:", one["trajectory_dlts"])

        st.write("Final posterior mean p per level:")
        st.code([round(x, 4) for x in one["final_post_mean_p"]])

        st.write("Final P(p > target) per level:")
        st.code([round(x, 4) for x in one["final_post_prob_over_target"]])

        st.write("Observed totals n per level:", one["n"])
        st.write("Observed totals y per level:", one["y"])

    with col2:
        st.subheader("Many trials summary")
        sim = simulate_many(crm_cfg, trial_cfg, true_tox, n_trials=int(n_trials), seed=int(seed))
        st.write("Final recommendation probabilities per dose level:")
        st.code([round(x, 4) for x in sim["final_probs"]])
        st.write("Stuck all cohorts at level 0:", sim["stuck_all_cohorts_at_0"])

        finals = np.array(sim["final_probs"], dtype=float)
        st.bar_chart(finals)

    st.success("Done.")


if __name__ == "__main__":
    main()
