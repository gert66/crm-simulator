"""
crm_tite_simulator.py

GitHub-ready Python implementation of:
- Classic CRM (binary DLT)
- Optional TITE-CRM (time-to-event weighted likelihood)
- Overdose Control
- Startup escape rule to avoid getting stuck at dose level 0
- Gauss-Hermite quadrature for robust posterior integration (no MCMC needed)

Dependencies:
    pip install numpy scipy

Run:
    python crm_tite_simulator.py

You can also import and use `run_simulation(...)` from another script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import math
import numpy as np
from scipy.special import expit  # sigmoid


# -----------------------------
# Utilities
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_log(x: float, eps: float = 1e-15) -> float:
    return math.log(max(eps, min(1 - eps, x)))


# -----------------------------
# Model and Config
# -----------------------------

@dataclass
class CRMConfig:
    # Dose skeleton (monotone increasing "prior guesses")
    skeleton: List[float]

    # Target toxicity probability
    target: float = 0.25

    # Prior on alpha for the power model p_i(alpha) = skeleton_i ^ exp(alpha)
    # We'll use alpha ~ Normal(mu, sigma^2)
    prior_mu: float = 0.0
    prior_sigma: float = 1.0

    # Dose transition constraints
    max_step: int = 1  # 1 = only move 1 level up/down per cohort

    # Overdose control
    use_overdose_control: bool = True
    overdose_cutoff: float = 0.25  # e.g. 0.25 means require P(p_i > target) < 0.25

    # Posterior integration
    gh_n: int = 61  # 41/61/81 typical; higher is slower but more stable

    # Startup escape: prevents being stuck at dose 0 due to early conservative posterior
    startup_escalate_if_zero: bool = True
    startup_margin: float = 0.90  # escalate if current dose estimated tox < target*margin and last cohort had <= cohort_dlt_threshold

    # TITE options
    use_tite: bool = False
    tite_assessment_window: float = 1.0  # normalized window length (e.g. 1.0 means full window)
    tite_weight_power: float = 1.0       # w = (t / window)^power, clamp [0,1]

    # Random seed for reproducibility (simulation)
    seed: int = 7


@dataclass
class TrialConfig:
    cohort_size: int = 3
    n_cohorts: int = 10
    start_level: int = 0


# -----------------------------
# CRM Core: likelihood and posterior via Gauss-Hermite
# -----------------------------

class CRMPosterior:
    """
    Power model:
        p_i(alpha) = skeleton_i ** exp(alpha)
    with alpha ~ Normal(mu, sigma^2).
    """

    def __init__(self, cfg: CRMConfig):
        self.cfg = cfg
        self.skeleton = np.array(cfg.skeleton, dtype=float)
        if np.any(self.skeleton <= 0) or np.any(self.skeleton >= 1):
            raise ValueError("All skeleton values must be strictly between 0 and 1.")

        # Precompute GH nodes/weights (for standard normal integral)
        self.gh_x, self.gh_w = np.polynomial.hermite.hermgauss(cfg.gh_n)
        # Transform for N(mu, sigma^2): alpha = mu + sqrt(2)*sigma*x
        self.alpha_nodes = cfg.prior_mu + math.sqrt(2.0) * cfg.prior_sigma * self.gh_x
        # Standard normal weight factor
        self.gh_weights = self.gh_w / math.sqrt(math.pi)

    def p_tox(self, alpha: np.ndarray) -> np.ndarray:
        """
        Returns matrix of p_i(alpha) with shape (len(alpha), n_doses)
        """
        a = np.exp(alpha).reshape(-1, 1)  # exp(alpha) positive
        # skeleton ** exp(alpha)
        return np.power(self.skeleton.reshape(1, -1), a)

    def log_likelihood_classic(self, alpha: np.ndarray, n: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Classic CRM likelihood with binomial data at each dose i:
            y_i ~ Binomial(n_i, p_i(alpha))
        Return log L(alpha) for each alpha node, shape (len(alpha),)
        """
        p = self.p_tox(alpha)  # (A, D)
        # log L = sum_i [ y_i log p_i + (n_i - y_i) log(1-p_i) ]
        ll = (y.reshape(1, -1) * np.log(np.clip(p, 1e-15, 1 - 1e-15)) +
              (n.reshape(1, -1) - y.reshape(1, -1)) * np.log(np.clip(1 - p, 1e-15, 1 - 1e-15))).sum(axis=1)
        return ll

    def log_likelihood_tite(self, alpha: np.ndarray, n_eff: np.ndarray, y_eff: np.ndarray) -> np.ndarray:
        """
        TITE approximation as weighted Bernoulli/binomial:
            contribution per patient: w * y * log p + w * (1-y) * log(1-p)
        Aggregated as:
            y_eff_i = sum(w_j * y_j), n_eff_i = sum(w_j)
        """
        p = self.p_tox(alpha)  # (A, D)
        ll = (y_eff.reshape(1, -1) * np.log(np.clip(p, 1e-15, 1 - 1e-15)) +
              (n_eff.reshape(1, -1) - y_eff.reshape(1, -1)) * np.log(np.clip(1 - p, 1e-15, 1 - 1e-15))).sum(axis=1)
        return ll

    def posterior_weights(self, log_like: np.ndarray) -> np.ndarray:
        """
        Returns normalized posterior weights over GH nodes.
        Prior is already built into GH integration via nodes/weights.
        """
        # unnormalized posterior mass at each node: w_k * exp(log_like_k)
        m = np.max(log_like)
        unnorm = self.gh_weights * np.exp(log_like - m)
        s = np.sum(unnorm)
        if s <= 0 or not np.isfinite(s):
            # fallback to prior if numerically broken
            return self.gh_weights / np.sum(self.gh_weights)
        return unnorm / s

    def posterior_mean_p(self, post_w: np.ndarray) -> np.ndarray:
        """
        Posterior mean of toxicity probability for each dose level.
        """
        p = self.p_tox(self.alpha_nodes)  # (A, D)
        return (post_w.reshape(-1, 1) * p).sum(axis=0)

    def posterior_prob_over_target(self, post_w: np.ndarray, target: float) -> np.ndarray:
        """
        For each dose i, compute P(p_i(alpha) > target | data).
        """
        p = self.p_tox(self.alpha_nodes)  # (A, D)
        ind = (p > target).astype(float)
        return (post_w.reshape(-1, 1) * ind).sum(axis=0)


# -----------------------------
# Dose selection rules
# -----------------------------

def select_dose_level(
    cfg: CRMConfig,
    current_level: int,
    post_mean_p: np.ndarray,
    post_prob_over_target: np.ndarray,
) -> int:
    """
    Select next level based on:
    - Pick dose with posterior mean closest to target
    - Apply overdose control by excluding doses where P(p > target) >= cutoff
    - Enforce max_step from current_level
    """
    n_levels = len(post_mean_p)
    candidates = list(range(n_levels))

    if cfg.use_overdose_control:
        candidates = [i for i in candidates if post_prob_over_target[i] < cfg.overdose_cutoff]
        if len(candidates) == 0:
            # If everything is "too risky", de-escalate to lowest available
            candidates = [0]

    # Choose closest to target among candidates
    distances = [(i, abs(post_mean_p[i] - cfg.target)) for i in candidates]
    best_i = min(distances, key=lambda t: t[1])[0]

    # Enforce max step
    lo = max(0, current_level - cfg.max_step)
    hi = min(n_levels - 1, current_level + cfg.max_step)
    best_i = int(clamp(best_i, lo, hi))

    return best_i


def startup_escape_rule(
    cfg: CRMConfig,
    current_level: int,
    next_level: int,
    post_mean_p: np.ndarray,
    last_cohort_dlts: int,
    cohort_dlt_threshold: int = 1,
) -> int:
    """
    If we're stuck at level 0 due to conservative overdose control, give a controlled push:
    - Only triggers when current level is 0 and next_level is 0.
    - Escalate to 1 if posterior mean at level 0 looks sufficiently below target,
      and last cohort at level 0 was not alarming (<= threshold DLTs).
    """
    if not cfg.startup_escalate_if_zero:
        return next_level
    if current_level != 0 or next_level != 0:
        return next_level

    safe_enough = post_mean_p[0] < (cfg.target * cfg.startup_margin)
    not_too_toxic_recently = last_cohort_dlts <= cohort_dlt_threshold

    if safe_enough and not_too_toxic_recently and len(post_mean_p) > 1:
        return 1
    return next_level


# -----------------------------
# Data structures for running a trial
# -----------------------------

@dataclass
class CohortOutcome:
    dose_level: int
    n: int
    dlts: int
    # TITE fields
    followup_times: Optional[List[float]] = None
    window: Optional[float] = None


@dataclass
class TrialState:
    level: int
    cohorts: List[CohortOutcome]


# -----------------------------
# Simulation: generating outcomes
# -----------------------------

def simulate_classic_cohort(rng: np.random.Generator, true_p: float, n: int) -> int:
    """
    Simulate DLT count for a cohort of size n with probability true_p.
    """
    return int(rng.binomial(n, true_p))


def simulate_tite_cohort(
    rng: np.random.Generator,
    true_p: float,
    n: int,
    window: float = 1.0,
) -> Tuple[int, List[float]]:
    """
    Simple TITE simulator:
    - DLT occurs with probability true_p during the window.
    - If DLT occurs, event time ~ Uniform(0, window)
    - If no DLT, observed followup time ~ Uniform(0, window) (censoring at random)
    Returns: (dlts_count, followup_times)
    """
    dlts = 0
    times: List[float] = []
    for _ in range(n):
        dlt = rng.random() < true_p
        if dlt:
            dlts += 1
            t = rng.random() * window
            times.append(float(t))
        else:
            # censored followup
            t = rng.random() * window
            times.append(float(t))
    return dlts, times


def tite_weights(times: List[float], window: float, power: float) -> List[float]:
    w = []
    for t in times:
        frac = 0.0 if window <= 0 else t / window
        frac = clamp(frac, 0.0, 1.0)
        w.append(float(frac ** power))
    return w


# -----------------------------
# Running a trial
# -----------------------------

def build_aggregated_data(
    cfg: CRMConfig,
    cohorts: List[CohortOutcome],
    n_levels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        classic: n, y
        tite: n_eff, y_eff
    If cfg.use_tite is False, tite arrays still returned but unused.
    """
    n = np.zeros(n_levels, dtype=float)
    y = np.zeros(n_levels, dtype=float)

    n_eff = np.zeros(n_levels, dtype=float)
    y_eff = np.zeros(n_levels, dtype=float)

    for c in cohorts:
        i = c.dose_level
        n[i] += c.n
        y[i] += c.dlts

        if cfg.use_tite:
            if c.followup_times is None or c.window is None:
                raise ValueError("TITE enabled but cohort missing followup_times/window.")
            w = tite_weights(c.followup_times, c.window, cfg.tite_weight_power)
            # For simplicity: assume dlts are the first `dlts` patients in this cohort for weighting.
            # If you track individual event indicators, replace this block with patient-level aggregation.
            indicators = [1.0] * c.dlts + [0.0] * (c.n - c.dlts)
            indicators = indicators[: c.n]
            if len(indicators) != len(w):
                # If mismatch due to the simplification, fall back to unweighted count
                w = [1.0] * c.n
                indicators = [1.0] * c.dlts + [0.0] * (c.n - c.dlts)

            n_eff[i] += sum(w)
            y_eff[i] += sum(w_j * y_j for w_j, y_j in zip(w, indicators))

    return n, y, n_eff, y_eff


def run_one_trial(
    crm_cfg: CRMConfig,
    trial_cfg: TrialConfig,
    true_tox: List[float],
) -> Dict[str, object]:
    """
    Runs one simulated trial and returns a dictionary with trajectory and final recommendation.
    """
    rng = np.random.default_rng(crm_cfg.seed)
    post = CRMPosterior(crm_cfg)

    n_levels = len(crm_cfg.skeleton)
    if len(true_tox) != n_levels:
        raise ValueError("true_tox must have same length as skeleton.")

    state = TrialState(level=trial_cfg.start_level, cohorts=[])

    for k in range(trial_cfg.n_cohorts):
        level = state.level
        p_true = true_tox[level]

        if crm_cfg.use_tite:
            dlts, times = simulate_tite_cohort(
                rng, p_true, trial_cfg.cohort_size, window=crm_cfg.tite_assessment_window
            )
            cohort = CohortOutcome(
                dose_level=level,
                n=trial_cfg.cohort_size,
                dlts=dlts,
                followup_times=times,
                window=crm_cfg.tite_assessment_window,
            )
        else:
            dlts = simulate_classic_cohort(rng, p_true, trial_cfg.cohort_size)
            cohort = CohortOutcome(dose_level=level, n=trial_cfg.cohort_size, dlts=dlts)

        state.cohorts.append(cohort)

        # Aggregate data and update posterior
        n, y, n_eff, y_eff = build_aggregated_data(crm_cfg, state.cohorts, n_levels)

        if crm_cfg.use_tite:
            ll = post.log_likelihood_tite(post.alpha_nodes, n_eff, y_eff)
        else:
            ll = post.log_likelihood_classic(post.alpha_nodes, n.astype(int), y.astype(int))

        post_w = post.posterior_weights(ll)
        post_mean_p = post.posterior_mean_p(post_w)
        post_prob_over_target = post.posterior_prob_over_target(post_w, crm_cfg.target)

        # Dose selection
        next_level = select_dose_level(crm_cfg, level, post_mean_p, post_prob_over_target)

        # Startup escape if stuck at level 0
        last_cohort_dlts = cohort.dlts
        next_level = startup_escape_rule(
            crm_cfg,
            current_level=level,
            next_level=next_level,
            post_mean_p=post_mean_p,
            last_cohort_dlts=last_cohort_dlts,
            cohort_dlt_threshold=1,
        )

        state.level = next_level

    # Final recommendation after last cohort, use the same selection logic one more time
    n, y, n_eff, y_eff = build_aggregated_data(crm_cfg, state.cohorts, n_levels)
    if crm_cfg.use_tite:
        ll = post.log_likelihood_tite(post.alpha_nodes, n_eff, y_eff)
    else:
        ll = post.log_likelihood_classic(post.alpha_nodes, n.astype(int), y.astype(int))

    post_w = post.posterior_weights(ll)
    post_mean_p = post.posterior_mean_p(post_w)
    post_prob_over_target = post.posterior_prob_over_target(post_w, crm_cfg.target)
    final_level = select_dose_level(crm_cfg, state.level, post_mean_p, post_prob_over_target)

    return {
        "final_recommended_level": int(final_level),
        "final_post_mean_p": post_mean_p.tolist(),
        "final_post_prob_over_target": post_prob_over_target.tolist(),
        "trajectory_levels": [c.dose_level for c in state.cohorts],
        "trajectory_dlts": [c.dlts for c in state.cohorts],
        "cohorts": state.cohorts,
        "config": crm_cfg,
        "trial_config": trial_cfg,
    }


def run_simulation(
    crm_cfg: CRMConfig,
    trial_cfg: TrialConfig,
    true_tox: List[float],
    n_trials: int = 200,
    seed: int = 123,
) -> Dict[str, object]:
    """
    Run many simulated trials. Returns distribution of final recommended dose and some diagnostics.
    """
    rng = np.random.default_rng(seed)
    finals = []
    stuck_at_0 = 0

    # To ensure independent trials, vary CRM seed each time
    for i in range(n_trials):
        cfg_i = CRMConfig(**{**crm_cfg.__dict__})
        cfg_i.seed = int(rng.integers(1, 2_000_000_000))

        out = run_one_trial(cfg_i, trial_cfg, true_tox)
        finals.append(out["final_recommended_level"])
        if all(lvl == 0 for lvl in out["trajectory_levels"]):
            stuck_at_0 += 1

    finals = np.array(finals, dtype=int)
    counts = np.bincount(finals, minlength=len(crm_cfg.skeleton))

    return {
        "n_trials": n_trials,
        "final_counts": counts.tolist(),
        "final_probs": (counts / n_trials).tolist(),
        "stuck_all_cohorts_at_0": int(stuck_at_0),
    }


# -----------------------------
# Example usage
# -----------------------------

def main():
    # Example skeleton (monotone increasing)
    skeleton = [0.05, 0.08, 0.12, 0.18, 0.25, 0.33]

    # Example true toxicity curve for simulation
    true_tox = [0.03, 0.07, 0.12, 0.20, 0.28, 0.40]

    crm_cfg = CRMConfig(
        skeleton=skeleton,
        target=0.25,
        prior_mu=0.0,
        prior_sigma=1.0,
        max_step=1,
        use_overdose_control=True,
        overdose_cutoff=0.25,
        gh_n=61,
        startup_escalate_if_zero=True,
        startup_margin=0.90,
        use_tite=False,  # set True to use TITE approximation
        tite_assessment_window=1.0,
        tite_weight_power=1.0,
    )

    trial_cfg = TrialConfig(
        cohort_size=3,
        n_cohorts=10,
        start_level=0,
    )

    # Run one trial
    one = run_one_trial(crm_cfg, trial_cfg, true_tox)
    print("One trial final recommended level:", one["final_recommended_level"])
    print("Trajectory (levels):", one["trajectory_levels"])
    print("Trajectory (DLTs):  ", one["trajectory_dlts"])
    print("Final posterior mean p:", np.round(np.array(one["final_post_mean_p"]), 3))
    print("Final P(p > target):  ", np.round(np.array(one["final_post_prob_over_target"]), 3))

    # Run many trials
    sim = run_simulation(crm_cfg, trial_cfg, true_tox, n_trials=300, seed=2026)
    print("\nSimulation results")
    print("n_trials:", sim["n_trials"])
    print("Final probs by dose level:", np.round(np.array(sim["final_probs"]), 3))
    print("Stuck all cohorts at level 0:", sim["stuck_all_cohorts_at_0"])


if __name__ == "__main__":
    main()
