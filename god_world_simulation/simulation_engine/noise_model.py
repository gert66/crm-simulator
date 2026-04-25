from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .population import Population
    from .truth_model import TruthResult


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class ObservedData:
    outcomes_photon: np.ndarray  # binary 0/1
    outcomes_proton: np.ndarray  # binary 0/1 (counterfactual, same patient)
    noise_sd: float
    beta_z: float


def add_noise(
    truth: TruthResult,
    noise_sd: float = 1.0,
    beta_z: float = 0.5,
    seed: int = 42,
) -> ObservedData:
    """Convert true response probabilities into noisy binary outcomes.

    A shared unobserved confounder Z (e.g. comorbidity burden) shifts both
    arms identically in logit space; independent per-plan noise captures
    biological heterogeneity not explained by the treatment plan.

    Parameters
    ----------
    truth    : TruthResult holding p_photon and p_proton arrays
    noise_sd : std-dev of the per-plan independent logit noise
    beta_z   : loading of the shared confounder Z ~ N(0,1) on logit scale
    seed     : RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    n = len(truth.p_photon)

    # Shared unobserved confounder — same value for both plans per patient.
    # Drawn first so its position in the RNG stream is stable.
    Z = rng.standard_normal(n)

    # Independent biological noise per plan
    eps_photon = rng.normal(0.0, noise_sd, size=n)
    eps_proton = rng.normal(0.0, noise_sd, size=n)

    logit_photon = logit(truth.p_photon) + eps_photon + beta_z * Z
    logit_proton = logit(truth.p_proton) + eps_proton + beta_z * Z

    # Bernoulli draws — separate uniform streams for each arm
    outcomes_photon = (rng.random(n) < sigmoid(logit_photon)).astype(np.int8)
    outcomes_proton = (rng.random(n) < sigmoid(logit_proton)).astype(np.int8)

    return ObservedData(
        outcomes_photon=outcomes_photon,
        outcomes_proton=outcomes_proton,
        noise_sd=noise_sd,
        beta_z=beta_z,
    )


def calibrate_noise(
    pop: Population,
    truth: TruthResult,
    target_auc: float = 0.64,
    beta_z: float = 0.5,
    seed: int = 42,
    tol: float = 0.005,
    max_iter: int = 40,
) -> float:
    """Binary-search for noise_sd so that a logistic model achieves target_auc.

    The logistic regression is fitted on [sqrt(GTV), sqrt(MHD_photon)] against
    outcomes_photon. AUC decreases monotonically as noise_sd increases, so a
    simple bisection is exact and guaranteed to converge.

    Returns the calibrated noise_sd; prints final noise_sd and AUC.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = np.column_stack([np.sqrt(pop.gtv), np.sqrt(pop.mhd_photon)])

    lo, hi = 0.01, 5.0
    noise_sd = (lo + hi) / 2.0
    auc = float("nan")

    for _ in range(max_iter):
        noise_sd = (lo + hi) / 2.0
        obs = add_noise(truth, noise_sd=noise_sd, beta_z=beta_z, seed=seed)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, obs.outcomes_photon)
        auc = roc_auc_score(obs.outcomes_photon, clf.predict_proba(X)[:, 1])

        if abs(auc - target_auc) < tol:
            break

        # More noise → lower AUC; adjust bounds accordingly
        if auc > target_auc:
            lo = noise_sd
        else:
            hi = noise_sd

    print(f"calibrate_noise: noise_sd={noise_sd:.4f}, AUC={auc:.4f} (target={target_auc})")
    return noise_sd


if __name__ == "__main__":
    from types import SimpleNamespace
    from god_world_simulation.simulation_engine.population import generate_population

    n = 2000
    pop = generate_population(n=n, seed=0)

    # True probabilities driven by covariates so calibrate_noise has signal to work with.
    # logit(p) = intercept + coeff * sqrt(GTV) + coeff * sqrt(MHD)
    logit_true = -2.0 + 0.25 * np.sqrt(pop.gtv) + 0.15 * np.sqrt(pop.mhd_photon)
    mock_truth = SimpleNamespace(
        p_photon=sigmoid(logit_true),
        p_proton=sigmoid(logit_true - 0.5),  # protons reduce risk
    )

    # --- add_noise smoke test --------------------------------------------
    obs = add_noise(mock_truth, noise_sd=1.0, beta_z=0.5, seed=42)

    print(f"n                       : {n}")
    print(f"outcomes_photon  mean   : {obs.outcomes_photon.mean():.3f}")
    print(f"outcomes_proton  mean   : {obs.outcomes_proton.mean():.3f}")
    print(f"true p_photon    mean   : {mock_truth.p_photon.mean():.3f}")
    print(f"true p_proton    mean   : {mock_truth.p_proton.mean():.3f}")
    print(f"noise_sd                : {obs.noise_sd}")
    print(f"beta_z                  : {obs.beta_z}")

    assert 0.0 < obs.outcomes_photon.mean() < 1.0
    assert 0.0 < obs.outcomes_proton.mean() < 1.0
    assert obs.outcomes_proton.mean() < obs.outcomes_photon.mean(), (
        "Expected proton arm to have lower mean outcome"
    )

    # --- calibrate_noise smoke test ---------------------------------------
    print("\n--- calibrate_noise ---")
    target = 0.64
    found_sd = calibrate_noise(pop, mock_truth, target_auc=target, seed=42)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = np.column_stack([np.sqrt(pop.gtv), np.sqrt(pop.mhd_photon)])
    final_obs = add_noise(mock_truth, noise_sd=found_sd, beta_z=0.5, seed=42)
    clf = LogisticRegression(max_iter=1000).fit(X, final_obs.outcomes_photon)
    final_auc = roc_auc_score(final_obs.outcomes_photon, clf.predict_proba(X)[:, 1])
    assert abs(final_auc - target) < 0.02, f"AUC {final_auc:.4f} too far from target {target}"

    print("\nAll assertions passed.")
