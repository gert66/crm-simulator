from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
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


if __name__ == "__main__":
    # Smoke test with synthetic true probabilities (no truth_model.py needed)
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    n = 2000
    mock_truth = SimpleNamespace(
        p_photon=rng.beta(2, 5, size=n),   # low baseline event rate
        p_proton=rng.beta(2, 6, size=n),   # slightly lower with protons
    )

    obs = add_noise(mock_truth, noise_sd=1.0, beta_z=0.5, seed=42)

    print(f"n                       : {n}")
    print(f"outcomes_photon  mean   : {obs.outcomes_photon.mean():.3f}")
    print(f"outcomes_proton  mean   : {obs.outcomes_proton.mean():.3f}")
    print(f"true p_photon    mean   : {mock_truth.p_photon.mean():.3f}")
    print(f"true p_proton    mean   : {mock_truth.p_proton.mean():.3f}")
    print(f"noise_sd                : {obs.noise_sd}")
    print(f"beta_z                  : {obs.beta_z}")

    # Observed rates should be in plausible range — not zero or one
    assert 0.0 < obs.outcomes_photon.mean() < 1.0
    assert 0.0 < obs.outcomes_proton.mean() < 1.0
    # Proton arm should on average have lower event rate
    assert obs.outcomes_proton.mean() < obs.outcomes_photon.mean(), (
        "Expected proton arm to have lower mean outcome"
    )
    print("\nAll assertions passed.")
