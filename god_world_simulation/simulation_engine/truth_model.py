from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .population import Population


@dataclass
class TruthResult:
    p_photon: np.ndarray      # true P(event | photon plan), shape (n,)
    p_proton: np.ndarray      # true P(event | proton plan), shape (n,)
    is_sensitive: np.ndarray  # latent flag copied from Population


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def compute_truth(pop: Population, truth_mode: str = "god_world") -> TruthResult:
    """Compute god-world true event probabilities for both treatment plans.

    Parameters
    ----------
    pop        : generated cohort (provides MHD arrays and is_sensitive)
    truth_mode : "god_world" uses a logistic dose-response with latent sensitivity;
                 "published" uses the Darby 2013 linear-exponential cardiac model.
    """
    if truth_mode == "god_world":
        return _god_world(pop)
    if truth_mode == "published":
        return _published(pop)
    raise ValueError(f"Unknown truth_mode: {truth_mode!r}")


def _god_world(pop: Population) -> TruthResult:
    """Logistic dose-response; only sensitive patients benefit from MHD reduction.

    Calibration: α = -2.5 → ~7.6% baseline at MHD=0;
                 β = 0.06 Gy⁻¹ → ~14% at median MHD=11 Gy (sensitive patients).
    Non-sensitive patients carry flat baseline risk regardless of plan.
    """
    alpha, beta = -2.5, 0.06

    p_sens_photon = _sigmoid(alpha + beta * pop.mhd_photon)
    p_sens_proton = _sigmoid(alpha + beta * pop.mhd_proton)
    p_baseline = float(_sigmoid(np.array(alpha))) * np.ones(pop.n)

    p_photon = np.where(pop.is_sensitive, p_sens_photon, p_baseline)
    p_proton = np.where(pop.is_sensitive, p_sens_proton, p_baseline)

    return TruthResult(p_photon=p_photon, p_proton=p_proton, is_sensitive=pop.is_sensitive)


def _published(pop: Population) -> TruthResult:
    """Darby et al. 2013 linear-exponential cardiac model.

    Every additional Gy of MHD raises relative cardiac risk by 7.4%.
    Baseline is calibrated so the expected rate at median MHD (11 Gy) is 12%.
    Non-sensitive patients: proton MHD reduction confers no benefit (god-world rule).
    """
    rr_per_gy = 0.074
    baseline = 0.12 * np.exp(-rr_per_gy * 11.0)

    p_all_photon = np.clip(baseline * np.exp(rr_per_gy * pop.mhd_photon), 1e-4, 0.999)
    p_all_proton = np.clip(baseline * np.exp(rr_per_gy * pop.mhd_proton), 1e-4, 0.999)

    # Non-sensitive: same risk with photons and protons
    p_photon = p_all_photon
    p_proton = np.where(pop.is_sensitive, p_all_proton, p_all_photon)

    return TruthResult(p_photon=p_photon, p_proton=p_proton, is_sensitive=pop.is_sensitive)


if __name__ == "__main__":
    from god_world_simulation.simulation_engine.population import generate_population

    pop = generate_population(n=1000, seed=42)

    for mode in ("god_world", "published"):
        t = compute_truth(pop, truth_mode=mode)
        delta = t.p_photon - t.p_proton
        print(f"\n[{mode}]")
        print(f"  p_photon  : mean={t.p_photon.mean():.3f}  range=[{t.p_photon.min():.3f}, {t.p_photon.max():.3f}]")
        print(f"  p_proton  : mean={t.p_proton.mean():.3f}")
        print(f"  true_delta: mean={delta.mean():.4f}  (sensitive only: {delta[t.is_sensitive].mean():.4f})")
        assert (delta[~t.is_sensitive] == 0).all(), "Non-sensitive patients must have zero true delta"

    print("\nAll assertions passed.")
