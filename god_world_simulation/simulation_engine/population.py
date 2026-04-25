from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class Population:
    n: int
    gtv: np.ndarray          # gross tumour volume, cc
    mhd_photon: np.ndarray   # mean heart dose with photons, Gy
    mhd_proton: np.ndarray   # mean heart dose with protons, Gy
    is_sensitive: np.ndarray  # latent sensitivity flag (bool); hidden from the model


def _truncated_lognormal(
    mu_log: float,
    sigma_log: float,
    low: float,
    high: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n samples from LogNormal(mu_log, sigma_log) truncated to [low, high].

    Uses rejection sampling with batches of 3n to minimise Python-loop iterations.
    """
    out: list[float] = []
    while len(out) < n:
        batch = rng.lognormal(mu_log, sigma_log, n * 3)
        batch = batch[(batch >= low) & (batch <= high)]
        out.extend(batch.tolist())
    return np.array(out[:n])


def generate_population(
    n: int = 1000,
    pi_sensitive: float = 0.7,
    proton_mode: Literal["percentage", "absolute", "proportional"] = "percentage",
    proton_reduction_pct: float = 0.50,
    proton_reduction_gy: float = 5.0,
    dist: Literal["clinical", "normal"] = "clinical",
    seed: int = 42,
) -> Population:
    """Generate a synthetic lung cancer cohort.

    Parameters
    ----------
    n               : cohort size
    pi_sensitive    : prevalence of latent cardiac-sensitive patients
    proton_mode     : how MHD is reduced with protons
                      "percentage"   – mhd_proton = mhd_photon * (1 - reduction_pct)
                      "absolute"     – mhd_proton = max(0, mhd_photon - reduction_gy)
                      "proportional" – percentage reduction + per-patient N(0, 0.15*mhd) noise
    proton_reduction_pct : fractional MHD reduction (percentage / proportional modes)
    proton_reduction_gy  : absolute Gy reduction (absolute mode)
    dist            : "clinical" uses truncated lognormal calibrated to Van Loon 2026 Table 1;
                      "normal" uses clipped Gaussian (quick sanity checks only)
    seed            : RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # --- covariate sampling ------------------------------------------------
    if dist == "clinical":
        # GTV:  median 70 cc,  range 0–1800 cc   (sigma_log chosen to give realistic spread)
        gtv = _truncated_lognormal(np.log(70.0), 1.2, low=1e-9, high=1800.0, n=n, rng=rng)
        # MHD:  median 11 Gy,  range 0.1–45 Gy
        mhd_photon = _truncated_lognormal(np.log(11.0), 0.8, low=0.1, high=45.0, n=n, rng=rng)
    elif dist == "normal":
        gtv = np.clip(rng.normal(70.0, 30.0, size=n), 0.0, 1800.0)
        mhd_photon = np.clip(rng.normal(11.0, 5.0, size=n), 0.1, 45.0)
    else:
        raise ValueError(f"Unknown dist: {dist!r}")

    # --- proton MHD --------------------------------------------------------
    if proton_mode == "percentage":
        mhd_proton = mhd_photon * (1.0 - proton_reduction_pct)
    elif proton_mode == "absolute":
        mhd_proton = np.maximum(0.0, mhd_photon - proton_reduction_gy)
    elif proton_mode == "proportional":
        noise = rng.normal(0.0, 0.15 * mhd_photon)
        mhd_proton = np.maximum(0.0, mhd_photon * (1.0 - proton_reduction_pct) + noise)
    else:
        raise ValueError(f"Unknown proton_mode: {proton_mode!r}")

    # --- latent sensitivity (hidden ground truth) --------------------------
    is_sensitive = rng.random(n) < pi_sensitive

    return Population(
        n=n,
        gtv=gtv,
        mhd_photon=mhd_photon,
        mhd_proton=mhd_proton,
        is_sensitive=is_sensitive,
    )


if __name__ == "__main__":
    pop = generate_population()

    median_gtv = float(np.median(pop.gtv))
    median_mhd = float(np.median(pop.mhd_photon))
    median_mhd_proton = float(np.median(pop.mhd_proton))
    pct_sensitive = float(pop.is_sensitive.mean()) * 100.0

    print(f"n                 : {pop.n}")
    print(f"GTV  median       : {median_gtv:.1f} cc   (target 70 cc)")
    print(f"GTV  range        : [{pop.gtv.min():.1f}, {pop.gtv.max():.1f}] cc")
    print(f"MHD  median       : {median_mhd:.1f} Gy  (target 11 Gy)")
    print(f"MHD  range        : [{pop.mhd_photon.min():.2f}, {pop.mhd_photon.max():.2f}] Gy")
    print(f"MHD proton median : {median_mhd_proton:.1f} Gy  (50% reduction)")
    print(f"Sensitive         : {pct_sensitive:.1f}%  (target 70%)")

    # within 20 % of targets
    assert 56.0 <= median_gtv <= 84.0, f"GTV median {median_gtv:.1f} outside ±20% of 70"
    assert 8.8 <= median_mhd <= 13.2, f"MHD median {median_mhd:.1f} outside ±20% of 11"
    print("\nAll assertions passed.")
