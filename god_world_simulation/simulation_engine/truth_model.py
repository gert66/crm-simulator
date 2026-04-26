from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .population import Population

# Van Loon et al. 2026 published logistic coefficients
_INTERCEPT  = -1.3409
_BETA_GTV   =  0.0590   # per unit √cc
_BETA_MHD   =  0.2635   # per unit √Gy


@dataclass
class TruthResult:
    p_photon: np.ndarray      # true P(event | photon plan), shape (n,)
    p_proton: np.ndarray      # true P(event | proton plan), shape (n,)
    is_sensitive: np.ndarray  # latent flag copied from Population


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _van_loon(gtv: np.ndarray, mhd: np.ndarray) -> np.ndarray:
    """Published Van Loon formula applied to arbitrary GTV and MHD arrays."""
    return _logistic(_INTERCEPT + _BETA_GTV * np.sqrt(gtv) + _BETA_MHD * np.sqrt(mhd))


def _find_nonsensitive_intercept(
    gtv: np.ndarray,
    target: float,
    tol: float = 1e-7,
    max_iter: int = 80,
) -> float:
    """Bisect for α such that mean(logistic(α + β_GTV·√GTV)) == target.

    Non-sensitive patients have β_MHD = 0 (MHD does not affect their risk).
    Their intercept is calibrated so their mean mortality equals the same
    target as the sensitive subgroup, preserving the marginal rate at any π.
    """
    lo, hi = -15.0, 15.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        val = float(_logistic(mid + _BETA_GTV * np.sqrt(gtv)).mean())
        if abs(val - target) < tol:
            return mid
        if val < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def compute_truth(pop: Population, truth_mode: str = "god_world") -> TruthResult:
    """Compute god-world true event probabilities for both treatment plans.

    Both modes use the Van Loon 2026 logistic formula as the reference.
    The target marginal mortality is the population mean under that formula,
    and is preserved at any value of π (sensitive fraction).

    Parameters
    ----------
    pop        : generated cohort (provides MHD arrays and is_sensitive)
    truth_mode : "god_world" — calibrated pi-stratification (only sensitive
                               patients have MHD-dependent risk);
                 "published" — Van Loon formula applied to all patients for
                               the photon arm; pi stratifies only proton benefit.
    """
    if truth_mode == "god_world":
        return _god_world(pop)
    if truth_mode == "published":
        return _published(pop)
    raise ValueError(f"Unknown truth_mode: {truth_mode!r}")


def _god_world(pop: Population) -> TruthResult:
    """Van Loon formula for sensitive patients; MHD-free formula for non-sensitive.

    Step 1 — target mortality: mean of the published formula across all patients.
    Step 2 — calibrate non-sensitive intercept via bisection so their subgroup
             mean also equals target_mortality (β_MHD = 0 for non-sensitives).
    Step 3 — mix subgroups according to is_sensitive.

    Result: pi × E[p_sens] + (1-pi) × E[p_nonsens] = target_mortality, always.
    """
    target_mortality = float(_van_loon(pop.gtv, pop.mhd_photon).mean())
    intercept_ns     = _find_nonsensitive_intercept(pop.gtv, target_mortality)

    # Sensitive: full Van Loon (MHD-dependent risk, proton benefit is real)
    p_sens_photon = _van_loon(pop.gtv, pop.mhd_photon)
    p_sens_proton = _van_loon(pop.gtv, pop.mhd_proton)

    # Non-sensitive: GTV only, β_MHD = 0 → same value for both plans
    p_ns = _logistic(intercept_ns + _BETA_GTV * np.sqrt(pop.gtv))

    p_photon = np.where(pop.is_sensitive, p_sens_photon, p_ns)
    p_proton = np.where(pop.is_sensitive, p_sens_proton, p_ns)

    return TruthResult(p_photon=p_photon, p_proton=p_proton, is_sensitive=pop.is_sensitive)


def _published(pop: Population) -> TruthResult:
    """Van Loon formula for all patients in the photon arm.

    Non-sensitive patients use the same photon probability for protons too
    (no MHD benefit), consistent with the god-world π rule.  The photon
    marginal rate is always target_mortality; π only shifts proton benefit.
    """
    p_photon          = _van_loon(pop.gtv, pop.mhd_photon)
    p_proton_sensitive = _van_loon(pop.gtv, pop.mhd_proton)

    # Non-sensitive: proton delivers no benefit → p_proton = p_photon
    p_proton = np.where(pop.is_sensitive, p_proton_sensitive, p_photon)

    return TruthResult(p_photon=p_photon, p_proton=p_proton, is_sensitive=pop.is_sensitive)


if __name__ == "__main__":
    from god_world_simulation.simulation_engine.population import generate_population

    rng_seeds = [42, 123, 999]
    for seed in rng_seeds:
        pop = generate_population(n=2000, seed=seed)

        target_mortality = float(_van_loon(pop.gtv, pop.mhd_photon).mean())
        intercept_ns     = _find_nonsensitive_intercept(pop.gtv, target_mortality)

        print(f"\n── seed={seed} ──")
        print(f"  target_mortality       : {target_mortality:.4f}")
        print(f"  intercept_nonsensitive : {intercept_ns:.4f}")

        for mode in ("god_world", "published"):
            t     = compute_truth(pop, truth_mode=mode)
            delta = t.p_photon - t.p_proton

            mean_p = float(t.p_photon.mean())
            print(f"\n  [{mode}]")
            print(f"    mean(p_photon)        : {mean_p:.4f}  (target {target_mortality:.4f})")
            print(f"    mean(p_proton)        : {t.p_proton.mean():.4f}")
            print(f"    mean(true_delta)      : {delta.mean():.4f}")
            print(f"    mean(delta|sensitive) : {delta[t.is_sensitive].mean():.4f}")

            # Non-sensitive must have zero true delta
            assert (delta[~t.is_sensitive] == 0).all(), \
                f"[{mode}] Non-sensitive patients must have true delta = 0"

            # God-world must preserve marginal mortality within 0.1%
            if mode == "god_world":
                assert abs(mean_p - target_mortality) < 0.001, (
                    f"[god_world] mean(p_photon)={mean_p:.4f} deviates from "
                    f"target={target_mortality:.4f}"
                )

    print("\nAll assertions passed.")
