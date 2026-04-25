from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .population import Population
from .noise_model import ObservedData


@dataclass
class FittedModel:
    coef_intercept: float
    coef_gtv: float
    coef_mhd: float
    auc: float
    predicted_photon: np.ndarray  # P(death) under photon plan
    predicted_proton: np.ndarray  # P(death) under proton plan (same coefficients, swapped MHD)
    predicted_delta: np.ndarray   # predicted_photon - predicted_proton


def fit_model(pop: Population, observed: ObservedData) -> FittedModel:
    """Fit a logistic regression on observed photon outcomes and apply it to both plans.

    Features are sqrt-transformed to reduce right-skew in GTV and MHD.
    Proton predictions reuse the photon-fitted coefficients with mhd_proton
    substituted, so predicted_delta reflects only the dosimetric difference.
    """
    X_photon = np.column_stack([np.sqrt(pop.gtv), np.sqrt(pop.mhd_photon)])
    X_proton = np.column_stack([np.sqrt(pop.gtv), np.sqrt(pop.mhd_proton)])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_photon, observed.outcomes_photon)

    predicted_photon = clf.predict_proba(X_photon)[:, 1]
    predicted_proton = clf.predict_proba(X_proton)[:, 1]

    auc = roc_auc_score(observed.outcomes_photon, predicted_photon)

    print(
        f"fit_model: intercept={clf.intercept_[0]:.4f}  "
        f"coef_gtv={clf.coef_[0][0]:.4f}  "
        f"coef_mhd={clf.coef_[0][1]:.4f}  "
        f"AUC={auc:.4f}"
    )

    return FittedModel(
        coef_intercept=float(clf.intercept_[0]),
        coef_gtv=float(clf.coef_[0][0]),
        coef_mhd=float(clf.coef_[0][1]),
        auc=auc,
        predicted_photon=predicted_photon,
        predicted_proton=predicted_proton,
        predicted_delta=predicted_photon - predicted_proton,
    )


if __name__ == "__main__":
    from types import SimpleNamespace
    from god_world_simulation.simulation_engine.population import generate_population
    from god_world_simulation.simulation_engine.noise_model import add_noise, sigmoid

    n = 1000
    pop = generate_population(n=n, seed=42)

    logit_true = -2.0 + 0.25 * np.sqrt(pop.gtv) + 0.15 * np.sqrt(pop.mhd_photon)
    mock_truth = SimpleNamespace(
        p_photon=sigmoid(logit_true),
        p_proton=sigmoid(logit_true - 0.5),
    )

    obs = add_noise(mock_truth, noise_sd=1.0, beta_z=0.5, seed=42)
    fitted = fit_model(pop, obs)

    print(f"\npredicted_photon  mean : {fitted.predicted_photon.mean():.3f}")
    print(f"predicted_proton  mean : {fitted.predicted_proton.mean():.3f}")
    print(f"predicted_delta   mean : {fitted.predicted_delta.mean():.4f}")

    assert fitted.coef_mhd > 0, "MHD coefficient should be positive (higher MHD → higher risk)"
    assert fitted.predicted_delta.mean() > 0, "Photon risk should exceed proton risk on average"
    assert 0.5 < fitted.auc < 1.0, f"AUC {fitted.auc:.4f} out of expected range"
    print("\nAll assertions passed.")
