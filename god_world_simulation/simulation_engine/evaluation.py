from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .truth_model import TruthResult
    from .fitting import FittedModel


@dataclass
class EvaluationResult:
    n_selected: int
    selection_rate: float
    predicted_nnt: float     # 1 / mean(predicted_delta[selected])
    true_nnt: float          # 1 / mean(true_delta[selected])
    naive_nnt: float         # 1 / mean(predicted_delta) across ALL patients
    nnt_inflation: float     # true_nnt / predicted_nnt
    fraction_true_benefit: float  # fraction of selected where true_delta > 0
    bins_df: pd.DataFrame    # per-bin summary for selected patients


def _safe_nnt(mean_delta: float) -> float:
    return 1.0 / mean_delta if mean_delta > 0 else float("inf")


def evaluate(
    truth: TruthResult,
    fitted: FittedModel,
    selection_threshold: float = 0.02,
) -> EvaluationResult:
    """Compare predicted NNT vs. god-world true NNT for the selected cohort.

    In the god-world, non-sensitive patients have zero treatment benefit
    regardless of how large their predicted_delta is, so true_delta is
    masked to 0 for non-sensitive patients before any NNT calculation.
    """
    # God-world true benefit: only sensitive patients benefit
    true_delta = (truth.p_photon - truth.p_proton) * truth.is_sensitive.astype(float)

    # Patient selection based on predicted benefit
    selected = fitted.predicted_delta >= selection_threshold
    n_selected = int(selected.sum())
    selection_rate = n_selected / len(selected)

    pred_sel = fitted.predicted_delta[selected]
    true_sel = true_delta[selected]

    predicted_nnt = _safe_nnt(float(pred_sel.mean())) if n_selected > 0 else float("inf")
    true_nnt = _safe_nnt(float(true_sel.mean())) if n_selected > 0 else float("inf")
    naive_nnt = _safe_nnt(float(fitted.predicted_delta.mean()))
    nnt_inflation = true_nnt / predicted_nnt if predicted_nnt not in (0.0, float("inf")) else float("inf")
    fraction_true_benefit = float((true_sel > 0).mean()) if n_selected > 0 else 0.0

    # --- bin analysis (selected patients only) ----------------------------
    bins_df = _build_bins_df(pred_sel, true_sel)

    return EvaluationResult(
        n_selected=n_selected,
        selection_rate=selection_rate,
        predicted_nnt=predicted_nnt,
        true_nnt=true_nnt,
        naive_nnt=naive_nnt,
        nnt_inflation=nnt_inflation,
        fraction_true_benefit=fraction_true_benefit,
        bins_df=bins_df,
    )


def _build_bins_df(pred_sel: np.ndarray, true_sel: np.ndarray) -> pd.DataFrame:
    """Build a 5-equal-width-bin summary table for selected patients."""
    labels = pd.cut(pred_sel, bins=5)
    df = pd.DataFrame({"predicted_delta_bin": labels, "pred": pred_sel, "true": true_sel})

    agg = (
        df.groupby("predicted_delta_bin", observed=True)
        .agg(
            mean_predicted_delta=("pred", "mean"),
            mean_true_delta=("true", "mean"),
            n_patients=("pred", "count"),
        )
        .reset_index()
    )

    agg["predicted_nnt"] = agg["mean_predicted_delta"].apply(_safe_nnt)
    agg["true_nnt"] = agg["mean_true_delta"].apply(_safe_nnt)

    return agg[[
        "predicted_delta_bin",
        "mean_predicted_delta",
        "mean_true_delta",
        "predicted_nnt",
        "true_nnt",
        "n_patients",
    ]]


def print_summary(result: EvaluationResult) -> None:
    """Print a clean scalar summary, highlighting NNT inflation."""
    sep = "=" * 54
    print(sep)
    print("  Evaluation Summary")
    print(sep)
    print(f"  n_selected             : {result.n_selected}")
    print(f"  selection_rate         : {result.selection_rate:.1%}")
    print(f"  naive_nnt  (all pts)   : {result.naive_nnt:.1f}")
    print(f"  predicted_nnt          : {result.predicted_nnt:.1f}")
    print(f"  true_nnt               : {result.true_nnt:.1f}")
    print(f"  *** nnt_inflation      : {result.nnt_inflation:.2f}x ***")
    print(f"  fraction_true_benefit  : {result.fraction_true_benefit:.1%}")
    print()
    print("  Bins (selected patients):")
    formatted = result.bins_df.copy()
    for col in ("mean_predicted_delta", "mean_true_delta"):
        formatted[col] = formatted[col].map("{:.4f}".format)
    for col in ("predicted_nnt", "true_nnt"):
        formatted[col] = formatted[col].apply(
            lambda x: f"{x:.1f}" if x != float("inf") else "inf"
        )
    print(formatted.to_string(index=False))
    print(sep)


if __name__ == "__main__":
    from types import SimpleNamespace
    import numpy as np
    from god_world_simulation.simulation_engine.population import generate_population
    from god_world_simulation.simulation_engine.noise_model import add_noise, sigmoid
    from god_world_simulation.simulation_engine.fitting import fit_model

    n = 2000
    pop = generate_population(n=n, seed=42)

    logit_true = -2.0 + 0.25 * np.sqrt(pop.gtv) + 0.15 * np.sqrt(pop.mhd_photon)
    p_photon = sigmoid(logit_true)
    p_proton = sigmoid(logit_true - 0.5)

    mock_truth = SimpleNamespace(
        p_photon=p_photon,
        p_proton=p_proton,
        is_sensitive=pop.is_sensitive,
    )

    obs = add_noise(mock_truth, noise_sd=1.5, beta_z=0.5, seed=42)
    fitted = fit_model(pop, obs)
    result = evaluate(mock_truth, fitted, selection_threshold=0.02)
    print_summary(result)

    assert result.n_selected > 0, "No patients selected"
    assert result.predicted_nnt > 0, "predicted_nnt must be positive"
    assert result.true_nnt > 0, "true_nnt must be positive"
    assert result.nnt_inflation > 0, "nnt_inflation must be positive"
    assert 0.0 <= result.fraction_true_benefit <= 1.0
    assert len(result.bins_df) <= 5
    assert list(result.bins_df.columns) == [
        "predicted_delta_bin", "mean_predicted_delta", "mean_true_delta",
        "predicted_nnt", "true_nnt", "n_patients",
    ]
    print("\nAll assertions passed.")
