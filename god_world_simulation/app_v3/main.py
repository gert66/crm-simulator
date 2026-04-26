"""God-World NNT Simulator — reactive Streamlit app (no Run button)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="God-World NNT Simulator", layout="wide")

# ── Guarded imports (sidebar still renders if these fail) ─────────────────────
try:
    from god_world_simulation.simulation_engine.population import generate_population
    from god_world_simulation.simulation_engine.truth_model import compute_truth
    from god_world_simulation.simulation_engine.noise_model import add_noise, calibrate_noise
    from god_world_simulation.simulation_engine.fitting import fit_model
    from god_world_simulation.simulation_engine.evaluation import evaluate
    IMPORTS_OK = True
except Exception as e:
    IMPORTS_OK = False
    import_error = str(e)


# ── Chart builders ────────────────────────────────────────────────────────────

def _fig_gtv(pop) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=pop.gtv, nbinsx=40, marker_color="#4C78A8", opacity=0.85,
    ))
    fig.update_layout(
        height=220, margin=dict(t=30, b=30, l=40, r=10),
        title_text="GTV distribution", xaxis_title="cc", yaxis_title="n",
    )
    return fig


def _fig_true_delta(truth) -> go.Figure:
    raw_delta = truth.p_photon - truth.p_proton
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=raw_delta[truth.is_sensitive], name="Sensitive",
        nbinsx=40, marker_color="#E45756", opacity=0.8,
    ))
    fig.add_trace(go.Histogram(
        x=raw_delta[~truth.is_sensitive], name="Non-sensitive (Δ=0)",
        nbinsx=5, marker_color="#4C78A8", opacity=0.6,
    ))
    fig.update_layout(
        barmode="overlay", height=220, margin=dict(t=30, b=30, l=40, r=10),
        title_text="True Δ by sensitivity", xaxis_title="True Δ", yaxis_title="n",
        legend=dict(orientation="h", y=1.15, font_size=11),
    )
    return fig


def _fig_nnt_bins(result) -> go.Figure:
    df = result.bins_df
    labels = [str(b) for b in df["predicted_delta_bin"]]
    fig = go.Figure([
        go.Bar(name="Predicted NNT", x=labels, y=df["predicted_nnt"],
               marker_color="#4C78A8"),
        go.Bar(name="True NNT",      x=labels, y=df["true_nnt"],
               marker_color="#F58518"),
    ])
    fig.update_layout(
        barmode="group",
        title_text="Predicted vs True NNT by benefit bin (selected patients)",
        xaxis_title="Predicted Δ bin", yaxis_title="NNT",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def _fig_delta_scatter(truth, fitted) -> go.Figure:
    true_delta = (truth.p_photon - truth.p_proton) * truth.is_sensitive.astype(float)
    fig = go.Figure()
    for flag, color, name in [
        (True,  "#E45756", "Sensitive"),
        (False, "#4C78A8", "Non-sensitive"),
    ]:
        mask = truth.is_sensitive == flag
        fig.add_trace(go.Scatter(
            x=fitted.predicted_delta[mask], y=true_delta[mask],
            mode="markers", name=name,
            marker=dict(color=color, size=3, opacity=0.45),
        ))
    lo, hi = float(fitted.predicted_delta.min()), float(fitted.predicted_delta.max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
        line=dict(color="black", dash="dash", width=1.5),
    ))
    fig.update_layout(
        title_text="Predicted Δ vs True Δ (god-world)",
        xaxis_title="Predicted Δ (model)", yaxis_title="True Δ (god-world)",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def _fig_mhd(pop) -> go.Figure:
    fig = go.Figure([
        go.Histogram(x=pop.mhd_photon, nbinsx=40, name="Photon MHD",
                     marker_color="#4C78A8", opacity=0.7),
        go.Histogram(x=pop.mhd_proton,  nbinsx=40, name="Proton MHD",
                     marker_color="#F58518", opacity=0.7),
    ])
    fig.update_layout(
        barmode="overlay", title_text="MHD: Photon vs Proton",
        xaxis_title="Gy", yaxis_title="n",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def _fig_roc(fpr, tpr, auc: float, title: str, color: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}",
        line=dict(color=color, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="grey", dash="dash", width=1),
    ))
    fig.update_layout(
        title_text=title,
        xaxis_title="False positive rate", yaxis_title="True positive rate",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=40, b=60, l=40, r=10),
    )
    return fig


def _fig_calibration_qc(truth_p: np.ndarray, fitted_p: np.ndarray) -> go.Figure:
    _df = pd.DataFrame({"truth": truth_p, "fitted": fitted_p})
    _df["decile"] = pd.qcut(truth_p, 10, labels=range(1, 11))
    _grp = _df.groupby("decile")[["truth", "fitted"]].mean().reset_index()
    fig = go.Figure([
        go.Bar(name="God-world", x=_grp["decile"], y=_grp["truth"],
               marker_color="#4C78A8", opacity=0.8),
        go.Bar(name="Fitted model", x=_grp["decile"], y=_grp["fitted"],
               marker_color="#F58518", opacity=0.8),
    ])
    fig.add_hline(y=0.51, line_dash="dash", line_color="red",
                  annotation_text="Van Loon 51%", annotation_position="top right")
    fig.update_layout(
        barmode="group",
        title_text="Mean mortality by risk decile",
        xaxis_title="Risk decile (1 = lowest, 10 = highest)",
        yaxis_title="Mean mortality",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def _fig_formula_comparison(truth, fitted) -> go.Figure:
    fig = go.Figure()
    for flag, color, name in [
        (True,  "#4C78A8", "Sensitive"),
        (False, "#E45756", "Non-sensitive"),
    ]:
        mask = truth.is_sensitive == flag
        fig.add_trace(go.Scatter(
            x=truth.p_photon[mask], y=fitted.predicted_photon[mask],
            mode="markers", name=name,
            marker=dict(color=color, size=3, opacity=0.4),
        ))
    lo = min(float(truth.p_photon.min()), float(fitted.predicted_photon.min()))
    hi = max(float(truth.p_photon.max()), float(fitted.predicted_photon.max()))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect calibration",
        line=dict(color="black", dash="dash", width=1.5),
    ))
    fig.update_layout(
        title_text="True vs predicted mortality per patient",
        xaxis_title="True probability (God-world)",
        yaxis_title="Predicted probability (fitted model)",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    with st.expander("ℹ️ How this app works", expanded=False):
        st.markdown(
            "1. A virtual patient population is generated\n\n"
            "2. True survival probabilities are computed (God-world)\n\n"
            "3. Noise is added until fitted model achieves target AUC\n\n"
            "4. A logistic model is fitted on the noisy outcomes\n\n"
            "5. Patients are selected based on predicted benefit\n\n"
            "6. Predicted NNT is compared to true NNT\n\n"
            "The gap between predicted and true NNT is the "
            "**NNT inflation factor**."
        )

    with st.expander("1 · Population", expanded=True):
        n_patients = st.slider(
            "N patients", 200, 5000, 1000, step=100,
            help=(
                "Number of virtual lung cancer patients to simulate. "
                "Larger cohorts give more stable NNT estimates but run slower."
            ),
        )
        dist = st.selectbox(
            "Distribution", ["clinical", "normal"],
            help=(
                "Clinical: uses lognormal distributions calibrated to "
                "Van Loon et al. 2026 (median GTV 70cc, median MHD 11Gy). "
                "Normal: legacy symmetric distributions, less realistic."
            ),
        )
        seed = int(st.number_input(
            "Seed", value=42, step=1, min_value=0,
            help=(
                "Random seed for reproducibility. Change this to see "
                "how results vary across different simulated cohorts."
            ),
        ))

    with st.expander("2 · Truth model (God-world)", expanded=True):
        pi_sensitive = st.slider(
            "Fraction truly sensitive to MHD (π)", 0.0, 1.0, 0.7, step=0.05,
            help=(
                "Fraction of patients for whom heart dose truly affects "
                "survival (π). The remaining (1-π) patients have zero cardiac "
                "mortality pathway — protons give them no benefit regardless "
                "of MHD reduction. This is hidden from the fitted model."
            ),
        )
        truth_mode = st.selectbox(
            "Truth mode", ["god_world", "published"],
            help=(
                "God-world: applies pi_sensitive so only a fraction of "
                "patients truly benefit from MHD reduction. "
                "Published: uses the Van Loon formula identically for all "
                "patients, ignoring susceptibility heterogeneity."
            ),
        )
        proton_mode = st.selectbox(
            "Proton mode", ["percentage", "absolute", "proportional"],
            help=(
                "How proton MHD is derived from photon MHD per patient. "
                "Percentage: MHD_proton = MHD_photon × (1 - reduction%). "
                "Absolute: MHD_proton = MHD_photon - fixed Gy value. "
                "Proportional: like percentage but with patient-level random "
                "variation (±15% noise around the reduction)."
            ),
        )
        if proton_mode == "absolute":
            proton_reduction_gy = st.slider(
                "Proton reduction (Gy)", 1, 15, 5,
                help=(
                    "Absolute reduction in mean heart dose (Gy) achieved "
                    "by proton therapy. Used only when proton mode is Absolute."
                ),
            )
            proton_reduction_pct = 0.50
        else:
            proton_reduction_pct = st.slider(
                "Proton reduction (%)", 0.10, 0.90, 0.50, step=0.05,
                help=(
                    "Fractional reduction in mean heart dose achieved by "
                    "proton therapy. 0.50 means protons deliver 50% less heart "
                    "dose than photons. Typical clinical range is 30-70%."
                ),
            )
            proton_reduction_gy = 5

    with st.expander("3 · Noise / Calibration", expanded=True):
        calibrate_auc_flag = st.checkbox(
            "Auto-calibrate noise to target AUC", value=True,
            help=(
                "When checked, the simulation automatically finds the "
                "noise level that makes the fitted model achieve exactly the "
                "target AUC. This reproduces the real-world performance loss "
                "seen in Van Loon et al. (AUC 0.64)."
            ),
        )
        if calibrate_auc_flag:
            target_auc = st.slider(
                "Target AUC", 0.50, 0.80, 0.64, step=0.01,
                help=(
                    "The AUC the fitted model should achieve after noise "
                    "is added. Set to 0.64 to match Van Loon et al. 2026. "
                    "Lower values = more noise = more NNT inflation."
                ),
            )
            manual_noise_sd = 1.5
        else:
            manual_noise_sd = st.slider(
                "Manual noise SD", 0.1, 4.0, 1.5, step=0.1,
                help=(
                    "Standard deviation of the noise added to the logit "
                    "scale. Higher values corrupt the true probabilities more, "
                    "reducing model discrimination and inflating NNT."
                ),
            )
            target_auc = 0.64
        beta_z = st.slider(
            "Confounder loading (β_z)", 0.0, 2.0, 0.5, step=0.1,
            help=(
                "Strength of the unobserved confounder Z (e.g. "
                "comorbidity, smoking history, performance status). "
                "Z affects both photon and proton outcomes equally within "
                "a patient, simulating shared unmeasured risk factors."
            ),
        )

    with st.expander("4 · Selection", expanded=True):
        threshold = st.slider(
            "Selection threshold Δ (e.g. 0.02 = 2%)",
            0.01, 0.10, 0.02, step=0.005, format="%.3f",
            help=(
                "The minimum predicted mortality reduction (Δ) required "
                "to select a patient for proton therapy. The Dutch national "
                "guideline uses 2% (0.02). Patients with predicted "
                "Δ < threshold receive photons instead."
            ),
        )


# ── Title + import guard ──────────────────────────────────────────────────────
st.title("God-World NNT Simulator")
st.subheader("What does a 2% predicted benefit actually mean?")

if not IMPORTS_OK:
    st.error(f"Import failed: {import_error}")
    st.stop()


# ── Reactive pipeline with incremental caching ────────────────────────────────

# Stage 1 — Population (+ both truth models)
pop_key = (
    n_patients, dist, seed,
    float(pi_sensitive), proton_mode,
    float(proton_reduction_pct), int(proton_reduction_gy),
)
if st.session_state.get("pop_key") != pop_key:
    pop = generate_population(
        n=n_patients, pi_sensitive=float(pi_sensitive),
        proton_mode=proton_mode,
        proton_reduction_pct=float(proton_reduction_pct),
        proton_reduction_gy=float(proton_reduction_gy),
        dist=dist, seed=seed,
    )
    st.session_state["pop"]       = pop
    st.session_state["truth_gw"]  = compute_truth(pop, "god_world")
    st.session_state["truth_pub"] = compute_truth(pop, "published")
    st.session_state["pop_key"]   = pop_key

pop       = st.session_state["pop"]
truth_gw  = st.session_state["truth_gw"]
truth_pub = st.session_state["truth_pub"]
truth     = truth_gw if truth_mode == "god_world" else truth_pub

# Stage 1b — AUC before noise (depends on pop + truth mode, not on noise params)
truth_key = (pop_key, truth_mode)
if st.session_state.get("truth_key") != truth_key:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score as _roc_auc, roc_curve as _roc_curve
    _rng_gw = np.random.default_rng(seed)
    _y_gw   = _rng_gw.binomial(n=1, p=truth.p_photon, size=pop.n)
    _X_gw   = np.column_stack([np.sqrt(pop.gtv), np.sqrt(pop.mhd_photon)])
    _pr_gw  = LogisticRegression(max_iter=1000).fit(_X_gw, _y_gw).predict_proba(_X_gw)[:, 1]
    _fpr_gw, _tpr_gw, _ = _roc_curve(_y_gw, _pr_gw)
    st.session_state["auc_before"]   = float(_roc_auc(_y_gw, _pr_gw))
    st.session_state["fpr_gw"]       = _fpr_gw
    st.session_state["tpr_gw"]       = _tpr_gw
    st.session_state["truth_key"]    = truth_key

auc_before = st.session_state["auc_before"]

# Stage 2 — Noise calibration + model fit
noise_key = (
    pop_key, truth_mode,
    bool(calibrate_auc_flag), float(target_auc),
    float(manual_noise_sd), float(beta_z), seed,
)
if st.session_state.get("noise_key") != noise_key:
    if calibrate_auc_flag:
        noise_sd = calibrate_noise(
            pop, truth,
            target_auc=float(target_auc), beta_z=float(beta_z), seed=seed,
        )
    else:
        noise_sd = float(manual_noise_sd)
    obs    = add_noise(truth, noise_sd=noise_sd, beta_z=float(beta_z), seed=seed)
    fitted = fit_model(pop, obs)
    st.session_state["noise_sd"]  = noise_sd
    st.session_state["observed"]  = obs
    st.session_state["fitted"]    = fitted
    st.session_state["noise_key"] = noise_key

noise_sd = st.session_state["noise_sd"]
obs      = st.session_state["observed"]
fitted   = st.session_state["fitted"]

# Stage 3 — Evaluation
eval_key = (noise_key, float(threshold))
if st.session_state.get("eval_key") != eval_key:
    result = evaluate(truth, fitted, selection_threshold=float(threshold))
    st.session_state["result"]   = result
    st.session_state["eval_key"] = eval_key

result = st.session_state["result"]


# ── Section A: Population Summary ─────────────────────────────────────────────
with st.expander("A · Population Summary", expanded=True):
    left, right = st.columns(2)
    with left:
        st.metric("N patients",  n_patients)
        st.metric("Median GTV",  f"{np.median(pop.gtv):.1f} cc")
        st.metric("Range GTV",   f"{pop.gtv.min():.0f} – {pop.gtv.max():.0f} cc")
    with right:
        st.metric("Distribution",        dist)
        st.metric("Median MHD (photon)", f"{np.median(pop.mhd_photon):.1f} Gy")
        st.metric("Range MHD",           f"{pop.mhd_photon.min():.1f} – {pop.mhd_photon.max():.1f} Gy")
    st.plotly_chart(_fig_gtv(pop), use_container_width=True, key="chart_gtv_a")


# ── Section B: Truth Model Summary ───────────────────────────────────────────
with st.expander("B · Truth Model Summary", expanded=True):
    delta_gw  = (truth_gw.p_photon  - truth_gw.p_proton)  * truth_gw.is_sensitive.astype(float)
    delta_pub = (truth_pub.p_photon - truth_pub.p_proton) * truth_pub.is_sensitive.astype(float)
    delta_sel = delta_gw if truth_mode == "god_world" else delta_pub
    reduction_str = (
        f"{proton_reduction_gy} Gy" if proton_mode == "absolute"
        else f"{proton_reduction_pct:.0%}"
    )

    left, right = st.columns(2)
    with left:
        st.metric("Proton mode",            proton_mode)
        st.metric("Reduction amount",       reduction_str)
        st.metric("π (sensitive fraction)", f"{pi_sensitive:.0%}")
    with right:
        st.metric("Mean true Δ (God-world)", f"{delta_gw.mean():.4f}")
        st.metric("Mean true Δ (published)", f"{delta_pub.mean():.4f}")
        st.metric("% with Δ > 2%",          f"{(delta_sel > 0.02).mean():.1%}")
    st.plotly_chart(_fig_true_delta(truth), use_container_width=True, key="chart_true_delta")


# ── Section QC-1: Survival calibration ───────────────────────────────────────
with st.expander("🔍 QC: Survival calibration", expanded=True):
    _p_all_sensitive = 1.0 / (1.0 + np.exp(-np.clip(
        -1.3409 + 0.0590 * np.sqrt(pop.gtv) + 0.2635 * np.sqrt(pop.mhd_photon),
        -500, 500,
    )))
    mean_p_allsensitive = float(_p_all_sensitive.mean())
    mean_p_current      = float(truth.p_photon.mean())

    qc1, qc2, qc3 = st.columns(3)
    qc1.metric(
        "Van Loon et al. (published)", "51.0%",
        help="Observed 2-year mortality in Van Loon 2026, n=1094",
    )
    qc2.metric(
        "God-world (all sensitive, π=1.0)", f"{mean_p_allsensitive:.1%}",
        help=(
            "Mean predicted mortality using published formula applied to all "
            "patients. Should be close to 51%."
        ),
    )
    qc3.metric(
        "God-world (mixed, current π)", f"{mean_p_current:.1%}",
        help=(
            "Mean predicted mortality after applying pi_sensitive split with "
            "intercept recalibration. Should also be close to 51%."
        ),
    )
    st.plotly_chart(
        _fig_calibration_qc(truth.p_photon, fitted.predicted_photon),
        use_container_width=True, key="chart_calibration_qc",
    )


# ── Section QC-2: Formula comparison ─────────────────────────────────────────
with st.expander("🔍 QC: Formula comparison", expanded=True):
    st.table(pd.DataFrame({
        "Parameter":           ["Intercept", "β GTV (sqrt)", "β MHD (sqrt)"],
        "Published (Van Loon)": [-1.3409, 0.0590, 0.2635],
        "Fitted model":        [fitted.coef_intercept, fitted.coef_gtv, fitted.coef_mhd],
    }))
    st.plotly_chart(
        _fig_formula_comparison(truth, fitted),
        use_container_width=True, key="chart_formula_comparison",
    )
    st.caption(
        "Points above the diagonal = model overestimates mortality. "
        "Points below = underestimates. "
        "Red points (non-sensitive) should scatter randomly around the diagonal "
        "if the model is well calibrated."
    )


# ── Section C: Noise & Calibration ───────────────────────────────────────────
with st.expander("C · Noise & Calibration", expanded=True):
    auc_col1, auc_col2 = st.columns(2)
    auc_delta = fitted.auc - auc_before
    auc_col1.metric("AUC (God-world, no noise)", f"{auc_before:.3f}")
    auc_col2.metric(
        "AUC (fitted model)", f"{fitted.auc:.3f}",
        delta=f"{auc_delta:.3f}", delta_color="inverse",
    )

    left, right = st.columns(2)
    with left:
        st.metric("Target AUC", f"{target_auc:.2f}" if calibrate_auc_flag else "Manual")
        st.metric("Noise SD",   f"{noise_sd:.3f}")
    with right:
        st.metric("N selected (Δ ≥ threshold)", result.n_selected)
        st.metric("Selection rate",             f"{result.selection_rate:.1%}")

    # ROC curves
    from sklearn.metrics import roc_curve, roc_auc_score as _roc_auc_score

    # Left: god-world binary outcomes → logistic fit (cached in Stage 1b)
    _fpr_b    = st.session_state["fpr_gw"]
    _tpr_b    = st.session_state["tpr_gw"]
    _auc_roc_b = auc_before

    # Right: fitted model predictions vs noisy observed outcomes
    _fpr_a, _tpr_a, _ = roc_curve(obs.outcomes_photon, fitted.predicted_photon)
    _auc_roc_a = float(_roc_auc_score(obs.outcomes_photon, fitted.predicted_photon))

    roc_col1, roc_col2 = st.columns(2)
    roc_col1.plotly_chart(
        _fig_roc(_fpr_b, _tpr_b, _auc_roc_b,
                 f"ROC — God-world outcomes  AUC={_auc_roc_b:.3f}",
                 "#4C78A8"),
        use_container_width=True, key="chart_roc_before",
    )
    roc_col2.plotly_chart(
        _fig_roc(_fpr_a, _tpr_a, _auc_roc_a,
                 f"ROC — Fitted model on noisy data  AUC={_auc_roc_a:.3f}",
                 "#F58518"),
        use_container_width=True, key="chart_roc_after",
    )


# ── Section D: Final Results ──────────────────────────────────────────────────
with st.expander("D · Final Results", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted NNT", f"{result.predicted_nnt:.1f}")
    c2.metric("True NNT",      f"{result.true_nnt:.1f}")
    c3.metric("NNT Inflation", f"{result.nnt_inflation:.2f}×")

    infl      = result.nnt_inflation
    box_color = "#d73027" if infl >= 2.0 else "#f46d43"
    st.markdown(
        f'<div style="background:{box_color};padding:16px;border-radius:8px;'
        f'text-align:center;margin:8px 0 16px 0;">'
        f'<b style="color:white;font-size:1.6em;">Inflation {infl:.1f}×</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs([
        "NNT by bin",
        "Predicted vs True Δ",
        "Calibration scatter",
    ])
    with tab1:
        st.plotly_chart(_fig_nnt_bins(result), use_container_width=True, key="chart_nnt")
    with tab2:
        st.plotly_chart(_fig_delta_scatter(truth, fitted), use_container_width=True, key="chart_scatter")
    with tab3:
        col_gtv, col_mhd = st.columns(2)
        col_gtv.plotly_chart(_fig_gtv(pop), use_container_width=True, key="chart_gtv_cal")
        col_mhd.plotly_chart(_fig_mhd(pop), use_container_width=True, key="chart_mhd")


# ── Debug expander ────────────────────────────────────────────────────────────
_true_delta = truth.p_photon - truth.p_proton

with st.expander("🐛 Debug: Internal data inspection", expanded=False):

    st.subheader("Population (first 10 patients)")
    st.dataframe(pd.DataFrame({
        "GTV (cc)":         pop.gtv[:10],
        "MHD photon (Gy)":  pop.mhd_photon[:10],
        "MHD proton (Gy)":  pop.mhd_proton[:10],
        "Is sensitive":     pop.is_sensitive[:10],
    }))

    st.subheader("True probabilities (God-world)")
    st.dataframe(pd.DataFrame({
        "p_photon":     truth.p_photon[:10],
        "p_proton":     truth.p_proton[:10],
        "true_delta":   _true_delta[:10],
        "is_sensitive": truth.is_sensitive[:10],
    }))

    st.subheader("Sampled binary outcomes")
    st.dataframe(pd.DataFrame({
        "p_photon (true)":           truth.p_photon[:10],
        "outcome_photon (sampled)":  obs.outcomes_photon[:10],
        "p > 0.5":                   (truth.p_photon[:10] > 0.5).astype(int),
        "outcome matches p>0.5":     (
            obs.outcomes_photon[:10] == (truth.p_photon[:10] > 0.5).astype(int)
        ).astype(int),
    }))

    st.subheader("Fitted logistic regression coefficients")
    st.dataframe(pd.DataFrame({
        "Parameter":          ["Intercept", "β sqrt(GTV)", "β sqrt(MHD)"],
        "Published (Van Loon)": [-1.3409, 0.0590, 0.2635],
        "Fitted model":       [fitted.coef_intercept, fitted.coef_gtv, fitted.coef_mhd],
        "Difference":         [
            fitted.coef_intercept - (-1.3409),
            fitted.coef_gtv - 0.0590,
            fitted.coef_mhd - 0.2635,
        ],
    }))

    st.subheader("Key diagnostic values")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Mean p_photon (God-world)", f"{truth.p_photon.mean():.3f}",
        help="Should be close to 0.51 — the Van Loon mortality rate",
    )
    col2.metric(
        "Mean outcome_photon (sampled)", f"{obs.outcomes_photon.mean():.3f}",
        help="Should also be close to 0.51 — random variation expected",
    )
    col3.metric(
        "Fraction sensitive", f"{pop.is_sensitive.mean():.3f}",
        help="Should match pi_sensitive slider value",
    )
    col4, col5, col6 = st.columns(3)
    col4.metric(
        "Mean true_delta (all patients)", f"{_true_delta.mean():.4f}",
        help="Mean true benefit of protons across entire population",
    )
    col5.metric(
        "Mean true_delta (sensitive only)", f"{_true_delta[truth.is_sensitive].mean():.4f}",
        help="Mean true benefit for truly sensitive patients only",
    )
    col6.metric(
        "Mean true_delta (non-sensitive)", f"{_true_delta[~truth.is_sensitive].mean():.4f}",
        help="Should be zero or very close to zero",
    )

    st.subheader("True probability distribution")
    fig_debug = go.Figure()
    fig_debug.add_trace(go.Histogram(
        x=truth.p_photon[truth.is_sensitive],
        name="Sensitive", opacity=0.6, nbinsx=30,
    ))
    fig_debug.add_trace(go.Histogram(
        x=truth.p_photon[~truth.is_sensitive],
        name="Non-sensitive", opacity=0.6, nbinsx=30,
    ))
    fig_debug.update_layout(
        barmode="overlay",
        title="Distribution of true p_photon by sensitivity group",
        xaxis_title="True 2-year mortality probability",
        yaxis_title="Count",
    )
    st.plotly_chart(fig_debug, use_container_width=True, key="chart_debug_hist")

    st.subheader("True delta distribution")
    fig_debug2 = go.Figure()
    fig_debug2.add_trace(go.Histogram(
        x=_true_delta[truth.is_sensitive],
        name="Sensitive", opacity=0.6, nbinsx=30,
    ))
    fig_debug2.add_trace(go.Histogram(
        x=_true_delta[~truth.is_sensitive],
        name="Non-sensitive", opacity=0.6, nbinsx=30,
    ))
    fig_debug2.add_vline(x=0.02, line_dash="dash", line_color="red",
                         annotation_text="2% threshold")
    fig_debug2.update_layout(
        barmode="overlay",
        title="Distribution of true_delta by sensitivity group",
        xaxis_title="True benefit of protons (Δ mortality)",
        yaxis_title="Count",
    )
    st.plotly_chart(fig_debug2, use_container_width=True, key="chart_debug_delta")
