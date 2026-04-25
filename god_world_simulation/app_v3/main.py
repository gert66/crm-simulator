"""God-World NNT Simulator — Streamlit app."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(page_title="God-World NNT Simulator", layout="wide")

# ── Simulation-engine imports (guarded so sidebar still renders on error) ─────
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


# ── Plotly chart builders ─────────────────────────────────────────────────────
def _chart_nnt_bins(result) -> go.Figure:
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
        title="Predicted vs True NNT by Benefit Bin (selected patients)",
        xaxis_title="Predicted Δ bin",
        yaxis_title="NNT",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _chart_gtv(pop) -> go.Figure:
    fig = go.Figure(
        go.Histogram(x=pop.gtv, nbinsx=40, marker_color="#4C78A8", opacity=0.85)
    )
    fig.update_layout(title="GTV Distribution", xaxis_title="GTV (cc)", yaxis_title="Count")
    return fig


def _chart_mhd(pop) -> go.Figure:
    fig = go.Figure([
        go.Histogram(x=pop.mhd_photon, nbinsx=40, name="Photon MHD",
                     marker_color="#4C78A8", opacity=0.7),
        go.Histogram(x=pop.mhd_proton, nbinsx=40, name="Proton MHD",
                     marker_color="#F58518", opacity=0.7),
    ])
    fig.update_layout(
        barmode="overlay",
        title="MHD: Photon vs Proton",
        xaxis_title="MHD (Gy)",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _chart_calibration(truth, fitted) -> go.Figure:
    true_delta = (truth.p_photon - truth.p_proton) * truth.is_sensitive.astype(float)
    fig = go.Figure()
    for flag, color, name in [
        (True,  "#E45756", "Sensitive"),
        (False, "#4C78A8", "Non-sensitive"),
    ]:
        mask = truth.is_sensitive == flag
        fig.add_trace(go.Scatter(
            x=fitted.predicted_delta[mask],
            y=true_delta[mask],
            mode="markers",
            marker=dict(color=color, size=3, opacity=0.45),
            name=name,
        ))
    lo = float(fitted.predicted_delta.min())
    hi = float(fitted.predicted_delta.max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines", name="y = x",
        line=dict(color="black", dash="dash", width=1.5),
    ))
    fig.update_layout(
        title="Predicted Δ vs True Δ (god-world)",
        xaxis_title="Predicted Δ (fitted model)",
        yaxis_title="True Δ (god-world)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ── Sidebar controls (always rendered, independent of import success) ─────────
with st.sidebar:
    st.header("Controls")

    with st.expander("1 · Population", expanded=True):
        n_patients = st.slider("N patients", 200, 5000, 1000, step=100)
        dist = st.selectbox("Distribution", ["clinical", "normal"])
        seed = int(st.number_input("Seed", value=42, step=1, min_value=0))

    with st.expander("2 · Truth model (God-world)", expanded=True):
        pi_sensitive = st.slider(
            "Fraction truly sensitive to MHD (π)", 0.0, 1.0, 0.7, step=0.05
        )
        truth_mode = st.selectbox("Truth mode", ["god_world", "published"])
        proton_mode = st.selectbox("Proton mode", ["percentage", "absolute", "proportional"])
        if proton_mode == "absolute":
            proton_reduction_gy = st.slider("Proton reduction (Gy)", 1, 15, 5)
            proton_reduction_pct = 0.50
        else:
            proton_reduction_pct = st.slider(
                "Proton reduction (%)", 0.10, 0.90, 0.50, step=0.05
            )
            proton_reduction_gy = 5

    with st.expander("3 · Noise / Calibration", expanded=True):
        calibrate_auc_flag = st.checkbox("Auto-calibrate noise to target AUC", value=True)
        if calibrate_auc_flag:
            target_auc = st.slider("Target AUC", 0.50, 0.80, 0.64, step=0.01)
            manual_noise_sd = 1.5
        else:
            manual_noise_sd = st.slider("Manual noise SD", 0.1, 4.0, 1.5, step=0.1)
            target_auc = 0.64
        beta_z = st.slider("Confounder loading (β_z)", 0.0, 2.0, 0.5, step=0.1)

    with st.expander("4 · Selection", expanded=True):
        threshold = st.slider(
            "Selection threshold Δ (e.g. 0.02 = 2%)",
            0.01, 0.10, 0.02, step=0.005, format="%.3f",
        )

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("God-World NNT Simulator")
st.subheader("What does a 2% predicted benefit actually mean?")

# Surface import errors before the Run button so the cause is visible
if not IMPORTS_OK:
    st.error(f"Import failed: {import_error}")
    st.stop()

# ── Run button — all pipeline work happens here, no caching ──────────────────
if st.button("▶  Run simulation", type="primary", use_container_width=True):
    with st.spinner("Running pipeline…"):
        pop = generate_population(
            n=n_patients,
            pi_sensitive=float(pi_sensitive),
            proton_mode=proton_mode,
            proton_reduction_pct=float(proton_reduction_pct),
            proton_reduction_gy=float(proton_reduction_gy),
            dist=dist,
            seed=seed,
        )
        truth = compute_truth(pop, truth_mode=truth_mode)

        if calibrate_auc_flag:
            noise_sd = calibrate_noise(
                pop, truth,
                target_auc=float(target_auc),
                beta_z=float(beta_z),
                seed=seed,
            )
        else:
            noise_sd = float(manual_noise_sd)

        obs = add_noise(truth, noise_sd=noise_sd, beta_z=float(beta_z), seed=seed)
        fitted = fit_model(pop, obs)
        result = evaluate(truth, fitted, selection_threshold=float(threshold))

    st.session_state["pop"] = pop
    st.session_state["truth"] = truth
    st.session_state["fitted"] = fitted
    st.session_state["result"] = result

# ── Results panel — only shown after at least one successful run ──────────────
if "result" in st.session_state:
    pop    = st.session_state["pop"]
    truth  = st.session_state["truth"]
    fitted = st.session_state["fitted"]
    result = st.session_state["result"]

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC", f"{fitted.auc:.3f}")
    c2.metric("N Selected", result.n_selected)
    c3.metric("Predicted NNT", f"{result.predicted_nnt:.1f}")
    c4.metric("True NNT", f"{result.true_nnt:.1f}")

    st.markdown("---")

    # NNT inflation box
    infl = result.nnt_inflation
    box_color = "#d73027" if infl >= 2.0 else "#f46d43"
    st.markdown(
        f'<div style="background:{box_color};padding:20px;border-radius:10px;'
        f'text-align:center;margin:8px 0 16px 0;">'
        f'<b style="color:white;font-size:1.8em;">NNT Inflation Factor: {infl:.1f}×</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "NNT by predicted benefit bin",
        "Population distributions",
        "Calibration detail",
    ])

    with tab1:
        st.plotly_chart(_chart_nnt_bins(result), use_container_width=True)

    with tab2:
        col_gtv, col_mhd = st.columns(2)
        col_gtv.plotly_chart(_chart_gtv(pop), use_container_width=True)
        col_mhd.plotly_chart(_chart_mhd(pop), use_container_width=True)

    with tab3:
        st.plotly_chart(_chart_calibration(truth, fitted), use_container_width=True)
