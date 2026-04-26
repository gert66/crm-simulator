"""God-World NNT Simulator — reactive Streamlit app (no Run button)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
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
    raw_delta = truth.p_photon - truth.p_proton  # 0 for non-sensitive by construction
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
    """Predicted Δ (x) vs True Δ (y), coloured by latent sensitivity."""
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
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
            proton_reduction_gy  = st.slider("Proton reduction (Gy)", 1, 15, 5)
            proton_reduction_pct = 0.50
        else:
            proton_reduction_pct = st.slider(
                "Proton reduction (%)", 0.10, 0.90, 0.50, step=0.05
            )
            proton_reduction_gy = 5

    with st.expander("3 · Noise / Calibration", expanded=True):
        calibrate_auc_flag = st.checkbox("Auto-calibrate noise to target AUC", value=True)
        if calibrate_auc_flag:
            target_auc     = st.slider("Target AUC", 0.50, 0.80, 0.64, step=0.01)
            manual_noise_sd = 1.5
        else:
            manual_noise_sd = st.slider("Manual noise SD", 0.1, 4.0, 1.5, step=0.1)
            target_auc      = 0.64
        beta_z = st.slider("Confounder loading (β_z)", 0.0, 2.0, 0.5, step=0.1)

    with st.expander("4 · Selection", expanded=True):
        threshold = st.slider(
            "Selection threshold Δ (e.g. 0.02 = 2%)",
            0.01, 0.10, 0.02, step=0.005, format="%.3f",
        )


# ── Title + import guard ──────────────────────────────────────────────────────
st.title("God-World NNT Simulator")
st.subheader("What does a 2% predicted benefit actually mean?")

if not IMPORTS_OK:
    st.error(f"Import failed: {import_error}")
    st.stop()


# ── Reactive pipeline with incremental caching ────────────────────────────────
#
# Each stage is keyed on its direct inputs. A stage reruns only when its key
# differs from the previous script execution stored in st.session_state.

# Stage 1 — Population (+ both truth models, cheap to always pair together)
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
    from sklearn.metrics import roc_auc_score as _roc_auc
    _X  = np.column_stack([np.sqrt(pop.gtv), np.sqrt(pop.mhd_photon)])
    _y  = (truth.p_photon > np.median(truth.p_photon)).astype(int)
    _pr = LogisticRegression(max_iter=1000).fit(_X, _y).predict_proba(_X)[:, 1]
    st.session_state["auc_before"] = float(_roc_auc(_y, _pr))
    st.session_state["truth_key"]  = truth_key

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
fitted   = st.session_state["fitted"]

# Stage 3 — Evaluation (cheap; reruns whenever threshold or upstream changes)
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
        st.metric("Proton mode",          proton_mode)
        st.metric("Reduction amount",     reduction_str)
        st.metric("π (sensitive fraction)", f"{pi_sensitive:.0%}")
    with right:
        st.metric("Mean true Δ (God-world)", f"{delta_gw.mean():.4f}")
        st.metric("Mean true Δ (published)", f"{delta_pub.mean():.4f}")
        st.metric("% with Δ > 2%",          f"{(delta_sel > 0.02).mean():.1%}")
    st.plotly_chart(_fig_true_delta(truth), use_container_width=True, key="chart_true_delta")


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
