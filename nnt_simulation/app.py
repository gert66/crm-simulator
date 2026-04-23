"""
NNT Simulation: Predicted vs True Number Needed to Treat
Proton vs Photon therapy — survival benefit analysis

Run with:
    streamlit run nnt_simulation/app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NNT Simulation — Proton vs Photon",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pure helper functions ─────────────────────────────────────────────────────

def sigmoid(x, midpoint, slope):
    """Logistic sigmoid mapping x to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-slope * (x - midpoint)))


def logit_to_prob(logit):
    return 1.0 / (1.0 + np.exp(-logit))


def survival_prob(gtv, mhd, intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope):
    """
    2-year OS probability via additive logistic model:
        logit(P) = intercept + sigmoid(GTV) + sigmoid(MHD)
    """
    logit = (
        intercept
        + sigmoid(gtv, gtv_mid, gtv_slope)
        + sigmoid(mhd, mhd_mid, mhd_slope)
    )
    return logit_to_prob(logit)


def sample_truncnorm(rng, mean, std, n, lower=0.0):
    """Rejection-sample from a normal truncated at `lower`."""
    out = np.empty(n)
    filled = 0
    while filled < n:
        batch = rng.normal(loc=mean, scale=std, size=(n - filled) * 3 + 100)
        batch = batch[batch >= lower]
        take = min(len(batch), n - filled)
        out[filled : filled + take] = batch[:take]
        filled += take
    return out


def apply_proton_mhd(mhd, mode, delta, factor):
    if mode == "Set to zero":
        return np.zeros_like(mhd)
    if mode == "Subtract fixed delta":
        return np.maximum(mhd - delta, 0.0)
    return mhd * factor  # Multiply by factor


# ── Bin definitions ───────────────────────────────────────────────────────────

BIN_EDGES = [0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.01]
BIN_LABELS = ["2–5 %", "5–10 %", "10–20 %", "20–40 %", "40–60 %", "60–80 %", "80–100 %"]


def bin_analysis(pred_delta, true_delta, threshold):
    """
    Group selected patients (pred_delta >= threshold) into fixed bins.
    Returns a DataFrame and the total number of selected patients.
    """
    sel = pred_delta >= threshold
    n_sel = int(sel.sum())
    rows = []
    for label, lo, hi in zip(BIN_LABELS, BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = sel & (pred_delta >= lo) & (pred_delta < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                dict(Bin=label, n=0, **{k: np.nan for k in
                     ["Share (%)", "Mean Δ pred", "Pred NNT", "True ARR", "True NNT"]})
            )
            continue
        mean_pred = pred_delta[mask].mean()
        true_arr = true_delta[mask].mean()
        rows.append({
            "Bin": label,
            "n": n,
            "Share (%)": n / n_sel * 100 if n_sel > 0 else np.nan,
            "Mean Δ pred": mean_pred,
            "Pred NNT": 1.0 / mean_pred if mean_pred > 1e-9 else np.nan,
            "True ARR": true_arr,
            "True NNT": 1.0 / true_arr if true_arr > 1e-9 else np.nan,
        })
    return pd.DataFrame(rows), n_sel


# ── Simulation (cached) ───────────────────────────────────────────────────────

@st.cache_data
def run_simulation(
    n, seed,
    gtv_mean, gtv_std,
    mhd_mean, mhd_std,
    intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope,
    proton_mode, proton_delta, proton_factor,
):
    rng = np.random.default_rng(seed)

    gtv = sample_truncnorm(rng, gtv_mean, gtv_std, n)
    mhd = sample_truncnorm(rng, mhd_mean, mhd_std, n)

    p_ph = survival_prob(gtv, mhd, intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope)
    out_ph = (rng.random(n) < p_ph).astype(float)

    mhd_pr = apply_proton_mhd(mhd, proton_mode, proton_delta, proton_factor)
    p_pr = survival_prob(gtv, mhd_pr, intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope)
    out_pr = (rng.random(n) < p_pr).astype(float)

    pred_delta = p_pr - p_ph
    true_delta = out_pr - out_ph
    return gtv, mhd, p_ph, p_pr, pred_delta, true_delta


# ── Sidebar inputs ────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Configuration")

    with st.expander("Population", expanded=True):
        n_patients = st.number_input("Number of patients", 200, 100_000, 5_000, 200)
        seed = int(st.number_input("Random seed", 0, 9_999, 42, 1))

    with st.expander("GTV distribution", expanded=True):
        gtv_mean = st.number_input("Mean (cc)", value=50.0, step=1.0, key="gm")
        gtv_std = st.number_input("Std (cc)", value=20.0, min_value=0.5, step=0.5, key="gs")

    with st.expander("MHD distribution", expanded=True):
        mhd_mean = st.number_input("Mean (Gy)", value=15.0, step=0.5, key="mm")
        mhd_std = st.number_input("Std (Gy)", value=5.0, min_value=0.5, step=0.5, key="ms")

    with st.expander("Survival model", expanded=True):
        st.caption(
            "logit(P) = intercept + σ(GTV) + σ(MHD). "
            "Each σ is a logistic sigmoid in (0, 1). "
            "Negative slope means higher values reduce survival."
        )
        intercept = st.slider("Intercept (logit scale)", -6.0, 2.0, -1.5, 0.05)
        st.markdown("**GTV sigmoid**")
        gtv_mid = st.number_input("Midpoint (cc)", value=50.0, step=1.0, key="gvmid")
        gtv_slope = st.number_input(
            "Slope", value=-0.04, step=0.01, format="%.3f", key="gvslp"
        )
        st.markdown("**MHD sigmoid**")
        mhd_mid = st.number_input("Midpoint (Gy)", value=15.0, step=0.5, key="mhmid")
        mhd_slope = st.number_input(
            "Slope", value=-0.30, step=0.05, format="%.3f", key="mhslp"
        )

    with st.expander("Proton effect", expanded=True):
        st.caption("Choose how proton therapy reduces mean heart dose compared with photons.")
        proton_mode = st.radio(
            "MHD reduction mode",
            ["Set to zero", "Subtract fixed delta", "Multiply by factor"],
        )
        proton_delta = 0.0
        proton_factor = 1.0
        if proton_mode == "Subtract fixed delta":
            proton_delta = float(
                st.number_input("Reduction (Gy)", value=5.0, min_value=0.0, step=0.5)
            )
        elif proton_mode == "Multiply by factor":
            proton_factor = st.slider(
                "Reduction factor",
                0.0, 1.0, 0.5, 0.05,
                help="0 = abolish MHD entirely; 1 = no change.",
            )

    with st.expander("Patient selection", expanded=True):
        st.caption(
            "Patients are offered proton therapy if their predicted survival benefit "
            "meets or exceeds this threshold."
        )
        delta_thresh = st.slider(
            "Predicted Δ threshold", 0.0, 0.50, 0.02, 0.005, format="%.3f"
        )

# ── Run simulation ────────────────────────────────────────────────────────────

gtv, mhd, p_ph, p_pr, pred_delta, true_delta = run_simulation(
    int(n_patients), seed,
    float(gtv_mean), float(gtv_std),
    float(mhd_mean), float(mhd_std),
    float(intercept), float(gtv_mid), float(gtv_slope),
    float(mhd_mid), float(mhd_slope),
    proton_mode, float(proton_delta), float(proton_factor),
)

selected = pred_delta >= delta_thresh
n_sel = int(selected.sum())

# ── Page title and description ────────────────────────────────────────────────

st.title("Predicted vs True NNT — Proton vs Photon")
st.markdown(
    "This simulator draws a population of patients with individual GTV and MHD values "
    "from truncated Gaussian distributions. A logistic survival model maps these covariates "
    "to 2-year OS probabilities. Proton therapy reduces MHD, increasing each patient's "
    "predicted survival probability. Binary outcomes are then sampled by Monte Carlo, "
    "which allows comparing the **predicted NNT** (from survival probabilities) "
    "with the **true NNT** (from sampled outcomes)."
)

# ── Summary cards ─────────────────────────────────────────────────────────────

st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Patients selected",
    f"{n_sel:,}",
    f"{n_sel / int(n_patients) * 100:.1f} % of total",
)
if n_sel > 0:
    mp = float(pred_delta[selected].mean())
    mt = float(true_delta[selected].mean())
    pred_nnt = 1.0 / mp if mp > 1e-9 else float("inf")
    true_nnt = 1.0 / mt if mt > 1e-9 else float("inf")
    c2.metric("Mean predicted Δ", f"{mp:.3f}")
    c3.metric("Predicted NNT", f"{pred_nnt:.1f}")
    c4.metric("True NNT", f"{true_nnt:.1f}")
else:
    c2.metric("Mean predicted Δ", "—")
    c3.metric("Predicted NNT", "—")
    c4.metric("True NNT", "—")

st.divider()

# ── Distribution plots ────────────────────────────────────────────────────────

st.subheader("Population Distributions")
st.markdown(
    "GTV and MHD are sampled independently from truncated Gaussian distributions "
    "(clipped at zero)."
)

col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(go.Histogram(x=gtv, nbinsx=40, marker_color="#4C8BF5"))
    fig.update_layout(
        title="GTV distribution",
        xaxis_title="GTV (cc)",
        yaxis_title="Count",
        height=300,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure(go.Histogram(x=mhd, nbinsx=40, marker_color="#F5824C"))
    fig.update_layout(
        title="MHD distribution (photon)",
        xaxis_title="MHD (Gy)",
        yaxis_title="Count",
        height=300,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Sigmoid overlay plots ─────────────────────────────────────────────────────

st.subheader("Survival Model: Sigmoid Contributions")
st.markdown(
    "Each sigmoid maps a covariate to a value in (0, 1) that is added to the logit of "
    "survival. Dots represent a random sample of patients coloured by their photon "
    "survival probability (red = low, green = high)."
)

rng_plot = np.random.default_rng(seed + 999)
idx = rng_plot.choice(int(n_patients), min(800, int(n_patients)), replace=False)

col1, col2 = st.columns(2)

with col1:
    lo_x = max(0.0, float(gtv_mean) - 4 * float(gtv_std))
    hi_x = float(gtv_mean) + 4 * float(gtv_std)
    x_curve = np.linspace(lo_x, hi_x, 300)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gtv[idx],
        y=sigmoid(gtv[idx], float(gtv_mid), float(gtv_slope)),
        mode="markers",
        marker=dict(
            size=4,
            color=p_ph[idx],
            colorscale="RdYlGn",
            cmin=0, cmax=1,
            opacity=0.55,
            colorbar=dict(title="P(OS2y)", thickness=12, len=0.8),
        ),
        name="Patients",
    ))
    fig.add_trace(go.Scatter(
        x=x_curve,
        y=sigmoid(x_curve, float(gtv_mid), float(gtv_slope)),
        mode="lines",
        line=dict(color="#1e3a5f", width=2.5),
        name="Sigmoid",
    ))
    fig.update_layout(
        title="GTV → logit contribution",
        xaxis_title="GTV (cc)",
        yaxis_title="σ(GTV)",
        yaxis=dict(range=[0, 1]),
        height=320,
        margin=dict(t=40, b=40, l=50, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    lo_x = max(0.0, float(mhd_mean) - 4 * float(mhd_std))
    hi_x = float(mhd_mean) + 4 * float(mhd_std)
    x_curve = np.linspace(lo_x, hi_x, 300)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mhd[idx],
        y=sigmoid(mhd[idx], float(mhd_mid), float(mhd_slope)),
        mode="markers",
        marker=dict(
            size=4,
            color=p_ph[idx],
            colorscale="RdYlGn",
            cmin=0, cmax=1,
            opacity=0.55,
        ),
        name="Patients",
    ))
    fig.add_trace(go.Scatter(
        x=x_curve,
        y=sigmoid(x_curve, float(mhd_mid), float(mhd_slope)),
        mode="lines",
        line=dict(color="#1e3a5f", width=2.5),
        name="Sigmoid",
    ))
    fig.update_layout(
        title="MHD → logit contribution",
        xaxis_title="MHD (Gy)",
        yaxis_title="σ(MHD)",
        yaxis=dict(range=[0, 1]),
        height=320,
        margin=dict(t=40, b=40, l=50, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Delta survival ────────────────────────────────────────────────────────────

st.divider()
st.subheader("Predicted Survival Benefit (Proton − Photon)")
st.markdown(
    f"The predicted benefit Δ = P_proton − P_photon is determined by how much MHD "
    f"is reduced and how sensitively the survival model responds to MHD. "
    f"The dashed red line marks the current selection threshold of "
    f"**{delta_thresh:.3f}** ({delta_thresh * 100:.1f} %)."
)

col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(go.Histogram(x=pred_delta, nbinsx=60, marker_color="#7B61FF"))
    fig.add_vline(
        x=delta_thresh,
        line_dash="dash",
        line_color="crimson",
        annotation_text=f"Threshold {delta_thresh:.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Predicted Δ — all patients",
        xaxis_title="Predicted Δ (P_proton − P_photon)",
        yaxis_title="Count",
        height=300,
        margin=dict(t=40, b=40, l=50, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if n_sel > 0:
        fig = go.Figure(
            go.Histogram(x=pred_delta[selected], nbinsx=40, marker_color="#00C9A7")
        )
        fig.update_layout(
            title=f"Predicted Δ — selected patients (n = {n_sel:,})",
            xaxis_title="Predicted Δ",
            yaxis_title="Count",
            height=300,
            margin=dict(t=40, b=40, l=50, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No patients exceed the selection threshold. "
            "Lower the threshold in the sidebar to see selected patients."
        )

# ── Bin analysis ──────────────────────────────────────────────────────────────

st.divider()
st.subheader("Bin Analysis: Predicted NNT vs True NNT")
st.markdown(
    "Selected patients are grouped into fixed bins by their predicted benefit. "
    "Within each bin the predicted NNT (1 / mean predicted Δ) is compared with the "
    "true NNT (1 / mean true Δ from Monte Carlo outcomes). "
    "Close agreement between the two lines indicates good calibration; "
    "divergence reveals where the model over- or under-predicts benefit."
)

if n_sel == 0:
    st.info(
        "No patients exceed the selection threshold. "
        "Lower the threshold to populate the bin analysis."
    )
else:
    bin_df, _ = bin_analysis(pred_delta, true_delta, float(delta_thresh))

    # Formatted display table
    def fmt(v, spec):
        return format(v, spec) if pd.notna(v) else "—"

    display = bin_df.copy()
    display["Share (%)"] = display["Share (%)"].map(lambda v: fmt(v, ".1f"))
    display["Mean Δ pred"] = display["Mean Δ pred"].map(lambda v: fmt(v, ".4f"))
    display["Pred NNT"] = display["Pred NNT"].map(lambda v: fmt(v, ".1f"))
    display["True ARR"] = display["True ARR"].map(lambda v: fmt(v, ".4f"))
    display["True NNT"] = display["True NNT"].map(lambda v: fmt(v, ".1f"))

    st.dataframe(display, use_container_width=True, hide_index=True)

    # Line chart — bins with both NNTs defined
    chart_df = bin_df.dropna(subset=["Pred NNT", "True NNT"])
    if len(chart_df) >= 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_df["Bin"],
            y=chart_df["Pred NNT"],
            mode="lines+markers",
            name="Predicted NNT",
            line=dict(color="#7B61FF", width=2.5),
            marker=dict(size=9, symbol="circle"),
        ))
        fig.add_trace(go.Scatter(
            x=chart_df["Bin"],
            y=chart_df["True NNT"],
            mode="lines+markers",
            name="True NNT",
            line=dict(color="#00C9A7", width=2.5),
            marker=dict(size=9, symbol="diamond"),
        ))
        fig.update_layout(
            title="Predicted NNT vs True NNT by predicted benefit bin",
            xaxis_title="Predicted benefit bin",
            yaxis_title="NNT",
            height=420,
            margin=dict(t=50, b=50, l=60, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Not enough populated bins to draw the comparison chart. "
            "Increase the number of patients or adjust the proton effect."
        )
