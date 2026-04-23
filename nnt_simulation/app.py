"""
NNT Simulation: Predicted vs True Number Needed to Treat
Proton vs Photon therapy — interactive model playground

Run with:
    streamlit run nnt_simulation/app.py
"""

import math
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

# ── Session state: one key per interactive parameter ──────────────────────────

_DEFAULTS = {
    "n_patients":     5_000,
    "seed":           42,
    "gtv_mean":       50.0,
    "gtv_std":        20.0,
    "mhd_mean":       15.0,
    "mhd_std":         5.0,
    "intercept":      -1.5,
    "gtv_mid":        50.0,
    "gtv_slope":      -0.04,
    "mhd_mid":        15.0,
    "mhd_slope":      -0.30,
    "proton_mode":    "Multiply by factor",
    "proton_delta":    5.0,
    "proton_factor":   0.5,
    "delta_thresh":    0.02,
    "hist_mode":      "Histogram",
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Pure math helpers ─────────────────────────────────────────────────────────

def sigmoid(x, midpoint, slope):
    return 1.0 / (1.0 + np.exp(-slope * (x - midpoint)))


def logit_to_prob(logit):
    return 1.0 / (1.0 + np.exp(-logit))


def survival_prob(gtv, mhd, intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope):
    logit = (
        intercept
        + sigmoid(gtv, gtv_mid, gtv_slope)
        + sigmoid(mhd, mhd_mid, mhd_slope)
    )
    return logit_to_prob(logit)


def sample_truncnorm(rng, mean, std, n, lower=0.0):
    out = np.empty(n)
    filled = 0
    while filled < n:
        batch = rng.normal(loc=mean, scale=std, size=(n - filled) * 3 + 100)
        batch = batch[batch >= lower]
        take = min(len(batch), n - filled)
        out[filled : filled + take] = batch[:take]
        filled += take
    return out


def truncnorm_pdf(x_arr, mean, std, lower=0.0):
    """Theoretical PDF of a normal truncated at `lower`, used for the density view."""
    alpha = (lower - mean) / std
    Z = 1.0 - 0.5 * (1.0 + math.erf(alpha / math.sqrt(2)))
    Z = max(Z, 1e-9)
    z = (x_arr - mean) / std
    pdf = np.exp(-0.5 * z ** 2) / (std * math.sqrt(2 * math.pi) * Z)
    return np.where(x_arr < lower, 0.0, pdf)


def apply_proton_mhd(mhd, mode, delta, factor):
    if mode == "Set to zero":
        return np.zeros_like(mhd)
    if mode == "Subtract fixed delta":
        return np.maximum(mhd - delta, 0.0)
    return mhd * factor


# ── Bin definitions ───────────────────────────────────────────────────────────

BIN_EDGES  = [0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.01]
BIN_LABELS = ["2–5 %", "5–10 %", "10–20 %", "20–40 %", "40–60 %", "60–80 %", "80–100 %"]


def bin_analysis(pred_delta, true_delta, threshold):
    sel   = pred_delta >= threshold
    n_sel = int(sel.sum())
    rows  = []
    for label, lo, hi in zip(BIN_LABELS, BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = sel & (pred_delta >= lo) & (pred_delta < hi)
        n    = int(mask.sum())
        if n == 0:
            rows.append(dict(Bin=label, n=0, **{k: np.nan for k in
                         ["Share (%)", "Mean Δ pred", "Pred NNT", "True ARR", "True NNT"]}))
            continue
        mean_pred = pred_delta[mask].mean()
        true_arr  = true_delta[mask].mean()
        rows.append({
            "Bin":         label,
            "n":           n,
            "Share (%)":   n / n_sel * 100 if n_sel > 0 else np.nan,
            "Mean Δ pred": mean_pred,
            "Pred NNT":    1.0 / mean_pred if mean_pred > 1e-9 else np.nan,
            "True ARR":    true_arr,
            "True NNT":    1.0 / true_arr  if true_arr  > 1e-9 else np.nan,
        })
    return pd.DataFrame(rows), n_sel


# ── Simulation (unchanged logic, cached) ──────────────────────────────────────

@st.cache_data
def run_simulation(
    n, seed,
    gtv_mean, gtv_std,
    mhd_mean, mhd_std,
    intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope,
    proton_mode, proton_delta, proton_factor,
):
    rng    = np.random.default_rng(seed)
    gtv    = sample_truncnorm(rng, gtv_mean, gtv_std, n)
    mhd    = sample_truncnorm(rng, mhd_mean, mhd_std, n)
    p_ph   = survival_prob(gtv, mhd, intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope)
    out_ph = (rng.random(n) < p_ph).astype(float)
    mhd_pr = apply_proton_mhd(mhd, proton_mode, proton_delta, proton_factor)
    p_pr   = survival_prob(gtv, mhd_pr, intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope)
    out_pr = (rng.random(n) < p_pr).astype(float)
    pred_delta = p_pr - p_ph
    true_delta = out_pr - out_ph
    return gtv, mhd, p_ph, p_pr, pred_delta, true_delta


# ── Dual-input: synchronized slider + number field ───────────────────────────

def _on_slider(key):
    v = st.session_state[f"_sl_{key}"]
    st.session_state[key]          = v
    st.session_state[f"_nu_{key}"] = v


def _on_num(key, lo, hi):
    v = float(st.session_state[f"_nu_{key}"])
    st.session_state[key]          = v
    st.session_state[f"_sl_{key}"] = float(np.clip(v, lo, hi))


def dual_param(label, key, lo, hi, step, fmt="%.2f", help_text=None):
    """
    Slider (wide) + number input (narrow) sharing session_state[key].
    Changing either widget immediately updates the other.
    Returns the current float value.
    """
    sl_key  = f"_sl_{key}"
    nu_key  = f"_nu_{key}"
    current = float(st.session_state[key])
    if sl_key not in st.session_state:
        st.session_state[sl_key] = float(np.clip(current, lo, hi))
    if nu_key not in st.session_state:
        st.session_state[nu_key] = current

    c1, c2 = st.columns([3, 1])
    with c1:
        st.slider(label, float(lo), float(hi), step=float(step),
                  key=sl_key, help=help_text,
                  on_change=_on_slider, args=(key,))
    with c2:
        st.number_input(label,
                        min_value=float(lo), max_value=float(hi),
                        step=float(step), format=fmt,
                        key=nu_key,
                        on_change=_on_num, args=(key, lo, hi),
                        label_visibility="collapsed")
    return float(st.session_state[key])


# ── Combined distribution + sigmoid figure ────────────────────────────────────

def make_combined_plot(
    data_a, mean, std, sig_mid, sig_slope,
    x_label, title,
    data_b=None, label_a="Photon", label_b="Proton",
    color_a="#4C8BF5", color_b="#00C9A7", color_sig="#E86510",
    hist_mode="Histogram", n_bins=40,
):
    """
    Left y-axis : histogram or scaled theoretical density of data_a
                  (and optional data_b overlay).
    Right y-axis: sigmoid σ(x) over [0, 1].
    X range is fixed to ±4 SD of the photon distribution so the axis
    does not jump when the proton overlay extends to zero.
    """
    x_lo    = max(0.0, mean - 4 * std)
    x_hi    = mean + 4 * std
    x_curve = np.linspace(x_lo, x_hi, 400)
    sig_y   = sigmoid(x_curve, sig_mid, sig_slope)

    fig = go.Figure()

    if hist_mode == "Histogram":
        fig.add_trace(go.Histogram(
            x=data_a, nbinsx=n_bins, name=label_a,
            marker_color=color_a, opacity=0.75, yaxis="y",
        ))
        if data_b is not None:
            fig.add_trace(go.Histogram(
                x=data_b, nbinsx=n_bins, name=label_b,
                marker_color=color_b, opacity=0.50, yaxis="y",
            ))
    else:  # Density: theoretical truncated-normal PDF scaled to match histogram area
        x_den = np.linspace(x_lo, x_hi, 600)
        pdf   = truncnorm_pdf(x_den, mean, std)
        scale = len(data_a) * (x_hi - x_lo) / n_bins
        ra, ga, ba = int(color_a[1:3], 16), int(color_a[3:5], 16), int(color_a[5:7], 16)
        fig.add_trace(go.Scatter(
            x=x_den, y=pdf * scale, mode="lines", fill="tozeroy",
            line=dict(color=color_a, width=2.5),
            fillcolor=f"rgba({ra},{ga},{ba},0.15)",
            name=label_a, yaxis="y",
        ))
        if data_b is not None:
            fig.add_trace(go.Histogram(
                x=data_b, nbinsx=n_bins, name=label_b,
                marker_color=color_b, opacity=0.40, yaxis="y",
            ))

    fig.add_trace(go.Scatter(
        x=x_curve, y=sig_y, mode="lines",
        line=dict(color=color_sig, width=2.5),
        name="σ (logit contrib.)", yaxis="y2",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title=x_label, range=[x_lo, x_hi]),
        yaxis=dict(
            title="Count" if hist_mode == "Histogram" else "Density (scaled)",
            side="left",
        ),
        yaxis2=dict(
            title="σ (logit contribution)", side="right", overlaying="y",
            range=[0, 1], showgrid=False, tickformat=".1f",
        ),
        height=410,
        margin=dict(t=55, b=50, l=65, r=75),
        legend=dict(orientation="h", y=1.16, x=0, xanchor="left", font=dict(size=11)),
        barmode="overlay",
        plot_bgcolor="rgba(248,249,250,1)",
        paper_bgcolor="white",
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Configuration")

    with st.expander("Population", expanded=True):
        st.number_input("Number of patients",
                        min_value=200, max_value=100_000, step=200,
                        key="n_patients")
        st.number_input("Random seed",
                        min_value=0, max_value=9_999, step=1,
                        key="seed")

    with st.expander("Proton effect", expanded=True):
        st.caption(
            "Choose how proton therapy reduces mean heart dose. "
            "Adjust the reduction magnitude in the main panel."
        )
        st.radio(
            "MHD reduction mode",
            ["Set to zero", "Subtract fixed delta", "Multiply by factor"],
            key="proton_mode",
        )

    with st.expander("Display", expanded=True):
        st.radio("Distribution view", ["Histogram", "Density"], key="hist_mode")

# ── Read current values from session state ────────────────────────────────────

n_patients    = int(st.session_state["n_patients"])
seed          = int(st.session_state["seed"])
gtv_mean      = float(st.session_state["gtv_mean"])
gtv_std       = float(st.session_state["gtv_std"])
mhd_mean      = float(st.session_state["mhd_mean"])
mhd_std       = float(st.session_state["mhd_std"])
intercept     = float(st.session_state["intercept"])
gtv_mid       = float(st.session_state["gtv_mid"])
gtv_slope     = float(st.session_state["gtv_slope"])
mhd_mid       = float(st.session_state["mhd_mid"])
mhd_slope     = float(st.session_state["mhd_slope"])
proton_mode   = st.session_state["proton_mode"]
proton_delta  = float(st.session_state["proton_delta"])
proton_factor = float(st.session_state["proton_factor"])
delta_thresh  = float(st.session_state["delta_thresh"])
hist_mode     = st.session_state["hist_mode"]

# ── Simulation ────────────────────────────────────────────────────────────────

gtv, mhd, p_ph, p_pr, pred_delta, true_delta = run_simulation(
    n_patients, seed,
    gtv_mean, gtv_std, mhd_mean, mhd_std,
    intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope,
    proton_mode, proton_delta, proton_factor,
)
mhd_pr   = apply_proton_mhd(mhd, proton_mode, proton_delta, proton_factor)
selected = pred_delta >= delta_thresh
n_sel    = int(selected.sum())

# ── Title ─────────────────────────────────────────────────────────────────────

st.title("Predicted vs True NNT — Proton vs Photon")
st.markdown(
    "Adjust any parameter and all plots update immediately. "
    "Each patient's GTV and MHD are sampled from truncated Gaussians; a logistic "
    "survival model converts them to 2-year OS probabilities. Proton therapy "
    "reduces MHD, generating a per-patient predicted benefit Δ. Monte Carlo "
    "binary outcomes then yield the **true NNT** alongside the **predicted NNT**."
)

# ── Summary cards ─────────────────────────────────────────────────────────────

ca, cb, cc, cd = st.columns(4)
ca.metric("Mean P(OS2y) — photon",  f"{p_ph.mean():.3f}")
cb.metric("Mean P(OS2y) — proton",  f"{p_pr.mean():.3f}")
cc.metric("Mean predicted Δ (all)", f"{pred_delta.mean():.3f}")
cd.metric("Mean true Δ (all)",      f"{true_delta.mean():.3f}")

ce, cf, cg, ch = st.columns(4)
ce.metric("Patients selected", f"{n_sel:,}",
          f"{n_sel / n_patients * 100:.1f} % of total")
if n_sel > 0:
    mp = float(pred_delta[selected].mean())
    mt = float(true_delta[selected].mean())
    cf.metric("Mean predicted Δ (selected)", f"{mp:.3f}")
    cg.metric("Predicted NNT (selected)", f"{1/mp:.1f}" if mp > 1e-9 else "∞")
    ch.metric("True NNT (selected)",      f"{1/mt:.1f}" if mt > 1e-9 else "∞")
else:
    cf.metric("Mean predicted Δ (selected)", "—")
    cg.metric("Predicted NNT (selected)", "—")
    ch.metric("True NNT (selected)", "—")

st.divider()

# ── Model Playground ──────────────────────────────────────────────────────────

st.subheader("Model Playground")
st.markdown(
    "The combined plots show where patients sit on the sigmoid curve. "
    "Where the curve is steep, small changes in the covariate produce large changes "
    "in predicted survival. Use the slider for quick exploration or type an exact "
    "value in the box on the right — both stay synchronised."
)

col_gtv, col_mhd = st.columns(2)

# ── GTV column ────────────────────────────────────────────────────────────────
with col_gtv:
    st.markdown("#### GTV")
    st.plotly_chart(
        make_combined_plot(
            gtv, gtv_mean, gtv_std, gtv_mid, gtv_slope,
            x_label="GTV (cc)",
            title="GTV distribution & sigmoid contribution",
            color_a="#4C8BF5", color_sig="#E86510",
            hist_mode=hist_mode,
        ),
        use_container_width=True,
    )
    st.markdown("**Distribution**")
    dual_param("Mean (cc)", "gtv_mean", 0.0, 200.0, 1.0,  "%.1f")
    dual_param("Std  (cc)", "gtv_std",  0.5,  80.0, 0.5,  "%.1f")
    st.markdown("**Sigmoid**")
    dual_param("Midpoint (cc)", "gtv_mid",   0.0, 200.0,  1.0, "%.1f")
    dual_param("Slope",         "gtv_slope", -1.0,   1.0, 0.01, "%.3f",
               help_text="Negative → larger GTV lowers survival.")

# ── MHD column ────────────────────────────────────────────────────────────────
with col_mhd:
    st.markdown("#### MHD")
    st.plotly_chart(
        make_combined_plot(
            mhd, mhd_mean, mhd_std, mhd_mid, mhd_slope,
            x_label="MHD (Gy)",
            title="MHD distribution & sigmoid (photon + proton)",
            data_b=mhd_pr,
            label_a="Photon MHD", label_b="Proton MHD",
            color_a="#E8543A", color_b="#00C9A7", color_sig="#7B61FF",
            hist_mode=hist_mode,
        ),
        use_container_width=True,
    )
    st.markdown("**Distribution**")
    dual_param("Mean (Gy)", "mhd_mean", 0.0, 60.0, 0.5, "%.1f")
    dual_param("Std  (Gy)", "mhd_std",  0.5, 25.0, 0.5, "%.1f")
    st.markdown("**Sigmoid**")
    dual_param("Midpoint (Gy)", "mhd_mid",   0.0, 60.0,  0.5,  "%.1f")
    dual_param("Slope",         "mhd_slope", -2.0,  2.0, 0.05, "%.3f",
               help_text="Negative → higher MHD lowers survival.")

# ── Intercept + proton reduction (full width, side by side) ───────────────────
c_left, c_right = st.columns(2)

with c_left:
    st.markdown("**Baseline (intercept)**")
    st.caption("Log-odds of survival when both sigmoid contributions equal 0.5.")
    dual_param("Intercept (logit scale)", "intercept", -6.0, 2.0, 0.05, "%.2f")

with c_right:
    st.markdown("**Proton MHD reduction**")
    if proton_mode == "Subtract fixed delta":
        dual_param("Reduction (Gy)", "proton_delta", 0.0, 40.0, 0.5, "%.1f")
    elif proton_mode == "Multiply by factor":
        dual_param("Reduction factor", "proton_factor", 0.0, 1.0, 0.05, "%.2f",
                   help_text="0 = abolish MHD entirely; 1 = no change.")
    else:
        st.info("MHD is set to zero for all proton patients (mode: Set to zero).")

st.divider()

# ── Patient selection ─────────────────────────────────────────────────────────

st.subheader("Predicted Survival Benefit (Proton − Photon)")
st.markdown(
    f"Δ = P_proton − P_photon per patient. Patients with Δ ≥ threshold are "
    f"selected for proton therapy. The current threshold is "
    f"**{delta_thresh:.3f}** ({delta_thresh * 100:.1f} %)."
)
dual_param("Predicted Δ selection threshold", "delta_thresh",
           0.0, 0.50, 0.005, "%.3f")

col1, col2 = st.columns(2)
with col1:
    fig = go.Figure(go.Histogram(x=pred_delta, nbinsx=60, marker_color="#7B61FF"))
    fig.add_vline(x=delta_thresh, line_dash="dash", line_color="crimson",
                  annotation_text=f"Threshold {delta_thresh:.3f}",
                  annotation_position="top right")
    fig.update_layout(
        title="Predicted Δ — all patients",
        xaxis_title="Predicted Δ (P_proton − P_photon)", yaxis_title="Count",
        height=300, margin=dict(t=40, b=40, l=50, r=20),
        plot_bgcolor="rgba(248,249,250,1)", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if n_sel > 0:
        fig = go.Figure(
            go.Histogram(x=pred_delta[selected], nbinsx=40, marker_color="#00C9A7")
        )
        fig.update_layout(
            title=f"Predicted Δ — selected patients (n = {n_sel:,})",
            xaxis_title="Predicted Δ", yaxis_title="Count",
            height=300, margin=dict(t=40, b=40, l=50, r=20),
            plot_bgcolor="rgba(248,249,250,1)", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No patients exceed the selection threshold. "
            "Lower the threshold to see selected patients."
        )

st.divider()

# ── Bin analysis ──────────────────────────────────────────────────────────────

st.subheader("Bin Analysis: Predicted NNT vs True NNT")
st.markdown(
    "Selected patients are grouped into fixed bins by their predicted benefit. "
    "Within each bin, predicted NNT (1 / mean predicted Δ) is compared with "
    "true NNT (1 / mean true Δ from Monte Carlo outcomes). "
    "Close agreement indicates good calibration; divergence shows where the model "
    "over- or under-predicts benefit."
)

if n_sel == 0:
    st.info("Lower the selection threshold to populate the bin analysis.")
else:
    bin_df, _ = bin_analysis(pred_delta, true_delta, float(delta_thresh))

    def _fmt(v, spec):
        return format(v, spec) if pd.notna(v) else "—"

    display = bin_df.copy()
    display["Share (%)"]   = display["Share (%)"].map(lambda v: _fmt(v, ".1f"))
    display["Mean Δ pred"] = display["Mean Δ pred"].map(lambda v: _fmt(v, ".4f"))
    display["Pred NNT"]    = display["Pred NNT"].map(lambda v: _fmt(v, ".1f"))
    display["True ARR"]    = display["True ARR"].map(lambda v: _fmt(v, ".4f"))
    display["True NNT"]    = display["True NNT"].map(lambda v: _fmt(v, ".1f"))
    st.dataframe(display, use_container_width=True, hide_index=True)

    chart_df = bin_df.dropna(subset=["Pred NNT", "True NNT"])
    if len(chart_df) >= 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_df["Bin"], y=chart_df["Pred NNT"],
            mode="lines+markers", name="Predicted NNT",
            line=dict(color="#7B61FF", width=2.5),
            marker=dict(size=9, symbol="circle"),
        ))
        fig.add_trace(go.Scatter(
            x=chart_df["Bin"], y=chart_df["True NNT"],
            mode="lines+markers", name="True NNT",
            line=dict(color="#00C9A7", width=2.5),
            marker=dict(size=9, symbol="diamond"),
        ))
        fig.update_layout(
            title="Predicted NNT vs True NNT by predicted benefit bin",
            xaxis_title="Predicted benefit bin",
            yaxis_title="NNT",
            height=420,
            margin=dict(t=50, b=50, l=60, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            plot_bgcolor="rgba(248,249,250,1)", paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Not enough populated bins to draw the comparison chart. "
            "Increase the number of patients or adjust the proton effect."
        )
