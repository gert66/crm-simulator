"""
NNT Simulation v2 — interactive model playground
Proton vs Photon therapy: predicted vs true NNT

Run with:  streamlit run nnt_simulation/nnt_sim_v2.py
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Chart config ──────────────────────────────────────────────────────────────

_CHART_CFG = {"displayModeBar": False}

# ── Default values ────────────────────────────────────────────────────────────

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
    "noise_enabled":  False,
    "noise_sd":        1.0,
    "z_enabled":      False,
    "z_mean":          0.0,
    "z_sd":            1.0,
    "z_beta":          0.5,
    "w_gtv":           1.0,
    "w_mhd":           1.0,
}

# ── Math helpers ──────────────────────────────────────────────────────────────

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


def _compute_auc(scores, labels):
    n_pos = float(labels.sum())
    n_neg = float(len(labels)) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(-scores)
    ls = labels[order].astype(float)
    tpr = np.concatenate([[0.0], np.cumsum(ls) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - ls) / n_neg])
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])) / 2)


def _compute_roc(scores, labels):
    """Return (fpr, tpr) arrays sorted by ascending fpr, pure numpy."""
    n_pos = float(labels.sum())
    n_neg = float(len(labels)) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-scores)
    ls = labels[order].astype(float)
    tpr = np.concatenate([[0.0], np.cumsum(ls) / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - ls) / n_neg, [1.0]])
    return fpr, tpr


# ── Logistic regression fitting ───────────────────────────────────────────────

@st.cache_data
def fit_logistic_regression(gtv, mhd, out_ph):
    """
    Fit LogisticRegression on standardised GTV+MHD to predict binary photon outcome.
    Returns (b0, b_gtv, b_mhd, gtv_mu, gtv_sig, mhd_mu, mhd_sig) or None on failure.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        gtv_mu, gtv_sig = float(gtv.mean()), float(gtv.std())
        mhd_mu, mhd_sig = float(mhd.mean()), float(mhd.std())
        X = np.column_stack([
            (gtv - gtv_mu) / gtv_sig,
            (mhd - mhd_mu) / mhd_sig,
        ])
        y = out_ph.astype(int)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y)
        return (
            float(lr.intercept_[0]),
            float(lr.coef_[0][0]),
            float(lr.coef_[0][1]),
            gtv_mu, gtv_sig,
            mhd_mu, mhd_sig,
        )
    except Exception:
        return None


# ── Bin definitions ───────────────────────────────────────────────────────────

BIN_EDGES  = [0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.01]
BIN_LABELS = ["2–5 %", "5–10 %", "10–20 %", "20–40 %", "40–60 %", "60–80 %", "80–100 %"]


def bin_analysis(pred_delta, true_delta, threshold, fit_delta=None):
    sel   = pred_delta >= threshold
    n_sel = int(sel.sum())
    rows  = []
    for label, lo, hi in zip(BIN_LABELS, BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = sel & (pred_delta >= lo) & (pred_delta < hi)
        n    = int(mask.sum())
        if n == 0:
            nan_cols = ["Share (%)", "Mean Δ pred", "Pred NNT", "True ARR", "True NNT"]
            if fit_delta is not None:
                nan_cols.append("Fitted NNT")
            rows.append(dict(Bin=label, n=0, **{k: np.nan for k in nan_cols}))
            continue
        mean_pred = pred_delta[mask].mean()
        true_arr  = true_delta[mask].mean()
        row = {
            "Bin":         label,
            "n":           n,
            "Share (%)":   n / n_sel * 100 if n_sel > 0 else np.nan,
            "Mean Δ pred": mean_pred,
            "Pred NNT":    1.0 / mean_pred if mean_pred > 1e-9 else np.nan,
            "True ARR":    true_arr,
            "True NNT":    1.0 / true_arr  if true_arr  > 1e-9 else np.nan,
        }
        if fit_delta is not None:
            mean_fit = float(fit_delta[mask].mean())
            row["Fitted NNT"] = 1.0 / mean_fit if mean_fit > 1e-9 else np.nan
        rows.append(row)
    return pd.DataFrame(rows), n_sel


# ── Simulation ────────────────────────────────────────────────────────────────

@st.cache_data
def run_simulation(
    n, seed,
    gtv_mean, gtv_std,
    mhd_mean, mhd_std,
    intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope,
    proton_mode, proton_delta, proton_factor,
    noise_enabled=False, noise_sd=1.0,
    z_enabled=False, z_mean=0.0, z_sd=1.0, z_beta=0.5,
    w_gtv=1.0, w_mhd=1.0,
):
    rng    = np.random.default_rng(seed)
    gtv    = sample_truncnorm(rng, gtv_mean, gtv_std, n)
    mhd    = sample_truncnorm(rng, mhd_mean, mhd_std, n)
    mhd_pr = apply_proton_mhd(mhd, proton_mode, proton_delta, proton_factor)

    # ── Prediction-world: model sees only GTV and MHD ────────────────────────
    # w_gtv / w_mhd scale each sigmoid contribution independently of its shape
    logit_ph = (intercept
                + w_gtv * sigmoid(gtv, gtv_mid, gtv_slope)
                + w_mhd * sigmoid(mhd, mhd_mid, mhd_slope))
    logit_pr = (intercept
                + w_gtv * sigmoid(gtv, gtv_mid, gtv_slope)
                + w_mhd * sigmoid(mhd_pr, mhd_mid, mhd_slope))

    # Predicted probabilities — Z and noise are never seen by the model
    p_ph = logit_to_prob(logit_ph)
    p_pr = logit_to_prob(logit_pr)

    # Noiseless, Z-free binary outcomes (AUC baseline reference)
    out_ph_base = (rng.random(n) < p_ph).astype(float)
    out_pr_base = (rng.random(n) < p_pr).astype(float)

    pred_delta = p_pr - p_ph  # always noiseless, always Z-free

    # ── Truth-world: true logit includes hidden factor Z and/or noise ─────────
    # Z is a patient-level characteristic; same value for both arms
    if z_enabled:
        z_rng  = np.random.default_rng(seed + 2000)
        z_vals = z_rng.normal(z_mean, z_sd, n)
        z_term = z_beta * z_vals
    else:
        z_rng  = None
        z_vals = None
        z_term = 0.0  # broadcasts cleanly with NumPy arrays

    # Additive noise — independent per patient, independent RNG stream
    if noise_enabled:
        noise_rng = np.random.default_rng(seed + 1000)
        eps_ph    = noise_rng.normal(0.0, noise_sd, n)
        eps_pr    = noise_rng.normal(0.0, noise_sd, n)
    else:
        noise_rng    = None
        eps_ph = eps_pr = 0.0

    if z_enabled or noise_enabled:
        p_true_ph  = logit_to_prob(logit_ph + z_term + eps_ph)
        p_true_pr  = logit_to_prob(logit_pr + z_term + eps_pr)
        # Binary draws use noise_rng when available, else z_rng
        draw_rng   = noise_rng if noise_enabled else z_rng
        out_ph     = (draw_rng.random(n) < p_true_ph).astype(float)
        out_pr     = (draw_rng.random(n) < p_true_pr).astype(float)
        true_delta = out_pr - out_ph
    else:
        out_ph     = out_ph_base
        true_delta = out_pr_base - out_ph_base

    # out_ph_base: Z-free noiseless outcomes (AUC reference)
    # out_ph:      outcomes from the full truth-world (Z + noise if enabled)
    # z_vals:      sampled Z values (None when Z is disabled)
    return gtv, mhd, p_ph, p_pr, pred_delta, true_delta, out_ph_base, out_ph, z_vals


# ── Dual-input widget ─────────────────────────────────────────────────────────

def _on_slider(key):
    v = st.session_state[f"_sl_{key}"]
    st.session_state[key]          = v
    st.session_state[f"_nu_{key}"] = v


def _on_num(key, lo, hi):
    v = float(st.session_state[f"_nu_{key}"])
    st.session_state[key]          = v
    st.session_state[f"_sl_{key}"] = float(np.clip(v, lo, hi))


def dual_param(label, key, lo, hi, step, fmt="%.2f", help_text=None):
    sl_key  = f"_sl_{key}"
    nu_key  = f"_nu_{key}"
    current = float(st.session_state[key])
    if sl_key not in st.session_state:
        st.session_state[sl_key] = float(np.clip(current, lo, hi))
    if nu_key not in st.session_state:
        st.session_state[nu_key] = current

    c_slider, c_num = st.columns([3, 1])
    with c_slider:
        st.slider(
            label,
            min_value=float(lo),
            max_value=float(hi),
            step=float(step),
            key=sl_key,
            help=help_text,
            on_change=_on_slider,
            args=(key,),
        )
    with c_num:
        st.number_input(
            label,
            min_value=float(lo),
            max_value=float(hi),
            step=float(step),
            format=fmt,
            key=nu_key,
            on_change=_on_num,
            args=(key, lo, hi),
            label_visibility="collapsed",
        )
    return float(st.session_state[key])


# ── Combined distribution + sigmoid figure ────────────────────────────────────

def make_combined_plot(
    data_a, mean, std, sig_mid, sig_slope,
    x_label, title,
    data_b=None, label_a="Photon", label_b="Proton",
    color_a="#4C8BF5", color_b="#00C9A7", color_sig="#E86510",
    hist_mode="Histogram", n_bins=40,
):
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
    else:
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
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("Simulation Setup")

        with st.expander("Population", expanded=True):
            st.number_input(
                "Number of patients",
                min_value=200,
                max_value=100_000,
                step=200,
                key="n_patients",
            )
            st.number_input(
                "Random seed",
                min_value=0,
                max_value=9_999,
                step=1,
                key="seed",
            )

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
            st.radio(
                "Distribution view",
                ["Histogram", "Density"],
                key="hist_mode",
            )

        with st.expander("Survival noise", expanded=False):
            st.checkbox(
                "Add random noise to survival mechanism",
                key="noise_enabled",
            )
            if st.session_state.get("noise_enabled", False):
                dual_param("Noise SD", "noise_sd", 0.1, 3.0, 0.1, "%.1f")

        with st.expander("Hidden prognostic factor Z", expanded=False):
            st.checkbox("Add hidden prognostic factor Z", key="z_enabled")
            if st.session_state.get("z_enabled", False):
                dual_param("Z mean",  "z_mean", -3.0, 3.0, 0.1, "%.1f")
                dual_param("Z SD",    "z_sd",    0.1, 3.0, 0.1, "%.1f")
                dual_param("β_Z",     "z_beta", -2.0, 2.0, 0.1, "%.1f")

    return (
        int(st.session_state["n_patients"]),
        int(st.session_state["seed"]),
        st.session_state["proton_mode"],
        st.session_state["hist_mode"],
        bool(st.session_state["noise_enabled"]),
        float(st.session_state["noise_sd"]),
        bool(st.session_state["z_enabled"]),
        float(st.session_state["z_mean"]),
        float(st.session_state["z_sd"]),
        float(st.session_state["z_beta"]),
    )


# ── Summary cards ─────────────────────────────────────────────────────────────

def render_summary_cards(n_patients, n_sel, p_ph, p_pr, pred_delta, true_delta, selected):
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


# ── Model playground ──────────────────────────────────────────────────────────

def render_playground(proton_mode, hist_mode, gtv, mhd, mhd_pr):
    st.subheader("Model Playground")
    st.markdown(
        "Adjust any parameter with the slider for quick exploration or type an "
        "exact value in the box on the right. All plots update on every change."
    )
    st.caption(
        "Shape and importance are now separate: you can have a sharp sigmoid "
        "that contributes little, or a shallow one that dominates."
    )

    col_gtv, col_mhd = st.columns(2)

    with col_gtv:
        st.markdown("#### GTV")
        st.caption(
            "Gross tumour volume contributes to survival through a sigmoid "
            "on the logit scale."
        )
        gtv_mean  = float(st.session_state["gtv_mean"])
        gtv_std   = float(st.session_state["gtv_std"])
        gtv_mid   = float(st.session_state["gtv_mid"])
        gtv_slope = float(st.session_state["gtv_slope"])
        st.plotly_chart(
            make_combined_plot(
                gtv, gtv_mean, gtv_std, gtv_mid, gtv_slope,
                x_label="GTV (cc)",
                title="GTV distribution & sigmoid contribution",
                color_a="#4C8BF5", color_sig="#E86510",
                hist_mode=hist_mode,
            ),
            use_container_width=True,
            config=_CHART_CFG,
        )
        st.markdown("**Distribution**")
        dual_param("Mean (cc)",  "gtv_mean",  0.0, 200.0, 1.0,  "%.1f")
        dual_param("Std  (cc)",  "gtv_std",   0.5,  80.0, 0.5,  "%.1f")
        st.markdown("**Sigmoid**")
        dual_param("Midpoint (cc)", "gtv_mid",   0.0, 200.0, 1.0,  "%.1f")
        dual_param(
            "Slope", "gtv_slope", -1.0, 1.0, 0.01, "%.3f",
            help_text="Negative → larger GTV reduces survival.",
        )
        dual_param(
            "Weight (w_GTV)", "w_gtv", -3.0, 3.0, 0.1, "%.1f",
            help_text="Scales the GTV sigmoid contribution to the logit.",
        )
        st.caption(
            "Midpoint controls where the transition happens. "
            "Slope controls how sharp it is. "
            "Weight controls how much GTV matters overall."
        )

    with col_mhd:
        st.markdown("#### MHD")
        st.caption(
            "Mean heart dose is reduced by proton therapy. "
            "The teal overlay shows the proton MHD distribution."
        )
        mhd_mean  = float(st.session_state["mhd_mean"])
        mhd_std   = float(st.session_state["mhd_std"])
        mhd_mid   = float(st.session_state["mhd_mid"])
        mhd_slope = float(st.session_state["mhd_slope"])
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
            config=_CHART_CFG,
        )
        st.markdown("**Distribution**")
        dual_param("Mean (Gy)", "mhd_mean",  0.0, 60.0, 0.5,  "%.1f")
        dual_param("Std  (Gy)", "mhd_std",   0.5, 25.0, 0.5,  "%.1f")
        st.markdown("**Sigmoid**")
        dual_param("Midpoint (Gy)", "mhd_mid",   0.0, 60.0,  0.5,  "%.1f")
        dual_param(
            "Slope", "mhd_slope", -2.0, 2.0, 0.05, "%.3f",
            help_text="Negative → higher MHD reduces survival.",
        )
        dual_param(
            "Weight (w_MHD)", "w_mhd", -3.0, 3.0, 0.1, "%.1f",
            help_text="Scales the MHD sigmoid contribution to the logit.",
        )
        st.caption(
            "Midpoint controls where the transition happens. "
            "Slope controls how sharp it is. "
            "Weight controls how much MHD matters overall."
        )

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Baseline (intercept)**")
        st.caption(
            "Sets the log-odds of survival when both sigmoids are at their midpoints. "
            "More negative means lower baseline survival."
        )
        dual_param("Intercept (logit scale)", "intercept", -6.0, 2.0, 0.05, "%.2f")

    with c_right:
        st.markdown("**Proton MHD reduction**")
        if proton_mode == "Subtract fixed delta":
            dual_param("Reduction (Gy)", "proton_delta", 0.0, 40.0, 0.5, "%.1f")
        elif proton_mode == "Multiply by factor":
            dual_param(
                "Reduction factor", "proton_factor", 0.0, 1.0, 0.05, "%.2f",
                help_text="0 = abolish MHD entirely; 1 = no change.",
            )
        else:
            st.info("MHD is set to zero for all proton patients.")


# ── Patient selection ─────────────────────────────────────────────────────────

def render_selection(pred_delta, true_delta, delta_thresh, selected, n_sel, noise_enabled=False):
    st.subheader("Predicted Survival Benefit (Proton − Photon)")
    st.markdown(
        f"Δ = P_proton − P_photon per patient. Patients with Δ ≥ threshold are "
        f"selected for proton therapy. The current threshold is "
        f"**{delta_thresh:.3f}** ({delta_thresh * 100:.1f} %)."
    )
    dual_param(
        "Predicted Δ threshold",
        "delta_thresh",
        lo=0.0, hi=0.50, step=0.005, fmt="%.3f",
    )

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
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)

    with col2:
        if n_sel > 0:
            fig = go.Figure(
                go.Histogram(x=pred_delta[selected], nbinsx=40, marker_color="#00C9A7")
            )
            fig.update_layout(
                title=f"Predicted Δ — selected patients (n = {n_sel:,})",
                xaxis_title="Predicted Δ", yaxis_title="Count",
                height=300, margin=dict(t=40, b=40, l=50, r=20),
            )
            st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)
        else:
            st.info(
                "No patients exceed the selection threshold. "
                "Lower the threshold to see selected patients."
            )

    if noise_enabled:
        st.caption(
            "True outcomes include additional random variation not visible to the model."
        )


# ── Bin analysis ──────────────────────────────────────────────────────────────

def render_bin_analysis(pred_delta, true_delta, delta_thresh, n_sel, fit_delta=None):
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
        return

    bin_df, _ = bin_analysis(pred_delta, true_delta, float(delta_thresh), fit_delta)

    def _fmt(v, spec):
        return format(v, spec) if pd.notna(v) else "—"

    display = bin_df.copy()
    display["Share (%)"]   = display["Share (%)"].map(lambda v: _fmt(v, ".1f"))
    display["Mean Δ pred"] = display["Mean Δ pred"].map(lambda v: _fmt(v, ".4f"))
    display["Pred NNT"]    = display["Pred NNT"].map(lambda v: _fmt(v, ".1f"))
    display["True ARR"]    = display["True ARR"].map(lambda v: _fmt(v, ".4f"))
    display["True NNT"]    = display["True NNT"].map(lambda v: _fmt(v, ".1f"))
    if "Fitted NNT" in display.columns:
        display["Fitted NNT"] = display["Fitted NNT"].map(lambda v: _fmt(v, ".1f"))
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
        if fit_delta is not None and "Fitted NNT" in chart_df.columns:
            fit_chart = bin_df.dropna(subset=["Fitted NNT"])
            if len(fit_chart) >= 1:
                fig.add_trace(go.Scatter(
                    x=fit_chart["Bin"], y=fit_chart["Fitted NNT"],
                    mode="lines+markers", name="Fitted NNT",
                    line=dict(color="#E8543A", width=2.5, dash="dot"),
                    marker=dict(size=9, symbol="square"),
                ))
        fig.update_layout(
            title="Predicted NNT vs True NNT by predicted benefit bin",
            xaxis_title="Predicted benefit bin",
            yaxis_title="NNT",
            height=420,
            margin=dict(t=50, b=50, l=60, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)
    else:
        st.info(
            "Not enough populated bins to draw the comparison chart. "
            "Increase the number of patients or adjust the proton effect."
        )


# ── Noise: AUC summary panel ──────────────────────────────────────────────────

def render_noise_section(p_ph, out_ph_base, out_ph, noise_sd):
    st.subheader("Random Noise: Model Discrimination")

    auc_no_noise   = _compute_auc(p_ph, out_ph_base)
    auc_with_noise = _compute_auc(p_ph, out_ph)

    c1, c2, c3 = st.columns(3)
    c1.metric("Noise SD",          f"{noise_sd:.1f}")
    c2.metric("AUC without noise", f"{auc_no_noise:.3f}")
    c3.metric("AUC with noise",    f"{auc_with_noise:.3f}")

    st.info(
        "Noise represents unexplained variation outside the measured predictors. "
        "As noise increases, AUC falls. A good-looking predicted delta can still "
        "correspond to a much weaker true effect."
    )


# ── Hidden factor Z: diagnostics panel ───────────────────────────────────────

def render_z_section(z_vals, z_mean, z_sd, z_beta):
    st.info(
        "The true world includes an unmeasured prognostic factor Z. "
        "The fitted model does not see Z."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Z mean",  f"{z_mean:.2f}")
    c2.metric("Z SD",    f"{z_sd:.2f}")
    c3.metric("β_Z",     f"{z_beta:.2f}")

    with st.expander("Hidden factor diagnostics"):
        fig = go.Figure(go.Histogram(
            x=z_vals, nbinsx=40,
            marker_color="#A855F7", opacity=0.80,
        ))
        fig.update_layout(
            title="Distribution of hidden factor Z",
            xaxis_title="Z", yaxis_title="Count",
            height=300, margin=dict(t=45, b=45, l=55, r=20),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)

    st.caption(
        "Z affects true survival but is invisible to the prediction model. "
        "This can reduce AUC and increase the gap between predicted and true NNT."
    )


# ── Fitted model ─────────────────────────────────────────────────────────────

def render_fitted_model(
    b0, b_gtv, b_mhd,
    p_ph, p_pr, pred_delta,
    p_fit_ph, p_fit_pr, fit_delta,
    true_delta,
):
    st.subheader("Fitted Model")

    # Coefficients row
    c1, c2, c3 = st.columns(3)
    c1.metric("Intercept (β₀)",  f"{b0:.3f}")
    c2.metric("β_GTV",           f"{b_gtv:.3f}")
    c3.metric("β_MHD",           f"{b_mhd:.3f}")

    st.markdown(
        f"**logit(P(OS2y)) = {b0:.3f}"
        f" {'+ ' if b_gtv >= 0 else '− '}{abs(b_gtv):.3f} × GTV"
        f" {'+ ' if b_mhd >= 0 else '− '}{abs(b_mhd):.3f} × MHD**"
    )
    st.caption(
        "GTV and MHD are standardised to zero mean and unit variance using training data. "
        "Logistic regression is fitted on binary photon survival outcomes."
    )

    # Comparison table
    comp_rows = [
        {
            "Metric":          "Mean photon survival",
            "Truth-generator": f"{p_ph.mean():.3f}",
            "Fitted model":    f"{p_fit_ph.mean():.3f}",
        },
        {
            "Metric":          "Mean proton survival",
            "Truth-generator": f"{p_pr.mean():.3f}",
            "Fitted model":    f"{p_fit_pr.mean():.3f}",
        },
        {
            "Metric":          "Mean predicted Δ",
            "Truth-generator": f"{pred_delta.mean():.4f}",
            "Fitted model":    f"{fit_delta.mean():.4f}",
        },
        {
            "Metric":          "Mean true Δ (outcomes)",
            "Truth-generator": f"{true_delta.mean():.4f}",
            "Fitted model":    "—",
        },
    ]
    st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)


# ── Model diagnostics ────────────────────────────────────────────────────────

def render_model_diagnostics(
    p_ph, p_pr, out_ph_base, out_ph,
    noise_enabled, noise_sd,
    intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope, w_gtv, w_mhd,
    p_fit_ph=None,
):
    st.subheader("Model Diagnostics")

    # ── ROC curve — use fitted predictions when available ─────────────────────
    roc_scores = p_fit_ph if p_fit_ph is not None else p_ph
    score_label = "Fitted LR" if p_fit_ph is not None else "Truth-generator"

    auc_base = _compute_auc(roc_scores, out_ph_base)
    fpr_base, tpr_base = _compute_roc(roc_scores, out_ph_base)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr_base, y=tpr_base, mode="lines",
        name=f"{score_label} vs noiseless outcomes (AUC = {auc_base:.3f})",
        line=dict(width=2.5),
    ))

    if noise_enabled:
        auc_noise = _compute_auc(roc_scores, out_ph)
        fpr_noise, tpr_noise = _compute_roc(roc_scores, out_ph)
        fig_roc.add_trace(go.Scatter(
            x=fpr_noise, y=tpr_noise, mode="lines",
            name=f"{score_label} vs noisy outcomes SD={noise_sd:.1f} (AUC = {auc_noise:.3f})",
            line=dict(width=2.5, dash="dot"),
        ))

    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="grey", width=1, dash="dash"),
        name="Random (AUC = 0.50)", showlegend=True,
    ))
    fig_roc.update_layout(
        title="ROC Curve — Predicted vs Actual Photon Survival",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
        margin=dict(t=50, b=50, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_roc, use_container_width=True, config=_CHART_CFG)

    col1, col2 = st.columns(2)

    # ── Prediction stats ───────────────────────────────────────────────────────
    with col1:
        st.markdown("**Predicted survival probability — summary**")
        stats_rows = [
            {"Arm": "Photon", "Mean": f"{p_ph.mean():.3f}", "SD": f"{p_ph.std():.3f}",
             "Min": f"{p_ph.min():.3f}", "Max": f"{p_ph.max():.3f}"},
            {"Arm": "Proton", "Mean": f"{p_pr.mean():.3f}", "SD": f"{p_pr.std():.3f}",
             "Min": f"{p_pr.min():.3f}", "Max": f"{p_pr.max():.3f}"},
        ]
        st.dataframe(pd.DataFrame(stats_rows), hide_index=True, use_container_width=True)

        # Histogram of predicted photon survival
        fig_hist = go.Figure(go.Histogram(
            x=p_ph, nbinsx=40, opacity=0.85,
        ))
        fig_hist.update_layout(
            title="Predicted photon survival probability",
            xaxis_title="Predicted P(survival)", yaxis_title="Count",
            height=280, margin=dict(t=45, b=45, l=55, r=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True, config=_CHART_CFG)

    # ── Model parameter table ──────────────────────────────────────────────────
    with col2:
        st.markdown("**Survival model parameters**")
        param_rows = [
            {"Parameter": "Intercept",    "Value": f"{intercept:.3f}"},
            {"Parameter": "GTV midpoint", "Value": f"{gtv_mid:.1f} cc"},
            {"Parameter": "GTV slope",    "Value": f"{gtv_slope:.4f}"},
            {"Parameter": "GTV weight",   "Value": f"{w_gtv:.2f}"},
            {"Parameter": "MHD midpoint", "Value": f"{mhd_mid:.1f} Gy"},
            {"Parameter": "MHD slope",    "Value": f"{mhd_slope:.4f}"},
            {"Parameter": "MHD weight",   "Value": f"{w_mhd:.2f}"},
        ]
        st.dataframe(pd.DataFrame(param_rows), hide_index=True, use_container_width=True)


# ── Extra exploration plots ───────────────────────────────────────────────────

def render_extra_plots(gtv, mhd, p_ph, p_pr, mhd_pr):
    st.divider()
    st.subheader("Additional Exploration")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Histogram(x=gtv, nbinsx=40, marker_color="#4C8BF5", opacity=0.85))
        fig.update_layout(
            title="GTV distribution",
            xaxis_title="GTV (cc)", yaxis_title="Count",
            height=320, margin=dict(t=45, b=45, l=55, r=20),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=mhd, nbinsx=40, name="Photon MHD",
            marker_color="#E8543A", opacity=0.75,
        ))
        fig.add_trace(go.Histogram(
            x=mhd_pr, nbinsx=40, name="Proton MHD",
            marker_color="#00C9A7", opacity=0.55,
        ))
        fig.update_layout(
            title="MHD distribution (photon vs proton)",
            xaxis_title="MHD (Gy)", yaxis_title="Count",
            barmode="overlay",
            height=320, margin=dict(t=45, b=45, l=55, r=20),
            legend=dict(orientation="h", y=1.12, x=0, xanchor="left", font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)

    # Survival probability vs MHD (10 equal-width bins)
    edges = np.linspace(mhd.min(), mhd.max(), 11)
    centers, p_ph_bin, p_pr_bin = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (mhd >= lo) & (mhd < hi)
        if mask.sum() > 0:
            centers.append(float((lo + hi) / 2))
            p_ph_bin.append(float(p_ph[mask].mean()))
            p_pr_bin.append(float(p_pr[mask].mean()))

    if centers:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=centers, y=p_ph_bin,
            mode="lines+markers", name="Photon",
            line=dict(color="#4C8BF5", width=2.5),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=centers, y=p_pr_bin,
            mode="lines+markers", name="Proton",
            line=dict(color="#00C9A7", width=2.5),
            marker=dict(size=8),
        ))
        fig.update_layout(
            title="Mean survival probability vs MHD (10 equal-width bins)",
            xaxis_title="MHD bin centre (Gy)",
            yaxis_title="Mean P(OS2y)",
            height=360,
            margin=dict(t=50, b=50, l=60, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, config=_CHART_CFG)


# ── App entry point ───────────────────────────────────────────────────────────

def main():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return
    except Exception:
        return

    st.set_page_config(
        page_title="NNT Simulation v2 — Proton vs Photon",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    for _k, _v in _DEFAULTS.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    n_patients, seed, proton_mode, hist_mode, noise_enabled, noise_sd, \
        z_enabled, z_mean, z_sd, z_beta = render_sidebar()

    gtv_mean      = float(st.session_state["gtv_mean"])
    gtv_std       = float(st.session_state["gtv_std"])
    mhd_mean      = float(st.session_state["mhd_mean"])
    mhd_std       = float(st.session_state["mhd_std"])
    intercept     = float(st.session_state["intercept"])
    gtv_mid       = float(st.session_state["gtv_mid"])
    gtv_slope     = float(st.session_state["gtv_slope"])
    mhd_mid       = float(st.session_state["mhd_mid"])
    mhd_slope     = float(st.session_state["mhd_slope"])
    proton_delta  = float(st.session_state["proton_delta"])
    proton_factor = float(st.session_state["proton_factor"])
    delta_thresh  = float(st.session_state["delta_thresh"])
    w_gtv         = float(st.session_state["w_gtv"])
    w_mhd         = float(st.session_state["w_mhd"])

    # ── Truth-world generation ────────────────────────────────────────────────
    gtv, mhd, p_ph, p_pr, pred_delta, true_delta, out_ph_base, out_ph, z_vals = run_simulation(
        n_patients, seed,
        gtv_mean, gtv_std, mhd_mean, mhd_std,
        intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope,
        proton_mode, proton_delta, proton_factor,
        noise_enabled, noise_sd,
        z_enabled, z_mean, z_sd, z_beta,
        w_gtv, w_mhd,
    )
    mhd_pr   = apply_proton_mhd(mhd, proton_mode, proton_delta, proton_factor)
    selected = pred_delta >= delta_thresh
    n_sel    = int(selected.sum())

    # ── Logistic regression fitting ───────────────────────────────────────────
    fit_result = fit_logistic_regression(gtv, mhd, out_ph)

    # ── Fitted photon and proton predictions ──────────────────────────────────
    if fit_result is not None:
        b0, b_gtv, b_mhd, gtv_mu, gtv_sig, mhd_mu, mhd_sig = fit_result
        gtv_z     = (gtv    - gtv_mu) / gtv_sig
        mhd_z_ph  = (mhd    - mhd_mu) / mhd_sig
        mhd_z_pr  = (mhd_pr - mhd_mu) / mhd_sig
        p_fit_ph  = logit_to_prob(b0 + b_gtv * gtv_z + b_mhd * mhd_z_ph)
        p_fit_pr  = logit_to_prob(b0 + b_gtv * gtv_z + b_mhd * mhd_z_pr)
        fit_delta = p_fit_pr - p_fit_ph
    else:
        b0 = b_gtv = b_mhd = None
        p_fit_ph = p_fit_pr = fit_delta = None

    st.title("Predicted vs True NNT — Proton vs Photon")
    st.markdown(
        "This app simulates a patient population to explore how well the "
        "**predicted NNT** (from survival probability differences) matches the "
        "**true NNT** (from Monte Carlo binary outcomes) when selecting patients "
        "for proton therapy. All parameters update instantly."
    )

    if fit_result is None:
        st.warning(
            "Logistic regression did not converge. "
            "Try adjusting the survival model parameters."
        )

    render_summary_cards(n_patients, n_sel, p_ph, p_pr, pred_delta, true_delta, selected)
    st.divider()
    render_playground(proton_mode, hist_mode, gtv, mhd, mhd_pr)
    st.divider()
    render_selection(pred_delta, true_delta, delta_thresh, selected, n_sel, noise_enabled)
    st.divider()
    # ── NNT analysis ─────────────────────────────────────────────────────────
    render_bin_analysis(pred_delta, true_delta, delta_thresh, n_sel, fit_delta)
    if noise_enabled:
        st.divider()
        render_noise_section(p_ph, out_ph_base, out_ph, noise_sd)
    if z_enabled:
        st.divider()
        render_z_section(z_vals, z_mean, z_sd, z_beta)
    if fit_result is not None:
        st.divider()
        render_fitted_model(
            b0, b_gtv, b_mhd,
            p_ph, p_pr, pred_delta,
            p_fit_ph, p_fit_pr, fit_delta,
            true_delta,
        )
    st.divider()
    render_model_diagnostics(
        p_ph, p_pr, out_ph_base, out_ph,
        noise_enabled, noise_sd,
        intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope, w_gtv, w_mhd,
        p_fit_ph=p_fit_ph,
    )
    render_extra_plots(gtv, mhd, p_ph, p_pr, mhd_pr)


main()
