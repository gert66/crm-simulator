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

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NNT Simulation v2 — Proton vs Photon",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Default values & session state init ──────────────────────────────────────
# All interactive parameters live here; widgets are wired to these keys.

_DEFAULTS = {
    # Population
    "n_patients":     5_000,
    "seed":           42,
    # GTV distribution
    "gtv_mean":       50.0,
    "gtv_std":        20.0,
    # MHD distribution
    "mhd_mean":       15.0,
    "mhd_std":         5.0,
    # Survival model
    "intercept":      -1.5,
    "gtv_mid":        50.0,
    "gtv_slope":      -0.04,
    "mhd_mid":        15.0,
    "mhd_slope":      -0.30,
    # Proton effect
    "proton_mode":    "Multiply by factor",
    "proton_delta":    5.0,
    "proton_factor":   0.5,
    # Patient selection
    "delta_thresh":    0.02,
    # Display
    "hist_mode":      "Histogram",
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Math helpers (placeholders — filled in next step) ────────────────────────

def sigmoid(x, midpoint, slope):
    pass

def logit_to_prob(logit):
    pass

def survival_prob(gtv, mhd, intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope):
    pass

def sample_truncnorm(rng, mean, std, n, lower=0.0):
    pass

def truncnorm_pdf(x_arr, mean, std, lower=0.0):
    pass

def apply_proton_mhd(mhd, mode, delta, factor):
    pass

def bin_analysis(pred_delta, true_delta, threshold):
    pass

def run_simulation(n, seed, gtv_mean, gtv_std, mhd_mean, mhd_std,
                   intercept, gtv_mid, gtv_slope, mhd_mid, mhd_slope,
                   proton_mode, proton_delta, proton_factor):
    pass

# ── Dual-input widget: slider + number input, synced via session_state ────────

def _on_slider(key):
    """Slider changed → push its value to the master key and the number input."""
    v = st.session_state[f"_sl_{key}"]
    st.session_state[key] = v
    st.session_state[f"_nu_{key}"] = v


def _on_num(key, lo, hi):
    """Number input changed → push its value to the master key and the slider."""
    v = float(st.session_state[f"_nu_{key}"])
    st.session_state[key] = v
    st.session_state[f"_sl_{key}"] = float(np.clip(v, lo, hi))


def dual_param(label, key, lo, hi, step, fmt="%.2f", help_text=None):
    """
    Render a slider and a number input that stay in sync through
    session_state[key].  Returns the current float value.
    """
    sl_key = f"_sl_{key}"
    nu_key = f"_nu_{key}"
    current = float(st.session_state[key])

    # Initialise widget-level state on first render.
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


# ── Plot placeholder ──────────────────────────────────────────────────────────

def make_combined_plot(data_a, mean, std, sig_mid, sig_slope,
                       x_label, title, data_b=None,
                       label_a="Photon", label_b="Proton",
                       color_a="#4C8BF5", color_b="#00C9A7", color_sig="#E86510",
                       hist_mode="Histogram", n_bins=40):
    pass


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    """
    Static setup controls that do not need a slider.
    Returns (n_patients, seed, proton_mode, hist_mode) read from session_state.
    """
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

    return (
        int(st.session_state["n_patients"]),
        int(st.session_state["seed"]),
        st.session_state["proton_mode"],
        st.session_state["hist_mode"],
    )


# ── Summary cards (placeholder) ───────────────────────────────────────────────

def render_summary_cards(n_patients, n_sel, pred_delta, true_delta, selected):
    pass


# ── Model playground: dual inputs + (later) combined plots ───────────────────

def render_playground(proton_mode, hist_mode):
    """
    Interactive parameter panel.  Dual inputs update session_state immediately;
    combined plots will be added here in the next step.
    """
    st.subheader("Model Playground")
    st.markdown(
        "Adjust any parameter with the slider for quick exploration or type an "
        "exact value in the box on the right. All plots update on every change."
    )

    col_gtv, col_mhd = st.columns(2)

    # ── GTV column ────────────────────────────────────────────────────────────
    with col_gtv:
        st.markdown("#### GTV")
        st.caption(
            "Gross tumour volume contributes to survival through a sigmoid "
            "on the logit scale."
        )

        # Plot slot — combined distribution + sigmoid inserted here next step.
        st.empty()

        st.markdown("**Distribution**")
        dual_param("Mean (cc)",  "gtv_mean",  0.0, 200.0, 1.0,  "%.1f")
        dual_param("Std  (cc)",  "gtv_std",   0.5,  80.0, 0.5,  "%.1f")

        st.markdown("**Sigmoid**")
        dual_param("Midpoint (cc)", "gtv_mid",   0.0, 200.0, 1.0,  "%.1f")
        dual_param(
            "Slope", "gtv_slope", -1.0, 1.0, 0.01, "%.3f",
            help_text="Negative → larger GTV reduces survival.",
        )

    # ── MHD column ────────────────────────────────────────────────────────────
    with col_mhd:
        st.markdown("#### MHD")
        st.caption(
            "Mean heart dose is reduced by proton therapy. "
            "The teal overlay shows the proton MHD distribution."
        )

        # Plot slot — combined distribution + sigmoid inserted here next step.
        st.empty()

        st.markdown("**Distribution**")
        dual_param("Mean (Gy)", "mhd_mean",  0.0, 60.0, 0.5,  "%.1f")
        dual_param("Std  (Gy)", "mhd_std",   0.5, 25.0, 0.5,  "%.1f")

        st.markdown("**Sigmoid**")
        dual_param("Midpoint (Gy)", "mhd_mid",   0.0, 60.0,  0.5,  "%.1f")
        dual_param(
            "Slope", "mhd_slope", -2.0, 2.0, 0.05, "%.3f",
            help_text="Negative → higher MHD reduces survival.",
        )

    # ── Intercept & proton reduction (full width) ─────────────────────────────
    st.markdown("**Baseline (intercept)**")
    st.caption(
        "Sets the log-odds of survival when both sigmoids are at their midpoints. "
        "More negative means lower baseline survival."
    )
    dual_param("Intercept (logit scale)", "intercept", -6.0, 2.0, 0.05, "%.2f")

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

def render_selection(pred_delta, true_delta, delta_thresh, selected, n_sel):
    """Selection threshold control.  Histograms added in the next step."""
    st.subheader("Patient Selection")
    st.markdown(
        "Patients are offered proton therapy when their predicted survival benefit "
        "meets or exceeds the threshold below."
    )
    dual_param(
        "Predicted Δ threshold",
        "delta_thresh",
        lo=0.0, hi=0.50, step=0.005, fmt="%.3f",
    )
    # Histogram slot — added next step.


# ── Bin analysis (placeholder) ────────────────────────────────────────────────

def render_bin_analysis(pred_delta, true_delta, delta_thresh, n_sel):
    pass


# ── App entry point ───────────────────────────────────────────────────────────

n_patients, seed, proton_mode, hist_mode = render_sidebar()

st.title("Predicted vs True NNT — Proton vs Photon")
st.markdown(
    "This app simulates a patient population to explore how well the "
    "**predicted NNT** (from survival probability differences) matches the "
    "**true NNT** (from Monte Carlo binary outcomes) when selecting patients "
    "for proton therapy. All parameters update instantly."
)

st.divider()
render_playground(proton_mode, hist_mode)
st.divider()
render_selection(None, None, None, None, None)
st.divider()
render_bin_analysis(None, None, None, None)
