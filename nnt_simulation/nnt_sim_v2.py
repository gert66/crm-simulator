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
# Each entry: key → default value

_DEFAULTS: dict = {
    "gtv_mean":      50.0,
    "gtv_std":       20.0,
    "mhd_mean":      15.0,
    "mhd_std":        5.0,
    "intercept":     -1.5,
    "gtv_mid":       50.0,
    "gtv_slope":     -0.04,
    "mhd_mid":       15.0,
    "mhd_slope":     -0.30,
    "proton_delta":   5.0,
    "proton_factor":  0.5,
    "delta_thresh":   0.02,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Placeholder functions ─────────────────────────────────────────────────────

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

def _on_slider(key):
    pass

def _on_num(key, lo, hi):
    pass

def dual_param(label, key, lo, hi, step, fmt="%.2f", help_text=None):
    pass

def make_combined_plot(data_a, mean, std, sig_mid, sig_slope,
                       x_label, title, data_b=None,
                       label_a="Photon", label_b="Proton",
                       color_a="#4C8BF5", color_b="#00C9A7", color_sig="#E86510",
                       hist_mode="Histogram", n_bins=40):
    pass

def render_sidebar():
    pass

def render_summary_cards(n_patients, n_sel, pred_delta, true_delta, selected):
    pass

def render_playground(gtv, mhd, mhd_pr, p_ph, hist_mode, proton_mode):
    pass

def render_selection(pred_delta, true_delta, delta_thresh, selected, n_sel):
    pass

def render_bin_analysis(pred_delta, true_delta, delta_thresh, n_sel):
    pass
