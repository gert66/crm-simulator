from __future__ import annotations
import streamlit as st

DEFAULTS = {
    # Essentials
    "target": 0.15,
    "start_dose": 1,          # 0-based index into dose labels
    "crm_max_n": 27,
    "crm_cohort": 3,
    "sixplus3_max_n": 27,     # you requested default 27
    "n_sims": 500,
    "seed": 123,

    # True curve (editable)
    "true_p": [0.01, 0.02, 0.12, 0.20, 0.35],
    "edit_true": True,

    # Skeleton / prior playground
    "skeleton_model": "empiric",   # or "logistic"
    "prior_target": 0.15,
    "delta": 0.10,
    "prior_mtd_nu": 3,             # 1-based “nu”
    "logit_intercept": 3.0,        # only used if logistic skeleton

    # CRM knobs
    "sigma_theta": 1.0,
    "burn_in_first_dlt": False,
    "ewoc": False,
    "ewoc_alpha": 0.25,
}

DOSE_LABELS = ["L0 (5×4 Gy)", "L1 (5×5 Gy)", "L2 (5×6 Gy)", "L3 (5×7 Gy)", "L4 (5×8 Gy)"]


def ensure_state():
    """
    Only set defaults if the key does not exist yet.
    This prevents resets on every rerun (the bug you saw).
    """
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "results" not in st.session_state:
        st.session_state["results"] = None


def reset_all():
    """
    Safe reset: overwrite keys, then rerun.
    """
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["results"] = None
    st.rerun()
