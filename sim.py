import streamlit as st
from core import DEFAULTS, DEFAULT_TRUE_P, DOSE_LABELS

st.set_page_config(page_title="CRM Simulator", layout="wide")

def _init_state():
    # defaults
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    # true curve
    for i in range(len(DOSE_LABELS)):
        st.session_state.setdefault(f"true_{i}", float(DEFAULT_TRUE_P[i]))
    # prior playground defaults (kept in session_state so Playground page can read)
    st.session_state.setdefault("prior_model", "empiric")
    st.session_state.setdefault("prior_target", 0.15)
    st.session_state.setdefault("halfwidth", 0.10)
    st.session_state.setdefault("prior_nu", 3)
    st.session_state.setdefault("logistic_intcpt", 3.0)

def _reset_to_defaults(keep_true_curve=True, keep_prior=True):
    # IMPORTANT: use clear+reseed to avoid Streamlit "cannot set after widget" issues
    current_true = {f"true_{i}": st.session_state.get(f"true_{i}", DEFAULT_TRUE_P[i]) for i in range(len(DOSE_LABELS))}
    current_prior = {
        "prior_model": st.session_state.get("prior_model", "empiric"),
        "prior_target": st.session_state.get("prior_target", 0.15),
        "halfwidth": st.session_state.get("halfwidth", 0.10),
        "prior_nu": st.session_state.get("prior_nu", 3),
        "logistic_intcpt": st.session_state.get("logistic_intcpt", 3.0),
    }

    st.session_state.clear()
    _init_state()

    if keep_true_curve:
        for k, v in current_true.items():
            st.session_state[k] = float(v)

    if keep_prior:
        for k, v in current_prior.items():
            st.session_state[k] = v

def essentials_page():
    st.title("Essentials")

    c1, c2 = st.columns([1.0, 0.35])
    with c1:
        st.caption("Study + simulation settings. Open Playground in another window for tuning priors and running.")
    with c2:
        st.button(
            "Reset to R defaults",
            help="Resets all tunable settings to code defaults. True curve and prior playground stay as-is.",
            on_click=_reset_to_defaults,
            kwargs={"keep_true_curve": True, "keep_prior": True},
            use_container_width=True,
        )

    st.divider()

    # keep essentials compact by using columns
    a, b, c, d = st.columns([1, 1, 1, 1], gap="large")

    with a:
        st.number_input(
            "Study target (acute)",
            min_value=0.05, max_value=0.50, step=0.01,
            key="target",
            help=f"Default from code: {DEFAULTS['target']}"
        )
        st.selectbox(
            "Start dose",
            options=list(range(0, len(DOSE_LABELS))),
            format_func=lambda i: f"Level {i} ({DOSE_LABELS[i]})",
            key="start_level",
            help=f"Default from code: {DEFAULTS['start_level']} (0-based)"
        )

    with b:
        st.number_input(
            "Max sample size (CRM)",
            min_value=6, max_value=200, step=3,
            key="max_n_crm",
            help=f"Default from code: {DEFAULTS['max_n_crm']}"
        )
        st.number_input(
            "CRM cohort size",
            min_value=1, max_value=12, step=1,
            key="cohort_size",
            help=f"Default from code: {DEFAULTS['cohort_size']}"
        )

    with c:
        st.number_input(
            "Number of simulated trials",
            min_value=50, max_value=5000, step=50,
            key="n_sims",
            help=f"Default from code: {DEFAULTS['n_sims']}"
        )
        st.number_input(
            "Random seed",
            min_value=1, max_value=10_000_000, step=1,
            key="seed",
            help=f"Default from code: {DEFAULTS['seed']}"
        )

    with d:
        st.selectbox(
            "Gauss–Hermite points",
            options=[31, 41, 61, 81],
            key="gh_n",
            help=f"Default from code: {DEFAULTS['gh_n']}"
        )
        st.selectbox(
            "Max dose step per CRM update",
            options=[1, 2],
            key="max_step",
            help=f"Default from code: {DEFAULTS['max_step']}"
        )

    st.divider()

    e1, e2, e3, e4 = st.columns([1, 1, 1, 1], gap="large")
    with e1:
        st.slider(
            "Max sample size (6+3)",
            min_value=6, max_value=200, step=3,
            key="max_n_63",
            help=f"Default from code: {DEFAULTS['max_n_63']}"
        )
    with e2:
        st.selectbox(
            "6+3 accept rule after expansion to 9",
            options=[1, 2],
            key="accept_rule_63",
            help=f"Default from code: {DEFAULTS['accept_rule_63']}"
        )
    with e3:
        st.toggle(
            "Guardrail: next dose ≤ highest tried + 1",
            key="guardrail",
            help=f"Default from code: {DEFAULTS['guardrail']}"
        )
        st.toggle(
            "Final MTD must be among tried doses",
            key="final_tried_only",
            help=f"Default from code: {DEFAULTS['final_tried_only']}"
        )
    with e4:
        st.toggle(
            "Burn-in until first DLT (R-like)",
            key="burn_in",
            help=f"Default from code: {DEFAULTS['burn_in']}"
        )
        st.toggle(
            "Enable EWOC overdose control",
            key="ewoc_on",
            help=f"Default from code: {DEFAULTS['ewoc_on']}"
        )
        st.slider(
            "EWOC alpha",
            min_value=0.05, max_value=0.99, step=0.01,
            key="ewoc_alpha",
            disabled=(not st.session_state.get("ewoc_on", False)),
            help=f"Default from code: {DEFAULTS['ewoc_alpha']}"
        )

    with st.expander("Current settings (from code)"):
        st.json(DEFAULTS)

# -------------------------
# Router (sidebar labels!)
# -------------------------

_init_state()

pages = [
    st.Page(essentials_page, title="Essentials", icon="⚙️"),
    st.Page("pages/playground.py", title="Playground", icon="🧪"),
]
nav = st.navigation(pages, position="sidebar")
nav.run()
