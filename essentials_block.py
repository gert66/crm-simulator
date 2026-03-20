with st.sidebar:
    st.markdown("#### Study endpoints")
    st.number_input(
        "Target tox1 (acute) rate",
        min_value=0.05, max_value=0.50, step=0.01, key="target_t1",
        help=h("target_t1", "Target acute DLT probability for MTD definition.")
    )
    st.number_input(
        "Target tox2 (subacute | surgery) rate",
        min_value=0.05, max_value=0.50, step=0.01, key="target_t2",
        help=h("target_t2",
               "Target subacute DLT probability conditional on surgery. "
               "Only surgery patients contribute to the tox2 model.")
    )
    st.number_input(
        "Probability of surgery",
        min_value=0.0, max_value=1.0, step=0.01, key="p_surgery",
        help=h("p_surgery",
               "Global probability that a patient proceeds to surgery. "
               "Dose-independent. Subacute toxicity only observed in these patients.")
    )
    st.number_input(
        "Start dose level (1-based)",
        min_value=1, max_value=5, step=1, key="start_level_1b",
        help=h("start_level_1b", "Starting dose level (1 = lowest).")
    )
    st.markdown("#### Simulation")
    st.number_input(
        "Number of simulated trials",
        min_value=50, max_value=5000, step=50, key="n_sims",
        help=h("n_sims", "Replicates for the simulation study.")
    )
    st.number_input(
        "Random seed",
        min_value=1, max_value=10_000_000, step=1, key="seed",
        help=h("seed", "Random seed for reproducibility.")
    )
    st.number_input(
        "Avg patients per month",
        min_value=0.1, max_value=20.0, step=0.1, key="accrual_per_month",
        help=h("accrual_per_month",
               "Average accrual rate. Arrivals simulated as a Poisson process "
               "(exponential inter-arrival times at this rate).")
    )

    st.markdown("#### Timing (days)")
    st.number_input(
        "Inclusion to RT start",
        min_value=0, max_value=180, step=1, key="incl_to_rt",
        help=h("incl_to_rt",
               "Days from enrolment to start of radiotherapy. "
               "Tox1 window begins at RT start. Default ≈ 3 weeks.")
    )
    st.number_input(
        "Radiotherapy duration",
        min_value=1, max_value=60, step=1, key="rt_dur",
        help=h("rt_dur",
               "Duration of radiotherapy in days. Default ≈ 2 weeks (10 fractions).")
    )
    st.number_input(
        "RT end to surgery",
        min_value=1, max_value=365, step=1, key="rt_to_surg",
        help=h("rt_to_surg",
               "Days from end of radiotherapy to surgery. Default 84 days ≈ 12 weeks. "
               "The tox1 (acute) follow-up window is derived as RT duration + this value, "
               "so it always extends from RT start to the moment of surgery.")
    )
    st.number_input(
        "Tox2 follow-up window (days)",
        min_value=7, max_value=180, step=1, key="tox2_win",
        help=h("tox2_win",
               "Post-surgery window for subacute toxicity assessment. Default 30 days.")
    )
    st.markdown("#### Sample size")
    st.number_input(
        "Max sample size (6+3)",
        min_value=6, max_value=200, step=3, key="max_n_63",
        help=h("max_n_63",
               "Maximum total enrolled patients in the 6+3 arm, including "
               "bridging patients treated at lower doses while awaiting evaluability.")
    )
    st.number_input(
        "Max sample size (CRM)",
        min_value=6, max_value=200, step=3, key="max_n_crm",
        help=h("max_n_crm", "Maximum total enrolled patients in the TITE-CRM arm.")
    )
    st.number_input(
        "Cohort size (CRM)",
        min_value=1, max_value=12, step=1, key="cohort_size",
        help=h("cohort_size",
               "Number of patients per CRM cohort. CRM updates after each "
               "cohort is fully enrolled, using TITE weights at that moment.")
    )

    st.markdown("#### CRM integration")
    st.selectbox(
        "Gauss–Hermite points",
        options=[31, 41, 61, 81], key="gh_n",
        help=h("gh_n",
               "Quadrature points for CRM posterior. Higher = more accurate, slower.")
    )
    st.selectbox(
        "Max dose step per update",
        options=[1, 2], key="max_step",
        help=h("max_step",
               "Max dose levels the CRM can move per cohort update.")
    )
    st.slider(
        "Prior sigma on theta",
        min_value=0.2, max_value=5.0, step=0.1, key="sigma",
        help=h("sigma",
               "SD of theta in the CRM prior (shared for both endpoints). "
               "Larger = more diffuse prior.",
               r_name="prior.sigma / sigma")
    )
    st.markdown("#### CRM safety / selection")
    st.toggle(
        "Guardrail: next dose ≤ highest tried + 1",
        key="enforce_guardrail",
        help=h("enforce_guardrail", "Prevent skipping untried dose levels.")
    )
    st.toggle(
        "Final MTD must be among tried doses",
        key="restrict_final_mtd",
        help=h("restrict_final_mtd",
               "Restrict final MTD selection to doses where n > 0.")
    )
    st.markdown("#### CRM behaviour")
    st.toggle(
        "Burn-in until first tox1 DLT",
        key="burn_in",
        help=h("burn_in",
               "Escalate one level at a time until the first observed acute DLT, "
               "then switch to CRM updates.")
    )
    st.toggle(
        "Enable EWOC joint overdose control",
        key="ewoc_on",
        help=h("ewoc_on",
               "Restrict doses where BOTH P(tox1 OD) and P(tox2 OD) < EWOC alpha.")
    )
    st.number_input(
        "EWOC alpha",
        min_value=0.01, max_value=0.99, step=0.01, key="ewoc_alpha",
        disabled=(not bool(st.session_state["ewoc_on"])),
        help=h("ewoc_alpha",
               "EWOC threshold applied to both endpoints independently.")
    )
    st.markdown("#### CRM decision trace")
    st.toggle(
        "Explain first CRM trial",
        key="show_crm_trace",
        help=h("show_crm_trace",
               "When ON, shows a detailed walkthrough for the first simulated "
               "CRM trial only: which dose each patient received, what follow-up "
               "data were available at each decision point, how the model judged "
               "safety for each dose level, and why the next dose was chosen. "
               "Has no effect on the summary results across all simulated trials.")
    )

    st.markdown("#### 6+3 stopping rules")
    st.info(
        "**Modified 6+3 (TITE version) — full evaluability required.**\n\n"
        "Decisions are only made once ALL enrolled patients in the evaluation "
        "cohort have completed their relevant follow-up windows.\n\n"
        "**Bridging rule:** while waiting for evaluability at the current "
        "dose, new arrivals are assigned to the next lower dose (*safe dose*). "
        "These bridging patients count toward the trial total but not toward "
        "the formal evaluation cohort.\n\n"
        "**Rate-based acute thresholds:** if the HOLD rule causes more than 6 "
        "(or 9) patients to be enrolled at eval dose, the acute threshold "
        "is scaled proportionally to preserve the original protocol ratio.",
        icon="ℹ️",
    )
    st.markdown(
        "<div style='font-size:0.79rem;font-weight:600;color:#555;'>"
        "Acute thresholds</div>",
        unsafe_allow_html=True,
    )
    _ar1, _ar2, _ar3 = st.columns(3, gap="small")
    with _ar1:
        st.number_input("≥6 — esc if tox1 ≤", min_value=0, max_value=5,
                        step=1, key="a6_esc_max",
                        help=h("a6_esc_max", "Phase 1 acute escalation threshold."))
    with _ar2:
        st.number_input("≥6 — stop if tox1 ≥", min_value=1, max_value=6,
                        step=1, key="a6_stop_min",
                        help=h("a6_stop_min", "Phase 1 acute stopping threshold."))
    with _ar3:
        st.number_input("≥9 — esc if tox1 ≤", min_value=0, max_value=8,
                        step=1, key="a9_esc_max",
                        help=h("a9_esc_max", "Phase 2 acute escalation threshold."))

    st.markdown(
        "<div style='font-size:0.79rem;font-weight:600;color:#555;margin-top:0.3rem;'>"
        "Subacute thresholds</div>",
        unsafe_allow_html=True,
    )
    _sr1, _sr2, _sr3, _sr4 = st.columns(4, gap="small")
    with _sr1:
        st.number_input("≥6 surg — esc if tox2 ≤", min_value=0, max_value=6,
                        step=1, key="s6_esc_max",
                        help=h("s6_esc_max", "Phase 1 subacute escalation threshold."))
    with _sr2:
        st.number_input("≥6 surg — stop if tox2 ≥", min_value=1, max_value=6,
                        step=1, key="s6_stop_min",
                        help=h("s6_stop_min", "Phase 1 subacute stopping threshold."))
    with _sr3:
        st.number_input("≥9 surg — esc if tox2 ≤", min_value=0, max_value=9,
                        step=1, key="s9_esc_max",
                        help=h("s9_esc_max", "Phase 2 subacute escalation threshold."))
    with _sr4:
        st.number_input("≥9 surg — stop if tox2 ≥", min_value=1, max_value=9,
                        step=1, key="s9_stop_min",
                        help=h("s9_stop_min", "Phase 2 subacute stopping threshold."))

    st.write("")
    st.button("Reset to defaults", on_click=_do_reset)

    # ── Timeline: rendered inside Essentials, spans full width below columns ──
    st.markdown(
        "<div style='font-size:0.78rem;font-weight:600;color:#555;margin-top:0.3rem;'>"
        "Patient timeline (based on current timing settings)</div>",
        unsafe_allow_html=True,
    )
    _tl_fig = _draw_timeline(
        int(st.session_state["incl_to_rt"]),
        int(st.session_state["rt_dur"]),
        int(st.session_state["rt_to_surg"]),
        int(st.session_state["tox2_win"]),
    )
    st.image(fig_to_png_bytes(_tl_fig), use_container_width=True)
