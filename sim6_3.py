import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# 6+3 design core
# ----------------------------

def simulate_one_trial(
    true_p,
    start_dose=0,
    cohort_init=6,
    cohort_expand=3,
    max_per_dose=9,
    no_skip=True,
    seed=1,
    stop_if_dose1_unsafe=True,
):
    """
    Returns:
      final_dose (0-based int or -1 if stopped),
      alloc (array K),
      dlts (array K),
      stopped (bool)
    """
    rng = np.random.default_rng(seed)
    true_p = np.asarray(true_p, float)
    K = len(true_p)

    alloc = np.zeros(K, dtype=int)
    dlts = np.zeros(K, dtype=int)

    current = int(start_dose)
    stopped = False

    # helper to treat n patients at current dose
    def treat_at(dose_idx, n_patients):
        y = rng.binomial(1, true_p[dose_idx], size=n_patients)
        alloc[dose_idx] += n_patients
        dlts[dose_idx] += int(np.sum(y))

    # run until we cannot escalate further
    while True:
        # Safety: stop if current is out of range
        if current < 0 or current >= K:
            stopped = True
            return -1, alloc, dlts, True

        # Treat initial cohort if not already treated at this dose
        already = alloc[current]
        if already == 0:
            treat_at(current, cohort_init)

        # Evaluate after 6
        dlt_6 = dlts[current]
        n_6 = alloc[current]  # should be 6 here if first time

        # In case someone changes cohort_init, keep logic based on n_6
        if n_6 < cohort_init:
            # should not happen, but safe guard
            treat_at(current, cohort_init - n_6)
            dlt_6 = dlts[current]
            n_6 = alloc[current]

        # Decision after initial cohort (default assumes cohort_init=6)
        if dlt_6 == 0:
            # escalate
            next_dose = current + 1
        elif dlt_6 == 1:
            # expand to 9 total (6+3)
            if alloc[current] < max_per_dose:
                treat_at(current, min(cohort_expand, max_per_dose - alloc[current]))

            dlt_9 = dlts[current]
            n_9 = alloc[current]  # should be 9

            # Decision after expansion
            if dlt_9 <= 1:
                next_dose = current + 1
            else:
                next_dose = current - 1
        else:
            # >=2 in initial cohort
            next_dose = current - 1

        # Safety stop at dose 1 (index 0) if unsafe and we would de-escalate below 0
        if next_dose < 0:
            if stop_if_dose1_unsafe:
                stopped = True
                return -1, alloc, dlts, True
            next_dose = 0

        # If next dose is same or lower, move there
        # If next dose is above max, we are done: recommend current (or best acceptable)
        if next_dose >= K:
            # reached above top, recommend current as highest tested dose
            return current, alloc, dlts, False

        # No skipping rule (usually irrelevant because step is 1)
        if no_skip:
            if next_dose > current + 1:
                next_dose = current + 1

        # If we are about to revisit a dose that already has decisions completed,
        # we still allow it (common in 3+3 family). In practice, we often stop once
        # de-escalation happens. You can choose either behavior.
        # Here: if we de-escalate, we stop and recommend the next_dose (more classic).
        if next_dose < current:
            return next_dose, alloc, dlts, False

        # Otherwise escalate and continue
        current = next_dose


def run_sims(
    n_sims,
    true_p,
    start_dose=0,
    cohort_init=6,
    cohort_expand=3,
    max_per_dose=9,
    no_skip=True,
    seed=1,
    stop_if_dose1_unsafe=True,
):
    rng = np.random.default_rng(seed)
    K = len(true_p)

    finals = np.zeros(n_sims, dtype=int)
    alloc_mat = np.zeros((n_sims, K), dtype=int)
    dlt_mat = np.zeros((n_sims, K), dtype=int)
    stopped = np.zeros(n_sims, dtype=bool)

    for i in range(n_sims):
        trial_seed = int(rng.integers(1, 2_000_000_000))
        final, alloc, dlts, did_stop = simulate_one_trial(
            true_p=true_p,
            start_dose=start_dose,
            cohort_init=cohort_init,
            cohort_expand=cohort_expand,
            max_per_dose=max_per_dose,
            no_skip=no_skip,
            seed=trial_seed,
            stop_if_dose1_unsafe=stop_if_dose1_unsafe,
        )
        finals[i] = final
        alloc_mat[i, :] = alloc
        dlt_mat[i, :] = dlts
        stopped[i] = did_stop

    # Convert finals: -1 (stopped) keep as -1
    final_dist = np.zeros(K, dtype=float)
    for k in range(K):
        final_dist[k] = np.mean(finals == k)

    res = {
        "final_dist": final_dist,
        "mean_alloc": np.mean(alloc_mat, axis=0),
        "mean_dlts": float(np.mean(np.sum(dlt_mat, axis=1))),
        "p_stopped": float(np.mean(stopped)),
        "mean_total_n": float(np.mean(np.sum(alloc_mat, axis=1))),
    }
    return res


def parse_list(txt):
    vals = [float(x.strip()) for x in txt.split(",") if x.strip() != ""]
    return vals


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="6+3 dose-escalation simulator", layout="wide")
st.title("6+3 dose-escalation simulator")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Design inputs")
    start_dose_1b = st.number_input("Starting dose level (1-based)", min_value=1, value=1, step=1)
    cohort_init = st.number_input("Initial cohort size", min_value=3, max_value=12, value=6, step=1)
    cohort_expand = st.number_input("Expansion cohort size", min_value=1, max_value=6, value=3, step=1)
    max_per_dose = st.number_input("Max patients per dose", min_value=cohort_init, max_value=24, value=9, step=1)
    no_skip = st.checkbox("No dose skipping", value=True)
    stop_if_dose1_unsafe = st.checkbox("Stop trial if dose level 1 appears unsafe", value=True)

    st.subheader("Simulation controls")
    n_sims = st.number_input("Number of simulated trials", min_value=50, max_value=20000, value=1000, step=50)
    seed = st.number_input("Random seed", min_value=1, max_value=10_000_000, value=1, step=1)

with col2:
    st.subheader("True scenario")
    true_text = st.text_input(
        "True DLT probabilities per dose (comma-separated)",
        value="0.01,0.02,0.05,0.08,0.12"
    )

if st.button("Run simulation"):
    true_p = parse_list(true_text)

    if any(p <= 0 or p >= 1 for p in true_p):
        st.error("All probabilities must be between 0 and 1 (exclusive).")
        st.stop()

    K = len(true_p)
    start_dose = int(start_dose_1b) - 1
    if start_dose < 0 or start_dose >= K:
        st.error("Starting dose must be within the number of dose levels.")
        st.stop()

    with st.spinner("Running simulations..."):
        res = run_sims(
            n_sims=int(n_sims),
            true_p=true_p,
            start_dose=start_dose,
            cohort_init=int(cohort_init),
            cohort_expand=int(cohort_expand),
            max_per_dose=int(max_per_dose),
            no_skip=bool(no_skip),
            seed=int(seed),
            stop_if_dose1_unsafe=bool(stop_if_dose1_unsafe),
        )

    st.subheader("Key results")
    st.write(f"Probability of early stop: **{res['p_stopped']:.3f}**")
    st.write(f"Mean total N per trial: **{res['mean_total_n']:.2f}**")
    st.write(f"Mean number of DLTs per trial: **{res['mean_dlts']:.2f}**")

    st.subheader("Plots")
    x = np.arange(1, K + 1)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(x, res["final_dist"])
    ax1.set_xlabel("Final recommended dose level")
    ax1.set_ylabel("Proportion of trials")
    ax1.set_title("Final recommendation distribution")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(x, res["mean_alloc"])
    ax2.set_xlabel("Dose level")
    ax2.set_ylabel("Mean # patients per trial")
    ax2.set_title("Mean allocation per dose")
    st.pyplot(fig2)
