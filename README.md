# CRM Simulator

Streamlit app for simulating a TITE-CRM dose-finding trial (`sim.py`).

## EWOC application modes

The sidebar "EWOC application" selector controls where the EWOC joint
overdose-control filter is applied during the CRM trial:

- **Dose assignment + final MTD** (default): EWOC filters the candidate
  doses at every cohort decision during the trial, and also filters the
  final MTD selection at study end. This is the original behavior.
- **Final MTD only**: cohort-by-cohort dose assignment during the trial
  ignores EWOC (doses are chosen by the standard argmin-to-target rule);
  EWOC is applied only once, when selecting the final MTD.
- **Off**: EWOC is never applied, neither during dose assignment nor at
  final MTD selection.

## METC amendment batch report

Run the reproducible METC amendment simulation grid with:

```bash
python metc_simulation_report.py --n-sim 200
```

The batch script uses the existing TITE-CRM engine and initializes every trial
with the current study state: six fully followed patients at L1 (5x5 Gy), zero
acute and subacute toxicity, and the first new cohort of three patients fixed at
L2 (5x6 Gy).  It evaluates five acute toxicity scenarios with the subacute probabilities held fixed, three EWOC application modes,
and burn-in on/off, then writes the HTML report, summary CSV, per-simulation CSV,
input probability/prior figures, and selected-MTD distribution figures grouped by EWOC/burn-in setting to `metc_outputs/`.

The METC batch uses the latest corrected prior clarification from Koen: the acute prior skeleton is `0.004, 0.021, 0.066, 0.150, 0.266`, so 0.150 is one dose below the highest dose level (L3, 5x7 Gy). The subacute prior from the prior revision was correct and remains unchanged. The fixed subacute probabilities are used across all five acute scenarios.
