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
