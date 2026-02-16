# simplesim.py
from __future__ import annotations

import numpy as np


def run(payload: dict) -> dict:
    """
    Temporary stub so the Streamlit app can run.
    Replace the internals with your real simulator later.

    Expected output keys:
      - mtd_probs_6p3
      - mtd_probs_crm
      - avg_n_per_dose_6p3
      - avg_n_per_dose_crm
      - p_dlt_per_patient_6p3
      - p_dlt_per_patient_crm
    """
    dose_labels = payload.get("dose_labels", ["L0", "L1", "L2", "L3", "L4"])
    n = len(dose_labels)

    rng = np.random.default_rng(int(payload.get("seed", 123)))

    # Fake but valid probability vectors
    a = rng.random(n); a = a / a.sum()
    b = rng.random(n); b = b / b.sum()

    # Fake but plausible averages
    avg_6p3 = rng.integers(low=0, high=8, size=n).astype(float)
    avg_crm = rng.integers(low=0, high=12, size=_
