#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# run_local.sh — launch the CRM simulator on localhost
# ---------------------------------------------------------------------------
# Usage:
#   chmod +x run_local.sh
#   ./run_local.sh
#
# Optional: pass extra streamlit arguments, e.g.
#   ./run_local.sh --server.port 8502
# ---------------------------------------------------------------------------

PYTHON="${PYTHON:-python3}"

# ── 1. Check Python ──────────────────────────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    echo "Error: '$PYTHON' not found. Install Python 3.9+ and retry." >&2
    exit 1
fi

PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 9 ) ]]; then
    echo "Error: Python 3.9 or newer is required (found $PY_MAJOR.$PY_MINOR)." >&2
    exit 1
fi

# ── 2. Install dependencies ───────────────────────────────────────────────────
echo "Installing dependencies from requirements.txt..."
"$PYTHON" -m pip install --quiet -r requirements.txt

# ── 3. Launch ─────────────────────────────────────────────────────────────────
echo "Starting CRM simulator at http://localhost:8501"
exec "$PYTHON" -m streamlit run sim.py \
    --server.headless true \
    "$@"
