# CRM Simulator

A Streamlit app for simulating TITE dual-endpoint dose-escalation trials with
TITE-CRM and TITE 6+3 designs.

## Running locally

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/gert66/crm-simulator.git
cd crm-simulator

# 2. (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Launch
chmod +x run_local.sh
./run_local.sh
```

`run_local.sh` installs all dependencies from `requirements.txt` and opens the
app at **http://localhost:8501**.

### Manual launch (without the script)

```bash
pip install -r requirements.txt
streamlit run sim.py
```

### Changing the port

```bash
./run_local.sh --server.port 8502
# or
streamlit run sim.py --server.port 8502
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit>=1.31` | UI framework |
| `numpy` | Numerical computation |
| `pandas` | Data tables |
| `matplotlib` | Charts and figures |
