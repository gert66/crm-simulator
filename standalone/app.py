"""
app.py  —  Flask backend for the standalone TITE CRM Simulator.
Run with:  python app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template, Response
import json, datetime
import numpy as np

import sim_core as sc

app = Flask(__name__)

# ==============================================================================
# Global state  (single-user local app — no sessions needed)
# ==============================================================================

def _default_state():
    s = {}
    s.update(sc.R_DEFAULTS)
    s.update(sc._TRUE_DEFAULTS)
    s.update(sc.DE_DEFAULTS)
    s["_results"] = None
    return s

_STATE = _default_state()


# ==============================================================================
# Helpers
# ==============================================================================

def _g(key, default=None):
    return _STATE.get(key, default)


def _true_tox():
    t1 = np.array([float(_STATE[k]) for k in sc.TRUE_T1_KEYS])
    t2 = np.array([float(_STATE[k]) for k in sc.TRUE_T2_KEYS])
    return t1, t2


def _skeletons():
    skel_t1 = sc.dfcrm_getprior(
        float(_g("halfwidth_t1")), float(_g("prior_target_t1")),
        int(_g("prior_nu_t1")), 5,
        model=str(_g("prior_model", "empiric")),
        intcpt=float(_g("logistic_intcpt", 3.0)),
    )
    skel_t2 = sc.dfcrm_getprior(
        float(_g("halfwidth_t2")), float(_g("prior_target_t2")),
        int(_g("prior_nu_t2")), 5,
        model=str(_g("prior_model", "empiric")),
        intcpt=float(_g("logistic_intcpt", 3.0)),
    )
    return np.array(skel_t1), np.array(skel_t2)


def _build_base_ss():
    tox1_win = int(_g("rt_dur")) + int(_g("rt_to_surg"))
    return {
        "target_tox1": float(_g("target_t1")),
        "target_tox2": float(_g("target_t2")),
        "p_surgery":   float(_g("p_surgery")),
        "sigma":       float(_g("sigma")),
        "ewoc_on":     bool(_g("ewoc_on")),
        "ewoc_alpha":  float(_g("ewoc_alpha")),
        "max_n":       int(_g("max_n_crm")),
        "cohort_size": int(_g("cohort_size")),
        "start_level": int(np.clip(int(_g("start_level_1b")) - 1, 0, 4)),
        "accrual_per_month": float(_g("accrual_per_month")),
        "incl_to_rt":  int(_g("incl_to_rt")),
        "rt_dur":      int(_g("rt_dur")),
        "rt_to_surg":  int(_g("rt_to_surg")),
        "tox1_win":    tox1_win,
        "tox2_win":    int(_g("tox2_win")),
        "max_step":    int(_g("max_step")),
        "gh_n":        int(_g("gh_n")),
        "burn_in":     bool(_g("burn_in")),
        "enforce_guardrail":       bool(_g("enforce_guardrail")),
        "restrict_final_to_tried": bool(_g("restrict_final_mtd")),
        "prior_hw1":   float(_g("halfwidth_t1")),
        "prior_pt1":   float(_g("prior_target_t1")),
        "prior_hw2":   float(_g("halfwidth_t2")),
        "prior_pt2":   float(_g("prior_target_t2")),
        "prior_model_str": str(_g("prior_model", "empiric")),
        "logistic_intcpt": float(_g("logistic_intcpt", 3.0)),
    }


def _np_clean(obj):
    """Make numpy types JSON-serialisable (used for custom encoder)."""
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, np.ndarray):   return obj.tolist()
    if isinstance(obj, np.bool_):     return bool(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


def _state_json():
    """Serialisable copy of _STATE — drop internal _results."""
    out = {}
    for k, v in _STATE.items():
        if k == "_results":
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating, np.bool_)):
            out[k] = _np_clean(v)
        else:
            out[k] = v
    return out


# ==============================================================================
# Routes — pages
# ==============================================================================

@app.route("/")
def index():
    prior_scenarios = list(sc._PRIOR_SCENARIOS.keys())
    dose_labels     = sc.dose_labels
    return render_template("index.html",
                           prior_scenarios=prior_scenarios,
                           dose_labels=dose_labels)


# ==============================================================================
# Routes — API
# ==============================================================================

@app.route("/api/state")
def api_get_state():
    return jsonify(_state_json())


@app.route("/api/update", methods=["POST"])
def api_update():
    data = request.get_json(force=True)
    for k, v in data.items():
        _STATE[k] = v
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    _STATE.clear()
    _STATE.update(_default_state())
    return jsonify({"ok": True})


@app.route("/api/prior-scenario", methods=["POST"])
def api_prior_scenario():
    """Apply a named prior scenario and return its parameter values."""
    name = request.get_json(force=True).get("name", "Custom")
    sc_data = sc._PRIOR_SCENARIOS.get(name, {})
    applied = {}
    for k in ("prior_target_t1", "halfwidth_t1", "prior_nu_t1",
               "prior_target_t2", "halfwidth_t2", "prior_nu_t2"):
        if k in sc_data:
            _STATE[k] = sc_data[k]
            applied[k] = sc_data[k]
    _STATE["prior_scenario"] = name
    return jsonify({"ok": True, "applied": applied,
                    "description": sc_data.get("description", "")})


@app.route("/api/timeline")
def api_timeline():
    import matplotlib.pyplot as plt
    fig = sc._draw_timeline(
        int(_g("incl_to_rt")), int(_g("rt_dur")),
        int(_g("rt_to_surg")), int(_g("tox2_win")),
    )
    b64 = sc.fig_to_b64(fig)
    plt.close(fig)
    return jsonify({"img": b64})


@app.route("/api/preview")
def api_preview():
    true_t1, true_t2 = _true_tox()
    skel_t1, skel_t2 = _skeletons()
    b64 = sc.generate_preview_chart(
        true_t1, skel_t1, float(_g("target_t1")),
        true_t2, skel_t2, float(_g("target_t2")),
    )
    return jsonify({"img": b64})


@app.route("/api/run", methods=["POST"])
def api_run():
    true_t1, true_t2 = _true_tox()
    skel_t1, skel_t2 = _skeletons()
    p_surg   = float(_g("p_surgery"))
    tgt1     = float(_g("target_t1"))
    tgt2     = float(_g("target_t2"))
    ns       = int(_g("n_sims"))
    seed     = int(_g("seed"))
    start_0b = int(np.clip(int(_g("start_level_1b")) - 1, 0, 4))
    tox1_win = int(_g("rt_dur")) + int(_g("rt_to_surg"))

    timing_kw = dict(
        accrual_per_month=float(_g("accrual_per_month")),
        incl_to_rt=int(_g("incl_to_rt")),
        rt_dur=int(_g("rt_dur")),
        rt_to_surg=int(_g("rt_to_surg")),
        tox1_win=tox1_win,
        tox2_win=int(_g("tox2_win")),
    )

    rng_master = np.random.default_rng(seed)
    sel_63  = np.zeros(5, dtype=int)
    sel_crm = np.zeros(5, dtype=int)
    nmat_63  = np.zeros((ns, 5)); nmat_crm  = np.zeros((ns, 5))
    nsurg_63 = np.zeros((ns, 5)); nsurg_crm = np.zeros((ns, 5))
    ya63 = np.zeros(ns); ys63 = np.zeros(ns); ns63 = np.zeros(ns)
    yacrm= np.zeros(ns); yscrm= np.zeros(ns); nscrm= np.zeros(ns)
    dur_63 = np.zeros(ns); dur_crm = np.zeros(ns)
    nbridg = np.zeros(ns, dtype=int)
    crm_trace_first = None

    for s in range(ns):
        rng_s = np.random.default_rng(rng_master.integers(0, 2**31))
        sel63, pts63, sd63, nb63 = sc.run_tite_6plus3(
            true_t1=true_t1, p_surgery=p_surg, true_t2=true_t2,
            start_level=start_0b, max_n=int(_g("max_n_63")),
            a6_esc_max=int(_g("a6_esc_max")),   a6_stop_min=int(_g("a6_stop_min")),
            a9_esc_max=int(_g("a9_esc_max")),   s6_esc_max=int(_g("s6_esc_max")),
            s6_stop_min=int(_g("s6_stop_min")), s9_esc_max=int(_g("s9_esc_max")),
            s9_stop_min=int(_g("s9_stop_min")), rng=rng_s, **timing_kw,
        )
        sel_63[sel63] += 1
        for p in pts63:
            nmat_63[s, p["dose"]]  += 1
            nsurg_63[s, p["dose"]] += int(p["has_surgery"])
            ya63[s] += int(p["has_tox1"]); ys63[s] += int(p["has_tox2"])
            ns63[s] += int(p["has_surgery"])
        dur_63[s] = sd63; nbridg[s] = nb63

        rng_s2 = np.random.default_rng(rng_master.integers(0, 2**31))
        selc, ptsc, sdc, trace_s = sc.run_tite_crm(
            true_t1=true_t1, p_surgery=p_surg, true_t2=true_t2,
            target1=tgt1, target2=tgt2, skel1=skel_t1, skel2=skel_t2,
            sigma=float(_g("sigma")),    start_level=start_0b,
            max_n=int(_g("max_n_crm")), cohort_size=int(_g("cohort_size")),
            max_step=int(_g("max_step")), gh_n=int(_g("gh_n")),
            enforce_guardrail=bool(_g("enforce_guardrail")),
            restrict_final_to_tried=bool(_g("restrict_final_mtd")),
            ewoc_on=bool(_g("ewoc_on")), ewoc_alpha=float(_g("ewoc_alpha")),
            burn_in=bool(_g("burn_in")), rng=rng_s2,
            collect_trace=(s == 0), **timing_kw,
        )
        if s == 0:
            crm_trace_first = {
                "patients":  ptsc, "decisions": trace_s,
                "true_t1":   true_t1.tolist(), "true_t2": true_t2.tolist(),
                "tox1_win":  tox1_win, "tox2_win": int(_g("tox2_win")),
                "final_mtd": selc,     "study_days": sdc,
                "sigma":     float(_g("sigma")),
                "skel_t1":   skel_t1.tolist(), "skel_t2": skel_t2.tolist(),
                "target_t1": tgt1,     "target_t2": tgt2,
                "gh_n":      int(_g("gh_n")),
                "ewoc_on":   bool(_g("ewoc_on")),
                "ewoc_alpha":float(_g("ewoc_alpha")),
                "restrict_final_mtd": bool(_g("restrict_final_mtd")),
            }
        sel_crm[selc] += 1
        for p in ptsc:
            nmat_crm[s, p["dose"]]  += 1
            nsurg_crm[s, p["dose"]] += int(p["has_surgery"])
            yacrm[s] += int(p["has_tox1"]); yscrm[s] += int(p["has_tox2"])
            nscrm[s] += int(p["has_surgery"])
        dur_crm[s] = sdc

    p63   = sel_63  / ns
    pcrm  = sel_crm / ns
    true_safe = sc.find_true_safe_dose(true_t1, true_t2, tgt1, tgt2)

    res = {
        "p63": p63, "pcrm": pcrm,
        "avg_n63":      nmat_63.mean(axis=0),
        "avg_ncrm":     nmat_crm.mean(axis=0),
        "avg_nsurg63":  nsurg_63.mean(axis=0),
        "avg_nsurgcrm": nsurg_crm.mean(axis=0),
        "acute_rate_63":   float(ya63.sum()  / max(1, nmat_63.sum())),
        "acute_rate_crm":  float(yacrm.sum() / max(1, nmat_crm.sum())),
        "sub_gs_rate_63":  float(ys63.sum()  / max(1, ns63.sum())),
        "sub_gs_rate_crm": float(yscrm.sum() / max(1, nscrm.sum())),
        "surg_rate_63":  float(ns63.mean()  / max(1, nmat_63.sum() / ns)),
        "surg_rate_crm": float(nscrm.mean() / max(1, nmat_crm.sum() / ns)),
        "dur63_mean":   float(dur_63.mean()        / sc.MONTH),
        "dur63_med":    float(np.median(dur_63)    / sc.MONTH),
        "durcrm_mean":  float(dur_crm.mean()       / sc.MONTH),
        "durcrm_med":   float(np.median(dur_crm)   / sc.MONTH),
        "avg_bridging": float(nbridg.mean()),
        "true_safe": true_safe, "ns": ns, "seed": seed, "p_surgery": p_surg,
        "crm_trace": crm_trace_first,
    }
    _STATE["_results"] = res

    # ── Charts ──────────────────────────────────────────────────────────────
    charts = {
        "mtd":      sc.generate_mtd_chart(p63, pcrm, true_safe),
        "patients": sc.generate_patients_chart(res),
    }
    trace = crm_trace_first
    if trace and trace.get("decisions"):
        charts["posterior_tracking"]   = sc.generate_posterior_tracking_chart(trace)
        charts["study_end_posteriors"] = sc.generate_study_end_posteriors_chart(trace)
        charts["dose_eligibility"]     = sc.generate_dose_eligibility_chart(trace)
        charts["trace_dose"]   = sc.generate_trace_dose_chart(trace["decisions"])
        charts["trace_safety"] = sc.generate_trace_safety_chart(
            trace["decisions"], bool(_g("ewoc_on")), float(_g("ewoc_alpha")))
        charts["trace_tite"]   = sc.generate_trace_tite_chart(trace["decisions"])

    # ── Tables ──────────────────────────────────────────────────────────────
    tables = {}
    if trace:
        tables["patient"]   = sc.build_patient_table_html(trace)
        tables["decision"]  = sc.build_decision_table_html(trace["decisions"])
        tables["final_mtd"] = sc.build_final_mtd_html(trace)

    # ── Scalar metrics ───────────────────────────────────────────────────────
    skip = {"p63","pcrm","avg_n63","avg_ncrm","avg_nsurg63","avg_nsurgcrm","crm_trace"}
    metrics = {}
    for k, v in res.items():
        if k in skip: continue
        if isinstance(v, (np.integer,)): metrics[k] = int(v)
        elif isinstance(v, (np.floating,)): metrics[k] = float(v)
        else: metrics[k] = v

    return jsonify({"charts": charts, "tables": tables, "metrics": metrics})


@app.route("/api/export-config")
def api_export_config():
    data = json.dumps(_state_json(), indent=2, default=_np_clean)
    return Response(data, mimetype="application/json",
                    headers={"Content-Disposition":
                             "attachment; filename=crm_config.json"})


@app.route("/api/import-config", methods=["POST"])
def api_import_config():
    try:
        data = request.get_json(force=True)
        allowed = (set(sc.R_DEFAULTS) | set(sc._TRUE_DEFAULTS) | set(sc.DE_DEFAULTS))
        for k, v in data.items():
            if k in allowed:
                _STATE[k] = v
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/sweep", methods=["POST"])
def api_sweep():
    import matplotlib.pyplot as plt
    data       = request.get_json(force=True)
    true_t1, true_t2 = _true_tox()
    skel_t1, skel_t2 = _skeletons()
    param_name = str(data.get("param_name", _g("de_param_name", "sigma")))
    n_sim      = int(data.get("n_sim",  _g("de_n_sim",  100)))
    seed       = int(data.get("seed",   _g("de_seed",   42)))
    speed      = bool(data.get("speed", _g("de_speed_mode", False)))

    # Persist sweep params into state
    _STATE["de_param_name"] = param_name
    _STATE["de_n_sim"]      = n_sim
    _STATE["de_seed"]       = seed

    pv_list, param_label = sc._de_pv_for_param(param_name, _STATE, speed=speed)
    base_ss = _build_base_ss()

    df = sc.run_parameter_sweep(
        param_name, pv_list, base_ss, true_t1, true_t2, skel_t1, skel_t2, n_sim, seed)

    fig = sc._plot_sweep_results(df, param_label, param_name=param_name)
    chart_b64 = sc.fig_to_b64(fig); plt.close(fig)

    ctx_b64 = None
    if param_name in ("prior_nu_t1", "prior_nu_t2"):
        is_t1 = (param_name == "prior_nu_t1")
        ctx_fig = sc._plot_prior_mtd_context(
            true_t1 if is_t1 else true_t2, pv_list,
            "tox1 (acute)" if is_t1 else "tox2 (subacute)",
            f"True tox {'1' if is_t1 else '2'} vs skeleton choices",
            float(_g("prior_target_t1" if is_t1 else "prior_target_t2")),
            float(_g("halfwidth_t1"    if is_t1 else "halfwidth_t2")),
            model=str(_g("prior_model", "empiric")),
            intcpt=float(_g("logistic_intcpt", 3.0)),
        )
        ctx_b64 = sc.fig_to_b64(ctx_fig); plt.close(ctx_fig)

    table_html = df[["param_label","n_patients","quality_score",
                      "pct_correct_selection","overdose_rate"]].rename(columns={
        "param_label": param_label, "n_patients": "N",
        "quality_score": "Quality score",
        "pct_correct_selection": "% Correct selection",
        "overdose_rate": "Overdose rate (%)",
    }).round({"Quality score": 4, "% Correct selection": 1, "Overdose rate (%)": 1}
    ).to_html(index=False, border=0, classes="de-tbl")

    return jsonify({"chart": chart_b64, "ctx_chart": ctx_b64,
                    "table_html": table_html, "param_label": param_label})


@app.route("/api/batch", methods=["POST"])
def api_batch():
    import matplotlib.pyplot as plt
    data      = request.get_json(force=True)
    true_t1, true_t2 = _true_tox()
    skel_t1, skel_t2 = _skeletons()
    n_sim     = int(data.get("n_sim",  _g("de_n_sim", 100)))
    seed      = int(data.get("seed",   _g("de_seed",   42)))
    run_label = str(data.get("run_label", ""))
    speed     = bool(data.get("speed", _g("de_speed_mode", False)))
    base_ss   = _build_base_ss()
    ts_str    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    results_list = []
    for param_name in sc._DE_ALL_PARAMS:
        pv_list, param_label = sc._de_pv_for_param(param_name, _STATE, speed=speed)
        df = sc.run_parameter_sweep(
            param_name, pv_list, base_ss, true_t1, true_t2, skel_t1, skel_t2, n_sim, seed)
        fig_l = sc._plot_sweep_results_light(df, param_label, param_name=param_name)
        fig_b64 = sc._fig_to_b64(fig_l); plt.close(fig_l)

        ctx_b64 = None
        if param_name in ("prior_nu_t1", "prior_nu_t2"):
            is_t1 = (param_name == "prior_nu_t1")
            ctx_fig = sc._plot_prior_mtd_context(
                true_t1 if is_t1 else true_t2, pv_list,
                "tox1 (acute)" if is_t1 else "tox2 (subacute)",
                f"True tox {'1' if is_t1 else '2'} vs skeleton choices",
                float(_g("prior_target_t1" if is_t1 else "prior_target_t2")),
                float(_g("halfwidth_t1"    if is_t1 else "halfwidth_t2")),
                model=str(_g("prior_model","empiric")),
                intcpt=float(_g("logistic_intcpt",3.0)), light=True,
            )
            ctx_b64 = sc._fig_to_b64(ctx_fig); plt.close(ctx_fig)

        results_list.append({
            "param_name": param_name, "param_label": param_label,
            "pv_list": pv_list, "result_df": df,
            "fig_b64": fig_b64, "context_fig_b64": ctx_b64,
        })

    html = sc._generate_de_all_html_report(
        results_list, base_ss, n_sim, seed, ts_str, run_label=run_label)
    return Response(html, mimetype="text/html",
                    headers={"Content-Disposition":
                             "attachment; filename=de_batch_report.html"})


# ==============================================================================
if __name__ == "__main__":
    print("Starting TITE CRM Simulator on http://localhost:5000")
    app.run(debug=True, port=5000, use_reloader=False)
