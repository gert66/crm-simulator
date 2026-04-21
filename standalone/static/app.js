/* app.js — TITE CRM Simulator frontend logic */
'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let state = {};

// ── Bootstrap ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  const resp = await fetch('/api/state');
  state = await resp.json();
  bindAllInputs();
  renderAllInputs();
  updatePreviewDebounced();
  updateTimelineDebounced();
  syncDEParamControls(state.de_param_name || 'sigma');
  syncEwocAlphaDisabled();
  syncLogisticIntcptRow();
  syncTraceSection();
  initDEListInputs();
});

// ── View switching ────────────────────────────────────────────────────────────
function switchView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById('view-' + name).classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
}

// ── Input binding ─────────────────────────────────────────────────────────────
function bindAllInputs() {
  document.querySelectorAll('[data-key]').forEach(el => {
    const key = el.dataset.key;
    const evt = (el.type === 'range' || el.tagName === 'SELECT') ? 'input' : 'change';
    el.addEventListener(evt, () => handleInput(el, key));
  });
}

function handleInput(el, key) {
  let value;
  if (el.type === 'checkbox') {
    value = el.checked;
  } else if (el.type === 'range' || el.type === 'number') {
    value = el.step && el.step.includes('.') ? parseFloat(el.value) : parseInt(el.value, 10);
    if (isNaN(value)) return;
  } else {
    value = el.value;
  }

  state[key] = value;
  updateBadge(key, value, el.dataset.fmt);
  sendUpdate({ [key]: value });

  // Side-effects
  const previewKeys = new Set([
    'true_t1_L0','true_t1_L1','true_t1_L2','true_t1_L3','true_t1_L4',
    'true_t2_L0','true_t2_L1','true_t2_L2','true_t2_L3','true_t2_L4',
    'target_t1','target_t2',
    'prior_target_t1','prior_target_t2','halfwidth_t1','halfwidth_t2',
    'prior_nu_t1','prior_nu_t2','prior_model','logistic_intcpt',
  ]);
  const timelineKeys = new Set(['incl_to_rt','rt_dur','rt_to_surg','tox2_win']);

  if (previewKeys.has(key))  updatePreviewDebounced();
  if (timelineKeys.has(key)) updateTimelineDebounced();
  if (key === 'ewoc_on')        syncEwocAlphaDisabled();
  if (key === 'prior_model')    syncLogisticIntcptRow();
  if (key === 'show_crm_trace') syncTraceSection();
}

function renderAllInputs() {
  document.querySelectorAll('[data-key]').forEach(el => {
    const key = el.dataset.key;
    if (!(key in state)) return;
    const v = state[key];
    if (el.type === 'checkbox') {
      el.checked = !!v;
    } else {
      el.value = v;
    }
    updateBadge(key, v, el.dataset.fmt);
  });
  // List text inputs
  setListInput('de-max-n-input',  state.de_max_n_vals  || [12,15,18,21,24,27,30,33,36]);
  setListInput('de-cohort-input', state.de_cohort_vals || [1,2,3,4]);
  setListInput('de-nu1-input',    state.de_nu1_vals    || [1,2,3,4,5]);
  setListInput('de-nu2-input',    state.de_nu2_vals    || [1,2,3,4,5]);
  // DE param selector
  const deSel = document.getElementById('de-param-sel');
  if (deSel) deSel.value = state.de_param_name || 'sigma';
}

function updateBadge(key, value, fmt) {
  const badge = document.getElementById(key + '-v');
  if (!badge) return;
  let display;
  if (fmt === 'f1') display = parseFloat(value).toFixed(1);
  else if (fmt === 'f2') display = parseFloat(value).toFixed(2);
  else if (fmt === 'f3') display = parseFloat(value).toFixed(3);
  else display = value;
  badge.textContent = display;
}

// ── API helpers ───────────────────────────────────────────────────────────────
const _sendQueue = {};
function sendUpdate(patch) {
  fetch('/api/update', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
}

// ── Debounced chart updates ───────────────────────────────────────────────────
let _previewTimer = null, _timelineTimer = null;

function updatePreviewDebounced() {
  clearTimeout(_previewTimer);
  _previewTimer = setTimeout(updatePreview, 250);
}

function updateTimelineDebounced() {
  clearTimeout(_timelineTimer);
  _timelineTimer = setTimeout(updateTimeline, 300);
}

async function updatePreview() {
  const img  = document.getElementById('preview-img');
  const spin = document.getElementById('preview-spinner');
  if (!img) return;
  spin && (spin.style.display = 'block');
  try {
    const r = await fetch('/api/preview');
    const d = await r.json();
    img.src = d.img;
    img.style.display = 'block';
  } finally {
    spin && (spin.style.display = 'none');
  }
}

async function updateTimeline() {
  const img  = document.getElementById('timeline-img');
  const spin = document.getElementById('timeline-spinner');
  if (!img) return;
  spin && (spin.style.display = 'block');
  try {
    const r = await fetch('/api/timeline');
    const d = await r.json();
    img.src = d.img;
    img.style.display = 'block';
  } finally {
    spin && (spin.style.display = 'none');
  }
}

// ── Run simulation ────────────────────────────────────────────────────────────
async function runSimulation() {
  const btn  = document.getElementById('run-btn');
  const spin = document.getElementById('run-spinner');
  const area = document.getElementById('results-area');
  const ph   = document.getElementById('run-placeholder');

  btn.disabled = true;
  spin.style.display = 'flex';
  ph.style.display   = 'none';
  area.style.display = 'none';

  try {
    const r = await fetch('/api/run', { method: 'POST' });
    const d = await r.json();
    renderResults(d);
    area.style.display = 'block';
  } catch(e) {
    alert('Simulation error: ' + e.message);
  } finally {
    btn.disabled = false;
    spin.style.display = 'none';
  }
}

function renderResults(d) {
  const { charts, tables, metrics } = d;

  // Main charts
  setImg('res-mtd-img',      charts.mtd);
  setImg('res-patients-img', charts.patients);

  // Posterior / eligibility charts
  setImg('res-post-track-img',  charts.posterior_tracking);
  setImg('res-study-end-img',   charts.study_end_posteriors);
  setImg('res-eligibility-img', charts.dose_eligibility);

  // Trace charts
  setImg('res-trace-dose-img',   charts.trace_dose);
  setImg('res-trace-safety-img', charts.trace_safety);
  setImg('res-trace-tite-img',   charts.trace_tite);

  // Metrics grid
  renderMetrics(metrics);

  // Tables
  if (tables.patient)   document.getElementById('patient-table-wrap').innerHTML = tables.patient;
  if (tables.decision)  document.getElementById('decision-table-wrap').innerHTML = tables.decision;
  if (tables.final_mtd) document.getElementById('final-mtd-wrap').innerHTML = tables.final_mtd;

  // EWOC caption
  const cap = document.getElementById('ewoc-caption');
  if (cap) {
    if (state.ewoc_on) {
      cap.textContent = `EWOC ON (α = ${parseFloat(state.ewoc_alpha).toFixed(2)}) — doses filtered where P(tox1 OD) < α AND P(tox2 OD) < α. Highest jointly admissible dose selected.`;
    } else {
      cap.textContent = 'EWOC OFF — no overdose-probability filter. Model picks dose with posterior mean P(tox1) closest to target1 (argmin rule).';
    }
  }

  // Results caption
  const ts = metrics.true_safe;
  document.getElementById('results-caption').textContent =
    `n_sims=${metrics.ns} | seed=${metrics.seed}` +
    (ts != null ? ` | True jointly safe dose: L${ts}` : ' | No jointly safe dose');

  // Sync trace section visibility
  syncTraceSection();
}

function renderMetrics(m) {
  const grid = document.getElementById('metrics-grid');
  if (!grid) return;
  const items = [
    ['Tox1/pt (6+3)',      fmt3(m.acute_rate_63),  'Acute DLT rate per treated patient — TITE 6+3'],
    ['Tox1/pt (CRM)',      fmt3(m.acute_rate_crm), 'Acute DLT rate per treated patient — TITE-CRM'],
    ['Tox2/surg (6+3)',    fmt3(m.sub_gs_rate_63),  'Subacute DLT rate per surgery-evaluable patient — 6+3'],
    ['Tox2/surg (CRM)',    fmt3(m.sub_gs_rate_crm), 'Subacute DLT rate per surgery-evaluable patient — CRM'],
    ['Duration mean (6+3)', fmt1(m.dur63_mean) + ' mo',  'Mean trial duration — TITE 6+3'],
    ['Duration mean (CRM)', fmt1(m.durcrm_mean) + ' mo', 'Mean trial duration — TITE-CRM'],
    ['Duration median (6+3)', fmt1(m.dur63_med) + ' mo',  'Median trial duration — TITE 6+3'],
    ['Duration median (CRM)', fmt1(m.durcrm_med) + ' mo', 'Median trial duration — TITE-CRM'],
    ['Avg bridging pts (6+3)', fmt1(m.avg_bridging), 'Avg patients at bridging dose per trial (6+3)'],
  ];
  grid.innerHTML = items.map(([lbl, val, title]) =>
    `<div class="metric-card" title="${title}">
       <div class="metric-label">${lbl}</div>
       <div class="metric-value">${val}</div>
     </div>`
  ).join('');
}

// ── UI sync helpers ───────────────────────────────────────────────────────────
function syncEwocAlphaDisabled() {
  const sl = document.getElementById('ewoc-alpha-slider');
  if (sl) sl.disabled = !state.ewoc_on;
}

function syncLogisticIntcptRow() {
  const row = document.getElementById('intcpt-row');
  if (row) row.style.display = (state.prior_model === 'logistic') ? 'flex' : 'none';
}

function syncTraceSection() {
  const sec = document.getElementById('trace-section');
  const cb  = document.getElementById('show-trace-cb');
  if (!sec) return;
  const show = state.show_crm_trace || (cb && cb.checked);
  sec.style.display = show ? 'block' : 'none';
}

function toggleTraceSection(checked) {
  state.show_crm_trace = checked;
  sendUpdate({ show_crm_trace: checked });
  syncTraceSection();
}

function toggleEwoc(checked) {
  state.ewoc_on = checked;
  sendUpdate({ ewoc_on: checked });
  syncEwocAlphaDisabled();
}

// ── Prior scenario ────────────────────────────────────────────────────────────
async function applyPriorScenario(name) {
  const r = await fetch('/api/prior-scenario', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  const d = await r.json();
  if (d.ok) {
    Object.assign(state, d.applied);
    // Update input elements for affected keys
    for (const [k, v] of Object.entries(d.applied)) {
      const el = document.querySelector(`[data-key="${k}"]`);
      if (el) { el.value = v; updateBadge(k, v, el.dataset.fmt); }
    }
    document.getElementById('scenario-desc').textContent = d.description || '';
    updatePreviewDebounced();
  }
}

// ── Config export / import ────────────────────────────────────────────────────
function exportConfig() {
  window.location.href = '/api/export-config';
}

async function importConfig(input) {
  const file = input.files[0];
  if (!file) return;
  const text = await file.text();
  let json;
  try { json = JSON.parse(text); }
  catch { alert('Invalid JSON file'); return; }

  const r = await fetch('/api/import-config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(json),
  });
  const d = await r.json();
  if (d.ok) {
    const sr = await fetch('/api/state');
    state = await sr.json();
    renderAllInputs();
    updatePreviewDebounced();
    updateTimelineDebounced();
    alert('Configuration imported.');
  } else {
    alert('Import failed: ' + d.error);
  }
  input.value = '';
}

async function resetConfig() {
  if (!confirm('Reset all settings to defaults?')) return;
  await fetch('/api/reset', { method: 'POST' });
  const sr = await fetch('/api/state');
  state = await sr.json();
  renderAllInputs();
  updatePreviewDebounced();
  updateTimelineDebounced();
}

// ── Design Exploration ────────────────────────────────────────────────────────
function onDEParamChange(value) {
  state.de_param_name = value;
  sendUpdate({ de_param_name: value });
  syncDEParamControls(value);
}

function syncDEParamControls(param) {
  document.querySelectorAll('.de-param-ctrl').forEach(el => el.style.display = 'none');
  const target = document.getElementById('de-ctrl-' + param);
  if (target) target.style.display = 'block';
}

function initDEListInputs() {
  setListInput('de-max-n-input',  state.de_max_n_vals  || [12,15,18,21,24,27,30,33,36]);
  setListInput('de-cohort-input', state.de_cohort_vals || [1,2,3,4]);
  setListInput('de-nu1-input',    state.de_nu1_vals    || [1,2,3,4,5]);
  setListInput('de-nu2-input',    state.de_nu2_vals    || [1,2,3,4,5]);
}

function setListInput(id, arr) {
  const el = document.getElementById(id);
  if (el) el.value = arr.join(', ');
}

function parseListInput(key, rawStr, type) {
  const vals = rawStr.split(',')
    .map(s => s.trim())
    .filter(s => s !== '')
    .map(s => type === 'int' ? parseInt(s, 10) : parseFloat(s))
    .filter(v => !isNaN(v));
  if (vals.length === 0) return;
  state[key] = vals;
  sendUpdate({ [key]: vals });
}

async function runSweep() {
  const btn  = document.getElementById('de-param-sel');
  const spin = document.getElementById('de-spinner');
  const ph   = document.getElementById('de-placeholder');
  const area = document.getElementById('de-results-area');

  spin.style.display = 'flex';
  ph.style.display   = 'none';
  area.style.display = 'none';

  const param = state.de_param_name || 'sigma';
  const payload = {
    param_name: param,
    n_sim:  parseInt(state.de_n_sim || 100),
    seed:   parseInt(state.de_seed  || 42),
    speed:  !!state.de_speed_mode,
  };

  try {
    const r = await fetch('/api/sweep', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    document.getElementById('de-result-title').textContent = 'Sweep: ' + d.param_label;
    setImg('de-sweep-img', d.chart);
    const ctxWrap = document.getElementById('de-ctx-wrap');
    if (d.ctx_chart) {
      setImg('de-ctx-img', d.ctx_chart);
      ctxWrap.style.display = 'block';
    } else {
      ctxWrap.style.display = 'none';
    }
    document.getElementById('de-table-wrap').innerHTML = d.table_html || '';
    area.style.display = 'block';
  } catch(e) {
    alert('Sweep error: ' + e.message);
  } finally {
    spin.style.display = 'none';
  }
}

function showBatchModal() {
  document.getElementById('batch-modal').style.display = 'flex';
  document.getElementById('batch-n-sim').value = state.de_n_sim || 100;
  document.getElementById('batch-seed').value  = state.de_seed  || 42;
  document.getElementById('batch-speed').checked = !!state.de_speed_mode;
}
function hideBatchModal() {
  document.getElementById('batch-modal').style.display = 'none';
  document.getElementById('batch-spinner').style.display = 'none';
}

async function runBatch() {
  const spin = document.getElementById('batch-spinner');
  spin.style.display = 'flex';

  const payload = {
    n_sim:     parseInt(document.getElementById('batch-n-sim').value),
    seed:      parseInt(document.getElementById('batch-seed').value),
    run_label: document.getElementById('batch-label').value.trim(),
    speed:     document.getElementById('batch-speed').checked,
  };

  try {
    const r = await fetch('/api/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const blob = await r.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'de_batch_report.html'; a.click();
    URL.revokeObjectURL(url);
    hideBatchModal();
  } catch(e) {
    alert('Batch error: ' + e.message);
  } finally {
    spin.style.display = 'none';
  }
}

// ── Utility ───────────────────────────────────────────────────────────────────
function setImg(id, src) {
  const el = document.getElementById(id);
  if (el && src) { el.src = src; el.style.display = 'block'; }
}
function fmt1(v) { return v != null ? parseFloat(v).toFixed(1) : '—'; }
function fmt3(v) { return v != null ? parseFloat(v).toFixed(3) : '—'; }
