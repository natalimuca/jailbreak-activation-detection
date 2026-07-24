const modelSelect = document.getElementById("model-select");
const promptInput = document.getElementById("prompt-input");
const analyzeButton = document.getElementById("analyze-button");
const buttonSpinner = analyzeButton.querySelector(".spinner");
const buttonIcon = analyzeButton.querySelector(".btn-icon");
const buttonLabel = analyzeButton.querySelector(".btn-label");
const statusLine = document.getElementById("status-line");
const modelStatsBody = document.querySelector("#model-stats .model-stats-body");

const ICONS = {
  clear: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.25" stroke-linecap="round" stroke-linejoin="round"><polyline points="4,12.5 9,17.5 20,6.5"/></svg>',
  flagged:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 L21.5 20 H2.5 Z"/><line x1="12" y1="9.5" x2="12" y2="14"/><circle cx="12" cy="16.8" r="0.6" fill="currentColor" stroke="none"/></svg>',
  unavailable:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><line x1="8" y1="12" x2="16" y2="12"/></svg>',
  idle: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><polyline points="12,7 12,12 15.5,14"/></svg>',
};

const LOADING_MESSAGES = [
  "Scoring keyword filter…",
  "Loading target model & extracting activations…",
  "Scoring dense-direction detector…",
  "Scoring SAE-feature detector…",
  "Swapping in the perplexity backbone…",
  "Scoring perplexity filter…",
];

// Real, published numbers from this repo's own RESULTS.md (dense-direction
// cross-model table + SAE-feature cross-model comparison) -- static
// reference context, never invented and never recomputed at request time.
// If RESULTS.md is ever updated, update this table to match; don't let it
// drift silently.
const MODEL_VALIDATION = {
  "Qwen/Qwen2.5-1.5B-Instruct": { layer: 20, testAcc: "89.6%", testAuroc: "0.970", pair: "38.1%", saeAuroc: null },
  "HuggingFaceTB/SmolLM2-1.7B-Instruct": { layer: 14, testAcc: "87.8%", testAuroc: "0.945", pair: "90.5%", saeAuroc: null },
  "Qwen/Qwen3-8B": { layer: 23, testAcc: "88.9%", testAuroc: "0.983", pair: "42.9%", saeAuroc: "0.975" },
  "meta-llama/Llama-3.1-8B-Instruct": { layer: 27, testAcc: "93.1%", testAuroc: "0.989", pair: "66.7%", saeAuroc: "0.978" },
  "google/gemma-2-9b-it": { layer: 34, testAcc: "93.1%", testAuroc: "0.984", pair: "47.6%", saeAuroc: "0.966" },
};

let loadingMessageTimer = null;

function setStatus(message, isError = false) {
  statusLine.textContent = message;
  statusLine.classList.toggle("error", isError);
}

function setBusy(isBusy) {
  analyzeButton.disabled = isBusy;
  buttonSpinner.hidden = !isBusy;
  buttonIcon.hidden = isBusy;
  buttonLabel.textContent = isBusy ? "Analyzing…" : "Analyze";

  if (isBusy) {
    let i = 0;
    setStatus(LOADING_MESSAGES[0]);
    // Not real server-reported progress (the API returns one response after
    // scoring everything) -- approximates the known GPU swap sequence so a
    // wait of up to ~a minute doesn't read as a frozen page.
    loadingMessageTimer = setInterval(() => {
      i = (i + 1) % LOADING_MESSAGES.length;
      setStatus(LOADING_MESSAGES[i]);
    }, 4000);
  } else {
    clearInterval(loadingMessageTimer);
  }
}

function renderModelStats(hfName) {
  const stats = MODEL_VALIDATION[hfName];
  if (!stats) {
    modelStatsBody.innerHTML = `<p class="empty-note">No validation record for this model.</p>`;
    return;
  }
  const rows = [
    ["Dense-direction layer", String(stats.layer)],
    ["TEST accuracy (n=288)", stats.testAcc],
    ["TEST AUROC", stats.testAuroc],
    ["PAIR-paraphrase robustness", stats.pair],
    ["SAE-feature AUROC", stats.saeAuroc ?? "n/a (no SAE suite)"],
  ];
  modelStatsBody.innerHTML = rows
    .map(([k, v]) => `<div class="model-stat-row"><span class="k">${k}</span><span class="v">${v}</span></div>`)
    .join("");
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error(`Failed to load models (${response.status})`);
    const models = await response.json();

    modelSelect.innerHTML = "";
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model.hf_name;
      option.textContent = model.sae_feature_available
        ? model.cache_label
        : `${model.cache_label} (no SAE)`;
      modelSelect.appendChild(option);
    }
    modelSelect.disabled = false;
    analyzeButton.disabled = false;
    renderModelStats(modelSelect.value);
    renderFindings(); // highlight the default-selected model as soon as the real model list loads
  } catch (err) {
    setStatus(`Could not load model list: ${err.message}`, true);
  }
}

function formatScore(value) {
  return Number.isFinite(value) ? value.toFixed(2) : String(value);
}

function renderPill(state, label) {
  return `<span class="pill ${state}">${ICONS[state]}${label}</span>`;
}

/* Magnitude axis (0 anchored at the left edge): for scores that are never
   meaningfully negative (keyword count, perplexity, SAE-feature sum). */
function renderMagnitudeAxis(flagged, score, threshold) {
  const stateClass = flagged ? "flagged" : "clear";
  const domain = Math.max(score, threshold, 1e-6) * 1.3;
  const fillPct = Math.min(100, (Math.max(score, 0) / domain) * 100);
  const thresholdPct = Math.min(98, (threshold / domain) * 100);
  return `
    <div class="score-stats">
      <span class="score-value ${stateClass}">${formatScore(score)}</span>
      <span class="score-threshold">/ ${formatScore(threshold)} threshold</span>
    </div>
    <div class="axis">
      <div class="axis-track"></div>
      <div class="axis-fill ${stateClass}" style="left:0; width:${fillPct}%"></div>
      <div class="axis-threshold" style="left:${thresholdPct}%"></div>
    </div>
  `;
}

/* Diverging axis (zero-centered): for dense-direction's real signed
   projection score and threshold, which are routinely negative (e.g.
   SmolLM2's calibrated threshold is -17.26) -- a 0-anchored bar would
   collapse every negative score to an identical "empty" reading. */
function renderDivergingAxis(flagged, score, threshold) {
  const stateClass = flagged ? "flagged" : "clear";
  const domain = Math.max(Math.abs(score), Math.abs(threshold), 1e-6) * 1.3;
  const scorePct = (score / domain) * 50;
  const thresholdPct = (threshold / domain) * 50;
  const fillLeft = score >= 0 ? 50 : 50 + scorePct;
  const fillWidth = Math.abs(scorePct);
  return `
    <div class="score-stats">
      <span class="score-value ${stateClass}">${score >= 0 ? "+" : ""}${formatScore(score)}</span>
      <span class="score-threshold">/ ${formatScore(threshold)} threshold</span>
    </div>
    <div class="axis">
      <div class="axis-track"></div>
      <div class="axis-zero" style="left:50%"></div>
      <div class="axis-fill ${stateClass}" style="left:${fillLeft}%; width:${fillWidth}%"></div>
      <div class="axis-threshold" style="left:${50 + thresholdPct}%"></div>
    </div>
  `;
}

function renderKeyword(result) {
  let html = renderPill(result.flagged ? "flagged" : "clear", result.flagged ? "Flagged" : "Clear");
  html += renderMagnitudeAxis(result.flagged, result.score, result.threshold);
  if (result.matched_terms.length > 0) {
    html += `<p class="detail-label">Matched terms</p><ul class="term-list">${result.matched_terms
      .map((term) => `<li>${term}</li>`)
      .join("")}</ul>`;
  } else {
    html += `<p class="empty-note">No lexicon terms matched.</p>`;
  }
  return html;
}

function renderPerplexity(result) {
  return (
    renderPill(result.flagged ? "flagged" : "clear", result.flagged ? "Flagged" : "Clear") +
    renderMagnitudeAxis(result.flagged, result.score, result.threshold)
  );
}

function renderDenseDirection(result) {
  return (
    renderPill(result.flagged ? "flagged" : "clear", result.flagged ? "Flagged" : "Clear") +
    renderDivergingAxis(result.flagged, result.score, result.threshold) +
    `<p class="detail-label">Layer</p><p class="empty-note">${result.layer}</p>`
  );
}

function renderSAEFeature(result) {
  if (!result.available) {
    return renderPill("unavailable", "Unavailable") + `<p class="empty-note">${result.reason}</p>`;
  }
  let html = renderPill(result.flagged ? "flagged" : "clear", result.flagged ? "Flagged" : "Clear");
  html += renderMagnitudeAxis(result.flagged, result.score, result.threshold);

  const maxContribution = Math.max(...result.top_features.map((f) => Math.abs(f.contribution)), 1e-6);
  html += `<p class="detail-label">Top contributing features</p><ul class="feature-list">${result.top_features
    .map((f) => {
      const pct = Math.min(100, (Math.abs(f.contribution) / maxContribution) * 100);
      return `<li>
        <span>L${f.layer}/F${f.feature}</span>
        <span class="feature-bar-wrap">
          <span class="feature-bar-track"><span class="feature-bar-fill" style="width:${pct}%"></span></span>
          <span class="contribution">${formatScore(f.contribution)}</span>
        </span>
      </li>`;
    })
    .join("")}</ul>`;
  return html;
}

const RENDERERS = {
  keyword: renderKeyword,
  perplexity: renderPerplexity,
  dense_direction: renderDenseDirection,
  sae_feature: renderSAEFeature,
};

function renderIdleCards() {
  for (const detector of Object.keys(RENDERERS)) {
    const card = document.getElementById(`card-${detector}`);
    card.dataset.state = "idle";
    card.querySelector(".card-body").innerHTML =
      renderPill("idle", "Not run yet") + `<p class="empty-note">Run an analysis to see this detector's result.</p>`;
  }
}

function renderSkeletonCards() {
  const skeletonBody = `
    <div class="skeleton skeleton-line w-40"></div>
    <div class="skeleton skeleton-line w-80"></div>
    <div class="skeleton skeleton-line w-60"></div>
  `;
  for (const detector of Object.keys(RENDERERS)) {
    const card = document.getElementById(`card-${detector}`);
    card.dataset.state = "loading";
    card.querySelector(".card-body").innerHTML = skeletonBody;
  }
}

function renderResults(analysis) {
  for (const [detector, render] of Object.entries(RENDERERS)) {
    const card = document.getElementById(`card-${detector}`);
    const result = analysis[detector];
    card.dataset.state = result ? "done" : "idle";
    card.querySelector(".card-body").innerHTML = result ? render(result) : `<p class="empty-note">Not requested.</p>`;
  }
}

async function analyze() {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    setStatus("Enter a prompt first.", true);
    return;
  }

  setBusy(true);
  renderSkeletonCards();

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, model_name: modelSelect.value }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);

    renderResults(body);
    setBusy(false);
    setStatus("Done.");
  } catch (err) {
    setBusy(false);
    renderIdleCards();
    setStatus(`Error: ${err.message}`, true);
  }
}

analyzeButton.addEventListener("click", analyze);
modelSelect.addEventListener("change", () => {
  renderModelStats(modelSelect.value);
  renderFindings(); // re-highlight the newly-selected model across the findings charts
});
renderIdleCards();
loadModels();

// ---------------------------------------------------------------------
// Findings dashboard -- static data read directly from this repo's own
// RESULTS.md (dense-direction cross-model table, PAIR-robustness ranking,
// SAE-feature cross-model comparison, Qwen3-8B baseline table). Never
// recomputed at request time and never fetched from the live API -- if
// RESULTS.md is ever updated, these arrays need updating to match, same
// discipline as MODEL_VALIDATION above.
// ---------------------------------------------------------------------

const AUROC_BY_MODEL = [
  { label: "Llama-3.1-8B-Instruct", value: 0.989 },
  { label: "gemma-2-9b-it", value: 0.984 },
  { label: "Qwen3-8B", value: 0.983 },
  { label: "Qwen2.5-1.5B-Instruct", value: 0.97 },
  { label: "SmolLM2-1.7B-Instruct", value: 0.945 },
];

const PAIR_ROBUSTNESS_BY_MODEL = [
  { label: "SmolLM2-1.7B-Instruct", value: 90.5 },
  { label: "Llama-3.1-8B-Instruct", value: 66.7 },
  { label: "gemma-2-9b-it", value: 47.6 },
  { label: "Qwen3-8B", value: 42.9 },
  { label: "Qwen2.5-1.5B-Instruct", value: 38.1 },
];

const DENSE_VS_SAE_AUROC = [
  { label: "Llama-3.1-8B-Instruct", dense: 0.989, sae: 0.978, note: "DeLong p=0.024" },
  { label: "gemma-2-9b-it", dense: 0.984, sae: 0.966, note: "DeLong p=0.0063" },
  { label: "Qwen3-8B", dense: 0.983, sae: 0.975, note: "DeLong p=0.068 (n.s.)" },
];

const BASELINE_COMPARISON = [
  { label: "Dense-direction", value: 0.983, group: "activation" },
  { label: "SAE-feature (top-15)", value: 0.975, group: "activation" },
  { label: "Keyword filter", value: 0.603, group: "baseline" },
  { label: "Perplexity filter", value: 0.52, group: "baseline" },
];

function currentModelLabel() {
  return modelSelect.value.split("/").pop();
}

/* Horizontal bar rows -- see the dataviz skill's marks-and-anatomy.md:
   thin bars, 4px rounded data-end (square at the baseline), a direct
   value label at the tip rather than gridline ticks.

   `highlightLabel` ties these findings back to the model actually selected
   in the live detector above (the dataviz skill's "emphasis" form: one
   series in the accent hue, the rest in the de-emphasis gray) -- only
   applied when that model actually appears in this particular chart's
   data, so switching to a model absent from a 3-model chart doesn't grey
   out everything for no reason. */
function renderBars(data, { formatValue, colorFor, domain, highlightLabel }) {
  const max = domain ?? Math.max(...data.map((d) => d.value));
  const hasMatch = Boolean(highlightLabel) && data.some((d) => d.label === highlightLabel);
  return data
    .map((d) => {
      const pct = Math.max(2, (d.value / max) * 100);
      const isCurrent = hasMatch && d.label === highlightLabel;
      const baseColor = colorFor ? colorFor(d) : "var(--chart-blue)";
      const color = hasMatch && !isCurrent ? "var(--chart-gray)" : baseColor;
      return `
        <div class="bar-row${isCurrent ? " bar-row-current" : ""}">
          <span class="bar-label" title="${d.label}">${d.label}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%; background:${color}"></div></div>
          <span class="bar-value">${formatValue(d.value)}</span>
        </div>
      `;
    })
    .join("");
}

/* The accessibility twin of every chart above (dataviz skill, Tier 0:
   "table-view toggle") -- collapsed by default so it doesn't compete with
   the chart, but always present with the exact same numbers. */
function appendTableView(container, headers, rows) {
  const details = document.createElement("details");
  details.className = "table-toggle";
  const summary = document.createElement("summary");
  summary.textContent = "View as table";
  details.appendChild(summary);
  const table = document.createElement("table");
  table.className = "data-table";
  table.innerHTML = `
    <thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>
  `;
  details.appendChild(table);
  container.appendChild(details);
}

function renderLegend(elementId, entries) {
  document.getElementById(elementId).innerHTML = entries
    .map(([label, colorVar]) => `<span class="legend-swatch" style="--swatch-color:${colorVar}">${label}</span>`)
    .join("");
}

function renderFindings() {
  const current = currentModelLabel();

  const aurocEl = document.getElementById("chart-auroc");
  aurocEl.innerHTML = `<div class="bar-chart">${renderBars(AUROC_BY_MODEL, {
    formatValue: (v) => v.toFixed(3),
    domain: 1.0,
    highlightLabel: current,
  })}</div>`;
  appendTableView(
    aurocEl,
    ["Model", "TEST AUROC"],
    AUROC_BY_MODEL.map((d) => [d.label, d.value.toFixed(3)])
  );

  const pairEl = document.getElementById("chart-pair");
  pairEl.innerHTML = `<div class="bar-chart">${renderBars(PAIR_ROBUSTNESS_BY_MODEL, {
    formatValue: (v) => `${v.toFixed(1)}%`,
    domain: 100,
    highlightLabel: current,
  })}</div>`;
  appendTableView(
    pairEl,
    ["Model", "PAIR detection rate"],
    PAIR_ROBUSTNESS_BY_MODEL.map((d) => [d.label, `${d.value.toFixed(1)}%`])
  );

  renderLegend("legend-dense-sae", [
    ["Dense-direction", "var(--chart-blue)"],
    ["SAE-feature", "var(--chart-amber)"],
  ]);
  const denseSaeEl = document.getElementById("chart-dense-sae");
  denseSaeEl.innerHTML = `<div class="bar-chart">${DENSE_VS_SAE_AUROC.map(
    (d) => `
      <div class="bar-group${d.label === current ? " bar-group-current" : DENSE_VS_SAE_AUROC.some((x) => x.label === current) ? " bar-group-dim" : ""}">
        <p class="bar-group-title">${d.label}<span class="bar-group-note">${d.note}</span></p>
        ${renderBars([{ label: "Dense-direction", value: d.dense }], {
          formatValue: (v) => v.toFixed(3),
          domain: 1.0,
          colorFor: () => "var(--chart-blue)",
        })}
        ${renderBars([{ label: "SAE-feature", value: d.sae }], {
          formatValue: (v) => v.toFixed(3),
          domain: 1.0,
          colorFor: () => "var(--chart-amber)",
        })}
      </div>
    `
  ).join("")}</div>`;
  appendTableView(
    denseSaeEl,
    ["Model", "Dense AUROC", "SAE AUROC", "DeLong test"],
    DENSE_VS_SAE_AUROC.map((d) => [d.label, d.dense.toFixed(3), d.sae.toFixed(3), d.note])
  );

  renderLegend("legend-baseline", [
    ["Activation-based", "var(--chart-blue)"],
    ["Baseline", "var(--chart-gray)"],
  ]);
  const baselineEl = document.getElementById("chart-baseline");
  baselineEl.innerHTML = `<div class="bar-chart">${renderBars(BASELINE_COMPARISON, {
    formatValue: (v) => v.toFixed(3),
    domain: 1.0,
    colorFor: (d) => (d.group === "activation" ? "var(--chart-blue)" : "var(--chart-gray)"),
  })}</div>`;
  appendTableView(
    baselineEl,
    ["Detector", "TEST AUROC"],
    BASELINE_COMPARISON.map((d) => [d.label, d.value.toFixed(3)])
  );
}

// ---------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------

const tabDetector = document.getElementById("tab-detector");
const tabFindings = document.getElementById("tab-findings");
const panelDetector = document.getElementById("panel-detector");
const panelFindings = document.getElementById("panel-findings");

function switchTab(target) {
  const showFindings = target === "findings";
  panelDetector.hidden = showFindings;
  panelFindings.hidden = !showFindings;
  tabDetector.classList.toggle("active", !showFindings);
  tabFindings.classList.toggle("active", showFindings);
  tabDetector.setAttribute("aria-selected", String(!showFindings));
  tabFindings.setAttribute("aria-selected", String(showFindings));
}

tabDetector.addEventListener("click", () => switchTab("detector"));
tabFindings.addEventListener("click", () => switchTab("findings"));

renderFindings();
