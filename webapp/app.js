const modelSelect = document.getElementById("model-select");
const promptInput = document.getElementById("prompt-input");
const analyzeButton = document.getElementById("analyze-button");
const statusLine = document.getElementById("status-line");
const resultsSection = document.getElementById("results");

let models = [];

function setStatus(message, isError = false) {
  statusLine.textContent = message;
  statusLine.classList.toggle("error", isError);
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error(`Failed to load models (${response.status})`);
    models = await response.json();

    modelSelect.innerHTML = "";
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model.hf_name;
      option.textContent = model.sae_feature_available
        ? model.cache_label
        : `${model.cache_label} (no SAE-feature detector)`;
      modelSelect.appendChild(option);
    }
    modelSelect.disabled = false;
    analyzeButton.disabled = false;
  } catch (err) {
    setStatus(`Could not load model list: ${err.message}`, true);
  }
}

function formatScore(value) {
  return Number.isFinite(value) ? value.toFixed(2) : String(value);
}

function scoreBarWidth(score, threshold) {
  // Cheap visual proxy, not a real probability: position the score
  // relative to 2x its threshold so both detectors well under and well
  // over threshold still render sensibly.
  const denom = Math.max(Math.abs(threshold) * 2, 1e-6);
  const pct = (Math.abs(score) / denom) * 100;
  return Math.min(100, Math.max(2, pct));
}

function renderScoreCard(flagged, score, threshold) {
  const badgeClass = flagged ? "flagged" : "clear";
  const badgeText = flagged ? "Flagged" : "Clear";
  const barWidth = scoreBarWidth(score, threshold);
  return `
    <span class="badge ${badgeClass}">${badgeText}</span>
    <div class="score-row">score ${formatScore(score)} &middot; threshold ${formatScore(threshold)}</div>
    <div class="score-bar"><div class="score-bar-fill ${badgeClass === "flagged" ? "flagged" : ""}" style="width:${barWidth}%"></div></div>
  `;
}

function renderKeyword(result) {
  let html = renderScoreCard(result.flagged, result.score, result.threshold);
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
  return renderScoreCard(result.flagged, result.score, result.threshold);
}

function renderDenseDirection(result) {
  return (
    renderScoreCard(result.flagged, result.score, result.threshold) +
    `<p class="detail-label">Layer</p><p>${result.layer}</p>`
  );
}

function renderSAEFeature(result) {
  if (!result.available) {
    return `<span class="badge unavailable">Unavailable</span><p class="empty-note">${result.reason}</p>`;
  }
  let html = renderScoreCard(result.flagged, result.score, result.threshold);
  html += `<p class="detail-label">Top contributing features</p><ul class="feature-list">${result.top_features
    .map((f) => `<li>layer ${f.layer}, feature ${f.feature}: ${formatScore(f.contribution)}</li>`)
    .join("")}</ul>`;
  return html;
}

const RENDERERS = {
  keyword: renderKeyword,
  perplexity: renderPerplexity,
  dense_direction: renderDenseDirection,
  sae_feature: renderSAEFeature,
};

function renderResults(analysis) {
  for (const [detector, render] of Object.entries(RENDERERS)) {
    const card = document.getElementById(`card-${detector}`);
    const body = card.querySelector(".card-body");
    const result = analysis[detector];
    body.innerHTML = result ? render(result) : `<p class="empty-note">Not requested.</p>`;
  }
  resultsSection.hidden = false;
}

async function analyze() {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    setStatus("Enter a prompt first.", true);
    return;
  }

  analyzeButton.disabled = true;
  setStatus("Running detectors… this can take up to a minute the first time a model loads.");

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, model_name: modelSelect.value }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);

    renderResults(body);
    setStatus("Done.");
  } catch (err) {
    setStatus(`Error: ${err.message}`, true);
  } finally {
    analyzeButton.disabled = false;
  }
}

analyzeButton.addEventListener("click", analyze);
loadModels();
