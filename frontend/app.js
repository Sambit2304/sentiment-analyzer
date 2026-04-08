const form = document.getElementById("predictForm");
const textInput = document.getElementById("textInput");
const submitBtn = document.getElementById("submitBtn");
const submitLabel = document.getElementById("submitLabel");
const spinner = document.getElementById("spinner");
const statusMsg = document.getElementById("statusMsg");

const resultWrap = document.getElementById("resultWrap");
const sentimentBadge = document.getElementById("sentimentBadge");
const sentimentCodeEl = document.getElementById("sentimentCode");
const cleanedTextEl = document.getElementById("cleanedText");
const probabilitiesEl = document.getElementById("probabilities");

const metaBase =
  document.querySelector('meta[name="api-base"]')?.content || "/";

// When running the frontend locally via `http.server` (default port 5173),
// talk to the dev backend on :8000. In production, the backend serves the
// UI from the same origin, so we can call relative `/predict`.
const API_BASE_URL =
  location.port === "5173"
    ? "http://localhost:8000"
    : metaBase === "/"
      ? ""
      : metaBase;

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  spinner.classList.toggle("hidden", !isLoading);
  submitLabel.textContent = isLoading ? "Analyzing" : "Analyze";
}

function setStatus(message) {
  statusMsg.textContent = message || "";
}

function applySentimentStyles(sentiment) {
  // Reset to neutral base
  sentimentBadge.className =
    "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold bg-slate-800/60 border-slate-700/60 text-slate-100 transition duration-300";

  const normalized = String(sentiment || "Neutral").trim();

  if (normalized === "Positive") {
    sentimentBadge.className =
      "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold bg-emerald-100/10 border-emerald-300/20 text-emerald-100 transition duration-300";
  } else if (normalized === "Negative") {
    sentimentBadge.className =
      "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold bg-rose-100/10 border-rose-300/20 text-rose-100 transition duration-300";
  } else if (normalized === "Neutral") {
    sentimentBadge.className =
      "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold bg-slate-800/60 border-slate-700/60 text-slate-100 transition duration-300";
  } else if (normalized === "Irrelevant") {
    sentimentBadge.className =
      "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold bg-slate-800/60 border-slate-700/60 text-slate-200 transition duration-300";
  }
}

function formatProbabilities(probabilities) {
  if (!probabilities) return "";

  // Show top entries for readability
  const entries = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);

  return entries
    .map(([label, p]) => `${label}: ${(p * 100).toFixed(1)}%`)
    .join(" • ");
}

async function predict(text) {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    const detail = data?.detail || "Request failed.";
    throw new Error(detail);
  }
  return data;
}

async function onSubmit(e) {
  e.preventDefault();
  setStatus("");

  const raw = textInput.value || "";
  const trimmed = raw.trim();

  if (!trimmed) {
    setStatus("Please enter some text.");
    return;
  }

  // Clear previous result until we have a new one
  resultWrap.classList.add("opacity-0", "translate-y-1", "pointer-events-none");
  resultWrap.classList.remove("opacity-100", "translate-y-0");
  cleanedTextEl.textContent = "";
  probabilitiesEl.textContent = "";

  setLoading(true);
  try {
    const data = await predict(trimmed);

    const sentiment = data?.sentiment || "Neutral";
    const code = data?.sentiment_code ?? 0;
    const cleaned = data?.cleaned_text || "";
    const probabilities = data?.probabilities || null;

    sentimentCodeEl.textContent = String(code);
    cleanedTextEl.textContent = cleaned;
    probabilitiesEl.textContent = formatProbabilities(probabilities);

    applySentimentStyles(sentiment);

    // Reveal with a smooth transition
    sentimentBadge.textContent = sentiment;
    resultWrap.classList.remove("opacity-0", "translate-y-1", "pointer-events-none");
    resultWrap.classList.add("opacity-100", "translate-y-0", "pointer-events-auto");
  } catch (err) {
    setStatus(err?.message || "Server error. Please try again.");
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", onSubmit);

// Convenience: Ctrl+Enter to submit
textInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    form.requestSubmit();
  }
});

