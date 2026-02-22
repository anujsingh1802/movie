const API = "";

  const state = {
    sessionId: null,
    currentQuestion: null,
    decisionPath: [],
    chart: null,
    statsSupported: null,
    nlpSupported: null,
    recommendSupported: null,
  };

  const el = {
    globalError: document.getElementById("global-error"),
    startBtn: document.getElementById("start-btn"),
    yesBtn: document.getElementById("yes-btn"),
    noBtn: document.getElementById("no-btn"),
    guessLoading: document.getElementById("guess-loading"),
    questionText: document.getElementById("question-text"),
    remainingCount: document.getElementById("remaining-count"),
    yesCount: document.getElementById("yes-count"),
    noCount: document.getElementById("no-count"),
    igValue: document.getElementById("ig-value"),
    guessResult: document.getElementById("guess-result"),
    decisionPath: document.getElementById("decision-path"),

    nlpInput: document.getElementById("nlp-input"),
    voiceBtn: document.getElementById("voice-btn"),
    nlpBtn: document.getElementById("nlp-btn"),
    recommendBtn: document.getElementById("recommend-btn"),
    recommendLoading: document.getElementById("recommend-loading"),
    nlpOutput: document.getElementById("nlp-output"),
  recommendations: document.getElementById("recommendations"),

  refreshStatsBtn: document.getElementById("refresh-stats-btn"),
  statsTotal: document.getElementById("stats-total"),
  statsTheoretical: document.getElementById("stats-theoretical"),
  statsActual: document.getElementById("stats-actual"),
  featureChart: document.getElementById("feature-chart"),
};

async function sendAnswer(answer) {
  setGuessLoading(true);
  setGlobalError("");

  try {
    const payload = {
      sessionId: state.sessionId,
      answer,
    };

    await apiFetch("/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

      state.decisionPath.push(`${state.currentQuestion.questionText} → ${answer ? "Yes" : "No"}`);
      renderDecisionPath();

      await fetchNextQuestion();
    } catch (err) {
      setGlobalError(normalizeError(err));
    } finally {
      setGuessLoading(false);
    }
  }

  function renderDecisionPath() {
    el.decisionPath.innerHTML = "";
    for (const step of state.decisionPath) {
      const li = document.createElement("li");
      li.textContent = step;
      el.decisionPath.appendChild(li);
    }
  }

  async function runNlpQuery() {
    if (!state.nlpSupported || !state.recommendSupported) {
      el.nlpOutput.innerHTML = '<div class="output-box">NLP/recommend endpoints are not available on this backend.</div>';
      return;
    }

    const query = el.nlpInput.value.trim();
    if (!query) return;

    setRecommendLoading(true);
    setGlobalError("");

    try {
      const data = await apiFetch("/nlp-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const rows = Array.isArray(data.rows) ? data.rows : [];
      el.nlpOutput.innerHTML = `
        <p><strong>Query:</strong> ${query}</p>
        <p><strong>Answer:</strong> ${data.answer || "No textual answer."}</p>
        <p><strong>Items:</strong> ${rows.length}</p>
      `;
    } catch (err) {
      setGlobalError(normalizeError(err));
    } finally {
      setRecommendLoading(false);
    }
  }

  async function fetchRecommendations() {
    if (!state.nlpSupported || !state.recommendSupported) {
      el.recommendations.innerHTML = '<div class="output-box">NLP/recommend endpoints are not available on this backend.</div>';
      return;
    }

    const movie = el.nlpInput.value.trim();
    if (!movie) return;

    setRecommendLoading(true);
    setGlobalError("");

    try {
      const data = await apiFetch(`/recommend?movie=${encodeURIComponent(movie)}`);
      const recs = Array.isArray(data.recommendations) ? data.recommendations : (Array.isArray(data) ? data : []);

      el.recommendations.innerHTML = "";
      recs.slice(0, 5).forEach((item) => {
        const card = document.createElement("article");
        card.className = "card";
        card.innerHTML = `
          <h4>${item.title || item.Title || "Unknown Title"}</h4>
          <p><strong>Reason:</strong> ${item.reason || item.explanation || "Similar profile."}</p>
        `;
        el.recommendations.appendChild(card);
      });
    } catch (err) {
      setGlobalError(normalizeError(err));
  } finally {
    setRecommendLoading(false);
  }
}

function setupVoiceInput() {
  if (!("webkitSpeechRecognition" in window) && !("SpeechRecognition" in window)) {
    el.voiceBtn.disabled = true;
    el.voiceBtn.textContent = "🎤 Not Supported";
    return;
  }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    el.voiceBtn.addEventListener("click", () => {
      setGlobalError("");
      recognition.start();
    });

    recognition.onresult = (event) => {
      const transcript = event.results?.[0]?.[0]?.transcript || "";
      if (transcript) el.nlpInput.value = transcript;
    };

    recognition.onerror = (event) => {
      setGlobalError(`Voice input error: ${event.error}`);
    };
  }

  function applyNlpRecommendationUnsupportedState() {
    el.nlpBtn.disabled = true;
    el.recommendBtn.disabled = true;
    el.nlpBtn.title = "Backend /nlp-query endpoint not available";
    el.recommendBtn.title = "Backend /recommend endpoint not available";
    el.nlpOutput.innerHTML = '<p>NLP/recommend endpoints are not available on this backend.</p>';
    el.recommendations.innerHTML = '<div class="output-box">N/A</div>';
  }

  async function detectCapabilities() {
    try {
      const openapi = await apiFetch("/openapi.json");
      state.nlpSupported = Boolean(openapi?.paths?.["/nlp-query"]);
      state.recommendSupported = Boolean(openapi?.paths?.["/recommend"]);
      state.statsSupported = Boolean(openapi?.paths?.["/stats"]);
    } catch {
      state.nlpSupported = false;
      state.recommendSupported = false;
      state.statsSupported = false;
    }

    if (!state.nlpSupported || !state.recommendSupported) {
      applyNlpRecommendationUnsupportedState();
    }

    if (!state.statsSupported) {
      el.refreshStatsBtn.disabled = true;
      el.refreshStatsBtn.title = "Backend /stats endpoint not available";
      el.statsTotal.textContent = "N/A";
      el.statsTheoretical.textContent = "N/A";
      el.statsActual.textContent = "N/A";
      el.featureChart.parentElement.innerHTML = '<div class="output-box">/stats endpoint is not available on this backend.</div>';
    }
  }

  async function loadStats() {
    if (!state.statsSupported) return;

    setStatsLoading(true);
    setGlobalError("");
    try {
      const data = await apiFetch("/stats");
      el.statsTotal.textContent = data.total_movies ?? "-";
      el.statsTheoretical.textContent = data.theoretical_min ?? "-";
      el.statsActual.textContent = data.actual_average ?? "-";
      renderFeatureChart(data.feature_importance || {});
    } catch (err) {
      setGlobalError(normalizeError(err));
  } finally {
    setStatsLoading(false);
  }
}

function renderFeatureChart(data) {
  const labels = Object.keys(data);
  const values = Object.values(data);
  if (!labels.length) return;

  const max = Math.max(...values);
  const bars = document.createElement("div");
  bars.className = "fallback-bars";
  labels.forEach((label, i) => {
      const row = document.createElement("div");
      row.className = "fallback-row";
      const pct = (Number(values[i]) / max) * 100;
      row.innerHTML = `
        <span>${label}</span>
        <div class="fallback-bar"><div class="fallback-fill" style="width:${pct}%"></div></div>
        <span>${Number(values[i]).toFixed(3)}</span>
      `;
      bars.appendChild(row);
    });
  }

  function bindEvents() {
    el.startBtn.addEventListener("click", startSession);
    el.yesBtn.addEventListener("click", () => sendAnswer(true));
    el.noBtn.addEventListener("click", () => sendAnswer(false));
    el.nlpBtn.addEventListener("click", runNlpQuery);
    el.recommendBtn.addEventListener("click", fetchRecommendations);
    el.refreshStatsBtn.addEventListener("click", loadStats);
  }

  async function init() {
    bindEvents();
    setupVoiceInput();
    await detectCapabilities();
    await loadStats();
  }

  init();