// Movie Assistant frontend controller.
(() => {
  const API = "";

  const state = {
    sessionId: null,
    currentQuestion: null,
    decisionPath: [],
    chart: null,
    statsSupported: null,
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
    statsLoading: document.getElementById("stats-loading"),
    statsTotal: document.getElementById("stats-total"),
    statsTheoretical: document.getElementById("stats-theoretical"),
    statsActual: document.getElementById("stats-actual"),
    featureChart: document.getElementById("feature-chart"),
  };

  function normalizeError(err) {
    if (!err) return "Unknown error";
    if (typeof err === "string") return err;
    if (err.message) return err.message;
    return JSON.stringify(err);
  }

  async function apiFetch(path, options = {}) {
    const response = await fetch(`${API}${path}`, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = data.detail ?? data.error ?? `Request failed: ${response.status}`;
      const message = typeof detail === "string" ? detail : JSON.stringify(detail);
      throw new Error(message);
    }
    return data;
  }

  function setGlobalError(message = "") {
    el.globalError.textContent = message;
  }

  function setGuessLoading(loading) {
    el.guessLoading.classList.toggle("hidden", !loading);
    el.startBtn.disabled = loading;
    const lockAnswerButtons = loading || !state.currentQuestion;
    el.yesBtn.disabled = lockAnswerButtons;
    el.noBtn.disabled = lockAnswerButtons;
  }

  function setRecommendLoading(loading) {
    el.recommendLoading.classList.toggle("hidden", !loading);
    el.nlpBtn.disabled = loading;
    el.recommendBtn.disabled = loading;
    el.voiceBtn.disabled = loading;
  }

  function setStatsLoading(loading) {
    el.statsLoading.classList.toggle("hidden", !loading);
    el.refreshStatsBtn.disabled = loading;
  }

  async function startSession() {
    setGlobalError("");
    setGuessLoading(true);
    try {
      let startData;
      try {
        startData = await apiFetch("/start"); // expected GET
      } catch {
        startData = await apiFetch("/start", { method: "POST" }); // fallback
      }

      state.sessionId = startData.session_id || null;
      state.currentQuestion = null;
      state.decisionPath = [];
      renderDecisionPath();
      el.guessResult.textContent = "";

      await fetchNextQuestion();
    } catch (err) {
      setGlobalError(normalizeError(err));
    } finally {
      setGuessLoading(false);
    }
  }

  async function fetchNextQuestion() {
    if (!state.sessionId) return;

    setGuessLoading(true);
    try {
      const data = await apiFetch(`/ask?session_id=${encodeURIComponent(state.sessionId)}`);

      if (data.result) {
        state.currentQuestion = null;
        el.questionText.textContent = "🎉 I guessed your movie!";
        el.guessResult.innerHTML = `<strong>${data.result}</strong>`;
        el.remainingCount.textContent = data.remaining ?? "-";
        el.yesCount.textContent = "-";
        el.noCount.textContent = "-";
        el.igValue.textContent = "-";
        return;
      }

      // Guard against malformed responses to avoid undefined payloads.
      if (!data.feature || !data.value) {
        state.currentQuestion = null;
        throw new Error("/ask response did not include feature/value fields.");
      }

      const questionText = data.question_text || data.question || `Is ${data.feature} = ${data.value}?`;
      state.currentQuestion = {
        feature: data.feature,
        value: data.value,
        questionText,
      };

      el.questionText.textContent = questionText;
      el.remainingCount.textContent = data.remaining ?? "-";
      el.yesCount.textContent = data.yes_count ?? "-";
      el.noCount.textContent = data.no_count ?? "-";
      el.igValue.textContent = Number(data.information_gain ?? 0).toFixed(4);
    } catch (err) {
      setGlobalError(normalizeError(err));
    } finally {
      setGuessLoading(false);
    }
  }

  async function sendAnswer(answer) {
    if (!state.currentQuestion || !state.sessionId) return;

    setGuessLoading(true);
    setGlobalError("");

    try {
      const payload = {
        session_id: state.sessionId,
        feature: state.currentQuestion.feature,
        value: state.currentQuestion.value,
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
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
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

  async function detectStatsSupport() {
    try {
      const openapi = await apiFetch("/openapi.json");
      state.statsSupported = Boolean(openapi?.paths?.["/stats"]);
    } catch {
      state.statsSupported = false;
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

  function renderFeatureChart(featureImportance) {
    const labels = Object.keys(featureImportance);
    const values = Object.values(featureImportance);

    // Preferred rendering with Chart.js if available.
    if (window.Chart) {
      if (state.chart) state.chart.destroy();
      state.chart = new Chart(el.featureChart, {
        type: "bar",
        data: {
          labels,
          datasets: [{
            label: "Feature Importance",
            data: values,
            backgroundColor: "rgba(93, 139, 255, 0.75)",
            borderColor: "rgba(93, 139, 255, 1)",
            borderWidth: 1,
          }],
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: "#eaf0ff" } },
            y: { ticks: { color: "#eaf0ff" }, beginAtZero: true },
          },
        },
      });
      return;
    }

    // Fallback rendering if third-party script is blocked.
    const wrap = el.featureChart.parentElement;
    wrap.innerHTML = '<div id="fallback-bars" class="fallback-bars"></div>';
    const bars = wrap.querySelector("#fallback-bars");
    const max = Math.max(...values, 1);
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
    await detectStatsSupport();
    await loadStats();
  }

  init();
})();