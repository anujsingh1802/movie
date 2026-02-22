const API_BASE = "";

const state = {
    sessionId: null,
    feature: null,
    value: null,
    pending: false,
    ended: false,
};

const ui = {
    question: document.getElementById("question"),
    result: document.getElementById("result"),
    remaining: document.getElementById("remaining"),
    buttons: document.querySelector(".buttons"),
    yes: document.querySelector(".yes-btn"),
    no: document.querySelector(".no-btn"),
};

function lockButtons(locked) {
    state.pending = locked;
    ui.yes.disabled = locked || state.ended;
    ui.no.disabled = locked || state.ended;
}

function showError(message) {
    ui.question.innerText = "⚠️ Request failed";
    ui.result.innerText = message || "Unknown error";
}

async function api(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, options);
    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(payload.detail || payload.error || "API error");
    }

    return payload;
}

function renderQuestion(payload) {
    state.feature = payload.feature;
    state.value = payload.value;

    ui.question.innerText = payload.question;
    ui.remaining.innerText = `Remaining Movies: ${payload.remaining}`;
    ui.result.innerText = `IG: ${Number(payload.information_gain).toFixed(3)} bits`;
}

function renderResult(payload) {
    state.ended = true;
    ui.question.innerText = "🎉 I guessed it!";
    ui.result.innerText = payload.result;
    ui.remaining.innerText = `Questions asked: ${payload.questions_asked}`;
    ui.buttons.style.display = "none";
}

async function startGame() {
    lockButtons(true);
    state.ended = false;
    ui.buttons.style.display = "flex";
    ui.result.innerText = "";

    try {
        const payload = await api("/start", { method: "POST" });
        state.sessionId = payload.session_id;
        await ask();
    } catch (error) {
        showError(error.message);
    } finally {
        lockButtons(false);
    }
}

async function ask() {
    if (!state.sessionId || state.ended) return;

    lockButtons(true);
    try {
        const payload = await api(`/ask?session_id=${encodeURIComponent(state.sessionId)}`);

        if (payload.result) {
            renderResult(payload);
            return;
        }

        renderQuestion(payload);
    } catch (error) {
        showError(error.message);
    } finally {
        lockButtons(false);
    }
}

async function sendAnswer(answer) {
    if (state.pending || state.ended || !state.sessionId || !state.feature) return;

    lockButtons(true);
    try {
        await api("/answer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: state.sessionId,
                feature: state.feature,
                value: state.value,
                answer,
            }),
        });

        await ask();
    } catch (error) {
        showError(error.message);
    } finally {
        lockButtons(false);
    }
}

window.sendAnswer = sendAnswer;
startGame();