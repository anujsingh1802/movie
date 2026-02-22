const API_BASE = "";

let currentFeature = null;
let currentValue = null;

async function startGame() {
    await fetch(`${API_BASE}/start`);
    loadQuestion();
}

async function loadQuestion() {
    const response = await fetch(`${API_BASE}/ask`);
    const data = await response.json();

    if (data.result) {
        document.getElementById("question").innerText = "🎉 I guessed it!";
        document.getElementById("result").innerText = data.result;
        document.querySelector(".buttons").style.display = "none";
        return;
    }

    currentFeature = data.feature;
    currentValue = data.value;

    document.getElementById("question").innerText =
        `Is ${currentFeature} equal to "${currentValue}"?`;

    document.getElementById("remaining").innerText =
        `Remaining Movies: ${data.remaining}`;
}

async function sendAnswer(answer) {
    await fetch(
        `${API_BASE}/answer?feature=${currentFeature}&value=${currentValue}&answer=${answer}`,
        { method: "POST" }
    );

    loadQuestion();
}

startGame();