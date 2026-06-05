const csvFileInput = document.getElementById("csvFile");
const uploadButton = document.getElementById("uploadButton");
const uploadStatus = document.getElementById("uploadStatus");

const statsButton = document.getElementById("statsButton");
const statsOutput = document.getElementById("statsOutput");

const chatWindow = document.getElementById("chatWindow");
const chatInput = document.getElementById("chatInput");
const chatButton = document.getElementById("chatButton");
const chatStatus = document.getElementById("chatStatus");

function setStatus(element, message, type = "") {
  element.textContent = message;
  element.classList.remove("error", "success");

  if (type) {
    element.classList.add(type);
  }
}

function appendChatMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;

  const label = document.createElement("span");
  label.className = "label";
  label.textContent = role === "user" ? "Du" : "AI";

  const body = document.createElement("div");
  body.textContent = text;

  message.appendChild(label);
  message.appendChild(body);
  chatWindow.appendChild(message);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function getApiError(error, fallback) {
  if (error && typeof error === "object" && "detail" in error) {
    return String(error.detail);
  }

  return fallback;
}

async function uploadCsv() {
  const selectedFile = csvFileInput.files?.[0];

  if (!selectedFile) {
    setStatus(uploadStatus, "Välj en CSV-fil först.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);

  uploadButton.disabled = true;
  setStatus(uploadStatus, "Laddar upp fil...", "");

  try {
    const response = await fetch("/data/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw data;
    }

    setStatus(
      uploadStatus,
      `Uppladdning klar. Rader: ${data.rows}, kolumner: ${data.columns.length}.`,
      "success"
    );
  } catch (error) {
    setStatus(uploadStatus, getApiError(error, "Kunde inte ladda upp filen."), "error");
  } finally {
    uploadButton.disabled = false;
  }
}

async function loadStats() {
  statsButton.disabled = true;
  statsOutput.textContent = "Hämtar statistik...";

  try {
    const response = await fetch("/data/stats");
    const data = await response.json();

    if (!response.ok) {
      throw data;
    }

    statsOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    statsOutput.textContent = getApiError(error, "Kunde inte hämta statistik.");
  } finally {
    statsButton.disabled = false;
  }
}

async function askAi() {
  const question = chatInput.value.trim();

  if (!question) {
    setStatus(chatStatus, "Skriv en fråga först.", "error");
    return;
  }

  appendChatMessage("user", question);
  chatInput.value = "";

  chatButton.disabled = true;
  chatInput.disabled = true;
  setStatus(chatStatus, "Tänker...", "");

  try {
    const response = await fetch("/ai/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw data;
    }

    appendChatMessage("ai", data.answer);
    setStatus(chatStatus, "", "");
  } catch (error) {
    const message = getApiError(error, "Något gick fel vid AI-anropet.");
    appendChatMessage("ai", `Fel: ${message}`);
    setStatus(chatStatus, message, "error");
  } finally {
    chatButton.disabled = false;
    chatInput.disabled = false;
    chatInput.focus();
  }
}

uploadButton.addEventListener("click", uploadCsv);
statsButton.addEventListener("click", loadStats);
chatButton.addEventListener("click", askAi);

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    askAi();
  }
});
