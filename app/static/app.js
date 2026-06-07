const csvFileInput = document.getElementById("csvFile");
const uploadButton = document.getElementById("uploadButton");
const uploadStatus = document.getElementById("uploadStatus");

const statsButton = document.getElementById("statsButton");
const clearButton = document.getElementById("clearButton");
const clearStatus = document.getElementById("clearStatus");
const friendlyViewButton = document.getElementById("friendlyViewButton");
const jsonViewButton = document.getElementById("jsonViewButton");
const statsFriendlyOutput = document.getElementById("statsFriendlyOutput");
const statsOutput = document.getElementById("statsOutput");
const metricRows = document.getElementById("metricRows");
const metricExercises = document.getElementById("metricExercises");
const metricTopLift = document.getElementById("metricTopLift");

const chatWindow = document.getElementById("chatWindow");
const chatInput = document.getElementById("chatInput");
const chatButton = document.getElementById("chatButton");
const chatStatus = document.getElementById("chatStatus");
const clearChatButton = document.getElementById("clearChatButton");

let currentStatsData = null;
let statsViewMode = "friendly";

function setStatus(element, message, type = "") {
  element.textContent = message;
  element.classList.remove("error", "success");

  if (type) {
    element.classList.add(type);
  }
}

function buildChatBody(text) {
  const body = document.createElement("div");
  body.className = "body";

  const lines = String(text).split("\n");
  let activeList = null;

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (!line) {
      activeList = null;
      continue;
    }

    if (line.startsWith("- ")) {
      if (!activeList) {
        activeList = document.createElement("ul");
        body.appendChild(activeList);
      }

      const listItem = document.createElement("li");
      listItem.textContent = line.slice(2);
      activeList.appendChild(listItem);
      continue;
    }

    activeList = null;

    const paragraph = document.createElement("p");
    paragraph.textContent = line;
    body.appendChild(paragraph);
  }

  if (!body.childNodes.length) {
    body.textContent = String(text);
  }

  return body;
}

function appendChatMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;

  const label = document.createElement("span");
  label.className = "label";
  label.textContent = role === "user" ? "Du" : "AI";

  const body = buildChatBody(text);

  message.appendChild(label);
  message.appendChild(body);
  chatWindow.appendChild(message);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function updateHeroMetrics(data) {
  if (!metricRows || !metricExercises || !metricTopLift) {
    return;
  }

  if (!data) {
    metricRows.textContent = "-";
    metricExercises.textContent = "-";
    metricTopLift.textContent = "-";
    return;
  }

  metricRows.textContent = String(data.rows ?? "-");
  metricExercises.textContent = String(data.exercise_count ?? "-");

  const heaviestLift = data.heaviest_lift;

  if (!heaviestLift || !heaviestLift.exercise) {
    metricTopLift.textContent = "-";
    return;
  }

  metricTopLift.textContent = `${heaviestLift.exercise} ${Number(heaviestLift.weight).toFixed(1)} kg`;
}

function clearChat() {
  chatWindow.replaceChildren();
  setStatus(chatStatus, "Chatten är rensad.", "success");
}

function resetClearStatusMessage() {
  setStatus(clearStatus, "", "");
}

function getApiError(error, fallback) {
  if (error && typeof error === "object" && "detail" in error) {
    const detail = error.detail;

    if (typeof detail === "string") {
      return detail;
    }

    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => {
          if (typeof item === "string") {
            return item;
          }

          if (item && typeof item === "object" && "msg" in item) {
            return String(item.msg);
          }

          return null;
        })
        .filter(Boolean);

      if (messages.length > 0) {
        return messages.join(". ");
      }
    }

    return fallback;
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
      "success",
    );
    // Ny uppladdning innebär ny datastatus, så vi tar bort gammalt
    // "Datan rensad"-meddelande från tidigare state.
    resetClearStatusMessage();
    if (metricRows) {
      metricRows.textContent = String(data.rows);
    }
  } catch (error) {
    setStatus(
      uploadStatus,
      getApiError(error, "Kunde inte ladda upp filen."),
      "error",
    );
  } finally {
    uploadButton.disabled = false;
  }
}

async function loadStats() {
  statsButton.disabled = true;

  if (statsViewMode === "friendly") {
    statsFriendlyOutput.textContent = "Hämtar statistik...";
  } else {
    statsOutput.textContent = "Hämtar statistik...";
  }

  try {
    const response = await fetch("/data/stats");
    const data = await response.json();

    if (!response.ok) {
      throw data;
    }

    currentStatsData = data;
    renderStats();
  } catch (error) {
    const message = getApiError(error, "Kunde inte hämta statistik.");

    currentStatsData = null;

    if (statsViewMode === "friendly") {
      statsFriendlyOutput.textContent = message;
    } else {
      statsOutput.textContent = message;
    }
  } finally {
    statsButton.disabled = false;
  }
}

function renderStats() {
  if (!currentStatsData) {
    statsFriendlyOutput.textContent = "Ingen statistik hämtad ännu.";
    statsOutput.textContent = "Ingen statistik hämtad ännu.";
    updateHeroMetrics(null);
    return;
  }

  statsOutput.textContent = JSON.stringify(currentStatsData, null, 2);
  statsFriendlyOutput.innerHTML = buildFriendlyStatsHtml(currentStatsData);
  wireAccordionControls();
  updateHeroMetrics(currentStatsData);
}

function wireAccordionControls() {
  const controls = statsFriendlyOutput.querySelector(".accordion-controls");

  if (!controls) {
    return;
  }

  const openAllButton = controls.querySelector('[data-action="open-all"]');
  const closeAllButton = controls.querySelector('[data-action="close-all"]');

  if (openAllButton) {
    openAllButton.addEventListener("click", () => {
      statsFriendlyOutput
        .querySelectorAll("details.stat-accordion")
        .forEach((section) => {
          section.open = true;
        });
    });
  }

  if (closeAllButton) {
    closeAllButton.addEventListener("click", () => {
      statsFriendlyOutput
        .querySelectorAll("details.stat-accordion")
        .forEach((section) => {
          section.open = false;
        });
    });
  }
}

function buildFriendlyStatsHtml(data) {
  const rows = data.rows ?? "-";
  const exerciseCount = data.exercise_count ?? "-";
  const exercises = Array.isArray(data.exercises) ? data.exercises : [];

  const heaviestLift = data.heaviest_lift ?? {};
  const heaviestText = heaviestLift.exercise
    ? `${heaviestLift.exercise} ${heaviestLift.weight} kg x ${heaviestLift.reps} reps`
    : "-";

  const oneRmEntries = Object.entries(data.estimated_1rm_by_exercise ?? {});
  const oneRmTop = oneRmEntries.length > 0 ? oneRmEntries[0] : null;

  const volumeEntries = Object.entries(data.total_volume_by_exercise ?? {});
  const volumeTop = volumeEntries.length > 0 ? volumeEntries[0] : null;

  const exercisesHtml =
    exercises.length > 0
      ? exercises.map((exercise) => `<li>${exercise}</li>`).join("")
      : "<li>Inga övningar hittades.</li>";

  const oneRmHtml =
    oneRmEntries.length > 0
      ? oneRmEntries
          .map(
            ([exercise, value]) =>
              `<li>${exercise}: ${Number(value).toFixed(1)} kg</li>`,
          )
          .join("")
      : "<li>Ingen 1RM-data hittades.</li>";

  const volumeHtml =
    volumeEntries.length > 0
      ? volumeEntries
          .map(
            ([exercise, value]) =>
              `<li>${exercise}: ${Number(value).toFixed(1)} kg</li>`,
          )
          .join("")
      : "<li>Ingen volymdata hittades.</li>";

  const exerciseCountLabel = exercises.length;
  const oneRmCountLabel = oneRmEntries.length;
  const volumeCountLabel = volumeEntries.length;

  return `
    <div class="stats-grid">
      <div class="stat-card">
        <span class="label">Antal rader</span>
        <span class="value">${rows}</span>
      </div>
      <div class="stat-card">
        <span class="label">Unika övningar</span>
        <span class="value">${exerciseCount}</span>
      </div>
      <div class="stat-card">
        <span class="label">Tyngsta lyft</span>
        <span class="value">${heaviestText}</span>
      </div>
      <div class="stat-card">
        <span class="label">Högst estimerad 1RM</span>
        <span class="value">${oneRmTop ? `${oneRmTop[0]} (${Number(oneRmTop[1]).toFixed(1)} kg)` : "-"}</span>
      </div>
      <div class="stat-card">
        <span class="label">Högst total volym</span>
        <span class="value">${volumeTop ? `${volumeTop[0]} (${Number(volumeTop[1]).toFixed(1)} kg)` : "-"}</span>
      </div>
    </div>

    <div class="accordion-controls" role="group" aria-label="Kontroller för statistiksektioner">
      <button type="button" class="soft-button accordion-control-button" data-action="open-all">
        Öppna alla
      </button>
      <button type="button" class="soft-button accordion-control-button" data-action="close-all">
        Stäng alla
      </button>
    </div>

    <details class="stat-accordion">
      <summary>
        <span>Övningar i datan</span>
        <span class="count-badge">${exerciseCountLabel}</span>
      </summary>
      <div class="accordion-content">
        <ul class="stat-list">${exercisesHtml}</ul>
      </div>
    </details>

    <details class="stat-accordion">
      <summary>
        <span>Estimerad 1RM per övning</span>
        <span class="count-badge">${oneRmCountLabel}</span>
      </summary>
      <div class="accordion-content">
        <ul class="stat-list">${oneRmHtml}</ul>
      </div>
    </details>

    <details class="stat-accordion">
      <summary>
        <span>Total volym per övning</span>
        <span class="count-badge">${volumeCountLabel}</span>
      </summary>
      <div class="accordion-content">
        <ul class="stat-list">${volumeHtml}</ul>
      </div>
    </details>
  `;
}

function setStatsView(mode) {
  statsViewMode = mode;

  const isFriendly = mode === "friendly";

  friendlyViewButton.classList.toggle("active", isFriendly);
  jsonViewButton.classList.toggle("active", !isFriendly);

  statsFriendlyOutput.classList.toggle("hidden", !isFriendly);
  statsOutput.classList.toggle("hidden", isFriendly);

  renderStats();
}

async function clearData() {
  clearButton.disabled = true;
  setStatus(clearStatus, "Rensar data...", "");

  try {
    const response = await fetch("/data/clear", {
      method: "DELETE",
    });

    const data = await response.json();

    if (!response.ok) {
      throw data;
    }

    setStatus(
      clearStatus,
      `Datan rensad. Borttagna rader: ${data.rows_removed}.`,
      "success",
    );
    currentStatsData = null;
    renderStats();
    updateHeroMetrics(null);
    chatStatus.textContent = "";
  } catch (error) {
    setStatus(
      clearStatus,
      getApiError(error, "Kunde inte rensa datan."),
      "error",
    );
  } finally {
    clearButton.disabled = false;
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
clearButton.addEventListener("click", clearData);
friendlyViewButton.addEventListener("click", () => setStatsView("friendly"));
jsonViewButton.addEventListener("click", () => setStatsView("json"));
chatButton.addEventListener("click", askAi);

if (clearChatButton) {
  clearChatButton.addEventListener("click", clearChat);
}

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    askAi();
  }
});

setStatsView("friendly");
updateHeroMetrics(null);
