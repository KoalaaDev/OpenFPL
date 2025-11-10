const form = document.getElementById("optimize-form");
const teamInsights = document.getElementById("team-insights");
const startingXiContainer = document.getElementById("starting-xi");
const benchContainer = document.getElementById("bench");
const chipContainer = document.getElementById("chip-recommendations");
const teamTitle = document.getElementById("team-title");
const savedAt = document.getElementById("saved-at");
const teamBudget = document.getElementById("team-budget");
const teamValue = document.getElementById("team-value");
const teamBank = document.getElementById("team-bank");
const feedback = document.getElementById("form-feedback");
const loadingOverlay = document.getElementById("loading-overlay");
const loadSavedButton = document.getElementById("load-saved");
const currentStarting = document.getElementById("current-starting");
const currentBench = document.getElementById("current-bench");

const currencyFormat = new Intl.NumberFormat("en-GB", {
  style: "currency",
  currency: "GBP",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const numberFormat = new Intl.NumberFormat("en-GB", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const teamId = form["team-id"].value.trim();
  const season = form["season"].value.trim();
  const gameweek = form["gameweek"].value.trim();

  if (!teamId) {
    return showError("Please enter a valid FPL manager ID before optimising.");
  }

  await optimiseTeam({ teamId, season, gameweek });
});

loadSavedButton.addEventListener("click", async () => {
  const teamId = form["team-id"].value.trim();
  if (!teamId) {
    return showError("Enter your team ID to load saved data.");
  }
  toggleLoading(true);
  try {
    const response = await fetch(`/api/saved/${teamId}`);
    if (!response.ok) {
      const message = (await response.json()).detail || "No saved squad found yet.";
      showError(message);
      return;
    }
    const data = await response.json();
    renderOptimisedTeam(data);
  } catch (error) {
    console.error(error);
    showError("We couldn't load your saved squad. Please try again.");
  } finally {
    toggleLoading(false);
  }
});

async function optimiseTeam({ teamId, season, gameweek }) {
  toggleLoading(true);
  clearFeedback();
  try {
    const response = await fetch("/api/optimize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        team_id: Number(teamId),
        season,
        gameweek: Number(gameweek),
      }),
    });

    if (!response.ok) {
      const message = (await response.json()).detail || "Failed to optimise squad.";
      throw new Error(message);
    }

    const data = await response.json();
    renderOptimisedTeam(data);
  } catch (error) {
    console.error(error);
    showError(error.message || "Unexpected error while optimising your XI.");
  } finally {
    toggleLoading(false);
  }
}

function renderOptimisedTeam(data) {
  teamTitle.textContent = `${data.team_name} · GW${data.gameweek}`;
  savedAt.textContent = data.saved_at ? `Saved ${new Date(data.saved_at).toLocaleString()}` : "";
  teamBudget.textContent = currencyFormat.format(data.budget || 0);
  teamValue.textContent = currencyFormat.format(data.value || 0);
  teamBank.textContent = currencyFormat.format(data.bank || 0);

  renderPlayers(startingXiContainer, data.optimized_starting, {
    badge: (player) => {
      if (data.captain && player.element_id === data.captain.element_id) {
        return `<span class="badge-pill">C</span>`;
      }
      if (data.vice_captain && player.element_id === data.vice_captain.element_id) {
        return `<span class="badge-pill">VC</span>`;
      }
      return `<span class="badge">${player.position}</span>`;
    },
  });

  renderPlayers(benchContainer, data.optimized_bench, {
    badge: (player) => `<span class="badge">${player.position}</span>`,
  });

  renderChips(data.chip_recommendations || {});
  renderCurrentTeam(data.current_team_projection || {});

  teamInsights.classList.remove("hidden");
  teamInsights.classList.add("fade-in");
}

function renderPlayers(container, players, options = {}) {
  container.innerHTML = "";
  if (!players || !players.length) {
    container.innerHTML = `<p class="text-sm text-slate-500">No players available.</p>`;
    return;
  }

  players.forEach((player) => {
    const row = document.createElement("div");
    row.className = "player-row fade-in";
    const badgeContent = options.badge ? options.badge(player) : `<span class="badge">${player.position}</span>`;
    row.innerHTML = `
      <div class="flex flex-col gap-1">
        <div class="flex flex-wrap items-center gap-2 text-sm font-medium text-white">
          <span>${player.name}</span>
          ${badgeContent}
        </div>
        <p class="text-xs uppercase tracking-[0.25em] text-slate-400">${player.team_name}</p>
      </div>
      <div class="stat">
        <span>Price</span>
        <strong>${currencyFormat.format(player.price)}</strong>
      </div>
      <div class="stat">
        <span>Projected</span>
        <strong>${numberFormat.format(player.prediction)}</strong>
      </div>
    `;
    container.appendChild(row);
  });
}

function renderChips(recommendations) {
  chipContainer.innerHTML = "";
  const entries = Object.entries(recommendations);
  if (!entries.length) {
    chipContainer.innerHTML = `<p class="text-sm text-slate-500">No chip recommendations available.</p>`;
    return;
  }
  entries.forEach(([chip, message]) => {
    const card = document.createElement("div");
    card.className = "chip-card fade-in";
    card.innerHTML = `
      <h5>${toTitleCase(chip)}</h5>
      <p>${message}</p>
    `;
    chipContainer.appendChild(card);
  });
}

function renderCurrentTeam(projection) {
  currentStarting.innerHTML = "";
  currentBench.innerHTML = "";
  if (!projection || (!projection.starting && !projection.bench)) {
    currentStarting.innerHTML = `<p class="text-xs text-slate-500">No current team data available.</p>`;
    currentBench.innerHTML = "";
    return;
  }
  const renderGroup = (target, players) => {
    if (!players || !players.length) {
      target.innerHTML = `<p class="text-xs text-slate-500">No data.</p>`;
      return;
    }
    target.innerHTML = "";
    players.forEach((player) => {
      const row = document.createElement("div");
      row.className = "player-row fade-in";
      row.innerHTML = `
        <div class="flex flex-col gap-1">
          <div class="flex flex-wrap items-center gap-2 text-xs font-semibold text-white">
            <span>${player.name}</span>
            ${player.is_captain ? '<span class="badge-pill">C</span>' : ""}
            ${player.is_vice_captain ? '<span class="badge-pill">VC</span>' : ""}
          </div>
          <p class="text-[0.65rem] uppercase tracking-[0.25em] text-slate-500">${player.team_name}</p>
        </div>
        <div class="stat">
          <span>Multiplier</span>
          <strong>${player.multiplier}×</strong>
        </div>
        <div class="stat">
          <span>Projected</span>
          <strong>${numberFormat.format(player.prediction || 0)}</strong>
        </div>
      `;
      target.appendChild(row);
    });
  };
  renderGroup(currentStarting, projection.starting);
  renderGroup(currentBench, projection.bench);
}

function showError(message) {
  feedback.textContent = message;
  feedback.classList.remove("hidden");
  feedback.classList.add("fade-in");
}

function clearFeedback() {
  feedback.classList.add("hidden");
  feedback.textContent = "";
}

function toggleLoading(show) {
  loadingOverlay.classList.toggle("hidden", !show);
  if (show) {
    loadingOverlay.classList.add("grid");
  } else {
    loadingOverlay.classList.remove("grid");
  }
}

function toTitleCase(text) {
  return text
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
