const form = document.getElementById("optimize-form");
const teamInsights = document.getElementById("team-insights");
const pitchOptimizedContainer = document.getElementById("pitch-optimized");
const pitchCurrentContainer = document.getElementById("pitch-current");
const benchOptimizedContainer = document.getElementById("optimized-bench");
const changeSummaryContainer = document.getElementById("change-summary");
const chipContainer = document.getElementById("chip-recommendations");
const teamTitle = document.getElementById("team-title");
const savedAt = document.getElementById("saved-at");
const teamBudget = document.getElementById("team-budget");
const teamValue = document.getElementById("team-value");
const teamBank = document.getElementById("team-bank");
const feedback = document.getElementById("form-feedback");
const loadingOverlay = document.getElementById("loading-overlay");
const loadSavedButton = document.getElementById("load-saved");
const currentBench = document.getElementById("current-bench");
const freeTransfersInput = document.getElementById("free-transfers");

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
  const freeTransfersRaw = freeTransfersInput ? freeTransfersInput.value.trim() : "";
  const freeTransfers = freeTransfersRaw === "" ? undefined : Number(freeTransfersRaw);

  if (!teamId) {
    return showError("Please enter a valid FPL manager ID before optimising.");
  }

  await optimiseTeam({ teamId, season, gameweek, freeTransfers });
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

async function optimiseTeam({ teamId, season, gameweek, freeTransfers }) {
  toggleLoading(true);
  clearFeedback();
  try {
    const payload = {
      team_id: Number(teamId),
      season,
      gameweek: Number(gameweek),
    };
    if (freeTransfers !== undefined && !Number.isNaN(freeTransfers)) {
      payload.free_transfers = freeTransfers;
    }
    const response = await fetch("/api/optimize", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
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
  if (form && form["gameweek"] && data.gameweek !== undefined && data.gameweek !== null) {
    form["gameweek"].value = String(data.gameweek);
  }
  savedAt.textContent = data.saved_at ? `Saved ${new Date(data.saved_at).toLocaleString()}` : "";
  teamBudget.textContent = currencyFormat.format(data.budget || 0);
  teamValue.textContent = currencyFormat.format(data.value || 0);
  teamBank.textContent = currencyFormat.format(data.bank || 0);
  const currentProjection = data.current_team_projection || {};
  const currentStarting = currentProjection.starting || [];
  const currentBenchPlayers = currentProjection.bench || [];

  const currentStartingIds = new Set(currentStarting.map((player) => player.element_id));
  const currentBenchIds = new Set(currentBenchPlayers.map((player) => player.element_id));
  const optimizedStartingIds = new Set(data.optimized_starting.map((player) => player.element_id));
  const optimizedBenchIds = new Set(data.optimized_bench.map((player) => player.element_id));

  const statusById = new Map();
  const currentStatusById = new Map();
  const changes = {
    promoted: [],
    newToXI: [],
    newToBench: [],
    demoted: [],
    removed: [],
  };

  data.optimized_starting.forEach((player) => {
    let status = "new";
    if (currentStartingIds.has(player.element_id)) {
      status = "kept";
    } else if (currentBenchIds.has(player.element_id)) {
      status = "promoted";
      changes.promoted.push(player);
    } else {
      changes.newToXI.push(player);
    }
    statusById.set(player.element_id, status);
  });

  data.optimized_bench.forEach((player) => {
    let status = "bench-new";
    if (currentStartingIds.has(player.element_id)) {
      status = "demoted";
      changes.demoted.push(player);
    } else if (currentBenchIds.has(player.element_id)) {
      status = "bench-kept";
    } else {
      changes.newToBench.push(player);
    }
    statusById.set(player.element_id, status);
  });

  currentStarting.forEach((player) => {
    if (!optimizedStartingIds.has(player.element_id) && !optimizedBenchIds.has(player.element_id)) {
      changes.removed.push(player);
    }
  });
  currentBenchPlayers.forEach((player) => {
    if (!optimizedStartingIds.has(player.element_id) && !optimizedBenchIds.has(player.element_id)) {
      changes.removed.push(player);
    }
  });

  const transferSummary = data.transfer_summary || {};
  if (transferSummary.transfers_out && transferSummary.transfers_out.length) {
    changes.removed = transferSummary.transfers_out;
  }

  currentStarting.forEach((player) => {
    let status = "kept";
    if (optimizedBenchIds.has(player.element_id)) {
      status = "demoted";
    } else if (!optimizedStartingIds.has(player.element_id)) {
      status = "removed";
    }
    currentStatusById.set(player.element_id, status);
  });

  currentBenchPlayers.forEach((player) => {
    let status = "bench-kept";
    if (optimizedStartingIds.has(player.element_id)) {
      status = "promoted";
    } else if (!optimizedBenchIds.has(player.element_id)) {
      status = "removed";
    }
    currentStatusById.set(player.element_id, status);
  });

  renderChangeSummary(changes, transferSummary, data.net_predicted_points);
  renderPitch(pitchOptimizedContainer, data.optimized_starting, {
    statusById,
    captainId: data.captain ? data.captain.element_id : undefined,
    viceCaptainId: data.vice_captain ? data.vice_captain.element_id : undefined,
  });
  renderBench(benchOptimizedContainer, data.optimized_bench, statusById);
  renderChips(data.chip_recommendations || {});
  renderCurrentTeam(currentProjection, currentStatusById);

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
    const statusMarkup = options.status ? options.status(player) : "";
    row.innerHTML = `
      <div class="flex flex-col gap-1">
        <div class="flex flex-wrap items-center gap-2 text-sm font-medium text-white">
          <span>${player.name}</span>
          ${badgeContent}
        </div>
        <p class="text-xs uppercase tracking-[0.25em] text-slate-400">${player.team_name}</p>
        ${statusMarkup}
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

function renderBench(container, players, statusById) {
  const statusLookup = statusById instanceof Map ? statusById : new Map();
  renderPlayers(container, players, {
    badge: (player) => `<span class="badge">${player.position}</span>`,
    status: (player) => renderStatusBadge(statusLookup.get(player.element_id)),
  });
}

function renderPitch(container, startingPlayers, options = {}) {
  container.innerHTML = "";
  if (!startingPlayers || !startingPlayers.length) {
    container.innerHTML = `<p class="text-sm text-slate-500">No starting XI available.</p>`;
    return;
  }

  const statusLookup = options.statusById instanceof Map ? options.statusById : new Map();
  const captainId = options.captainId;
  const viceCaptainId = options.viceCaptainId;

  const rows = [
    { key: "GKP", label: "Goalkeeper" },
    { key: "DEF", label: "Defenders" },
    { key: "MID", label: "Midfielders" },
    { key: "FWD", label: "Forwards" },
  ];

  const grouped = rows.reduce((acc, row) => ({ ...acc, [row.key]: [] }), {});
  startingPlayers.forEach((player) => {
    const key = rows.find((row) => row.key === player.position) ? player.position : "MID";
    grouped[key].push(player);
  });

  rows.forEach((row) => {
    const rowWrapper = document.createElement("div");
    rowWrapper.className = "pitch-row";
    const players = grouped[row.key];
    if (!players.length) {
      return;
    }
    const label = document.createElement("div");
    label.className = "pitch-row__label";
    label.textContent = row.label;
    rowWrapper.appendChild(label);
    [...players]
      .sort((a, b) => b.prediction - a.prediction)
      .forEach((player) => {
      const status = statusLookup.get(player.element_id);
      const meta = resolveStatusMeta(status);
      const isCaptain = captainId && player.element_id === captainId;
      const isVice = viceCaptainId && player.element_id === viceCaptainId;
      const card = document.createElement("div");
      card.className = `pitch-player fade-in pitch-player--${meta.tone}`;
      card.innerHTML = `
        <div class="pitch-player__badge">
          ${isCaptain ? '<span class="badge-pill">C</span>' : isVice ? '<span class="badge-pill">VC</span>' : `<span class="badge">${player.position}</span>`}
        </div>
        <div class="pitch-player__name">${player.name}</div>
        <div class="pitch-player__team">${player.team_name}</div>
        <div class="pitch-player__metrics">
          <span>${currencyFormat.format(player.price)}</span>
          <span>${numberFormat.format(player.prediction)}</span>
        </div>
        ${meta.label ? `<span class="status-tag status-tag--${meta.tone}">${meta.label}</span>` : ""}
      `;
      rowWrapper.appendChild(card);
      });
    container.appendChild(rowWrapper);
  });
}

function renderChangeSummary(changes, summary = {}, netPoints) {
  changeSummaryContainer.innerHTML = "";

  const totalTransfers = summary.transfers_needed ?? (summary.transfers_in ? summary.transfers_in.length : 0);
  const freeTransfers = summary.free_transfers;
  const transferHits = summary.transfer_hits ?? (freeTransfers !== undefined && freeTransfers !== null
    ? Math.max(totalTransfers - freeTransfers, 0)
    : 0);
  const hitPoints = summary.hit_points ?? transferHits * 4;
  const netProjected = typeof netPoints === "number"
    ? netPoints
    : typeof summary.net_points === "number"
      ? summary.net_points
      : null;

  const headline = document.createElement("div");
  headline.className = "change-headline";
  const headlineParts = [`${totalTransfers} move${totalTransfers === 1 ? "" : "s"}`];
  if (freeTransfers !== undefined && freeTransfers !== null) {
    headlineParts.push(`${freeTransfers} free transfer${freeTransfers === 1 ? "" : "s"}`);
  }
  const hitLabel = hitPoints ? `-${Math.round(hitPoints)} pts hit` : "No hit";
  headlineParts.push(hitLabel);
  if (netProjected !== null) {
    headlineParts.push(`Net XI ${numberFormat.format(netProjected)} pts`);
  }
  headline.innerHTML = headlineParts.map((block) => `<span>${block}</span>`).join("");
  changeSummaryContainer.appendChild(headline);

  if (summary.free_transfers_source === "estimated") {
    const note = document.createElement("p");
    note.className = "change-note";
    note.textContent = "Free transfers estimated from FPL data (defaults to 1). Update the field above if you have more.";
    changeSummaryContainer.appendChild(note);
  }

  const groups = [
    { title: "Transfers In", tone: "positive", players: summary.transfers_in || [] },
    { title: "Transfers Out", tone: "danger", players: summary.transfers_out || [] },
    { title: "New Starters", tone: "positive", players: changes.newToXI || [] },
    { title: "New Bench Cover", tone: "neutral", players: changes.newToBench || [] },
    { title: "Promoted from Bench", tone: "positive", players: changes.promoted || [] },
    { title: "Dropped to Bench", tone: "warning", players: changes.demoted || [] },
  ];

  const hasContent = groups.some((group) => group.players && group.players.length);
  if (!hasContent) {
    const emptyMessage = document.createElement("p");
    emptyMessage.className = "change-note";
    emptyMessage.textContent = "No squad changes recommended versus your current team.";
    changeSummaryContainer.appendChild(emptyMessage);
    return;
  }

  groups
    .filter((group) => group.players && group.players.length)
    .forEach((group) => {
      const card = document.createElement("div");
      card.className = `change-card change-card--${group.tone} fade-in`;
      card.innerHTML = `
        <h4>${group.title}</h4>
        <ul>
          ${group.players
            .map((player) => {
              const info = normalizePlayerInfo(player);
              const detail = [info.position, info.team].filter(Boolean).join(" · ");
              return `
                <li>
                  <span>${info.name}</span>
                  <span>${detail}</span>
                </li>
              `;
            })
            .join("")}
        </ul>
      `;
      changeSummaryContainer.appendChild(card);
    });
}

function resolveStatusMeta(status) {
  const fallbacks = { label: "", tone: "neutral" };
  const map = {
    kept: { label: "Unchanged", tone: "neutral" },
    promoted: { label: "Promoted", tone: "positive" },
    new: { label: "New", tone: "positive" },
    demoted: { label: "To Bench", tone: "warning" },
    "bench-kept": { label: "Bench Hold", tone: "neutral" },
    "bench-new": { label: "Bench Cover", tone: "neutral" },
    removed: { label: "Removed", tone: "danger" },
  };
  return map[status] || fallbacks;
}

function renderStatusBadge(status) {
  if (!status) {
    return "";
  }
  const meta = resolveStatusMeta(status);
  if (!meta.label) {
    return "";
  }
  return `<span class="status-tag status-tag--${meta.tone}">${meta.label}</span>`;
}

function normalizePlayerInfo(player) {
  if (!player) {
    return { name: "", team: "", position: "" };
  }
  const name = player.name || player.player || "";
  const team = player.team_name || player.team || "";
  const position = player.position || "";
  return { name, team, position };
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

function renderCurrentTeam(projection, statusById) {
  pitchCurrentContainer.innerHTML = "";
  currentBench.innerHTML = "";

  const hasStarting = Array.isArray(projection?.starting) && projection.starting.length;
  const hasBench = Array.isArray(projection?.bench) && projection.bench.length;
  if (!hasStarting && !hasBench) {
    pitchCurrentContainer.innerHTML = `<p class="text-sm text-slate-500">No current team data available.</p>`;
    return;
  }

  const statusLookup = statusById instanceof Map ? statusById : new Map();
  const captain = hasStarting ? projection.starting.find((player) => player.is_captain) : undefined;
  const viceCaptain = hasStarting ? projection.starting.find((player) => player.is_vice_captain) : undefined;

  renderPitch(pitchCurrentContainer, projection.starting || [], {
    statusById: statusLookup,
    captainId: captain ? captain.element_id : undefined,
    viceCaptainId: viceCaptain ? viceCaptain.element_id : undefined,
  });

  renderBench(currentBench, projection.bench || [], statusLookup);
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
