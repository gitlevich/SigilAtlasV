/**
 * Status bar — shows import progress at the bottom of the window.
 *
 * Polls /ingest/progress while an import is active. Displays per-stage
 * progress bars with completed/total and images/sec rate.
 */

import * as api from "../api";
import type { ImportProgress, StageProgress } from "../types";

let pollTimer: ReturnType<typeof setInterval> | null = null;
let statusBarEl: HTMLElement | null = null;
let onImportComplete: (() => void) | null = null;

/** Previous snapshot for computing rate. */
let prevSnapshot: { time: number; completed: Map<string, number> } | null = null;

export function initStatusBar(opts: { onComplete: () => void }): void {
  onImportComplete = opts.onComplete;
  statusBarEl = document.getElementById("status-bar");
  if (!statusBarEl) return;
  statusBarEl.innerHTML = "";
  statusBarEl.classList.add("hidden");
}

export function startPolling(): void {
  if (pollTimer) return;
  prevSnapshot = null;
  if (statusBarEl) {
    statusBarEl.classList.remove("hidden");
  }
  pollTimer = setInterval(poll, 500);
  poll();
}

export function stopPolling(): void {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function poll(): Promise<void> {
  try {
    const progress = await api.getImportProgress();
    render(progress);

    if (progress.status === "completed") {
      stopPolling();
      setTimeout(() => {
        if (statusBarEl) statusBarEl.classList.add("hidden");
      }, 3000);
      onImportComplete?.();
    } else if (progress.status === "error") {
      stopPolling();
    } else if (progress.status === "paused") {
      stopPolling();
    }
  } catch {
    // Sidecar might be busy; skip this tick
  }
}

function render(progress: ImportProgress): void {
  if (!statusBarEl) return;

  const now = performance.now() / 1000;
  const rates = new Map<string, number>();

  if (prevSnapshot) {
    const dt = now - prevSnapshot.time;
    if (dt > 0.1) {
      for (const stage of progress.stages) {
        const prev = prevSnapshot.completed.get(stage.name) ?? 0;
        const delta = stage.completed - prev;
        if (delta > 0) {
          rates.set(stage.name, delta / dt);
        }
      }
    }
  }

  prevSnapshot = {
    time: now,
    completed: new Map(progress.stages.map((s) => [s.name, s.completed])),
  };

  statusBarEl.innerHTML = "";

  if (progress.status === "completed") {
    const msg = document.createElement("span");
    msg.className = "status-message";
    msg.textContent = "Import complete";
    statusBarEl.appendChild(msg);
    return;
  }

  if (progress.status === "error") {
    const msg = document.createElement("span");
    msg.className = "status-message status-error";
    msg.textContent = "Import failed";
    statusBarEl.appendChild(msg);
    return;
  }

  if (progress.status === "paused") {
    const msg = document.createElement("span");
    msg.className = "status-message";
    msg.textContent = "Import paused";
    statusBarEl.appendChild(msg);
    return;
  }

  for (const stage of progress.stages) {
    statusBarEl.appendChild(renderStage(stage, rates.get(stage.name)));
  }
}

function renderStage(stage: StageProgress, rate?: number): HTMLElement {
  const el = document.createElement("div");
  el.className = "status-stage";

  const label = document.createElement("span");
  label.className = "stage-label";
  label.textContent = stage.name;
  el.appendChild(label);

  const barOuter = document.createElement("div");
  barOuter.className = "stage-bar-outer";
  const barInner = document.createElement("div");
  barInner.className = "stage-bar-inner";
  const pct = stage.total > 0 ? (stage.completed / stage.total) * 100 : 0;
  barInner.style.width = `${pct}%`;
  barOuter.appendChild(barInner);
  el.appendChild(barOuter);

  const info = document.createElement("span");
  info.className = "stage-info";
  let text = `${stage.completed}/${stage.total}`;
  if (rate !== undefined && rate > 0) {
    text += ` (${rate.toFixed(1)}/s)`;
  }
  info.textContent = text;
  el.appendChild(info);

  return el;
}
