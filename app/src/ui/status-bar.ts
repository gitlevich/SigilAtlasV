/**
 * Status bar — pure view of state.importProgress.
 *
 * Subscribes to app state. Renders when importProgress changes.
 * No polling, no lifecycle management — that's the import service's job.
 */

import { subscribe, type AppState } from "../state";
import { pauseImport, resumeImport } from "../import";
import type { ImportProgress, StageProgress } from "../types";

/** Previous snapshot for computing rate between renders. */
let prevCompleted: Map<string, number> = new Map();
let prevTime = 0;

export function initStatusBar(): void {
  subscribe(render);
}

function render(s: AppState): void {
  const el = document.getElementById("status-bar");
  if (!el) return;

  const progress = s.importProgress;
  const hasError = !!s.lastError;
  const hasProgress = progress && progress.status !== "idle";

  if (!hasError && !hasProgress) {
    el.classList.add("hidden");
    prevCompleted = new Map();
    prevTime = 0;
    return;
  }

  el.classList.remove("hidden");
  el.innerHTML = "";

  if (hasError) {
    el.appendChild(statusMsg(s.lastError!, true));
  }

  if (!progress || progress.status === "idle") return;

  if (progress.status === "completed") {
    el.appendChild(statusMsg("Import complete"));
    return;
  }

  if (progress.status === "error") {
    el.appendChild(statusMsg("Import failed", true));
    return;
  }

  // Running or paused — show stages
  if (progress.stages.length === 0) {
    el.appendChild(statusMsg("Scanning..."));
  }

  const now = performance.now() / 1000;
  const rates = computeRates(progress, now);

  for (const stage of progress.stages) {
    el.appendChild(renderStage(stage, rates.get(stage.name)));
  }

  const spacer = document.createElement("div");
  spacer.style.flex = "1";
  el.appendChild(spacer);

  if (progress.status === "paused") {
    el.appendChild(statusMsg("Paused"));
    el.appendChild(playPauseBtn(true));
  } else {
    el.appendChild(playPauseBtn(false));
  }
}

function computeRates(progress: ImportProgress, now: number): Map<string, number> {
  const rates = new Map<string, number>();
  const dt = now - prevTime;
  if (dt > 0.2 && prevTime > 0) {
    for (const stage of progress.stages) {
      const prev = prevCompleted.get(stage.name) ?? 0;
      const delta = stage.completed - prev;
      if (delta > 0) rates.set(stage.name, delta / dt);
    }
  }
  prevCompleted = new Map(progress.stages.map((s) => [s.name, s.completed]));
  prevTime = now;
  return rates;
}

function statusMsg(text: string, error = false): HTMLElement {
  const span = document.createElement("span");
  span.className = error ? "status-message status-error" : "status-message";
  span.textContent = text;
  return span;
}

function playPauseBtn(paused: boolean): HTMLElement {
  const btn = document.createElement("button");
  btn.className = "status-play-pause";
  btn.textContent = paused ? "\u25B6" : "\u23F8";
  btn.title = paused ? "Resume import" : "Pause import";
  btn.addEventListener("click", () => {
    (paused ? resumeImport() : pauseImport()).catch((e) =>
      console.error("[status-bar] pause/resume failed:", e),
    );
  });
  return btn;
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
