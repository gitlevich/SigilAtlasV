/**
 * Import service — owns the import lifecycle.
 *
 * Mutates state.importProgress and calls notify() so any subscriber
 * (status bar, viewport refresh) reacts automatically.
 */

import { state, notify } from "./state";
import * as api from "./api";

let pollTimer: ReturnType<typeof setInterval> | null = null;

export async function startImport(source: string): Promise<void> {
  // Show immediately — don't wait for the POST round-trip
  state.importProgress = { status: "running", stages: [], started_at: null };
  notify();

  const res = await api.startImport(source);
  if (res.status === "started") {
    startPolling();
  }
}

export async function pauseImport(): Promise<void> {
  await api.pauseImport();
  stopPolling();
}

export async function resumeImport(): Promise<void> {
  const res = await api.resumeImport();
  if (res.status === "started") {
    startPolling();
  }
}

function startPolling(): void {
  if (pollTimer) return;
  pollTimer = setInterval(poll, 500);
  poll();
}

function stopPolling(): void {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function poll(): Promise<void> {
  try {
    const progress = await api.getImportProgress();
    state.importProgress = progress;
    notify();

    if (progress.status === "completed" || progress.status === "error" || progress.status === "paused") {
      stopPolling();
    }
  } catch (e) {
    console.error("[import] poll failed:", e);
  }
}
