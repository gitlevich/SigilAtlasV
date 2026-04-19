/**
 * Menu actions — every app action exposed as a callable function so the
 * native macOS menu, keyboard shortcuts, and on-screen panel buttons all
 * dispatch to the same code. Keeps the menu definition flat and avoids
 * drift between menu / keybinding / button paths.
 */

import { state, notify } from "./state";
import * as api from "./api";
import { recomputeSliceAndLayout, imageAtWorld } from "./ui/controls";
import { saveCurrentAsCollage, openCollage } from "./collages";
import { openLightbox, isLightboxOpen, closeLightbox } from "./ui/lightbox";
import { startImport, startPolling } from "./import";

// ── Open / Save (Collage) ────────────────────────────────────────────────

export function actSaveCollage(): void {
  saveCurrentAsCollage().catch((e) => console.error("[save-collage]", e));
}

export function actOpenCollage(): void {
  openCollage().catch((e) => console.error("[open-collage]", e));
}

export async function actImportPhotos(): Promise<void> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  const selected = await open({ directory: true, title: "Choose source folder" });
  if (typeof selected === "string") {
    startImport(selected).catch((e) => console.error("[import]", e));
  }
}

export async function actImportApplePhotos(): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  const { message } = await import("@tauri-apps/plugin-dialog");

  state.importProgress = { status: "running", stages: [], started_at: null };
  notify();

  try {
    const auth = await invoke<{ status: string }>("photos_auth");
    if (auth.status !== "authorized" && auth.status !== "limited") {
      state.lastError = `Photos access: ${auth.status}`;
      state.importProgress = { status: "error", stages: [], started_at: null };
      notify();
      await message(
        "Sigil Atlas needs read access to Photos. Grant it in System Settings > Privacy & Security > Photos, then try again.",
        { title: "Photos access denied" },
      );
      return;
    }
    // Start polling before awaiting the enumerate — Python begins emitting
    // progress on the reporter the moment the first batch arrives. The poll
    // loop stops itself once status flips to "completed" / "error".
    startPolling();
    const summary = await invoke<{
      registered: number;
      skipped: number;
      thumbnails: number;
      thumbnail_failures: number;
    }>("photos_enumerate", {});
    console.log("[photos-import]", summary);
  } catch (e) {
    console.error("[photos-import]", e);
    const msg = e instanceof Error ? e.message : String(e);
    // Cancel surfaces as an error from the invoke, but the backend has
    // already set status=paused. Leave that alone so the UI matches.
    if (!/cancell?ed/i.test(msg)) {
      state.lastError = msg;
      state.importProgress = { status: "error", stages: [], started_at: null };
      notify();
    }
  }
}


// ── Mode ─────────────────────────────────────────────────────────────────

export function actModeSpacelike(): void {
  if (state.mode === "spacelike") return;
  state.mode = "spacelike";
  recomputeSliceAndLayout().catch((e) => console.error("[mode]", e));
}

export function actModeTimelike(): void {
  if (state.mode === "timelike") return;
  state.mode = "timelike";
  recomputeSliceAndLayout().catch((e) => console.error("[mode]", e));
}

// ── Layers ───────────────────────────────────────────────────────────────

export function actToggleLayerPhotos(): void {
  state.layers.photos = !state.layers.photos;
  notify();
}

export function actToggleLayerNeighborhoods(): void {
  state.layers.neighborhoods = !state.layers.neighborhoods;
  notify();
  // Match the panel button behaviour: when turning ON, fetch clusters.
  if (state.layers.neighborhoods) {
    api.computeNeighborhoods({
      image_ids: state.imageIds,
      model: state.model,
      k: 50,
    }).catch((e) => console.error("[neighborhoods]", e));
  }
}

// ── Camera ───────────────────────────────────────────────────────────────

export function actZoomIn(): void {
  applyZoom(0.85);
}

export function actZoomOut(): void {
  applyZoom(1.18);
}

export function actFrameAll(): void {
  if (!state.layout) return;
  const canvas = document.getElementById("viewport") as HTMLCanvasElement | null;
  if (!canvas) return;
  const aspect = canvas.clientWidth / canvas.clientHeight;
  const tw = state.torusWidth || 1;
  const th = state.torusHeight || 1;
  state.pov.x = tw / 2;
  state.pov.y = th / 2;
  state.pov.z = Math.min(tw, th * aspect);
}

function applyZoom(factor: number): void {
  const canvas = document.getElementById("viewport") as HTMLCanvasElement | null;
  if (!canvas) return;
  const aspect = canvas.clientWidth / canvas.clientHeight;
  const tw = state.torusWidth || 10000;
  const th = state.torusHeight || 10000;
  const maxZoom = Math.min(tw, th * aspect);
  state.pov.z = Math.max(50, Math.min(maxZoom, state.pov.z * factor));
}

// ── Image at screen centre / Lightbox ────────────────────────────────────

function centeredImageId(): string | null {
  if (!state.layout) return null;
  return imageAtWorld(state.layout, state.pov.x, state.pov.y);
}

export function actOpenLightbox(): void {
  if (isLightboxOpen()) {
    closeLightbox();
    return;
  }
  const id = centeredImageId();
  if (id) openLightbox(id);
}

export function actSetTargetToCenter(): void {
  const id = centeredImageId();
  if (!id) return;
  state.attractors = state.attractors.filter((a) => a.kind !== "target_image");
  state.attractors.push({ kind: "target_image", ref: id });
  recomputeSliceAndLayout({ anchorImageId: id })
    .catch((e) => console.error("[set-target]", e));
}

export function actReleaseTarget(): void {
  const before = state.attractors.length;
  state.attractors = state.attractors.filter((a) => a.kind !== "target_image");
  if (state.attractors.length !== before) {
    recomputeSliceAndLayout().catch((e) => console.error("[release-target]", e));
  }
}

// ── Filter clearing ──────────────────────────────────────────────────────

export function actClearAttractors(): void {
  if (state.attractors.length === 0) return;
  state.attractors = [];
  recomputeSliceAndLayout().catch((e) => console.error("[clear-attractors]", e));
}

export function actClearAllFilters(): void {
  const dirty =
    state.attractors.length > 0 ||
    state.contrastControls.length > 0 ||
    state.rangeFilters.length > 0;
  if (!dirty) return;
  state.attractors = [];
  state.contrastControls = [];
  state.rangeFilters = [];
  recomputeSliceAndLayout().catch((e) => console.error("[clear-filters]", e));
}

// ── Find (focus the Attract input) ───────────────────────────────────────

export function actFocusAttractInput(): void {
  // The Attract section's input is the only `.pill-input` inside the
  // right-panel "Attract" section. Easiest selector that doesn't depend on
  // ids: find the input under the section whose header reads "Attract".
  const sections = document.querySelectorAll(
    "#neighborhood-panel .section",
  );
  for (const sec of Array.from(sections)) {
    const header = sec.querySelector(".section-header");
    if (header?.textContent?.trim().toLowerCase() === "attract") {
      const input = sec.querySelector(".pill-input") as HTMLInputElement | null;
      if (input) {
        sec.classList.remove("collapsed");
        input.focus();
        input.select();
      }
      return;
    }
  }
}

// ── Settings (focus the Settings section) ────────────────────────────────

export function actOpenSettings(): void {
  const sections = document.querySelectorAll("#neighborhood-panel .section");
  for (const sec of Array.from(sections)) {
    const header = sec.querySelector(".section-header");
    if (header?.textContent?.trim().toLowerCase() === "settings") {
      sec.classList.remove("collapsed");
      sec.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
  }
}

// ── Sidebar ──────────────────────────────────────────────────────────────

export function actToggleSidebar(): void {
  const panel = document.getElementById("neighborhood-panel");
  if (!panel) return;
  panel.classList.toggle("folded");
  if (panel.classList.contains("folded")) {
    panel.style.width = "";
    panel.style.minWidth = "";
  }
}

/** Toggle both side panels as a unit. If either is visible, hide both; if
 *  both are hidden, show both. Bound to Ctrl+Tab. */
export function actToggleBothPanels(): void {
  const left = document.getElementById("slice-panel");
  const right = document.getElementById("neighborhood-panel");
  if (!left || !right) return;
  const anyOpen = !left.classList.contains("folded") || !right.classList.contains("folded");
  for (const p of [left, right]) {
    if (anyOpen) {
      p.classList.add("folded");
      p.style.width = "";
      p.style.minWidth = "";
    } else {
      p.classList.remove("folded");
    }
  }
}

// ── Tools ────────────────────────────────────────────────────────────────

export async function actEmbedMissing(): Promise<void> {
  await api.runMissingEmbeddings();
  startPolling();
}

export async function actRecomputePixelFeatures(): Promise<void> {
  await api.runPixelFeatures();
  startPolling();
}

export async function actRegeneratePreviews(): Promise<void> {
  try {
    await api.regeneratePreviews();
    state.lastError = null;
    notify();
    startPolling();
  } catch (e) {
    // Precheck failure — commonly the source drive is disconnected.
    // Surface the sidecar's message in the status bar instead of
    // silently failing into the console.
    const raw = e instanceof Error ? e.message : String(e);
    const msg = raw.replace(/^\/tools\/regenerate-previews:\s*/i, "").trim();
    state.lastError = msg || "Could not regenerate previews.";
    notify();
    console.error("[regenerate-previews]", e);
  }
}

export async function actNukeCorpus(): Promise<void> {
  const { confirm } = await import("@tauri-apps/plugin-dialog");
  const yes = await confirm(
    "This will permanently delete all images, embeddings, and metadata from the corpus. This cannot be undone.",
    { title: "Nuke the Corpus" },
  );
  if (yes) {
    await api.nukeCorpus();
  }
}
