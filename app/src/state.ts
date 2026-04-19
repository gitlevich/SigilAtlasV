/**
 * Reactive application state.
 *
 * Holds the current slice, layout, camera position, and neighborhood settings.
 * Uses a simple listener pattern for reactivity.
 */

import type {
  AnyLayout,
  Attractor,
  LayerToggles,
  PointOfView,
  RangeFilter,
  ContrastControl,
  ImportProgress,
} from "./types";
import * as api from "./api";

export interface AppState {
  // RelevanceFilter inputs — UI-shaped; buildFilter projects them into the AST.
  // The filter is a membrane: these pieces are intent, the slice is the result.
  attractors: Attractor[];
  contrastControls: ContrastControl[];
  rangeFilters: RangeFilter[];

  // ThingsLibrary — every @thing the user has named, persistent across sessions
  // per `Explore/Control/ThingsLibrary/invariant-persistent`. Activation moves
  // a thing into AttractorControl; #delete removes it from the library entirely.
  thingsLibrary: string[];
  thingsLibraryFolded: boolean;

  // Relevance gate — how strictly semantic atoms (Thing, TargetImage) admit
  // images into the slice. [0, 1], 0 = loose, 1 = strict. Controlled by the
  // Relevance slider adjacent to AttractorControl.
  relevance: number;

  // Layout
  selectedAxes: string[];
  feathering: number;
  model: string;

  // Layout result (StripLayout for timelike, SpaceLikeLayout for spacelike)
  layout: AnyLayout | null;
  imageIds: string[];
  orderValues: Record<string, number>;

  // Mode: how images arrange on the torus
  mode: "timelike" | "spacelike";

  // Time direction (timelike mode): "capture_date" or a contrast index
  timeDirection: string;

  // Camera
  pov: PointOfView;

  // Torus dimensions
  torusWidth: number;
  torusHeight: number;
  stripHeight: number;
  cellSize: number;

  // Layer toggles — three independent layers (photos, wireframe, relief)
  layers: LayerToggles;

  // Relief height scaling in world units
  reliefScale: number;

  // Import
  importProgress: ImportProgress | null;

  // Last error (shown in status bar, cleared on next successful recompute)
  lastError: string | null;

  // SpaceLike field expansion mode. "echo" preserves the moiré-on-small-slices
  // visual that emerges from radial padding cycle; "tight" drops the
  // least-similar overflow so every cell carries a unique image.
  fieldExpansion: "echo" | "tight";

  // SpaceLike arrangement (single-attractor only). "rings" = radial Chebyshev
  // shells; "field" = biased-UMAP continuous deformation; "axis" (target
  // image only) = two-pole gradient between the target's embedding and its
  // antipode.
  arrangement: "rings" | "field" | "axis";

  // Lightbox — one image inhabited in isolation. While open, the @slice,
  // @arrangement and @control are not recomputing; the field is frozen.
  // See sigil_atlas.sigil/Explore/Lightbox.
  lightbox: LightboxState;

  // Collages — saved views (SigilML expression + camera + arrangement params).
  // Loaded from the workspace on init; refreshed on save/rename/delete.
  collages: api.CollageSummary[];

  // Sigils (.sigil subdirectories of the @Workspace). Each is a self-contained
  // saved context. Populated by scanning the workspace directory on startup
  // and after save. Null until the first scan completes.
  workspaceSigils: SigilEntry[];
}

export interface SigilEntry {
  name: string;
  folder_path: string;
  preview_data_url: string | null;
  modified_at: number | null;
}

export interface LightboxState {
  imageId: string | null;
  entryPov: PointOfView | null;
  showMetadata: boolean;
}

type Listener = (state: AppState) => void;

const listeners: Listener[] = [];

// The library belongs to the @Workspace (per spec Explore/Control/
// ThingsLibrary/invariant-persistent.md) and is persisted in the workspace
// SQLite via /things/library endpoints. Only the folded/unfolded header state
// stays in localStorage — it's a UI preference, not workspace data.
const LIB_KEY_LEGACY = "sigil-atlas/things-library";
const LIB_FOLDED_KEY = "sigil-atlas/things-library-folded";

function loadLibraryFolded(): boolean {
  return localStorage.getItem(LIB_FOLDED_KEY) === "1";
}

export function persistLibraryFolded(): void {
  try {
    localStorage.setItem(LIB_FOLDED_KEY, state.thingsLibraryFolded ? "1" : "0");
  } catch (e) {
    console.error("[library] persist folded failed:", e);
  }
}

/**
 * Load the library from the workspace sidecar. On first run after migration,
 * drain any legacy localStorage entries into the workspace then clear them.
 */
export async function initThingsLibrary(): Promise<void> {
  let legacy: string[] = [];
  try {
    const raw = localStorage.getItem(LIB_KEY_LEGACY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        legacy = parsed.filter((s): s is string => typeof s === "string");
      }
    }
  } catch {
    // Ignore malformed legacy payload; we'll clear it below.
  }

  let names: string[];
  try {
    names = await api.getThingsLibrary();
  } catch (e) {
    console.error("[library] load failed:", e);
    state.thingsLibrary = legacy;
    return;
  }

  if (legacy.length > 0) {
    const known = new Set(names);
    for (const name of legacy) {
      if (known.has(name)) continue;
      try {
        names = await api.addThingToLibrary(name);
      } catch (e) {
        console.error("[library] migration add failed:", name, e);
      }
    }
    try {
      localStorage.removeItem(LIB_KEY_LEGACY);
    } catch {
      // Best-effort; clearing localStorage is not load-bearing.
    }
  }

  state.thingsLibrary = names;
}

export const state: AppState = {
  attractors: [],
  contrastControls: [],
  rangeFilters: [],
  thingsLibrary: [],
  thingsLibraryFolded: loadLibraryFolded(),
  relevance: 0.5,
  selectedAxes: [],
  feathering: 0.5,
  model: "clip-vit-b-32",
  layout: null,
  imageIds: [],
  orderValues: {},
  mode: "spacelike",
  timeDirection: "capture_date",
  pov: { x: 0, y: 0, z: 1000, pitch: 0, yaw: 0 },
  torusWidth: 0,
  torusHeight: 0,
  stripHeight: 100,
  cellSize: 100,
  layers: { photos: true, neighborhoods: false },
  reliefScale: 600,
  importProgress: null,
  lastError: null,
  lightbox: { imageId: null, entryPov: null, showMetadata: false },
  collages: [],
  workspaceSigils: [],
  fieldExpansion: "echo",
  arrangement: "rings",
};

const SIGILS_FOLDER_KEY = "sigil-atlas.sigils-folder";

/** The user-chosen directory where `.sigil` archives live. Distinct from the
 *  sidecar's data-store workspace: this one only controls the left-panel
 *  listing, not the corpus DB or image cache. Null until the user picks. */
export function currentSigilsFolder(): string | null {
  try {
    return localStorage.getItem(SIGILS_FOLDER_KEY);
  } catch {
    return null;
  }
}

export function setSigilsFolder(path: string): void {
  try {
    localStorage.setItem(SIGILS_FOLDER_KEY, path);
  } catch (e) {
    console.error("[sigils folder] persist failed:", e);
  }
}

/** Scan the sigils folder for `.sigil` subdirectories. No-op outside Tauri. */
export async function refreshWorkspaceSigils(): Promise<void> {
  const folder = currentSigilsFolder();
  if (!folder) { state.workspaceSigils = []; notify(); return; }
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    state.workspaceSigils = await invoke<SigilEntry[]>("list_sigils", { workspace: folder });
  } catch (e) {
    console.error("[sigils] list failed:", e);
    state.workspaceSigils = [];
  }
  notify();
}

/** Refresh state.collages from the workspace. Call after any save/rename/delete. */
export async function refreshCollages(): Promise<void> {
  try {
    state.collages = await api.listCollages();
  } catch (e) {
    console.error("[collages] list failed:", e);
  }
}

export function subscribe(fn: Listener): () => void {
  listeners.push(fn);
  return () => {
    const idx = listeners.indexOf(fn);
    if (idx >= 0) listeners.splice(idx, 1);
  };
}

export function notify(): void {
  for (const fn of listeners) fn(state);
}

// ── Workspace persistent state ────────────────────────────────────────────
// Per sigil_atlas.sigil/Explore/invariant-persistent-state.md: closing and
// reopening the app returns to the same @POV, same @arrangement, same UI.
// The shape is a deliberate subset of AppState — only things that survive
// restart; not imageIds, layout, torus dims, progress, lightbox.

export interface WorkspacePersistedState {
  mode: AppState["mode"];
  arrangement: AppState["arrangement"];
  fieldExpansion: AppState["fieldExpansion"];
  selectedAxes: string[];
  feathering: number;
  cellSize: number;
  model: string;
  timeDirection: string;
  relevance: number;
  layers: LayerToggles;
  reliefScale: number;
  pov: PointOfView;
}

export function snapshotPersistentState(): WorkspacePersistedState {
  return {
    mode: state.mode,
    arrangement: state.arrangement,
    fieldExpansion: state.fieldExpansion,
    selectedAxes: [...state.selectedAxes],
    feathering: state.feathering,
    cellSize: state.cellSize,
    model: state.model,
    timeDirection: state.timeDirection,
    relevance: state.relevance,
    layers: { ...state.layers },
    reliefScale: state.reliefScale,
    pov: { ...state.pov },
  };
}

/** Apply a persisted payload back into live state, in place. Called once at
 *  init before the first slice/layout so mode/model/etc take effect for the
 *  very first recompute. POV is applied separately after layout, because the
 *  initial framing depends on computed torus dimensions. */
export function applyPersistedExceptPov(p: WorkspacePersistedState): void {
  state.mode = p.mode;
  state.arrangement = p.arrangement;
  state.fieldExpansion = p.fieldExpansion;
  state.selectedAxes = [...p.selectedAxes];
  state.feathering = p.feathering;
  state.cellSize = p.cellSize;
  state.model = p.model;
  state.timeDirection = p.timeDirection;
  state.relevance = p.relevance;
  state.layers = { ...p.layers };
  state.reliefScale = p.reliefScale;
}

export async function loadPersistedState(): Promise<WorkspacePersistedState | null> {
  try {
    return await api.getWorkspaceState<WorkspacePersistedState>();
  } catch (e) {
    console.error("[workspace-state] load failed:", e);
    return null;
  }
}

// Debounced persister. Multiple rapid mutations (drag, slider tweak) collapse
// into a single POST. The watcher also ticks at a coarse interval to catch
// POV changes from the camera loop which don't call notify().
let persistTimer: number | null = null;
let lastPersistedJson: string | null = null;
const PERSIST_DEBOUNCE_MS = 400;

export function schedulePersist(): void {
  if (persistTimer !== null) return;
  persistTimer = window.setTimeout(() => {
    persistTimer = null;
    const snap = snapshotPersistentState();
    const json = JSON.stringify(snap);
    if (json === lastPersistedJson) return;
    lastPersistedJson = json;
    api.saveWorkspaceState(snap).catch((e) => {
      console.error("[workspace-state] save failed:", e);
      lastPersistedJson = null; // retry on next tick
    });
  }, PERSIST_DEBOUNCE_MS);
}

/** Mark the current state as already-persisted so the first schedulePersist()
 *  after init doesn't POST unchanged state back to the server. */
export function markPersistedBaseline(): void {
  lastPersistedJson = JSON.stringify(snapshotPersistentState());
}
