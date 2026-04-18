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

  // Tonal data — per-image [brightness, color_temperature] for every image
  // in the corpus that has pixel features. Fetched once on startup, used
  // every frame to compute the ambient tint under the current viewport per
  // the @Lighting light-follows-attention coupling.
  tonal: Record<string, [number, number]>;

  // Lightbox — one image inhabited in isolation. While open, the @slice,
  // @arrangement and @control are not recomputing; the field is frozen.
  // See sigil_atlas.sigil/Explore/Lightbox.
  lightbox: LightboxState;

  // Collages — saved views (SigilML expression + camera + arrangement params).
  // Loaded from the workspace on init; refreshed on save/rename/delete.
  collages: api.CollageSummary[];
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
  tonal: {},
  collages: [],
  fieldExpansion: "echo",
  arrangement: "rings",
};

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
