/**
 * Reactive application state.
 *
 * Holds the current slice, layout, camera position, and neighborhood settings.
 * Uses a simple listener pattern for reactivity.
 */

import type {
  AnyLayout,
  Attractor,
  PointOfView,
  RangeFilter,
  ProximityFilter,
  ContrastControl,
  ImportProgress,
} from "./types";

export interface AppState {
  // Slice controls
  rangeFilters: RangeFilter[];
  proximityFilters: ProximityFilter[];
  contrastControls: ContrastControl[];

  // Layout
  selectedAxes: string[];
  feathering: number;
  model: string;

  // SpaceLike attractors
  attractors: Attractor[];

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

  // Import
  importProgress: ImportProgress | null;

  // Last error (shown in status bar, cleared on next successful recompute)
  lastError: string | null;
}

type Listener = (state: AppState) => void;

const listeners: Listener[] = [];

export const state: AppState = {
  rangeFilters: [],
  proximityFilters: [],
  contrastControls: [],
  selectedAxes: [],
  feathering: 0.5,
  model: "clip-vit-b-32",
  attractors: [],
  layout: null,
  imageIds: [],
  orderValues: {},
  mode: "spacelike",
  timeDirection: "capture_date",
  pov: { x: 0, y: 0, z: 1000 },
  torusWidth: 0,
  torusHeight: 0,
  stripHeight: 100,
  cellSize: 100,
  importProgress: null,
  lastError: null,
};

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
