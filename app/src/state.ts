/**
 * Reactive application state.
 *
 * Holds the current slice, layout, camera position, and neighborhood settings.
 * Uses a simple listener pattern for reactivity.
 */

import type { StripLayout, PointOfView, RangeFilter, ProximityFilter, ContrastControl } from "./types";

export interface AppState {
  // Slice controls
  rangeFilters: RangeFilter[];
  proximityFilters: ProximityFilter[];
  contrastControls: ContrastControl[];

  // Layout
  selectedAxes: string[];
  tightness: number;
  model: string;

  // Layout result
  layout: StripLayout | null;
  imageIds: string[];
  orderValues: Record<string, number>;

  // Time direction: "similarity" | "capture_date" | index into contrastControls
  timeDirection: "similarity" | "capture_date";

  // Camera
  pov: PointOfView;

  // Torus dimensions
  torusWidth: number;
  torusHeight: number;
  stripHeight: number;
}

type Listener = (state: AppState) => void;

const listeners: Listener[] = [];

export const state: AppState = {
  rangeFilters: [],
  proximityFilters: [],
  contrastControls: [],
  selectedAxes: [],
  tightness: 0.5,
  model: "clip-vit-b-32",
  layout: null,
  imageIds: [],
  orderValues: {},
  timeDirection: "capture_date",
  pov: { x: 0, y: 0, z: 1000 },
  torusWidth: 0,
  torusHeight: 0,
  stripHeight: 100,
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
