/**
 * Reactive application state.
 *
 * Holds the current slice, layout, camera position, and neighborhood settings.
 * Uses a simple listener pattern for reactivity.
 */

import type { StripLayout, PointOfView, RangeFilter, ProximityFilter } from "./types";

export interface AppState {
  // SliceMode
  rangeFilters: RangeFilter[];
  proximityFilters: ProximityFilter[];

  // NeighborhoodMode
  selectedAxes: string[];
  tightness: number;
  model: string;

  // Layout result
  layout: StripLayout | null;
  imageIds: string[];

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
  selectedAxes: [],
  tightness: 0.5,
  model: "clip-vit-b-32",
  layout: null,
  imageIds: [],
  pov: { x: 0, y: 0, z: 1000 },
  torusWidth: 0,  // derived from layout
  torusHeight: 0, // derived from layout
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
