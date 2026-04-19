/** Shared types matching the Python sidecar API shapes. */

export interface ImagePosition {
  id: string;
  x: number;
  width: number;
  thumbnail_path: string | null;
}

export interface Strip {
  y: number;
  height: number;
  images: ImagePosition[];
}

export interface StripLayout {
  strips: Strip[];
  torus_width: number;
  torus_height: number;
  strip_height: number;
}

export interface CellPosition {
  id: string;
  col: number;
  row: number;
  elevation: number; // [0, 1]; density peak of the continuous target field
}

export interface AttractorPosition {
  kind: "thing" | "target_image";
  ref: string;
  col: number;
  row: number;
}

export interface SpaceLikeLayout {
  positions: CellPosition[];
  attractor_positions: AttractorPosition[];
  cell_size: number;
  cols: number;
  rows: number;
  torus_width: number;
  torus_height: number;
}

export type AnyLayout = StripLayout | SpaceLikeLayout;

export function isSpaceLikeLayout(layout: AnyLayout): layout is SpaceLikeLayout {
  return (layout as SpaceLikeLayout).positions !== undefined;
}

export interface Attractor {
  kind: "thing" | "target_image";
  ref: string;
}

export interface ContrastAxis {
  pole_a: string;
  pole_b: string;
}

export interface RangeFilter {
  dimension: string;
  min: number;
  max: number;
}

export interface ContrastControl {
  pole_a: string;
  pole_b: string;
  band_min: number;
  band_max: number;
}

export interface SliceRequest {
  filter: import("./relevance").Expression | null;
  relevance: number;
  model: string;
  order_contrast?: { pole_a: string; pole_b: string } | null;
}

export interface SpaceLikeRequest {
  filter: import("./relevance").Expression | null;
  relevance: number;
  model: string;
  feathering: number;
  cell_size: number;
  // "echo" (default): grid >= slice; surplus cells cycle, creating moiré
  //   on small slices. "tight": grid <= slice; every cell unique, the
  //   layout drops the least-similar overflow.
  field_expansion?: "echo" | "tight";
  // Single-attractor arrangement. "rings": radial Chebyshev rings, sharp
  //   similarity tiers from centre. "field": biased-UMAP deformation,
  //   continuous gradient. "axis" (TargetImage only): synthesises a two-pole
  //   axis between the target's embedding and its antipode, smooth gradient
  //   along that axis. Default "rings".
  arrangement?: "rings" | "field" | "axis";
  // Viewport aspect = cols/rows target. Sizing the torus to match the
  // viewport aspect means a fully zoomed-out view shows exactly one torus
  // — no repeated area visible. Default 1.0 (near-square, legacy).
  aspect?: number;
}

export interface SliceResponse {
  image_ids: string[];
  count: number;
  has_order_axis: boolean;
  order_values: Record<string, number>;
}

export interface LayoutRequest {
  image_ids: string[];
  axes: string[] | null;
  feathering: number;
  model: string;
  strip_height: number;
  preserve_order?: boolean;
  order_values?: Record<string, number>;
  layout_mode?: "auto" | "strips";
}

export interface Dimension {
  name: string;
  type: "range" | "enum";
  min?: number;
  max?: number;
  values?: string[];
}

export interface PointOfView {
  x: number;
  y: number;
  z: number; // world-space distance covered by the horizontal view (zoom)
  pitch: number; // radians; 0 = top-down, PI/2 = horizon
  yaw: number; // radians; rotation around the vertical axis
}

export interface LayerToggles {
  photos: boolean;
  neighborhoods: boolean;
}

export interface VocabularyTree {
  [sigil: string]: VocabularyNode[];
}

export interface VocabularyNode {
  name: string;
  prompt: string;
  children?: VocabularyNode[];
}


export interface Sibling {
  name: string;
  prompt: string;
}

export interface StageProgress {
  name: string;
  completed: number;
  total: number;
  timestamp: number;
}

export interface ImportProgress {
  status: "idle" | "running" | "paused" | "completed" | "error";
  stages: StageProgress[];
  started_at: number | null;
}
