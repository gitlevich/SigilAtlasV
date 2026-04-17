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

export interface SpaceLikeRequest {
  image_ids: string[];
  attractors: Attractor[];
  model: string;
  feathering: number;
  cell_size: number;
}

export interface RangeFilter {
  dimension: string;
  min: number;
  max: number;
}

export interface ProximityFilter {
  text: string;
  weight: number;
}

export interface ContrastControl {
  pole_a: string;
  pole_b: string;
  role: "filter" | "attract" | "order";
  band_min: number;
  band_max: number;
}

export interface SliceRequest {
  range_filters: RangeFilter[];
  proximity_filters: ProximityFilter[];
  contrast_controls: ContrastControl[];
  model: string;
  feathering: number;
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

export interface VocabTerm {
  name: string;
  path: string;
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
