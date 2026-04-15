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
  tightness: number;
  model: string;
  strip_height: number;
  preserve_order?: boolean;
  order_values?: Record<string, number>;
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
  z: number; // 0 = one image fills frame, higher = zoomed out
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
