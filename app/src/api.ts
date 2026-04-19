/**
 * API client for the Python sidecar.
 */

import type {
  SliceRequest,
  SliceResponse,
  LayoutRequest,
  StripLayout,
  SpaceLikeLayout,
  SpaceLikeRequest,
  Dimension,
  VocabularyTree,
  Sibling,
  ImportProgress,
} from "./types";

let sidecarPort: number | null = null;

// Caller registers a revive callback that spawns a fresh sidecar (through
// Tauri) and returns its port. Invoked when fetch fails with a network error,
// implying the sidecar has died.
type ReviveFn = () => Promise<number>;
let reviveSidecar: ReviveFn | null = null;
let reviveInFlight: Promise<number> | null = null;

export function setSidecarPort(port: number): void {
  sidecarPort = port;
}

/** Base URL for the sidecar — useful for `<img src>` references that need
 *  a direct GET. Throws if the port hasn't been set yet. */
export function getThumbnailBaseUrl(): string {
  if (!sidecarPort) throw new Error("Sidecar port not set");
  return `http://127.0.0.1:${sidecarPort}`;
}

export function setReviveFn(fn: ReviveFn | null): void {
  reviveSidecar = fn;
}

function isNetworkError(err: unknown): boolean {
  // fetch() throws TypeError on connection-refused / ECONNREFUSED.
  return err instanceof TypeError && /fetch|network|refused|load/i.test(err.message);
}

async function tryRevive(): Promise<number | null> {
  if (!reviveSidecar) return null;
  if (!reviveInFlight) {
    reviveInFlight = reviveSidecar()
      .then((port) => {
        sidecarPort = port;
        return port;
      })
      .finally(() => {
        reviveInFlight = null;
      });
  }
  try {
    return await reviveInFlight;
  } catch {
    return null;
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  if (!sidecarPort) throw new Error("Sidecar port not set");
  const url = () => `http://127.0.0.1:${sidecarPort}${path}`;
  try {
    const res = await fetch(url(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`${path}: ${detail || res.status}`);
    }
    return res.json();
  } catch (err) {
    if (isNetworkError(err)) {
      const p = await tryRevive();
      if (p) {
        const res = await fetch(url(), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(`${path}: ${res.status}`);
        return res.json();
      }
    }
    throw err;
  }
}

async function get<T>(path: string): Promise<T> {
  if (!sidecarPort) throw new Error("Sidecar port not set");
  const url = () => `http://127.0.0.1:${sidecarPort}${path}`;
  try {
    const res = await fetch(url());
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`${path}: ${detail || res.status}`);
    }
    return res.json();
  } catch (err) {
    if (isNetworkError(err)) {
      const p = await tryRevive();
      if (p) {
        const res = await fetch(url());
        if (!res.ok) throw new Error(`${path}: ${res.status}`);
        return res.json();
      }
    }
    throw err;
  }
}

export async function health(): Promise<boolean> {
  try {
    await get("/health");
    return true;
  } catch {
    return false;
  }
}

export async function getDimensions(): Promise<Dimension[]> {
  const data = await get<{ dimensions: Dimension[] }>("/dimensions");
  return data.dimensions;
}

export interface ModelsResponse {
  models: string[];
  total: number;
  counts: Record<string, number>;
}

export async function getModels(): Promise<ModelsResponse> {
  return get("/models");
}

export async function getVocabularyTree(): Promise<VocabularyTree> {
  return get("/vocabulary/tree");
}

export async function getSiblings(term: string): Promise<Sibling[]> {
  const data = await get<{ siblings: Sibling[] }>(`/siblings/${encodeURIComponent(term)}`);
  return data.siblings;
}

export async function computeSlice(req: SliceRequest): Promise<SliceResponse> {
  return post("/slice", req);
}

export async function computeLayout(req: LayoutRequest): Promise<StripLayout> {
  return post("/layout", req);
}

export async function computeSpacelike(req: SpaceLikeRequest): Promise<SpaceLikeLayout> {
  return post("/spacelike", req);
}

export async function computeWireframe(req: { image_ids: string[]; model: string; k?: number }): Promise<{ edges: [string, string][] }> {
  return post("/wireframe", req);
}

export async function computeNeighborhoods(req: { image_ids: string[]; model: string; k?: number }): Promise<{ k: number; cluster_ids: Record<string, number> }> {
  return post("/neighborhoods", req);
}

export async function startImport(source: string): Promise<{ status: string }> {
  return post("/ingest/start", { source });
}

export async function getImportProgress(): Promise<ImportProgress> {
  return get("/ingest/progress");
}

export async function pauseImport(): Promise<{ status: string }> {
  return post("/ingest/pause", {});
}

export async function resumeImport(): Promise<{ status: string }> {
  return post("/ingest/resume", {});
}

export async function nukeCorpus(): Promise<{ status: string }> {
  return post("/corpus/nuke", {});
}

export async function runPixelFeatures(): Promise<{ status: string }> {
  return post("/tools/pixel-features", {});
}

export async function runMissingEmbeddings(): Promise<{ status: string }> {
  return post("/tools/embed-missing", {});
}

export async function regeneratePreviews(): Promise<{ status: string; count: number }> {
  return post("/tools/regenerate-previews", {});
}

export interface ImageMetadata {
  id: string;
  source_path: string | null;
  capture_date: number | null;
  pixel_width: number | null;
  pixel_height: number | null;
  gps_latitude: number | null;
  gps_longitude: number | null;
  camera_model: string | null;
  lens_model: string | null;
  focal_length: number | null;
  aperture: number | null;
  shutter_speed: number | null;
  iso: number | null;
}

export async function getImageMetadata(id: string): Promise<ImageMetadata | null> {
  try {
    return await get<ImageMetadata>(`/image/info/${encodeURIComponent(id)}`);
  } catch {
    return null;
  }
}

export function imageSourceUrl(id: string): string {
  return `http://127.0.0.1:${sidecarPort}/image/source/${encodeURIComponent(id)}`;
}

export function imagePreviewUrl(id: string): string {
  return `http://127.0.0.1:${sidecarPort}/preview/${encodeURIComponent(id)}`;
}

export function imageThumbnailUrl(id: string): string {
  return `http://127.0.0.1:${sidecarPort}/thumbnail/${encodeURIComponent(id)}`;
}

export async function getThingsLibrary(): Promise<string[]> {
  const data = await get<{ names: string[] }>("/things/library");
  return data.names;
}

export async function addThingToLibrary(name: string): Promise<string[]> {
  const data = await post<{ names: string[] }>("/things/library/add", { name });
  return data.names;
}

export async function removeThingFromLibrary(name: string): Promise<string[]> {
  const data = await post<{ names: string[] }>("/things/library/remove", { name });
  return data.names;
}

// ── Collages ──────────────────────────────────────────────────────────────

export interface CollageSummary {
  id: string;
  name: string;
  created_at: number;
  modified_at: number;
  mode: string;
  model: string;
  relevance: number;
  feathering: number;
  cell_size: number;
  has_thumbnail: boolean;
}

export interface CollageDetail extends Omit<CollageSummary, "has_thumbnail"> {
  expression_json: string; // JSON-stringified Expression | null
  pov_json: string;        // JSON-stringified PointOfView
}

export interface SaveCollageRequest {
  name: string;
  expression: import("./relevance").Expression | null;
  pov: import("./types").PointOfView;
  mode: "spacelike" | "timelike";
  model: string;
  relevance: number;
  feathering: number;
  cell_size: number;
  thumbnail_base64?: string;
}

export async function listCollages(): Promise<CollageSummary[]> {
  const data = await get<{ collages: CollageSummary[] }>("/collages");
  return data.collages;
}

export async function fetchCollage(id: string): Promise<CollageDetail> {
  return get<CollageDetail>(`/collages/${encodeURIComponent(id)}`);
}

export function collageThumbnailUrl(id: string): string {
  if (!sidecarPort) throw new Error("Sidecar port not set");
  return `http://127.0.0.1:${sidecarPort}/collages/${encodeURIComponent(id)}/thumbnail`;
}

export async function saveCollage(req: SaveCollageRequest): Promise<{ id: string; collages: CollageSummary[] }> {
  return post("/collages/save", req);
}

export async function renameCollage(id: string, name: string): Promise<CollageSummary[]> {
  const data = await post<{ collages: CollageSummary[] }>("/collages/rename", { id, name });
  return data.collages;
}

export async function deleteCollage(id: string): Promise<CollageSummary[]> {
  const data = await post<{ collages: CollageSummary[] }>("/collages/delete", { id });
  return data.collages;
}

// Collage as a `.sigil` directory (file-based, browseable in Finder).

export interface ExportCollageRequest {
  parent_path: string;
  name_hint?: string;
  expression: import("./relevance").Expression | null;
  pov: import("./types").PointOfView;
  mode: "spacelike" | "timelike";
  model: string;
  relevance: number;
  feathering: number;
  cell_size: number;
  attractors: import("./types").Attractor[]; // for naming hint (Thing pills)
  image_ids: string[]; // for CLIP-centroid naming when no pills
  screenshot_base64?: string; // full-resolution canvas PNG
}

export interface ExportCollageResponse {
  folder_path: string;
  name: string;
}

export async function exportCollage(req: ExportCollageRequest): Promise<ExportCollageResponse> {
  return post("/collages/export", req);
}

export interface CollageManifest {
  version: number;
  name: string;
  saved_at: number;
  expression: import("./relevance").Expression | null;
  pov: import("./types").PointOfView;
  mode: "spacelike" | "timelike";
  model: string;
  relevance: number;
  feathering: number;
  cell_size: number;
  image_ids: string[];
}

export async function importCollage(folder_path: string): Promise<{ collage: CollageManifest; folder_path: string }> {
  return post("/collages/import", { folder_path });
}
