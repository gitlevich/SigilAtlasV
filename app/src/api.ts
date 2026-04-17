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
  VocabTerm,
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

export async function getVocabularyFlat(): Promise<VocabTerm[]> {
  return get("/vocabulary/flat");
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
