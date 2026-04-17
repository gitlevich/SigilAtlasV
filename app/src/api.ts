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

export function setSidecarPort(port: number): void {
  sidecarPort = port;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  if (!sidecarPort) throw new Error("Sidecar port not set");
  const res = await fetch(`http://127.0.0.1:${sidecarPort}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${path}: ${detail || res.status}`);
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  if (!sidecarPort) throw new Error("Sidecar port not set");
  const res = await fetch(`http://127.0.0.1:${sidecarPort}${path}`);
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${path}: ${detail || res.status}`);
  }
  return res.json();
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

export async function getModels(): Promise<string[]> {
  const data = await get<{ models: string[] }>("/models");
  return data.models;
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
