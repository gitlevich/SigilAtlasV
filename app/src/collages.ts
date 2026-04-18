/**
 * Collage save/load — file-based via Tauri's native dialogs.
 *
 * Per spec (root `affordance-create-sigil` + `Explore/Collage/language.md`):
 * a saved view is a `.sigil` directory carrying narrative (`language.md`),
 * machine state (`collage.json`), and assets (`thumbnail.png`). Save asks the
 * user for a parent location via the native folder picker; the folder name is
 * auto-derived (active Thing pills, else CLIP-centroid against a 30k-word
 * dictionary, else timestamp). Open also uses the native folder picker.
 */

import { state, notify } from "./state";
import * as api from "./api";
import { buildFilter } from "./relevance";
import type { Attractor, ContrastControl, RangeFilter, PointOfView } from "./types";
import { recomputeSliceAndLayout } from "./ui/controls";

/** Capture the WebGL canvas at native resolution as a PNG, base64-encoded
 *  (no `data:` prefix). On Retina displays the canvas is already at
 *  device-pixel size, so this gives the user an honest full-rez screenshot. */
function captureScreenshot(): string | null {
  const canvas = document.getElementById("viewport") as HTMLCanvasElement | null;
  if (!canvas) return null;
  if (canvas.width === 0 || canvas.height === 0) return null;
  // toDataURL on the WebGL canvas works because we set
  // preserveDrawingBuffer:true at context creation.
  const dataUrl = canvas.toDataURL("image/png");
  const idx = dataUrl.indexOf(",");
  return idx >= 0 ? dataUrl.slice(idx + 1) : null;
}

async function pickParentFolder(title: string): Promise<string | null> {
  try {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const selected = await open({
      directory: true,
      multiple: false,
      title,
    });
    return typeof selected === "string" ? selected : null;
  } catch (e) {
    console.error("[collage dialog]", e);
    return null;
  }
}

/**
 * Save the current view as a `.sigil` collage folder.
 * Native folder picker → user chooses parent location.
 * Returns the path of the new folder, or null if cancelled / failed.
 */
export async function saveCurrentAsCollage(): Promise<string | null> {
  const parent = await pickParentFolder("Save Sigil Into\u2026");
  if (!parent) return null;

  const expression = buildFilter({
    attractors: state.attractors,
    contrastControls: state.contrastControls,
    rangeFilters: state.rangeFilters,
  });
  const screenshot_base64 = captureScreenshot() ?? undefined;

  try {
    const res = await api.exportCollage({
      parent_path: parent,
      expression,
      pov: { ...state.pov },
      mode: state.mode,
      model: state.model,
      relevance: state.relevance,
      feathering: state.feathering,
      cell_size: state.cellSize,
      attractors: state.attractors,
      image_ids: state.imageIds,
      screenshot_base64,
    });
    state.lastError = null;
    notify();
    console.info("[collage] saved to", res.folder_path);
    return res.folder_path;
  } catch (e) {
    console.error("[collage save]", e);
    state.lastError = `Save failed: ${e instanceof Error ? e.message : e}`;
    notify();
    return null;
  }
}

/**
 * Load a `.sigil` collage folder. Native folder picker → user selects the
 * `.sigil` folder. Re-evaluates the saved expression against the current
 * corpus, applies arrangement params, and restores the camera.
 */
export async function openCollage(): Promise<void> {
  const folder = await pickParentFolder("Open Sigil\u2026");
  if (!folder) return;
  await loadCollageFromFolder(folder);
}

/** Apply a collage manifest to state. Triggers recompute and restores POV. */
export async function loadCollageFromFolder(folderPath: string): Promise<void> {
  let manifest: api.CollageManifest;
  try {
    const res = await api.importCollage(folderPath);
    manifest = res.collage;
  } catch (e) {
    console.error("[collage load]", e);
    state.lastError = `Load failed: ${e instanceof Error ? e.message : e}`;
    notify();
    return;
  }

  const { attractors, contrastControls, rangeFilters } = projectExpressionToState(manifest.expression);
  state.attractors = attractors;
  state.contrastControls = contrastControls;
  state.rangeFilters = rangeFilters;
  state.relevance = manifest.relevance;
  state.feathering = manifest.feathering;
  state.cellSize = manifest.cell_size;
  state.model = manifest.model;
  state.mode = (manifest.mode === "timelike" ? "timelike" : "spacelike");

  await recomputeSliceAndLayout();
  state.pov = { ...manifest.pov };
  notify();
}

/** Best-effort: turn a saved AST back into the UI-shaped attractors/contrasts/ranges. */
function projectExpressionToState(expr: unknown): {
  attractors: Attractor[];
  contrastControls: ContrastControl[];
  rangeFilters: RangeFilter[];
} {
  const attractors: Attractor[] = [];
  const contrastControls: ContrastControl[] = [];
  const rangeFilters: RangeFilter[] = [];

  const visit = (node: unknown) => {
    if (!node || typeof node !== "object") return;
    const n = node as Record<string, unknown>;
    switch (n.type) {
      case "thing":
        attractors.push({ kind: "thing", ref: String(n.name) });
        return;
      case "target_image":
        attractors.push({ kind: "target_image", ref: String(n.image_id) });
        return;
      case "contrast":
        contrastControls.push({
          pole_a: String(n.pole_a),
          pole_b: String(n.pole_b),
          band_min: Number(n.band_min),
          band_max: Number(n.band_max),
        });
        return;
      case "range":
        rangeFilters.push({
          dimension: String(n.dimension),
          min: Number(n.min),
          max: Number(n.max),
        });
        return;
      case "and":
      case "or":
        for (const c of (n.children as unknown[]) ?? []) visit(c);
        return;
      case "not":
        visit(n.child);
        return;
    }
  };
  visit(expr);
  return { attractors, contrastControls, rangeFilters };
}

// Legacy in-panel collages (SQLite) wrappers — kept so the existing panel can
// still browse old saves until the panel itself is removed in a follow-up.

import { refreshCollages } from "./state";

export async function renameCollageById(id: string, name: string): Promise<void> {
  try {
    state.collages = await api.renameCollage(id, name);
    notify();
  } catch (e) {
    console.error("[collage rename]", e);
  }
}

export async function deleteCollageById(id: string): Promise<void> {
  try {
    state.collages = await api.deleteCollage(id);
    notify();
  } catch (e) {
    console.error("[collage delete]", e);
  }
}

export async function loadCollage(id: string): Promise<void> {
  let detail: api.CollageDetail;
  try {
    detail = await api.fetchCollage(id);
  } catch (e) {
    console.error("[collage load]", e);
    state.lastError = `Load failed: ${e instanceof Error ? e.message : e}`;
    notify();
    return;
  }
  const expression = JSON.parse(detail.expression_json);
  const pov: PointOfView = JSON.parse(detail.pov_json);
  const { attractors, contrastControls, rangeFilters } = projectExpressionToState(expression);
  state.attractors = attractors;
  state.contrastControls = contrastControls;
  state.rangeFilters = rangeFilters;
  state.relevance = detail.relevance;
  state.feathering = detail.feathering;
  state.cellSize = detail.cell_size;
  state.model = detail.model;
  state.mode = (detail.mode === "timelike" ? "timelike" : "spacelike");
  await recomputeSliceAndLayout();
  state.pov = pov;
  notify();
}

export { refreshCollages };
