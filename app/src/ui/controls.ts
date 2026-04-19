/**
 * UI controls — SigilControls per spec.
 *
 * Right panel with Lightroom-style collapsible sections:
 * Mode, Attract, Contrasts, Color, Tone, Settings.
 */

import {
  state, notify, subscribe,
  setPanelSectionCollapsed, setLeftPanelFolded, setRightPanelFolded, setRightPanelWidth,
  currentSigilsFolder, setSigilsFolder, refreshWorkspaceSigils,
} from "../state";
import * as api from "../api";
import { collageThumbnailUrl } from "../api";
import type { AnyLayout, Dimension, ContrastControl, PointOfView, VocabularyTree, VocabularyNode } from "../types";
import { isSpaceLikeLayout } from "../types";
import { buildFilter } from "../relevance";
import { loadCollageFromFolder } from "../collages";
import type { TorusViewport } from "../renderer/torus-viewport";
import { createDiscriminateWidget } from "./discriminate-widget";
import { createColorWheel, hueRangeToFilter, type HueRange } from "./color-wheel";
import { startImport } from "../import";

// Vocabulary cached as a tree — the SpaceLike form of the sigil. Fetched
// once per init. Both autocomplete and the taxonomy browser walk this tree;
// the flat list is not used here (per spec: the tree is the simultaneous
// shape, flat is a TimeLike enumeration of trajectories through it).
// Per spec (Thing/language.md): "the @taxonomy is scaffolding, not a gate" —
// we never reject input that isn't in the tree.
let vocabulary: VocabularyTree = {};

// Subscribers that repaint when vocabulary arrives from the sidecar.
const vocabReadyListeners: Array<() => void> = [];
function notifyVocabReady(): void {
  for (const fn of vocabReadyListeners) fn();
}

// Walk the tree, emitting {node, ancestry} for every node. Ancestry is the
// chain of ancestor names from the root sigil down to (but not including)
// the node itself. Used by autocomplete and by the taxonomy browser.
interface WalkedNode {
  node: VocabularyNode;
  ancestry: string[];
  sigil: string;
}
function walkVocabulary(tree: VocabularyTree, visit: (w: WalkedNode) => void): void {
  function recurse(node: VocabularyNode, sigil: string, ancestry: string[]): void {
    visit({ node, ancestry, sigil });
    if (node.children) {
      const next = ancestry.concat(node.name);
      for (const c of node.children) recurse(c, sigil, next);
    }
  }
  for (const [sigil, roots] of Object.entries(tree)) {
    for (const root of roots) recurse(root, sigil, []);
  }
}

// Re-render hooks — set by build*Section so cross-section actions (e.g.
// activating a library pill into Attract) can refresh the affected views
// without rebuilding the entire panel.
let renderAttractPills: (() => void) | null = null;
let renderLibraryPills: (() => void) | null = null;

/** Add a Thing as an active attractor. Releases any TargetImage first —
 *  naming a new search is signalling intent to attend to something new,
 *  and the layout would otherwise still rank by similarity to a stale
 *  focal point. De-dupes the Thing by name. */
function addThingAttractor(name: string): boolean {
  const hadTarget = state.attractors.some((a) => a.kind === "target_image");
  const hasThing = state.attractors.some((a) => a.kind === "thing" && a.ref === name);
  if (hasThing && !hadTarget) return false;
  state.attractors = state.attractors.filter((a) => a.kind !== "target_image");
  if (!hasThing) state.attractors.push({ kind: "thing", ref: name });
  return true;
}

/** Add a name to the ThingsLibrary if not already present, and persist to the workspace. */
function captureInLibrary(name: string): void {
  if (!name) return;
  if (state.thingsLibrary.includes(name)) return;
  state.thingsLibrary.push(name);
  renderLibraryPills?.();
  api.addThingToLibrary(name)
    .then((names) => {
      state.thingsLibrary = names;
      renderLibraryPills?.();
    })
    .catch((e) => console.error("[library] add failed:", e));
}

let sliceDebounceTimer: ReturnType<typeof setTimeout> | null = null;
function debouncedRecompute(): void {
  if (sliceDebounceTimer) clearTimeout(sliceDebounceTimer);
  sliceDebounceTimer = setTimeout(() => {
    recomputeSliceAndLayout().catch((e) => console.error("Slice failed:", e));
  }, 250);
}

let viewport: TorusViewport | null = null;

export function setViewport(vp: TorusViewport): void {
  viewport = vp;
}

// Cached on the last initControls call so callers (e.g. sigil load) can
// rebuild the panels from the current state without a server round-trip.
let cachedDimensions: Dimension[] = [];
let cachedModelsRes: api.ModelsResponse | null = null;

/** Re-fetch dimensions and models from the server and rebuild the control panels. */
export async function refreshControls(): Promise<void> {
  const [dimensions, modelsRes] = await Promise.all([
    api.getDimensions(),
    api.getModels(),
  ]);
  const complete = modelsRes.models.filter(
    (m) => (modelsRes.counts[m] ?? 0) >= modelsRes.total,
  );
  if (complete.length > 0 && !complete.includes(state.model)) {
    state.model = complete[0];
  } else if (modelsRes.models.length > 0 && !modelsRes.models.includes(state.model)) {
    state.model = modelsRes.models[0];
  }
  await initControls(dimensions, modelsRes);
  notify();
}

/** Rebuild every control widget from current state, using the dimensions and
 *  models already fetched on startup. Cheap (no API calls) and idempotent;
 *  the sigil-load path calls this so opened sigils flow through the same
 *  control surface the user types into. */
export async function rebuildControlsFromState(): Promise<void> {
  if (!cachedModelsRes) return;
  await initControls(cachedDimensions, cachedModelsRes);
  notify();
}

export async function recomputeSliceAndLayout(opts?: { anchorImageId?: string }): Promise<void> {
  // Cancel any pending debounced recompute to avoid stale overwrites
  if (sliceDebounceTimer) { clearTimeout(sliceDebounceTimer); sliceDebounceTimer = null; }

  try {
    const filter = buildFilter({
      attractors: state.attractors,
      contrastControls: state.contrastControls,
      rangeFilters: state.rangeFilters,
    });
    const res = await api.computeSlice({
      filter,
      relevance: state.relevance,
      model: state.model,
    });
    state.imageIds = res.image_ids;
    state.orderValues = res.order_values || {};

    // Anchor: the image at screen centre in the current layout. We'll try to
    // keep this same image at screen centre after the recompute, regardless
    // of mode switch or slice change.
    const prevLayout = state.layout;
    // Explicit anchor (e.g. option-click target) overrides the
    // "whatever is at screen centre" auto-anchor.
    const anchorImageId = opts?.anchorImageId ?? (prevLayout ? imageAtPov(prevLayout, state.pov) : null);
    const isFirstLoad = state.torusWidth === 0;

    const canvas = document.getElementById("viewport") as HTMLCanvasElement;
    const aspect = canvas.clientWidth / canvas.clientHeight;

    let newLayout: AnyLayout;
    let attractorCell: { col: number; row: number; cell_size: number } | null = null;

    if (state.mode === "spacelike") {
      const layout = await api.computeSpacelike({
        filter,
        relevance: state.relevance,
        model: state.model,
        feathering: state.feathering,
        cell_size: state.cellSize,
        field_expansion: state.fieldExpansion,
        arrangement: state.arrangement,
      });
      newLayout = layout;
      if (layout.attractor_positions.length > 0) {
        const a = layout.attractor_positions[0];
        attractorCell = { col: a.col, row: a.row, cell_size: layout.cell_size };
      }
    } else {
      const hasOV = Object.keys(state.orderValues).length > 0;
      const orderValues = hasOV ? state.orderValues : undefined;
      newLayout = await api.computeLayout({
        image_ids: state.imageIds,
        axes: state.selectedAxes.length > 0 ? state.selectedAxes : null,
        feathering: state.feathering,
        model: state.model,
        strip_height: state.stripHeight,
        preserve_order: false,
        order_values: orderValues,
      });
    }

    state.layout = newLayout;
    state.torusWidth = newLayout.torus_width;
    state.torusHeight = newLayout.torus_height;
    const maxZoom = Math.min(newLayout.torus_width, newLayout.torus_height * aspect);

    // Framing priority, high to low:
    //  1. Attractor just added: snap to the attractor peak cell, tight zoom.
    //  2. Anchor image from before still exists: centre on its new world pos,
    //     preserving zoom (clamped to new maxZoom).
    //  3. First ever load: full-torus splash.
    //  4. Fallback (rare): geometric centre.
    if (attractorCell && !anchorImageId) {
      // New attractor with no prior anchor: zoom in on it. ~10 visible rows
      // keeps display cell height above the per-image-atlas threshold (80px),
      // so cells render at the sharp tier rather than the blurry mid-atlas.
      const visibleCells = 10;
      const desiredZoom = visibleCells * attractorCell.cell_size * aspect;
      state.pov.x = (attractorCell.col + 0.5) * attractorCell.cell_size;
      state.pov.y = (attractorCell.row + 0.5) * attractorCell.cell_size;
      state.pov.z = Math.min(desiredZoom, maxZoom);
    } else if (anchorImageId) {
      const newCentre = worldCenterOfImage(newLayout, anchorImageId);
      if (newCentre) {
        state.pov.x = newCentre.x;
        state.pov.y = newCentre.y;
        state.pov.z = Math.min(state.pov.z, maxZoom);
      } else if (attractorCell) {
        // Anchor fell out of slice; fall back to attractor framing.
        const visibleCells = 10;
        const desiredZoom = visibleCells * attractorCell.cell_size * aspect;
        state.pov.x = (attractorCell.col + 0.5) * attractorCell.cell_size;
        state.pov.y = (attractorCell.row + 0.5) * attractorCell.cell_size;
        state.pov.z = Math.min(desiredZoom, maxZoom);
      } else {
        state.pov.x = newLayout.torus_width / 2;
        state.pov.y = newLayout.torus_height / 2;
        state.pov.z = Math.min(state.pov.z, maxZoom);
      }
    } else if (isFirstLoad) {
      // ~6 rows visible: legible cells, fast mode switches.
      const cellSize = isSpaceLikeLayout(newLayout) ? newLayout.cell_size : newLayout.strip_height;
      const desiredZoom = 6 * cellSize * aspect;
      state.pov.x = newLayout.torus_width / 2;
      state.pov.y = newLayout.torus_height / 2;
      state.pov.z = Math.min(desiredZoom, maxZoom);
    } else {
      state.pov.x = newLayout.torus_width / 2;
      state.pov.y = newLayout.torus_height / 2;
      state.pov.z = Math.min(state.pov.z, maxZoom);
    }

    if (viewport) viewport.setLayout(newLayout);

    updateImageCount();
    state.lastError = null;

    // Wireframe survives re-layouts but its edges must be recomputed
    // against the new slice. Refresh in the background.
    if (state.mode === "spacelike" && state.layers.neighborhoods) {
      fetchAndSetNeighborhoods().catch((err) => console.error("[neighborhoods]", err));
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("[recompute]", msg);
    state.lastError = msg;
  }

  notify();
}

function updateImageCount(): void {
  const el = document.getElementById("image-count");
  if (el) el.textContent = `${state.imageIds.length} images`;
}


// ── Camera anchoring: the image under screen centre stays put across
// recomputes, mode switches, and slice changes. Pan/zoom follow the image,
// not the geometry.

/** World centre of the cell/image under a world position, or null if the
 *  position falls outside any cell. Used by double-click to snap-centre on
 *  the clicked image rather than the literal clicked pixel.
 */
export function cellCenterAtWorld(layout: AnyLayout, worldX: number, worldY: number): { x: number; y: number } | null {
  if (isSpaceLikeLayout(layout)) {
    if (layout.cols === 0 || layout.rows === 0) return null;
    const wx = ((worldX % layout.torus_width) + layout.torus_width) % layout.torus_width;
    const wy = ((worldY % layout.torus_height) + layout.torus_height) % layout.torus_height;
    const col = Math.min(layout.cols - 1, Math.max(0, Math.floor(wx / layout.cell_size)));
    const row = Math.min(layout.rows - 1, Math.max(0, Math.floor(wy / layout.cell_size)));
    return {
      x: (col + 0.5) * layout.cell_size,
      y: (row + 0.5) * layout.cell_size,
    };
  }
  const tw = layout.torus_width, th = layout.torus_height;
  if (tw === 0 || th === 0) return null;
  const wx = ((worldX % tw) + tw) % tw;
  const wy = ((worldY % th) + th) % th;
  for (const strip of layout.strips) {
    if (wy >= strip.y && wy < strip.y + strip.height) {
      for (const img of strip.images) {
        if (wx >= img.x && wx < img.x + img.width) {
          return { x: img.x + img.width / 2, y: strip.y + strip.height / 2 };
        }
      }
      break;
    }
  }
  return null;
}

export function imageAtWorld(layout: AnyLayout, worldX: number, worldY: number): string | null {
  if (isSpaceLikeLayout(layout)) {
    if (layout.cols === 0 || layout.rows === 0) return null;
    const wx = ((worldX % layout.torus_width) + layout.torus_width) % layout.torus_width;
    const wy = ((worldY % layout.torus_height) + layout.torus_height) % layout.torus_height;
    const col = Math.min(layout.cols - 1, Math.max(0, Math.floor(wx / layout.cell_size)));
    const row = Math.min(layout.rows - 1, Math.max(0, Math.floor(wy / layout.cell_size)));
    for (const p of layout.positions) {
      if (p.col === col && p.row === row) return p.id;
    }
    return null;
  }
  const tw = layout.torus_width, th = layout.torus_height;
  if (tw === 0 || th === 0) return null;
  const wx = ((worldX % tw) + tw) % tw;
  const wy = ((worldY % th) + th) % th;
  for (const strip of layout.strips) {
    if (wy >= strip.y && wy < strip.y + strip.height) {
      for (const img of strip.images) {
        if (wx >= img.x && wx < img.x + img.width) return img.id;
      }
      break;
    }
  }
  return null;
}

function imageAtPov(layout: AnyLayout, pov: PointOfView): string | null {
  return imageAtWorld(layout, pov.x, pov.y);
}

/** Step to an adjacent image on the layout's lattice. Directions are
 *  screen-aligned: "up" goes to the row above (smaller row index in the
 *  SpaceLike grid, earlier strip in TimeLike), "left" to the previous image
 *  in reading order on the same row. The torus wraps on both axes, so the
 *  walk never hits a hard edge.
 *
 *  Used by the @Lightbox to walk the suspended @arrangement's lattice with
 *  arrow keys.
 */
export function imageNeighbor(
  layout: AnyLayout,
  id: string,
  dir: "up" | "down" | "left" | "right",
): string | null {
  if (isSpaceLikeLayout(layout)) {
    const here = layout.positions.find((p) => p.id === id);
    if (!here) return null;
    const dc = dir === "left" ? -1 : dir === "right" ? 1 : 0;
    const dr = dir === "up" ? -1 : dir === "down" ? 1 : 0;
    const cols = layout.cols, rows = layout.rows;
    if (cols === 0 || rows === 0) return null;
    const col = ((here.col + dc) % cols + cols) % cols;
    const row = ((here.row + dr) % rows + rows) % rows;
    for (const p of layout.positions) {
      if (p.col === col && p.row === row) return p.id;
    }
    return null;
  }
  // Strip layout: up/down move between strips, left/right within the strip.
  const strips = layout.strips;
  if (strips.length === 0) return null;
  let stripIdx = -1, imgIdx = -1;
  for (let i = 0; i < strips.length && stripIdx < 0; i++) {
    const j = strips[i].images.findIndex((im) => im.id === id);
    if (j >= 0) { stripIdx = i; imgIdx = j; }
  }
  if (stripIdx < 0) return null;
  if (dir === "left" || dir === "right") {
    const imgs = strips[stripIdx].images;
    const n = imgs.length;
    if (n === 0) return null;
    const next = ((imgIdx + (dir === "right" ? 1 : -1)) % n + n) % n;
    return imgs[next].id;
  }
  const n = strips.length;
  const nextStrip = ((stripIdx + (dir === "down" ? 1 : -1)) % n + n) % n;
  const imgs = strips[nextStrip].images;
  if (imgs.length === 0) return null;
  // Pick the image nearest by x to the one we came from, so vertical walks
  // don't snap to the start of each strip.
  const here = strips[stripIdx].images[imgIdx];
  const hereCenter = here.x + here.width / 2;
  let best = imgs[0];
  let bestDist = Math.abs((best.x + best.width / 2) - hereCenter);
  for (let i = 1; i < imgs.length; i++) {
    const c = imgs[i].x + imgs[i].width / 2;
    const d = Math.abs(c - hereCenter);
    if (d < bestDist) { best = imgs[i]; bestDist = d; }
  }
  return best.id;
}

function worldCenterOfImage(layout: AnyLayout, id: string): { x: number; y: number } | null {
  if (isSpaceLikeLayout(layout)) {
    for (const p of layout.positions) {
      if (p.id === id) {
        return {
          x: (p.col + 0.5) * layout.cell_size,
          y: (p.row + 0.5) * layout.cell_size,
        };
      }
    }
    return null;
  }
  for (const strip of layout.strips) {
    for (const img of strip.images) {
      if (img.id === id) {
        return { x: img.x + img.width / 2, y: strip.y + strip.height / 2 };
      }
    }
  }
  return null;
}


// ── Section builder ──

function createSection(title: string, collapsed = false): { section: HTMLElement; body: HTMLElement } {
  // Persisted workspace fold state trumps the section's own default.
  const saved = state.panels.sections[title];
  const initialCollapsed = saved ?? collapsed;

  const section = document.createElement("div");
  section.className = "section" + (initialCollapsed ? " collapsed" : "");

  const header = document.createElement("div");
  header.className = "section-header";
  header.textContent = title;
  header.addEventListener("click", () => {
    const isCollapsed = section.classList.toggle("collapsed");
    setPanelSectionCollapsed(title, isCollapsed);
  });
  section.appendChild(header);

  const body = document.createElement("div");
  body.className = "section-body";
  section.appendChild(body);

  return { section, body };
}


// ── Range filter helper ──

function setRangeFilter(dimension: string, min: number, max: number): void {
  const existing = state.rangeFilters.findIndex((f) => f.dimension === dimension);
  const filter = { dimension, min, max };
  if (existing >= 0) state.rangeFilters[existing] = filter;
  else state.rangeFilters.push(filter);
}

function removeRangeFilter(dimension: string): void {
  const idx = state.rangeFilters.findIndex((f) => f.dimension === dimension);
  if (idx >= 0) state.rangeFilters.splice(idx, 1);
}


async function fetchAndSetNeighborhoods(): Promise<void> {
  if (!viewport || state.imageIds.length === 0) return;
  const btn = [...document.querySelectorAll(".layer-toggle")].find(
    (b) => b.textContent === "Neighborhoods" || b.textContent === "Neighborhoods\u2026",
  ) as HTMLButtonElement | undefined;
  if (btn) btn.textContent = "Neighborhoods\u2026";
  try {
    const res = await api.computeNeighborhoods({
      image_ids: state.imageIds,
      model: state.model,
      k: 50,
    });
    viewport.setNeighborhoodClusters(res.cluster_ids);
  } finally {
    if (btn) btn.textContent = "Neighborhoods";
  }
}


// ── Layers (Photos / Neighborhoods) ──

function buildLayersSection(body: HTMLElement): void {
  const row = document.createElement("div");
  row.className = "layer-toggles";

  const defs: Array<{ key: keyof typeof state.layers; label: string; hint: string }> = [
    { key: "photos", label: "Photos", hint: "image tiles" },
    { key: "neighborhoods", label: "Neighborhoods", hint: "KMeans cluster boundaries" },
  ];

  for (const def of defs) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "layer-toggle" + (state.layers[def.key] ? " on" : "");
    btn.textContent = def.label;
    btn.title = def.hint;
    btn.addEventListener("click", () => {
      state.layers[def.key] = !state.layers[def.key];
      btn.classList.toggle("on", state.layers[def.key]);
      if (def.key === "neighborhoods" && state.layers.neighborhoods) {
        fetchAndSetNeighborhoods().catch((err) => console.error("[neighborhoods]", err));
      }
    });
    row.appendChild(btn);
  }

  body.appendChild(row);

  const hint = document.createElement("div");
  hint.className = "section-hint";
  hint.textContent = "";
  body.appendChild(hint);
}


// ── Attract input ──
//
// Plain free-text entry. Per spec (Thing/language.md): "the @taxonomy is
// scaffolding, not a gate" — anything typed is a valid @thing. The taxonomy
// browser below handles discovery; this input stays out of the way.

function createAttractInput(commit: (name: string) => void): HTMLElement {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "pill-input";
  input.placeholder = "type a thing, press enter...";

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const chosen = input.value.trim();
      if (!chosen) return;
      input.value = "";
      commit(chosen);
    }
  });

  return input;
}


// ── Things Library ──
//
// Per `Explore/Control/ThingsLibrary/language.md`: every @thing the user has
// named lives here, persistent across sessions. Click a pill to #activate it
// into AttractorControl (the slice recomputes as if just typed). Press × to
// #delete from the library entirely (with confirmation). #search filters the
// pills by substring; #fold collapses the section to its header.

// ── Collages ──
//
// Per `Explore/Collage/language.md`: a saved view is a serialized SigilML
// expression plus camera POV plus arrangement params. Click a collage to
// re-evaluate and restore the camera; double-click name to rename; × deletes.

let renderCollages: (() => void) | null = null;

function buildCollagesSection(body: HTMLElement): void {
  const list = document.createElement("div");
  list.className = "collages-list";
  body.appendChild(list);

  async function pickSigilsFolder(): Promise<void> {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({ directory: true, title: "Choose sigils folder" });
      if (typeof selected !== "string") return;
      setSigilsFolder(selected);
      await refreshWorkspaceSigils();
    } catch (e) {
      console.error("[sigils folder pick]", e);
    }
  }

  const render = () => {
    list.innerHTML = "";
    const folder = currentSigilsFolder();

    if (!folder) {
      const empty = document.createElement("div");
      empty.className = "library-empty";
      empty.textContent = "no sigils folder chosen";
      list.appendChild(empty);
      const btn = document.createElement("button");
      btn.className = "workspace-pick-btn";
      btn.textContent = "Choose Sigils Folder\u2026";
      btn.addEventListener("click", () => pickSigilsFolder());
      list.appendChild(btn);
      return;
    }

    if (state.workspaceSigils.length === 0) {
      const empty = document.createElement("div");
      empty.className = "library-empty";
      empty.textContent = "no sigils in this folder \u2014 press \u2318S to save one";
      list.appendChild(empty);
      return;
    }

    for (const s of state.workspaceSigils) {
      const row = document.createElement("div");
      row.className = "collage-row";
      if (s.modified_at) {
        row.title = `Saved ${new Date(s.modified_at * 1000).toLocaleString()}`;
      }

      const thumb = document.createElement("div");
      thumb.className = "collage-thumb";
      if (s.preview_data_url) {
        const img = document.createElement("img");
        img.src = s.preview_data_url;
        img.alt = s.name;
        thumb.appendChild(img);
      }
      thumb.addEventListener("click", () => {
        loadCollageFromFolder(s.folder_path).catch((e) => console.error("[sigil load]", e));
      });
      row.appendChild(thumb);

      const meta = document.createElement("div");
      meta.className = "collage-meta";
      const nameEl = document.createElement("div");
      nameEl.className = "collage-name";
      nameEl.textContent = s.name;
      nameEl.addEventListener("click", () => {
        loadCollageFromFolder(s.folder_path).catch((e) => console.error("[sigil load]", e));
      });
      meta.appendChild(nameEl);
      row.appendChild(meta);

      const remove = document.createElement("span");
      remove.className = "pill-remove collage-delete";
      remove.textContent = "\u00d7";
      remove.title = "click to arm, click again to delete";
      remove.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!remove.classList.contains("armed")) {
          remove.classList.add("armed");
          remove.title = "click again to delete \u2014 leave the sigil to cancel";
          return;
        }
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          await invoke("delete_sigil", { folderPath: s.folder_path });
          const { refreshWorkspaceSigils } = await import("../state");
          await refreshWorkspaceSigils();
        } catch (err) {
          console.error("[sigil delete]", err);
          alert(`Delete failed: ${err}`);
        }
      });
      row.appendChild(remove);

      // Leaving the row disarms the delete so a stray subsequent hover
      // can't step into the armed state by accident.
      row.addEventListener("mouseleave", () => {
        remove.classList.remove("armed");
        remove.title = "click to arm, click again to delete";
      });

      list.appendChild(row);
    }
  };

  renderCollages = render;
  render();
}


function buildThingsLibrarySection(section: HTMLElement, body: HTMLElement): void {
  // Search row — toggled by clicking the magnifier in the header.
  const searchRow = document.createElement("div");
  searchRow.className = "library-search-row";
  searchRow.style.display = "none";

  const searchInput = document.createElement("input");
  searchInput.type = "text";
  searchInput.className = "pill-input";
  searchInput.placeholder = "filter library...";
  searchRow.appendChild(searchInput);
  body.appendChild(searchRow);

  const holder = document.createElement("div");
  holder.className = "pill-holder library-holder";
  body.appendChild(holder);

  let searchQuery = "";

  const render = () => {
    holder.innerHTML = "";

    const sorted = [...state.thingsLibrary].sort((a, b) => a.localeCompare(b));
    const filtered = searchQuery
      ? sorted.filter((n) => n.toLowerCase().includes(searchQuery))
      : sorted;

    if (filtered.length === 0) {
      const empty = document.createElement("span");
      empty.className = "library-empty";
      empty.textContent = state.thingsLibrary.length === 0
        ? "things you name will appear here"
        : "no matches";
      holder.appendChild(empty);
      return;
    }

    for (const name of filtered) {
      const isActive = state.attractors.some((a) => a.kind === "thing" && a.ref === name);
      const pill = document.createElement("span");
      pill.className = "pill library-pill" + (isActive ? " active" : "");
      pill.textContent = name;
      pill.title = isActive
        ? `${name} (active in Attract — click here is a no-op; remove via × in Attract)`
        : `click to activate as an attractor`;

      pill.addEventListener("click", (e) => {
        if ((e.target as HTMLElement).classList.contains("pill-remove")) return;
        if (isActive) return;
        if (!addThingAttractor(name)) return;
        renderAttractPills?.();
        render();
        recomputeSliceAndLayout().catch((err) => console.error("[library activate]", err));
      });

      const remove = document.createElement("span");
      remove.className = "pill-remove";
      remove.textContent = "\u00d7";
      remove.title = "delete from library (cannot be undone)";
      remove.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!confirm(`Delete "${name}" from the things library?`)) return;
        state.thingsLibrary = state.thingsLibrary.filter((n) => n !== name);
        // If active, also deactivate so we don't keep a name we've forgotten.
        const wasActive = state.attractors.some((a) => a.kind === "thing" && a.ref === name);
        state.attractors = state.attractors.filter(
          (a) => !(a.kind === "thing" && a.ref === name),
        );
        render();
        renderAttractPills?.();
        api.removeThingFromLibrary(name)
          .then((names) => {
            state.thingsLibrary = names;
            render();
          })
          .catch((err) => console.error("[library] remove failed:", err));
        if (wasActive) {
          recomputeSliceAndLayout().catch((err) => console.error("[library delete]", err));
        }
      });
      pill.appendChild(remove);
      holder.appendChild(pill);
    }
  };

  // Wire fold / search affordances into the section header.
  const header = section.querySelector(".section-header");
  if (header) {
    const tools = document.createElement("span");
    tools.className = "library-header-tools";

    const searchBtn = document.createElement("span");
    searchBtn.className = "library-tool";
    searchBtn.textContent = "\u2315"; // ⌕ search
    searchBtn.title = "search library";
    searchBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const showing = searchRow.style.display !== "none";
      searchRow.style.display = showing ? "none" : "";
      if (!showing) searchInput.focus();
      else { searchInput.value = ""; searchQuery = ""; render(); }
    });

    tools.appendChild(searchBtn);
    header.appendChild(tools);
  }

  searchInput.addEventListener("input", () => {
    searchQuery = searchInput.value.trim().toLowerCase();
    render();
  });

  renderLibraryPills = render;
  render();
}


// ── Attractors (things that bend the SpaceLike gravity field) ──

// ── Taxonomy tree browser ──
//
// Renders the vocabulary tree with collapsible branches. Clicking a node
// name adds it as a @Thing @Attractor — crude or rich (leaf or interior
// node both work; richness determines how the layout responds).

function buildTaxonomySection(body: HTMLElement): void {
  const container = document.createElement("div");
  container.className = "taxonomy-tree";
  body.appendChild(container);

  // Re-render on vocabulary load and on attractor changes (to mark which
  // nodes are currently active with a subtle highlight).
  const render = () => renderTaxonomyTree(container);
  render();
  vocabReadyListeners.push(render);
  subscribe(render);
}

function renderTaxonomyTree(container: HTMLElement): void {
  container.innerHTML = "";
  const entries = Object.entries(vocabulary);
  if (entries.length === 0) {
    const loading = document.createElement("div");
    loading.className = "taxonomy-empty";
    loading.textContent = "loading\u2026";
    container.appendChild(loading);
    return;
  }
  const active = new Set(
    state.attractors.filter((a) => a.kind === "thing").map((a) => a.ref),
  );
  for (const [sigil, roots] of entries) {
    // The sigil root is itself a clickable attractor — drop it to organize
    // the world from the point of view of the entire taxonomy (articulates
    // every top-level child as a peer anchor). Children render nested
    // below so you can also drop any sub-node directly.
    const group = document.createElement("div");
    group.className = "taxonomy-group";
    const header = document.createElement("div");
    header.className = "taxonomy-group-header";
    if (active.has(sigil)) header.classList.add("active");
    header.textContent = sigil;
    header.title = `Drop the entire ${sigil} taxonomy as an attractor`;
    header.addEventListener("click", () => {
      captureInLibrary(sigil);
      if (!addThingAttractor(sigil)) return;
      renderAttractPills?.();
      renderLibraryPills?.();
      recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
    });
    group.appendChild(header);
    for (const root of roots) {
      group.appendChild(renderTaxonomyNode(root, 0, active));
    }
    container.appendChild(group);
  }
}

function renderTaxonomyNode(
  node: VocabularyNode,
  depth: number,
  active: Set<string>,
): HTMLElement {
  const row = document.createElement("div");
  row.className = "taxonomy-node";
  row.style.paddingLeft = `${depth * 10}px`;

  const header = document.createElement("div");
  header.className = "taxonomy-row";

  const hasChildren = !!(node.children && node.children.length > 0);
  const caret = document.createElement("span");
  caret.className = "taxonomy-caret";
  caret.textContent = hasChildren ? "\u25b8" : "\u00a0";
  header.appendChild(caret);

  const name = document.createElement("span");
  name.className = "taxonomy-name";
  if (active.has(node.name)) name.classList.add("active");
  name.textContent = node.name;
  name.title = node.prompt;
  name.addEventListener("click", () => {
    captureInLibrary(node.name);
    if (!addThingAttractor(node.name)) return;
    renderAttractPills?.();
    renderLibraryPills?.();
    recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
  });
  header.appendChild(name);
  row.appendChild(header);

  if (hasChildren) {
    const childContainer = document.createElement("div");
    childContainer.className = "taxonomy-children collapsed";
    for (const c of node.children!) {
      childContainer.appendChild(renderTaxonomyNode(c, depth + 1, active));
    }
    row.appendChild(childContainer);
    const toggle = () => {
      const isCollapsed = childContainer.classList.toggle("collapsed");
      caret.textContent = isCollapsed ? "\u25b8" : "\u25be";
    };
    caret.addEventListener("click", (e) => {
      e.stopPropagation();
      toggle();
    });
  }
  return row;
}

function buildAttractSection(body: HTMLElement): void {
  const holder = document.createElement("div");
  holder.className = "pill-holder";

  const renderPills = () => {
    holder.querySelectorAll(".pill").forEach((el) => el.remove());
    for (let i = 0; i < state.attractors.length; i++) {
      const att = state.attractors[i];
      const pill = document.createElement("span");
      const idx = i;

      if (att.kind === "target_image") {
        // TargetImage pill carries a thumbnail of the actual image so the
        // user can see *which* image they pointed at — invisible state was
        // the original confusion and the thumbnail directly answers it.
        pill.className = "pill pill-target";
        const thumb = document.createElement("img");
        thumb.className = "pill-thumb";
        thumb.src = `${api.getThumbnailBaseUrl()}/thumbnail/${encodeURIComponent(att.ref)}`;
        thumb.alt = "target image";
        pill.appendChild(thumb);
        const label = document.createElement("span");
        label.className = "pill-label";
        label.textContent = "this image";
        pill.appendChild(label);
        pill.title = `Target image (option-clicked).\nClick × or press Escape to release.`;
      } else {
        pill.className = "pill";
        pill.textContent = att.ref;
      }

      const remove = document.createElement("span");
      remove.className = "pill-remove";
      remove.textContent = "\u00d7";
      remove.addEventListener("click", () => {
        state.attractors.splice(idx, 1);
        renderPills();
        renderLibraryPills?.();
        recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
      });
      pill.appendChild(remove);
      holder.insertBefore(pill, holder.lastElementChild);
    }
  };

  const attractInput = createAttractInput((name: string) => {
    captureInLibrary(name);
    if (!addThingAttractor(name)) return;
    renderPills();
    recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
  });

  holder.appendChild(attractInput);
  body.appendChild(holder);
  renderAttractPills = renderPills;
  renderPills();

  // Repaint when state.attractors changes by reference — covers option+click
  // (which assigns a new array and pushes) and any future external mutations.
  // The local renderPills() call after typed adds is still in place to avoid
  // depending on the next notify() tick for typed-pill visibility.
  let lastAttractors = state.attractors;
  subscribe((s) => {
    if (s.attractors !== lastAttractors) {
      lastAttractors = s.attractors;
      renderPills();
      renderLibraryPills?.();
    }
  });

  // Relevance slider — how strictly the named things gate the slice.
  // Per spec (Neighborhood/language.md): "distinctness of neighborhoods
  // comes from the @slice's @relevanceFilter, not from spatial separation."
  // This slider controls that membrane's permissiveness.
  const relevanceGroup = document.createElement("div");
  relevanceGroup.className = "control-group relevance-group";

  const relevanceLabel = document.createElement("label");
  relevanceLabel.textContent = "Relevance";
  relevanceGroup.appendChild(relevanceLabel);

  const relevanceSlider = document.createElement("input");
  relevanceSlider.type = "range";
  relevanceSlider.className = "thin-slider";
  relevanceSlider.min = "0";
  relevanceSlider.max = "1";
  relevanceSlider.step = "0.05";
  relevanceSlider.value = String(state.relevance);
  relevanceSlider.addEventListener("change", () => {
    state.relevance = parseFloat(relevanceSlider.value);
    recomputeSliceAndLayout().catch((e) => console.error("Relevance change failed:", e));
  });
  relevanceGroup.appendChild(relevanceSlider);

  const relevanceLabels = document.createElement("div");
  relevanceLabels.className = "slider-labels";
  relevanceLabels.innerHTML = "<span>loose</span><span>strict</span>";
  relevanceGroup.appendChild(relevanceLabels);

  body.appendChild(relevanceGroup);
}


// ── Contrasts ──

function buildContrastsSection(body: HTMLElement): void {
  const list = document.createElement("div");
  list.className = "contrast-list";

  const onChange = () => recomputeSliceAndLayout().catch((e) => console.error("Slice failed:", e));

  const renderContrasts = () => {
    list.innerHTML = "";
    for (let i = 0; i < state.contrastControls.length; i++) {
      list.appendChild(createContrastWidget(state.contrastControls[i], i, onChange, renderContrasts));
    }
  };

  const addBtn = document.createElement("div");
  addBtn.className = "add-contrast";

  const leftInput = document.createElement("input");
  leftInput.type = "text";
  leftInput.className = "pill-input";
  leftInput.placeholder = "pole A...";

  const vs = document.createElement("span");
  vs.className = "contrast-vs";
  vs.textContent = "vs";

  const rightInput = document.createElement("input");
  rightInput.type = "text";
  rightInput.className = "pill-input";
  rightInput.placeholder = "pole B...";

  const addContrast = () => {
    const a = leftInput.value.trim();
    const b = rightInput.value.trim();
    if (!a || !b) return;
    state.contrastControls.push({
      pole_a: `a photograph that is ${a}`,
      pole_b: `a photograph that is ${b}`,
      band_min: -1.0,
      band_max: 1.0,
    });
    leftInput.value = "";
    rightInput.value = "";
    renderContrasts();
    onChange();
  };

  rightInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); addContrast(); }
  });

  addBtn.appendChild(leftInput);
  addBtn.appendChild(vs);
  addBtn.appendChild(rightInput);

  body.appendChild(list);
  body.appendChild(addBtn);
}


function createEditablePole(
  cc: ContrastControl,
  pole: "pole_a" | "pole_b",
  onChange: () => void,
): HTMLElement {
  const stripPrefix = (s: string) => s.replace(/^a photograph (?:of |that is )/, "");
  const addPrefix = (s: string) => `a photograph that is ${s}`;

  const span = document.createElement("span");
  span.className = "pole-label";
  span.textContent = stripPrefix(cc[pole]);

  span.addEventListener("click", (e) => {
    e.stopPropagation();
    const input = document.createElement("input");
    input.type = "text";
    input.className = "pole-edit";
    input.value = stripPrefix(cc[pole]);
    span.replaceWith(input);
    input.focus();
    input.select();

    const commit = () => {
      const text = input.value.trim();
      if (text) {
        cc[pole] = addPrefix(text);
      }
      const newSpan = createEditablePole(cc, pole, onChange);
      input.replaceWith(newSpan);
      if (text) onChange();
    };

    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") { ev.preventDefault(); commit(); }
      if (ev.key === "Escape") { input.replaceWith(createEditablePole(cc, pole, onChange)); }
    });
    input.addEventListener("blur", commit);
  });

  return span;
}


function createContrastWidget(
  cc: ContrastControl, index: number,
  onChange: () => void, rerender: () => void,
): HTMLElement {
  const widget = document.createElement("div");
  widget.className = "contrast-widget";

  const stripPromptPrefix = (s: string) => s.replace(/^a photograph (?:of |that is )/, "");

  const header = document.createElement("div");
  header.className = "contrast-header";
  header.appendChild(createEditablePole(cc, "pole_b", onChange));
  const vsSpan = document.createElement("span");
  vsSpan.className = "contrast-vs";
  vsSpan.textContent = "vs";
  header.appendChild(vsSpan);
  header.appendChild(createEditablePole(cc, "pole_a", onChange));

  const removeBtn = document.createElement("span");
  removeBtn.className = "pill-remove";
  removeBtn.textContent = "\u00d7";
  removeBtn.addEventListener("click", () => {
    state.contrastControls.splice(index, 1);
    rerender();
    onChange();
  });
  header.appendChild(removeBtn);
  widget.appendChild(header);

  const bandWidget = createDiscriminateWidget({
    rangeMin: -1,
    rangeMax: 1,
    initial: { min: cc.band_min, max: cc.band_max },
    onCommit: (band) => {
      cc.band_min = band.min;
      cc.band_max = band.max;
      onChange();
    },
  });

  const labels = document.createElement("div");
  labels.className = "slider-labels";
  labels.innerHTML = `<span>${stripPromptPrefix(cc.pole_b)}</span><span>${stripPromptPrefix(cc.pole_a)}</span>`;

  widget.appendChild(bandWidget);
  widget.appendChild(labels);
  return widget;
}


// ── Color section ──

function buildColorSection(body: HTMLElement, dimensions: Dimension[]): void {
  const hueDim = dimensions.find((d) => d.name === "hue_dominant");
  if (!hueDim) return;

  const wheel = createColorWheel({
    size: 140,
    initial: { center: 0, width: 0 },
    onChange: (range: HueRange) => {
      const filter = hueRangeToFilter(range);
      if (filter) {
        // Handle wrapping: if min > max, it wraps around 360
        if (filter.min <= filter.max) {
          setRangeFilter("hue_dominant", filter.min, filter.max);
        } else {
          // For wrapping hue, set to full range and let the backend handle it
          // TODO: proper wrapping filter support
          setRangeFilter("hue_dominant", filter.min, filter.max);
        }
      } else {
        removeRangeFilter("hue_dominant");
      }
      debouncedRecompute();
    },
  });
  body.appendChild(wheel);

  const hint = document.createElement("div");
  hint.className = "section-hint";
  hint.textContent = "click ring to select hue, scroll to adjust range";
  body.appendChild(hint);
}


// ── Tone section ──

function buildToneSection(body: HTMLElement, dimensions: Dimension[]): void {
  const toneControls: Array<{ name: string; label: string; leftLabel?: string; rightLabel?: string }> = [
    { name: "brightness", label: "Brightness", leftLabel: "dark", rightLabel: "bright" },
    { name: "contrast", label: "Contrast", leftLabel: "flat", rightLabel: "punchy" },
    { name: "color_temperature", label: "White balance", leftLabel: "cool", rightLabel: "warm" },
  ];

  for (const tc of toneControls) {
    const dim = dimensions.find((d) => d.name === tc.name);
    if (!dim || dim.min === undefined || dim.max === undefined) continue;

    const group = document.createElement("div");
    group.className = "control-group";

    const label = document.createElement("label");
    label.textContent = tc.label;
    group.appendChild(label);

    const widget = createDiscriminateWidget({
      rangeMin: dim.min,
      rangeMax: dim.max,
      initial: { min: dim.min, max: dim.max },
      dimension: tc.name,
      onChange: (band) => {
        setRangeFilter(tc.name, band.min, band.max);
        debouncedRecompute();
      },
    });
    group.appendChild(widget);

    if (tc.leftLabel || tc.rightLabel) {
      const labels = document.createElement("div");
      labels.className = "slider-labels";
      labels.innerHTML = `<span>${tc.leftLabel || ""}</span><span>${tc.rightLabel || ""}</span>`;
      group.appendChild(labels);
    }

    body.appendChild(group);
  }
}


// ── Import section (browser dev mode fallback) ──

function buildImportSection(body: HTMLElement): void {
  const row = document.createElement("div");
  row.className = "import-row";

  const pathInput = document.createElement("input");
  pathInput.type = "text";
  pathInput.className = "import-path";
  pathInput.placeholder = "/path/to/photos...";

  const btn = document.createElement("button");
  btn.className = "import-btn";
  btn.textContent = "Import";
  btn.addEventListener("click", () => {
    const source = pathInput.value.trim();
    if (source) startImport(source).catch((e) => console.error("[import]", e));
  });

  row.appendChild(pathInput);
  row.appendChild(btn);
  body.appendChild(row);
}


// ── Main init ──

export async function initControls(dimensions: Dimension[], modelsRes: api.ModelsResponse): Promise<void> {
  cachedDimensions = dimensions;
  cachedModelsRes = modelsRes;

  const models = modelsRes.models;
  const counts = modelsRes.counts;
  const total = modelsRes.total;

  // Load vocabulary in the background — autocomplete and the taxonomy
  // browser both walk this tree when it arrives. Free-text typing works
  // either way per !taxonomy-is-scaffolding.
  if (Object.keys(vocabulary).length === 0) {
    api.getVocabularyTree()
      .then((tree) => {
        vocabulary = tree;
        notifyVocabReady();
      })
      .catch((e) => console.error("[vocabulary]", e));
  }

  // --- Left panel: Sigils (saved views) above the advanced dim sliders ---
  const slicePanel = document.getElementById("slice-panel")!;
  slicePanel.innerHTML = "";
  slicePanel.classList.toggle("folded", state.panels.leftFolded);

  const leftHeader = document.createElement("div");
  leftHeader.className = "panel-header";
  const leftCollapse = document.createElement("button");
  leftCollapse.className = "panel-collapse-btn";
  leftCollapse.textContent = "\u25C0";
  leftCollapse.title = "hide";
  leftCollapse.addEventListener("click", (e) => {
    e.stopPropagation();
    slicePanel.classList.add("folded");
    setLeftPanelFolded(true);
  });
  leftHeader.appendChild(document.createElement("span"));
  leftHeader.appendChild(leftCollapse);
  slicePanel.appendChild(leftHeader);

  const leftUnfoldTab = document.createElement("div");
  leftUnfoldTab.className = "panel-unfold-tab left";
  leftUnfoldTab.textContent = "\u25B6";
  leftUnfoldTab.addEventListener("click", (e) => {
    e.stopPropagation();
    slicePanel.classList.remove("folded");
    setLeftPanelFolded(false);
  });
  slicePanel.parentElement!.insertBefore(leftUnfoldTab, slicePanel.nextSibling);

  const collages = createSection("Sigils");
  buildCollagesSection(collages.body);
  slicePanel.appendChild(collages.section);
  let lastSigils = state.workspaceSigils;
  subscribe((s) => {
    if (s.workspaceSigils !== lastSigils) {
      lastSigils = s.workspaceSigils;
      renderCollages?.();
    }
  });


  // --- Right panel: Lightroom-style sections ---
  const panel = document.getElementById("neighborhood-panel")!;
  panel.innerHTML = "";
  panel.classList.toggle("folded", state.panels.rightFolded);
  // Restore user-dragged width. Null means "never resized" → use the CSS
  // default by leaving inline styles unset.
  if (state.panels.rightWidth !== null) {
    panel.style.width = `${state.panels.rightWidth}px`;
    panel.style.minWidth = `${state.panels.rightWidth}px`;
  }

  // Gutter: resize by dragging, separate from the panel
  const gutter = document.getElementById("panel-gutter")!;
  gutter.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    gutter.setPointerCapture(e.pointerId);
    gutter.classList.add("dragging");
    const startX = e.clientX;
    const startWidth = panel.offsetWidth;
    let lastWidth = startWidth;

    const onMove = (ev: PointerEvent) => {
      const delta = startX - ev.clientX;
      const newWidth = Math.max(180, Math.min(400, startWidth + delta));
      panel.style.width = `${newWidth}px`;
      panel.style.minWidth = `${newWidth}px`;
      lastWidth = newWidth;
    };

    const onUp = () => {
      gutter.classList.remove("dragging");
      gutter.removeEventListener("pointermove", onMove);
      gutter.removeEventListener("pointerup", onUp);
      gutter.removeEventListener("lostpointercapture", onUp);
      setRightPanelWidth(lastWidth);
    };

    gutter.addEventListener("pointermove", onMove);
    gutter.addEventListener("pointerup", onUp);
    gutter.addEventListener("lostpointercapture", onUp);
  });

  // Unfold tab — sibling after the panel so it isn't clipped by overflow:hidden
  const unfoldTab = document.createElement("div");
  unfoldTab.className = "panel-unfold-tab";
  unfoldTab.textContent = "\u25C0";
  unfoldTab.addEventListener("click", (e) => {
    e.stopPropagation();
    panel.classList.remove("folded");
    setRightPanelFolded(false);
  });
  panel.parentElement!.appendChild(unfoldTab);

  // Panel header with collapse
  const panelHeader = document.createElement("div");
  panelHeader.className = "panel-header";
  const panelTitle = document.createElement("span");
  panelTitle.textContent = "Sigil Controls";
  panelTitle.className = "panel-title";
  const collapseBtn = document.createElement("button");
  collapseBtn.className = "panel-collapse-btn";
  collapseBtn.textContent = "\u25B6";
  collapseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    panel.style.width = "";
    panel.style.minWidth = "";
    panel.classList.add("folded");
    setRightPanelFolded(true);
  });
  panelHeader.appendChild(panelTitle);
  panelHeader.appendChild(collapseBtn);
  panel.appendChild(panelHeader);

  // Mode tabs
  const modeTabs = document.createElement("div");
  modeTabs.className = "mode-tabs";
  let timeSection: HTMLElement;
  const modes: Array<{ value: "spacelike" | "timelike"; label: string }> = [
    { value: "spacelike", label: "Spacelike" },
    { value: "timelike", label: "Timelike" },
  ];
  for (const m of modes) {
    const tab = document.createElement("button");
    tab.className = "mode-tab" + (m.value === state.mode ? " active" : "");
    tab.textContent = m.label;
    tab.addEventListener("click", () => {
      state.mode = m.value;
      modeTabs.querySelectorAll(".mode-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      if (timeSection) timeSection.style.display = state.mode === "timelike" ? "" : "none";
      recomputeSliceAndLayout().catch((e) => console.error("Mode switch failed:", e));
    });
    modeTabs.appendChild(tab);
  }
  panel.appendChild(modeTabs);

  // Layers section — three independent toggles per spec
  const layers = createSection("Layers");
  buildLayersSection(layers.body);
  panel.appendChild(layers.section);

  // ThingsLibrary — sits ABOVE Attract per spec; persists across sessions.
  // Fold state flows through the general createSection → state.panels mechanism.
  const library = createSection("Things");
  buildThingsLibrarySection(library.section, library.body);
  panel.appendChild(library.section);

  // Taxonomy browser — the sigil's tree, clickable. Picking a node turns
  // it into a @Thing @Attractor (same codepath as typing it in Attract).
  // Collapsed by default — most sessions don't need it, but when you want
  // to drop a rich taxonomy like Animalia as an attractor, this is the
  // affordance: browse the tree, click the node, watch the field re-form.
  const taxonomy = createSection("Taxonomy", true);
  buildTaxonomySection(taxonomy.body);
  panel.appendChild(taxonomy.section);

  // Attract section
  const attract = createSection("Attract");
  buildAttractSection(attract.body);
  panel.appendChild(attract.section);

  // Contrasts section
  const contrasts = createSection("Contrasts");
  buildContrastsSection(contrasts.body);
  panel.appendChild(contrasts.section);

  // Color section — only shown if hue dimension exists
  const color = createSection("Color");
  buildColorSection(color.body, dimensions);
  if (color.body.childElementCount > 0) panel.appendChild(color.section);

  // Tone section — only shown if tone dimensions exist
  const tone = createSection("Tone");
  buildToneSection(tone.body, dimensions);
  if (tone.body.childElementCount > 0) panel.appendChild(tone.section);

  // Settings section (collapsed by default)
  const settings = createSection("Settings", true);

  // Gravity softness — softmax temperature over attractor pulls in SpaceLike.
  // Distinct from Relevance (slice membrane): this one shapes how sharply
  // neighborhoods compete for cells inside the already-gated field.
  const featherGroup = document.createElement("div");
  featherGroup.className = "control-group";
  const featherLabel = document.createElement("label");
  featherLabel.textContent = "Gravity softness";
  featherGroup.appendChild(featherLabel);

  const featherSlider = document.createElement("input");
  featherSlider.type = "range";
  featherSlider.className = "thin-slider";
  featherSlider.min = "0";
  featherSlider.max = "1";
  featherSlider.step = "0.05";
  featherSlider.value = String(state.feathering);

  const featherLabels = document.createElement("div");
  featherLabels.className = "slider-labels";
  featherLabels.innerHTML = "<span>hard</span><span>soft</span>";

  featherSlider.addEventListener("change", () => {
    state.feathering = parseFloat(featherSlider.value);
    recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
  });
  featherGroup.appendChild(featherSlider);
  featherGroup.appendChild(featherLabels);
  settings.body.appendChild(featherGroup);

  // Field expansion — controls how the SpaceLike grid relates to the slice.
  // Echo (default) cycles surplus cells, creating moiré on small slices;
  // Tight drops the least-similar overflow so every cell is unique.
  const fieldGroup = document.createElement("div");
  fieldGroup.className = "control-group";
  const fieldLabel = document.createElement("label");
  fieldLabel.textContent = "Field";
  fieldGroup.appendChild(fieldLabel);
  const fieldSelect = document.createElement("select");
  for (const opt of [
    { value: "echo", label: "Echo (cycle to fill)" },
    { value: "tight", label: "Tight (no repeats)" },
  ]) {
    const o = document.createElement("option");
    o.value = opt.value;
    o.textContent = opt.label;
    fieldSelect.appendChild(o);
  }
  fieldSelect.value = state.fieldExpansion;
  fieldSelect.addEventListener("change", () => {
    state.fieldExpansion = fieldSelect.value as "echo" | "tight";
    recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
  });
  fieldGroup.appendChild(fieldSelect);
  settings.body.appendChild(fieldGroup);

  // Arrangement — how a single attractor lays out its neighbourhood.
  // Rings produces sharp similarity tiers; Field is a continuous deformation
  // of UMAP that preserves local mutual proximity.
  const arrGroup = document.createElement("div");
  arrGroup.className = "control-group";
  const arrLabel = document.createElement("label");
  arrLabel.textContent = "Arrangement";
  arrGroup.appendChild(arrLabel);
  const arrSelect = document.createElement("select");
  for (const opt of [
    { value: "rings", label: "Rings (radial shells)" },
    { value: "field", label: "Field (continuous)" },
    { value: "axis",  label: "Axis (similar \u2194 opposite)" },
  ]) {
    const o = document.createElement("option");
    o.value = opt.value;
    o.textContent = opt.label;
    arrSelect.appendChild(o);
  }
  arrSelect.value = state.arrangement;
  arrSelect.addEventListener("change", () => {
    state.arrangement = arrSelect.value as "rings" | "field" | "axis";
    recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
  });
  arrGroup.appendChild(arrSelect);
  settings.body.appendChild(arrGroup);

  // Time direction
  const timeDirGroup = document.createElement("div");
  timeDirGroup.className = "control-group";
  const timeDirLabel = document.createElement("label");
  timeDirLabel.textContent = "Time direction";
  timeDirGroup.appendChild(timeDirLabel);

  const timeSelect = document.createElement("select");
  const captureOpt = document.createElement("option");
  captureOpt.value = "capture_date";
  captureOpt.textContent = "Capture date";
  timeSelect.appendChild(captureOpt);
  timeSelect.value = state.timeDirection;
  timeSelect.addEventListener("change", () => {
    state.timeDirection = timeSelect.value;
    recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
  });
  timeDirGroup.appendChild(timeSelect);
  timeDirGroup.style.display = state.mode === "timelike" ? "" : "none";
  settings.body.appendChild(timeDirGroup);
  timeSection = timeDirGroup;

  // Model — advanced: which embedding space the attractors and contrasts
  // resolve in. Similarity is defined by named Contrasts; this picks the
  // raw space those Contrasts project out of.
  const modelGroup = document.createElement("div");
  modelGroup.className = "control-group";
  const modelLabel = document.createElement("label");
  modelLabel.textContent = "Embedding space";
  modelGroup.appendChild(modelLabel);
  const modelSelect = document.createElement("select");
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m;
    const coverage = counts[m] ?? 0;
    const complete = coverage >= total;
    const base = m === "clip-vit-b-32" ? "CLIP B-32 (semantic)"
      : m === "clip-vit-l-14" ? "CLIP L-14 (semantic, fine)"
      : m === "dinov2-vitb14" ? "DINOv2 (visual form)"
      : m;
    if (!complete) {
      const pct = Math.round(100 * coverage / Math.max(1, total));
      opt.textContent = `${base} — ${pct}% embedded`;
      opt.disabled = true;
    } else {
      opt.textContent = base;
    }
    modelSelect.appendChild(opt);
  }
  modelSelect.value = state.model;
  modelSelect.addEventListener("change", () => {
    state.model = modelSelect.value;
    recomputeSliceAndLayout().catch((e) => console.error("Model switch failed:", e));
  });
  modelGroup.appendChild(modelSelect);
  settings.body.appendChild(modelGroup);

  // Image count
  const countEl = document.createElement("div");
  countEl.id = "image-count";
  countEl.textContent = `${state.imageIds.length} images`;
  settings.body.appendChild(countEl);

  // Import (browser dev mode fallback; primary path is File > Import...)
  const importGroup = document.createElement("div");
  importGroup.className = "control-group";
  buildImportSection(importGroup);
  settings.body.appendChild(importGroup);

  panel.appendChild(settings.section);
}
