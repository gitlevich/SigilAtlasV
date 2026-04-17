/**
 * UI controls — SigilControls per spec.
 *
 * Right panel with Lightroom-style collapsible sections:
 * Mode, Attract, Contrasts, Color, Tone, Settings.
 */

import { state, notify } from "../state";
import * as api from "../api";
import type { AnyLayout, Dimension, ContrastControl, PointOfView } from "../types";
import { isSpaceLikeLayout } from "../types";
import type { TorusViewport } from "../renderer/torus-viewport";
import { createBandpassWidget } from "./bandpass-widget";
import { createColorWheel, hueRangeToFilter, type HueRange } from "./color-wheel";
import { startImport } from "../import";

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

/** Re-fetch dimensions and models from the server and rebuild the control panels. */
export async function refreshControls(): Promise<void> {
  const [dimensions, models] = await Promise.all([
    api.getDimensions(),
    api.getModels(),
  ]);
  if (models.length > 0 && !models.includes(state.model)) {
    state.model = models[0];
  }
  await initControls(dimensions, models);
  notify();
}

export async function recomputeSliceAndLayout(): Promise<void> {
  // Cancel any pending debounced recompute to avoid stale overwrites
  if (sliceDebounceTimer) { clearTimeout(sliceDebounceTimer); sliceDebounceTimer = null; }

  try {
    const res = await api.computeSlice({
      range_filters: state.rangeFilters,
      proximity_filters: state.proximityFilters,
      contrast_controls: state.contrastControls,
      model: state.model,
      feathering: state.feathering,
    });
    state.imageIds = res.image_ids;
    state.orderValues = res.order_values || {};

    // Anchor: the image at screen centre in the current layout. We'll try to
    // keep this same image at screen centre after the recompute, regardless
    // of mode switch or slice change.
    const prevLayout = state.layout;
    const anchorImageId = prevLayout ? imageAtPov(prevLayout, state.pov) : null;
    const isFirstLoad = state.torusWidth === 0;

    const canvas = document.getElementById("viewport") as HTMLCanvasElement;
    const aspect = canvas.clientWidth / canvas.clientHeight;

    let newLayout: AnyLayout;
    let attractorCell: { col: number; row: number; cell_size: number } | null = null;

    if (state.mode === "spacelike") {
      const layout = await api.computeSpacelike({
        image_ids: state.imageIds,
        attractors: state.attractors,
        model: state.model,
        feathering: state.feathering,
        cell_size: state.cellSize,
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
      // New attractor with no prior anchor: zoom in on it.
      const visibleCells = 16;
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
        const visibleCells = 16;
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
      state.pov.x = newLayout.torus_width / 2;
      state.pov.y = newLayout.torus_height / 2;
      state.pov.z = maxZoom;
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

function imageAtPov(layout: AnyLayout, pov: PointOfView): string | null {
  if (isSpaceLikeLayout(layout)) {
    if (layout.cols === 0 || layout.rows === 0) return null;
    // Wrap pov into the torus range, then find the cell it falls in.
    const wx = ((pov.x % layout.torus_width) + layout.torus_width) % layout.torus_width;
    const wy = ((pov.y % layout.torus_height) + layout.torus_height) % layout.torus_height;
    const col = Math.min(layout.cols - 1, Math.max(0, Math.floor(wx / layout.cell_size)));
    const row = Math.min(layout.rows - 1, Math.max(0, Math.floor(wy / layout.cell_size)));
    for (const p of layout.positions) {
      if (p.col === col && p.row === row) return p.id;
    }
    return null;
  }
  // Strip layout.
  const tw = layout.torus_width, th = layout.torus_height;
  if (tw === 0 || th === 0) return null;
  const wx = ((pov.x % tw) + tw) % tw;
  const wy = ((pov.y % th) + th) % th;
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
  const section = document.createElement("div");
  section.className = "section" + (collapsed ? " collapsed" : "");

  const header = document.createElement("div");
  header.className = "section-header";
  header.textContent = title;
  header.addEventListener("click", () => section.classList.toggle("collapsed"));
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


// ── Layers (Photos / Wireframe / Relief) ──

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


// ── Attractors (things that bend the SpaceLike gravity field) ──

function buildAttractSection(body: HTMLElement): void {
  const holder = document.createElement("div");
  holder.className = "pill-holder";

  const renderPills = () => {
    holder.querySelectorAll(".pill").forEach((el) => el.remove());
    for (let i = 0; i < state.attractors.length; i++) {
      const att = state.attractors[i];
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = att.ref;
      const remove = document.createElement("span");
      remove.className = "pill-remove";
      remove.textContent = "\u00d7";
      const idx = i;
      remove.addEventListener("click", () => {
        state.attractors.splice(idx, 1);
        renderPills();
        recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
      });
      pill.appendChild(remove);
      holder.insertBefore(pill, holder.lastElementChild);
    }
  };

  const input = document.createElement("input");
  input.type = "text";
  input.className = "pill-input";
  input.placeholder = "type a thing, press enter...";
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      state.attractors.push({ kind: "thing", ref: text });
      input.value = "";
      renderPills();
      recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
    }
  });

  holder.appendChild(input);
  body.appendChild(holder);
  renderPills();
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
      role: "filter",
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

  const bandWidget = createBandpassWidget({
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

    const widget = createBandpassWidget({
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

export async function initControls(dimensions: Dimension[], models: string[]): Promise<void> {

  // --- Left panel: start folded, keep dimension sliders for advanced use ---
  const slicePanel = document.getElementById("slice-panel")!;
  slicePanel.innerHTML = "<h3>Slice</h3>";
  slicePanel.classList.add("folded");

  for (const dim of dimensions) {
    if (dim.type !== "range" || dim.min === undefined || dim.max === undefined) continue;
    const container = document.createElement("div");
    container.className = "control-group";
    const label = document.createElement("label");
    label.textContent = dim.name;
    container.appendChild(label);
    const widget = createBandpassWidget({
      rangeMin: dim.min,
      rangeMax: dim.max,
      initial: { min: dim.min, max: dim.max },
      onChange: (band) => {
        setRangeFilter(dim.name, band.min, band.max);
        debouncedRecompute();
      },
    });
    container.appendChild(widget);
    slicePanel.appendChild(container);
  }

  // --- Right panel: Lightroom-style sections ---
  const panel = document.getElementById("neighborhood-panel")!;
  panel.innerHTML = "";

  // Gutter: resize by dragging, separate from the panel
  const gutter = document.getElementById("panel-gutter")!;
  gutter.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    gutter.setPointerCapture(e.pointerId);
    gutter.classList.add("dragging");
    const startX = e.clientX;
    const startWidth = panel.offsetWidth;

    const onMove = (ev: PointerEvent) => {
      const delta = startX - ev.clientX;
      const newWidth = Math.max(180, Math.min(400, startWidth + delta));
      panel.style.width = `${newWidth}px`;
      panel.style.minWidth = `${newWidth}px`;
    };

    const onUp = () => {
      gutter.classList.remove("dragging");
      gutter.removeEventListener("pointermove", onMove);
      gutter.removeEventListener("pointerup", onUp);
      gutter.removeEventListener("lostpointercapture", onUp);
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

  // Feathering — soft vs hard edge of attractor neighborhoods
  const featherGroup = document.createElement("div");
  featherGroup.className = "control-group";
  const featherLabel = document.createElement("label");
  featherLabel.textContent = "Feather";
  featherGroup.appendChild(featherLabel);

  const featherSlider = document.createElement("input");
  featherSlider.type = "range";
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

  // Model
  const modelGroup = document.createElement("div");
  modelGroup.className = "control-group";
  const modelLabel = document.createElement("label");
  modelLabel.textContent = "Model";
  modelGroup.appendChild(modelLabel);

  const modelSelect = document.createElement("select");
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m === "clip-vit-b-32" ? "Semantic" : m === "clip-vit-l-14" ? "Semantic XR" : m === "dinov2-vitb14" ? "Visual" : m;
    modelSelect.appendChild(opt);
  }
  modelSelect.value = state.model;
  modelSelect.addEventListener("change", () => {
    state.model = modelSelect.value;
    recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
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
