/**
 * UI controls — SigilControls per spec.
 *
 * SigilControls = the right panel. A sigil builder.
 * The active set of ContrastControls IS the query. The slice is the result.
 *
 * Things = proximity filters (attract role). Name a concept, bias toward it.
 * Contrasts = two-pole bandpass filters. Define a tension, set the range.
 * Both feed into slice -> layout pipeline.
 */

import { state, notify } from "../state";
import * as api from "../api";
import type { Dimension, ContrastControl } from "../types";
import type { TorusViewport } from "../renderer/torus-viewport";
import { createBandpassWidget } from "./bandpass-widget";

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

async function recomputeSliceAndLayout(): Promise<void> {
  // Slice: apply all controls
  const res = await api.computeSlice({
    range_filters: state.rangeFilters,
    proximity_filters: state.proximityFilters,
    contrast_controls: state.contrastControls,
    model: state.model,
  });
  state.imageIds = res.image_ids;
  state.orderValues = res.order_values || {};

  // Layout: mode determines how images arrange on the torus
  let orderValues: Record<string, number> | undefined;
  let preserveOrder = false;

  if (state.mode === "timelike") {
    // Timelike: order by capture date or contrast projection
    const hasOV = Object.keys(state.orderValues).length > 0;
    orderValues = hasOV ? state.orderValues : undefined;
  } else if (state.mode === "tastelike") {
    // Tastelike: preserve score-based order from slice
    preserveOrder = state.proximityFilters.length > 0 ||
      state.contrastControls.some((c) => c.role === "attract");
  }
  // Spacelike: no order_values, no preserve_order — UMAP + Hilbert

  const layout = await api.computeLayout({
    image_ids: state.imageIds,
    axes: state.selectedAxes.length > 0 ? state.selectedAxes : null,
    tightness: state.tightness,
    model: state.model,
    strip_height: state.stripHeight,
    preserve_order: preserveOrder,
    order_values: orderValues,
  });
  state.layout = layout;
  state.torusWidth = layout.torus_width;
  state.torusHeight = layout.torus_height;

  // Recenter camera only if surface size changed significantly
  const prevArea = state.torusWidth * state.torusHeight;
  const newArea = layout.torus_width * layout.torus_height;
  if (prevArea === 0 || Math.abs(newArea - prevArea) / prevArea > 0.5) {
    const canvas = document.getElementById("viewport") as HTMLCanvasElement;
    const aspect = canvas.clientWidth / canvas.clientHeight;
    const maxZoom = Math.min(layout.torus_width, layout.torus_height * aspect);
    const visibleStrips = 8;
    const desiredZoom = visibleStrips * layout.strip_height * aspect;
    state.pov.x = layout.torus_width / 2;
    state.pov.y = layout.torus_height / 2;
    state.pov.z = Math.min(desiredZoom, maxZoom);
  }

  if (viewport) viewport.setLayout(layout);
  updateImageCount();
  notify();
}

function updateImageCount(): void {
  const el = document.getElementById("image-count");
  if (el) el.textContent = `${state.imageIds.length} images`;
}

function makeFoldable(panel: HTMLElement): void {
  const h3 = panel.querySelector("h3");
  if (h3) {
    h3.addEventListener("click", () => panel.classList.toggle("folded"));
  }
}


// ── Things (proximity filters) ──

const thingNames: string[] = [];

function createThingsHolder(
  container: HTMLElement,
  onChange: () => void,
): void {
  const holder = document.createElement("div");
  holder.className = "pill-holder";

  const renderPills = () => {
    holder.querySelectorAll(".pill").forEach((el) => el.remove());
    for (let i = 0; i < thingNames.length; i++) {
      const name = thingNames[i];
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = name;
      const remove = document.createElement("span");
      remove.className = "pill-remove";
      remove.textContent = "\u00d7";
      const idx = i;
      remove.addEventListener("click", () => {
        state.proximityFilters.splice(idx, 1);
        thingNames.splice(idx, 1);
        renderPills();
        onChange();
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
      state.proximityFilters.push({ text: `a photograph of ${text}`, weight: 1.0 });
      thingNames.push(text);
      input.value = "";
      renderPills();
      onChange();
    }
  });

  holder.appendChild(input);
  container.appendChild(holder);
}


// ── Contrast (two-pole bandpass) ──

function createContrastBuilder(
  container: HTMLElement,
  onChange: () => void,
): void {
  const list = document.createElement("div");
  list.className = "contrast-list";

  const renderContrasts = () => {
    list.innerHTML = "";
    for (let i = 0; i < state.contrastControls.length; i++) {
      const cc = state.contrastControls[i];
      list.appendChild(createContrastWidget(cc, i, onChange, renderContrasts));
    }
  };

  // Add contrast: two text inputs with "vs" between them
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

  container.appendChild(list);
  container.appendChild(addBtn);
}


function createContrastWidget(
  cc: ContrastControl,
  index: number,
  onChange: () => void,
  rerender: () => void,
): HTMLElement {
  const widget = document.createElement("div");
  widget.className = "contrast-widget";

  // Header: pole_a vs pole_b [x]
  const stripPromptPrefix = (s: string) => s.replace(/^a photograph (?:of |that is )/, "");
  const header = document.createElement("div");
  header.className = "contrast-header";
  header.innerHTML = `<span>${stripPromptPrefix(cc.pole_a)}</span><span class="contrast-vs">vs</span><span>${stripPromptPrefix(cc.pole_b)}</span>`;
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

  // Bandpass widget
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
  labels.innerHTML = `<span>${stripPromptPrefix(cc.pole_a)}</span><span>${stripPromptPrefix(cc.pole_b)}</span>`;

  widget.appendChild(bandWidget);
  widget.appendChild(labels);

  return widget;
}


// ── Main init ──

export async function initControls(dimensions: Dimension[], models: string[]): Promise<void> {

  // --- Slice panel (left) ---
  const slicePanel = document.getElementById("slice-panel")!;
  slicePanel.innerHTML = "<h3>Slice</h3>";

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
        const existing = state.rangeFilters.findIndex((f) => f.dimension === dim.name);
        const filter = { dimension: dim.name, min: band.min, max: band.max };
        if (existing >= 0) state.rangeFilters[existing] = filter;
        else state.rangeFilters.push(filter);
        debouncedRecompute();
      },
    });
    container.appendChild(widget);
    slicePanel.appendChild(container);
  }

  // --- SigilControls panel (right) ---
  const nbPanel = document.getElementById("neighborhood-panel")!;
  nbPanel.innerHTML = "<h3>Sigil Controls</h3>";

  // Mode tabs
  const modeTabs = document.createElement("div");
  modeTabs.className = "mode-tabs";
  const modes: Array<{ value: "spacelike" | "timelike" | "tastelike"; label: string }> = [
    { value: "spacelike", label: "Spacelike" },
    { value: "timelike", label: "Timelike" },
    { value: "tastelike", label: "Tastelike" },
  ];
  for (const m of modes) {
    const tab = document.createElement("button");
    tab.className = "mode-tab" + (m.value === state.mode ? " active" : "");
    tab.textContent = m.label;
    tab.addEventListener("click", () => {
      state.mode = m.value;
      modeTabs.querySelectorAll(".mode-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      timeGroup.style.display = state.mode === "timelike" ? "" : "none";
      recomputeSliceAndLayout().catch((e) => console.error("Mode switch failed:", e));
    });
    modeTabs.appendChild(tab);
  }
  nbPanel.appendChild(modeTabs);

  // Things (proximity attract)
  const thingsGroup = document.createElement("div");
  thingsGroup.className = "control-group";
  const thingsLabel = document.createElement("label");
  thingsLabel.textContent = "Attract";
  thingsGroup.appendChild(thingsLabel);
  createThingsHolder(thingsGroup, () => {
    recomputeSliceAndLayout().catch((e) => console.error("Slice failed:", e));
  });
  nbPanel.appendChild(thingsGroup);

  // Contrasts
  const contrastGroup = document.createElement("div");
  contrastGroup.className = "control-group";
  const contrastLabel = document.createElement("label");
  contrastLabel.textContent = "Contrasts";
  contrastGroup.appendChild(contrastLabel);
  createContrastBuilder(contrastGroup, () => {
    recomputeSliceAndLayout().catch((e) => console.error("Slice failed:", e));
  });
  nbPanel.appendChild(contrastGroup);

  // Time direction
  const timeGroup = document.createElement("div");
  timeGroup.className = "control-group";
  const timeLabel = document.createElement("label");
  timeLabel.textContent = "Time direction";
  timeGroup.appendChild(timeLabel);

  const timeSelect = document.createElement("select");
  const captureOpt = document.createElement("option");
  captureOpt.value = "capture_date";
  captureOpt.textContent = "Capture date";
  timeSelect.appendChild(captureOpt);
  // Order-role contrasts will be added dynamically as they're created
  timeSelect.value = state.timeDirection;
  timeSelect.addEventListener("change", () => {
    state.timeDirection = timeSelect.value as "capture_date";
    recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
  });
  timeGroup.appendChild(timeSelect);
  timeGroup.style.display = state.mode === "timelike" ? "" : "none";
  nbPanel.appendChild(timeGroup);

  // Tightness slider
  const tightnessGroup = document.createElement("div");
  tightnessGroup.className = "control-group";
  const tightnessLabel = document.createElement("label");
  tightnessLabel.textContent = "Tightness";
  tightnessGroup.appendChild(tightnessLabel);

  const tightnessSlider = document.createElement("input");
  tightnessSlider.type = "range";
  tightnessSlider.min = "0";
  tightnessSlider.max = "1";
  tightnessSlider.step = "0.05";
  tightnessSlider.value = String(state.tightness);

  const tightnessLabels = document.createElement("div");
  tightnessLabels.className = "slider-labels";
  tightnessLabels.innerHTML = "<span>tight</span><span>loose</span>";

  tightnessSlider.addEventListener("change", () => {
    state.tightness = parseFloat(tightnessSlider.value);
    recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
  });
  tightnessGroup.appendChild(tightnessSlider);
  tightnessGroup.appendChild(tightnessLabels);
  nbPanel.appendChild(tightnessGroup);

  // Model dropdown
  const modelGroup = document.createElement("div");
  modelGroup.className = "control-group";
  const modelLabel = document.createElement("label");
  modelLabel.textContent = "Model";
  modelGroup.appendChild(modelLabel);

  const modelSelect = document.createElement("select");
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m === "clip-vit-l-14" ? "CLIP (semantic)" : m === "dinov2-vitb14" ? "DINOv2 (texture)" : m;
    modelSelect.appendChild(opt);
  }
  modelSelect.value = state.model;
  modelSelect.addEventListener("change", () => {
    state.model = modelSelect.value;
    recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
  });
  modelGroup.appendChild(modelSelect);
  nbPanel.appendChild(modelGroup);

  // Image count
  const countEl = document.createElement("div");
  countEl.id = "image-count";
  countEl.textContent = `${state.imageIds.length} images`;
  nbPanel.appendChild(countEl);

  makeFoldable(slicePanel);
  makeFoldable(nbPanel);
}
