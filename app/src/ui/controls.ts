/**
 * UI controls — SigilControls per spec.
 *
 * Right panel with Lightroom-style collapsible sections:
 * Mode, Attract, Contrasts, Color, Tone, Settings.
 */

import { state, notify } from "../state";
import * as api from "../api";
import type { Dimension, ContrastControl } from "../types";
import type { TorusViewport } from "../renderer/torus-viewport";
import { createBandpassWidget } from "./bandpass-widget";
import { createColorWheel, hueRangeToFilter, type HueRange } from "./color-wheel";

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
  // Cancel any pending debounced recompute to avoid stale overwrites
  if (sliceDebounceTimer) { clearTimeout(sliceDebounceTimer); sliceDebounceTimer = null; }
  console.log("[recompute] proximity:", state.proximityFilters.length, "contrasts:", state.contrastControls.length, "range:", state.rangeFilters.length);
  const res = await api.computeSlice({
    range_filters: state.rangeFilters,
    proximity_filters: state.proximityFilters,
    contrast_controls: state.contrastControls,
    model: state.model,
    tightness: state.tightness,
  });
  state.imageIds = res.image_ids;
  state.orderValues = res.order_values || {};

  let orderValues: Record<string, number> | undefined;
  let preserveOrder = false;

  if (state.mode === "timelike") {
    const hasOV = Object.keys(state.orderValues).length > 0;
    orderValues = hasOV ? state.orderValues : undefined;
  } else if (state.mode === "tastelike" || state.proximityFilters.length > 0) {
    preserveOrder = true;
  }

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

  const prevArea = state.torusWidth * state.torusHeight;
  const newArea = layout.torus_width * layout.torus_height;
  state.torusWidth = layout.torus_width;
  state.torusHeight = layout.torus_height;

  const canvas = document.getElementById("viewport") as HTMLCanvasElement;
  const aspect = canvas.clientWidth / canvas.clientHeight;
  const maxZoom = Math.min(layout.torus_width, layout.torus_height * aspect);

  if (prevArea === 0 || Math.abs(newArea - prevArea) / prevArea > 0.3) {
    const visibleStrips = 8;
    const desiredZoom = visibleStrips * layout.strip_height * aspect;
    state.pov.x = layout.torus_width / 2;
    state.pov.y = layout.torus_height / 2;
    state.pov.z = Math.min(desiredZoom, maxZoom);
  } else {
    state.pov.z = Math.min(state.pov.z, maxZoom);
  }

  if (viewport) viewport.setLayout(layout);
  updateImageCount();
  notify();
}

function updateImageCount(): void {
  const el = document.getElementById("image-count");
  if (el) el.textContent = `${state.imageIds.length} images`;
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


// ── Things (proximity filters) ──

const thingNames: string[] = [];

function buildAttractSection(body: HTMLElement): void {
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
        recomputeSliceAndLayout().catch((e) => console.error("Slice failed:", e));
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
      recomputeSliceAndLayout().catch((e) => console.error("Slice failed:", e));
    }
  });

  holder.appendChild(input);
  body.appendChild(holder);
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

  // Resize handle
  const resizeHandle = document.createElement("div");
  resizeHandle.className = "panel-resize-handle";
  panel.appendChild(resizeHandle);

  let resizing = false;
  resizeHandle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    resizeHandle.setPointerCapture(e.pointerId);
    resizing = true;
    resizeHandle.classList.add("dragging");
    const startX = e.clientX;
    const startWidth = panel.offsetWidth;

    const onMove = (ev: PointerEvent) => {
      const delta = startX - ev.clientX;
      const newWidth = Math.max(180, Math.min(400, startWidth + delta));
      panel.style.width = `${newWidth}px`;
      panel.style.minWidth = `${newWidth}px`;
    };

    const onUp = () => {
      resizing = false;
      resizeHandle.classList.remove("dragging");
      resizeHandle.removeEventListener("pointermove", onMove);
      resizeHandle.removeEventListener("pointerup", onUp);
      resizeHandle.removeEventListener("lostpointercapture", onUp);
    };

    resizeHandle.addEventListener("pointermove", onMove);
    resizeHandle.addEventListener("pointerup", onUp);
    resizeHandle.addEventListener("lostpointercapture", onUp);
  });

  // Mode tabs (not in a section — always visible at top)
  const modeTabs = document.createElement("div");
  modeTabs.className = "mode-tabs";
  let timeSection: HTMLElement;
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
      if (timeSection) timeSection.style.display = state.mode === "timelike" ? "" : "none";
      recomputeSliceAndLayout().catch((e) => console.error("Mode switch failed:", e));
    });
    modeTabs.appendChild(tab);
  }
  panel.appendChild(modeTabs);

  // Attract section
  const attract = createSection("Attract");
  buildAttractSection(attract.body);
  panel.appendChild(attract.section);

  // Contrasts section
  const contrasts = createSection("Contrasts");
  buildContrastsSection(contrasts.body);
  panel.appendChild(contrasts.section);

  // Color section
  const color = createSection("Color");
  buildColorSection(color.body, dimensions);
  panel.appendChild(color.section);

  // Tone section
  const tone = createSection("Tone");
  buildToneSection(tone.body, dimensions);
  panel.appendChild(tone.section);

  // Settings section (collapsed by default)
  const settings = createSection("Settings", true);

  // Tightness
  const tightnessGroup = document.createElement("div");
  tightnessGroup.className = "control-group";
  const tightnessLabel = document.createElement("label");
  tightnessLabel.textContent = "Feather";
  tightnessGroup.appendChild(tightnessLabel);

  const tightnessSlider = document.createElement("input");
  tightnessSlider.type = "range";
  tightnessSlider.min = "0";
  tightnessSlider.max = "1";
  tightnessSlider.step = "0.05";
  tightnessSlider.value = String(state.tightness);

  const tightnessLabels = document.createElement("div");
  tightnessLabels.className = "slider-labels";
  tightnessLabels.innerHTML = "<span>strict</span><span>permissive</span>";

  tightnessSlider.addEventListener("change", () => {
    state.tightness = parseFloat(tightnessSlider.value);
    recomputeSliceAndLayout().catch((e) => console.error("Layout failed:", e));
  });
  tightnessGroup.appendChild(tightnessSlider);
  tightnessGroup.appendChild(tightnessLabels);
  settings.body.appendChild(tightnessGroup);

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
    opt.textContent = m === "clip-vit-b-32" ? "Semantic" : m === "clip-vit-l-14" ? "Semantic LG" : m === "dinov2-vitb14" ? "Visual" : m;
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

  panel.appendChild(settings.section);
}
