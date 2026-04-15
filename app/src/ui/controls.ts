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
import type { Dimension, ContrastControl, VocabTerm } from "../types";
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
let allVocab: VocabTerm[] = [];

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

  // Layout: arrange the slice on the torus
  // Pass order_values so layout sorts by time or contrast projection
  const hasScoring = state.proximityFilters.length > 0 ||
    state.contrastControls.some((c) => c.role === "attract" || c.role === "order");
  const hasOrderValues = Object.keys(state.orderValues).length > 0;
  const layout = await api.computeLayout({
    image_ids: state.imageIds,
    axes: state.selectedAxes.length > 0 ? state.selectedAxes : null,
    tightness: state.tightness,
    model: state.model,
    strip_height: state.stripHeight,
    preserve_order: hasScoring && !hasOrderValues,
    order_values: hasOrderValues ? state.orderValues : undefined,
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


// ── Autocomplete ──

interface AutocompleteResult {
  name: string;
  path: string;
  prompt: string;
}

function createAutocomplete(
  container: HTMLElement,
  candidates: () => AutocompleteResult[],
  onSelect: (term: AutocompleteResult) => void,
  placeholder = "name a thing...",
): HTMLInputElement {
  const wrapper = document.createElement("div");
  wrapper.className = "autocomplete-wrapper";

  const input = document.createElement("input");
  input.type = "text";
  input.className = "pill-input";
  input.placeholder = placeholder;

  const dropdown = document.createElement("div");
  dropdown.className = "autocomplete-dropdown";
  dropdown.style.display = "none";

  let matches: AutocompleteResult[] = [];
  let activeIdx = -1;

  const highlightItem = (idx: number) => {
    const items = dropdown.querySelectorAll(".autocomplete-item");
    items.forEach((el, i) => {
      (el as HTMLElement).classList.toggle("active", i === idx);
    });
    activeIdx = idx;
  };

  const selectMatch = (term: AutocompleteResult) => {
    input.value = "";
    dropdown.style.display = "none";
    activeIdx = -1;
    onSelect(term);
  };

  const updateDropdown = () => {
    const query = input.value.trim().toLowerCase();
    if (!query) {
      dropdown.style.display = "none";
      matches = [];
      activeIdx = -1;
      return;
    }
    const all = candidates().filter((t) => t.name.toLowerCase().includes(query) || t.path.toLowerCase().includes(query));
    const startsWith = all.filter((t) => t.name.toLowerCase().startsWith(query));
    const rest = all.filter((t) => !t.name.toLowerCase().startsWith(query));
    matches = [...startsWith, ...rest].slice(0, 12);
    if (matches.length === 0) {
      dropdown.style.display = "none";
      activeIdx = -1;
      return;
    }
    dropdown.innerHTML = "";
    for (const m of matches) {
      const item = document.createElement("div");
      item.className = "autocomplete-item";
      // Show path so user can distinguish homonyms
      const pathDisplay = m.path.replace(/_/g, " ");
      item.innerHTML = `<span class="ac-path">${pathDisplay}</span>`;
      item.addEventListener("mousedown", (e) => {
        e.preventDefault();
        selectMatch(m);
      });
      dropdown.appendChild(item);
    }
    activeIdx = 0;
    highlightItem(0);
    dropdown.style.display = "block";
  };

  input.addEventListener("input", updateDropdown);
  input.addEventListener("focus", updateDropdown);
  input.addEventListener("blur", () => {
    setTimeout(() => { dropdown.style.display = "none"; }, 150);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      dropdown.style.display = "none";
      input.blur();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (matches.length > 0) highlightItem(Math.min(activeIdx + 1, matches.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (matches.length > 0) highlightItem(Math.max(activeIdx - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (activeIdx >= 0 && activeIdx < matches.length) selectMatch(matches[activeIdx]);
    }
  });

  wrapper.appendChild(input);
  wrapper.appendChild(dropdown);
  container.appendChild(wrapper);
  return input;
}


// ── Things (proximity filters) ──

// Track which paths are in the proximity filters for pill display
const proximityPaths: Array<{ name: string; path: string }> = [];

function createThingsHolder(
  container: HTMLElement,
  onChange: () => void,
): void {
  const holder = document.createElement("div");
  holder.className = "pill-holder";

  const renderPills = () => {
    holder.querySelectorAll(".pill").forEach((el) => el.remove());
    for (let i = 0; i < proximityPaths.length; i++) {
      const pp = proximityPaths[i];
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = pp.name.replace(/_/g, " ");
      pill.title = pp.path.replace(/_/g, " ");  // hover reveals full path
      const remove = document.createElement("span");
      remove.className = "pill-remove";
      remove.textContent = "\u00d7";
      const idx = i;
      remove.addEventListener("click", () => {
        state.proximityFilters.splice(idx, 1);
        proximityPaths.splice(idx, 1);
        renderPills();
        onChange();
      });
      pill.appendChild(remove);
      holder.insertBefore(pill, holder.lastElementChild);
    }
  };

  createAutocomplete(holder, () => allVocab, (term) => {
    if (!proximityPaths.some((p) => p.path === term.path)) {
      state.proximityFilters.push({ text: term.prompt, weight: 1.0 });
      proximityPaths.push({ name: term.name, path: term.path });
      renderPills();
      onChange();
    }
  });

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

  // Add contrast button
  const addBtn = document.createElement("div");
  addBtn.className = "add-contrast";

  const leftSlot = document.createElement("div");
  leftSlot.className = "contrast-slot";
  const vs = document.createElement("span");
  vs.className = "contrast-vs";
  vs.textContent = "vs";
  const rightSlot = document.createElement("div");
  rightSlot.className = "contrast-slot";

  let pendingPoleA: AutocompleteResult | null = null;
  let siblingTerms: AutocompleteResult[] = [];

  const resetBuilder = () => {
    pendingPoleA = null;
    siblingTerms = [];
    leftSlot.innerHTML = "";
    rightSlot.innerHTML = "";
    createAutocomplete(leftSlot, () => allVocab, async (term) => {
      pendingPoleA = term;
      leftSlot.innerHTML = "";
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = term.name.replace(/_/g, " ");
      pill.title = term.path.replace(/_/g, " ");
      leftSlot.appendChild(pill);
      // Fetch siblings for constrained right pole
      const sibs = await api.getSiblings(term.name);
      siblingTerms = sibs.map((s) => ({ name: s.name, path: term.path.replace(/\/[^/]+$/, "/" + s.name), prompt: s.prompt }));
      rightSlot.innerHTML = "";
      createAutocomplete(rightSlot, () => siblingTerms, (term2) => {
        state.contrastControls.push({
          pole_a: pendingPoleA!.prompt,
          pole_b: term2.prompt,
          role: "filter",
          band_min: -1.0,
          band_max: 1.0,
        });
        resetBuilder();
        renderContrasts();
        onChange();
      }, "opposite...");
    });
  };

  resetBuilder();

  addBtn.appendChild(leftSlot);
  addBtn.appendChild(vs);
  addBtn.appendChild(rightSlot);

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
  const header = document.createElement("div");
  header.className = "contrast-header";
  header.innerHTML = `<span>${cc.pole_a.replace(/_/g, " ")}</span><span class="contrast-vs">vs</span><span>${cc.pole_b.replace(/_/g, " ")}</span>`;
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
  labels.innerHTML = `<span>${cc.pole_a.replace(/_/g, " ")}</span><span>${cc.pole_b.replace(/_/g, " ")}</span>`;

  widget.appendChild(bandWidget);
  widget.appendChild(labels);

  return widget;
}


// ── Main init ──

export async function initControls(dimensions: Dimension[], models: string[]): Promise<void> {
  // Fetch vocabulary for autocomplete
  allVocab = await api.getVocabularyFlat();
  allVocab.sort((a, b) => a.name.localeCompare(b.name));

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
    opt.textContent = m === "clip-vit-b-32" ? "CLIP (semantic)" : m === "dinov2-vitb14" ? "DINOv2 (texture)" : m;
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
