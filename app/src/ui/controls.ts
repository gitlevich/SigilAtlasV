/**
 * UI controls for SliceMode and NeighborhoodMode.
 */

import { state, notify } from "../state";
import * as api from "../api";
import type { Dimension } from "../types";
import type { TorusViewport } from "../renderer/torus-viewport";

let viewport: TorusViewport | null = null;

export function setViewport(vp: TorusViewport): void {
  viewport = vp;
}

async function recomputeLayout(): Promise<void> {
  const layout = await api.computeLayout({
    image_ids: state.imageIds,
    axes: state.selectedAxes.length > 0 ? state.selectedAxes : null,
    tightness: state.tightness,
    model: state.model,
    strip_height: state.stripHeight,
  });
  state.layout = layout;
  state.torusWidth = layout.torus_width;
  state.torusHeight = layout.torus_height;
  if (viewport) viewport.setLayout(layout);
  notify();
}

async function recomputeSlice(): Promise<void> {
  const res = await api.computeSlice({
    range_filters: state.rangeFilters,
    proximity_filters: state.proximityFilters,
    model: state.model,
  });
  state.imageIds = res.image_ids;
  await recomputeLayout();
}

function makeFoldable(panel: HTMLElement, side: "left" | "right"): void {
  const h3 = panel.querySelector("h3");
  if (h3) {
    h3.addEventListener("click", () => panel.classList.toggle("folded"));
  }
}

export async function initControls(dimensions: Dimension[], models: string[]): Promise<void> {
  // --- Slice panel (left) ---
  const slicePanel = document.getElementById("slice-panel")!;
  slicePanel.innerHTML = "<h3>Slice</h3>";

  // Range filters per dimension
  for (const dim of dimensions) {
    if (dim.type !== "range" || dim.min === undefined || dim.max === undefined) continue;

    const container = document.createElement("div");
    container.className = "control-group";

    const label = document.createElement("label");
    label.textContent = dim.name;
    container.appendChild(label);

    const minInput = document.createElement("input");
    minInput.type = "range";
    minInput.min = String(dim.min);
    minInput.max = String(dim.max);
    minInput.step = String((dim.max - dim.min) / 100);
    minInput.value = String(dim.min);

    const maxInput = document.createElement("input");
    maxInput.type = "range";
    maxInput.min = String(dim.min);
    maxInput.max = String(dim.max);
    maxInput.step = String((dim.max - dim.min) / 100);
    maxInput.value = String(dim.max);

    const updateRange = () => {
      const existing = state.rangeFilters.findIndex((f) => f.dimension === dim.name);
      const filter = { dimension: dim.name, min: parseFloat(minInput.value), max: parseFloat(maxInput.value) };
      if (existing >= 0) {
        state.rangeFilters[existing] = filter;
      } else {
        state.rangeFilters.push(filter);
      }
    };

    minInput.addEventListener("change", () => { updateRange(); recomputeSlice(); });
    maxInput.addEventListener("change", () => { updateRange(); recomputeSlice(); });

    container.appendChild(minInput);
    container.appendChild(maxInput);
    slicePanel.appendChild(container);
  }

  // Proximity filter text input
  const proxContainer = document.createElement("div");
  proxContainer.className = "control-group";
  const proxLabel = document.createElement("label");
  proxLabel.textContent = "Proximity";
  proxContainer.appendChild(proxLabel);

  const proxInput = document.createElement("input");
  proxInput.type = "text";
  proxInput.placeholder = "e.g. sunset, forest...";
  proxInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && proxInput.value.trim()) {
      state.proximityFilters.push({ text: proxInput.value.trim(), weight: 1.0 });
      proxInput.value = "";
      recomputeSlice();
    }
  });
  proxContainer.appendChild(proxInput);
  slicePanel.appendChild(proxContainer);

  // --- Neighborhood panel (right) ---
  const nbPanel = document.getElementById("neighborhood-panel")!;
  nbPanel.innerHTML = "<h3>Neighborhood</h3>";

  // Axis checkboxes
  const axesGroup = document.createElement("div");
  axesGroup.className = "control-group";
  const axesLabel = document.createElement("label");
  axesLabel.textContent = "Axes";
  axesGroup.appendChild(axesLabel);

  // Axes that define similarity for layout: metadata dimensions + embedding
  // (raw characterization paths are not useful as axes)
  const axisNames = ["time", "location"];
  for (const axis of axisNames) {
    const row = document.createElement("div");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.id = `axis-${axis}`;
    cb.addEventListener("change", () => {
      if (cb.checked) {
        state.selectedAxes.push(axis);
      } else {
        state.selectedAxes = state.selectedAxes.filter((a) => a !== axis);
      }
      recomputeLayout();
    });
    const lbl = document.createElement("label");
    lbl.htmlFor = cb.id;
    lbl.textContent = axis;
    row.appendChild(cb);
    row.appendChild(lbl);
    axesGroup.appendChild(row);
  }
  nbPanel.appendChild(axesGroup);

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
  tightnessSlider.addEventListener("change", () => {
    state.tightness = parseFloat(tightnessSlider.value);
    recomputeLayout();
  });
  tightnessGroup.appendChild(tightnessSlider);
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
    recomputeLayout();
  });
  modelGroup.appendChild(modelSelect);
  nbPanel.appendChild(modelGroup);

  // Image count display
  const countEl = document.createElement("div");
  countEl.id = "image-count";
  countEl.textContent = `${state.imageIds.length} images`;
  nbPanel.appendChild(countEl);

  // Make panels foldable (!foldable invariant)
  makeFoldable(slicePanel, "left");
  makeFoldable(nbPanel, "right");
}
