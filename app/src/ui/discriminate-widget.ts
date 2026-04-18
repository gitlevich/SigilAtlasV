/**
 * Discriminate range slider widget.
 *
 * Renders a `Discriminate(contrast, min, max)` SigilML operator: a track with
 * a draggable, resizable band spanning [min, max] within [rangeMin, rangeMax].
 *
 * Interactions:
 * - Drag left edge to narrow from the left
 * - Drag right edge to narrow from the right
 * - Drag the band body to slide the whole window
 *
 * Two callbacks:
 * - onChange: fires on every drag pixel (for live visual feedback)
 * - onCommit: fires on pointer up (for expensive recomputes)
 * Supply at least one.
 */

export interface DiscriminateState {
  min: number;
  max: number;
}

export interface DiscriminateOptions {
  rangeMin: number;
  rangeMax: number;
  initial: DiscriminateState;
  onChange?: (state: DiscriminateState) => void;
  onCommit?: (state: DiscriminateState) => void;
  dimension?: string;
}

export function createDiscriminateWidget(options: DiscriminateOptions): HTMLElement {
  const { rangeMin, rangeMax, initial, onChange, onCommit, dimension } = options;
  const span = rangeMax - rangeMin;
  if (span <= 0) {
    return document.createElement("div");
  }

  const current: DiscriminateState = { min: initial.min, max: initial.max };

  const track = document.createElement("div");
  track.className = "discriminate-track";
  if (dimension) track.dataset.dim = dimension;

  const bg = document.createElement("div");
  bg.className = "discriminate-track-bg";
  track.appendChild(bg);

  const band = document.createElement("div");
  band.className = "discriminate-band";
  track.appendChild(band);

  const handleL = document.createElement("div");
  handleL.className = "discriminate-handle left";
  band.appendChild(handleL);

  const handleR = document.createElement("div");
  handleR.className = "discriminate-handle right";
  band.appendChild(handleR);

  const toFrac = (v: number) => (v - rangeMin) / span;

  function updateDOM(): void {
    const l = toFrac(current.min) * 100;
    const r = toFrac(current.max) * 100;
    band.style.left = `${l}%`;
    band.style.width = `${r - l}%`;
  }

  updateDOM();

  // --- drag machinery ---
  type DragMode = "left" | "right" | "move";
  let mode: DragMode | null = null;
  let startX = 0;
  let startMin = 0;
  let startMax = 0;
  let trackWidth = 0;

  function clamp(v: number, lo: number, hi: number): number {
    return Math.max(lo, Math.min(hi, v));
  }

  function beginDrag(e: PointerEvent, m: DragMode): void {
    e.preventDefault();
    e.stopPropagation();
    mode = m;
    startX = e.clientX;
    startMin = current.min;
    startMax = current.max;
    trackWidth = track.getBoundingClientRect().width;
    (e.target as HTMLElement).classList.add("dragging");
    if (m === "move") band.classList.add("dragging");
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  function onMove(e: PointerEvent): void {
    if (!mode) return;
    const dx = e.clientX - startX;
    const dv = (dx / trackWidth) * span;

    if (mode === "left") {
      current.min = clamp(startMin + dv, rangeMin, current.max - span * 0.01);
    } else if (mode === "right") {
      current.max = clamp(startMax + dv, current.min + span * 0.01, rangeMax);
    } else {
      const w = startMax - startMin;
      let newMin = startMin + dv;
      let newMax = startMax + dv;
      if (newMin < rangeMin) { newMin = rangeMin; newMax = rangeMin + w; }
      if (newMax > rangeMax) { newMax = rangeMax; newMin = rangeMax - w; }
      current.min = newMin;
      current.max = newMax;
    }

    updateDOM();
    if (onChange) onChange(current);
  }

  function onUp(): void {
    band.classList.remove("dragging");
    track.querySelectorAll(".dragging").forEach((el) => el.classList.remove("dragging"));
    mode = null;
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    if (onCommit) onCommit(current);
  }

  // Double-click resets to full range
  track.addEventListener("dblclick", (e) => {
    e.preventDefault();
    current.min = rangeMin;
    current.max = rangeMax;
    updateDOM();
    if (onChange) onChange(current);
    if (onCommit) onCommit(current);
  });

  handleL.addEventListener("pointerdown", (e) => beginDrag(e, "left"));
  handleR.addEventListener("pointerdown", (e) => beginDrag(e, "right"));
  band.addEventListener("pointerdown", (e) => {
    if (e.target === handleL || e.target === handleR) return;
    beginDrag(e, "move");
  });

  return track;
}
