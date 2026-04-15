/**
 * Color wheel widget for hue selection.
 *
 * A canvas-drawn hue ring with a draggable arc selection.
 * - Drag the arc body to rotate (change center hue)
 * - Drag the arc edge handles to widen/narrow the range
 * - Click the ring outside the arc to set a new center
 */

export interface HueRange {
  center: number;  // degrees 0-360
  width: number;   // degrees of arc, 0 = off, 360 = all
}

export interface ColorWheelOptions {
  size: number;
  initial: HueRange;
  onChange: (range: HueRange) => void;
}

function normAngle(a: number): number {
  return ((a % 360) + 360) % 360;
}

function angleDiff(a: number, b: number): number {
  let d = a - b;
  while (d > 180) d -= 360;
  while (d < -180) d += 360;
  return d;
}

export function createColorWheel(options: ColorWheelOptions): HTMLElement {
  const { size, initial, onChange } = options;
  const current: HueRange = { ...initial };

  const container = document.createElement("div");
  container.className = "color-wheel-container";

  const canvas = document.createElement("canvas");
  canvas.width = size * 2;
  canvas.height = size * 2;
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;
  canvas.className = "color-wheel-canvas";
  container.appendChild(canvas);

  const ctx = canvas.getContext("2d")!;
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const outerR = cx - 4;
  const innerR = outerR * 0.65;
  const midR = (outerR + innerR) / 2;
  const handleR = 7;

  function startAngle(): number { return normAngle(current.center - current.width / 2); }
  function endAngle(): number { return normAngle(current.center + current.width / 2); }

  function toCanvas(deg: number): number { return (deg - 90) * Math.PI / 180; }

  function draw(): void {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw hue ring
    for (let i = 0; i < 360; i++) {
      const a0 = toCanvas(i);
      const a1 = toCanvas(i + 1);
      ctx.beginPath();
      ctx.arc(cx, cy, outerR, a0, a1);
      ctx.arc(cx, cy, innerR, a1, a0, true);
      ctx.closePath();
      ctx.fillStyle = `hsl(${i}, 80%, 50%)`;
      ctx.fill();
    }

    if (current.width <= 0 || current.width >= 360) return;

    // Dim unselected portion
    const sa = toCanvas(startAngle());
    const ea = toCanvas(endAngle());

    ctx.globalCompositeOperation = "source-atop";
    ctx.fillStyle = "rgba(13, 13, 18, 0.7)";
    ctx.beginPath();
    ctx.arc(cx, cy, outerR + 1, 0, Math.PI * 2);
    ctx.arc(cx, cy, innerR - 1, 0, Math.PI * 2, true);
    ctx.fill();

    // Cut out the selected arc
    ctx.globalCompositeOperation = "destination-out";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, outerR + 2, sa, ea);
    ctx.closePath();
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";

    // Edge handles — small circles on the ring at start and end angles
    for (const angle of [startAngle(), endAngle()]) {
      const a = toCanvas(angle);
      const hx = cx + midR * Math.cos(a);
      const hy = cy + midR * Math.sin(a);
      ctx.beginPath();
      ctx.arc(hx, hy, handleR, 0, Math.PI * 2);
      ctx.fillStyle = "#c8c8d0";
      ctx.strokeStyle = "#0d0d12";
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();
    }
  }

  draw();

  // Interaction
  function angleFromEvent(e: MouseEvent): number {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * canvas.width - cx;
    const y = (e.clientY - rect.top) / rect.height * canvas.height - cy;
    let deg = Math.atan2(y, x) * 180 / Math.PI + 90;
    if (deg < 0) deg += 360;
    return deg;
  }

  function distFromEvent(e: MouseEvent): number {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * canvas.width - cx;
    const y = (e.clientY - rect.top) / rect.height * canvas.height - cy;
    return Math.sqrt(x * x + y * y);
  }

  function isNearHandle(e: MouseEvent, handleAngle: number): boolean {
    const a = toCanvas(handleAngle);
    const hx = cx + midR * Math.cos(a);
    const hy = cy + midR * Math.sin(a);
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width * canvas.width;
    const my = (e.clientY - rect.top) / rect.height * canvas.height;
    const dx = mx - hx;
    const dy = my - hy;
    return Math.sqrt(dx * dx + dy * dy) < handleR * 3;
  }

  type DragMode = "start-edge" | "end-edge" | "rotate" | "none";
  let mode: DragMode = "none";
  let dragStartAngle = 0;
  let dragStartCenter = 0;
  let dragStartWidth = 0;

  // Double-click resets the color selection
  canvas.addEventListener("dblclick", (e) => {
    e.preventDefault();
    e.stopPropagation();
    current.width = 0;
    current.center = 0;
    draw();
    onChange(current);
  });

  canvas.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const dist = distFromEvent(e);
    if (dist < innerR - 20 || dist > outerR + 20) return;

    const clickAngle = angleFromEvent(e);

    if (current.width > 0 && current.width < 360) {
      // Check if near an edge handle
      if (isNearHandle(e, startAngle())) {
        mode = "start-edge";
      } else if (isNearHandle(e, endAngle())) {
        mode = "end-edge";
      } else {
        // Check if inside the arc — rotate
        const diff = angleDiff(clickAngle, current.center);
        if (Math.abs(diff) < current.width / 2) {
          mode = "rotate";
        } else {
          // Click outside arc — set new center
          current.center = clickAngle;
          if (current.width === 0) current.width = 40;
          mode = "rotate";
        }
      }
    } else {
      // No selection yet — create one
      current.center = clickAngle;
      current.width = 40;
      mode = "rotate";
    }

    dragStartAngle = clickAngle;
    dragStartCenter = current.center;
    dragStartWidth = current.width;

    draw();
    onChange(current);

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  });

  function onMove(e: PointerEvent): void {
    if (mode === "none") return;
    const angle = angleFromEvent(e);

    if (mode === "rotate") {
      const delta = angleDiff(angle, dragStartAngle);
      current.center = normAngle(dragStartCenter + delta);
    } else if (mode === "start-edge") {
      // Move the start edge: changes width, keeps end fixed
      const end = normAngle(dragStartCenter + dragStartWidth / 2);
      const newWidth = angleDiff(end, angle);
      current.width = Math.max(10, Math.min(350, ((newWidth % 360) + 360) % 360));
      current.center = normAngle(end - current.width / 2);
    } else if (mode === "end-edge") {
      // Move the end edge: changes width, keeps start fixed
      const start = normAngle(dragStartCenter - dragStartWidth / 2);
      const newWidth = angleDiff(angle, start);
      current.width = Math.max(10, Math.min(350, ((newWidth % 360) + 360) % 360));
      current.center = normAngle(start + current.width / 2);
    }

    draw();
    onChange(current);
  }

  function onUp(): void {
    mode = "none";
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
  }

  return container;
}

/** Convert a hue center + width to a min/max range for the hue_dominant filter. */
export function hueRangeToFilter(range: HueRange): { min: number; max: number } | null {
  if (range.width <= 0 || range.width >= 360) return null;
  let min = range.center - range.width / 2;
  let max = range.center + range.width / 2;
  if (min < 0) min += 360;
  if (max > 360) max -= 360;
  return { min, max };
}
