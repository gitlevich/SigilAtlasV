/**
 * Color wheel widget for hue selection.
 *
 * A canvas-drawn hue ring. Click/drag to select a hue center.
 * A draggable arc shows the selected hue range.
 * Outputs { min, max } in degrees [0, 360], wrapping around.
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

export function createColorWheel(options: ColorWheelOptions): HTMLElement {
  const { size, initial, onChange } = options;
  const current: HueRange = { ...initial };

  const container = document.createElement("div");
  container.className = "color-wheel-container";

  const canvas = document.createElement("canvas");
  canvas.width = size * 2;  // retina
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

  function draw(): void {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw hue ring
    const steps = 360;
    for (let i = 0; i < steps; i++) {
      const a0 = (i - 90) * Math.PI / 180;
      const a1 = (i + 1 - 90) * Math.PI / 180;
      ctx.beginPath();
      ctx.arc(cx, cy, outerR, a0, a1);
      ctx.arc(cx, cy, innerR, a1, a0, true);
      ctx.closePath();
      ctx.fillStyle = `hsl(${i}, 80%, 50%)`;
      ctx.fill();
    }

    // Dim the ring outside the selected range
    if (current.width > 0 && current.width < 360) {
      const startAngle = current.center - current.width / 2;
      const endAngle = current.center + current.width / 2;
      // Draw a dim overlay on the unselected part
      ctx.globalCompositeOperation = "source-atop";
      ctx.fillStyle = "rgba(13, 13, 18, 0.7)";

      // Fill the entire ring, then clear the selected arc
      ctx.beginPath();
      ctx.arc(cx, cy, outerR + 1, 0, Math.PI * 2);
      ctx.arc(cx, cy, innerR - 1, 0, Math.PI * 2, true);
      ctx.fill();

      // Restore the selected arc
      const sa = (startAngle - 90) * Math.PI / 180;
      const ea = (endAngle - 90) * Math.PI / 180;
      ctx.globalCompositeOperation = "destination-out";
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, outerR + 2, sa, ea);
      ctx.closePath();
      ctx.fill();

      ctx.globalCompositeOperation = "source-over";

      // Draw selection indicator lines
      for (const angle of [startAngle, endAngle]) {
        const a = (angle - 90) * Math.PI / 180;
        ctx.beginPath();
        ctx.moveTo(cx + innerR * Math.cos(a), cy + innerR * Math.sin(a));
        ctx.lineTo(cx + outerR * Math.cos(a), cy + outerR * Math.sin(a));
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    // Draw center dot showing selected hue
    if (current.width > 0) {
      const a = (current.center - 90) * Math.PI / 180;
      const dotR = midR;
      ctx.beginPath();
      ctx.arc(cx + dotR * Math.cos(a), cy + dotR * Math.sin(a), 6, 0, Math.PI * 2);
      ctx.fillStyle = `hsl(${current.center}, 80%, 50%)`;
      ctx.strokeStyle = "#fff";
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

  let dragging = false;

  canvas.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * canvas.width - cx;
    const y = (e.clientY - rect.top) / rect.height * canvas.height - cy;
    const dist = Math.sqrt(x * x + y * y);

    // Only respond to clicks on the ring
    if (dist < innerR - 10 || dist > outerR + 10) return;

    dragging = true;
    current.center = angleFromEvent(e);
    if (current.width === 0) current.width = 40; // activate with default range
    draw();
    onChange(current);

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  });

  function onMove(e: PointerEvent): void {
    if (!dragging) return;
    current.center = angleFromEvent(e);
    draw();
    onChange(current);
  }

  function onUp(): void {
    dragging = false;
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
  }

  // Width control: scroll on the wheel to widen/narrow the range
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    current.width = Math.max(0, Math.min(360, current.width + e.deltaY * 0.5));
    draw();
    onChange(current);
  }, { passive: false });

  return container;
}

/** Convert a hue center + width to a min/max range for the hue_dominant filter. */
export function hueRangeToFilter(range: HueRange): { min: number; max: number } | null {
  if (range.width <= 0 || range.width >= 360) return null;
  let min = range.center - range.width / 2;
  let max = range.center + range.width / 2;
  // Normalize to [0, 360]
  if (min < 0) min += 360;
  if (max > 360) max -= 360;
  return { min, max };
}
